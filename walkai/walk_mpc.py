"""Walking feasibility check (no learning): closed-loop predictive-sampling MPC with the env's own physics.

  ../standai/.venv/bin/python -I walkai/walk_mpc.py --seeds 0 1 2 3 4 5 --steps 400
Each seed: settle with the stand policy (e2e1) for 3 s, then at every control step sample N smooth action plans over a
0.5 s horizon, roll them out on a copy of the physics state, apply the first action of the best plan to the REAL WalkEnv
(so real_steps / gait quality are measured by the same code that scores training). Cost = forward-speed tracking +
single-support (one foot lifted, the other planted) bonus - slip / foot-height / joint-limit / foot-gap violations.
Reports, per seed, whether it stayed up, how far it went, how many real steps (foot lifted >= 0.1 s, lands ahead of the
other foot) and the mean gait quality. A "case" = no fall, >= 8 real steps, quality >= 0.8, mean speed >= 0.2 m/s.
"""
import argparse
import sys
import pathlib as _pl
import time
from multiprocessing import Pool

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from walk_env import (CHECKPOINT_DIR, FEET_GAP_MAX, FOOT_Z_MAX, HIP_MAX_RAD, KNEE_MIN_RAD, PLANTED_Z,  # noqa: E402
                      WalkGetUpEnv)
import env as env_module  # noqa: E402  (walk_env puts standai/scripts on sys.path)
from env import FRAME_SKIP  # noqa: E402

H, N, KN, SIGMA = 40, 160, 8, 0.45   # 0.8 s horizon
TARGET_V = 0.4
import os  # noqa: E402
GAP_MAX = float(os.environ.get('MPC_GAP_MAX', FEET_GAP_MAX))   # |x_r - x_l| above this is penalised in the MPC cost
GAP_K = float(os.environ.get('MPC_GAP_K', 3.0))


def apply_ctrl(env, a):
    d = env.data
    for _ in range(FRAME_SKIP):
        d.ctrl[env.a_act] = a
        d.ctrl[env.a_act[env.a_arm_act_idx]] *= env._power_scale(env.a_arm_qpos, env.a_arm_range, env_module.ARM_MIN_POWER)
        d.ctrl[env.a_act[env.a_leg_act_idx]] *= env._power_scale(env.a_leg_qpos, env.a_leg_range, env_module.LEG_MIN_POWER)
        d.ctrl[env.a_root_act] = 0.0
        d.ctrl[env.a_balance_act] = 0.0
        mujoco.mj_step(env.model, d)


def snap(env, J):
    d = env.data
    zf = [d.site_xpos[s][2] for s in env.a_feet]
    xf = [d.site_xpos[s][0] for s in env.a_feet]
    return np.array([d.xpos[env.a_torso_id][2], d.qpos[env.a_ry_qpos], d.qvel[env._a_root_x_dof], zf[0], zf[1], xf[0], xf[1],
                     d.qpos[J[0]], d.qpos[J[1]], d.qpos[J[2]], d.qpos[J[3]]])


def horizon_cost(tr, prev_xf):
    z, ry, vx, z0, z1, x0, x1, hr, hl, kr, kl = tr.T
    up = (z >= 0.9) & (np.abs(ry) < 0.5)
    track = np.exp(-((vx - TARGET_V) ** 2) / (2 * 0.15 ** 2))
    lifted0, lifted1 = z0 > 0.15, z1 > 0.15
    single = (lifted0 ^ lifted1) & (np.where(lifted0, z1, z0) < PLANTED_Z)
    vf0 = np.diff(np.concatenate([[prev_xf[0]], x0])) / 0.02
    vf1 = np.diff(np.concatenate([[prev_xf[1]], x1])) / 0.02
    slip = 12.0 * (np.abs(vf0) * (z0 < PLANTED_Z) + np.abs(vf1) * (z1 < PLANTED_Z))
    viol = (slip + 30 * (np.maximum(z0 - FOOT_Z_MAX, 0) + np.maximum(z1 - FOOT_Z_MAX, 0))
            + 20 * (np.maximum(hr - 0.7, 0) + np.maximum(hl - 0.7, 0))
            + 10 * (np.maximum(KNEE_MIN_RAD - kr, 0) + np.maximum(KNEE_MIN_RAD - kl, 0))
            + GAP_K * np.maximum(np.abs(x0 - x1) - GAP_MAX, 0))
    flight = ((z0 > 0.12) & (z1 > 0.12)).astype(float)          # both feet off the ground = hopping, not walking
    viol = viol + 8.0 * flight
    fell = (z < 0.6).any()
    return (np.mean(up * (2.0 * track + 0.6 * single)) - 0.15 * np.mean(viol) + 3.0 * up[-1] - 8.0 * fell)


def run_seed(args):
    seed, steps, settle = args
    env = WalkGetUpEnv(fixed_cmd=0.5, stand_frac=1.0)
    m = env.model
    J = [m.joint(n).qposadr[0] for n in ("a_hip_r", "a_hip_l", "a_knee_r", "a_knee_l")]
    J = [J[0], J[1], J[2], J[3]]
    stand = PPO.load(CHECKPOINT_DIR / "best_getup_e2e1.zip", device="cpu")
    obs, _ = env.reset(seed=seed)
    for _ in range(settle):
        obs, *_ = env.step(stand.predict(obs, deterministic=True)[0])
    rng = np.random.default_rng(seed)
    plan = np.zeros((H, 12))
    xs_k, tt = np.linspace(0, H - 1, KN), np.arange(H)
    cur = mujoco.MjData(m)
    x_start = float(env.data.qpos[m.joint("a_root_x").qposadr[0]])
    real, cross, qs, ups, vxs, alive, falls = 0, 0, [], [], [], 0, 0
    gaps, ss = [], []
    acts = []
    for k in range(steps):
        mujoco.mj_copyData(cur, m, env.data)
        prev_xf = [float(env.data.site_xpos[s][0]) for s in env.a_feet]
        noise = rng.standard_normal((N, KN, 12)) * SIGMA
        noise[0] = 0.0
        best, best_c = 0, -1e9
        cands = np.empty((N, H, 12))
        for i in range(N):
            for j in range(12):
                cands[i, :, j] = plan[:, j] + np.interp(tt, xs_k, noise[i, :, j])
        cands = np.clip(cands, -1, 1)
        for i in range(N):
            mujoco.mj_copyData(env.data, m, cur)
            tr = np.empty((H, 11))
            for t in range(H):
                apply_ctrl(env, cands[i, t])
                tr[t] = snap(env, J)
            c = horizon_cost(tr, prev_xf)
            if c > best_c:
                best_c, best = c, i
        plan = cands[best]
        mujoco.mj_copyData(env.data, m, cur)
        a0 = plan[0].copy()
        acts.append(a0)
        _, _, term, trunc, info = env.step(np.concatenate([[0.0], a0]).astype(np.float32))
        b = info["reward_breakdown"]
        real += b["real_steps"]; cross += b["cross_steps"]; ss.append(b["single"] > 0)
        gaps.append(float(env.data.site_xpos[env.a_feet[0]][0] - env.data.site_xpos[env.a_feet[1]][0])); qs.append(b["quality"]); ups.append(info["upright"]); vxs.append(info["vx"]); alive = k + 1
        if info["torso_z"] < 0.6:
            falls = 1
            break
        plan = np.vstack([plan[1:], plan[-1:]])
    dx = float(info["root_x"]) - x_start
    ok = (not falls) and real >= 8 and np.mean(qs) >= 0.8 and np.mean(vxs) >= 0.2
    return dict(cross=cross, gap_lo=float(min(gaps)), gap_hi=float(max(gaps)), single=float(np.mean(ss)), seed=seed, alive=alive, dx=dx, real_steps=real, quality=float(np.mean(qs)), upright=float(np.mean(ups)),
                mean_vx=float(np.mean(vxs)), fell=bool(falls), case=bool(ok), acts=np.array(acts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--settle", type=int, default=150)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="checkpoints/walk_mpc.npz")
    a = ap.parse_args()
    t0 = time.time()
    with Pool(min(a.procs, len(a.seeds))) as p:
        out = p.map(run_seed, [(s, a.steps, a.settle) for s in a.seeds])
    print(f"done in {time.time() - t0:.0f}s ({a.steps} steps = {a.steps * 0.02:.0f}s per seed)")
    print(f"{'seed':>4} {'alive':>5} {'dx(m)':>6} {'vx':>5} {'steps':>5} {'cross':>5} {'single':>6} {'quality':>7} {'upright':>7} {'foot gap range(m)':>18}  case")
    for r in out:
        print(f"{r['seed']:>4} {r['alive']:>5} {r['dx']:>6.2f} {r['mean_vx']:>5.2f} {r['real_steps']:>5} {r['cross']:>5} {r['single']*100:>5.0f}% {r['quality']:>7.2f} {r['upright']*100:>6.0f}% {r['gap_lo']:>+8.2f} {r['gap_hi']:>+8.2f}  {'CASE' if r['case'] else ('fell' if r['fell'] else '-')}")
    print(f"cases: {sum(r['case'] for r in out)}/{len(out)}")
    np.savez(a.out, seeds=[r['seed'] for r in out], acts=np.stack([r['acts'] for r in out if len(r['acts']) == a.steps] or [np.zeros(1)]),
             case=[r['case'] for r in out])


if __name__ == "__main__":
    main()
