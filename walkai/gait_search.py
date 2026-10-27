"""Feasibility test: can this body walk at all? Open-loop sinusoidal gait + simple pitch feedback, optimised with CEM.

  ../standai/.venv/bin/python -I walkai/gait_search.py --gens 60 --out checkpoints/gait_best.json
Gait: right leg joint j = bias_j + amp_j * sin(2 pi f t + phase_j); left leg is the same shifted by half a period.
Hips get + kp * pitch + kd * pitch_rate, waist gets kpw * pitch. Fitness = forward distance x mean gait quality x
survival, with a floor on the number of real steps (see walk_env.walk_reward): shaking in place or sliding scores ~0.
Action vector: [root_x, waist, head, sh_r, el_r, sh_l, el_l, hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l].
"""
import argparse
import json
import sys
import pathlib as _pl
import time
from multiprocessing import Pool

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from walk_env import WalkEnv  # noqa: E402

NAMES = ["freq", "hip_b", "hip_a", "hip_p", "knee_b", "knee_a", "knee_p", "ankle_b", "ankle_a", "ankle_p",
         "waist_b", "kp_hip", "kd_hip", "kp_waist"]
LO = np.array([0.6, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, -4, -1, -4], dtype=float)
HI = np.array([2.6, 1, 1, 2 * np.pi, 1, 1, 2 * np.pi, 1, 1, 2 * np.pi, 1, 4, 1, 4], dtype=float)
MEAN0 = np.array([1.4, 0.0, 0.4, 0.0, -0.2, 0.4, 1.5, 0.0, 0.2, 3.0, 0.0, 0.0, 0.0, 0.0])
SIGMA0 = (HI - LO) * 0.25
STEPS, TMAX = 400, 400
_env = None


def _get_env():
    global _env
    if _env is None:
        _env = WalkEnv(fixed_cmd=0.5)
    return _env


def action_at(p, t, pitch, dpitch):
    f, ph = p[0], 2 * np.pi * p[0] * t
    a = np.zeros(13)
    for j, (b, amp, phase) in enumerate(((p[1], p[2], p[3]), (p[4], p[5], p[6]), (p[7], p[8], p[9]))):
        a[7 + j] = b + amp * np.sin(ph + phase)               # right hip, knee, ankle
        a[10 + j] = b + amp * np.sin(ph + phase + np.pi)      # left, half a period later
    a[7] += p[11] * pitch + p[12] * dpitch
    a[10] += p[11] * pitch + p[12] * dpitch
    a[1] = p[10] + p[13] * pitch
    return np.clip(a, -1, 1)


def rollout(p, seed):
    env = _get_env()
    env.reset(seed=seed)
    x0 = float(env.data.qpos[env.model.joint("a_root_x").qposadr[0]])
    qs, steps, alive = [], 0, 0
    for k in range(STEPS):
        pitch = float(env.data.qpos[env.a_ry_qpos]); dpitch = float(env.data.qvel[env.a_ry_dof]) if hasattr(env, "a_ry_dof") else 0.0
        _, _, term, trunc, info = env.step(action_at(p, k * 0.02, pitch, dpitch))
        b = info["reward_breakdown"]
        qs.append(b["quality"]); steps += b["real_steps"]; alive = k + 1
        if info["torso_z"] < 0.6 or term or trunc:
            break
    dx = info["root_x"] - x0
    return dx, float(np.mean(qs)), steps, alive


def fitness(p, seeds=(11, 12)):
    p = np.clip(p, LO, HI)
    res = [rollout(p, s) for s in seeds]
    fits = []
    for dx, q, steps, alive in res:
        f = (max(dx, 0.0) + 0.05) * q * (alive / STEPS)      # small floor so step count alone gives a gradient
        f *= min(1.0, steps / 8.0) ** 2          # a walk needs >= 8 real steps in 8 s; shuffling scores ~0
        fits.append(f)
    return float(np.mean(fits)), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=84)
    ap.add_argument("--elite", type=int, default=12)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--out", default="checkpoints/gait_best.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    mean, sigma, best = MEAN0.copy(), SIGMA0.copy(), (-1.0, None, None)
    t0 = time.time()
    with Pool(a.procs) as pool:
        for g in range(a.gens):
            pop = np.clip(mean + sigma * rng.standard_normal((a.pop, len(mean))), LO, HI)
            out = pool.map(fitness, list(pop))
            fit = np.array([o[0] for o in out])
            idx = np.argsort(-fit)[:a.elite]
            mean = 0.3 * mean + 0.7 * pop[idx].mean(axis=0)
            sigma = np.maximum(0.5 * sigma + 0.5 * pop[idx].std(axis=0), (HI - LO) * 0.01)
            if fit[idx[0]] > best[0]:
                best = (float(fit[idx[0]]), pop[idx[0]].copy(), out[idx[0]][1])
                json.dump(dict(fitness=best[0], params=dict(zip(NAMES, best[1].tolist())), rollouts=best[2]), open(a.out, "w"), indent=1)
            dx, q, st, al = out[idx[0]][1][0]
            print(f"gen {g:3d} [{time.time() - t0:5.0f}s] best {fit[idx[0]]:6.2f} mean {fit.mean():5.2f} | top: dx {dx:+.2f} m quality {q:.2f} steps {st} alive {al}/{STEPS} f {pop[idx[0]][0]:.2f} Hz", flush=True)


if __name__ == "__main__":
    main()
