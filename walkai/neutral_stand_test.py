"""Can the stand policies hold a feet-together pose? Put the body in a symmetric neutral pose (legs straight, feet under the hip),
run a policy with command 0 for 10 s and report torso height, upright share and how far apart the feet drift."""
import sys
import pathlib as _pl

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from walk_env import CHECKPOINT_DIR, WalkGetUpEnv  # noqa: E402


def neutral(env, hip=0.05, knee=-0.05, ankle=0.0, jitter=0.0, rng=None):
    m, d = env.model, env.data
    for n, v in (("a_hip_r", hip), ("a_hip_l", hip), ("a_knee_r", knee), ("a_knee_l", knee), ("a_ankle_r", ankle), ("a_ankle_l", ankle)):
        q = v + (rng.normal(0, jitter) if jitter else 0.0)
        d.qpos[m.joint(n).qposadr[0]] = q
    d.qpos[env.a_ry_qpos] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    low = min(d.site_xpos[s][2] for s in env.a_feet)
    d.qpos[m.joint("a_root_z").qposadr[0]] += 0.075 - low          # feet resting on the ground
    mujoco.mj_forward(m, d)


def run(policy, seeds=8, steps=500, **kw):
    env = WalkGetUpEnv(fixed_cmd=0.0, stand_frac=1.0)
    rows = []
    for sd in range(seeds):
        env.reset(seed=sd)
        rng = np.random.default_rng(sd)
        neutral(env, jitter=kw.get("jitter", 0.0), rng=rng)
        obs = env._obs()
        up, gap, z = [], [], []
        first_fall = None
        for t in range(steps):
            obs, r, te, tr, info = env.step(policy.predict(obs, deterministic=True)[0])
            x = [float(env.data.site_xpos[s][0]) for s in env.a_feet]
            up.append(info["upright"]); gap.append(abs(x[0] - x[1])); z.append(info["torso_z"])
            if first_fall is None and info["torso_z"] < 0.6:
                first_fall = t * 0.02
        rows.append((np.mean(up), np.mean(gap), gap[-1], np.mean(z), first_fall))
    r = rows
    falls = sum(1 for x in r if x[4] is not None)
    return (f"upright {np.mean([x[0] for x in r])*100:3.0f}% | feet gap mean {np.mean([x[1] for x in r]):.2f} m, at 10 s {np.mean([x[2] for x in r]):.2f} m | "
            f"torso z {np.mean([x[3] for x in r]):.2f} | fell in {falls}/{len(r)}")


if __name__ == "__main__":
    for name in ("best_getup_e2e1", "best_stand_big1"):
        pol = PPO.load(CHECKPOINT_DIR / f"{name}.zip", device="cpu")
        print(f"{name:18s} neutral pose          : {run(pol)}")
        print(f"{name:18s} neutral +-0.1 rad jitter: {run(pol, jitter=0.1)}")
