"""Play back the action sequences found by walk_mpc.py in the viewer (playback only, nothing is learned from them).

DISPLAY=:1 ../standai/.venv/bin/python -I walkai/replay_mpc.py checkpoints/walk_mpc2.npz --seeds 1 6 0
"""
import argparse
import sys
import pathlib as _pl
import time

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from walk_env import CHECKPOINT_DIR, WalkGetUpEnv  # noqa: E402
from env import FRAME_SKIP  # noqa: E402
from viewer_utils import hide_opponent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--seeds", type=int, nargs="+", default=[1])
ap.add_argument("--settle", type=int, default=150)
a = ap.parse_args()
d = np.load(a.npz)
seeds = list(d["seeds"])
stand = PPO.load(CHECKPOINT_DIR / "best_getup_e2e1.zip", device="cpu")
env = WalkGetUpEnv(fixed_cmd=0.5, stand_frac=1.0)
dt = env.model.opt.timestep * FRAME_SKIP
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    hide_opponent(env.model, viewer)
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, -90, -10
    while viewer.is_running():
        for s in a.seeds:
            obs, _ = env.reset(seed=s)
            for _ in range(a.settle):
                obs, *_ = env.step(stand.predict(obs, deterministic=True)[0])
                viewer.cam.lookat[:] = [env.data.xpos[env.a_torso_id][0], 0, 0.8]; viewer.sync(); time.sleep(dt)
            x0 = float(env.data.qpos[env.model.joint("a_root_x").qposadr[0]]); steps = 0
            for act in d["acts"][seeds.index(s)]:
                if not viewer.is_running():
                    break
                _, _, _, _, info = env.step(np.concatenate([[0.0], act]).astype(np.float32))
                steps += info["reward_breakdown"]["real_steps"]
                viewer.cam.lookat[:] = [env.data.xpos[env.a_torso_id][0], 0, 0.8]; viewer.sync(); time.sleep(dt)
            print(f"seed {s}: moved {info['root_x'] - x0:+.2f} m in {len(d['acts'][seeds.index(s)]) * dt:.0f} s, counter real steps {steps}", flush=True)
            time.sleep(1.0)
