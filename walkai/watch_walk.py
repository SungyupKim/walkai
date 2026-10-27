"""Watch a walking policy in the MuJoCo viewer.

DISPLAY=:1 ../standai/.venv/bin/python -I walkai/watch_walk.py <ckpt.zip> [--env getup|walk] [--cmd 0.5] [--seed 0]
Each episode is 20 s; console prints speed and how long the body stood. --cmd omitted = a random command per episode.
"""
import argparse
import sys
import pathlib as _pl
import time

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from walk_env import WalkEnv, WalkGetUpEnv  # noqa: E402
from env import FRAME_SKIP  # noqa: E402
from viewer_utils import hide_opponent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--env", choices=["walk", "getup"], default="getup")
    p.add_argument("--cmd", type=float, default=None)
    p.add_argument("--vmax", type=float, default=0.6)
    p.add_argument("--stand-frac", type=float, default=1.0, help="getup env: share of standing starts (rest crouched / lying)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args()
    model = PPO.load(a.ckpt, device="cpu")
    env = WalkEnv(fixed_cmd=a.cmd, vmax=a.vmax) if a.env == "walk" else WalkGetUpEnv(fixed_cmd=a.cmd, vmax=a.vmax, stand_frac=a.stand_frac)
    dt = env.model.opt.timestep * FRAME_SKIP / a.speed
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        hide_opponent(env.model, viewer)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, -90, -10
        seed = a.seed
        while viewer.is_running():
            obs, _ = env.reset(seed=seed)
            seed += 1
            vxs, ups, steps = [], [], 0
            while viewer.is_running():
                act, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(act)
                vxs.append(info["vx"]); ups.append(info["upright"]); steps += info["real_steps"]
                viewer.cam.lookat[:] = [env.data.xpos[env.a_torso_id][0], 0, 0.8]
                viewer.sync()
                time.sleep(dt)
                if term or trunc:
                    break
            print(f"seed {seed - 1}: cmd {info['cmd']:.2f} m/s, mean vx {np.mean(vxs):+.2f}, upright {np.mean(ups) * 100:.0f}% of the time, "
                  f"moved {info['root_x']:+.2f} m, real steps {steps}", flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
