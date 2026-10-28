"""Overnight report: for the newest checkpoint of each run, 6 standing starts x 8 s at a fixed command, deterministic policy.

  ../standai/.venv/bin/python -I walkai/report.py walk11a walk11b [--cmd 0.4]
Prints upright %, distance, real steps (counter), single-support share, gait quality, and the training log's latest numbers.
"""
import argparse
import glob
import os
import sys
import pathlib as _pl

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from walk_env import WalkGetUpEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--cmd", type=float, default=0.4)
ap.add_argument("--seeds", type=int, default=6)
a = ap.parse_args()
for run in a.runs:
    cks = sorted(glob.glob(f"checkpoints/autosave/{run}_*_steps.zip"), key=os.path.getmtime)
    if not cks:
        print(f"{run}: no checkpoint yet"); continue
    ck = cks[-1]
    m = PPO.load(ck, device="cpu")
    e = WalkGetUpEnv(fixed_cmd=a.cmd, stand_frac=1.0)
    rows = []
    for sd in range(a.seeds):
        o, _ = e.reset(seed=900 + sd)
        x0 = float(e.data.qpos[e.model.joint("a_root_x").qposadr[0]]); st = 0; cr = 0; up = []; ss = []; q = []
        for _ in range(400):
            o, r, te, tr, i = e.step(m.predict(o, deterministic=True)[0])
            b = i["reward_breakdown"]; st += b["real_steps"]; cr += b["cross_steps"]; up.append(i["upright"]); ss.append(b["single"] > 0); q.append(b["quality"])
        rows.append((np.mean(up), i["root_x"] - x0, st, np.mean(ss), np.mean(q), cr))
    r = np.array(rows).mean(0)
    print(f"{run} [{os.path.basename(ck).split('_')[-2]} steps] cmd {a.cmd}: upright {r[0]*100:3.0f}% | moved {r[1]:+.2f} m/8s | real steps {r[2]:.1f} (rear foot passes {r[5]:.1f}) | single-support {r[3]*100:3.0f}% | quality {r[4]:.2f}", flush=True)
