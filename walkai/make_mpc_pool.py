"""Collect MID-STRIDE start states from the MPC solutions (state distribution only; no actions are ever imitated).

  ../standai/.venv/bin/python -I walkai/make_mpc_pool.py --out checkpoints/mpc_pool.npz
Replays every saved MPC action sequence through WalkGetUpEnv (same command and stand-policy settle as when it was computed) and keeps the
full MuJoCo (qpos, qvel) of upright states where exactly one foot is swinging and the other is planted. Swings of the REAR foot that have not yet
passed the other foot (the moment a crossing step is decided) are kept in full; other swing states are subsampled.
"""
import argparse
import sys
import pathlib as _pl

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from walk_env import CHECKPOINT_DIR, WalkGetUpEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", default=["checkpoints/walk_mpc2.npz", "checkpoints/walk_mpc_narrow.npz", "checkpoints/walk_mpc_narrow2.npz"])
ap.add_argument("--out", default="checkpoints/mpc_pool.npz")
a = ap.parse_args()
stand = PPO.load(CHECKPOINT_DIR / "best_getup_e2e1.zip", device="cpu")
rng = np.random.default_rng(0)
Q, V, tag = [], [], []
for f in a.files:
    d = np.load(f)
    seeds = [int(s) for s in d["seeds"]]
    acts = d["acts"]
    if len(acts) != len(seeds):                      # only the full-length solutions were stored
        print(f"skip {f}: {len(acts)} action sets for {len(seeds)} seeds"); continue
    for k, seed in enumerate(seeds):
        e = WalkGetUpEnv(fixed_cmd=0.5, stand_frac=1.0)
        o, _ = e.reset(seed=seed)
        for _ in range(150):
            o, *_ = e.step(stand.predict(o, deterministic=True)[0])
        n_rear = n_other = 0
        for t, act in enumerate(acts[k]):
            _, _, _, _, info = e.step(np.concatenate([[0.0], act]).astype(np.float32))
            if t < 40 or not info["upright"]:
                continue
            z = [float(e.data.site_xpos[s][2]) for s in e.a_feet]
            x = [float(e.data.site_xpos[s][0]) for s in e.a_feet]
            sw = [i for i in (0, 1) if z[i] > 0.15 and z[1 - i] < 0.12]
            if len(sw) != 1:
                continue
            i = sw[0]
            rear_pending = x[i] < x[1 - i] + 0.3          # swing foot has not yet got 0.3 m ahead of the planted one
            if rear_pending or rng.random() < 0.25:
                Q.append(e.data.qpos.copy()); V.append(e.data.qvel.copy()); tag.append(int(rear_pending))
                n_rear += rear_pending; n_other += (not rear_pending)
        print(f"{f} seed {seed}: kept {n_rear} rear-swing + {n_other} other states", flush=True)
np.savez(a.out, qpos=np.array(Q), qvel=np.array(V), rear=np.array(tag))
print(f"pool: {len(Q)} states ({int(np.sum(tag))} rear-foot swings) -> {a.out}")
