"""Fixed-seed walking evaluation: for each commanded speed, N 20 s episodes with the deterministic policy.

  ../standai/.venv/bin/python -I walkai/eval_walk.py checkpoints/walk_final.zip --cmds 0 0.3 0.5 0.8 --n 50
'walks' = no fall, still upright at the end, mean vx within 20% of the command (cmd 0: |mean vx| < 0.05), and a
displacement of at least 0.8 * cmd * duration. Several policies can be given to compare on the same seeds.
"""
import argparse
import sys
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import pathlib

import numpy as np
from stable_baselines3 import PPO

from walk_env import WalkEnv, WalkGetUpEnv


def run_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)
    x0 = None
    vxs, ups, fell = [], [], False
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        x0 = info["root_x"] if x0 is None else x0
        vxs.append(info["vx"])
        ups.append(info["upright"])
        if term or trunc:
            fell = term
            late = slice(len(vxs) // 2, None)   # second half: after any get-up
            return dict(late_vx=float(np.mean(vxs[late])), late_up=float(np.mean(ups[late])), fell=fell, upright=info["upright"], mean_vx=float(np.mean(vxs)), dx=info["root_x"] - x0, steps=len(vxs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--cmds", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.8])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--env", choices=["walk", "getup"], default="walk",
                   help="getup: lying/crouched starts, never terminates; walks = upright >=80% and vx within 20% of cmd over the last 10 s")
    p.add_argument("--seed0", type=int, default=777000)
    args = p.parse_args()
    for ck in args.ckpts:
        model = PPO.load(ck, device="cpu")
        print(f"== {ck}")
        for cmd in args.cmds:
            env = WalkEnv(fixed_cmd=cmd) if args.env == "walk" else WalkGetUpEnv(fixed_cmd=cmd)
            rs = [run_episode(model, env, args.seed0 + i) for i in range(args.n)]
            dur = np.mean([r["steps"] for r in rs]) * 0.02
            if args.env == "getup":
                ok = [r["late_up"] >= 0.8 and (abs(r["late_vx"]) < 0.05 if cmd == 0 else abs(r["late_vx"] - cmd) <= 0.2 * cmd) for r in rs]
            else:
              ok = [(not r["fell"] and r["upright"]
                   and (abs(r["mean_vx"]) < 0.05 if cmd == 0 else abs(r["mean_vx"] - cmd) <= 0.2 * cmd and r["dx"] >= 0.8 * cmd * 20.0))
                  for r in rs]
            print(f"cmd {cmd:.1f}: walks {np.mean(ok)*100:5.1f}%  fell {np.mean([r['fell'] for r in rs])*100:5.1f}%  "
                  f"mean vx {np.mean([r['mean_vx'] for r in rs]):+.3f}  dx {np.mean([r['dx'] for r in rs]):+.2f} m  ({dur:.1f}s)")


if __name__ == "__main__":
    main()
