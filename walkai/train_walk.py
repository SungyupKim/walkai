"""PPO fine-tune of the stand policy into a walking policy (WalkEnv, commanded forward speed).

  ../standai/.venv/bin/python -I walkai/train_walk.py --out walk1 --timesteps 30_000_000
The speed ceiling (vmax) ramps up automatically while the commanded-walk episodes stay on their feet.
"""
import argparse
import sys
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import pathlib
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from walk_env import CHECKPOINT_DIR, ROOT, WalkEnv, WalkGetUpEnv

ENVS = {"walk": WalkEnv, "getup": WalkGetUpEnv}

OUT_DIR = ROOT / "checkpoints"


class StdFloor(BaseCallback):
    """Keep the policy's action std >= std_min (more exploration for lifting the feet); re-applied before every rollout."""

    def __init__(self, std_min):
        super().__init__()
        self.log_min = float(np.log(std_min))

    def _clamp(self):
        import torch
        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(min=self.log_min)

    def _on_training_start(self):
        self._clamp()

    def _on_rollout_start(self):
        self._clamp()

    def _on_step(self):
        return True


class VmaxCurriculum(BaseCallback):
    """Raise the command ceiling by `step` once recent walk-command episodes rarely fall and track the command."""

    def __init__(self, vmax, vmax_end, step=0.1, window=64, max_fall=0.15, min_track=0.7):
        super().__init__()
        self.vmax, self.vmax_end, self.step, self.window = vmax, vmax_end, step, window
        self.max_fall, self.min_track = max_fall, min_track
        self.sum_vx = None
        self.recent = []

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.sum_vx, self.n_steps, self.cmd, self.up, self.steps = (np.zeros(n) for _ in range(5))
        self.ep_steps = []
        self.training_env.env_method("set_vmax", self.vmax)

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            self.sum_vx[i] += info["vx"]
            self.n_steps[i] += 1
            self.up[i] += info["upright"]
            self.steps[i] += info["real_steps"]
            self.cmd[i] = info["cmd"]
            if done:
                # no-termination env (getup): "fell" = was upright for under 70% of the episode
                fell = (not info.get("TimeLimit.truncated", False)) or self.up[i] / self.n_steps[i] < 0.7
                if self.cmd[i] > 0:
                    track = (self.sum_vx[i] / self.n_steps[i]) / self.cmd[i]
                    self.recent.append((fell, track))
                if self.cmd[i] > 0.1:
                    self.ep_steps.append(self.steps[i])
                self.sum_vx[i] = self.n_steps[i] = self.up[i] = self.steps[i] = 0
        if len(self.recent) >= self.window:
            fall = np.mean([f for f, _ in self.recent])
            track = np.mean([t for f, t in self.recent if not f] or [0.0])
            self.logger.record("walk/fall_rate", fall)
            self.logger.record("walk/track_ratio", track)
            self.logger.record("walk/vmax", self.vmax)
            if self.ep_steps:
                self.logger.record("walk/real_steps_per_ep", float(np.mean(self.ep_steps)))
                self.ep_steps = []
            if fall <= self.max_fall and track >= self.min_track and self.vmax < self.vmax_end:
                self.vmax = round(min(self.vmax + self.step, self.vmax_end), 3)
                self.training_env.env_method("set_vmax", self.vmax)
                print(f"[walk] vmax -> {self.vmax}", flush=True)
            self.recent = []
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="walk")
    p.add_argument("--env", choices=list(ENVS), default="walk", help="getup = lying/crouched starts, no termination, init from best_getup_e2e1")
    p.add_argument("--init-from", default=str(CHECKPOINT_DIR / "best_stand_big1.zip"))
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=14)
    p.add_argument("--learning-rate", type=float, default=5e-5, help="standai: low lr keeps the warm start stable")
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--std-min", type=float, default=0.4, help="floor for the action std (exploration)")
    p.add_argument("--vmax", type=float, default=0.3)
    p.add_argument("--stand-frac", type=float, default=1.0, help="getup env: share of calm standing starts (rest: crouched / lying)")
    p.add_argument("--vmax-end", type=float, default=1.0)
    p.add_argument("--save-freq", type=int, default=1_000_000)
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    run = f"{args.out}_{time.strftime('%Y%m%d_%H%M%S')}"
    vec_env = make_vec_env(ENVS[args.env], n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, env_kwargs={"vmax": args.vmax, **({"stand_frac": args.stand_frac} if args.env == "getup" else {})})
    lr = args.learning_rate
    model = PPO.load(args.init_from, env=vec_env, device="cpu",
                     custom_objects={"learning_rate": lr, "lr_schedule": lambda _p: lr, "target_kl": args.target_kl},
                     tensorboard_log=str(OUT_DIR / "tb"))
    print(f"[walk] warm-start {args.init_from}; run {run}", flush=True)
    model.learn(args.timesteps, reset_num_timesteps=True, tb_log_name=run, log_interval=5, callback=[
        StdFloor(args.std_min),
        VmaxCurriculum(args.vmax, args.vmax_end),
        CheckpointCallback(max(args.save_freq // args.n_envs, 1), str(OUT_DIR / "autosave"), name_prefix=run)])
    model.save(OUT_DIR / f"{args.out}_final.zip")


if __name__ == "__main__":
    main()
