"""Print the latest values of the main training scalars: tb_tail.py checkpoints/tb/<run>_1"""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); ea.Reload()
for tag in ["time/total_timesteps", "rollout/ep_rew_mean", "rollout/ep_len_mean", "walk/fall_rate", "walk/track_ratio", "walk/vmax", "train/approx_kl", "train/std"]:
    if tag in ea.Tags()["scalars"]:
        ev = ea.Scalars(tag)
        print(f"{tag:24s} " + "  ".join(f"{e.value:.3g}" for e in ev[-4:]))
