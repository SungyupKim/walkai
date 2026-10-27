# walkai

Walking on top of [`standai`](https://github.com/SungyupKim/standai) (git submodule in `standai/`): the same 2D MuJoCo biped
(`standai/models/fighter2d.xml`), warm-started from the stand policy `../standai/checkpoints/best_stand_big1.zip`
(checkpoints are not in git, so they are read from the sibling `../standai` folder; override with `STANDAI_CHECKPOINTS`).

Goal 1: from standing, walk forward at a commanded speed (0.3-1.0 m/s) for 20 s without falling; a share of episodes command 0 to keep the stand.

```bash
git clone --recurse-submodules <this repo>        # or: git submodule update --init
PY=../standai/.venv/bin/python
$PY -I walkai/train_walk.py --out walk1            # PPO, lr 5e-5, vmax curriculum 0.3 -> 1.0
$PY -I walkai/tb_tail.py checkpoints/tb/<run>_1    # latest scalars (fall_rate, track_ratio, vmax)
$PY -I walkai/eval_walk.py checkpoints/walk1_final.zip --cmds 0 0.3 0.5 0.8 --n 50
```

Files: `walkai/walk_env.py` (WalkEnv = standai LocomotionEnv(stand_only) + velocity-tracking reward; gait-hostile stand terms scaled down),
`train_walk.py`, `eval_walk.py`. Baseline (stand policy, 10 seeds, 20 s): cmd 0 holds 70%, cmd 0.5 walks 0%.
