"""WalkEnv: stand-policy-compatible environment for learning to walk forward at a commanded speed.

Built on standai's LocomotionEnv(stand_only=True), so observations (70) and actions (13) are the same as
the stand policy's and it can be warm-started from `standai/checkpoints/best_stand_big1.zip`.
stand_only already pins the unphysical root_x thrust to 0, so all motion comes from the legs.

Per episode a forward speed command v* is drawn (the obs's target_vx slot) and kept for the whole episode;
a fraction of episodes command 0 so the policy keeps its stand. Starts are calm (near-standing) only.
Reward = standai's stand reward with the terms that fight a gait scaled down, plus a Gaussian
velocity-tracking term that only pays while the body is genuinely upright.
"""
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
STANDAI_SCRIPTS = ROOT / "standai" / "scripts"
if str(STANDAI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STANDAI_SCRIPTS))

import numpy as np  # noqa: E402

from env import FALL_HEIGHT  # noqa: E402
from getup_env import GetUpEnv  # noqa: E402
from locomotion_env import LocomotionEnv  # noqa: E402

CHECKPOINT_DIR = pathlib.Path(os.environ.get("STANDAI_CHECKPOINTS", ROOT.parent / "standai" / "checkpoints"))

WALK_VEL_SCALE = 5.0        # peak per-step reward of the velocity-tracking term
WALK_VEL_SIGMA = 0.2        # m/s; narrow so standing still against a real command earns little
STOP_FRAC = 0.2             # share of episodes commanded v* = 0 (keep the stand)
MIN_CMD = 0.2               # nonzero commands are drawn from [MIN_CMD, vmax]
UPRIGHT_PITCH_DEG = 30.0
# Stand-reward terms that penalise a gait (feet-inside-COM support, "feet under COM", always-straight knees).
# Kept at this fraction of their stand value.
GAIT_FRIENDLY = {"com": 0.3, "step": 0.0, "knee_straight": 0.3}


# ---- gait shaping: pay for real steps, not for shaking the legs ----
FOOT_LIFT_Z = 0.20          # foot site z (m) above which the foot counts as lifted (planted ~0.165)
STEP_MIN_AIR_S = 0.2        # a "real" step: a foot stays lifted at least this long, then lands with the other foot planted
STEP_AIR_CAP_S = 0.4        # airtime beyond this earns nothing extra
STEP_BONUS = 40.0           # reward at touchdown per second of airtime above STEP_MIN_AIR_S (capped)
STEP_GATE_S = 1.0           # velocity reward is full only within this long after the last real step ...
NO_STEP_GATE = 0.2          # ... and this fraction of it otherwise (vibrating / sliding forward pays little)
ACTION_RATE_K = 1.0         # penalty * mean((a_t - a_{t-1})^2): high-frequency vibration is expensive
DT = 0.02


class GaitTracker:
    """Per-foot airtime bookkeeping shared by WalkEnv and WalkGetUpEnv."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.air = [0.0, 0.0]
        self.since_step = 10 ** 6
        self.prev_action = None

    def update(self, env, action, moving):
        """Returns (step_bonus, vel_gate, action_penalty, n_real_steps_this_step)."""
        lifted = [float(env.data.site_xpos[s][2]) > FOOT_LIFT_Z for s in env.a_feet]
        bonus, real = 0.0, 0
        for f in (0, 1):
            if lifted[f]:
                self.air[f] += DT
            elif self.air[f] > 0.0:                       # touchdown
                if self.air[f] >= STEP_MIN_AIR_S and not lifted[1 - f] and moving:
                    bonus += STEP_BONUS * (min(self.air[f], STEP_AIR_CAP_S) - STEP_MIN_AIR_S + 0.05)
                    real += 1
                self.air[f] = 0.0
        self.since_step = 0 if real else self.since_step + 1
        gate = 1.0 if (not moving or self.since_step * DT <= STEP_GATE_S) else NO_STEP_GATE
        a = np.asarray(action, dtype=np.float32)
        pen = 0.0 if self.prev_action is None else ACTION_RATE_K * float(np.mean((a[1:] - self.prev_action[1:]) ** 2))
        self.prev_action = a.copy()
        return bonus, gate, pen, real


def draw_cmd(rng, vmax, fixed_cmd=None):
    """Forward-speed command for one episode."""
    if fixed_cmd is not None:
        return float(fixed_cmd)
    if rng.random() < STOP_FRAC:
        return 0.0
    return float(rng.uniform(min(MIN_CMD, vmax), vmax))


def walk_reward(env, upright, action):
    """Velocity tracking (paid only while upright, gated on real steps) + step bonus - action-rate penalty.
    Returns (total reward, vx, breakdown dict)."""
    vx = float(env.data.qvel[env._a_root_x_dof])
    moving = env._target_vx > 0.1
    bonus, gate, pen, real = env._gait.update(env, action, moving)
    vel = WALK_VEL_SCALE * float(np.exp(-((vx - env._target_vx) ** 2) / (2.0 * WALK_VEL_SIGMA ** 2))) * upright * gate
    bonus *= upright
    return vel + bonus - pen, vx, {"walk_vel": vel, "step_bonus": bonus, "action_rate": -pen, "real_steps": real}


class WalkEnv(LocomotionEnv):
    def __init__(self, render_mode=None, vmax=0.5, fixed_cmd=None, horizon=1000):
        super().__init__(render_mode=render_mode, stand_only=True)
        self._calm_frac = 1.0
        self._vmax = float(vmax)
        self._fixed_cmd = fixed_cmd
        self._horizon = int(horizon)
        self._t = 0
        self._gait = GaitTracker()

    def set_vmax(self, vmax):
        self._vmax = float(vmax)

    def _resample_target_vx(self):
        self._target_vx = draw_cmd(self.np_random, self._vmax, self._fixed_cmd)

    def reset(self, seed=None, options=None):
        self._t = 0
        self._gait.reset()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._t += 1
        bd = info["reward_breakdown"]
        for key, keep in GAIT_FRIENDLY.items():
            reward -= (1.0 - keep) * bd.get(key, 0.0)
            bd[key] = keep * bd.get(key, 0.0)

        pitch = (self.data.qpos[self.a_ry_qpos] + np.pi) % (2 * np.pi) - np.pi
        upright = bool(self.data.xpos[self.a_torso_id][2] >= FALL_HEIGHT
                       and abs(pitch) < np.deg2rad(UPRIGHT_PITCH_DEG))
        walk, vx, parts = walk_reward(self, upright, action)
        reward += walk
        bd.update(parts)
        info.update(real_steps=parts["real_steps"], vx=vx, upright=upright, cmd=self._target_vx, root_x=float(self.data.qpos[self.model.joint("a_root_x").qposadr[0]]))
        if self._t >= self._horizon and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


class WalkGetUpEnv(GetUpEnv):
    """Walking on top of the single get-up network (standai `best_getup_e2e1`, same 70-dim obs / 13 actions).

    No termination on a fall: the episode runs the full horizon (20 s) and the policy has to get up again.
    Starts: with probability `stand_frac` a calm standing start (walking can be practised from t=0); otherwise
    lying (supine / prone, 50:50) or, with probability `crouch_frac` of those, a crouched near-standing start.
    Reward = GetUpEnv's standing reward + the velocity-tracking term while standing (torso >= 0.95 m, |pitch| < 30 deg).
    """

    def __init__(self, vmax=0.5, fixed_cmd=None, crouch_frac=0.5, stand_frac=0.5, **kw):
        super().__init__(**kw)
        self._stand_frac = float(stand_frac)
        self._fixed_start = None   # None = random mix, 'stand' / 'lying' to force one kind (evaluation)
        self._crouch_frac = float(crouch_frac)
        self._vmax = float(vmax)
        self._fixed_cmd = fixed_cmd
        self._gait = GaitTracker()

    def set_vmax(self, vmax):
        self._vmax = float(vmax)

    def _resample_target_vx(self):
        self._target_vx = draw_cmd(self.np_random, self._vmax, self._fixed_cmd)

    def reset(self, seed=None, options=None):
        self._gait.reset()
        if seed is not None:
            self._side_rng = np.random.default_rng(seed + 7919)
        if self._fixed_start != "stand" and not (self._stand_draw(seed) and self._fixed_start is None):
            return super().reset(seed=seed, options=options)
        # calm standing start: LocomotionEnv's own reset, then FloorEnv's per-episode bookkeeping that step() reads
        self._t = 0
        self._calm_frac = 1.0
        obs, info = LocomotionEnv.reset(self, seed=seed, options=options)
        self._calm_frac = 0.0
        self._start_kind, self._start_side = "stand", "pos"
        head = float(self.data.xpos[self.a_head_id][2])
        self._floor_head_best = self._floor_head_max = self._prev_head_z = head
        self._hold = 0
        self._arr_prev_com_x = None
        return obs, info

    def _stand_draw(self, seed):
        rng = self._side_rng if seed is not None else self.np_random
        return rng.random() < self._stand_frac

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        walk, vx, parts = walk_reward(self, info["standing"], action)
        reward += walk
        info.setdefault("reward_breakdown", {}).update(parts)
        info.update(real_steps=parts["real_steps"], vx=vx, upright=info["standing"], cmd=self._target_vx,
                    root_x=float(self.data.qpos[self.model.joint("a_root_x").qposadr[0]]))
        return obs, reward, terminated, truncated, info
