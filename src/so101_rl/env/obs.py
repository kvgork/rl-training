"""Frozen policy / critic observation layout (FS3 of ``plans/sim-track-implementation-plan.md``; supersedes F7's
15-D ``observation``). One definition for the batched sim env (torch) and the real-arm env (numpy).

Policy observation (``PolicyObsLayout``), in order:

=================  ====  =================================================================================
``q_norm``            6  ``(present_ticks - CENTER_TICK) / 2048`` for all six joints (gripper included)
``dq_norm``           6  ``velocity / servo_hz / max_ticks_per_tick`` clipped to [-2, 2]; ``velocity`` is the
                         bus/state-record Present_Speed in ticks per SECOND (``MujocoBus``, STS3215), so the
                         ratio is ticks per servo tick over the per-tick velocity cap
``last_applied``      6  ``applied_delta_ticks / max_delta_ticks`` clipped to [-1, 1] (post-clamp, L4)
``tip_xyz``           3  So101FK tip from PRESENT ticks, base frame, metres
``die_xyz``           3  last-known die centroid, base frame (zeros until first seen)        [die tasks]
``die_visible``       1  1.0 if the tracker saw the die this policy step, else 0.0          [die tasks]
``die_age``           1  ``min(frames_since_seen, 30) / 30``                                [die tasks]
=================  ====  =================================================================================

Gymnasium ``Dict`` contract (F7) is unchanged: ``observation`` = this layout, ``achieved_goal`` = tip xyz (reach)
or last-known die xyz (die), ``desired_goal`` = goal xyz / zone centre; the policy input is ``observation`` with
``desired_goal`` appended (``policy_input_dim``). The privileged critic layout is sim-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from so101_rl.config.schema import CENTER_TICK, So101Config

if TYPE_CHECKING:
    import torch

OBS_LAYOUT_VERSION: Final[int] = 1
GOAL_DIM: Final[int] = 3
Q_SCALE_TICKS: Final[float] = 2048.0
DQ_CLIP: Final[float] = 2.0
DIE_AGE_MAX_FRAMES: Final[int] = 30


@dataclass(frozen=True)
class PolicyObsLayout:
    q_norm: slice
    dq_norm: slice
    last_applied: slice
    tip_xyz: slice
    die_xyz: slice | None
    die_visible: slice | None
    die_age: slice | None
    dim: int

    @property
    def has_die(self) -> bool:
        return self.die_xyz is not None


REACH_OBS: Final[PolicyObsLayout] = PolicyObsLayout(
    q_norm=slice(0, 6),
    dq_norm=slice(6, 12),
    last_applied=slice(12, 18),
    tip_xyz=slice(18, 21),
    die_xyz=None,
    die_visible=None,
    die_age=None,
    dim=21,
)
DIE_OBS: Final[PolicyObsLayout] = PolicyObsLayout(
    q_norm=slice(0, 6),
    dq_norm=slice(6, 12),
    last_applied=slice(12, 18),
    tip_xyz=slice(18, 21),
    die_xyz=slice(21, 24),
    die_visible=slice(24, 25),
    die_age=slice(25, 26),
    dim=26,
)


def policy_input_dim(layout: PolicyObsLayout) -> int:
    """``layout.dim + GOAL_DIM``: desired_goal appended (the TD-MPC2 "state" / SB3 flattened input)."""
    return layout.dim + GOAL_DIM


@dataclass(frozen=True)
class DieObs:
    """Tracker block for one env (numpy). ``xyz`` is the HELD last-known centroid while invisible."""

    xyz: np.ndarray  # float[..., 3], base frame, zeros until first seen
    visible: np.ndarray | bool  # bool[...]
    age_frames: np.ndarray | int  # int[...], camera frames since last seen (0 when visible)


@dataclass(frozen=True)
class DieObsT:
    """Tracker block for N worlds (torch)."""

    xyz: torch.Tensor  # float32[N, 3]
    visible: torch.Tensor  # bool[N]
    age_frames: torch.Tensor  # int64[N]


@dataclass(frozen=True)
class PrivObsLayout:
    q_true: slice  # 6  joint angles (rad), unquantised
    dq_true: slice  # 6  rad/s
    tip_xyz_true: slice  # 3
    die_pos: slice  # 3  true die centre
    die_quat: slice  # 4  wxyz
    die_vel: slice  # 3  linear velocity
    die_in_gripper: slice  # 1  jaw_col geoms touching the die this step
    die_on_table: slice  # 1
    dr_params: slice  # 33: (kp, kv, damping, frictionloss, armature) x 6 / centre, delay_ticks/5, die_mass, friction
    tracker_delay: slice  # 1  frames / 3
    dim: int  # 61


PRIV_OBS: Final[PrivObsLayout] = PrivObsLayout(
    q_true=slice(0, 6),
    dq_true=slice(6, 12),
    tip_xyz_true=slice(12, 15),
    die_pos=slice(15, 18),
    die_quat=slice(18, 22),
    die_vel=slice(22, 25),
    die_in_gripper=slice(25, 26),
    die_on_table=slice(26, 27),
    dr_params=slice(27, 60),
    tracker_delay=slice(60, 61),
    dim=61,
)


def _check_die(layout: PolicyObsLayout, die: object) -> None:
    if layout.has_die and die is None:
        raise ValueError("this layout has a die block: pass die=")
    if not layout.has_die and die is not None:
        raise ValueError("this layout has no die block: die must be None")


def build_policy_obs_np(
    layout: PolicyObsLayout,
    *,
    present_ticks: np.ndarray,
    velocity: np.ndarray,
    applied_delta_ticks: np.ndarray,
    tip_xyz: np.ndarray,
    cfg: So101Config,
    die: DieObs | None,
    out: np.ndarray,
) -> np.ndarray:
    """Fill ``out`` (``[..., layout.dim]``, any leading shape) and return it. See the module table.

    ``velocity`` is Present_Speed in ticks per second (bus / state record); it is converted to ticks per servo tick.
    """
    _check_die(layout, die)
    if out.shape[-1] != layout.dim:
        raise ValueError(f"out has {out.shape[-1]} columns, layout needs {layout.dim}")
    per_tick = cfg.action.max_ticks_per_tick.astype(np.float64)
    max_delta = cfg.action.max_delta_ticks.astype(np.float64)
    out[..., layout.q_norm] = (np.asarray(present_ticks, np.float64) - CENTER_TICK) / Q_SCALE_TICKS
    dq = np.asarray(velocity, np.float64) / cfg.rates.servo_hz / per_tick
    out[..., layout.dq_norm] = np.clip(dq, -DQ_CLIP, DQ_CLIP)
    out[..., layout.last_applied] = np.clip(np.asarray(applied_delta_ticks, np.float64) / max_delta, -1.0, 1.0)
    out[..., layout.tip_xyz] = tip_xyz
    if die is not None:
        out[..., layout.die_xyz] = die.xyz
        out[..., layout.die_visible] = np.asarray(die.visible, np.float64)[..., None]
        age = np.minimum(np.asarray(die.age_frames, np.float64), DIE_AGE_MAX_FRAMES) / DIE_AGE_MAX_FRAMES
        out[..., layout.die_age] = age[..., None]
    return out


_TORCH_SCALES: dict[tuple[int, str], tuple[object, ...]] = {}


def _torch_scales(cfg: So101Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``max_ticks_per_tick`` / ``max_delta_ticks`` as float64 tensors, built once per (cfg, device): no host->device
    copy per step (CUDA-graph friendly). The cfg is kept alive in the cache so its id cannot be reused."""
    import torch

    key = (id(cfg), str(device))
    hit = _TORCH_SCALES.get(key)
    if hit is None or hit[0] is not cfg:
        per_tick = torch.tensor(cfg.action.max_ticks_per_tick.tolist(), dtype=torch.float64, device=device)
        max_delta = torch.tensor(cfg.action.max_delta_ticks.tolist(), dtype=torch.float64, device=device)
        hit = _TORCH_SCALES[key] = (cfg, per_tick, max_delta)
    return hit[1], hit[2]


def build_policy_obs_torch(
    layout: PolicyObsLayout,
    *,
    present_ticks: torch.Tensor,
    velocity: torch.Tensor,
    applied_delta_ticks: torch.Tensor,
    tip_xyz: torch.Tensor,
    cfg: So101Config,
    die: DieObsT | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Same semantics as :func:`build_policy_obs_np` for ``[N, ...]`` tensors; fills and returns ``out [N, dim]``."""
    import torch

    _check_die(layout, die)
    if out.shape[-1] != layout.dim:
        raise ValueError(f"out has {out.shape[-1]} columns, layout needs {layout.dim}")
    dt = out.dtype
    per_tick, max_delta = _torch_scales(cfg, out.device)
    q = (present_ticks.to(torch.float64) - CENTER_TICK) / Q_SCALE_TICKS
    dq = (velocity.to(torch.float64) / cfg.rates.servo_hz / per_tick).clamp(-DQ_CLIP, DQ_CLIP)
    last = (applied_delta_ticks.to(torch.float64) / max_delta).clamp(-1.0, 1.0)
    out[..., layout.q_norm] = q.to(dt)
    out[..., layout.dq_norm] = dq.to(dt)
    out[..., layout.last_applied] = last.to(dt)
    out[..., layout.tip_xyz] = tip_xyz.to(dt)
    if die is not None:
        out[..., layout.die_xyz] = die.xyz.to(dt)
        out[..., layout.die_visible] = die.visible.to(dt).unsqueeze(-1)
        age = die.age_frames.to(torch.float64).clamp(max=DIE_AGE_MAX_FRAMES) / DIE_AGE_MAX_FRAMES
        out[..., layout.die_age] = age.to(dt).unsqueeze(-1)
    return out

