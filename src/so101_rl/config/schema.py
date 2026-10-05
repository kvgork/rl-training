"""Frozen configuration schema (plan F1).

Unit convention: raw servo ticks (4096 per revolution) inside the servo process, radians only in FK/env.
The only [-1, 1] -> ticks scaling in the code base lives in ``env/goal_env.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

TICKS_PER_REV: Final[int] = 4096
JOINT_NAMES: Final[tuple[str, ...]] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
N_JOINTS: Final[int] = 6

# LeRobot homing puts the middle of the range of motion of the ARM joints at 2047 (the servo firmware applies
# homing_offset, so Present_Position is already homed and the conversions below never touch homing_offset).
# The MJCF (so101_new_calib.xml) is defined relative to the same pose, so arm qpos = 0 <-> tick CENTER_TICK.
# The gripper's range is NOT centred on 2047; its 0-100 % <-> rad mapping lives in sim/ (uses range_min/max).
CENTER_TICK: Final[int] = 2047
_RAD_PER_TICK: Final[float] = 2.0 * math.pi / TICKS_PER_REV


@dataclass(frozen=True)
class MotorCalibration:
    """One entry of the LeRobot calibration JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def _signs(calib: dict[str, MotorCalibration]) -> np.ndarray:
    return np.array([-1.0 if calib[n].drive_mode else 1.0 for n in JOINT_NAMES], dtype=np.float64)


def ticks_to_rad(ticks: np.ndarray, calib: dict[str, MotorCalibration]) -> np.ndarray:
    """Present_Position ticks (homing offset already applied by the servo) -> joint angle [rad], shape [6].

    Arm joints only: the gripper is not centred on CENTER_TICK (see above).
    """
    t = np.asarray(ticks, dtype=np.float64)
    return _signs(calib) * (t - CENTER_TICK) * _RAD_PER_TICK


def rad_to_ticks(q: np.ndarray, calib: dict[str, MotorCalibration]) -> np.ndarray:
    """Inverse of :func:`ticks_to_rad`. int32, rounded, deliberately NOT clipped to 0..4095."""
    q = np.asarray(q, dtype=np.float64)
    return np.rint(_signs(calib) * q / _RAD_PER_TICK + CENTER_TICK).astype(np.int32)


@dataclass(frozen=True, eq=False)
class JointLimits:
    """Ticks, int32[6]; intersection of config and calibration range_min/max."""

    min_ticks: np.ndarray
    max_ticks: np.ndarray


@dataclass(frozen=True, eq=False)
class ActionConfig:
    max_delta_ticks: np.ndarray  # int32[6]; |delta| per 10 Hz command
    tracking_error_ticks: np.ndarray  # E, int32[6]
    max_ticks_per_tick: np.ndarray  # per 30 Hz tick velocity cap
    interp_ticks: int = 3


@dataclass(frozen=True)
class ThermalConfig:
    trip_c: float = 60.0
    resume_c: float = 50.0
    resume_dwell_s: float = 30.0
    max_trips_per_hour: int = 3
    duty_on_s: float = 1200.0
    duty_off_s: float = 600.0
    stale_temp_s: float = 1.0  # temperature older than this is a fault


@dataclass(frozen=True)
class RateConfig:
    servo_hz: float = 30.0
    policy_hz: float = 10.0
    stale_cmd_ms: float = 150.0
    max_consecutive_misses: int = 5
    heartbeat_timeout_ms: int = 1000
    max_bus_failures: int = 3  # consecutive read/write failures -> LATCHED


@dataclass(frozen=True, eq=False)
class SafeStateConfig:
    bound_s: float = 5.0
    ramp_ticks_per_tick: int = 8
    home_ticks: np.ndarray | None = None  # int32[6]; FK inside the box (loader-checked)
    rest_ticks: np.ndarray | None = None  # int32[6]; gravity-stable, verified by H5
    torque_off_order: tuple[str, ...] = (
        "gripper",
        "wrist_roll",
        "wrist_flex",
        "elbow_flex",
        "shoulder_lift",
        "shoulder_pan",
    )
    # Transit zone (review B1): the rest pose is outside the box, so the ramps between rest and home are checked
    # against the table instead. Tip z must stay >= table_z_m + table_clearance_m; below
    # table_z_m + table_clearance_m + transit_zone_m, joints that lower the tip wait for the lifting joints.
    table_z_m: float = -0.02
    table_clearance_m: float = 0.005
    transit_zone_m: float = 0.05
    home_box_margin_m: float = 0.010  # FK(home) at least this far inside every box face
    rest_tolerance_ticks: int = 100  # reenergize starts only within this of rest (arm joints) or inside the box


@dataclass(frozen=True)
class WorkspaceBox:
    x: tuple[float, float] = (0.08, 0.34)
    y: tuple[float, float] = (-0.22, 0.22)
    z: tuple[float, float] = (0.010, 0.28)
    # BOX_PRESENT trips only when the PRESENT tip is more than this outside the box (review S3: 1 mm of sag at
    # the floor must not end the run). Commanded targets are still clamped to the exact box.
    present_tolerance_m: float = 0.005


@dataclass(frozen=True)
class RegisterConfig:
    """Written in configure(); ALL of them, every connect."""

    return_delay_time: int = 0
    operating_mode: int = 0  # POSITION
    p_coefficient: int = 16
    i_coefficient: int = 0
    d_coefficient: int = 32
    maximum_acceleration: int = 254
    acceleration: int = 254
    protection_current: int = 250  # gripper default from lerobot; arm value measured in H6
    overload_torque: int = 25
    over_current_protection_time: int = 20  # UNVERIFIED default; H6 proves fold-back


@dataclass(frozen=True)
class SimWarpConfig:
    """MuJoCo Warp buffer sizes (FS1, sim-first track). Measured in the S0 spike (2026-10-05)."""

    njmax: int = 256  # constraint rows per world; the MJWarp default 64 overflows (peak nefc 188 measured)
    nconmax_per_world: int = 32  # contacts per world (CPU max ncon 18); naconmax = nconmax_per_world * nworld
    nworld_default: int = 1024
    overflow_check_every_steps: int = 10  # policy steps between device-side overflow reads (one host sync each)
    overflow_policy: Literal["fail"] = "fail"  # there is deliberately no "warn" value


# FS4 reward terms (plan §10 2026-10-06 "Reward structure"): name -> the term's own parameter keys in so101.yaml
# (``weight`` is common to all). The implementations live in ``env/reward.py`` (test-enforced: same names).
REWARD_TERM_PARAMS: Final[dict[str, frozenset[str]]] = {
    "distance": frozenset({"scale_m", "zero_on_success"}),
    "success": frozenset(),
    "clamp": frozenset(),
    "action_l2": frozenset(),
    "time": frozenset(),
}
TASK_NAMES: Final[tuple[str, ...]] = ("reach", "die_to_zone")


@dataclass(frozen=True)
class RewardTermSpec:
    """One weighted reward term. ``scale_m`` / ``zero_on_success`` are used by ``distance`` only."""

    name: str
    weight: float  # >= 0; every term's value is <= 0, so a negative weight would pay a positive reward
    scale_m: float | None = None
    zero_on_success: bool = False


@dataclass(frozen=True)
class RewardSpec:
    """Per-task reward: ``sum(weight * term)``. Validated by ``env.reward.validate_reward_spec``."""

    terms: tuple[RewardTermSpec, ...]

    def term(self, name: str) -> RewardTermSpec | None:
        return next((t for t in self.terms if t.name == name), None)


@dataclass(frozen=True)
class TaskConfig:
    """The configurable part of an FS4 task (``tasks.<name>`` in so101.yaml); the rest is frozen in env/tasks.py."""

    reward: RewardSpec
    max_height_m: float | None = None  # die: centroid at most this far above the goal (plan §10 2026-10-06)
    settle_mm: float | None = None  # die: tracker displacement between the debounce steps (plan §10 2026-10-06)


@dataclass(frozen=True)
class TasksConfig:
    reach: TaskConfig
    die_to_zone: TaskConfig


@dataclass(frozen=True, eq=False)
class So101Config:
    port: str
    heartbeat_port: str | None
    calibration_path: Path
    shm_name: str
    joints: JointLimits
    action: ActionConfig
    thermal: ThermalConfig
    rates: RateConfig
    safe_state: SafeStateConfig
    box: WorkspaceBox
    registers: RegisterConfig
    kinematics_path: Path  # config/so101_kinematics.yaml
    sim_warp: SimWarpConfig  # FS1 (sim-first track)
    tasks: TasksConfig  # FS4 per-task reward and die-success parameters
