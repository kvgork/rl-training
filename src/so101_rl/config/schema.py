"""Frozen configuration schema (plan F1).

Unit convention: raw servo ticks (4096 per revolution) inside the servo process, radians only in FK/env.
The only [-1, 1] -> ticks scaling in the code base lives in ``env/goal_env.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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

# LeRobot homes every joint so that the middle of its range of motion reads 2047 (post-homing-offset).
# The MJCF (so101_new_calib.xml) is defined relative to the same pose, so qpos = 0 <-> tick CENTER_TICK.
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
    """Present_Position ticks (homing offset already applied by the servo) -> joint angle [rad], shape [6]."""
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
    home_ticks: np.ndarray | None = None  # int32[6]
    rest_ticks: np.ndarray | None = None  # int32[6]; gravity-stable, verified by H5
    torque_off_order: tuple[str, ...] = (
        "gripper",
        "wrist_roll",
        "wrist_flex",
        "elbow_flex",
        "shoulder_lift",
        "shoulder_pan",
    )


@dataclass(frozen=True)
class WorkspaceBox:
    x: tuple[float, float] = (0.08, 0.34)
    y: tuple[float, float] = (-0.22, 0.22)
    z: tuple[float, float] = (0.010, 0.28)


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
