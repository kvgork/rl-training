"""Gripper 0-100 % <-> joint angle [rad] <-> servo ticks (plan F10).

Unlike the arm joints the gripper range is not centred on ``CENTER_TICK``, so it does not go through
``ticks_to_rad``. The mapping is linear between the two ends of its motion:

* 0 %   = closed = ``range_min`` tick (calibration) = lower end of the MJCF gripper joint range
* 100 % = open   = ``range_max`` tick (calibration) = upper end of the MJCF gripper joint range

If the calibration says ``drive_mode = 1`` the tick direction is reversed (0 % at ``range_max``).
The joint range is read from ``config/so101_kinematics.yaml`` (generated from the upstream MJCF), not hard-coded.
"""

from __future__ import annotations

from functools import cache

import yaml

from so101_rl.config import DEFAULT_CONFIG_PATH, MotorCalibration


@cache
def gripper_range_rad() -> tuple[float, float]:
    """(closed, open) joint angle of the MJCF gripper joint, from the kinematics yaml."""
    kin = yaml.safe_load((DEFAULT_CONFIG_PATH.parent / "so101_kinematics.yaml").read_text())
    (joint,) = (j for j in kin["joints"] if j["name"] == "gripper")
    lo, hi = joint["range_rad"]
    return float(lo), float(hi)


def _check_percent(p: float) -> None:
    if not 0.0 <= float(p) <= 100.0:  # also rejects NaN
        raise ValueError(f"gripper percent must be within [0, 100], got {p!r}")


def gripper_percent_to_rad(p: float) -> float:
    """0 % (closed) .. 100 % (open) -> joint angle. Raises ``ValueError`` outside 0..100 (or NaN)."""
    _check_percent(p)
    lo, hi = gripper_range_rad()
    return lo + float(p) / 100.0 * (hi - lo)


def gripper_rad_to_percent(q: float) -> float:
    """Joint angle -> percent, clipped to 0..100 (sim positions may overshoot the limit by solver slack)."""
    lo, hi = gripper_range_rad()
    return min(100.0, max(0.0, (float(q) - lo) / (hi - lo) * 100.0))


def gripper_ticks_to_percent(t: float, calib: dict[str, MotorCalibration]) -> float:
    """Present_Position ticks of the gripper motor -> percent (clipped to 0..100)."""
    c = calib["gripper"]
    frac = (float(t) - c.range_min) / (c.range_max - c.range_min)
    if c.drive_mode:
        frac = 1.0 - frac
    return min(100.0, max(0.0, frac * 100.0))


def gripper_percent_to_ticks(p: float, calib: dict[str, MotorCalibration]) -> float:
    """Percent -> ticks (float, unrounded). Raises ``ValueError`` outside 0..100."""
    _check_percent(p)
    c = calib["gripper"]
    frac = float(p) / 100.0
    if c.drive_mode:
        frac = 1.0 - frac
    return c.range_min + frac * (c.range_max - c.range_min)


def gripper_ticks_to_rad(t: float, calib: dict[str, MotorCalibration]) -> float:
    return gripper_percent_to_rad(gripper_ticks_to_percent(t, calib))


def gripper_rad_to_ticks(q: float, calib: dict[str, MotorCalibration]) -> int:
    return round(gripper_percent_to_ticks(gripper_rad_to_percent(q), calib))


def gripper_ticks_per_rad(calib: dict[str, MotorCalibration]) -> float:
    """Magnitude of d(ticks)/d(rad) for the gripper."""
    c = calib["gripper"]
    lo, hi = gripper_range_rad()
    return abs(c.range_max - c.range_min) / (hi - lo)


__all__ = [
    "gripper_percent_to_rad",
    "gripper_percent_to_ticks",
    "gripper_rad_to_percent",
    "gripper_rad_to_ticks",
    "gripper_range_rad",
    "gripper_ticks_per_rad",
    "gripper_ticks_to_percent",
    "gripper_ticks_to_rad",
]
