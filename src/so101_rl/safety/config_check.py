"""Safe-state geometry validation (review B1 a/b, B5). Lives in ``safety`` so that ``config`` stays a leaf
package (everything imports config, config imports nothing above it).

``validate_geometry(cfg, calibration=None, chain=None)`` checks, with the real forward kinematics:

* (a) FK(home_ticks) lies ``home_box_margin_m`` inside the box; FK(rest_ticks) is above the table floor and
  inside the box footprint;
* (b) the exact runtime :class:`~so101_rl.safety.transit.TransitPlanner` replayed tick by tick, rest -> home and
  home -> rest, neither stalls ("transit stalls") nor dips below the floor ("transit dips");
* B5: ``bound_s`` covers the worst joint limit -> home plus the exact home -> rest transit, plus a margin.

``load_config`` no longer runs it. Call it wherever the safety stack is built (``scripts``, the servo process
entry point, the tests that ship a config).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from so101_rl.config import (
    JOINT_NAMES,
    TICKS_PER_REV,
    MotorCalibration,
    SafeStateConfig,
    So101Config,
    WorkspaceBox,
    load_calibration,
)

from .fk import KinematicChain, So101FK
from .transit import TransitPlanner, simulate_transit

# safe_state.bound_s must cover (worst joint limit -> home) + (exact home -> rest transit) + this margin (B5).
SAFE_STATE_BOUND_MARGIN_S = 1.0


def validate_geometry(
    cfg: So101Config,
    calibration: dict[str, MotorCalibration] | None = None,
    chain: KinematicChain | None = None,
) -> None:
    """Raise ``ValueError("safe_state: ...")`` listing every geometry problem of ``cfg.safe_state``.

    ``calibration`` defaults to the JSON at ``cfg.calibration_path`` (only ``drive_mode`` is used); ``chain``
    defaults to the yaml at ``cfg.kinematics_path`` (results cached per hashable input when it is defaulted).
    """
    problems = geometry_problems(cfg, calibration, chain)
    if problems:
        raise ValueError("safe_state: " + "; ".join(problems))


def geometry_problems(
    cfg: So101Config,
    calibration: dict[str, MotorCalibration] | None = None,
    chain: KinematicChain | None = None,
) -> tuple[str, ...]:
    """The list of problems ``validate_geometry`` raises on (empty when the configuration is sound)."""
    if calibration is None:
        calibration = load_calibration(cfg.calibration_path)
    safe, box, limits = cfg.safe_state, cfg.box, cfg.joints
    args = (
        tuple(int(v) for v in safe.home_ticks),
        tuple(int(v) for v in safe.rest_ticks),
        int(safe.ramp_ticks_per_tick),
        float(safe.bound_s),
        float(safe.table_z_m),
        float(safe.table_clearance_m),
        float(safe.transit_zone_m),
        float(safe.home_box_margin_m),
        (box.x, box.y, box.z),
        tuple(int(v) for v in limits.min_ticks),
        tuple(int(v) for v in limits.max_ticks),
        float(cfg.rates.servo_hz),
        tuple(int(calibration[n].drive_mode) for n in JOINT_NAMES),
    )
    if chain is not None:
        return _compute_problems(*args, chain)
    return _cached_problems(*args, str(cfg.kinematics_path))


@functools.lru_cache(maxsize=64)
def _cached_problems(*args: object) -> tuple[str, ...]:
    *rest, path = args
    return _compute_problems(*rest, KinematicChain.from_yaml(Path(str(path))))  # type: ignore[arg-type]


def _compute_problems(
    home: tuple[int, ...],
    rest: tuple[int, ...],
    step: int,
    bound_s: float,
    table_z: float,
    clearance: float,
    zone: float,
    home_margin: float,
    box_xyz: tuple[tuple[float, float], ...],
    lim_min: tuple[int, ...],
    lim_max: tuple[int, ...],
    servo_hz: float,
    drive_modes: tuple[int, ...],
    chain: KinematicChain,
) -> tuple[str, ...]:
    calib = {
        n: MotorCalibration(id=i + 1, drive_mode=d, homing_offset=0, range_min=0, range_max=TICKS_PER_REV - 1)
        for i, (n, d) in enumerate(zip(JOINT_NAMES, drive_modes, strict=True))
    }
    fk = So101FK(chain, calib)
    box = WorkspaceBox(x=box_xyz[0], y=box_xyz[1], z=box_xyz[2])
    planner = TransitPlanner(
        fk, SafeStateConfig(ramp_ticks_per_tick=step, table_z_m=table_z, table_clearance_m=clearance, transit_zone_m=zone)
    )
    home_a = np.array(home, dtype=np.int32)
    rest_a = np.array(rest, dtype=np.int32)
    floor = planner.floor_z
    out: list[str] = []
    xyz = np.zeros(3)

    # (a) home well inside the box; rest above the table and inside the box footprint (away from the D435 mount)
    fk.gripper_xyz_from_ticks(home_a, xyz)
    if not fk.in_box(xyz, box, margin=-home_margin):
        out.append(f"FK(home_ticks) = {np.round(xyz, 4).tolist()} m is not {home_margin} m inside the box")
    fk.gripper_xyz_from_ticks(rest_a, xyz)
    if xyz[2] < floor:
        out.append(f"FK(rest_ticks) z = {xyz[2]:.4f} m is below the table floor {floor:.4f} m (table + clearance)")
    if not (box.x[0] <= xyz[0] <= box.x[1] and box.y[0] <= xyz[1] <= box.y[1]):
        out.append(f"FK(rest_ticks) xy = {np.round(xyz[:2], 4).tolist()} m is outside the box footprint")

    # (b) the exact ramp (the runtime TransitPlanner), tick by tick, in both directions
    max_ticks = int(4 * TICKS_PER_REV / max(step, 1)) + 1
    legs: dict[str, int] = {}
    for name, a, b in (("rest -> home", rest_a, home_a), ("home -> rest", home_a, rest_a)):
        n, z_min, reached = simulate_transit(planner, a, b, max_ticks)
        legs[name] = n
        if not reached:
            out.append(f"{name} transit stalls after {n} ticks (the table floor holds every joint)")
        if z_min < floor - 1e-9:
            out.append(f"{name} transit dips to z = {z_min:.4f} m, below the table floor {floor:.4f} m")

    # B5: bound_s covers the worst joint limit -> home plus the exact home -> rest transit, plus a margin
    lo, hi = np.array(lim_min), np.array(lim_max)
    worst = int(max(np.abs(lo - home_a).max(), np.abs(hi - home_a).max()))
    need_s = worst / (step * servo_hz) + legs["home -> rest"] / servo_hz + SAFE_STATE_BOUND_MARGIN_S
    if bound_s < need_s:
        out.append(
            f"bound_s = {bound_s} s is below the {need_s:.2f} s the ramp needs (worst limit -> home {worst} ticks, "
            f"home -> rest {legs['home -> rest']} ticks at {step} ticks/tick and {servo_hz} Hz, "
            f"+{SAFE_STATE_BOUND_MARGIN_S} s)"
        )
    return tuple(out)


