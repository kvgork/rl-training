"""Load and validate ``so101.yaml`` + the LeRobot calibration JSON."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .schema import (
    JOINT_NAMES,
    N_JOINTS,
    REWARD_TERM_PARAMS,
    TASK_NAMES,
    TICKS_PER_REV,
    ActionConfig,
    JointLimits,
    MotorCalibration,
    RateConfig,
    RegisterConfig,
    RewardSpec,
    RewardTermSpec,
    SafeStateConfig,
    SimWarpConfig,
    So101Config,
    TaskConfig,
    TasksConfig,
    ThermalConfig,
    WorkspaceBox,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("so101.yaml")
ENV_PREFIXES = ("SO101_",)
ENV_NAMES = ("REQUIRE_RT",)


# Registers written by configure(). Bounds are the byte-width / documented ranges of the STS3215 control table
# as encoded in lerobot (lerobot/motors/feetech/tables.py, STS_SMS_SERIES_CONTROL_TABLE: 1-byte registers are
# 0..254 here because 255 is the broadcast/reserved value; Overload_Torque is a percentage; Protection_Current is
# a 2-byte register whose usable range is 0..511 in the Feetech memory table). The current and time bounds are
# deliberately conservative and are confirmed on the real arm in H6.
REGISTER_RANGES: dict[str, tuple[int, int]] = {
    "return_delay_time": (0, 254),
    "operating_mode": (0, 0),  # 0 = POSITION; the servo core only supports position mode
    "p_coefficient": (0, 254),
    "i_coefficient": (0, 254),
    "d_coefficient": (0, 254),
    "maximum_acceleration": (0, 254),
    "acceleration": (0, 254),
    "protection_current": (1, 511),
    "overload_torque": (1, 100),
    "over_current_protection_time": (0, 254),
}
# STS3215 homing offset is a 12-bit sign-magnitude register (bit 11 = sign): +-2047.
HOMING_OFFSET_RANGE = (-2047, 2047)
# Automatic thermal resumes per hour. The plan's value is 3; the state machine keeps max + 1 trip times.
MAX_TRIPS_PER_HOUR_RANGE = (1, 10)
# Plausible table height below the base frame origin (m); the box floor is the upper bound.
TABLE_Z_RANGE_M = (-0.2, 0.0)
# Gripper floor (plan §10 2026-10-07): clearance of the lowest gripper point above the table, and how far its present
# tolerance may let it press into the table.
GRIPPER_CLEARANCE_MAX_M = 0.02
GRIPPER_MAX_PENETRATION_M = 0.001
# Lowest allowed safe_state.start_min_tip_z_m above the table (plan §10 2026-10-06: "tip >= 1 cm above the table").
START_MIN_ABOVE_TABLE_M = 0.010
SAFE_STATE_KEYS = frozenset(
    {
        "bound_s",
        "ramp_ticks_per_tick",
        "home_ticks",
        "rest_ticks",
        "torque_off_order",
        "table_z_m",
        "table_clearance_m",
        "transit_zone_m",
        "home_box_margin_m",
        "rest_tolerance_ticks",
        "start_min_tip_z_m",
    }
)

TOP_KEYS = frozenset(
    {
        "verified_on_hardware",
        "port",
        "heartbeat_port",
        "calibration_path",
        "shm_name",
        "kinematics_path",
        "firmware_max_temp_c",
        "joints",
        "action",
        "thermal",
        "rates",
        "safe_state",
        "box",
        "registers",
        "sim_warp",
        "tasks",
    }
)
# FS1: MuJoCo Warp sizes. Lower bounds from the S0 spike (njmax 64 is MJWarp's own default and already overflows on
# this scene, so nothing smaller is accepted); upper bounds only reject absurd values (VRAM).
SIM_WARP_RANGES: dict[str, tuple[int, int]] = {
    "njmax": (64, 4096),
    "nconmax_per_world": (1, 1024),
    "nworld_default": (1, 65536),
    "overflow_check_every_steps": (1, 100000),
}
SIM_WARP_POLICIES = ("fail",)
CALIB_KEYS = frozenset({"id", "drive_mode", "homing_offset", "range_min", "range_max"})
# FS4 tasks (plan §10 2026-10-06). Keys per task; ranges reject typos and absurd values, the sign rules of the
# reward are proven by env.reward.validate_reward_spec (run whenever a TaskSpec is built).
TASK_KEYS: dict[str, frozenset[str]] = {
    "reach": frozenset({"reward", "trip_penalty"}),
    "die_to_zone": frozenset({"reward", "trip_penalty", "max_height_m", "settle_mm"}),
}
# < 0 (rewards are non-positive); the bounds only reject absurd values. "Worse than any trip-free episode" needs the
# task's horizon and workspace box: env.reward.validate_trip_penalty proves it whenever a TaskSpec is built.
TRIP_PENALTY_RANGE = (-1.0e4, -1.0)
REWARD_WEIGHT_RANGE = (0.0, 100.0)  # >= 0: every term is <= 0, a negative weight would pay a positive reward
REWARD_SCALE_RANGE_M = (1e-3, 10.0)
# Die centroid above the goal: > 0 and below ~2 cm, where a die "held clearly in the air" must still fail.
DIE_MAX_HEIGHT_RANGE_M = (1e-3, 0.020)
DIE_SETTLE_RANGE_MM = (0.1, 10.0)


def _keys(node: Any, expected: frozenset[str] | set[str], where: str) -> dict[str, Any]:
    """Mapping with EXACTLY the expected keys: typos, removed keys and missing keys are all errors."""
    if not isinstance(node, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(node).__name__}")
    unknown = sorted(str(k) for k in node if k not in expected)
    missing = sorted(k for k in expected if k not in node)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    return node


def _int(value: Any, key: str) -> int:
    """Integers only: bool, str and non-integral floats are rejected (integral floats such as 45.0 are fine)."""
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ValueError(f"{key}: expected an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{key}: expected an integer, got {value!r}")


def _num(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{key}: expected a number, got {value!r}")  # noqa: TRY004 - value errors by contract
    if not math.isfinite(value):
        raise ValueError(f"{key}: non-finite value {value!r}")
    return float(value)


def _positive(value: float, key: str) -> float:
    if not value > 0:
        raise ValueError(f"{key}: must be > 0, got {value}")
    return value


def _in_range(value: int, lo: int, hi: int, key: str) -> int:
    if not lo <= value <= hi:
        raise ValueError(f"{key}: {value} outside the allowed range {lo}..{hi}")
    return value


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    """Parse the LeRobot calibration JSON. Refuses ids other than 1-6, missing/extra joints and bad fields."""
    raw = json.loads(Path(path).expanduser().read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"calibration {path}: top level must be a mapping")
    missing = [n for n in JOINT_NAMES if n not in raw]
    if missing:
        raise ValueError(f"calibration {path} is missing joints: {missing}")
    extra = [n for n in raw if n not in JOINT_NAMES]
    if extra:
        raise ValueError(f"calibration {path} has unexpected motors: {extra}")
    calib = {}
    for n in JOINT_NAMES:
        entry = _keys(raw[n], CALIB_KEYS, f"calibration {path}: {n}")
        calib[n] = MotorCalibration(**{k: _int(entry[k], f"calibration {n}.{k}") for k in sorted(CALIB_KEYS)})
    _validate_calibration(calib, path)
    return calib


def _validate_calibration(calib: dict[str, MotorCalibration], where: object) -> None:
    ids = sorted(c.id for c in calib.values())
    if ids != list(range(1, N_JOINTS + 1)):
        raise ValueError(f"calibration {where}: motor ids must be exactly 1..6, got {ids}")
    for name, c in calib.items():
        if c.drive_mode not in (0, 1):
            raise ValueError(f"calibration {where}: {name}.drive_mode must be 0 or 1, got {c.drive_mode}")
        lo, hi = HOMING_OFFSET_RANGE
        if not lo <= c.homing_offset <= hi:
            raise ValueError(f"calibration {where}: {name}.homing_offset {c.homing_offset} outside {lo}..{hi}")
        if not (0 <= c.range_min < c.range_max <= TICKS_PER_REV - 1):
            raise ValueError(f"calibration {where}: bad range for {name}: {c.range_min}..{c.range_max}")


def _finite(value: Any, key: str) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key}: non-finite value {value!r}")


def _walk_finite(node: Any, prefix: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_finite(v, f"{prefix}{k}.")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk_finite(v, f"{prefix}{i}")
    elif isinstance(node, float):
        _finite(node, prefix.rstrip("."))


def _vec(value: Any, key: str) -> np.ndarray:
    """Exactly six integral values -> read-only int32[6]. Scalars are rejected (no silent broadcast)."""
    if not isinstance(value, (list, tuple)) or len(value) != N_JOINTS:
        raise ValueError(f"{key}: expected a list of {N_JOINTS} integers, got {value!r}")
    out = np.array([_int(v, f"{key}[{i}]") for i, v in enumerate(value)], dtype=np.int32)
    out.setflags(write=False)
    return out


def _pair(value: Any, key: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key}: expected [low, high], got {value!r}")
    return (_num(value[0], f"{key}[0]"), _num(value[1], f"{key}[1]"))


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    _walk_finite(raw)
    return raw


def _num_in(value: Any, lo: float, hi: float, key: str) -> float:
    v = _num(value, key)
    if not lo <= v <= hi:
        raise ValueError(f"{key}: {v} outside the allowed range {lo}..{hi}")
    return v


def _parse_reward(node: Any, where: str) -> RewardSpec:
    """``tasks.<name>.reward``: a mapping term name -> {weight, <term params>}. Strict: unknown term names, unknown
    or missing parameter keys and out-of-range values are errors; at least one term."""
    if not isinstance(node, dict) or not node:
        raise ValueError(f"{where}: expected a non-empty mapping of reward terms, got {node!r}")
    terms = []
    for name, params in node.items():
        if name not in REWARD_TERM_PARAMS:
            raise ValueError(f"{where}: unknown reward term {name!r} (known: {sorted(REWARD_TERM_PARAMS)})")
        p = _keys(params, {"weight", *REWARD_TERM_PARAMS[name]}, f"{where}.{name}")
        kw: dict[str, Any] = {"weight": _num_in(p["weight"], *REWARD_WEIGHT_RANGE, f"{where}.{name}.weight")}
        if "scale_m" in p:
            kw["scale_m"] = _num_in(p["scale_m"], *REWARD_SCALE_RANGE_M, f"{where}.{name}.scale_m")
        if "zero_on_success" in p:
            if not isinstance(p["zero_on_success"], bool):
                raise ValueError(f"{where}.{name}.zero_on_success: expected true or false")
            kw["zero_on_success"] = p["zero_on_success"]
        terms.append(RewardTermSpec(name=name, **kw))
    return RewardSpec(terms=tuple(terms))


def parse_tasks(node: Any, where: str = "tasks") -> TasksConfig:
    """Strictly parse the ``tasks:`` section (FS4): ``reach`` and ``die_to_zone``, each with exactly its keys."""
    t = _keys(node, set(TASK_NAMES), where)
    out = {}
    for name in TASK_NAMES:
        tw = f"{where}.{name}"
        e = _keys(t[name], TASK_KEYS[name], tw)
        kw: dict[str, Any] = {
            "reward": _parse_reward(e["reward"], f"{tw}.reward"),
            "trip_penalty": _num_in(e["trip_penalty"], *TRIP_PENALTY_RANGE, f"{tw}.trip_penalty"),
        }
        if "max_height_m" in e:
            kw["max_height_m"] = _num_in(e["max_height_m"], *DIE_MAX_HEIGHT_RANGE_M, f"{tw}.max_height_m")
        if "settle_mm" in e:
            kw["settle_mm"] = _num_in(e["settle_mm"], *DIE_SETTLE_RANGE_MM, f"{tw}.settle_mm")
        out[name] = TaskConfig(**kw)
    return TasksConfig(**out)


def load_tasks_config(path: Path | None = None) -> TasksConfig:
    """Only the ``tasks:`` section (no calibration needed): ``env/tasks.py`` builds REACH / DIE_TO_ZONE from it."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)
    if "tasks" not in raw:
        raise ValueError(f"{path}: missing keys ['tasks']")
    return parse_tasks(raw["tasks"])


def load_config(
    path: Path | None = None, *, calibration: dict[str, MotorCalibration] | None = None
) -> So101Config:
    """Build a validated :class:`So101Config`.

    Strict: every section must contain exactly the expected keys (no unknown, no missing, no defaults).
    Joint limits are the intersection of the yaml limits and the calibration ``range_min/max``
    (narrowest wins). ``calibration`` overrides reading ``calibration_path`` (tests, sim).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _keys(_read_yaml(path), TOP_KEYS, str(path))

    if not isinstance(raw["verified_on_hardware"], bool):
        raise ValueError("verified_on_hardware: expected true or false")  # noqa: TRY004 - value errors by contract
    for k in ("port", "shm_name", "calibration_path", "kinematics_path"):
        if not isinstance(raw[k], str) or not raw[k]:
            raise ValueError(f"{k}: expected a non-empty string, got {raw[k]!r}")
    hb = raw["heartbeat_port"]
    if hb is not None and (not isinstance(hb, str) or not hb):
        raise ValueError(f"heartbeat_port: expected a string or null, got {hb!r}")

    calib_path = Path(raw["calibration_path"]).expanduser()
    if calibration is None:
        calibration = load_calibration(calib_path)
    else:
        _validate_calibration(calibration, "<argument>")

    j = _keys(raw["joints"], {"min_ticks", "max_ticks"}, "joints")
    cal_min = np.array([calibration[n].range_min for n in JOINT_NAMES], dtype=np.int32)
    cal_max = np.array([calibration[n].range_max for n in JOINT_NAMES], dtype=np.int32)
    cfg_min = _vec(j["min_ticks"], "joints.min_ticks")
    cfg_max = _vec(j["max_ticks"], "joints.max_ticks")
    lo = np.maximum(cfg_min, cal_min).astype(np.int32)
    hi = np.minimum(cfg_max, cal_max).astype(np.int32)
    if np.any(lo >= hi):
        raise ValueError(f"empty joint limit intersection: min={lo.tolist()} max={hi.tolist()}")
    lo.setflags(write=False)
    hi.setflags(write=False)
    limits = JointLimits(min_ticks=lo, max_ticks=hi)

    r = _keys(
        raw["rates"],
        {"servo_hz", "policy_hz", "stale_cmd_ms", "max_consecutive_misses", "heartbeat_timeout_ms", "max_bus_failures"},
        "rates",
    )
    rates = RateConfig(
        servo_hz=_positive(_num(r["servo_hz"], "rates.servo_hz"), "rates.servo_hz"),
        policy_hz=_positive(_num(r["policy_hz"], "rates.policy_hz"), "rates.policy_hz"),
        stale_cmd_ms=_positive(_num(r["stale_cmd_ms"], "rates.stale_cmd_ms"), "rates.stale_cmd_ms"),
        max_consecutive_misses=_in_range(
            _int(r["max_consecutive_misses"], "rates.max_consecutive_misses"), 1, 10**6, "rates.max_consecutive_misses"
        ),
        heartbeat_timeout_ms=_in_range(
            _int(r["heartbeat_timeout_ms"], "rates.heartbeat_timeout_ms"), 1, 10**7, "rates.heartbeat_timeout_ms"
        ),
        max_bus_failures=_in_range(
            _int(r["max_bus_failures"], "rates.max_bus_failures"), 1, 10**6, "rates.max_bus_failures"
        ),
    )
    ratio = rates.servo_hz / rates.policy_hz
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
        raise ValueError("servo_hz must be a positive integer multiple of policy_hz")
    ticks_per_command = round(ratio)

    a = _keys(raw["action"], {"max_delta_ticks", "tracking_error_ticks", "max_ticks_per_tick", "interp_ticks"}, "action")
    action = ActionConfig(
        max_delta_ticks=_vec(a["max_delta_ticks"], "action.max_delta_ticks"),
        tracking_error_ticks=_vec(a["tracking_error_ticks"], "action.tracking_error_ticks"),
        max_ticks_per_tick=_vec(a["max_ticks_per_tick"], "action.max_ticks_per_tick"),
        interp_ticks=_int(a["interp_ticks"], "action.interp_ticks"),
    )
    for name in ("max_delta_ticks", "tracking_error_ticks", "max_ticks_per_tick"):
        if np.any(getattr(action, name) <= 0):
            raise ValueError(f"action.{name}: every joint must be > 0, got {getattr(action, name).tolist()}")
    if action.interp_ticks < 1:
        raise ValueError("action.interp_ticks must be >= 1")
    # The velocity cap must be able to deliver one full command within one policy period, or the goal is
    # never reached and the tracking-error clamp fights the velocity cap.
    reachable = action.max_ticks_per_tick.astype(np.int64) * ticks_per_command
    if np.any(reachable < action.max_delta_ticks):
        raise ValueError(
            f"action.max_ticks_per_tick * (servo_hz/policy_hz) = {reachable.tolist()} is below "
            f"max_delta_ticks {action.max_delta_ticks.tolist()}"
        )

    t = _keys(
        raw["thermal"],
        {"trip_c", "resume_c", "resume_dwell_s", "max_trips_per_hour", "duty_on_s", "duty_off_s", "stale_temp_s"},
        "thermal",
    )
    thermal = ThermalConfig(
        trip_c=_num(t["trip_c"], "thermal.trip_c"),
        resume_c=_num(t["resume_c"], "thermal.resume_c"),
        resume_dwell_s=_positive(_num(t["resume_dwell_s"], "thermal.resume_dwell_s"), "thermal.resume_dwell_s"),
        max_trips_per_hour=_in_range(
            _int(t["max_trips_per_hour"], "thermal.max_trips_per_hour"),
            *MAX_TRIPS_PER_HOUR_RANGE,
            "thermal.max_trips_per_hour",
        ),
        duty_on_s=_positive(_num(t["duty_on_s"], "thermal.duty_on_s"), "thermal.duty_on_s"),
        duty_off_s=_positive(_num(t["duty_off_s"], "thermal.duty_off_s"), "thermal.duty_off_s"),
        stale_temp_s=_positive(_num(t["stale_temp_s"], "thermal.stale_temp_s"), "thermal.stale_temp_s"),
    )
    firmware_max = _num(raw["firmware_max_temp_c"], "firmware_max_temp_c")
    if not thermal.resume_c < thermal.trip_c < firmware_max:
        raise ValueError(
            f"need resume_c < trip_c < firmware_max_temp_c, got {thermal.resume_c} / {thermal.trip_c} / "
            f"{firmware_max}"
        )

    b = _keys(
        raw["box"], {"x", "y", "z", "present_tolerance_m", "gripper_floor_clearance_m", "gripper_present_tolerance_m"}, "box"
    )
    box = WorkspaceBox(
        x=_pair(b["x"], "box.x"),
        y=_pair(b["y"], "box.y"),
        z=_pair(b["z"], "box.z"),
        present_tolerance_m=_num(b["present_tolerance_m"], "box.present_tolerance_m"),
        gripper_present_tolerance_m=_num(b["gripper_present_tolerance_m"], "box.gripper_present_tolerance_m"),
    )
    gripper_clearance = _num(b["gripper_floor_clearance_m"], "box.gripper_floor_clearance_m")
    if not 0.0 <= gripper_clearance <= GRIPPER_CLEARANCE_MAX_M:
        raise ValueError(
            f"box.gripper_floor_clearance_m: must be within 0..{GRIPPER_CLEARANCE_MAX_M} m, got {gripper_clearance}"
        )
    for axis, (lo_m, hi_m) in (("x", box.x), ("y", box.y), ("z", box.z)):
        if not lo_m < hi_m:
            raise ValueError(f"box.{axis}: lower bound must be below upper bound")
    if not 0.0 <= box.present_tolerance_m <= 0.02:
        raise ValueError(f"box.present_tolerance_m: must be within 0..0.02 m, got {box.present_tolerance_m}")

    # The gripper-point present tolerance may let a point press at most GRIPPER_MAX_PENETRATION_M into the table
    # (plan §10 2026-10-07: 1 mm clearance, 2 mm tolerance -> ~1 mm); checked below once table_z_m is known.
    if not 0.0 <= box.gripper_present_tolerance_m <= gripper_clearance + GRIPPER_MAX_PENETRATION_M:
        raise ValueError(
            f"box.gripper_present_tolerance_m: must be within 0..gripper_floor_clearance_m + "
            f"{GRIPPER_MAX_PENETRATION_M} = {gripper_clearance + GRIPPER_MAX_PENETRATION_M:.4f} m, "
            f"got {box.gripper_present_tolerance_m}"
        )

    s = _keys(raw["safe_state"], SAFE_STATE_KEYS, "safe_state")
    order = s["torque_off_order"]
    if not isinstance(order, (list, tuple)) or not all(isinstance(n, str) for n in order):
        raise ValueError("safe_state.torque_off_order: expected a list of joint names")
    safe = SafeStateConfig(
        bound_s=_positive(_num(s["bound_s"], "safe_state.bound_s"), "safe_state.bound_s"),
        ramp_ticks_per_tick=_int(s["ramp_ticks_per_tick"], "safe_state.ramp_ticks_per_tick"),
        home_ticks=_vec(s["home_ticks"], "safe_state.home_ticks"),
        rest_ticks=_vec(s["rest_ticks"], "safe_state.rest_ticks"),
        torque_off_order=tuple(order),
        table_z_m=_num(s["table_z_m"], "safe_state.table_z_m"),
        table_clearance_m=_num(s["table_clearance_m"], "safe_state.table_clearance_m"),
        transit_zone_m=_num(s["transit_zone_m"], "safe_state.transit_zone_m"),
        home_box_margin_m=_num(s["home_box_margin_m"], "safe_state.home_box_margin_m"),
        rest_tolerance_ticks=_in_range(
            _int(s["rest_tolerance_ticks"], "safe_state.rest_tolerance_ticks"), 1, 400, "safe_state.rest_tolerance_ticks"
        ),
        start_min_tip_z_m=_num(s["start_min_tip_z_m"], "safe_state.start_min_tip_z_m"),
    )
    _in_range(safe.ramp_ticks_per_tick, 1, 64, "safe_state.ramp_ticks_per_tick")
    if not TABLE_Z_RANGE_M[0] <= safe.table_z_m < box.z[0]:
        raise ValueError(
            f"safe_state.table_z_m: must be within {TABLE_Z_RANGE_M[0]} m and below box.z[0] = {box.z[0]} m, "
            f"got {safe.table_z_m}"
        )
    # The start floor is at least 1 cm above the table (plan §10 2026-10-06 "Re-energise start rule"; shipped: 0.010 m,
    # the old box floor) and below the box top. It is independent of box.z[0]: lowering the running floor must never
    # widen the start rule again.
    start_lo = safe.table_z_m + START_MIN_ABOVE_TABLE_M
    if not start_lo <= safe.start_min_tip_z_m < box.z[1]:
        raise ValueError(
            f"safe_state.start_min_tip_z_m: must be within [table_z_m + {START_MIN_ABOVE_TABLE_M} = {start_lo:.4f} m, "
            f"box.z[1] = {box.z[1]} m), got {safe.start_min_tip_z_m}"
        )
    for key in ("table_clearance_m", "transit_zone_m", "home_box_margin_m"):
        if not getattr(safe, key) >= 0.0:
            raise ValueError(f"safe_state.{key}: must be >= 0, got {getattr(safe, key)}")
    # rounded to 1e-9 m so the shipped value equals the schema default exactly (-0.02 + 0.001 is not -0.019 in binary)
    box = dataclasses.replace(box, gripper_floor_z_m=round(safe.table_z_m + gripper_clearance, 9))
    if sorted(safe.torque_off_order) != sorted(JOINT_NAMES):
        raise ValueError("safe_state.torque_off_order must list every joint exactly once")
    for key, ticks in (("home_ticks", safe.home_ticks), ("rest_ticks", safe.rest_ticks)):
        if np.any(ticks < limits.min_ticks) or np.any(ticks > limits.max_ticks):
            raise ValueError(f"safe_state.{key} outside the joint limits: {ticks.tolist()}")
    kinematics_path = (path.parent / raw["kinematics_path"]).resolve()

    g = _keys(raw["registers"], set(REGISTER_RANGES), "registers")
    registers = RegisterConfig(
        **{k: _in_range(_int(g[k], f"registers.{k}"), *REGISTER_RANGES[k], f"registers.{k}") for k in REGISTER_RANGES}
    )

    w = _keys(raw["sim_warp"], {*SIM_WARP_RANGES, "overflow_policy"}, "sim_warp")
    if w["overflow_policy"] not in SIM_WARP_POLICIES:
        raise ValueError(f"sim_warp.overflow_policy: must be one of {SIM_WARP_POLICIES}, got {w['overflow_policy']!r}")
    sim_warp = SimWarpConfig(
        **{k: _in_range(_int(w[k], f"sim_warp.{k}"), *SIM_WARP_RANGES[k], f"sim_warp.{k}") for k in SIM_WARP_RANGES},
        overflow_policy=w["overflow_policy"],
    )

    return So101Config(
        port=raw["port"],
        heartbeat_port=hb,
        calibration_path=calib_path,
        shm_name=raw["shm_name"],
        joints=limits,
        action=action,
        thermal=thermal,
        rates=rates,
        safe_state=safe,
        box=box,
        registers=registers,
        kinematics_path=kinematics_path,
        sim_warp=sim_warp,
        tasks=parse_tasks(raw["tasks"]),
    )


def is_verified_on_hardware(path: Path | None = None) -> bool:
    """``verified_on_hardware`` flag from the yaml (read by check_milestone)."""
    flag = _read_yaml(Path(path) if path is not None else DEFAULT_CONFIG_PATH).get("verified_on_hardware")
    if not isinstance(flag, bool):
        raise ValueError("verified_on_hardware: expected true or false")  # noqa: TRY004 - value errors by contract
    return flag


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _jsonable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump_env(cfg: So101Config | None = None) -> dict[str, Any]:
    """Config plus the ``SO101_*`` / ``REQUIRE_RT`` environment, for diffing launch environments (L4)."""
    cfg = cfg if cfg is not None else load_config()
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES) or k in ENV_NAMES}
    return {"config": _jsonable(cfg), "env": env}
