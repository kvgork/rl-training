"""Load and validate ``so101.yaml`` + the LeRobot calibration JSON."""

from __future__ import annotations

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
    TICKS_PER_REV,
    ActionConfig,
    JointLimits,
    MotorCalibration,
    RateConfig,
    RegisterConfig,
    SafeStateConfig,
    So101Config,
    ThermalConfig,
    WorkspaceBox,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("so101.yaml")
ENV_PREFIXES = ("SO101_",)
ENV_NAMES = ("REQUIRE_RT",)


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    """Parse the LeRobot calibration JSON. Refuses ids other than 1-6 and missing joints."""
    raw = json.loads(Path(path).expanduser().read_text())
    missing = [n for n in JOINT_NAMES if n not in raw]
    if missing:
        raise ValueError(f"calibration {path} is missing joints: {missing}")
    extra = [n for n in raw if n not in JOINT_NAMES]
    if extra:
        raise ValueError(f"calibration {path} has unexpected motors: {extra}")
    calib = {
        n: MotorCalibration(
            id=int(raw[n]["id"]),
            drive_mode=int(raw[n]["drive_mode"]),
            homing_offset=int(raw[n]["homing_offset"]),
            range_min=int(raw[n]["range_min"]),
            range_max=int(raw[n]["range_max"]),
        )
        for n in JOINT_NAMES
    }
    _validate_calibration(calib, path)
    return calib


def _validate_calibration(calib: dict[str, MotorCalibration], where: object) -> None:
    ids = sorted(c.id for c in calib.values())
    if ids != list(range(1, N_JOINTS + 1)):
        raise ValueError(f"calibration {where}: motor ids must be exactly 1..6, got {ids}")
    for name, c in calib.items():
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


def _vec(value: Any, key: str, dtype: Any = np.int32) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(N_JOINTS, float(arr))
    if arr.shape != (N_JOINTS,):
        raise ValueError(f"{key}: expected {N_JOINTS} values, got shape {arr.shape}")
    _finite(arr, key)
    out = np.rint(arr).astype(dtype) if np.issubdtype(dtype, np.integer) else arr.astype(dtype)
    out.setflags(write=False)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    _walk_finite(raw)
    return raw


def load_config(
    path: Path | None = None, *, calibration: dict[str, MotorCalibration] | None = None
) -> So101Config:
    """Build a validated :class:`So101Config`.

    Joint limits are the intersection of the yaml limits and the calibration ``range_min/max``
    (narrowest wins). ``calibration`` overrides reading ``calibration_path`` (tests, sim).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)

    calib_path = Path(str(raw["calibration_path"])).expanduser()
    if calibration is None:
        calibration = load_calibration(calib_path)
    else:
        _validate_calibration(calibration, "<argument>")

    cal_min = np.array([calibration[n].range_min for n in JOINT_NAMES], dtype=np.int32)
    cal_max = np.array([calibration[n].range_max for n in JOINT_NAMES], dtype=np.int32)
    cfg_min = _vec(raw["joints"]["min_ticks"], "joints.min_ticks")
    cfg_max = _vec(raw["joints"]["max_ticks"], "joints.max_ticks")
    lo = np.maximum(cfg_min, cal_min).astype(np.int32)
    hi = np.minimum(cfg_max, cal_max).astype(np.int32)
    if np.any(lo >= hi):
        raise ValueError(f"empty joint limit intersection: min={lo.tolist()} max={hi.tolist()}")
    lo.setflags(write=False)
    hi.setflags(write=False)
    limits = JointLimits(min_ticks=lo, max_ticks=hi)

    a = raw["action"]
    action = ActionConfig(
        max_delta_ticks=_vec(a["max_delta_ticks"], "action.max_delta_ticks"),
        tracking_error_ticks=_vec(a["tracking_error_ticks"], "action.tracking_error_ticks"),
        max_ticks_per_tick=_vec(a["max_ticks_per_tick"], "action.max_ticks_per_tick"),
        interp_ticks=int(a.get("interp_ticks", 3)),
    )
    if np.any(action.max_delta_ticks <= 0) or np.any(action.tracking_error_ticks <= 0):
        raise ValueError("action limits must be positive")
    if action.interp_ticks < 1:
        raise ValueError("action.interp_ticks must be >= 1")

    thermal = ThermalConfig(**raw["thermal"])
    firmware_max = float(raw["firmware_max_temp_c"])
    if not thermal.resume_c < thermal.trip_c < firmware_max:
        raise ValueError(
            f"need resume_c < trip_c < firmware_max_temp_c, got {thermal.resume_c} / {thermal.trip_c} / "
            f"{firmware_max}"
        )

    rates = RateConfig(**raw["rates"])
    if not (rates.servo_hz > 0 and rates.policy_hz > 0 and math.isclose(rates.servo_hz % rates.policy_hz, 0)):
        raise ValueError("servo_hz must be a positive multiple of policy_hz")

    s = raw["safe_state"]
    safe = SafeStateConfig(
        bound_s=float(s["bound_s"]),
        ramp_ticks_per_tick=int(s["ramp_ticks_per_tick"]),
        home_ticks=_vec(s["home_ticks"], "safe_state.home_ticks"),
        rest_ticks=_vec(s["rest_ticks"], "safe_state.rest_ticks"),
        torque_off_order=tuple(s["torque_off_order"]),
    )
    if sorted(safe.torque_off_order) != sorted(JOINT_NAMES):
        raise ValueError("safe_state.torque_off_order must list every joint exactly once")
    for key, ticks in (("home_ticks", safe.home_ticks), ("rest_ticks", safe.rest_ticks)):
        if np.any(ticks < limits.min_ticks) or np.any(ticks > limits.max_ticks):
            raise ValueError(f"safe_state.{key} outside the joint limits: {ticks.tolist()}")

    b = raw["box"]
    box = WorkspaceBox(x=tuple(b["x"]), y=tuple(b["y"]), z=tuple(b["z"]))
    for axis, (lo_m, hi_m) in (("x", box.x), ("y", box.y), ("z", box.z)):
        if not lo_m < hi_m:
            raise ValueError(f"box.{axis}: lower bound must be below upper bound")

    hb = raw.get("heartbeat_port")
    return So101Config(
        port=str(raw["port"]),
        heartbeat_port=None if hb is None else str(hb),
        calibration_path=calib_path,
        shm_name=str(raw["shm_name"]),
        joints=limits,
        action=action,
        thermal=thermal,
        rates=rates,
        safe_state=safe,
        box=box,
        registers=RegisterConfig(**raw["registers"]),
        kinematics_path=(path.parent / str(raw["kinematics_path"])).resolve(),
    )


def is_verified_on_hardware(path: Path | None = None) -> bool:
    """``verified_on_hardware`` flag from the yaml (read by check_milestone)."""
    return bool(_read_yaml(Path(path) if path is not None else DEFAULT_CONFIG_PATH).get("verified_on_hardware"))


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
