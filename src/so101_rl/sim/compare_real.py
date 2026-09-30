"""Sysid *check* (no fitting): replay a logged real episode through the sim and report the per-joint error.

Reads the F8 episode layout (``target_ticks`` / ``present_ticks``, int32 ``[N, 6]``, one row per policy step).
Convention: ``present_ticks[i]`` is the reading after ``target_ticks[i]`` was applied for one policy step
(``servo_hz / policy_hz`` servo ticks). The sim starts from ``present_ticks[0]`` (the pose the arm was at when the
log began, so record 0 is applied but not scored) and records ``1..N-1`` are compared.

Usage: ``python -m so101_rl.sim.compare_real EPISODE.h5 [--calibration CALIB.json]``; exits 1 if any joint's RMS
error exceeds ``MAX_RMS_PCT`` of its calibrated range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

from so101_rl.config import JOINT_NAMES, MotorCalibration, So101Config, load_calibration, load_config
from so101_rl.servo.bus import BlockRead
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

MAX_RMS_PCT = 5.0  # sysid milestone gate: per-joint RMS error as % of the joint's calibrated range


def rms_joint_error_pct(
    real_h5: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    calibration: dict[str, MotorCalibration],
    *,
    cfg: So101Config | None = None,
) -> dict[str, float]:
    """Replay the logged ``target_ticks`` and return ``{joint: RMS(present_sim - present_real) / range * 100}``."""
    with h5py.File(real_h5, "r") as f:
        targets = np.asarray(f["target_ticks"][:], dtype=np.int32)
        real = np.asarray(f["present_ticks"][:], dtype=np.int32)
    if targets.shape != real.shape or targets.ndim != 2 or targets.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"{real_h5}: target_ticks {targets.shape} / present_ticks {real.shape} are not [N, 6]")
    if len(targets) < 2:
        raise ValueError(f"{real_h5}: need at least 2 steps to compare, got {len(targets)}")

    cfg = cfg if cfg is not None else load_config(calibration=calibration)
    bus = MujocoBus(model, data, calibration, cfg)
    bus.connect()

    # Start from the logged pose, at rest.
    data.qvel[:] = 0.0
    data.qpos[bus.qpos_adr] = bus.ticks_to_q(real[0])
    data.ctrl[bus.act_ids] = data.qpos[bus.qpos_adr]
    mujoco.mj_forward(model, data)

    ticks_per_step = max(1, round(cfg.rates.servo_hz / cfg.rates.policy_hz))
    out = BlockRead()
    sim = np.empty_like(real)
    for i, goal in enumerate(targets):
        if not bus.write_goal(goal):
            raise ValueError(f"{real_h5}: step {i}: target_ticks {goal.tolist()} rejected (must be int 0..4095)")
        bus.step_ticks(ticks_per_step)
        bus.read_block(out)
        sim[i] = out.position

    err = (sim[1:] - real[1:]).astype(np.float64)
    rms = np.sqrt(np.mean(err**2, axis=0))
    span = np.array([calibration[n].range_max - calibration[n].range_min for n in JOINT_NAMES], dtype=np.float64)
    return {n: float(rms[i] / span[i] * 100.0) for i, n in enumerate(JOINT_NAMES)}


def format_table(errors: dict[str, float], max_pct: float = MAX_RMS_PCT) -> str:
    rows = [f"{'joint':<15}{'rms % of range':>16}  verdict"]
    for n, e in errors.items():
        rows.append(f"{n:<15}{e:>16.3f}  {'ok' if e <= max_pct else 'FAIL'}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episode", type=Path, help="F8 episode .h5")
    ap.add_argument("--calibration", type=Path, default=None, help="LeRobot calibration JSON (default: from config)")
    ap.add_argument("--max-pct", type=float, default=MAX_RMS_PCT)
    args = ap.parse_args(argv)

    calib_path = args.calibration if args.calibration is not None else load_config().calibration_path
    calibration = load_calibration(calib_path)
    model, data = load_model()
    errors = rms_joint_error_pct(args.episode, model, data, calibration)
    print(format_table(errors, args.max_pct))
    bad = [n for n, e in errors.items() if e > args.max_pct]
    if bad:
        print(f"FAIL: {', '.join(bad)} exceed {args.max_pct:.1f} % of range", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
