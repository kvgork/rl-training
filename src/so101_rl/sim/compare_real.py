"""Sysid *check* (no fitting): replay a logged real run through the sim and report the per-joint error.

Two inputs:

* an F8 episode (``target_ticks`` / ``present_ticks``, int32 ``[N, 6]``, one row per policy step). Convention:
  ``present_ticks[i]`` is the reading after ``target_ticks[i]`` was applied for one policy step
  (``servo_hz / policy_hz`` servo ticks). The sim starts from ``present_ticks[0]`` (the pose the arm was at when the
  log began, so record 0 is applied but not scored) and records ``1..N-1`` are compared.
* ``--tick-log``: the servo's 30 Hz ``TICK_LOG_DTYPE`` log. Every run of consecutive RUN loop ticks is replayed open
  loop (target per tick -> present per tick, ``present[k + 1]`` is the reading after ``target[k]``; the procedure of
  ``sim.sysid.replay`` with one shooting window per run) and the error is pooled over all runs.

``--sim-params PATH`` (``calib/sim_params.yaml``, FS7) applies those parameters (physics, the command delay FIFO and
the dead-band) before the replay; without it the MJCF prior is used with no delay. compare-real is the check itself,
so an unverified file (the release-0 prior) is accepted here.

Usage: ``python -m so101_rl.sim.compare_real [EPISODE.h5 | --tick-log TICKS.bin] [--sim-params YAML]
[--calibration CALIB.json]``; exits 1 if any joint's RMS error exceeds ``MAX_RMS_PCT`` of its calibrated range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import mujoco
import numpy as np

from so101_rl.config import JOINT_NAMES, MotorCalibration, So101Config, load_calibration, load_config
from so101_rl.servo.bus import BlockRead
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

if TYPE_CHECKING:
    from so101_rl.sim.sysid.params import SimParams

MAX_RMS_PCT = 5.0  # sysid milestone gate: per-joint RMS error as % of the joint's calibrated range


def rms_joint_error_pct(
    real_h5: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    calibration: dict[str, MotorCalibration],
    *,
    cfg: So101Config | None = None,
    sim_params: SimParams | None = None,
) -> dict[str, float]:
    """Replay the logged ``target_ticks`` and return ``{joint: RMS(present_sim - present_real) / range * 100}``.

    ``sim_params`` (optional) is applied to ``model`` first (it stays applied), with its ``delay_ticks`` as the bus's
    command FIFO and ``deadband_ticks`` as target hysteresis on the replayed goals."""
    from so101_rl.sim.sysid.params import apply_deadband

    with h5py.File(real_h5, "r") as f:
        targets = np.asarray(f["target_ticks"][:], dtype=np.int32)
        real = np.asarray(f["present_ticks"][:], dtype=np.int32)
    if targets.shape != real.shape or targets.ndim != 2 or targets.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"{real_h5}: target_ticks {targets.shape} / present_ticks {real.shape} are not [N, 6]")
    if len(targets) < 2:
        raise ValueError(f"{real_h5}: need at least 2 steps to compare, got {len(targets)}")
    if targets.min() < 0 or targets.max() > 4095:
        bad = int(np.flatnonzero((targets < 0).any(axis=1) | (targets > 4095).any(axis=1))[0])
        raise ValueError(f"{real_h5}: step {bad}: target_ticks {targets[bad].tolist()} rejected (must be int 0..4095)")

    cfg = cfg if cfg is not None else load_config(calibration=calibration)
    delay = 0
    if sim_params is not None:
        sim_params.apply_cpu(model)  # before the bus: MujocoBus caches the gains it restores on torque-on
        delay = sim_params.delay_ticks
        targets = apply_deadband(targets, sim_params.deadband_ticks, initial=real[0]).astype(np.int32)
    bus = MujocoBus(model, data, calibration, cfg, action_delay_ticks=delay)
    bus.connect()
    bus.set_torque(True)  # replaying commanded goals: the servos are energised (the bus itself starts limp)

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


def rms_tick_log_error_pct(
    tick_log: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    calibration: dict[str, MotorCalibration],
    *,
    cfg: So101Config | None = None,
    sim_params: SimParams | None = None,
    min_ticks: int = 10,
) -> dict[str, float]:
    """Open-loop 30 Hz replay of every RUN run of a ``TICK_LOG_DTYPE`` log; per-joint RMS pooled over all runs."""
    from so101_rl.sim.sysid.dataset import tick_log_segments
    from so101_rl.sim.sysid.params import SimParams
    from so101_rl.sim.sysid.replay import TickMap, as_joint_dict, pooled_rms_pct, replay_segment_cpu

    cfg = cfg if cfg is not None else load_config(calibration=calibration)
    params = sim_params if sim_params is not None else SimParams.from_mjcf(model)
    segs = tick_log_segments(Path(tick_log), min_ticks=min_ticks)
    if not segs:
        raise ValueError(f"{tick_log}: no run of >= {min_ticks} consecutive RUN ticks to replay")
    params.apply_cpu(model)
    tmap = TickMap(calibration, model)
    sims = [
        replay_segment_cpu(s, model, data, calibration, cfg, params, shooting_window_ticks=len(s), tmap=tmap,
                           apply_params=False)
        for s in segs
    ]
    return as_joint_dict(pooled_rms_pct(sims, segs, calibration))


def format_table(errors: dict[str, float], max_pct: float = MAX_RMS_PCT) -> str:
    rows = [f"{'joint':<15}{'rms % of range':>16}  verdict"]
    for n, e in errors.items():
        rows.append(f"{n:<15}{e:>16.3f}  {'ok' if e <= max_pct else 'FAIL'}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episode", type=Path, nargs="?", default=None, help="F8 episode .h5")
    ap.add_argument("--tick-log", type=Path, default=None, help="30 Hz servo tick log (TICK_LOG_DTYPE) instead of an episode")
    ap.add_argument("--sim-params", type=Path, default=None, help="sim_params yaml (FS7) to apply before the replay")
    ap.add_argument("--calibration", type=Path, default=None, help="LeRobot calibration JSON (default: from config)")
    ap.add_argument("--max-pct", type=float, default=MAX_RMS_PCT)
    args = ap.parse_args(argv)
    if (args.episode is None) == (args.tick_log is None):
        ap.error("give exactly one of EPISODE or --tick-log")

    calib_path = args.calibration if args.calibration is not None else load_config().calibration_path
    calibration = load_calibration(Path(calib_path).expanduser())
    model, data = load_model()
    params = None
    if args.sim_params is not None:
        from so101_rl.sim.sysid.params import read_sim_params_yaml

        params, meta = read_sim_params_yaml(args.sim_params, allow_unverified=True)
        status = "verified" if not meta.verification_problems(MAX_RMS_PCT) else "UNVERIFIED"
        print(f"sim params: {args.sim_params} (release {meta.version}, {status}, delay {params.delay_ticks} ticks)")
    if args.tick_log is not None:
        errors = rms_tick_log_error_pct(args.tick_log, model, data, calibration, sim_params=params)
    else:
        errors = rms_joint_error_pct(args.episode, model, data, calibration, sim_params=params)
    print(format_table(errors, args.max_pct))
    bad = [n for n, e in errors.items() if not e <= args.max_pct]  # NaN fails too
    if bad:
        print(f"FAIL: {', '.join(bad)} exceed {args.max_pct:.1f} % of range", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
