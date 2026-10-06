"""Excitation runner (FS7, sim-first track S2.2; pixi task ``sysid-excite``).

``python -m so101_rl.sim.sysid.run_excitation --backend {sim,real,synthetic} --plan {default,short,fit40}
--out runs/sysid/<ts>/ [--attended] [--with-gripper-die]``

* ``sim`` (the dry run): spawns ``so101_rl.servo.main --backend mujoco --shm-name sysid-<pid> --tick-log
  OUT/tick_log.bin`` as a child (like ``servo/soak.py``), waits for RUN and drives it over the servo's shared memory
  at ``policy_hz`` in real time; the child is stopped with SIGINT (the normal shutdown). **Never opens a serial
  device**: the mujoco backend has no port, and this process never opens one in any mode.
* ``real`` (HS1/HS2, attended only): refuses (exit 2) without ``--attended`` and without the H2/H3/H5/H7 artifacts
  ``<hw-dir>/{estop,deadman,rest-pose,calib-check}-*.json``. It does not start the servo: the operator runs
  ``pixi run servo --backend feetech ... --tick-log OUT/tick_log.bin`` in another terminal (the servo process owns the
  bus, ARCHITECTURE §3) and this runner attaches to its shm segment (``--shm-name``, default from the config).
* ``synthetic``: an in-process ``ServoCore`` + ``MujocoBus`` rig in simulated time (the ``tests/test_action_units``
  Rig: limp at rest, button, ARM, re-energise to home, RUN) with ground-truth :class:`SimParams` (``--truth``) and
  ``--present-noise`` ticks of uniform integer measurement noise on the logged present positions. It writes the same
  run layout fast and deterministically; it is the data source of the ground-truth recovery tests.

Outputs in ``--out``: ``tick_log.bin`` (TICK_LOG_DTYPE), ``episodes/ep_*.h5`` (F8, one per segment; rows sampled at
10 Hz from the state record, an approximation of ``servo.core.log_row``: the 30 Hz tick log is the sysid data),
``plan.json`` (segment table with ``t_start_ns`` / ``t_stop_ns`` on the servo clock and the measured clamp-active
fraction per segment). Exit codes: 0 ok, 1 an artifact check failed (L12: every episode passes ``check_episode``, the
tick log is non-empty and every segment has RUN ticks) or the arm left RUN, 2 refused (real-mode gates, bad plan).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so101_rl.config import JOINT_NAMES, MotorCalibration, So101Config, load_calibration, load_config
from so101_rl.sim.sysid.excitation import (
    ExcitationPlan,
    SegmentSpan,
    check_plan_in_box,
    generate_deltas,
    named_plan,
    plan_table,
    segment_spans,
)

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
HW_ARTIFACTS: tuple[str, ...] = ("estop", "deadman", "rest-pose", "calib-check")  # H2, H3, H5, H7
RUN_TIMEOUT_S = 90.0
_NS = 1_000_000_000


# --------------------------------------------------------------------------------------------- gates
def real_mode_refusals(*, attended: bool, hw_dir: Path) -> list[str]:
    """Why ``--backend real`` must not run ([] = allowed). Checked before anything else happens."""
    reasons = []
    if not attended:
        reasons.append("--backend real needs --attended (Tier 0: a human at the e-stop for the whole run)")
    missing = [a for a in HW_ARTIFACTS if not sorted(Path(hw_dir).glob(f"{a}-*.json"))]
    if missing:
        reasons.append(
            f"missing hardware artifacts in {hw_dir}: " + ", ".join(f"{a}-*.json" for a in missing) + " (H2/H3/H5/H7)"
        )
    return reasons


# --------------------------------------------------------------------------------------------- clamp statistics
def clamp_active_fraction(recs: np.ndarray, cfg: So101Config) -> dict[str, float]:
    """Fraction of the RUN loop ticks in ``recs`` on which a clamp stage acted.

    From the tick log: the TRACKING / JOINT_LIMIT / BOX bits of ``trips`` (the pipeline reports them for
    information) and the velocity cap reconstructed as ``|target[k] - target[k-1]| >= max_ticks_per_tick`` (the cap
    sets no trip bit). ``any`` is the union (the plan's clamp-active fraction)."""
    from so101_rl.safety.types import TripCause

    n = len(recs)
    if n == 0:
        return {"any": 0.0, "tracking": 0.0, "joint_limit": 0.0, "box": 0.0, "velocity": 0.0, "ticks": 0}
    trips = recs["trips"].astype(np.int64)
    tr = (trips & int(TripCause.TRACKING)) != 0
    jl = (trips & int(TripCause.JOINT_LIMIT)) != 0
    bx = (trips & int(TripCause.BOX)) != 0
    tgt = recs["target"].astype(np.int64)
    vel = np.zeros(n, dtype=bool)
    vel[1:] = (np.abs(np.diff(tgt, axis=0)) >= np.asarray(cfg.action.max_ticks_per_tick)[None, :]).any(axis=1)
    anyc = tr | jl | bx | vel
    return {
        "any": float(anyc.mean()),
        "tracking": float(tr.mean()),
        "joint_limit": float(jl.mean()),
        "box": float(bx.mean()),
        "velocity": float(vel.mean()),
        "ticks": int(n),
    }


def _segment_records(recs: np.ndarray, t0: int, t1: int) -> np.ndarray:
    from so101_rl.safety.types import SmState
    from so101_rl.servo.core import PHASE_LOOP

    m = (recs["t_ns"] >= t0) & (recs["t_ns"] < t1) & (recs["phase"] == PHASE_LOOP) & (recs["sm_state"] == int(SmState.RUN))
    return recs[m]


# --------------------------------------------------------------------------------------------- driver
@dataclass
class _Driven:
    segments: list[dict[str, Any]]
    left_run: bool
    steps: int


def _obs_row(st: np.ndarray, cfg: So101Config, fk: Any) -> dict[str, np.ndarray]:
    from so101_rl.config import CENTER_TICK

    present = st["present_ticks"][0].astype(np.int32)
    tip = fk.gripper_xyz_from_ticks(present, np.zeros(3))
    vmax = float(cfg.rates.servo_hz) * np.asarray(cfg.action.max_ticks_per_tick, dtype=np.float64)
    obs = np.concatenate(
        [(present - CENTER_TICK) / 2048.0, np.clip(st["velocity"][0] / vmax, -2.0, 2.0), tip]
    ).astype(np.float32)
    return {"observation": obs, "achieved_goal": tip.astype(np.float32), "desired_goal": tip.astype(np.float32)}


def drive_plan(
    ipc: Any,
    plan: ExcitationPlan,
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    out_dir: Path,
    *,
    advance: Callable[[], None],
    now_ns: Callable[[], int],
    alive: Callable[[], bool] = lambda: True,
    run_id: str | None = None,
) -> _Driven:
    """Send the plan's deltas at one command per ``advance()`` (one policy period) and log one F8 episode per segment.

    ``ipc`` is anything with the shm block API (``ServoShm`` or ``LocalIpc``). The loop does no file I/O: episode rows
    are buffered and written after the plan (an ``fsync`` inside the 10 Hz loop made the command stale on a loaded
    host, HOLD -> safe stop). HOLD (a late command) is tolerated, the next command resumes RUN; any other state, or
    ``alive()`` turning false, stops the plan (``left_run``)."""
    from so101_rl.logging.episode_logger import EpisodeLogger, StepRecord
    from so101_rl.safety.fk import KinematicChain, So101FK
    from so101_rl.safety.types import SmState
    from so101_rl.servo.shm import CmdMode, new_cmd, new_state

    fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
    deltas = generate_deltas(plan, cfg)
    spans = segment_spans(plan, cfg)
    max_d = np.asarray(cfg.action.max_delta_ticks, dtype=np.float32)
    st, c = new_state(), new_cmd()
    starts = {s.start: s for s in spans}
    info: list[dict[str, Any]] = []
    rows: list[list[StepRecord]] = []
    cur: SegmentSpan | None = None
    pending: tuple[np.ndarray, dict[str, np.ndarray]] | None = None  # (delta, obs) of the open episode's last step
    left_run = False
    step = 0
    driving = (int(SmState.RUN), int(SmState.HOLD))

    def log_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        delta, obs = pending
        rows[-1].append(
            StepRecord(
                t_mono_ns=int(st["t_mono_ns"][0]),
                obs=obs,
                action_cmd=(delta / max_d).astype(np.float32),
                applied_delta_ticks=st["applied_delta_ticks"][0].astype(np.float32),
                target_ticks=st["target_ticks"][0].astype(np.int32),
                present_ticks=st["present_ticks"][0].astype(np.int32),
                reward=0.0,
                terminated=False,
                truncated=False,
                trips=int(st["fault_bits"][0]),
                verifier=0,
            )
        )
        pending = None

    def close_segment(aborted: bool) -> None:
        nonlocal cur
        if cur is None:
            return
        log_pending()
        info[-1]["t_stop_ns"] = int(now_ns())
        info[-1]["aborted"] = aborted
        cur = None

    try:
        for step in range(len(deltas) + 1):
            ipc.read_state(st)
            if cur is not None and cur.stop == step:
                close_segment(False)
            if step == len(deltas):
                break
            if int(st["sm_state"][0]) not in driving or not alive():
                left_run = True
                break
            if step in starts:
                cur = starts[step]
                info.append({"index": cur.index, "t_start_ns": int(now_ns())})
                rows.append([])
            if cur is not None:
                log_pending()
                pending = (deltas[step].copy(), _obs_row(st, cfg, fk))
            c["t_mono_ns"] = now_ns()
            c["state_seq"] = st["seq"][0]
            c["mode"] = int(CmdMode.RUN)
            c["delta_ticks"][0] = deltas[step]
            ipc.write_cmd(c)
            advance()
    finally:
        close_segment(True)
        logger = EpisodeLogger(out_dir.parent, out_dir.name if run_id is None else run_id, keep_full_frames="never")
        try:
            for seg_info, recs in zip(info, rows, strict=True):
                if not recs:
                    continue
                span = spans[seg_info["index"]]
                logger.start_episode({"segment": span.index, "kind": span.kind, "label": span.label, "plan": plan.name})
                for r in recs:
                    logger.log_step(r)
                seg_info["episode"] = str(logger.end_episode("trip" if seg_info.get("aborted") else "truncated"))
        finally:
            logger.close()
    return _Driven(info, left_run, step)


def write_plan_json(
    out_dir: Path,
    plan: ExcitationPlan,
    cfg: So101Config,
    driven: _Driven,
    *,
    backend: str,
    with_gripper_die: bool,
    extra: dict[str, Any] | None = None,
    extra_errors: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Merge the plan table, the driven segment times and the tick-log clamp statistics; run the artifact checks."""
    from so101_rl.logging.checks import check_episode
    from so101_rl.servo.core import TICK_LOG_DTYPE

    errors: list[str] = list(extra_errors or [])
    log_path = out_dir / "tick_log.bin"
    recs = np.zeros(0, dtype=TICK_LOG_DTYPE)
    if not log_path.is_file() or log_path.stat().st_size == 0:
        errors.append("tick_log.bin missing or empty")
    elif log_path.stat().st_size % TICK_LOG_DTYPE.itemsize:
        errors.append("tick_log.bin size is not a multiple of the record size")
    else:
        recs = np.fromfile(log_path, dtype=TICK_LOG_DTYPE)
    table = plan_table(plan, cfg)
    by_index = {d["index"]: d for d in driven.segments}
    for row in table:
        d = by_index.get(row["index"])
        if d is None:
            row["completed"] = False
            errors.append(f"segment {row['index']} ({row['label']}) was not run")
            continue
        row.update({k: v for k, v in d.items() if k != "index"})
        row["completed"] = not d.get("aborted", False)
        seg = _segment_records(recs, d["t_start_ns"], d.get("t_stop_ns", d["t_start_ns"]))
        stats = clamp_active_fraction(seg, cfg)
        row["clamp_active_frac"] = stats["any"]
        row["clamp_active_by_stage"] = {k: v for k, v in stats.items() if k not in ("any", "ticks")}
        row["run_ticks"] = stats["ticks"]
        if stats["ticks"] == 0:
            errors.append(f"segment {row['index']} ({row['label']}): no RUN ticks in the tick log")
        ep = d.get("episode")
        if ep:
            rep = check_episode(Path(ep), cfg)
            if not rep.ok:
                errors.append(f"{ep}: {'; '.join(rep.errors)}")
    if driven.left_run:
        errors.append(f"the servo left RUN (or the child died) at policy step {driven.steps}")
    doc = {
        "name": plan.name,
        "seed": plan.seed,
        "policy_hz": plan.policy_hz,
        "servo_hz": cfg.rates.servo_hz,
        "backend": backend,
        "with_gripper_die": with_gripper_die,
        "created": dt.datetime.now(dt.UTC).isoformat(),
        "joint_names": list(JOINT_NAMES),
        "segments": table,
        "artifact_errors": errors,
        **(extra or {}),
    }
    (out_dir / "plan.json").write_text(json.dumps(doc, indent=2, default=_json_default))
    return doc, errors


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------------------------- backends
class SimTimeClock:
    """MuJoCo-time clock: ``sleep_until`` steps the physics (the role of the real 30 Hz loop's sleep)."""

    def __init__(self, bus: Any) -> None:
        self.bus = bus
        self.period_ns = round(bus.steps_per_tick * bus.model.opt.timestep * _NS)
        self._offset = 1_000 * _NS

    def monotonic_ns(self) -> int:
        return round(self.bus.data.time * _NS) + self._offset

    def sleep_until(self, t_ns: int) -> None:
        n = -(-(t_ns - self.monotonic_ns()) // self.period_ns)
        if n > 0:
            self.bus.step_ticks(n)


class _TickCollector:
    def __init__(self, dtype: np.dtype) -> None:
        self.rows: list[np.ndarray] = []
        self.dtype = dtype

    def push(self, rec: np.ndarray) -> None:
        self.rows.append(np.array(rec, dtype=self.dtype).copy())

    def array(self) -> np.ndarray:
        return np.array(self.rows, dtype=self.dtype) if self.rows else np.zeros(0, dtype=self.dtype)


def run_synthetic(
    plan: ExcitationPlan,
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    out_dir: Path,
    *,
    truth: Any,
    present_noise_ticks: int = 0,
    seed: int = 0,
    with_gripper_die: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Ground-truth run: ``ServoCore`` + ``MujocoBus`` with ``truth`` (a :class:`SimParams`) in simulated time."""
    from so101_rl.servo.core import TICK_LOG_DTYPE, build_servo_core
    from so101_rl.servo.heartbeat import FakeHeartbeatLink, FakeMcu
    from so101_rl.servo.shm import LocalIpc
    from so101_rl.sim.model import load_model
    from so101_rl.sim.mujoco_bus import MujocoBus

    out_dir.mkdir(parents=True, exist_ok=True)
    model, data = load_model()
    truth.apply_cpu(model)  # before the bus: MujocoBus caches the gains it restores on torque-on
    bus = _deadband_bus(MujocoBus, truth.deadband_ticks)(
        model, data, calibration, cfg, action_delay_ticks=truth.delay_ticks
    )
    bus.connect()
    bus.reset_to_keyframe("rest")
    bus.set_torque(False)  # limp at rest, like the arm after power-up
    bus.written_goals = _NoList()
    clock = SimTimeClock(bus)
    mcu = FakeMcu(clock)
    ipc = LocalIpc()
    collector = _TickCollector(TICK_LOG_DTYPE)
    core = build_servo_core(cfg, calibration, bus, ipc, FakeHeartbeatLink(mcu), clock, tick_log=collector, log=lambda m: None)
    mcu.set_button(True)
    core.start(arm_timeout_s=5.0)
    mcu.set_button(False)
    period = core.period_ns
    ticks_per_step = max(1, round(cfg.rates.servo_hz / cfg.rates.policy_hz))
    t = clock.monotonic_ns()
    core.tick()  # one RUN tick publishes a RUN state record (start() leaves the boot-phase record behind)
    clock.sleep_until(t + period)

    def advance() -> None:
        for _ in range(ticks_per_step):
            t = clock.monotonic_ns()
            core.tick()
            clock.sleep_until(t + period)

    driven = drive_plan(ipc, plan, cfg, calibration, out_dir, advance=advance, now_ns=clock.monotonic_ns)
    recs = collector.array()
    if present_noise_ticks:
        rng = np.random.default_rng(seed)
        noise = rng.integers(-present_noise_ticks, present_noise_ticks + 1, size=recs["present"].shape)
        recs["present"] = np.clip(recs["present"].astype(np.int64) + noise, 0, 4095).astype(np.int32)
    recs.tofile(out_dir / "tick_log.bin")
    extra = {
        "truth": truth.to_dict(),
        "present_noise_ticks": int(present_noise_ticks),
        "noise_seed": int(seed),
        "note": "synthetic ground-truth run: ServoCore + MujocoBus in simulated time",
    }
    return write_plan_json(out_dir, plan, cfg, driven, backend="synthetic", with_gripper_die=with_gripper_die, extra=extra)


class _NoList(list):
    def append(self, _item: Any) -> None:
        return None


def _deadband_bus(base: type, band: np.ndarray) -> type:
    """``base`` (MujocoBus) whose written goal follows the D-BACKLASH target hysteresis (identity for band 0)."""
    band = np.asarray(band, dtype=np.int64)
    if not band.any():
        return base

    class DeadbandBus(base):  # type: ignore[misc, valid-type]
        _held: np.ndarray | None = None

        def write_goal(self, goal_ticks: np.ndarray) -> bool:
            g = np.asarray(goal_ticks)
            if g.shape == band.shape and np.issubdtype(g.dtype, np.integer):
                g64 = g.astype(np.int64)
                held = g64 if self._held is None else self._held
                held = np.where(g64 > held + band, g64 - band, np.where(g64 < held - band, g64 + band, held))
                self._held = held
                g = held.astype(np.int32)
            return super().write_goal(g)

    return DeadbandBus


def _wait_for_run(name: str, proc: subprocess.Popen, timeout_s: float) -> Any:
    from so101_rl.safety.types import SmState
    from so101_rl.servo.shm import ServoShm, new_state

    deadline = time.monotonic() + timeout_s
    shm = None
    st = new_state()
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"servo process exited early with code {proc.returncode}")
            if shm is None:
                try:
                    shm = ServoShm.attach(name)
                except (FileNotFoundError, RuntimeError):
                    time.sleep(0.05)
                    continue
            if shm.read_state(st) and int(st["sm_state"][0]) == int(SmState.RUN):
                return shm
            time.sleep(0.01)
        raise TimeoutError(f"servo did not reach RUN within {timeout_s} s")
    except BaseException:
        if shm is not None:
            shm.close()
        raise


def _realtime_advance(policy_hz: float) -> tuple[Callable[[], None], Callable[[], int]]:
    period_ns = round(_NS / policy_hz)
    nxt = [time.monotonic_ns()]

    def advance() -> None:
        nxt[0] += period_ns
        rem = nxt[0] - time.monotonic_ns()
        if rem > 0:
            time.sleep(rem / _NS)
        else:  # overrun: restart the grid, never burst commands (the servo would only see the last one)
            nxt[0] = time.monotonic_ns()

    return advance, time.monotonic_ns


def run_sim(
    plan: ExcitationPlan,
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    out_dir: Path,
    *,
    calibration_path: Path | None,
    with_gripper_die: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Dry run against a ``servo --backend mujoco`` child (real time, the same shm path as the arm)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shm_name = f"sysid-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    cmd = [
        sys.executable, "-m", "so101_rl.servo.main", "--backend", "mujoco", "--shm-name", shm_name,
        "--tick-log", str(out_dir / "tick_log.bin"), "--summary", str(out_dir / "servo_summary.json"),
    ]  # fmt: skip
    if calibration_path is not None:
        cmd += ["--calibration", str(calibration_path)]
    proc = subprocess.Popen(cmd)
    shm = None
    driven = _Driven([], True, 0)
    try:
        shm = _wait_for_run(shm_name, proc, RUN_TIMEOUT_S)
        advance, now = _realtime_advance(plan.policy_hz)
        driven = drive_plan(shm, plan, cfg, calibration, out_dir, advance=advance, now_ns=now,
                            alive=lambda: proc.poll() is None)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=cfg.safe_state.bound_s + 30.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if shm is not None:
            shm.close()
    child = [] if proc.returncode == 0 else [f"servo child exited with code {proc.returncode}"]
    return write_plan_json(out_dir, plan, cfg, driven, backend="sim", with_gripper_die=with_gripper_die,
                           extra={"servo_exit_code": proc.returncode}, extra_errors=child)


def run_real(
    plan: ExcitationPlan,
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    out_dir: Path,
    *,
    shm_name: str,
    with_gripper_die: bool,
) -> tuple[dict[str, Any], list[str]]:  # pragma: no cover - hardware (HS1/HS2), never run without the arm
    """Attach to the operator-started servo (it owns the bus and writes OUT/tick_log.bin) and drive the plan."""
    from so101_rl.servo.shm import ServoShm

    out_dir.mkdir(parents=True, exist_ok=True)
    with ServoShm.attach(shm_name) as shm:
        advance, now = _realtime_advance(plan.policy_hz)
        driven = drive_plan(shm, plan, cfg, calibration, out_dir, advance=advance, now_ns=now)
    # The servo is still writing OUT/tick_log.bin: wait until the operator stopped it (its shm segment is gone)
    # so the log is complete before it is analysed.
    print("sysid-excite: plan done. Stop the servo now (Ctrl-C in its terminal); waiting for it to exit ...")
    deadline = time.monotonic() + cfg.safe_state.bound_s + 600.0
    while time.monotonic() < deadline:
        try:
            ServoShm.attach(shm_name).close()
        except (FileNotFoundError, RuntimeError):
            break
        time.sleep(0.5)
    else:
        return write_plan_json(out_dir, plan, cfg, driven, backend="real", with_gripper_die=with_gripper_die,
                               extra_errors=["the servo did not exit: tick_log.bin may be incomplete"])
    return write_plan_json(out_dir, plan, cfg, driven, backend="real", with_gripper_die=with_gripper_die)


# --------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sysid-excite", description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("sim", "real", "synthetic"), required=True)
    ap.add_argument("--plan", default="default", help="default | short | fit40")
    ap.add_argument("--out", type=Path, required=True, help="run directory (runs/sysid/<ts>/)")
    ap.add_argument("--attended", action="store_true", help="real: a human is at the e-stop (Tier 0)")
    ap.add_argument("--with-gripper-die", action="store_true", help="add the gripper/die segments (HS2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", type=Path, default=None, help="LeRobot calibration JSON (default: from config)")
    ap.add_argument("--hw-dir", type=Path, default=Path("runs/hw"), help="real: H-track artifact directory")
    ap.add_argument("--shm-name", default=None, help="real: shm segment of the running servo (default: config)")
    ap.add_argument("--truth", type=Path, default=None, help="synthetic: ground-truth sim_params yaml (default: prior)")
    ap.add_argument("--present-noise", type=int, default=0, help="synthetic: ±N ticks uniform noise on logged present")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend == "real":
        refusals = real_mode_refusals(attended=args.attended, hw_dir=args.hw_dir)
        if refusals:
            for r in refusals:
                print(f"sysid-excite: refused: {r}", file=sys.stderr)
            return EXIT_REFUSED
    base = load_config() if args.calibration is None else None
    calib_path = args.calibration if args.calibration is not None else base.calibration_path
    calibration = load_calibration(Path(calib_path).expanduser())
    cfg = load_config(calibration=calibration)
    try:
        plan = named_plan(args.plan, cfg, with_gripper_die=args.with_gripper_die, seed=args.seed)
    except ValueError as exc:
        print(f"sysid-excite: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    from so101_rl.safety.fk import KinematicChain

    chk = check_plan_in_box(plan, cfg, calibration, KinematicChain.from_yaml(cfg.kinematics_path),
                            np.asarray(cfg.safe_state.home_ticks))
    if not chk.ok:
        print(f"sysid-excite: plan {plan.name!r} fails the in-box preview: {'; '.join(chk.reasons)}", file=sys.stderr)
        return EXIT_REFUSED
    out = args.out
    # real mode: the operator-started servo may already have created OUT/tick_log.bin
    allowed = {"tick_log.bin", "servo_summary.json"} if args.backend == "real" else set()
    if out.exists() and any(p.name not in allowed for p in out.iterdir()):
        print(f"sysid-excite: {out} is not empty", file=sys.stderr)
        return EXIT_REFUSED
    if args.backend == "sim":
        doc, errors = run_sim(plan, cfg, calibration, out, calibration_path=args.calibration,
                              with_gripper_die=args.with_gripper_die)
    elif args.backend == "synthetic":
        from so101_rl.sim.model import load_model
        from so101_rl.sim.sysid.params import SimParams

        truth = (SimParams.from_yaml(args.truth, allow_unverified=True) if args.truth is not None
                 else SimParams.from_mjcf(load_model()[0]))
        doc, errors = run_synthetic(plan, cfg, calibration, out, truth=truth, present_noise_ticks=args.present_noise,
                                    seed=args.seed, with_gripper_die=args.with_gripper_die)
    else:  # pragma: no cover - hardware
        doc, errors = run_real(plan, cfg, calibration, out, shm_name=args.shm_name or cfg.shm_name,
                               with_gripper_die=args.with_gripper_die)
    for row in doc["segments"]:
        print(f"{row['label']:<28} run_ticks {row.get('run_ticks', 0):>5}  clamp_active_frac {row.get('clamp_active_frac', 0):.3f}")
    print(f"plan.json: {out / 'plan.json'}")
    if errors:
        for e in errors:
            print(f"sysid-excite: FAIL: {e}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
