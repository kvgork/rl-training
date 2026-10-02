"""Soak harness and gate (plan F6, Task 3.3).

``python -m so101_rl.servo.soak --backend {fake,mujoco,feetech} --minutes M`` starts the servo process
(``so101_rl.servo.main``) as a child with a tick log, attaches its shm segment as a random-policy client
(uniform deltas within ``max_delta_ticks``, pulled back toward home so the arm keeps exercising the box instead
of parking against a face) at ``policy_hz`` for ``M`` minutes, then stops the child with SIGINT (the normal
shutdown) and writes ``runs/soak-<ts>/report.json``.

Sources of the report fields: ``tick_stats`` (jitter/duration over the loop ticks of the soak window),
``hb_max_gap_ms`` / ``hb_gaps_over_200ms`` (the window's start and end count as gap endpoints, so silence before
the first or after the last heartbeat is a gap too), ``hold_entries`` (entries into HOLD: the policy was late),
``max_temp_c`` / ``thermal_trips`` (rising edges of THERMAL) / ``unexpected_safe_states`` (entries into
SAFE_STOPPING, TORQUE_OFF or LATCHED while the policy was driving) come from the tick log; ``calib_mismatches`` (failed calibration checks at each CHECK / re-energise),
``gc_collections`` and ``rt_active`` from the child's summary; ``rss_growth_mb`` from ``psutil`` on the child
(RUN start -> end of the soak window); ``kernel`` from ``platform``.

``check_soak_report(path, strict=...)`` is the Tier-1 gate: at least ``GATE_MIN_MINUTES`` (30) of soak, p99.9
tick jitter < 5 ms, no heartbeat gap over 200 ms, no HOLD entry, no thermal trip, no calibration mismatch.
``strict=False`` (a non-RT host) turns those into warnings. A run that is not a valid soak fails on any host (U3
review item 3, the L12 failure mode: an empty run must never pass): the servo exited nonzero, wrote no summary,
entered a safe state while the random policy was driving it (``unexpected_safe_states``), ran shorter than 90 %
of the requested minutes, or has fewer than 90 % of the expected loop-tick samples
(``minutes * 60 * servo_hz``) or policy steps (``minutes * 60 * policy_hz``). The CLI uses ``strict=rt_active``
and exits 1 when the gate fails.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from so101_rl.config import JOINT_NAMES, load_config
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo.core import PHASE_LOOP, TICK_LOG_DTYPE
from so101_rl.servo.shm import CmdMode, ServoShm, new_cmd, new_state
from so101_rl.servo.timing import TickStats

GATE_P999_MS = 5.0
GATE_HB_GAP_MS = 200.0
GATE_MIN_MINUTES = 30.0  # the first Tier-1 soak (ARCHITECTURE section 6.6); strict gate only
MIN_FRACTION = 0.9  # samples / policy steps / minutes below this fraction of the expected count: not a valid soak
_NS = 1_000_000_000
_STOP_STATES = (SmState.SAFE_STOPPING, SmState.TORQUE_OFF, SmState.LATCHED)
_RUN_TIMEOUT_S = 60.0


@dataclass
class SoakReport:
    rt_active: bool
    kernel: str
    minutes: float
    tick_stats: dict
    hb_max_gap_ms: float
    hb_gaps_over_200ms: int
    thermal_trips: int
    calib_mismatches: int
    max_temp_c: dict[str, int]
    rss_growth_mb: float
    gc_collections: int
    unexpected_safe_states: int
    hold_entries: int = 0
    requested_minutes: float = 0.0
    servo_hz: float = 0.0
    policy_hz: float = 0.0
    backend: str = ""
    servo_exit_code: int | None = None
    policy_steps: int = 0
    summary_found: bool = True
    notes: list[str] = field(default_factory=list)


def check_soak_report(path: Path, *, strict: bool) -> tuple[bool, list[str]]:
    """Gate a report.json. Returns ``(ok, messages)``; with ``strict=False`` failures are warnings and ok is True."""
    rep = json.loads(Path(path).read_text())
    problems: list[str] = []
    invalid: list[str] = []  # the run itself is not a valid soak: fails even on a non-RT host
    if rep.get("servo_exit_code", 0) != 0:
        invalid.append(f"servo process exited with code {rep.get('servo_exit_code')}")
    if not rep.get("summary_found", True):
        invalid.append("the servo wrote no summary (crashed?)")
    if int(rep.get("unexpected_safe_states", 0)) != 0:
        invalid.append(f"{rep['unexpected_safe_states']} unexpected safe-state entr(y/ies) while the policy was driving")
    minutes = float(rep.get("minutes", 0.0))
    requested = float(rep.get("requested_minutes", 0.0)) or minutes
    servo_hz = float(rep.get("servo_hz", 0.0))
    policy_hz = float(rep.get("policy_hz", 0.0))
    if not (servo_hz > 0 and policy_hz > 0):
        invalid.append("report has no servo_hz / policy_hz: cannot check the sample counts")
    if not minutes > 0 or minutes < MIN_FRACTION * requested:
        invalid.append(f"soak ran {minutes:.3f} min of the {requested:.3f} min requested")
    want_samples = MIN_FRACTION * minutes * 60.0 * servo_hz
    samples = int(rep["tick_stats"].get("samples", 0))
    if samples < max(1.0, want_samples):
        invalid.append(f"{samples} loop-tick samples, expected >= {want_samples:.0f} ({minutes:.3f} min at {servo_hz:g} Hz)")
    want_steps = MIN_FRACTION * minutes * 60.0 * policy_hz
    steps = int(rep.get("policy_steps", 0))
    if steps < max(1.0, want_steps):
        invalid.append(f"{steps} policy steps, expected >= {want_steps:.0f} ({minutes:.3f} min at {policy_hz:g} Hz)")
    if minutes < GATE_MIN_MINUTES:
        problems.append(f"soak of {minutes:.2f} min is shorter than the {GATE_MIN_MINUTES:g} min Tier-1 floor")
    p999 = float(rep["tick_stats"]["p999_ms"])
    if not p999 < GATE_P999_MS:
        problems.append(f"p99.9 tick jitter {p999:.3f} ms >= {GATE_P999_MS} ms")
    if int(rep["hb_gaps_over_200ms"]) != 0:
        problems.append(f"{rep['hb_gaps_over_200ms']} heartbeat gap(s) over {GATE_HB_GAP_MS:.0f} ms")
    if int(rep.get("hold_entries", 0)) != 0:
        problems.append(f"{rep['hold_entries']} HOLD entr(y/ies): the policy was late")
    if int(rep["thermal_trips"]) != 0:
        problems.append(f"{rep['thermal_trips']} thermal trip(s)")
    if int(rep["calib_mismatches"]) != 0:
        problems.append(f"{rep['calib_mismatches']} calibration mismatch(es)")
    if strict:
        return (not (problems or invalid), invalid + problems)
    return (not invalid, invalid + [f"WARN (non-RT host, advisory): {p}" for p in problems])


def analyse_tick_log(recs: np.ndarray, *, t_start_ns: int, t_end_ns: int, period_ns: int) -> dict:
    """Soak metrics from TICK_LOG_DTYPE records inside ``[t_start_ns, t_end_ns]`` (the policy-driven window)."""
    win = recs[(recs["t_ns"] >= t_start_ns) & (recs["t_ns"] <= t_end_ns)]
    loop = win[win["phase"] == PHASE_LOOP]
    stats = TickStats(period_ns, capacity=max(1, len(loop)))
    for j, d in zip(loop["jitter_ns"].tolist(), loop["duration_ns"].tolist(), strict=True):
        stats.record(int(j), int(d))
    # the window's start and end are gap endpoints: silence before the first / after the last heartbeat counts
    hb_t = win["t_ns"][win["hb_sent"].astype(bool)]
    if t_end_ns > t_start_ns:
        gaps = np.diff(np.concatenate(([t_start_ns], hb_t.astype(np.int64), [t_end_ns])))
        hb_max_gap_ms = float(gaps.max()) / 1e6
        over = int((gaps > GATE_HB_GAP_MS * 1e6).sum())
    else:
        hb_max_gap_ms, over = 0.0, 0  # no window at all: the sample-count checks reject the run
    thermal = (win["trips"] & int(TripCause.THERMAL)) != 0
    thermal_trips = int(np.count_nonzero(thermal[1:] & ~thermal[:-1]) + (1 if len(thermal) and thermal[0] else 0))
    st = win["sm_state"].astype(np.int64)
    stop = np.isin(st, [int(s) for s in _STOP_STATES])
    unexpected = int(np.count_nonzero(stop[1:] & ~stop[:-1]) + (1 if len(stop) and stop[0] else 0))
    hold = st == int(SmState.HOLD)
    hold_entries = int(np.count_nonzero(hold[1:] & ~hold[:-1]) + (1 if len(hold) and hold[0] else 0))
    tmax = win["temp_c"].max(axis=0) if len(win) else np.zeros(len(JOINT_NAMES), dtype=np.int64)
    return {
        "tick_stats": stats.summary(),
        "hb_max_gap_ms": hb_max_gap_ms,
        "hb_gaps_over_200ms": over,
        "thermal_trips": thermal_trips,
        "unexpected_safe_states": unexpected,
        "hold_entries": hold_entries,
        "max_temp_c": {n: int(t) for n, t in zip(JOINT_NAMES, tmax, strict=True)},
    }


def _rss_mb(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss / (1 << 20)
    except Exception:  # noqa: BLE001 - psutil missing or the child already exited
        return None


def _wait_for_run(name: str, proc: subprocess.Popen, timeout_s: float) -> ServoShm:
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
            if shm.read_state(st) and int(st["sm_state"][0]) == SmState.RUN:
                return shm
            time.sleep(0.01)
        raise TimeoutError(f"servo did not reach RUN within {timeout_s} s")
    except BaseException:
        if shm is not None:
            shm.close()
        raise


def soak(
    *,
    backend: str,
    minutes: float,
    out_root: Path = Path("runs"),
    calibration: Path | None = None,
    seed: int = 0,
    extra_servo_args: list[str] | None = None,
) -> tuple[Path, SoakReport]:
    """Run one soak and write ``<out_root>/soak-<ts>/report.json``. Returns (report path, report)."""
    cfg = load_config()
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / f"soak-{ts}-{uuid.uuid4().hex[:4]}"
    out_dir.mkdir(parents=True, exist_ok=False)
    tick_log = out_dir / "ticks.bin"
    summary_path = out_dir / "servo_summary.json"
    shm_name = f"so101_soak_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    cmd = [
        sys.executable,
        "-m",
        "so101_rl.servo.main",
        "--backend",
        backend,
        "--shm-name",
        shm_name,
        "--tick-log",
        str(tick_log),
        "--summary",
        str(summary_path),
    ]
    if calibration is not None:
        cmd += ["--calibration", str(calibration)]
    cmd += list(extra_servo_args or [])
    proc = subprocess.Popen(cmd)
    notes: list[str] = []
    shm = None
    steps = 0
    t_start = t_end = 0
    rss0 = rss1 = None
    try:
        shm = _wait_for_run(shm_name, proc, _RUN_TIMEOUT_S)
        rss0 = _rss_mb(proc.pid)
        rng = np.random.default_rng(seed)
        max_d = cfg.action.max_delta_ticks.astype(np.float64)
        home = np.asarray(cfg.safe_state.home_ticks, dtype=np.float64)
        period = 1.0 / cfg.rates.policy_hz
        st = new_state()
        c = new_cmd()
        t_start = time.monotonic_ns()
        stop_at = time.monotonic() + minutes * 60.0
        nxt = time.monotonic()
        while time.monotonic() < stop_at and proc.poll() is None:
            if shm.read_state(st):
                present = st["present_ticks"][0].astype(np.float64)
                pull = np.clip(0.1 * (home - present), -max_d, max_d)
                delta = np.clip(rng.uniform(-1.0, 1.0, size=len(max_d)) * max_d + pull, -max_d, max_d)
                c["t_mono_ns"] = time.monotonic_ns()
                c["state_seq"] = st["seq"][0]
                c["mode"] = int(CmdMode.RUN)
                c["delta_ticks"][0] = delta.astype(np.float32)
                shm.write_cmd(c)
                steps += 1
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
        t_end = time.monotonic_ns()
        rss1 = _rss_mb(proc.pid)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=cfg.safe_state.bound_s + 30.0)
        except subprocess.TimeoutExpired:
            notes.append("servo did not exit after SIGINT; killed")
            proc.kill()
            proc.wait()
        if shm is not None:
            shm.close()

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    recs = np.fromfile(tick_log, dtype=TICK_LOG_DTYPE) if tick_log.exists() else np.zeros(0, TICK_LOG_DTYPE)
    m = analyse_tick_log(recs, t_start_ns=t_start, t_end_ns=t_end, period_ns=int(_NS / cfg.rates.servo_hz))
    if summary.get("tick_log_dropped"):
        notes.append(f"tick log dropped {summary['tick_log_dropped']} records")
    report = SoakReport(
        rt_active=bool(summary.get("rt_active", False)),
        kernel=str(summary.get("kernel", "")),
        minutes=(t_end - t_start) / 60e9 if t_end else 0.0,
        tick_stats=m["tick_stats"],
        hb_max_gap_ms=m["hb_max_gap_ms"],
        hb_gaps_over_200ms=m["hb_gaps_over_200ms"],
        thermal_trips=m["thermal_trips"],
        calib_mismatches=int(summary.get("calib_mismatches", 0)),
        max_temp_c=m["max_temp_c"],
        rss_growth_mb=(rss1 - rss0) if (rss0 is not None and rss1 is not None) else float("nan"),
        gc_collections=int(summary.get("gc_collections", 0)),
        unexpected_safe_states=m["unexpected_safe_states"],
        hold_entries=m["hold_entries"],
        requested_minutes=float(minutes),
        servo_hz=float(cfg.rates.servo_hz),
        policy_hz=float(cfg.rates.policy_hz),
        backend=backend,
        servo_exit_code=proc.returncode,
        policy_steps=steps,
        summary_found=bool(summary),
        notes=notes,
    )
    path = out_dir / "report.json"
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True, default=float))
    return path, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="so101-soak", description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("fake", "mujoco", "feetech"), default="fake")
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args, extra = ap.parse_known_args(argv)
    path, rep = soak(
        backend=args.backend,
        minutes=args.minutes,
        out_root=args.out,
        calibration=args.calibration,
        seed=args.seed,
        extra_servo_args=extra,
    )
    ok, msgs = check_soak_report(path, strict=rep.rt_active)
    print(f"soak report: {path}")
    for msg in msgs:
        print(f"  {msg}")
    print(f"gate: {'PASS' if ok else 'FAIL'} (strict={rep.rt_active})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
