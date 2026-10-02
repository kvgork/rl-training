"""Servo process runner (plan F6, Task 3.2): RT setup, absolute-deadline loop, signals, exit codes.

``run()`` owns the whole life of the servo process:

1. Real-time setup, best effort unless ``require_rt``: ``mlockall``, ``SCHED_FIFO`` priority 50 and CPU affinity.
   ``require_rt`` (the ``--require-rt`` flag or ``REQUIRE_RT=1``) makes any failure fatal: exit code 3.
2. The single-instance locks: the shm segment (``ServoShm.create``) and every serial port in ``serial_ports``
   (``SerialPortLock``; review item 14); exit 4 if another servo owns one. Then the bus and heartbeat factories and
   ``build_servo_core``, which runs ``safety.config_check.validate_geometry`` (exit 2 on a bad safe-state geometry).
3. ``ServoCore.start()`` (port without handshake -> BOOT -> button + ARM -> ping -> re-energise -> RUN; exit 2 if
   it fails), then ``gc.collect(); gc.freeze(); gc.disable()``.
4. The absolute-deadline loop: ``deadline += period`` every tick, so the rate does not drift with the tick body.
   After a large overrun (the tick finished a full period or more past its next deadline) the missed deadlines are
   SKIPPED and the loop waits for the next point of the original grid (resync, counted) instead of running the
   missed ticks back to back. Overruns are counted in the state record and the
   tick log, and reported on stderr at most once per second.
5. Every exit (``--max-ticks``, SIGINT, SIGTERM, SIGHUP, SIGQUIT, an exception) goes through
   ``ServoCore.shutdown``. The signal handlers only set a flag; the loop checks it once per tick, and a running
   re-energise polls it too. SIGHUP matters: closing the ssh session that started the servo must ramp the arm
   down, not kill the process (the arm would fall about 1 s later, when the MCU times out).

Exit codes: 0 normal stop, 1 exception in the loop (after a clean shutdown), 2 start-up failed, 3 RT required
but unavailable, 4 another servo process owns the shm segment.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
import os
import platform
import resource
import signal
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from so101_rl.config import JOINT_NAMES, MotorCalibration, So101Config, load_calibration
from so101_rl.safety.reenergize import torque_off_retry
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo.bus import Bus
from so101_rl.servo.core import TICK_LOG_DTYPE, ServoCore, StartupError, build_servo_core
from so101_rl.servo.heartbeat import HeartbeatLink
from so101_rl.servo.shm import SerialPortLock, ServoAlreadyRunning, ServoIpc, ServoShm
from so101_rl.servo.timing import Clock, RealClock, TickStats

EXIT_OK, EXIT_EXCEPTION, EXIT_STARTUP, EXIT_RT, EXIT_LOCKED = 0, 1, 2, 3, 4
RT_PRIORITY = 50
_MCL_CURRENT, _MCL_FUTURE = 1, 2
_NS = 1_000_000_000
_MIN_MEMLOCK = 512 << 20
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)  # all go through shutdown()


@dataclass
class RtStatus:
    active: bool = False  # SCHED_FIFO and mlockall both succeeded
    fifo: bool = False
    mlock: bool = False
    affinity: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def require_rt_from_env() -> bool:
    return os.environ.get("REQUIRE_RT", "") == "1"


def setup_realtime(*, priority: int = RT_PRIORITY, cpu: int | None = None) -> RtStatus:
    """Best-effort RT setup. Never raises; the caller decides whether a failure is fatal."""
    st = RtStatus()
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and soft < _MIN_MEMLOCK and os.geteuid() != 0:
            # MCL_FUTURE under a small limit would make later allocations fail: do not lock at all
            raise OSError(f"RLIMIT_MEMLOCK is {soft} B (needs >= {_MIN_MEMLOCK} B or unlimited)")
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        if libc.mlockall(_MCL_CURRENT | _MCL_FUTURE) == 0:
            st.mlock = True
        else:
            st.errors.append(f"mlockall: {os.strerror(ctypes.get_errno())}")
    except OSError as exc:
        st.errors.append(f"mlockall: {exc}")
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        st.fifo = True
    except (OSError, AttributeError) as exc:
        st.errors.append(f"SCHED_FIFO({priority}): {exc}")
    try:
        cpus = sorted(os.sched_getaffinity(0))
        want = cpu if cpu is not None else int(os.environ.get("SO101_SERVO_CPU", cpus[-1]))
        if want in cpus:
            os.sched_setaffinity(0, {want})
            st.affinity = [want]
        else:
            st.errors.append(f"affinity: cpu {want} not in {cpus}")
            st.affinity = cpus
    except (OSError, AttributeError, ValueError) as exc:
        st.errors.append(f"affinity: {exc}")
    st.active = st.fifo and st.mlock
    return st


class ServoRunner:
    """The absolute-deadline loop around one :class:`ServoCore`. ``request_stop`` is what the signal handlers call
    (it only sets a flag)."""

    def __init__(self, core: ServoCore, clock: Clock, *, log: Callable[[str], None] | None = None) -> None:
        self.core = core
        self.clock = clock
        self.stats = TickStats(core.period_ns)
        self.resyncs = 0
        self.ticks = 0
        self.signal_received: int | None = None
        self._stop = False
        self._log = log if log is not None else (lambda m: print(f"servo: {m}", file=sys.stderr, flush=True))
        self._last_overrun_log_ns = -(10**18)
        self._overruns_since_log = 0

    def request_stop(self, signum: int | None = None) -> None:
        if signum is not None and self.signal_received is None:
            self.signal_received = signum
        self._stop = True

    @property
    def stopping(self) -> bool:
        return self._stop

    def loop(self, max_ticks: int | None = None) -> int:
        """Tick until ``request_stop`` or ``max_ticks``; returns the number of ticks run."""
        clock, core = self.clock, self.core
        period = core.period_ns
        deadline = clock.monotonic_ns()
        n = 0
        while not self._stop and (max_ticks is None or n < max_ticks):
            clock.sleep_until(deadline)
            t0 = clock.monotonic_ns()
            res = core.tick(deadline_ns=deadline)
            n += 1
            t1 = clock.monotonic_ns()
            self.stats.record(t0 - deadline, t1 - t0)
            deadline += period
            if t1 - deadline >= period:  # a full period or more behind: skip to the next grid point, never burst
                deadline += -(-(t1 - deadline) // period) * period
                self.resyncs += 1
            if res.overrun_ns:
                self._overruns_since_log += 1
                if t1 - self._last_overrun_log_ns >= _NS:
                    self._log(
                        f"overrun: tick {core.tick_count} ended {res.overrun_ns / 1e6:.2f} ms past its period "
                        f"({self._overruns_since_log} since the last report, state {res.state.name})"
                    )
                    self._last_overrun_log_ns = t1
                    self._overruns_since_log = 0
        self.ticks += n
        return n


def _install_signals(runner: ServoRunner) -> dict[int, Any]:
    old: dict[int, Any] = {}

    def handler(signum, _frame):  # flag only: no I/O, no allocation-heavy work in the handler
        runner.request_stop(signum)

    for sig in STOP_SIGNALS:
        try:
            old[sig] = signal.signal(sig, handler)
        except ValueError:  # not the main thread
            pass
    return old


def _restore_signals(old: dict[int, Any]) -> None:
    for sig, h in old.items():
        signal.signal(sig, h)


def _gc_collections() -> int:
    return sum(s.get("collections", 0) for s in gc.get_stats())


def run(
    cfg: So101Config,
    bus_factory: Callable[[], Bus],
    hb_factory: Callable[[], HeartbeatLink | None],
    *,
    require_rt: bool,
    max_ticks: int | None = None,
    clock: Clock | None = None,
    ipc: ServoIpc | None = None,
    calibration: dict[str, MotorCalibration] | None = None,
    tick_log_path: Path | None = None,
    summary_path: Path | None = None,
    arm_timeout_s: float | None = None,
    install_signals: bool = True,
    freeze_gc: bool = True,
    on_core: Callable[[ServoCore, ServoRunner], None] | None = None,
    serial_ports: tuple[str, ...] = (),
) -> int:
    """Run the servo process (see the module docstring). Returns the exit code. ``serial_ports``: every serial
    device this servo will own (bus, watchdog MCU), each locked for the life of the process."""
    log = lambda m: print(f"servo: {m}", file=sys.stderr, flush=True)
    clock = clock if clock is not None else RealClock()
    summary: dict[str, Any] = {"kernel": platform.release(), "exit_code": None}

    rt = setup_realtime() if (require_rt or isinstance(clock, RealClock)) else RtStatus(errors=["fake clock"])
    summary["rt"] = asdict(rt)
    summary["rt_active"] = rt.active
    if not rt.active:
        msg = "real-time setup incomplete: " + "; ".join(rt.errors)
        if require_rt:
            log(msg + " (REQUIRE_RT: refusing to run; grant rtprio/memlock, e.g. limits.conf, or unset REQUIRE_RT)")
            summary["exit_code"] = EXIT_RT
            _write_summary(summary_path, summary)
            return EXIT_RT
        log(msg + " (best effort: continuing without RT)")

    if calibration is None:
        calibration = load_calibration(cfg.calibration_path)
    own_ipc = None
    tick_log = None
    port_locks: list[SerialPortLock] = []
    try:
        if ipc is None:
            own_ipc = ipc = ServoShm.create(cfg.shm_name)
        for port in dict.fromkeys(serial_ports):
            port_locks.append(SerialPortLock(port))
    except ServoAlreadyRunning as exc:
        log(str(exc))
        for lock in port_locks:
            lock.release()
        if own_ipc is not None:
            own_ipc.close()
        summary["exit_code"] = EXIT_LOCKED
        _write_summary(summary_path, summary)
        return EXIT_LOCKED

    code = EXIT_OK
    core: ServoCore | None = None
    runner: ServoRunner | None = None
    old_signals: dict[int, Any] = {}
    gc_before = _gc_collections()
    hb = None
    bus = None
    try:
        try:
            if tick_log_path is not None:
                from so101_rl.logging.ring_writer import RingWriter

                tick_log = RingWriter(Path(tick_log_path), TICK_LOG_DTYPE, capacity=1 << 16)
            bus = bus_factory()
            hb = hb_factory()
            core = build_servo_core(cfg, calibration, bus, ipc, hb, clock, tick_log=tick_log, log=log)
        except Exception as exc:  # noqa: BLE001 - bad geometry (ValueError), a port that will not open, ...
            log(f"refusing to start: {exc!r}")
            summary["error"] = repr(exc)
            code = EXIT_STARTUP
            if bus is not None:
                try:
                    torque_off_retry(bus)  # never connected by us, but the bus object exists: leave it limp
                    bus.disconnect()
                except Exception:  # noqa: BLE001, S110 - best effort on the way out
                    pass
            return code
        runner = ServoRunner(core, clock, log=log)
        if install_signals:
            old_signals = _install_signals(runner)
        if on_core is not None:
            on_core(core, runner)
        try:
            core.start(should_abort=lambda: runner.stopping, arm_timeout_s=arm_timeout_s)
        except StartupError as exc:
            if runner.signal_received is not None:
                log(f"start-up interrupted by signal {runner.signal_received}")
            else:
                log(str(exc))
                summary["error"] = repr(exc)
                code = EXIT_STARTUP
        except Exception as exc:  # noqa: BLE001 - e.g. OSError from bus.connect(): still through shutdown()
            traceback.print_exc()
            summary["error"] = repr(exc)
            code = EXIT_STARTUP
        if code == EXIT_OK and not runner.stopping:
            log(f"running at {cfg.rates.servo_hz:g} Hz (rt_active={rt.active}, max_ticks={max_ticks})")
            gc.collect()
            if freeze_gc:
                gc.freeze()
                gc.disable()
            gc_before = _gc_collections()
            try:
                runner.loop(max_ticks)
            except Exception as exc:  # noqa: BLE001 - every exception still goes through shutdown()
                traceback.print_exc()
                summary["error"] = repr(exc)
                code = EXIT_EXCEPTION
    finally:
        gc.enable()
        if core is not None:
            reason = TripCause.USER_STOP if code == EXIT_OK else TripCause.NONE
            try:
                core.shutdown(reason)
            except Exception as exc:  # noqa: BLE001 - report, exit nonzero
                traceback.print_exc()
                summary["shutdown_error"] = repr(exc)
                code = code or EXIT_EXCEPTION
        _restore_signals(old_signals)
        if core is not None and runner is not None:
            summary.update(_core_summary(core, runner, gc_before))
        if tick_log is not None:
            tick_log.close()
            summary["tick_log_dropped"] = tick_log.dropped
        if hb is not None and hasattr(hb, "close"):
            try:
                hb.close()
            except Exception:  # noqa: BLE001, S110 - best effort at exit
                pass
        if own_ipc is not None:
            own_ipc.close()
        for lock in port_locks:
            lock.release()
        summary["exit_code"] = code
        _write_summary(summary_path, summary)
    return code


def _core_summary(core: ServoCore, runner: ServoRunner, gc_before: int) -> dict[str, Any]:
    thermal = core.thermal
    return {
        "ticks": core.tick_count,
        "loop_ticks": runner.ticks,
        "overruns": core.overruns,
        "resyncs": runner.resyncs,
        "tick_stats": runner.stats.summary(),
        "final_state": SmState(core.state).name,
        "fault_bits": int(core.sm.fault_bits),
        "fault_names": [c.name for c in TripCause if c and core.sm.fault_bits & c],
        "calib_mismatches": core.calib_mismatches,
        "reenergize_count": core.reenergize_count,
        "thermal_trips_monitor": thermal.trips_last_hour(),
        "max_temp_c": {n: int(t) for n, t in zip(JOINT_NAMES, thermal.max_seen_c, strict=True)},
        "gc_collections": _gc_collections() - gc_before,
        "signal": runner.signal_received,
        "last_error": repr(core.last_error) if core.last_error is not None else None,
    }


def _write_summary(path: Path | None, summary: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    tmp.replace(path)
