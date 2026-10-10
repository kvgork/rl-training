"""Servo process runner and CLI (plan Task 3.2). Subprocess tests use the fake backend only; the watchdog MCU is a
``FakeMcu`` behind a pty pair (never a real serial device)."""

from __future__ import annotations

import json
import os
import pty
import resource
import select
import signal
import subprocess
import sys
import threading
import time
import tty
import uuid
from pathlib import Path

import numpy as np
import pytest

from so101_rl.safety.types import SmState
from so101_rl.servo import process
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.heartbeat import REASON_TIMEOUT, FakeHeartbeatLink, FakeMcu
from so101_rl.servo.shm import LocalIpc, ServoAlreadyRunning, ServoShm, lock_path, new_state
from so101_rl.servo.timing import FakeClock, RealClock

ROOT = Path(__file__).parent.parent
CALIB = ROOT / "tests" / "data" / "so101_follower.calib.json"
NS = 1_000_000_000


def _servo_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "so101_rl.servo.main", "--calibration", str(CALIB), *args]


def _name() -> str:
    return f"so101test_{uuid.uuid4().hex[:10]}"


def _rt_available() -> bool:
    return os.geteuid() == 0 or resource.getrlimit(resource.RLIMIT_RTPRIO)[0] > 0


# ---------------------------------------------------------------------------------------------------------------
# subprocess: --backend fake --max-ticks 300


def test_fake_backend_300_ticks_exits_0_and_publishes_state(tmp_path):
    name = _name()
    summary = tmp_path / "summary.json"
    t0 = time.monotonic()
    proc = subprocess.Popen(_servo_cmd("--backend", "fake", "--max-ticks", "300", "--shm-name", name, "--summary", str(summary)))
    seen_ticks, seen_states = [], set()
    shm = None
    st = new_state()
    try:
        while proc.poll() is None and time.monotonic() - t0 < 30:
            if shm is None:
                try:
                    shm = ServoShm.attach(name)
                except (FileNotFoundError, RuntimeError):
                    time.sleep(0.02)
                    continue
            if shm.read_state(st):
                seen_ticks.append(int(st["tick"][0]))
                seen_states.add(SmState(int(st["sm_state"][0])))
            time.sleep(0.005)
        code = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        if shm is not None:
            shm.close()
    elapsed = time.monotonic() - t0
    assert code == 0
    assert elapsed < 15.0, f"took {elapsed:.1f} s"
    assert seen_ticks and max(seen_ticks) >= 290 and seen_ticks == sorted(seen_ticks)
    assert SmState.RUN in seen_states
    rep = json.loads(summary.read_text())
    assert rep["ticks"] == 300 and rep["loop_ticks"] == 300
    assert rep["overruns"] <= 15  # "small": < 5 % of the ticks on a loaded, non-RT host
    assert rep["exit_code"] == 0 and rep["final_state"] in ("TORQUE_OFF", "SAFE_STOPPING")
    with pytest.raises(FileNotFoundError):
        ServoShm.attach(name)  # unlinked on the way out


def test_require_rt_without_privileges_exits_3_with_a_clear_message(tmp_path):
    if _rt_available():
        pytest.skip("this host grants real-time scheduling")
    proc = subprocess.run(
        _servo_cmd("--backend", "fake", "--max-ticks", "5", "--shm-name", _name()),
        env={**os.environ, "REQUIRE_RT": "1"},
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 3
    assert "REQUIRE_RT" in proc.stderr and "SCHED_FIFO" in proc.stderr


def test_feetech_backend_refuses_without_watchdog(monkeypatch, capsys):
    """In-process and hermetic: FeetechBus is replaced by a tripwire and the config's heartbeat port is forced
    to None, so this can never reach a real serial device (hardware rule: sim/fake only)."""
    import dataclasses

    from so101_rl.servo import feetech_bus, main

    def tripwire(*_a, **_k):
        raise AssertionError("FeetechBus must not be constructed")

    monkeypatch.setattr(feetech_bus, "FeetechBus", tripwire)
    real_load = main.load_config
    monkeypatch.setattr(main, "load_config", lambda *a, **k: dataclasses.replace(real_load(*a, **k), heartbeat_port=None))
    monkeypatch.setattr(main, "run", lambda *a, **k: pytest.fail("must refuse before run()"))
    code = main.main(["--backend", "feetech", "--calibration", str(CALIB), "--shm-name", _name()])
    assert code == 2 and "watchdog MCU" in capsys.readouterr().err


def test_feetech_backend_refuses_the_cuda_torch_environment(monkeypatch, capsys):
    """lerobot imports torch; a CUDA build cannot be mapped after mlockall: refuse before any port is opened."""
    from so101_rl.servo import feetech_bus, main

    monkeypatch.setattr(feetech_bus, "FeetechBus", lambda *_a, **_k: pytest.fail("FeetechBus must not be constructed"))
    monkeypatch.setattr(main, "run", lambda *a, **k: pytest.fail("must refuse before run()"))
    monkeypatch.setattr(main, "_cuda_torch_installed", lambda: True)
    code = main.main(["--backend", "feetech", "--no-mcu", "--calibration", str(CALIB), "--shm-name", _name()])
    assert code == 2 and "pixi run -e robot servo" in capsys.readouterr().err


def test_cuda_torch_detection_matches_the_installed_build():
    import torch

    from so101_rl.servo import main

    assert main._cuda_torch_installed() == (torch.version.cuda is not None)


@pytest.mark.slow
def test_mujoco_backend_headless_runs_and_exits_0(tmp_path):
    summary = tmp_path / "summary.json"
    proc = subprocess.run(
        _servo_cmd("--backend", "mujoco", "--max-ticks", "30", "--shm-name", _name(), "--summary", str(summary)),
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rep = json.loads(summary.read_text())
    assert rep["reenergize_count"] == 1 and rep["ticks"] == 30


# ---------------------------------------------------------------------------------------------------------------
# kill -9: the deadman (FakeMcu behind a pty) latches within 1.5 s


class PtyMcu(threading.Thread):
    """FakeMcu on the master side of a pty; the servo opens the slave as its heartbeat port."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.master, self._slave = pty.openpty()
        tty.setraw(self._slave)
        self.port = os.ttyname(self._slave)
        self.mcu = FakeMcu(RealClock())
        self.lock = threading.Lock()
        self.armed_at: float | None = None
        self.latched_at: float | None = None
        self.heartbeats = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            r, _, _ = select.select([self.master], [], [], 0.001)
            with self.lock:
                if r:
                    try:
                        data = os.read(self.master, 256)
                    except OSError:
                        data = b""
                    if data:
                        self.heartbeats += data.count(0xA5)
                        reply = self.mcu.feed(data)
                        if reply:
                            os.write(self.master, reply)
                self.mcu.poll()
                now = time.monotonic()
                if self.mcu.armed and self.armed_at is None:
                    self.armed_at = now
                if self.armed_at is not None and self.mcu.latched and self.latched_at is None:
                    self.latched_at = now

    def close(self) -> None:
        self._halt.set()
        self.join(timeout=2)
        os.close(self.master)
        os.close(self._slave)


def test_kill_9_lets_the_watchdog_latch_within_1_5_s(tmp_path):
    hw = PtyMcu()
    hw.mcu.reset_button(500)  # the operator presses the physical button (the MCU boots LATCHED)
    assert not hw.mcu.latched
    hw.start()
    name = _name()
    proc = subprocess.Popen(_servo_cmd("--backend", "fake", "--heartbeat-port", hw.port, "--shm-name", name))
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (hw.armed_at is None or time.monotonic() - hw.armed_at < 2.0):
            assert proc.poll() is None, "servo exited early"
            time.sleep(0.02)
        assert hw.armed_at is not None and hw.latched_at is None and hw.heartbeats > 30
        t_kill = time.monotonic()
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        while hw.latched_at is None and time.monotonic() - t_kill < 3.0:
            time.sleep(0.005)
        assert hw.latched_at is not None, "the MCU never latched"
        assert hw.latched_at - t_kill <= 1.5
        assert hw.mcu.reason == REASON_TIMEOUT
        # single-instance lock: the dead owner's segment is replaced, not blocked
        ServoShm.create(name).close()
    finally:
        if proc.poll() is None:
            proc.kill()
        hw.close()


# ---------------------------------------------------------------------------------------------------------------
# single-instance lock (review carry-forward C1)


def test_second_servo_on_the_same_segment_is_refused():
    name = _name()
    with ServoShm.create(name):
        with pytest.raises(ServoAlreadyRunning):
            ServoShm.create(name)
        assert Path(lock_path(name)).exists()
    ServoShm.create(name).close()  # released on close


def test_second_servo_process_exits_4(cfg, calibration, tmp_path):
    import dataclasses

    name = _name()
    cfg2 = dataclasses.replace(cfg, shm_name=name)
    with ServoShm.create(name):
        code = process.run(
            cfg2,
            lambda: pytest.fail("bus must not be built"),
            lambda: None,
            require_rt=False,
            clock=FakeClock(),
            calibration=calibration,
            install_signals=False,
        )
    assert code == process.EXIT_LOCKED


# ---------------------------------------------------------------------------------------------------------------
# in-process runner


def _in_process(cfg, calibration, *, max_ticks, summary=None, **kw):
    clock = FakeClock(start_ns=1_000 * NS)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.home_ticks)
    mcu = FakeMcu(clock)
    mcu.set_button(True)
    code = process.run(
        cfg,
        lambda: bus,
        lambda: FakeHeartbeatLink(mcu),
        require_rt=False,
        max_ticks=max_ticks,
        clock=clock,
        ipc=LocalIpc(),
        calibration=calibration,
        summary_path=summary,
        arm_timeout_s=5.0,
        install_signals=False,
        freeze_gc=False,
        **kw,
    )
    return code, bus, mcu


def test_in_process_run_writes_summary_and_shuts_down(cfg, calibration, tmp_path):
    summary = tmp_path / "s.json"
    code, bus, _ = _in_process(cfg, calibration, max_ticks=50, summary=summary)
    rep = json.loads(summary.read_text())
    assert code == 0 and rep["ticks"] == 50 and rep["exit_code"] == 0
    assert rep["rt_active"] is False and "kernel" in rep and "tick_stats" in rep
    assert not bus.torque_enabled().any() and not bus.connected


def test_bad_geometry_refuses_to_start_with_exit_2(cfg, calibration):
    import dataclasses

    bad = dataclasses.replace(
        cfg, safe_state=dataclasses.replace(cfg.safe_state, home_ticks=np.full(6, 2047))  # tip x 0.39 m: outside the box
    )
    code, bus, _ = _in_process(bad, calibration, max_ticks=5)
    assert code == process.EXIT_STARTUP
    assert not bus.torque_enabled().any()


def test_exception_in_the_loop_still_shuts_down(cfg, calibration):
    clock = FakeClock(start_ns=1_000 * NS)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.home_ticks)
    calls = {"n": 0}
    real = bus.read_block

    def boom(out):
        calls["n"] += 1
        if calls["n"] == 400:
            raise RuntimeError("injected loop fault")
        return real(out)

    bus.read_block = boom
    code = process.run(
        cfg,
        lambda: bus,
        lambda: None,
        require_rt=False,
        max_ticks=1000,
        clock=clock,
        ipc=LocalIpc(),
        calibration=calibration,
        install_signals=False,
        freeze_gc=False,
    )
    assert code == process.EXIT_EXCEPTION
    assert not bus.torque_enabled().any() and not bus.connected


class _SlowCore:
    """Stub core: every 10th tick takes 3.5 periods."""

    def __init__(self, clock: FakeClock, period: int) -> None:
        from so101_rl.servo.core import TickResult

        self.clock, self.period_ns, self.tick_count, self.starts = clock, period, 0, []
        self._res = TickResult()

    def tick(self, deadline_ns=None):
        self.starts.append(self.clock.monotonic_ns())
        self.tick_count += 1
        if self.tick_count % 10 == 0:
            self.clock.advance(int(3.5 * self.period_ns))
            self._res.overrun_ns = int(2.5 * self.period_ns)
        else:
            self._res.overrun_ns = 0
        return self._res


def test_deadline_loop_resyncs_after_a_large_overrun_instead_of_bursting():
    clock = FakeClock()
    period = 33_333_333
    core = _SlowCore(clock, period)
    logs = []
    runner = process.ServoRunner(core, clock, log=logs.append)
    runner.loop(max_ticks=40)
    starts = np.array(core.starts)
    gaps = np.diff(starts)
    assert runner.resyncs == 4
    assert (gaps >= period - 1).all()  # never two ticks back to back (no burst)
    assert ((starts - starts[0]) % period == 0).all()  # still on the absolute grid
    assert runner.stats.overruns == 4 and logs  # overruns reported (rate-limited)


def test_setup_realtime_never_raises_and_reports():
    st = process.setup_realtime()
    assert isinstance(st.active, bool) and isinstance(st.errors, list)
    if not _rt_available():
        assert not st.fifo and any("SCHED_FIFO" in e for e in st.errors)
    os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))  # undo the pinning for the rest of the session


# ---------------------------------------------------------------------------------------------------------------
# U3 review items 6, 7 and 14


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT], ids=["SIGTERM", "SIGHUP", "SIGQUIT"])
def test_item6_stop_signals_go_through_the_bounded_shutdown(cfg, calibration, sig):
    """SIGHUP (ssh session closed) and SIGQUIT must ramp the arm down like SIGINT, not kill the process."""
    clock = FakeClock(start_ns=1_000 * NS)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.home_ticks)
    mcu = FakeMcu(clock)
    mcu.set_button(True)
    n = {"reads": 0, "t_sig": None}
    real_read = bus.read_block
    seen = {}

    def read(out):
        n["reads"] += 1
        if n["t_sig"] is None and "core" in seen and seen["core"].state is SmState.RUN:  # first loop tick
            n["t_sig"] = clock.monotonic_ns()
            os.kill(os.getpid(), sig)
        return real_read(out)

    bus.read_block = read
    before = signal.getsignal(sig)
    code = process.run(
        cfg,
        lambda: bus,
        lambda: FakeHeartbeatLink(mcu),
        require_rt=False,
        clock=clock,
        ipc=LocalIpc(),
        calibration=calibration,
        arm_timeout_s=5.0,
        freeze_gc=False,
        max_ticks=600,  # a test that never signals must fail, not hang
        on_core=lambda core, runner: seen.update(core=core, runner=runner),
    )
    assert code == process.EXIT_OK
    assert n["t_sig"] is not None and seen["runner"].signal_received == sig
    assert clock.monotonic_ns() - n["t_sig"] <= cfg.safe_state.bound_s * NS
    assert not bus.torque_enabled().any() and not bus.connected
    np.testing.assert_allclose(bus._pos[:5], np.asarray(cfg.safe_state.rest_ticks)[:5], atol=30)  # ramped down
    assert signal.getsignal(sig) == before  # restored
    assert set(process.STOP_SIGNALS) >= {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT}


def _boot_ramp_signal_run(cfg, calibration, *, second_after_s=None):
    """In-process ``run`` from rest (a ~5 s boot ramp); SIGINT 1 s into the ramp, optionally a second SIGINT
    ``second_after_s`` later. Returns (exit code, times, bus, mcu, runner, heartbeat times, torque-off times)."""
    clock = FakeClock(start_ns=1_000 * NS)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.rest_ticks)  # long ramp
    mcu = FakeMcu(clock)
    mcu.set_button(True)
    t = {"on": None, "sig": None, "sig2": None}
    hb, offs, seen = [], [], {}
    real_torque, real_read = bus.set_torque, bus.read_block

    def set_torque(enabled, ids=None):
        if enabled and t["on"] is None:
            t["on"] = clock.monotonic_ns()
        if not enabled:
            offs.append(clock.monotonic_ns())
        return real_torque(enabled, ids)

    def read(out):
        now = clock.monotonic_ns()
        if t["on"] is not None and t["sig"] is None and now - t["on"] >= 1 * NS:
            t["sig"] = now
            os.kill(os.getpid(), signal.SIGINT)  # Ctrl-C during the boot ramp
        elif second_after_s is not None and t["sig"] is not None and t["sig2"] is None:
            if now - t["sig"] >= second_after_s * NS:
                t["sig2"] = now
                os.kill(os.getpid(), signal.SIGINT)  # second Ctrl-C during the hand-over: escalate
        return real_read(out)

    bus.set_torque, bus.read_block = set_torque, read

    class Link(FakeHeartbeatLink):
        def send(self, seq):
            hb.append(clock.monotonic_ns())
            return super().send(seq)

    code = process.run(
        cfg,
        lambda: bus,
        lambda: Link(mcu),
        require_rt=False,
        clock=clock,
        ipc=LocalIpc(),
        calibration=calibration,
        arm_timeout_s=5.0,
        freeze_gc=False,
        on_core=lambda core, runner: seen.update(runner=runner),
    )
    return code, t, bus, mcu, seen["runner"], hb, offs


def test_item7_signal_during_the_boot_ramp_hands_over_to_the_safe_state_sequence(cfg, calibration):
    code, t, bus, mcu, runner, hb, offs = _boot_ramp_signal_run(cfg, calibration)
    assert code == process.EXIT_OK  # interrupted by a signal, not a start-up failure
    assert t["sig"] is not None and runner.stop_signals == 1 and not runner.immediate
    assert min(o for o in offs if o >= t["sig"]) - t["sig"] > 1 * NS  # no torque off in place: ramped back
    np.testing.assert_allclose(bus._pos[:5], np.asarray(cfg.safe_state.rest_ticks)[:5], atol=30)  # at rest
    assert not bus.torque_enabled().any()
    assert max(offs) - t["sig"] <= cfg.safe_state.bound_s * NS
    assert np.diff([h for h in hb if h >= t["on"]]).max() < 0.5 * NS and not mcu.latched  # heartbeat continuous


def test_item7_second_signal_during_the_handover_cuts_torque_within_one_tick(cfg, calibration):
    code, t, bus, _mcu, runner, _hb, offs = _boot_ramp_signal_run(cfg, calibration, second_after_s=0.5)
    assert code == process.EXIT_OK
    assert t["sig2"] is not None and runner.stop_signals == 2 and runner.immediate
    period = round(NS / cfg.rates.servo_hz)
    assert min(o for o in offs if o >= t["sig2"]) - t["sig2"] <= period
    assert not bus.torque_enabled().any()
    assert np.abs(bus._pos[:5] - np.asarray(cfg.safe_state.rest_ticks)[:5]).max() > 60  # cut, not ramped to rest


def _sigint_during_boot_ramp_subprocess(tmp_path, *, second_after_s=None):
    """``--backend fake --fake-start rest`` subprocess (FakeMcu in-process, real clock). SIGINT once the boot ramp
    has visibly left rest (state record: BOOT, present moving), optionally a second SIGINT later."""
    name = _name()
    summary = tmp_path / "summary.json"
    proc = subprocess.Popen(
        _servo_cmd("--backend", "fake", "--fake-start", "rest", "--shm-name", name, "--summary", str(summary)),
        stderr=subprocess.PIPE,
        text=True,
    )
    rest = np.asarray(json.loads("[2151, 828, 3071, 2854, 2203, 2010]"))
    shm, st, last = None, new_state(), None
    t_sig = t_sig2 = None
    try:
        t0 = time.monotonic()
        while proc.poll() is None and time.monotonic() - t0 < 40:
            if shm is None:
                try:
                    shm = ServoShm.attach(name)
                except (FileNotFoundError, RuntimeError):
                    time.sleep(0.02)
                    continue
            if shm.read_state(st):
                last = st.copy()
                moved = np.abs(st["present_ticks"][0][:5].astype(int) - rest[:5]).max()
                booting = SmState(int(st["sm_state"][0])) is SmState.BOOT and bool(st["read_ok"][0])
                if t_sig is None and booting and moved > 60:
                    proc.send_signal(signal.SIGINT)
                    t_sig = time.monotonic()
            due = t_sig is not None and t_sig2 is None and time.monotonic() - t_sig >= (second_after_s or 0.0)
            if second_after_s is not None and due:
                proc.send_signal(signal.SIGINT)
                t_sig2 = time.monotonic()
            time.sleep(0.005)
        code = proc.wait(timeout=40)
        t_exit = time.monotonic()
        err = proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
        if shm is not None:
            shm.close()
    return code, json.loads(summary.read_text()), err, last, rest, t_sig, t_sig2, t_exit


def test_item7_subprocess_sigint_during_boot_ramp_exits_via_safe_state(tmp_path):
    code, rep, err, last, rest, t_sig, _t2, t_exit = _sigint_during_boot_ramp_subprocess(tmp_path)
    assert t_sig is not None, err
    assert code == 0, err
    assert rep["signal"] == signal.SIGINT and rep["stop_signals"] == 1 and not rep["immediate_stop"]
    assert "ReenergizeAborted" in rep["last_error"] and "handing over to the safe-state sequence" in err
    assert "escalation" not in err
    np.testing.assert_allclose(last["present_ticks"][0][:5], rest[:5], atol=30)  # ramped back to rest
    assert t_exit - t_sig <= 20.0 + 2.0  # bound_s plus process exit


def test_item7_subprocess_second_sigint_cuts_torque_immediately(tmp_path):
    code, rep, err, last, rest, _t_sig, t_sig2, t_exit = _sigint_during_boot_ramp_subprocess(tmp_path, second_after_s=0.4)
    assert t_sig2 is not None, err
    assert code == 0, err
    assert rep["stop_signals"] == 2 and rep["immediate_stop"]
    assert "escalation" in err
    assert t_exit - t_sig2 < 1.5  # no ramp: torque off on the next tick, then the normal exit path
    assert np.abs(last["present_ticks"][0][:5].astype(int) - rest[:5]).max() > 30  # cut, not at rest


def test_item14_a_held_serial_port_refuses_a_second_servo_with_exit_4(cfg, calibration, tmp_path):
    from so101_rl.servo.shm import SerialPortLock

    port = tmp_path / "ttyFAKE_bus"
    port.write_text("")
    alias = tmp_path / "serial-by-id"
    alias.symlink_to(port)
    held = SerialPortLock(str(port))  # another servo (with a different shm name) owns the bus
    try:
        code = process.run(
            cfg,
            lambda: pytest.fail("bus must not be built"),
            lambda: None,
            require_rt=False,
            clock=FakeClock(),
            ipc=LocalIpc(),
            calibration=calibration,
            install_signals=False,
            serial_ports=(str(alias),),
        )
    finally:
        held.release()
    assert code == process.EXIT_LOCKED
    code, _bus, _ = _in_process(cfg, calibration, max_ticks=5, serial_ports=(str(alias),))
    assert code == process.EXIT_OK  # free again, and released by run() itself:
    SerialPortLock(str(port)).release()


def test_item14_cli_locks_the_feetech_bus_and_watchdog_ports(monkeypatch):
    from so101_rl.servo import main as main_mod

    seen = {}
    monkeypatch.setattr(main_mod, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr(main_mod, "_cuda_torch_installed", lambda: False)
    assert main_mod.main(["--backend", "feetech", "--calibration", str(CALIB), "--port", "/dev/fake-bus",
                          "--heartbeat-port", "/dev/fake-mcu"]) == 0
    assert seen["serial_ports"] == ("/dev/fake-bus", "/dev/fake-mcu")
    assert main_mod.main(["--backend", "fake", "--calibration", str(CALIB)]) == 0
    assert seen["serial_ports"] == ()
