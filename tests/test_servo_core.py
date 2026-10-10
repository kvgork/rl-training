"""ServoCore.tick(), start-up and shutdown on FakeBus + FakeClock + FakeMcu + LocalIpc (plan Task 3.1).

Every binding obligation of the wave-1 safety layer (ARCHITECTURE section 5, plan Task 3.1 "Contract") has at
least one test here; the test name says which.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import signal
import tracemalloc

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES
from so101_rl.safety.state_machine import SafetyStateMachine, SmInputs, SmOutputs
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo import core as core_mod
from so101_rl.servo.core import StartupError, build_servo_core, log_row
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.heartbeat import REASON_TIMEOUT, FakeHeartbeatLink, FakeMcu, McuStatus
from so101_rl.servo.shm import CmdMode, LocalIpc, new_cmd, new_state
from so101_rl.servo.timing import FakeClock

NS = 1_000_000_000
MS = 1_000_000


class Harness:
    def __init__(self, cfg, calibration, *, start=None, mcu=True, fake_bus_kwargs=None, relay_powered=False, boot_ms=0.0):
        self.cfg = cfg
        self.clock = FakeClock(start_ns=1_000 * NS)
        self.start_ticks = np.asarray(cfg.safe_state.home_ticks if start is None else start, dtype=np.int32)
        self.mcu = FakeMcu(self.clock, timeout_ms=cfg.rates.heartbeat_timeout_ms) if mcu else None
        kw = dict(fake_bus_kwargs or {})
        if relay_powered:  # the servo side is live only while the MCU's relay is closed (the real wiring)
            kw.update(power=self.mcu.relay_closed, boot_ms=boot_ms)
        self.bus = FakeBus(self.clock, calibration=calibration, initial_ticks=self.start_ticks, **kw)
        self.ipc = LocalIpc()
        self.link = FakeHeartbeatLink(self.mcu) if mcu else None
        self.logs: list[str] = []
        self.core = build_servo_core(cfg, calibration, self.bus, self.ipc, self.link, self.clock, log=self.logs.append)
        self.period = self.core.period_ns
        self.cmd = new_cmd()
        self.state = new_state()
        self.records: list[np.ndarray] = []

    def boot(self):
        if self.mcu is not None:
            self.mcu.set_button(True)  # the operator holds the reset button (the MCU boots LATCHED)
        self.core.start(arm_timeout_s=5.0)
        if self.mcu is not None:
            self.mcu.set_button(False)
        return self

    def send(self, delta=None, mode=CmdMode.RUN, joint=None, value=None):
        d = np.zeros(6, dtype=np.float32) if delta is None else np.asarray(delta, dtype=np.float32)
        if joint is not None:
            d[joint] = value
        self.ipc.read_state(self.state)
        self.cmd["t_mono_ns"] = self.clock.monotonic_ns()
        self.cmd["state_seq"] = self.state["seq"][0]
        self.cmd["mode"] = int(mode)
        self.cmd["delta_ticks"][0] = d
        self.ipc.write_cmd(self.cmd)

    def tick(self, n=1, *, record=False):
        res = None
        for _ in range(n):
            res = self.core.tick()
            if record:
                self.ipc.read_state(self.state)
                self.records.append(self.state.copy())
            self.clock.advance(self.period)
        return res

    def run_policy(self, steps, delta=None, *, record=False):
        """``steps`` policy steps of 3 servo ticks each, a fresh command at the start of every step."""
        for _ in range(steps):
            self.send(delta)
            self.tick(3, record=record)

    def read_state(self):
        assert self.ipc.read_state(self.state)
        return self.state[0]


@pytest.fixture
def h(cfg, calibration):
    return Harness(cfg, calibration).boot()


def _stop_states_seen(h, n, delta=None):
    seen = []
    for _ in range(n):
        if delta is not None:
            h.send(delta)
        h.tick()
        seen.append(h.core.state)
    return seen


# ---------------------------------------------------------------------------------------------------------------
# start-up: boot latch, ARM, heartbeat immediately before reenergize


def test_boot_latch_is_handled_without_entering_latched(cfg, calibration):
    hn = Harness(cfg, calibration)
    assert hn.mcu.latched and hn.mcu.reason == 6  # MCU_RESET: the MCU boots latched
    seen_states = []
    real_idle = hn.core._boot_idle_tick

    def spy(st):
        seen_states.append((hn.core.state, st.latched))
        real_idle(st)

    hn.core._boot_idle_tick = spy
    hn.boot()
    assert any(latched for _, latched in seen_states), "start() must wait while the MCU is latched"
    assert all(s is SmState.BOOT for s, _ in seen_states)  # the latch was never fed to the state machine
    assert hn.core.state is SmState.RUN
    assert hn.mcu.armed and not hn.mcu.latched
    assert not hn.core.sm.fault_bits & (TripCause.DEADMAN | TripCause.HEARTBEAT_FAIL)


def test_feeding_the_boot_latch_would_be_absorbing(cfg, fake_clock):
    """Why the core must not pass McuStatus.latched before ARM: LATCHED never leaves (control experiment)."""
    sm = SafetyStateMachine(cfg, fake_clock)
    out = SmOutputs()
    sm.step(SmInputs(mcu=McuStatus(False, True, False, False, 6)), out)
    assert out.state is SmState.LATCHED
    sm.mark_energized()
    sm.step(SmInputs(), out)
    assert out.state is SmState.LATCHED


def test_start_times_out_without_button_and_never_energises(cfg, calibration):
    hn = Harness(cfg, calibration)
    with pytest.raises(StartupError, match="did not arm"):
        hn.core.start(arm_timeout_s=1.0)
    assert not hn.bus.torque_enabled().any()
    assert hn.core.state is SmState.BOOT


def test_heartbeat_sent_immediately_before_every_reenergize(cfg, calibration, monkeypatch):
    entries = []
    real = core_mod.reenergize

    def spy(bus, cfg_, calib, clock, block, **kw):
        entries.append((clock.monotonic_ns(), kw["tick_hook"], kw["fk"]))
        return real(bus, cfg_, calib, clock, block, **kw)

    monkeypatch.setattr(core_mod, "reenergize", spy)
    hn = Harness(cfg, calibration)
    seqs = []
    real_send = hn.link.send

    def send_spy(seq):
        seqs.append(hn.clock.monotonic_ns())
        return real_send(seq)

    hn.link.send = send_spy
    hn.boot()
    t_entry, hook, fk = entries[0]
    assert seqs and seqs[0] <= t_entry and t_entry - max(t for t in seqs if t <= t_entry) < hn.period
    assert hook is not None and fk is hn.core._fk  # reenergize(..., tick_hook=hb, fk=fk)
    assert hn.mcu.armed and not hn.mcu.latched


def test_hook_heartbeats_keep_mcu_armed_through_reenergize_from_rest(cfg, calibration):
    hn = Harness(cfg, calibration, start=cfg.safe_state.rest_ticks)  # long ramp: rest -> home (several seconds)
    t0 = hn.clock.monotonic_ns()
    hn.boot()
    assert hn.clock.monotonic_ns() - t0 > 2 * NS  # the ramp took longer than the MCU timeout
    assert hn.mcu.armed and not hn.mcu.latched
    assert hn.core.state is SmState.RUN


def test_heartbeat_refused_during_reenergize_aborts_with_torque_off(cfg, calibration):
    hn = Harness(cfg, calibration, start=cfg.safe_state.rest_ticks)
    calls = {"n": 0}
    real = hn.link.send

    def flaky(seq):
        calls["n"] += 1
        return real(seq) if calls["n"] < 20 else False

    hn.link.send = flaky
    hn.mcu.set_button(True)
    with pytest.raises(StartupError):
        hn.core.start(arm_timeout_s=5.0)
    assert isinstance(hn.core.last_error, core_mod.HeartbeatLost)
    assert not hn.bus.torque_enabled().any()


# ---------------------------------------------------------------------------------------------------------------
# commands


def test_command_is_applied_exactly_once_across_three_ticks(h):
    h.tick(2)
    base = h.bus.written_goals[-1].copy()
    h.send(joint=0, value=45)
    h.tick(9)  # the same seq is read 9 times
    goals = np.array(h.bus.written_goals[-9:])
    assert goals[2, 0] - base[0] == 45  # lands on the goal after interp_ticks
    assert (goals[2:, 0] == goals[2, 0]).all()  # and never integrates the delta again
    assert h.core.state is SmState.HOLD or h.core.state is SmState.RUN


def test_late_policy_holds_target_without_extrapolation_then_fresh_seq_resumes(h):
    h.run_policy(3, delta=[20, 0, 0, 0, 0, 0])
    states = _stop_states_seen(h, 7)  # 233 ms without a command
    assert SmState.HOLD in states and SmState.SAFE_STOPPING not in states
    i = states.index(SmState.HOLD)
    held = np.array(h.bus.written_goals[-(7 - i) :])
    assert (held == held[0]).all()  # held, not extrapolated
    h.send(joint=0, value=10)
    h.tick()
    assert h.core.state is SmState.RUN


def test_n_misses_safe_stop_then_torque_off(h):
    states = _stop_states_seen(h, 400)
    assert states.index(SmState.HOLD) < states.index(SmState.SAFE_STOPPING) < states.index(SmState.TORQUE_OFF)
    assert h.core.sm.fault_bits & TripCause.STALE_CMD
    assert not h.bus.torque_enabled().any()
    # the safe sequence went to rest before the torque off
    np.testing.assert_allclose(h.bus._pos[:5], np.asarray(h.cfg.safe_state.rest_ticks)[:5], atol=30)


def test_stop_req_command_is_a_user_stop(h):
    h.send(mode=CmdMode.STOP_REQ)
    h.tick()
    assert h.core.state is SmState.SAFE_STOPPING
    assert h.core.sm.fault_bits & TripCause.USER_STOP


# ---------------------------------------------------------------------------------------------------------------
# heartbeat only after a good read AND an accepted write


def test_read_failure_tick_writes_nothing_and_sends_no_heartbeat(h):
    h.run_policy(2)
    n_goals = len(h.bus.written_goals)
    last_hb = h.mcu._last_hb_ns
    h.bus.faults.fail_next_read(1)
    h.send(joint=1, value=10)
    res = h.tick()
    assert not res.ok and not res.wrote and not res.hb_sent
    assert len(h.bus.written_goals) == n_goals
    assert h.mcu._last_hb_ns == last_hb
    assert h.core.state is SmState.RUN  # a single failure only holds


def test_write_failure_tick_sends_no_heartbeat(h):
    h.run_policy(2)
    last_hb = h.mcu._last_hb_ns
    h.bus.faults.fail_next_write(1)
    h.send(joint=1, value=10)
    res = h.tick()
    assert res.ok and not res.wrote and not res.hb_sent
    assert h.mcu._last_hb_ns == last_hb


def test_three_consecutive_bus_failures_latch_and_stop_heartbeats(h):
    h.run_policy(2)
    h.bus.faults.fail_next_read(3)
    h.tick(3)
    assert h.core.state is SmState.LATCHED
    assert h.core.sm.fault_bits & TripCause.BUS_FAIL
    assert not h.bus.torque_enabled().any()
    hb_before = h.mcu._last_hb_ns
    h.tick(40)  # reads are fine again, but LATCHED is absorbing: torque off, no heartbeat
    assert h.mcu._last_hb_ns == hb_before
    h.mcu.poll()
    assert h.mcu.latched and h.mcu.reason == REASON_TIMEOUT  # the deadman takes over


def test_mcu_latches_when_the_servo_stops_ticking(h):
    """SIGSTOP model: no tick() for 1.1 s -> the MCU opens the relay; the next heartbeat's echo reports the MCU
    latched -> LATCHED at once (DEADMAN, plan section 10 2026-10-02: an MCU latch never waits for the miss budget)."""
    h.run_policy(3)
    assert h.mcu.armed
    h.clock.advance(1100 * MS)
    h.mcu.poll()
    assert h.mcu.latched and h.mcu.reason == REASON_TIMEOUT
    h.send()
    h.tick(2)
    assert h.core.state is SmState.LATCHED
    assert h.core.sm.fault_bits & TripCause.DEADMAN


def test_state_is_published_every_tick_also_when_latched(h):
    h.bus.faults.fail_next_read(3)
    h.tick(3)
    assert h.core.state is SmState.LATCHED
    seqs = []
    for _ in range(5):
        h.tick()
        seqs.append(int(h.read_state()["seq"]))
    assert seqs == sorted(set(seqs)) and len(seqs) == 5
    assert int(h.read_state()["sm_state"]) == SmState.LATCHED


# ---------------------------------------------------------------------------------------------------------------
# thermal: rest -> torque off -> dwell (heartbeat continues) -> reenergize -> RUN


def _thermal_trip_to_torque_off(h, joint="elbow_flex"):
    h.run_policy(3)
    h.bus.set_temperature(joint, 61.0)
    for _ in range(600):
        h.tick()
        if h.core.state is SmState.TORQUE_OFF:
            break
    assert h.core.state is SmState.TORQUE_OFF
    assert h.core.sm.fault_bits & TripCause.THERMAL


def test_thermal_trip_goes_to_rest_then_torque_off(h):
    _thermal_trip_to_torque_off(h)
    assert not h.bus.torque_enabled().any()
    np.testing.assert_allclose(h.bus._pos[:5], np.asarray(h.cfg.safe_state.rest_ticks)[:5], atol=30)


def test_torque_off_dwell_keeps_torque_off_and_heartbeating(h):
    """The dwell re-sends torque off every tick (acked: the heartbeat's write) and never writes a goal, which on the
    STS3215 would switch torque back on (found on the arm 2026-10-10)."""
    _thermal_trip_to_torque_off(h)
    h.bus.set_temperature("elbow_flex", 55.0)  # still above resume_c: the dwell has not started
    n_goals = len(h.bus.written_goals)
    for _ in range(60):  # 2 s, twice the MCU timeout
        res = h.tick()
        assert res.wrote and res.hb_sent
        assert not h.bus.torque_enabled().any()
    assert len(h.bus.written_goals) == n_goals
    assert h.mcu.armed and not h.mcu.latched
    assert h.core.state is SmState.TORQUE_OFF


def test_monitor_only_runs_every_tick_outside_run(h, monkeypatch):
    calls = []
    real = h.core.pipeline.monitor_only
    monkeypatch.setattr(h.core.pipeline, "monitor_only", lambda b, v: (calls.append(h.core.state), real(b, v))[1])
    _thermal_trip_to_torque_off(h)
    n0 = len(calls)
    h.tick(30)
    assert len(calls) - n0 == 30
    assert {SmState.SAFE_STOPPING, SmState.TORQUE_OFF} <= set(calls)


def test_thermal_resume_full_cycle_and_stale_misses_do_not_stop_after_resume(h, monkeypatch):
    entries = []
    real = core_mod.reenergize
    monkeypatch.setattr(
        core_mod,
        "reenergize",
        lambda *a, **kw: (entries.append((h.clock.monotonic_ns(), h.core.last_hb_ns)), real(*a, **kw))[1],
    )
    _thermal_trip_to_torque_off(h)
    for n in JOINT_NAMES:
        h.bus.set_temperature(n, 45.0)  # cooled below resume_c; the 30 s dwell starts
    # No policy commands during the whole dwell: the miss count grows to ~300 windows.
    states = []
    for _ in range(35 * 30):
        h.tick()
        states.append(h.core.state)
        if h.core.state is SmState.RUN:
            break
    assert h.core.state is SmState.RUN, states[-5:]
    assert len(entries) == 1  # the full reenergize ran once
    t_entry, t_hb = entries[0]
    assert 0 <= t_entry - t_hb < h.period  # a heartbeat immediately before (same tick)
    assert h.mcu.armed and not h.mcu.latched  # heartbeat never stopped
    assert h.core.consecutive_misses() == 0  # reset after mark_energized
    # the policy is back within one policy step: the arm keeps running, no STALE_CMD stop
    for _ in range(20):
        h.send(joint=0, value=5)
        h.tick(3)
        assert h.core.state is SmState.RUN
    assert not h.core.sm.fault_bits & TripCause.STALE_CMD
    # resync: the first RUN target started from present, not from the pre-trip target
    assert abs(int(h.bus.written_goals[-1][1]) - int(h.core.block.position[1])) <= h.cfg.action.tracking_error_ticks[1]


def test_failed_thermal_reenergize_notifies_and_needs_a_human(h, monkeypatch):
    _thermal_trip_to_torque_off(h)
    for n in JOINT_NAMES:
        h.bus.set_temperature(n, 45.0)

    def broken(*a, **kw):
        raise RuntimeError("injected re-energise failure")

    monkeypatch.setattr(core_mod, "reenergize", broken)
    for _ in range(35 * 30):
        h.tick()
    assert h.core.state is SmState.TORQUE_OFF
    assert h.core.reenergize_count == 2  # boot + the failed automatic one; never retried
    assert h.core.sm.resume_policy().name == "HUMAN_PHYSICAL_RESET"


# ---------------------------------------------------------------------------------------------------------------
# shutdown


def test_shutdown_runs_safe_state_to_rest_with_heartbeat_and_is_idempotent(h):
    h.run_policy(5, delta=[10, 0, 0, 0, 0, 0])
    t0 = h.clock.monotonic_ns()
    h.core.shutdown()
    assert h.clock.monotonic_ns() - t0 <= h.cfg.safe_state.bound_s * NS
    assert not h.bus.torque_enabled().any()
    np.testing.assert_allclose(h.bus._pos[:5], np.asarray(h.cfg.safe_state.rest_ticks)[:5], atol=30)
    assert h.mcu.armed and not h.mcu.latched  # heartbeats continued through the shutdown ramp
    assert not h.bus.connected
    n_events = len(h.bus.torque_events)
    h.core.shutdown()  # idempotent
    assert len(h.bus.torque_events) == n_events
    with pytest.raises(RuntimeError):
        h.core.tick()


def test_shutdown_uses_sequencer_run_with_tick_hook(h, monkeypatch):
    seen = {}
    real = h.core._safe.run

    def spy(bus, block, *, servo_hz, tick_hook=None):
        seen["hz"], seen["hook"] = servo_hz, tick_hook
        return real(bus, block, servo_hz=servo_hz, tick_hook=tick_hook)

    monkeypatch.setattr(h.core._safe, "run", spy)
    h.run_policy(2)
    h.core.shutdown()
    assert seen["hz"] == h.cfg.rates.servo_hz and seen["hook"] is not None
    # the sequencer was built with the required protections
    assert h.core._safe.torque_off_ids == (6, 5, 4, 3, 2, 1)  # calibration ids, torque_off_order


def test_sigint_sets_a_flag_and_shutdown_completes_within_bound(cfg, calibration):
    from so101_rl.servo import process

    clock = FakeClock(start_ns=1_000 * NS)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=cfg.safe_state.home_ticks)
    mcu = FakeMcu(clock)
    mcu.set_button(True)
    ipc = LocalIpc()
    ticks = {"n": 0, "t_sig": None}
    real_read = bus.read_block

    def read_and_maybe_signal(out):
        ticks["n"] += 1
        if ticks["n"] == 60:
            ticks["t_sig"] = clock.monotonic_ns()
            os.kill(os.getpid(), signal.SIGINT)  # the handler only sets the runner's flag
        return real_read(out)

    bus.read_block = read_and_maybe_signal
    seen = {}

    def on_core(core, runner):
        seen["core"], seen["runner"] = core, runner
        cmd = new_cmd()
        cmd["mode"] = int(CmdMode.RUN)
        ipc.write_cmd(cmd)

    code = process.run(
        cfg,
        lambda: bus,
        lambda: FakeHeartbeatLink(mcu),
        require_rt=False,
        clock=clock,
        ipc=ipc,
        calibration=calibration,
        arm_timeout_s=5.0,
        freeze_gc=False,
        on_core=on_core,
    )
    assert code == 0
    assert seen["runner"].signal_received == signal.SIGINT
    assert clock.monotonic_ns() - ticks["t_sig"] <= cfg.safe_state.bound_s * NS
    assert not bus.torque_enabled().any()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored


# ---------------------------------------------------------------------------------------------------------------
# log convention, allocation, wiring


def test_log_convention_present_is_read_after_the_step_target_was_applied(h):
    h.tick(2)
    h.records.clear()
    steps = 6
    t_writes = []
    for i in range(steps):
        h.send(joint=0, value=45 if i % 2 == 0 else -10)
        for _ in range(3):
            h.core.tick()
            h.ipc.read_state(h.state)
            h.records.append(h.state.copy())
            t_writes.append(h.bus.written_goal_t_ns[-1])
            h.clock.advance(h.period)
    for i in range(steps - 1):
        last, first_next = h.records[3 * i + 2], h.records[3 * i + 3]
        row = log_row(last, first_next)
        # applied post-clamp delta of command i = goal change of that step
        assert row["applied_delta_ticks"][0] == (45 if i % 2 == 0 else -10)
        np.testing.assert_array_equal(row["target_ticks"], h.bus.written_goals[-(3 * steps) + 3 * i + 2])
        # present of the next step's first record was read after the step's last write ...
        assert int(first_next["t_mono_ns"][0]) > t_writes[3 * i + 2]
        # ... and one full policy step after the step's first write
        assert int(first_next["t_mono_ns"][0]) - t_writes[3 * i] >= 3 * h.period - 1


def test_logged_action_is_the_applied_post_clamp_delta(h):
    h.tick(2)
    h.send(joint=0, value=1000.0)  # far beyond max_delta_ticks
    h.tick(3)
    st = h.read_state()
    assert float(st["applied_delta_ticks"][0]) == float(h.cfg.action.max_delta_ticks[0])


_SCALAR_SHUFFLE_BYTES = 32  # an int <-> float re-attribution through the free lists: 24..32 B, never per tick


@pytest.mark.parametrize("leak", [False, True], ids=["core", "control-leak-detected"])
def test_zero_net_allocation_over_1000_ticks_after_warmup(cfg, calibration, leak):
    hn = Harness(cfg, calibration, fake_bus_kwargs={"record_writes": False}).boot()
    leak_bytes = 1024
    if leak:  # control: a deliberate per-tick leak must be caught by exactly this measurement
        kept = []
        real_hb = hn.core._send_heartbeat
        hn.core._send_heartbeat = lambda: (kept.append(bytearray(leak_bytes)), real_hb())[1]
    hn.run_policy(40, delta=[3, -2, 2, 1, 0, 1])  # warm-up: every code path of a RUN tick has run
    pkg = os.path.dirname(core_mod.__file__).rsplit(os.sep, 1)[0]
    filters = [
        tracemalloc.Filter(True, os.path.join(pkg, "servo", "core.py")),
        tracemalloc.Filter(True, os.path.join(pkg, "safety", "*")),
        tracemalloc.Filter(True, os.path.join(pkg, "servo", "shm.py")),
        tracemalloc.Filter(True, __file__ if leak else os.path.join(pkg, "servo", "core.py")),
    ]
    def run(n):
        for k in range(n):
            if k % 3 == 0:
                hn.send([3 if k % 2 else -3, 2, -2, 1, 0, 1])
            hn.core.tick()
            hn.clock.advance(hn.period)

    # U3 verification N5 (flaky under load / test order): tracemalloc attributes a block to the line that last
    # malloc'ed it, and CPython recycles small scalars through free lists that tracemalloc does not see. Whether a
    # long-lived scalar is re-malloc'ed inside the window therefore depends on free-list state left by earlier
    # tests and GC passes. Observed: an int freed at core.py (``age``: -1 object, -28 B) and a float malloc'ed at
    # thermal.py (``hottest_c``: +1 object, +24 B), net 0 objects, and the old "sum of the growing lines == 0"
    # failed on it. The core is now judged by NET object count (exact: <= 0) and net bytes (<= a scalar-size
    # mismatch, 32 B, far below any per-tick growth over 1000 ticks). Determinism: one full collection before
    # tracing, no cyclic-GC pass inside the traced region.
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        run(600)  # traced warm-up (seq numbers past the small-int cache): the long-lived scalars (seq, timestamps) are now traced objects too
        before = tracemalloc.take_snapshot().filter_traces(filters)
        run(1000)
        after = tracemalloc.take_snapshot().filter_traces(filters)
    finally:
        tracemalloc.stop()
        if gc_was_enabled:
            gc.enable()
    assert hn.core.state is SmState.RUN
    diff = after.compare_to(before, "lineno")
    growth = [s for s in diff if s.size_diff > 0]
    if leak:
        # the control leaks 1000 heartbeats x 1 KiB on one line of this file: at least 90 % of that must be seen
        here = sum(s.size_diff for s in growth if s.traceback[0].filename == __file__)
        assert here >= 0.9 * 1000 * leak_bytes, [(str(s.traceback), s.size_diff) for s in growth]
        return
    # the core allocates nothing net over 1000 ticks: a leaked object of any kind, once per tick or even once in
    # the window, raises the net object count; a container that grows without new objects raises the bytes
    detail = [(str(s.traceback), s.count_diff, s.size_diff) for s in diff if s.count_diff or s.size_diff]
    assert sum(s.count_diff for s in diff) <= 0, detail
    assert sum(s.size_diff for s in diff) <= _SCALAR_SHUFFLE_BYTES, detail


def test_tick_returns_the_same_preallocated_result(h):
    assert h.tick() is h.tick()


def test_build_servo_core_validates_the_safe_state_geometry(cfg, calibration):
    bad = dataclasses.replace(
        cfg, safe_state=dataclasses.replace(cfg.safe_state, home_ticks=np.full(6, 2047))  # tip x 0.39 m: outside the box
    )
    clock = FakeClock()
    with pytest.raises(ValueError, match="safe_state"):
        build_servo_core(bad, calibration, FakeBus(clock, calibration=calibration), LocalIpc(), None, clock)


def test_without_mcu_link_the_core_still_runs(cfg, calibration):
    hn = Harness(cfg, calibration, mcu=False).boot()
    hn.run_policy(3)
    assert hn.core.state is SmState.RUN and not hn.core.armed


def test_boot_fails_cleanly_when_no_good_tick_follows_arm(cfg, calibration):
    """No heartbeat can be sent immediately before reenergize (servos not answering): refuse, torque off."""
    hn = Harness(cfg, calibration)
    hn.bus.faults.fail_next_read(10_000)
    hn.mcu.set_button(True)
    with pytest.raises(StartupError, match="after ARM"):
        hn.core.start(arm_timeout_s=5.0)
    assert hn.core.reenergize_count == 0  # reenergize was never entered without the heartbeat before it
    assert not hn.bus.torque_enabled().any()


def test_shutdown_still_torques_off_and_disconnects_when_the_sequence_raises(h, monkeypatch):
    h.run_policy(2)

    def broken():
        raise RuntimeError("injected shutdown fault")

    monkeypatch.setattr(h.core, "_shutdown_sequence", broken)
    h.bus.set_torque(True)
    h.core.shutdown()
    assert not h.bus.torque_enabled().any() and not h.bus.connected
    assert "injected" in repr(h.core.last_error)


def test_safe_stop_tick_with_failed_write_sends_no_heartbeat(h):
    h.send(mode=CmdMode.STOP_REQ)
    h.tick()
    assert h.core.state is SmState.SAFE_STOPPING
    h.bus.faults.fail_next_write(1)
    res = h.tick()
    assert res.ok and not res.wrote and not res.hb_sent


def test_refused_writes_with_good_reads_latch_even_without_an_mcu(cfg, calibration):
    hn = Harness(cfg, calibration, mcu=False).boot()
    hn.run_policy(2)
    hn.bus.faults.fail_next_write(cfg.rates.max_bus_failures)
    for _ in range(cfg.rates.max_bus_failures):
        hn.send(joint=0, value=5)
        res = hn.tick()
        assert res.ok and not res.wrote and not res.hb_sent
    hn.tick()
    assert hn.core.state is SmState.LATCHED
    assert hn.core.sm.fault_bits & TripCause.BUS_FAIL
    assert not hn.bus.torque_enabled().any()


def test_hold_target_is_latched_on_entry(h):
    h.run_policy(3, delta=[20, 0, 0, 0, 0, 0])
    states = _stop_states_seen(h, 12)
    i = states.index(SmState.HOLD)
    held = np.array(h.bus.written_goals[-(12 - i) :])[: states[i:].count(SmState.HOLD)]
    assert len(held) >= 3 and (held == held[0]).all()
    np.testing.assert_array_equal(held[0], h.core.pipeline.prev_target)
