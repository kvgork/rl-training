"""Regression tests for the U3 review (``plans/reviews/2026-10-01-u3-review.md``), items 1-11.

Every test names the review item it covers. The item-4 tests are mutation-killing: each one fails when the safety
rule it names is removed from ``servo/core.py`` (verified by applying the mutation; see the commit message).
FakeBus + FakeClock + FakeMcu + LocalIpc only: nothing here touches a serial device.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_servo_core import MS, NS, Harness

from so101_rl.config import JOINT_NAMES
from so101_rl.safety.reenergize import ReenergizeAborted
from so101_rl.safety.state_machine import HB_MISS_BUDGET
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo import core as core_mod
from so101_rl.servo.core import HeartbeatLost, StartupError
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.heartbeat import REASON_ESTOP_PIN, REASON_SEQ_ERROR, HbSend
from so101_rl.servo.shm import CmdMode
from so101_rl.servo.timing import FakeClock


@pytest.fixture
def h(cfg, calibration):
    return Harness(cfg, calibration).boot()


def _to_hold(h):
    h.run_policy(3, delta=[20, 0, 0, 0, 0, 0])
    for _ in range(20):
        h.tick()
        if h.core.state is SmState.HOLD:
            return
    raise AssertionError(f"never reached HOLD: {h.core.state}")


def _to_torque_off_by_stop(h):
    h.run_policy(2)
    h.send(mode=CmdMode.STOP_REQ)
    for _ in range(1200):
        h.tick()
        if h.core.state is SmState.TORQUE_OFF:
            return
    raise AssertionError(h.core.state)


def _thermal_to_torque_off(h, joint="elbow_flex"):
    h.bus.set_temperature(joint, 61.0)
    for _ in range(1200):
        h.send()
        h.tick()
        if h.core.state is SmState.TORQUE_OFF:
            return
    raise AssertionError(h.core.state)


def _cool_until_run(h, max_ticks=35 * 30):
    for n in JOINT_NAMES:
        h.bus.set_temperature(n, 45.0)
    for _ in range(max_ticks):
        h.tick()
        if h.core.state is SmState.RUN:
            return
    raise AssertionError(h.core.state)


# ---------------------------------------------------------------------------------------------------------------
# item 1: real-arm start order (port without handshake -> button + ARM -> ping -> reenergize)


def test_item1_old_order_handshake_before_arm_fails_on_a_relay_powered_bus(cfg, calibration):
    hn = Harness(cfg, calibration, relay_powered=True)
    assert hn.mcu.latched and not hn.mcu.relay_on  # boot latch: the servo side is dead
    with pytest.raises(ConnectionError, match="handshake"):
        hn.bus.connect(handshake=True)  # what FeetechBus.connect() used to do before the button + ARM


def test_item1_new_order_succeeds_on_a_relay_powered_bus_and_never_pings_before_arm(cfg, calibration):
    hn = Harness(cfg, calibration, relay_powered=True, boot_ms=200.0)  # servos need 200 ms after the relay closes
    pings_at_arm = []
    real_arm = hn.link.arm
    hn.link.arm = lambda: (pings_at_arm.append(hn.bus.pings), real_arm())[1]
    connect_kwargs = []
    real_connect = hn.bus.connect
    hn.bus.connect = lambda **kw: (connect_kwargs.append(kw), real_connect(**kw))[1]
    hn.boot()
    assert connect_kwargs == [{}]  # the core opens the port without a handshake
    assert pings_at_arm and pings_at_arm[0] == 0  # no bus traffic while the servo side is unpowered
    assert hn.bus.pings >= 2  # pinged after ARM until the booting servos answered
    assert hn.core.state is SmState.RUN and hn.bus.torque_enabled().all()
    assert hn.mcu.armed and not hn.mcu.latched
    hn.run_policy(5, delta=[5, 0, 0, 0, 0, 0])
    assert hn.core.state is SmState.RUN


def test_item1_servos_that_never_answer_after_arm_fail_start_up_cleanly(cfg, calibration):
    hn = Harness(cfg, calibration, relay_powered=True)
    hn.bus.faults.drop_motor(3)
    hn.mcu.set_button(True)
    with pytest.raises(StartupError, match="answer a ping"):
        hn.core.start(arm_timeout_s=5.0)
    assert hn.core.reenergize_count == 0 and not hn.bus.torque_enabled().any()


def test_item1_relay_opening_cuts_servo_power_and_the_core_latches(cfg, calibration):
    hn = Harness(cfg, calibration, relay_powered=True).boot()
    hn.run_policy(3)
    hn.mcu.set_estop(True)  # relay opens: the servos lose power and torque
    hn.run_policy(2)
    assert hn.core.state is SmState.LATCHED
    assert not hn.bus.torque_enabled().any()


# ---------------------------------------------------------------------------------------------------------------
# item 2: tri-state heartbeat; the sequence advances only when the frame was delivered


def test_item2_undelivered_shutdown_heartbeat_no_longer_latches_seq_error(h):
    """The review's reproduction: fail_writes(1) during the shutdown ramp used to latch with reason 3."""
    h.run_policy(3)
    real_send = h.link.send
    n = {"sends": 0}

    def send(seq):
        n["sends"] += 1
        if n["sends"] == 5:
            h.link.fail_writes(1)  # a write timeout in the middle of the ramp
        return real_send(seq)

    h.link.send = send
    h.core.shutdown()
    h.mcu.poll()
    assert h.mcu.reason != REASON_SEQ_ERROR
    assert h.mcu.armed and not h.mcu.latched


def test_item2_undelivered_heartbeat_reuses_its_sequence_number_in_run(h):
    h.run_policy(2)
    seq_before = h.core._hb_seq
    h.link.fail_writes(1)
    h.send()
    h.tick()
    assert h.core._hb_seq == seq_before and h.core.hb_misses == 1
    h.tick()
    assert h.core._hb_seq == (seq_before + 1) & 0xFF and h.core.hb_misses == 0
    assert h.mcu.armed and not h.mcu.latched and h.core.state is SmState.RUN


def test_item2_lost_echo_still_advances_the_sequence(h):
    h.run_policy(2)
    seq_before = h.core._hb_seq
    h.link.drop_echoes(1)
    h.send()
    h.tick()
    assert h.core._hb_seq == (seq_before + 1) & 0xFF  # the MCU counted it: repeating it would latch
    h.tick(3)
    assert h.mcu.armed and not h.mcu.latched and h.core.state is SmState.RUN


# ---------------------------------------------------------------------------------------------------------------
# item 5: the decided miss budget (5 consecutive missed acks, or the MCU latched); reenergize stays strict


@pytest.mark.parametrize("fault", ["fail_writes", "drop_echoes"])
def test_item5_four_misses_then_an_ack_keep_running(h, fault):
    h.run_policy(2)
    getattr(h.link, fault)(HB_MISS_BUDGET - 1)
    for _ in range(HB_MISS_BUDGET - 1):
        h.send()
        h.tick()
        assert h.core.state is SmState.RUN
    assert h.core.hb_misses == HB_MISS_BUDGET - 1
    h.run_policy(3)
    assert h.core.hb_misses == 0 and h.core.state is SmState.RUN
    assert h.mcu.armed and not h.mcu.latched


@pytest.mark.parametrize("fault", ["fail_writes", "drop_echoes"])
def test_item5_five_consecutive_misses_latch(h, fault):
    h.run_policy(2)
    getattr(h.link, fault)(HB_MISS_BUDGET)
    for _ in range(HB_MISS_BUDGET):
        h.send()
        h.tick()
        assert h.core.state is SmState.RUN
    h.tick()
    assert h.core.state is SmState.LATCHED and h.core.sm.fault_bits & TripCause.HEARTBEAT_FAIL
    assert not h.bus.torque_enabled().any()


def test_item5_mcu_latched_latches_at_once(h):
    h.run_policy(2)
    h.mcu.set_estop(True)
    h.send()
    h.tick()  # this tick's heartbeat echo reports the latch
    h.tick()
    assert h.core.state is SmState.LATCHED and h.core.sm.fault_bits & TripCause.DEADMAN
    assert h.mcu.reason == REASON_ESTOP_PIN


def test_item5_one_miss_during_boot_reenergize_aborts_with_torque_off(cfg, calibration):
    hn = Harness(cfg, calibration, start=cfg.safe_state.rest_ticks)  # a long ramp: rest -> home
    real = hn.link.send
    n = {"k": 0}

    def send(seq):
        n["k"] += 1
        if n["k"] == 20:  # well inside the ramp, torque on
            hn.link.fail_writes(1)
        return real(seq)

    hn.link.send = send
    hn.mcu.set_button(True)
    with pytest.raises(StartupError):
        hn.core.start(arm_timeout_s=5.0)
    assert isinstance(hn.core.last_error, HeartbeatLost)
    assert not hn.bus.torque_enabled().any()


def test_item5_one_miss_during_thermal_resume_aborts_with_torque_off(h):
    _thermal_to_torque_off(h)
    for n in JOINT_NAMES:
        h.bus.set_temperature(n, 45.0)
    real = h.link.send
    state = {"armed": False}

    def send(seq):
        if h.core.state is SmState.CHECK and not state["armed"] and h.bus.torque_enabled().any():
            state["armed"] = True
            return HbSend.NOT_DELIVERED  # one miss inside the re-energise ramp
        return real(seq)

    h.link.send = send
    for _ in range(35 * 30):
        h.tick()
        if state["armed"] and h.core.state is not SmState.CHECK:
            break
    assert state["armed"]
    assert isinstance(h.core.last_error, HeartbeatLost)
    assert not h.bus.torque_enabled().any()
    h.tick()
    assert h.core.state is SmState.LATCHED  # strict path: the refused heartbeat latches


# ---------------------------------------------------------------------------------------------------------------
# item 4: mutation-killing tests (each fails when the named rule is removed from servo/core.py)


def test_item4_pre_reenergize_heartbeat_must_be_acked(cfg, calibration):
    """Kills: forcing the pre-reenergize heartbeat result to True (core ``_pre_reenergize_tick``)."""
    hn = Harness(cfg, calibration)
    hn.link.send = lambda seq: HbSend.NOT_DELIVERED  # the MCU never gets a heartbeat after ARM
    hn.mcu.set_button(True)
    with pytest.raises(StartupError, match="after ARM"):
        hn.core.start(arm_timeout_s=5.0)
    assert hn.core.reenergize_count == 0  # never entered without an ACKED heartbeat right before it
    assert not hn.bus.torque_enabled().any()


def test_item4_first_run_target_after_a_thermal_resume_is_resynced_to_present(h):
    """Kills: deleting ``pipe.resync`` on ``resync_target``. (At boot the pipeline resyncs itself because it was
    never synced, so the observable case is the thermal resume: the pre-trip target is far from home.)"""
    h.run_policy(6, delta=[0, 40, -40, 0, 0, 0])  # leave home: the pipeline's prev_target is now far from home
    _thermal_to_torque_off(h)
    _cool_until_run(h)  # reenergize ramped the arm back to home
    res = h.tick()  # the resync tick, no command
    assert res.wrote
    np.testing.assert_array_equal(h.bus.written_goals[-1], h.core.block.position)
    np.testing.assert_array_equal(h.core.pipeline.goal, h.core.block.position)


def test_item4_shutdown_ramp_longer_than_the_mcu_timeout_keeps_heartbeating(h):
    """Kills: deleting the heartbeat in ``_shutdown_hook``. The ramp (home -> rest) takes several seconds."""
    h.run_policy(3)
    latched = []
    real_read = h.bus.read_block

    def read(out):
        h.mcu.poll()  # the MCU loop runs concurrently with the ramp
        latched.append(h.mcu.latched)
        return real_read(out)

    h.bus.read_block = read
    t0 = h.clock.monotonic_ns()
    h.core.shutdown()
    assert h.clock.monotonic_ns() - t0 > 1.5 * NS
    h.mcu.poll()
    assert not any(latched) and h.mcu.armed and not h.mcu.latched


def test_item4_thermal_resume_waits_for_an_acked_heartbeat(h, monkeypatch):
    """Kills: changing the thermal-resume gate (``hb_sent and _last_hb_ok``) to ``if True``."""
    results = []
    real_send = h.link.send
    refused = {"done": False}

    def send(seq):
        if h.core.state is SmState.CHECK and not refused["done"] and not h.bus.torque_enabled().any():
            refused["done"] = True
            results.append(HbSend.NOT_DELIVERED)
            return HbSend.NOT_DELIVERED  # the heartbeat of the CHECK tick is refused
        r = real_send(seq)
        results.append(r)
        return r

    entries = []
    real = core_mod.reenergize
    monkeypatch.setattr(core_mod, "reenergize", lambda *a, **kw: (entries.append(results[-1]), real(*a, **kw))[1])
    _thermal_to_torque_off(h)
    h.link.send = send
    _cool_until_run(h)
    assert refused["done"]
    assert entries == [HbSend.ACKED]  # entered once, right after an ACKED heartbeat (one CHECK tick later)
    assert h.mcu.armed and not h.mcu.latched


def test_item4_run_write_is_range_checked_by_the_core(h, monkeypatch):
    """Kills: dropping the core's ``check_range`` before ``write_goal`` in RUN (FakeBus would get 4096)."""
    h.run_policy(2)
    real_apply = h.core.pipeline.apply

    def bad_apply(block, ticks_remaining, v):
        real_apply(block, ticks_remaining, v)
        v.target_ticks[0] = 4096  # a pipeline bug that does not report RANGE
        return v

    monkeypatch.setattr(h.core.pipeline, "apply", bad_apply)
    n = len(h.bus.written_goals)
    h.send()
    res = h.tick()
    assert not res.wrote and not res.hb_sent
    assert len(h.bus.written_goals) == n


def test_item4_hold_writes_nothing_on_a_failed_read(h):
    """Kills: dropping the ``ok`` guard of the HOLD write."""
    _to_hold(h)
    n = len(h.bus.written_goals)
    h.bus.faults.fail_next_read(1)
    res = h.tick()
    assert h.core.state is SmState.HOLD and not res.ok and not res.wrote
    assert len(h.bus.written_goals) == n


def test_item4_refused_torque_off_in_torque_off_sends_no_heartbeat(h):
    """Kills: dropping ``torque_ok`` from the TORQUE_OFF write condition."""
    _to_torque_off_by_stop(h)
    last_hb = h.mcu._last_hb_ns
    h.bus.faults.fail_next_write(1)  # consumed by the torque-off of this tick
    res = h.tick()
    assert res.ok and not res.wrote and not res.hb_sent
    assert h.mcu._last_hb_ns == last_hb


def test_item4_second_safe_stop_starts_a_fresh_sequence_to_rest(h):
    """Kills: dropping the ``_safe_started`` reset when a safe sequence is done (the second thermal trip would
    finish at once and torque off in place instead of going to rest)."""
    h.run_policy(3)
    _thermal_to_torque_off(h)
    _cool_until_run(h)
    h.run_policy(5, delta=[0, 30, -30, 0, 0, 0])  # leave home again
    starts = []
    real_start = h.core._safe.start
    h.core._safe.start = lambda present: (starts.append(present.copy()), real_start(present))[1]
    _thermal_to_torque_off(h, joint="wrist_flex")
    assert len(starts) == 1
    np.testing.assert_allclose(h.bus._pos[:5], np.asarray(h.cfg.safe_state.rest_ticks)[:5], atol=30)


# ---------------------------------------------------------------------------------------------------------------
# item 7: reenergize abort; no thermal resume after a stop


def test_item7_stop_during_boot_reenergize_aborts_within_a_tick(cfg, calibration):
    hn = Harness(cfg, calibration, start=cfg.safe_state.rest_ticks)
    t = {"abort_at": None}

    def should_abort():
        if t["abort_at"] is None and hn.bus.torque_enabled().any():
            t["abort_at"] = hn.clock.monotonic_ns() + 300 * MS  # 0.3 s into the ramp
        return t["abort_at"] is not None and hn.clock.monotonic_ns() >= t["abort_at"]

    hn.mcu.set_button(True)
    with pytest.raises(StartupError):
        hn.core.start(should_abort=should_abort, arm_timeout_s=5.0)
    assert isinstance(hn.core.last_error, ReenergizeAborted)
    assert hn.clock.monotonic_ns() - t["abort_at"] <= hn.period
    assert not hn.bus.torque_enabled().any()


def test_item7_no_thermal_resume_once_stop_is_set(cfg, calibration, monkeypatch):
    hn = Harness(cfg, calibration)
    stop = {"set": False}
    hn.mcu.set_button(True)
    hn.core.start(should_abort=lambda: stop["set"], arm_timeout_s=5.0)
    hn.mcu.set_button(False)
    hn.run_policy(3)
    _thermal_to_torque_off(hn)
    stop["set"] = True  # e.g. SIGINT arrived during the dwell
    calls = []
    monkeypatch.setattr(core_mod, "reenergize", lambda *a, **kw: calls.append(1))
    for n in JOINT_NAMES:
        hn.bus.set_temperature(n, 45.0)
    hn.tick(35 * 30)
    assert calls == [] and hn.core.reenergize_count == 1
    assert not hn.bus.torque_enabled().any()


# ---------------------------------------------------------------------------------------------------------------
# item 8: command handling


def test_item8a_delta_arriving_on_a_failed_read_tick_is_applied_later(h):
    h.run_policy(2)
    base = int(h.bus.written_goals[-1][0])
    h.bus.faults.fail_next_read(1)
    h.send(joint=0, value=30)
    h.tick()  # failed read: use_policy is false on this tick
    h.tick(3)
    assert int(h.bus.written_goals[-1][0]) - base == 30


def test_item8b_command_written_before_arm_is_not_fresh_after_start(cfg, calibration):
    hn = Harness(cfg, calibration)
    hn.send(joint=0, value=40)  # written while the core waits for the button
    hn.boot()
    hn.tick(3)
    np.testing.assert_array_equal(hn.bus.written_goals[-1], hn.core.block.position)  # never integrated
    assert hn.core.sm.state is SmState.RUN


@pytest.mark.parametrize("mode", [CmdMode.IDLE, CmdMode.HOLD])
def test_item8c_idle_or_hold_command_does_not_resume_run(h, mode, monkeypatch):
    _to_hold(h)
    resyncs = []
    real = h.core.pipeline.resync
    monkeypatch.setattr(h.core.pipeline, "resync", lambda p: (resyncs.append(1), real(p))[1])
    h.send(mode=mode)
    h.tick()
    assert h.core.state is SmState.HOLD and resyncs == []
    h.send(joint=0, value=5)  # a RUN command resumes
    h.tick()
    assert h.core.state is SmState.RUN and resyncs == [1]


# ---------------------------------------------------------------------------------------------------------------
# items 9, 10: state record read_ok, applied delta zero outside apply


def test_item9_read_ok_is_false_on_failed_read_ticks(h):
    h.run_policy(2)
    assert h.read_state()["read_ok"] == 1
    h.bus.faults.fail_next_read(1)
    h.tick()
    st = h.read_state()
    assert st["read_ok"] == 0 and st["hb_ok"] == 0
    h.tick()
    assert h.read_state()["read_ok"] == 1


def test_item9_boot_wait_records_have_read_ok_false(cfg, calibration):
    hn = Harness(cfg, calibration)
    with pytest.raises(StartupError):
        hn.core.start(arm_timeout_s=0.2)  # nobody presses the button
    st = hn.read_state()
    assert st["read_ok"] == 0 and st["sm_state"] == SmState.BOOT


def test_item10_applied_delta_is_zero_when_apply_did_not_run(h):
    h.run_policy(2, delta=[20, 0, 0, 0, 0, 0])
    assert float(h.read_state()["applied_delta_ticks"][0]) == 20.0
    _to_hold(h)
    st = h.read_state()
    assert not st["applied_delta_ticks"].any() and not st["clamped_mask"].any()


# ---------------------------------------------------------------------------------------------------------------
# item 11: the ARM wait reminds the operator


def test_item11_arm_wait_repeats_the_operator_message(cfg, calibration):
    hn = Harness(cfg, calibration)
    with pytest.raises(StartupError):
        hn.core.start(arm_timeout_s=3 * core_mod.ARM_REMINDER_S + 1)
    reminders = [m for m in hn.logs if "press the physical reset button" in m]
    assert len(reminders) >= 4 and "still waiting" in reminders[-1]


# ---------------------------------------------------------------------------------------------------------------
# item 13: FakeBus write semantics = transmitted


def test_item13_fakebus_goal_write_on_an_unpowered_bus_is_transmitted_not_acked(calibration):
    clock = FakeClock()
    bus = FakeBus(clock, calibration=calibration, power=lambda: False)
    goal = np.full(6, 2047, dtype=np.int32)
    assert bus.write_goal(goal)  # un-acked sync write: transmitted
    assert not bus.set_torque(True)  # acked: refused
    assert not any(bus.ping_all().values())
    with pytest.raises(ConnectionError):
        bus.read_reg("Present_Position", 1)
