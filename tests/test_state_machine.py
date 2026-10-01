"""SafetyStateMachine transitions (plan task 2.5). Table-driven, fake clock."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from so101_rl.safety.state_machine import SafetyStateMachine, SmInputs, SmOutputs
from so101_rl.safety.types import RESUME_BY_CAUSE, ResumePolicy, SmState, TripCause
from so101_rl.servo.timing import FakeClock

MS = 1_000_000
S = 1_000_000_000
T = TripCause


@pytest.fixture
def clock():
    return FakeClock(start_ns=1000 * S)


@pytest.fixture
def sm(cfg, clock):
    return SafetyStateMachine(cfg, clock)


@pytest.fixture
def out():
    return SmOutputs()


def run_state(sm, out):
    """Energise and take the first RUN step (fresh command)."""
    sm.mark_energized()
    sm.step(SmInputs(cmd_fresh=True, cmd_age_ns=0), out)
    assert out.state is SmState.RUN
    return sm


def fresh(**kw):
    return SmInputs(cmd_fresh=True, cmd_age_ns=0, **kw)


# ---- start-up -------------------------------------------------------------------------------------------------
def test_boot_holds_until_energized_then_enters_run_with_resync(sm, out):
    for _ in range(3):
        sm.step(fresh(), out)
        assert out.state is SmState.BOOT and not out.use_policy and not out.resync_target
    sm.mark_energized()
    sm.step(fresh(), out)
    assert out.state is SmState.RUN and out.use_policy and out.resync_target
    sm.step(fresh(), out)
    assert out.state is SmState.RUN and not out.resync_target  # only on ENTRY to RUN


def test_fatal_fault_before_energise_goes_straight_to_torque_off(sm, out):
    sm.step(SmInputs(trips=T.NAN), out)
    assert out.state is SmState.TORQUE_OFF and out.torque_off_now and not out.run_safe_sequence


# ---- stale command / HOLD --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "age_ms,expected",
    [(0, SmState.RUN), (100, SmState.RUN), (149, SmState.RUN), (150, SmState.RUN), (151, SmState.HOLD), (400, SmState.HOLD)],
)
def test_run_to_hold_threshold(sm, out, age_ms, expected):
    run_state(sm, out)
    sm.step(SmInputs(cmd_age_ns=age_ms * MS), out)
    assert out.state is expected
    assert out.use_policy is (expected is SmState.RUN)
    if expected is SmState.HOLD:
        assert sm.fault_bits & T.STALE_CMD


def test_hold_to_run_only_on_fresh_seq_with_resync(sm, out):
    run_state(sm, out)
    sm.step(SmInputs(cmd_age_ns=200 * MS), out)
    assert out.state is SmState.HOLD and not out.use_policy
    sm.step(SmInputs(cmd_age_ns=230 * MS, cmd_fresh=False), out)
    assert out.state is SmState.HOLD
    sm.step(SmInputs(cmd_age_ns=30 * MS, cmd_fresh=False), out)
    assert out.state is SmState.HOLD  # young but not a NEW seq
    sm.step(fresh(), out)
    assert out.state is SmState.RUN and out.resync_target and out.use_policy
    assert sm.fault_bits == T.NONE  # STALE_CMD cleared on resume
    sm.step(fresh(), out)
    assert not out.resync_target


def test_hold_ignores_fresh_flag_if_the_command_is_already_stale(sm, out):
    run_state(sm, out)
    sm.step(SmInputs(cmd_age_ns=200 * MS), out)
    sm.step(SmInputs(cmd_age_ns=151 * MS, cmd_fresh=True), out)
    assert out.state is SmState.HOLD


@pytest.mark.parametrize("n_miss,expected", [(1, SmState.HOLD), (4, SmState.HOLD), (5, SmState.SAFE_STOPPING), (9, SmState.SAFE_STOPPING)])
def test_consecutive_misses(sm, cfg, out, n_miss, expected):
    assert cfg.rates.max_consecutive_misses == 5
    run_state(sm, out)
    sm.step(SmInputs(cmd_age_ns=200 * MS, consecutive_misses=1), out)  # into HOLD
    sm.step(SmInputs(cmd_age_ns=200 * MS + n_miss * 100 * MS, consecutive_misses=n_miss), out)
    assert out.state is expected
    if expected is SmState.SAFE_STOPPING:
        assert out.run_safe_sequence and not out.use_policy
        assert sm.resume_policy() is ResumePolicy.FRESH_PROCESS  # after SAFE_STOPPING only a fresh process


def test_safe_stopping_ignores_policy_commands(sm, out):
    run_state(sm, out)
    sm.step(SmInputs(consecutive_misses=5, cmd_age_ns=600 * MS), out)
    assert out.state is SmState.SAFE_STOPPING
    for _ in range(50):
        sm.step(fresh(may_resume_thermal=True), out)
        assert out.state is SmState.SAFE_STOPPING and not out.use_policy and out.run_safe_sequence
    sm.notify_safe_done()
    sm.step(fresh(), out)
    assert out.state is SmState.TORQUE_OFF and out.torque_off_now and not out.run_safe_sequence


# ---- thermal (plan section 10, 2026-10-01: to rest, torque off; resume only through a full reenergize) ----------
def thermal_trip_to_torque_off(sm, out):
    """RUN -> (THERMAL) SAFE_STOPPING -> sequencer done -> TORQUE_OFF."""
    sm.step(fresh(trips=T.THERMAL), out)
    assert out.state is SmState.SAFE_STOPPING and out.run_safe_sequence and not out.use_policy
    assert not out.torque_off_now  # the sequencer ramps to rest first; it turns torque off itself
    sm.notify_safe_done()
    sm.step(fresh(trips=T.THERMAL), out)
    assert out.state is SmState.TORQUE_OFF and out.torque_off_now


def resume_after_dwell(sm, out):
    """TORQUE_OFF (thermal only) -> CHECK (reenergize requested) -> mark_energized -> RUN (resync)."""
    sm.step(fresh(may_resume_thermal=True), out)
    assert out.state is SmState.CHECK and out.reenergize_requested and out.torque_off_now and not out.use_policy
    sm.mark_energized()
    sm.step(fresh(), out)
    assert out.state is SmState.RUN and out.resync_target and out.use_policy


def test_thermal_trip_goes_to_rest_and_torque_off_never_holds_energised(sm, out):
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    assert sm.fault_bits & T.THERMAL and sm.resume_policy() is ResumePolicy.AUTO
    for _ in range(20):
        sm.step(fresh(trips=T.THERMAL), out)  # the monitor keeps reporting while latched
        assert out.state is SmState.TORQUE_OFF and out.torque_off_now and not out.reenergize_requested
    sm.step(fresh(trips=T.NONE, may_resume_thermal=False), out)
    assert out.state is SmState.TORQUE_OFF  # dwell not over
    sm.step(fresh(trips=T.THERMAL, may_resume_thermal=True), out)
    assert out.state is SmState.TORQUE_OFF  # still reporting THERMAL: do not resume


def test_thermal_resume_is_a_full_reenergize_then_run_with_resync(sm, out):
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    resume_after_dwell(sm, out)
    assert sm.fault_bits == T.NONE
    sm.step(fresh(), out)
    assert not out.resync_target and not out.reenergize_requested  # one-shot flags


def test_cooldown_state_is_never_entered(sm, out):
    run_state(sm, out)
    seen = set()
    for inp in (fresh(trips=T.THERMAL), fresh(trips=T.THERMAL), fresh(may_resume_thermal=True)):
        sm.step(inp, out)
        seen.add(out.state)
        sm.notify_safe_done()
    assert SmState.COOLDOWN not in seen


def test_thermal_resume_requires_a_good_block_read(sm, out):
    """S6: no resume path may ignore block_ok."""
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    sm.step(fresh(may_resume_thermal=True, block_ok=False), out)
    assert out.state is SmState.TORQUE_OFF and not out.reenergize_requested
    sm.step(fresh(may_resume_thermal=True), out)
    assert out.state is SmState.CHECK


def test_resync_after_energise_waits_for_a_good_block_read(sm, out):
    """S6: the first RUN step after mark_energized must not resync to a stale (failed) read."""
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    sm.step(fresh(may_resume_thermal=True), out)
    sm.mark_energized()
    sm.step(fresh(block_ok=False), out)
    assert out.state is SmState.RUN and not out.resync_target and not out.use_policy
    sm.step(fresh(block_ok=True), out)
    assert out.resync_target and out.use_policy


def test_thermal_relatched_during_check_goes_back_to_torque_off(sm, out):
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    sm.step(fresh(may_resume_thermal=True), out)
    assert out.state is SmState.CHECK
    sm.step(fresh(trips=T.THERMAL), out)
    assert out.state is SmState.TORQUE_OFF and out.torque_off_now
    assert sm.resume_policy() is ResumePolicy.AUTO
    resume_after_dwell(sm, out)


def test_failed_automatic_reenergize_needs_a_human(sm, out):
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    sm.step(fresh(may_resume_thermal=True), out)
    sm.notify_reenergize_failed()
    for _ in range(10):
        sm.step(fresh(may_resume_thermal=True), out)
        assert out.state is SmState.TORQUE_OFF and not out.reenergize_requested
    assert sm.resume_policy() is ResumePolicy.HUMAN_PHYSICAL_RESET


def test_thermal_from_hold_also_goes_to_rest(sm, out):
    run_state(sm, out)
    sm.step(SmInputs(cmd_age_ns=300 * MS), out)
    assert out.state is SmState.HOLD
    sm.step(SmInputs(cmd_age_ns=300 * MS, trips=T.THERMAL), out)
    assert out.state is SmState.SAFE_STOPPING
    sm.notify_safe_done()
    sm.step(SmInputs(cmd_age_ns=300 * MS, may_resume_thermal=True), out)
    assert out.state is SmState.CHECK  # THERMAL + the STALE_CMD of the HOLD before it is still thermal-only


def test_another_fault_during_thermal_torque_off_blocks_the_automatic_resume(sm, out):
    run_state(sm, out)
    thermal_trip_to_torque_off(sm, out)
    sm.step(fresh(trips=T.PROTECTION_BIT), out)
    sm.step(fresh(may_resume_thermal=True), out)
    assert out.state is SmState.TORQUE_OFF
    assert sm.resume_policy() is ResumePolicy.HUMAN_PHYSICAL_RESET


def test_fourth_thermal_trip_in_an_hour_needs_a_human(sm, out, clock):
    run_state(sm, out)
    for i in range(3):
        thermal_trip_to_torque_off(sm, out)
        resume_after_dwell(sm, out)
        clock.advance(600 * S)
        assert sm.thermal_trips_last_hour() == i + 1
    sm.step(fresh(trips=T.THERMAL), out)
    assert out.state is SmState.SAFE_STOPPING and out.run_safe_sequence
    assert sm.resume_policy() is ResumePolicy.HUMAN_PHYSICAL_RESET
    sm.notify_safe_done()
    for _ in range(10):
        sm.step(fresh(may_resume_thermal=True), out)
        assert out.state is SmState.TORQUE_OFF and not out.reenergize_requested  # no automatic resume


def test_thermal_trips_older_than_an_hour_do_not_count(sm, out, clock):
    run_state(sm, out)
    for _ in range(3):
        thermal_trip_to_torque_off(sm, out)
        resume_after_dwell(sm, out)
        clock.advance(1300 * S)
    # trips at t=0, 1300, 2600; now t=3900 -> the first is out of the window
    thermal_trip_to_torque_off(sm, out)
    assert sm.resume_policy() is ResumePolicy.AUTO
    assert sm.thermal_trips_last_hour() == 3
    resume_after_dwell(sm, out)


@pytest.mark.parametrize("max_per_hour", [1, 10])
def test_trip_ring_follows_the_configured_limit(cfg, clock, max_per_hour):
    """S10: the ring is sized from the config (the loader bounds max_trips_per_hour to 1..10)."""
    cfg2 = replace(cfg, thermal=replace(cfg.thermal, max_trips_per_hour=max_per_hour))
    sm, out = SafetyStateMachine(cfg2, clock), SmOutputs()
    run_state(sm, out)
    for _ in range(max_per_hour):
        thermal_trip_to_torque_off(sm, out)
        resume_after_dwell(sm, out)
        clock.advance(S)
    assert sm.thermal_trips_last_hour() == max_per_hour
    sm.step(fresh(trips=T.THERMAL), out)
    assert sm.resume_policy() is ResumePolicy.HUMAN_PHYSICAL_RESET


# ---- fatal classes --------------------------------------------------------------------------------------------------
def test_nan_goes_safe_stopping_then_torque_off_and_never_resumes(sm, out):
    run_state(sm, out)
    sm.step(fresh(trips=T.NAN), out)
    assert out.state is SmState.SAFE_STOPPING and out.run_safe_sequence and not out.use_policy
    assert sm.resume_policy() is ResumePolicy.FRESH_PROCESS
    sm.notify_safe_done()
    for _ in range(100):
        sm.step(fresh(may_resume_thermal=True), out)
        assert out.state is SmState.TORQUE_OFF and out.torque_off_now and not out.use_policy


FRESH = [T.NAN, T.STUCK, T.OOD, T.RANGE, T.BOX_PRESENT]
HUMAN_STOP = [T.PROTECTION_BIT, T.CALIB_MISMATCH, T.USER_STOP]
LINK = [T.DEADMAN, T.HEARTBEAT_FAIL, T.BUS_FAIL]


@pytest.mark.parametrize("cause", FRESH + HUMAN_STOP)
def test_stop_causes_go_through_safe_stopping(sm, out, cause):
    run_state(sm, out)
    sm.step(fresh(trips=cause), out)
    assert out.state is SmState.SAFE_STOPPING
    assert sm.fault_bits & cause
    assert sm.resume_policy() is RESUME_BY_CAUSE[cause]


@pytest.mark.parametrize("cause", LINK)
def test_link_causes_latch(sm, out, cause):
    run_state(sm, out)
    sm.step(fresh(trips=cause), out)
    assert out.state is SmState.LATCHED
    assert sm.resume_policy() is RESUME_BY_CAUSE[cause] is ResumePolicy.HUMAN_PHYSICAL_RESET


def test_user_stop_request(sm, out):
    run_state(sm, out)
    sm.step(fresh(stop_requested=True), out)
    assert out.state is SmState.SAFE_STOPPING and sm.fault_bits & T.USER_STOP


@pytest.mark.parametrize("clamp", [T.TRACKING, T.JOINT_LIMIT, T.BOX, T.TRACKING | T.BOX | T.JOINT_LIMIT])
def test_clamps_in_trips_are_not_transitions(sm, out, clamp):
    run_state(sm, out)
    for _ in range(10):
        sm.step(fresh(trips=clamp), out)
        assert out.state is SmState.RUN and out.use_policy
    assert sm.fault_bits == T.NONE


def test_safe_stopping_after_thermal_escalates_on_a_fatal_cause(sm, out):
    run_state(sm, out)
    sm.step(fresh(trips=T.THERMAL), out)
    sm.step(fresh(trips=T.THERMAL | T.NAN), out)
    assert out.state is SmState.SAFE_STOPPING
    sm.notify_safe_done()
    sm.step(fresh(may_resume_thermal=True), out)
    assert out.state is SmState.TORQUE_OFF  # NAN accumulated: no automatic resume
    assert sm.resume_policy() is ResumePolicy.FRESH_PROCESS


# ---- LATCHED ----------------------------------------------------------------------------------------------------------
def test_heartbeat_fail_latches_and_nothing_leaves_latched(sm, out):
    run_state(sm, out)
    sm.step(fresh(hb_ok=False), out)
    assert out.state is SmState.LATCHED and out.torque_off_now and not out.use_policy
    assert sm.fault_bits & T.HEARTBEAT_FAIL
    for i in range(200):
        sm.step(fresh(hb_ok=True, may_resume_thermal=True, mcu=SimpleNamespace(latched=False)), out)
        assert out.state is SmState.LATCHED and not out.use_policy and not out.resync_target
        assert out.torque_off_now  # S4: every tick, not only on entry
    sm.mark_energized()  # also no way back via the start-up hook
    sm.notify_safe_done()
    sm.step(fresh(), out)
    assert out.state is SmState.LATCHED
    assert sm.resume_policy() is ResumePolicy.HUMAN_PHYSICAL_RESET


def test_mcu_latched_latches(sm, out):
    run_state(sm, out)
    sm.step(fresh(mcu=SimpleNamespace(latched=True)), out)
    assert out.state is SmState.LATCHED and sm.fault_bits & T.DEADMAN


def test_mcu_not_latched_is_fine(sm, out):
    run_state(sm, out)
    sm.step(fresh(mcu=SimpleNamespace(latched=False)), out)
    assert out.state is SmState.RUN


def test_latched_from_safe_stopping_and_torque_off(sm, out):
    run_state(sm, out)
    sm.step(fresh(trips=T.NAN), out)
    sm.step(fresh(hb_ok=False), out)
    assert out.state is SmState.LATCHED


def test_single_bus_failures_hold_but_three_in_a_row_latch(sm, out, cfg):
    assert cfg.rates.max_bus_failures == 3
    run_state(sm, out)
    for _ in range(2):
        sm.step(fresh(block_ok=False), out)
        assert out.state is SmState.RUN and not out.use_policy  # hold the target, do not integrate
    sm.step(fresh(block_ok=True), out)  # a good read resets the count
    assert out.use_policy
    for _ in range(2):
        sm.step(fresh(block_ok=False), out)
    assert out.state is SmState.RUN
    sm.step(fresh(block_ok=False), out)
    assert out.state is SmState.LATCHED and sm.fault_bits & T.BUS_FAIL


def test_fault_bits_accumulate_across_causes(sm, out):
    run_state(sm, out)
    sm.step(fresh(trips=T.STUCK), out)
    sm.step(fresh(trips=T.NAN), out)
    assert sm.fault_bits == T.STUCK | T.NAN
    assert sm.resume_policy() is ResumePolicy.FRESH_PROCESS


def test_use_policy_only_in_run(sm, out):
    seen = []
    sm.step(SmInputs(), out)  # BOOT
    seen.append((out.state, out.use_policy))
    sm.mark_energized()
    for inp in (fresh(), fresh(trips=T.THERMAL), fresh(trips=T.NAN)):
        sm.step(inp, out)
        seen.append((out.state, out.use_policy))
    assert [s for s, _ in seen] == [SmState.BOOT, SmState.RUN, SmState.SAFE_STOPPING, SmState.SAFE_STOPPING]
    assert [u for _, u in seen] == [False, True, False, False]


def test_hold_to_run_requires_a_good_block_read(sm, out):
    """S6: a fresh command with a failed read does not resume (and does not resync to stale data)."""
    run_state(sm, out)
    sm.step(SmInputs(cmd_age_ns=200 * MS), out)
    assert out.state is SmState.HOLD
    sm.step(fresh(block_ok=False), out)
    assert out.state is SmState.HOLD and not out.resync_target and not out.use_policy
    sm.step(fresh(), out)
    assert out.state is SmState.RUN and out.resync_target


def test_latched_emits_torque_off_every_tick_even_with_bad_reads(sm, out):
    run_state(sm, out)
    sm.step(fresh(trips=T.DEADMAN), out)
    for ok in (True, False, True):
        sm.step(fresh(block_ok=ok), out)
        assert out.state is SmState.LATCHED and out.torque_off_now


def test_a_thermal_trip_during_a_stale_command_stop_does_not_auto_resume(sm, out):
    """Reviewer follow-up: N stale commands need a fresh process; a THERMAL on top must not unlock the auto path."""
    run_state(sm, out)
    sm.step(SmInputs(consecutive_misses=5, cmd_age_ns=600 * MS), out)
    assert out.state is SmState.SAFE_STOPPING
    sm.step(fresh(trips=T.THERMAL), out)
    sm.notify_safe_done()
    for _ in range(5):
        sm.step(fresh(may_resume_thermal=True), out)
        assert out.state is SmState.TORQUE_OFF and not out.reenergize_requested
    assert sm.resume_policy() is ResumePolicy.FRESH_PROCESS
