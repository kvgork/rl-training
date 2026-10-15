"""ThermalMonitor with a fake clock (plan task 2.3)."""

import numpy as np
import pytest

from so101_rl.config import N_JOINTS, ThermalConfig
from so101_rl.safety.thermal import CONFIRM_READS, MEDIAN_READS, OUTLIER_C, HotCheck, MedianTemperature, ThermalMonitor
from so101_rl.safety.types import TripCause
from so101_rl.servo.timing import FakeClock

S = 1_000_000_000
CFG = ThermalConfig()  # the dataclass defaults mirror so101.yaml; asserted below


def temps(v):
    return np.full(N_JOINTS, v, dtype=np.int32)


ZERO_STATUS = np.zeros(N_JOINTS, dtype=np.int32)


@pytest.fixture
def clock():
    return FakeClock(start_ns=10 * S)


@pytest.fixture
def mon(clock):
    return ThermalMonitor(CFG, clock)


def step(mon, clock, t, *, dt_s=1 / 30, status=ZERO_STATUS, ok=True):
    clock.advance(int(dt_s * S))
    return mon.update(temps(t) if np.isscalar(t) else t, status, ok)


LAG = MEDIAN_READS // 2  # a new level reaches the median on read LAG + 1


def trip(mon, clock, t, **kw):
    """``LAG + CONFIRM_READS`` reads of ``t`` in a row: enough for any earlier level to leave the median, then
    ``CONFIRM_READS`` hot medians (a hot joint trips on the last one); the last result."""
    for _ in range(LAG + CONFIRM_READS - 1):
        step(mon, clock, t, **kw)
    return step(mon, clock, t, **kw)


def settle(mon, clock, t, **kw):
    """``LAG + 1`` reads of ``t``: the median is ``t`` from the last one on."""
    for _ in range(LAG + 1):
        step(mon, clock, t, **kw)


def test_defaults_match_yaml(cfg):
    assert cfg.thermal == CFG


def test_ramp_to_59_no_trip_then_60_trips(mon, clock):
    for t in range(30, 60):
        assert step(mon, clock, t) == TripCause.NONE
    assert step(mon, clock, 59) == TripCause.NONE
    for _ in range(LAG + CONFIRM_READS - 1):  # the median reaches 60 on read LAG + 1, then CONFIRM_READS of it
        assert step(mon, clock, 60) == TripCause.NONE
    assert step(mon, clock, 60) == TripCause.THERMAL
    assert mon.trips_last_hour() == 1
    assert mon.hottest_c == 60.0


def test_single_hot_joint_trips(mon, clock):
    t = temps(40)
    t[4] = 61
    assert trip(mon, clock, t) == TripCause.THERMAL


def test_one_tick_spikes_never_trip(mon, clock):
    """The real bus, 2026-10-11: 26 -> 69 -> 26 C on one joint for one tick; also two spikes a tick apart."""
    for spikes in ([69], [57, 69], [69, 26, 69]):
        for v in spikes:
            t = temps(26)
            t[0] = v
            assert step(mon, clock, t) == TripCause.NONE
        step(mon, clock, 26)
    assert mon.trips_last_hour() == 0 and not mon.latched


def test_hot_reads_must_be_consecutive_on_the_same_joint(mon, clock):
    for j in range(CONFIRM_READS):  # each read a different joint hot: no joint hot twice in a row
        t = temps(26)
        t[j] = 61
        assert step(mon, clock, t) == TripCause.NONE
    assert trip(mon, clock, 61) == TripCause.THERMAL


def test_failed_reads_neither_count_nor_reset_the_hot_run(mon, clock):
    step(mon, clock, 61)
    step(mon, clock, 61, ok=False)  # the arrays hold the old values: not a read
    step(mon, clock, 61)
    assert not mon.latched
    assert step(mon, clock, 61) == TripCause.THERMAL


def test_no_resume_at_51_and_dwell_is_enforced(mon, clock):
    trip(mon, clock, 61)
    assert not mon.may_resume()
    for _ in range(300):
        step(mon, clock, 51, dt_s=0.5)  # 150 s at 51 C
    assert not mon.may_resume()
    assert mon.latched
    # now cool to exactly resume_c and hold (the median is 50 from the last settle read on)
    settle(mon, clock, 50, dt_s=0.0)
    for _ in range(29):
        step(mon, clock, 50, dt_s=1.0)
    assert not mon.may_resume()  # 29 s
    step(mon, clock, 50, dt_s=0.5)
    assert not mon.may_resume()  # 29.5 s
    assert step(mon, clock, 50, dt_s=0.5) == TripCause.NONE  # 30.0 s -> latch clears on this update
    assert mon.may_resume()
    assert not mon.latched


def test_dwell_restarts_when_temperature_rises_again(mon, clock):
    trip(mon, clock, 61)
    settle(mon, clock, 50)
    for _ in range(25):
        step(mon, clock, 50, dt_s=1.0)
    step(mon, clock, 52, dt_s=1.0)  # a one-read spike above resume_c does not restart the dwell
    trip(mon, clock, 52, dt_s=1.0)  # but a held 52 does
    settle(mon, clock, 50, dt_s=1.0)  # cooling restarts on the last of these (t0)
    for _ in range(29):
        step(mon, clock, 50, dt_s=1.0)
    assert not mon.may_resume()  # 29 s since t0
    step(mon, clock, 50, dt_s=1.0)
    assert mon.may_resume()  # 30 s


def test_oscillation_59_61_is_exactly_one_trip(mon, clock):
    results = []
    trip(mon, clock, 61, dt_s=0.1)
    for i in range(40):
        results.append(step(mon, clock, 61 if i % 2 == 0 else 59, dt_s=0.1))
    assert mon.trips_last_hour() == 1
    assert all(r & TripCause.THERMAL for r in results)  # stays tripped while latched


def test_three_trips_in_an_hour_are_counted(mon, clock):
    for _ in range(3):
        trip(mon, clock, 61)
        settle(mon, clock, 40, dt_s=1.0)  # cool
        step(mon, clock, 40, dt_s=31.0)  # dwell satisfied -> unlatched
        assert not mon.latched
        clock.advance(600 * S)
    assert mon.trips_last_hour() == 3
    assert mon.trips_last_hour() == 3  # idempotent
    clock.advance(3600 * S)
    assert mon.trips_last_hour() == 0  # trailing window


@pytest.mark.parametrize("bad", [0, 255])
def test_dead_sensor_values_trip(mon, clock, bad):
    t = temps(35)
    t[2] = bad
    assert step(mon, clock, t) & TripCause.THERMAL


def test_zero_temperature_blocks_resume(mon, clock):
    step(mon, clock, 61)
    t = temps(40)
    t[0] = 0
    for _ in range(100):
        step(mon, clock, t, dt_s=1.0)
    assert not mon.may_resume() and mon.latched


def test_stale_temperature_trips_after_one_second(mon, clock):
    assert step(mon, clock, 35) == TripCause.NONE
    assert step(mon, clock, 35, dt_s=0.5, ok=False) == TripCause.NONE  # 0.5 s old
    assert step(mon, clock, 35, dt_s=0.4, ok=False) == TripCause.NONE  # 0.9 s old
    assert step(mon, clock, 35, dt_s=0.2, ok=False) & TripCause.THERMAL  # 1.1 s old
    assert mon.trips_last_hour() == 1


def test_stale_timer_does_not_start_at_construction(clock):
    """S10: a slow re-energise between construction and the first update must not look like a dead sensor."""
    mon = ThermalMonitor(CFG, clock)
    clock.advance(9 * S)  # e.g. reenergize ramped home for 9 s; nobody fed the monitor
    assert mon.update(temps(30), ZERO_STATUS, False) == TripCause.NONE
    # ...but a sensor that keeps failing while being fed every tick still trips after stale_temp_s
    for _ in range(int(CFG.stale_temp_s * 30)):
        assert step(mon, clock, 30, ok=False) == TripCause.NONE  # < stale_temp_s since the first failed read
    assert step(mon, clock, 30, dt_s=0.1, ok=False) & TripCause.THERMAL


def test_a_gap_in_feeding_restarts_the_stale_timer(mon, clock):
    """S10: after good reads, a multi-second blocking reenergize (no updates) followed by one failed read is not a
    stale-temperature trip."""
    for _ in range(10):
        step(mon, clock, 35)
    assert step(mon, clock, 35, dt_s=8.0, ok=False) == TripCause.NONE
    assert step(mon, clock, 35, ok=True) == TripCause.NONE
    assert mon.trips_last_hour() == 0


def test_failed_block_never_counts_as_cooling(mon, clock):
    """S12: run the failed reads for LONGER than the dwell, fed every tick, so a broken implementation that
    counted them as cooling would resume."""
    step(mon, clock, 61)
    step(mon, clock, 40)
    assert not mon.may_resume()
    for _ in range(int((CFG.resume_dwell_s + 10) * 30)):
        step(mon, clock, 40, ok=False)  # previous arrays say 40 C, but they are not fresh
        assert not mon.may_resume()
    assert mon.latched
    # and once reads come back, the dwell starts from zero
    for _ in range(int(CFG.resume_dwell_s * 30) - 2):
        assert step(mon, clock, 40) & TripCause.THERMAL
    assert not mon.may_resume()


def test_good_reads_for_the_dwell_do_resume(mon, clock):
    """Positive control for the test above: the same timeline with good reads clears the latch."""
    step(mon, clock, 61)
    for _ in range(int(CFG.resume_dwell_s * 30) + 30):
        step(mon, clock, 40)
    assert mon.may_resume() and not mon.latched
    assert step(mon, clock, 40) == TripCause.NONE


@pytest.mark.parametrize("joint", range(N_JOINTS))
@pytest.mark.parametrize("bit", [1, 4, 32, 128])
def test_any_status_bit_is_protection_trip(mon, clock, joint, bit):
    status = ZERO_STATUS.copy()
    status[joint] = bit
    cause = step(mon, clock, 35, status=status)
    assert cause == TripCause.PROTECTION_BIT
    assert mon.trips_last_hour() == 0  # protection bits are not thermal latch events


def test_protection_bit_and_thermal_combine(mon, clock):
    status = ZERO_STATUS.copy()
    status[1] = 4
    assert trip(mon, clock, 65, status=status) == TripCause.THERMAL | TripCause.PROTECTION_BIT


def test_max_seen_tracks_per_joint(mon, clock):
    t = np.array([30, 41, 35, 52, 33, 38], dtype=np.int32)
    settle(mon, clock, t)
    for _ in range(MEDIAN_READS):
        step(mon, clock, temps(31))
    np.testing.assert_array_equal(mon.max_seen_c, np.maximum(t, 31))
    assert mon.hottest_c == 31.0


# ---------------------------------------------------------------- median filter (plan section 10, 2026-10-15)


def test_real_spike_runs_never_trip(mon, clock):
    """Recorded runs: 33 -> 94 -> 98 -> 65 -> 33 C (tripped THERMAL on 2026-10-12 through CONFIRM_READS alone), a 150 C
    read (2026-10-14), and a 4-read run (the longest in 740k real reads)."""
    settle(mon, clock, 33)
    for run in ([94, 98, 65], [150], [57, 64, 63, 84]):
        for v in run:
            t = temps(33)
            t[0] = v
            assert step(mon, clock, t) == TripCause.NONE
        for _ in range(MEDIAN_READS):
            assert step(mon, clock, 33) == TripCause.NONE
    assert mon.trips_last_hour() == 0 and not mon.latched
    assert mon.outlier_reads[0] == 3 + 1 + 4 and mon.outlier_reads[1:].sum() == 0


def test_a_held_hot_level_still_trips_within_the_median_lag(mon, clock):
    settle(mon, clock, 40)
    t = temps(40)
    t[2] = 62
    for _ in range(LAG + CONFIRM_READS - 1):
        assert step(mon, clock, t) == TripCause.NONE
    assert step(mon, clock, t) == TripCause.THERMAL


def test_dead_sensor_bypasses_the_median(mon, clock):
    settle(mon, clock, 30)
    t = temps(30)
    t[5] = 255
    assert step(mon, clock, t) == TripCause.THERMAL  # one raw 255 among ten good medians: at once


def test_median_filter_before_the_window_fills():
    f = MedianTemperature()
    np.testing.assert_array_equal(f.update(temps(30)), temps(30))
    np.testing.assert_array_equal(f.update(temps(40)), temps(35))
    assert len(f._buf) == MEDIAN_READS and OUTLIER_C == 3.0


def test_hot_check_ignores_a_spike_run_after_good_reads():
    h = HotCheck(CFG.trip_c)
    for _ in range(MEDIAN_READS):
        assert not h.update(temps(30))
    assert not any(h.update(temps(v)) for v in (94, 98, 65))
