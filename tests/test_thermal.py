"""ThermalMonitor with a fake clock (plan task 2.3)."""

import numpy as np
import pytest

from so101_rl.config import N_JOINTS, ThermalConfig
from so101_rl.safety.thermal import ThermalMonitor
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


def test_defaults_match_yaml(cfg):
    assert cfg.thermal == CFG


def test_ramp_to_59_no_trip_then_60_trips(mon, clock):
    for t in range(30, 60):
        assert step(mon, clock, t) == TripCause.NONE
    assert step(mon, clock, 59) == TripCause.NONE
    assert step(mon, clock, 60) == TripCause.THERMAL
    assert mon.trips_last_hour() == 1
    assert mon.hottest_c == 60.0


def test_single_hot_joint_trips(mon, clock):
    t = temps(40)
    t[4] = 61
    assert step(mon, clock, t) == TripCause.THERMAL


def test_no_resume_at_51_and_dwell_is_enforced(mon, clock):
    step(mon, clock, 61)
    assert not mon.may_resume()
    for _ in range(300):
        step(mon, clock, 51, dt_s=0.5)  # 150 s at 51 C
    assert not mon.may_resume()
    assert mon.latched
    # now cool to exactly resume_c and hold
    step(mon, clock, 50, dt_s=0.0)
    for _ in range(29):
        step(mon, clock, 50, dt_s=1.0)
    assert not mon.may_resume()  # 29 s
    step(mon, clock, 50, dt_s=0.5)
    assert not mon.may_resume()  # 29.5 s
    assert step(mon, clock, 50, dt_s=0.5) == TripCause.NONE  # 30.0 s -> latch clears on this update
    assert mon.may_resume()
    assert not mon.latched


def test_dwell_restarts_when_temperature_rises_again(mon, clock):
    step(mon, clock, 61)
    step(mon, clock, 50)
    for _ in range(25):
        step(mon, clock, 50, dt_s=1.0)
    step(mon, clock, 52, dt_s=1.0)  # bump above resume_c
    step(mon, clock, 50, dt_s=1.0)  # cooling restarts here (t0)
    for _ in range(29):
        step(mon, clock, 50, dt_s=1.0)
    assert not mon.may_resume()  # 29 s since t0
    step(mon, clock, 50, dt_s=1.0)
    assert mon.may_resume()  # 30 s


def test_oscillation_59_61_is_exactly_one_trip(mon, clock):
    results = []
    for i in range(40):
        results.append(step(mon, clock, 61 if i % 2 == 0 else 59, dt_s=0.1))
    assert mon.trips_last_hour() == 1
    assert all(r & TripCause.THERMAL for r in results)  # stays tripped while latched


def test_three_trips_in_an_hour_are_counted(mon, clock):
    for _ in range(3):
        step(mon, clock, 61)
        step(mon, clock, 40, dt_s=1.0)  # cool
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
    assert step(mon, clock, 35, dt_s=0.9, ok=False) == TripCause.NONE  # 0.9 s old
    assert step(mon, clock, 35, dt_s=0.2, ok=False) & TripCause.THERMAL  # 1.1 s old
    assert mon.trips_last_hour() == 1


def test_failed_block_never_counts_as_cooling(mon, clock):
    step(mon, clock, 61)
    step(mon, clock, 40)
    for _ in range(10):
        step(mon, clock, 40, dt_s=0.05, ok=False)  # previous arrays say 40 C, but they are not fresh
    assert not mon.may_resume()


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
    assert step(mon, clock, 65, status=status) == TripCause.THERMAL | TripCause.PROTECTION_BIT


def test_max_seen_tracks_per_joint(mon, clock):
    t = np.array([30, 41, 35, 52, 33, 38], dtype=np.int32)
    step(mon, clock, t)
    step(mon, clock, temps(31))
    np.testing.assert_array_equal(mon.max_seen_c, np.maximum(t, 31))
    assert mon.hottest_c == 31.0
