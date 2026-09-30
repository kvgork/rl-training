"""StuckActionMonitor (plan task 2.3)."""

import numpy as np

from so101_rl.config import N_JOINTS
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.types import TripCause

STILL = np.zeros(N_JOINTS, dtype=np.int32)
MOVING = np.ones(N_JOINTS, dtype=np.int32)


def vec(x):
    return np.array([x, 0, 0, 0, 0, 0], dtype=np.float32)


def test_identical_nonzero_delta_for_90_ticks_trips_on_the_90th():
    m = StuckActionMonitor()
    for _ in range(89):
        assert m.update(vec(10.0), STILL) == TripCause.NONE
    assert m.update(vec(10.0), STILL) == TripCause.STUCK
    assert m.update(vec(10.0), STILL) == TripCause.STUCK  # stays tripped while it persists


def test_changing_delta_resets():
    m = StuckActionMonitor()
    for _ in range(80):
        m.update(vec(10.0), STILL)
    assert m.update(vec(12.0), STILL) == TripCause.NONE  # differs by 2 > epsilon
    assert m.count == 1
    for _ in range(88):
        assert m.update(vec(12.0), STILL) == TripCause.NONE
    assert m.update(vec(12.0), STILL) == TripCause.STUCK


def test_within_epsilon_counts_as_identical():
    m = StuckActionMonitor()
    for i in range(89):
        m.update(vec(10.0 + (0.3 if i % 2 else 0.0)), STILL)
    assert m.update(vec(10.0), STILL) == TripCause.STUCK


def test_reference_does_not_drift():
    m = StuckActionMonitor()
    # creeping by 0.3 each tick leaves epsilon of the FIRST value after two ticks -> never accumulates 90
    for i in range(300):
        assert m.update(vec(10.0 + 0.3 * i), STILL) == TripCause.NONE


def test_motion_resets():
    m = StuckActionMonitor()
    for _ in range(89):
        m.update(vec(10.0), STILL)
    assert m.update(vec(10.0), MOVING) == TripCause.NONE
    assert m.count == 0
    for _ in range(89):
        assert m.update(vec(10.0), STILL) == TripCause.NONE


def test_one_moving_joint_is_enough_to_reset():
    m = StuckActionMonitor()
    moving = STILL.copy()
    moving[3] = 1
    for _ in range(200):
        assert m.update(vec(10.0), moving) == TripCause.NONE


def test_zero_delta_never_trips():
    m = StuckActionMonitor()
    for _ in range(500):
        assert m.update(np.zeros(N_JOINTS, dtype=np.float32), STILL) == TripCause.NONE


def test_sub_epsilon_delta_is_treated_as_zero():
    m = StuckActionMonitor()
    for _ in range(500):
        assert m.update(vec(0.4), STILL) == TripCause.NONE


def test_negative_delta_counts():
    m = StuckActionMonitor(repeat_ticks=5)
    out = [m.update(vec(-7.0), STILL) for _ in range(5)]
    assert out[-1] == TripCause.STUCK and all(o == TripCause.NONE for o in out[:-1])


def test_custom_repeat_ticks():
    m = StuckActionMonitor(repeat_ticks=3)
    assert [m.update(vec(1.0), STILL) for _ in range(3)] == [TripCause.NONE, TripCause.NONE, TripCause.STUCK]


def test_reset_method():
    m = StuckActionMonitor(repeat_ticks=3)
    m.update(vec(1.0), STILL)
    m.update(vec(1.0), STILL)
    m.reset()
    assert m.update(vec(1.0), STILL) == TripCause.NONE
