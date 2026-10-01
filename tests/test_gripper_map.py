import math

import pytest

from so101_rl.sim.gripper_map import (
    gripper_percent_to_rad,
    gripper_percent_to_ticks,
    gripper_rad_to_percent,
    gripper_rad_to_ticks,
    gripper_range_rad,
    gripper_ticks_to_percent,
    gripper_ticks_to_rad,
)


def test_ends_and_midpoint():
    lo, hi = gripper_range_rad()
    assert gripper_percent_to_rad(0) == pytest.approx(lo)  # closed = lower joint-range end
    assert gripper_percent_to_rad(100) == pytest.approx(hi)  # open = upper end
    assert gripper_percent_to_rad(50) == pytest.approx((lo + hi) / 2)


def test_percent_rad_round_trip():
    for p in (0, 1, 25, 50, 73.5, 100):
        assert gripper_rad_to_percent(gripper_percent_to_rad(p)) == pytest.approx(p, abs=1e-9)


def test_ticks_use_calibration_range_not_center(calibration):
    c = calibration["gripper"]
    assert gripper_ticks_to_percent(c.range_min, calibration) == 0.0
    assert gripper_ticks_to_percent(c.range_max, calibration) == 100.0
    assert gripper_percent_to_ticks(50, calibration) == pytest.approx((c.range_min + c.range_max) / 2)


def test_ticks_round_trip_within_one_tick(calibration):
    c = calibration["gripper"]
    for t in range(c.range_min, c.range_max + 1, 7):
        back = gripper_rad_to_ticks(gripper_ticks_to_rad(t, calibration), calibration)
        assert abs(back - t) < 1
    for p in (0, 12.5, 50, 99, 100):
        t = gripper_percent_to_ticks(p, calibration)
        assert abs(gripper_ticks_to_percent(round(t), calibration) - p) <= 100.0 / (c.range_max - c.range_min)


def test_drive_mode_reverses_direction(calibration):
    from dataclasses import replace

    flipped = dict(calibration, gripper=replace(calibration["gripper"], drive_mode=1))
    c = flipped["gripper"]
    assert gripper_ticks_to_percent(c.range_max, flipped) == 0.0
    assert gripper_ticks_to_percent(c.range_min, flipped) == 100.0
    assert gripper_rad_to_ticks(gripper_percent_to_rad(0), flipped) == c.range_max


@pytest.mark.parametrize("bad", [-0.1, 100.1, 1e9, -5, math.nan, math.inf])
def test_out_of_range_percent_raises(bad, calibration):
    with pytest.raises(ValueError):
        gripper_percent_to_rad(bad)
    with pytest.raises(ValueError):
        gripper_percent_to_ticks(bad, calibration)


def test_out_of_range_ticks_and_rad_clip():
    assert gripper_rad_to_percent(-10.0) == 0.0
    assert gripper_rad_to_percent(10.0) == 100.0
