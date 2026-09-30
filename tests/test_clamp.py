"""Property tests for the pure-numpy clamps (plan task 2.1)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_rl.config import N_JOINTS, JointLimits
from so101_rl.safety.clamp import (
    check_range,
    clamp_joint_limits,
    clamp_tracking,
    clamp_velocity,
    sanitize_delta,
)
from so101_rl.safety.types import TripCause

MAX_DELTA = np.array([45, 45, 45, 45, 45, 60], dtype=np.int32)  # test fixture, not a config value
E = np.array([120, 120, 120, 120, 120, 400], dtype=np.int32)

any_float = st.floats(allow_nan=True, allow_infinity=True, width=64)
finite_huge = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e300, max_value=1e300)
vec_any = st.lists(any_float, min_size=N_JOINTS, max_size=N_JOINTS).map(lambda x: np.array(x, dtype=np.float64))
vec_finite = st.lists(finite_huge, min_size=N_JOINTS, max_size=N_JOINTS).map(lambda x: np.array(x, dtype=np.float64))
ticks = st.integers(min_value=0, max_value=4095)
tick_vec = st.lists(ticks, min_size=N_JOINTS, max_size=N_JOINTS).map(lambda x: np.array(x, dtype=np.int32))
wide_vec = st.lists(st.integers(-(10**6), 10**6), min_size=N_JOINTS, max_size=N_JOINTS).map(
    lambda x: np.array(x, dtype=np.int32)
)


@settings(max_examples=300, deadline=None)
@given(vec_any)
def test_sanitize_delta_output_always_finite_and_bounded(delta):
    out = np.empty(N_JOINTS, dtype=np.float32)
    trip = sanitize_delta(delta, MAX_DELTA, out)
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= MAX_DELTA)
    if not np.all(np.isfinite(delta)):
        assert trip == TripCause.NAN and not out.any()
    else:
        assert trip == TripCause.NONE


@settings(max_examples=300, deadline=None)
@given(vec_finite)
def test_sanitize_delta_finite_inputs_clip_monotonically(delta):
    out = np.empty(N_JOINTS, dtype=np.float32)
    assert sanitize_delta(delta, MAX_DELTA, out) == TripCause.NONE
    # sign preserved, small values untouched
    assert np.all(np.sign(out) * np.sign(delta) >= 0)
    small = np.abs(delta) <= MAX_DELTA
    np.testing.assert_allclose(out[small], delta[small].astype(np.float32), rtol=1e-6, atol=1e-6)


def test_sanitize_delta_single_nan_zeroes_whole_vector():
    out = np.full(N_JOINTS, 7.0, dtype=np.float32)
    delta = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert sanitize_delta(delta, MAX_DELTA, out) == TripCause.NAN
    assert not out.any()


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_sanitize_delta_inf_and_nan_rejected(bad):
    out = np.ones(N_JOINTS, dtype=np.float32)
    delta = np.zeros(N_JOINTS)
    delta[3] = bad
    assert sanitize_delta(delta, MAX_DELTA, out) == TripCause.NAN


@settings(max_examples=300, deadline=None)
@given(wide_vec, tick_vec)
def test_clamp_tracking_bounds_error_to_E(target, present):
    out = np.empty(N_JOINTS, dtype=np.int32)
    mask = np.empty(N_JOINTS, dtype=bool)
    clamp_tracking(target, present, E, out, mask)
    assert np.all(np.abs(out.astype(np.int64) - present) <= E)
    np.testing.assert_array_equal(mask, out != target)
    inside = np.abs(target.astype(np.int64) - present) <= E
    np.testing.assert_array_equal(out[inside], target[inside])


@settings(max_examples=300, deadline=None)
@given(tick_vec, tick_vec)
def test_clamp_velocity_bounds_step(target, prev):
    vmax = np.full(N_JOINTS, 20, dtype=np.int32)
    out = np.empty(N_JOINTS, dtype=np.int32)
    mask = np.empty(N_JOINTS, dtype=bool)
    clamp_velocity(target, prev, vmax, out, mask)
    assert np.all(np.abs(out.astype(np.int64) - prev) <= vmax)
    np.testing.assert_array_equal(mask, out != target)


@settings(max_examples=300, deadline=None)
@given(wide_vec)
def test_clamp_joint_limits_inside_limits(target):
    lim = JointLimits(
        min_ticks=np.array([1426, 869, 1010, 500, 0, 100], dtype=np.int32),
        max_ticks=np.array([2735, 3159, 3000, 3500, 4095, 3900], dtype=np.int32),
    )
    out = np.empty(N_JOINTS, dtype=np.int32)
    mask = np.empty(N_JOINTS, dtype=bool)
    clamp_joint_limits(target, lim, out, mask)
    assert np.all(out >= lim.min_ticks) and np.all(out <= lim.max_ticks)
    np.testing.assert_array_equal(mask, out != target)


def _good():
    return np.array([0, 100, 2047, 4095, 1, 4094], dtype=np.int32)


def test_check_range_accepts_valid():
    assert check_range(_good())
    assert check_range(_good().astype(np.int64))
    assert check_range(_good().astype(np.uint16))


@pytest.mark.parametrize("idx,val", [(0, 4096), (2, -1), (5, 100000), (3, -4096)])
def test_check_range_rejects_out_of_range(idx, val):
    g = _good()
    g[idx] = val
    assert not check_range(g)


def test_check_range_rejects_float_dtype_even_if_in_range():
    assert not check_range(_good().astype(np.float32))
    assert not check_range(_good().astype(np.float64))


def test_check_range_rejects_nan_and_inf():
    for bad in (np.nan, np.inf, -np.inf):
        g = _good().astype(np.float64)
        g[1] = bad
        assert not check_range(g)


def test_check_range_rejects_bool_and_wrong_shape():
    assert not check_range(np.ones(N_JOINTS, dtype=bool))
    assert not check_range(_good()[:5])
    assert not check_range(_good().reshape(2, 3))
    assert not check_range([0, 1, 2, 3, 4, 5])  # type: ignore[arg-type]


def test_out_of_range_goal_rejected_before_serialisation():
    """Action-unit Test 3 (unit level): a goal that only becomes bad at the very end is caught by check_range."""
    target = np.array([10, 10, 10, 10, 10, 10], dtype=np.int32)
    target[2] = 4096
    sent = []

    def write_goal(g):  # the "serialiser": must never see an unchecked goal
        sent.append(g.copy())

    if check_range(target):
        write_goal(target)
    assert sent == []


def test_clamps_do_not_modify_inputs():
    target = np.array([5000, -5, 100, 100, 100, 100], dtype=np.int32)
    before = target.copy()
    out = np.empty(N_JOINTS, dtype=np.int32)
    mask = np.empty(N_JOINTS, dtype=bool)
    clamp_tracking(target, np.full(N_JOINTS, 100, dtype=np.int32), E, out, mask)
    np.testing.assert_array_equal(target, before)
