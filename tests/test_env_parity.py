"""Real env (``So101GoalEnv`` on the CPU reference rig) vs batched env (``BatchedGoalEnv``): the action mapping and
the episode-end rule are one definition (``env/tasks.py``), driven through both envs (integration review 2026-10-07).

* Action mapping: ``tasks.action_to_delta_np`` == ``tasks.action_to_delta_torch`` bit for bit, including ±inf, 1e3,
  NaN and the values on a .5 rounding boundary where a float32 product rounds differently from the float64 one; the
  two envs apply the same deltas for the same actions (GPU).
* Episode end: a success predicate on a tripped step is never a success (trip-on-success: truncated only), and a
  success on the step-limit step is ``terminated`` AND ``truncated`` in both envs (GPU).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from so101_rl.env.backend import SimBackend
from so101_rl.env.goal_env import So101GoalEnv
from so101_rl.env.tasks import REACH, action_to_delta_np, action_to_delta_torch
from so101_rl.safety.types import TripCause

INF = np.float32(np.inf)
NAN = np.float32(np.nan)
# Rows: one policy action each. Row 1: out of range (inf, 1e3) and exact halves (0.5 * 45 = 22.5 -> 22, 0.125 * 60 =
# 7.5 -> 8). Row 2: float32 values whose float64 product is just off .5 (0.1 * 45 = 4.50000007 -> 5, but the float32
# product rounds to 4.5 -> 4; 0.7 -> 31 vs 32; gripper 0.275 -> 17 vs 16). Row 3: a NaN (the sanitizer must trip).
ACTIONS = np.array(
    [
        [INF, -INF, 1e3, -1e3, 0.5, 0.125],
        [0.1, -0.1, 0.7, -0.7, 0.0, 0.275],
        [NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    np.float32,
)
EXPECTED = np.array(
    [
        [45, -45, 45, -45, 22, 8],
        [5, -5, 31, -31, 0, 17],
        [np.nan, 0, 0, 0, 0, 0],
    ],
    np.float32,
)


# ---------------------------------------------------------------------------------------------------- CPU
def test_action_mapping_np_equals_torch_on_special_values(cfg):
    md = cfg.action.max_delta_ticks
    d_np = action_to_delta_np(ACTIONS, md)
    d_t = action_to_delta_torch(torch.as_tensor(ACTIONS), torch.tensor(np.asarray(md, np.float64))).numpy()
    np.testing.assert_array_equal(d_np, EXPECTED)  # NaN == NaN for assert_array_equal
    np.testing.assert_array_equal(d_t, EXPECTED)
    assert d_np.dtype == d_t.dtype == np.float32


def test_action_mapping_np_equals_torch_on_random_and_boundary_values(cfg):
    md = cfg.action.max_delta_ticks
    rng = np.random.default_rng(0)
    # every float32 k/20000 (all the .5 boundaries of both joint scales) plus random values in and out of range
    grid = (np.arange(-20000, 20001) / 20000).astype(np.float32)
    a = np.concatenate([np.repeat(grid[:, None], 6, axis=1), rng.uniform(-3, 3, (20000, 6)).astype(np.float32)])
    d_np = action_to_delta_np(a, md)
    d_t = action_to_delta_torch(torch.as_tensor(a), torch.tensor(np.asarray(md, np.float64))).numpy()
    np.testing.assert_array_equal(d_np, d_t)
    # the float64 product matters: the old float32 rule disagrees on some of these values
    old = np.rint(a * np.asarray(md, np.float32))
    assert (old != d_np).any()


def test_real_env_applies_the_mapping_for_special_values(cfg, calibration):
    """±inf / 1e3 are clipped to ±1 (no trip), halves round to even, NaN trips (the CPU half of the GPU parity)."""
    for row, want in zip(ACTIONS, EXPECTED, strict=True):
        env = So101GoalEnv(SimBackend(cfg, calibration), cfg, REACH)
        try:
            env.reset(seed=0, options={"goal": np.array([0.30, 0.15, 0.25])})  # far: no success
            _o, _r, terminated, truncated, info = env.step(row)
            if np.isnan(row).any():
                assert info["trips"] & TripCause.NAN and truncated and not terminated
                assert not info["applied_delta_ticks"].any()
            else:
                assert info["trips"] == 0 and not truncated
                np.testing.assert_array_equal(info["applied_delta_ticks"], want)
        finally:
            env.close()


# ---------------------------------------------------------------------------------------------------- GPU
gpu = pytest.mark.gpu


@pytest.fixture(scope="module")
def scene():
    from so101_rl.sim.model import load_model

    return load_model()


def _batched(scene, cfg, cal, task, n):
    from so101_rl.env.batched_env import BatchedGoalEnv, model_sim_params

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    env = BatchedGoalEnv(cfg, cal, task, n, sim_params=model_sim_params(scene[0]), model=scene[0], seed=0)
    env.start_noise_ticks = 0  # exact home, like the real env's start_pose "home"
    return env


def _real(cfg, cal, task):
    return So101GoalEnv(SimBackend(cfg, cal), cfg, task)


@gpu
def test_both_envs_apply_the_same_deltas_and_trips(scene, cfg, calibration):
    env = _batched(scene, cfg, calibration, REACH, len(ACTIONS))
    try:
        env.reset(phase_randomise=False)
        env.goal.fill_(5.0)  # unreachable: no success, no auto-reset
        st = env.step(torch.as_tensor(ACTIONS, device=env.device))
        b_applied = st.info["applied_delta_ticks"].cpu().numpy()
        b_trips = st.info["trips"].cpu().numpy()
        b_trunc = st.truncated.cpu().numpy()
    finally:
        env.close()
    for i, row in enumerate(ACTIONS):
        real = _real(cfg, calibration, REACH)
        try:
            real.reset(seed=0, options={"goal": np.array([0.30, 0.15, 0.25])})
            _o, _r, _te, truncated, info = real.step(row)
        finally:
            real.close()
        np.testing.assert_array_equal(b_applied[i], info["applied_delta_ticks"], err_msg=f"row {i}")
        assert bool(b_trips[i] & int(TripCause.NAN)) == bool(info["trips"] & TripCause.NAN), i
        assert bool(b_trunc[i]) == truncated, i


def _real_episode_end(cfg, cal, task, action) -> tuple[bool, bool, bool]:
    env = _real(cfg, cal, task)
    try:
        obs, _ = env.reset(seed=0)
        env.reset(seed=0, options={"goal": obs["achieved_goal"].astype(np.float64)})  # predicate holds at once
        _o, _r, terminated, truncated, info = env.step(action)
        return bool(terminated), bool(truncated), bool(info["is_success"])
    finally:
        env.close()


def _batched_episode_end(scene, cfg, cal, task, action) -> tuple[bool, bool, bool]:
    env = _batched(scene, cfg, cal, task, 2)
    try:
        _obs, _priv, achieved, _des = env.reset(phase_randomise=False)
        env.goal.copy_(achieved)  # predicate holds at once
        st = env.step(torch.as_tensor(np.tile(action, (2, 1)), device=env.device))
        return bool(st.terminated[0]), bool(st.truncated[0]), bool(st.info["success"][0])
    finally:
        env.close()


@gpu
def test_trip_on_success_is_truncated_not_terminated_in_both_envs(scene, cfg, calibration):
    """The tip sits on the goal (predicate true) but the step trips (NaN): no success in either env."""
    action = np.array([np.nan, 0, 0, 0, 0, 0], np.float32)
    real = _real_episode_end(cfg, calibration, REACH, action)
    batched = _batched_episode_end(scene, cfg, calibration, REACH, action)
    assert real == batched == (False, True, False), (real, batched)


@gpu
def test_success_on_the_step_limit_is_terminated_and_truncated_in_both_envs(scene, cfg, calibration):
    task = dataclasses.replace(REACH, episode_steps=1)
    action = np.zeros(6, np.float32)
    real = _real_episode_end(cfg, calibration, task, action)
    batched = _batched_episode_end(scene, cfg, calibration, task, action)
    assert real == batched == (True, True, True), (real, batched)


@gpu
def test_success_without_trip_or_limit_is_terminated_only_in_both_envs(scene, cfg, calibration):
    action = np.zeros(6, np.float32)
    real = _real_episode_end(cfg, calibration, REACH, action)
    batched = _batched_episode_end(scene, cfg, calibration, REACH, action)
    assert real == batched == (True, False, True), (real, batched)
