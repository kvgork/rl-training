"""Goal samplers (FS12 ``BoxGoalSampler``; sim plan R1.2): the real env's goals come from the SAME ``GoalBounds``
the batched env samples from (``REACH.goal_bounds``), plus an FK-reachability filter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from so101_rl.env.goals import BoxGoalSampler, FixedGoalSampler, GoalSampler, fk_tip_batch
from so101_rl.env.tasks import REACH, GoalBounds
from so101_rl.safety.fk import KinematicChain, So101FK


@pytest.fixture
def fk(cfg, calibration):
    return So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)


@pytest.fixture
def sampler(cfg, fk):
    return BoxGoalSampler(REACH.goal_bounds, fk=fk, limits=cfg.joints, seed=0)


def _inside_torch(bounds: GoalBounds, xyz: np.ndarray) -> torch.Tensor:
    """The bounds evaluated the way a torch sampler does (lo/hi from the same GoalBounds, radius ball)."""
    x = torch.as_tensor(xyz, dtype=torch.float64)
    lo = torch.as_tensor(bounds.lo)
    hi = torch.as_tensor(bounds.hi)
    return ((x >= lo) & (x <= hi)).all(-1) & (x.norm(dim=-1) <= bounds.radius_max_m)


def test_sampler_bounds_equal_batched_env_bounds(sampler, cfg):
    assert sampler.bounds is REACH.goal_bounds  # one GoalBounds object for the numpy and the torch sampler
    assert REACH.goal_bounds.box == cfg.box  # the bounds are the configured workspace box (shrunk by the margin)
    g = sampler.sample_n(100_000, np.random.default_rng(1))
    assert g.shape == (100_000, 3)
    assert REACH.goal_bounds.contains_np(g).all()
    assert bool(_inside_torch(REACH.goal_bounds, g).all())
    b = REACH.goal_bounds
    assert (g[:, 2] >= b.z_min_m).all() and (np.linalg.norm(g, axis=-1) <= 0.30 + 1e-12).all()
    # the samples fill the bounds (not a degenerate corner): every axis spans most of its range
    span = (g.max(0) - g.min(0)) / (b.hi - b.lo)
    assert (span > 0.8).all()


def test_the_bounds_are_shown_to_reject_outside_points():
    b = REACH.goal_bounds
    outside = np.array([[b.hi[0] + 0.001, 0.0, 0.1], [0.2, 0.0, b.z_min_m - 0.001], [0.29, 0.19, 0.2]])
    assert not b.contains_np(outside).any()
    assert not bool(_inside_torch(b, outside).any())


def test_every_sample_is_fk_reachable(sampler, cfg, fk):
    g = sampler.sample_n(2000, np.random.default_rng(2))
    assert sampler.reachable(g).all()
    # an independent check: each goal lies within one voxel diagonal of some FK point of a joint sample
    rng = np.random.default_rng(3)
    lo, hi = cfg.joints.min_ticks.astype(np.int64), cfg.joints.max_ticks.astype(np.int64)
    pts = fk_tip_batch(fk, rng.integers(lo, hi + 1, size=(400_000, 6)))
    pts = pts[REACH.goal_bounds.contains_np(pts)]
    d = torch.cdist(torch.as_tensor(g[:200]), torch.as_tensor(pts)).min(dim=1).values.numpy()
    assert np.median(d) < 0.01 and d.max() < 0.03


def test_unreachable_voxels_are_rejected(sampler):
    cand = REACH.goal_bounds.sample_np(20_000, np.random.default_rng(4))
    frac = sampler.reachable(cand).mean()
    assert 0.3 < frac < 1.0  # part of the box is out of the arm's reach and is filtered


def test_fk_batch_matches_the_safety_fk(fk, cfg):
    rng = np.random.default_rng(5)
    ticks = rng.integers(cfg.joints.min_ticks, cfg.joints.max_ticks + 1, size=(500, 6))
    out = fk_tip_batch(fk, ticks)
    ref = np.empty(3)
    for t, o in zip(ticks, out, strict=True):
        np.testing.assert_allclose(fk.gripper_xyz_from_ticks(t, ref), o, atol=1e-12)


def test_sampling_is_deterministic_in_the_rng(sampler, cfg, fk):
    a = sampler.sample_n(50, np.random.default_rng(7))
    again = BoxGoalSampler(REACH.goal_bounds, fk=fk, limits=cfg.joints, seed=0)
    np.testing.assert_array_equal(a, again.sample_n(50, np.random.default_rng(7)))
    one = sampler.sample(np.random.default_rng(7))
    assert one.shape == (3,)
    np.testing.assert_array_equal(one, a[0])


def test_fixed_sampler_and_protocol(sampler):
    f = FixedGoalSampler(np.array([0.22, 0.10, -0.012]))
    assert isinstance(f, GoalSampler) and isinstance(sampler, GoalSampler)
    g = f.sample(np.random.default_rng(0))
    g[0] = 9.0  # a copy: the fixed goal is not mutable through the sample
    np.testing.assert_array_equal(f.sample(np.random.default_rng(0)), [0.22, 0.10, -0.012])


# ---------------------------------------------------------------- one sampler for both envs (integration)
def test_torch_front_end_draws_from_the_same_reachable_set(sampler):
    """``sample_torch`` (the batched env's path) and ``sample_n`` (the real env's path) share bounds and keys:
    every torch goal passes the numpy bounds and the numpy reachability lookup, and the two lookups agree bit
    for bit on arbitrary points."""
    gen = torch.Generator().manual_seed(0)
    g = torch.cat([sampler.sample_torch(4096, gen) for _ in range(10)])
    assert g.dtype == torch.float32 and g.shape == (40960, 3)
    gn = g.double().numpy()
    assert REACH.goal_bounds.contains_np(gn).all()
    assert sampler.reachable(gn).all()
    pts = REACH.goal_bounds.sample_np(50_000, np.random.default_rng(11)) + np.random.default_rng(12).normal(
        0, 0.02, (50_000, 3)
    )
    np.testing.assert_array_equal(sampler.reachable_torch(torch.as_tensor(pts)).numpy(), sampler.reachable(pts))
    np.testing.assert_array_equal(sampler.contains_torch(torch.as_tensor(pts)).numpy(), REACH.goal_bounds.contains_np(pts))


def test_torch_and_numpy_goal_distributions_match(sampler):
    """Same set, same (uniform) distribution: per-axis means and spreads of the two front ends agree."""
    gt = sampler.sample_torch(50_000, torch.Generator().manual_seed(1)).double().numpy()
    gn = sampler.sample_n(50_000, np.random.default_rng(1))
    np.testing.assert_allclose(gt.mean(0), gn.mean(0), atol=3e-3)
    np.testing.assert_allclose(gt.std(0), gn.std(0), atol=3e-3)


def test_torch_sampler_falls_back_when_nothing_is_reachable(sampler, monkeypatch):
    from so101_rl.env.goals import FALLBACK_GOAL

    with pytest.raises(ValueError, match="candidates"):
        sampler.sample_torch(8, torch.Generator(), candidates=0)
    monkeypatch.setattr(sampler, "reachable_torch", lambda x: torch.zeros(x.shape[:-1], dtype=torch.bool))
    g = sampler.sample_torch(8, torch.Generator().manual_seed(0))
    np.testing.assert_allclose(g.numpy(), np.tile(FALLBACK_GOAL, (8, 1)), atol=1e-7)
    assert sampler.reachable(np.asarray(FALLBACK_GOAL)[None]).all() and REACH.goal_bounds.contains_np(FALLBACK_GOAL)


def test_the_real_and_batched_envs_share_one_table(sampler, cfg, fk):
    """Both envs construct ``BoxGoalSampler(task.goal_bounds, fk, cfg.joints, seed=0)``: the per-process cache
    hands both the same key array, so they cannot drift apart."""
    again = BoxGoalSampler(REACH.goal_bounds, fk=fk, limits=cfg.joints, seed=0)
    assert again.keys is sampler.keys and not sampler.keys.flags.writeable


def test_torch_sampler_refuses_a_device_mismatch(sampler):
    with pytest.raises(ValueError, match="generator on"):
        sampler.sample_torch(4, torch.Generator(), device=torch.device("meta"))
