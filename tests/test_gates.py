"""L1/L2 reach gates (MVP M3, sim plan S4.2 reach part), each with a shown-to-fail twin."""

from pathlib import Path

import numpy as np
import pytest

from so101_rl.env.tasks import REACH, GoalBounds
from so101_rl.rl import gates

ROOT = Path(__file__).parent.parent
CAL = str(ROOT / "tests/data/so101_follower.calib.json")


@pytest.fixture(scope="module")
def rig():
    return gates._setup(CAL)


def _fk_goals(rig, spread):
    """Goals that are FK images of joint vectors near home (well-posed: reachable, short travel)."""
    from so101_rl.env.goals import fk_tip_batch

    cfg, cal, fk = rig
    home = gates._home_ticks(cal)

    def sample(n, rng):
        off = rng.integers(-spread, spread + 1, size=(n, 6))
        off[:, 4:] = 0
        return fk_tip_batch(fk, np.clip(home + off, cfg.joints.min_ticks, cfg.joints.max_ticks))

    return sample


def _l1(rig, sample, **kw):
    cfg, cal, fk = rig
    rows = gates.l1_reach_cpu(cfg, cal, fk, sample_goals=sample, success_radius_m=REACH.success_radius_m,
                              episode_steps=REACH.episode_steps, n_joint=40_000, n_goal=300, **kw)  # fmt: skip
    return {r.name: r for r in rows}


def test_l1_reach_passes_on_a_well_posed_distribution(rig):
    rows = _l1(rig, _fk_goals(rig, 400))
    assert rows["fk_coverage"].passed, rows["fk_coverage"]
    assert rows["travel_budget"].passed, rows["travel_budget"]
    assert rows["start_not_success"].passed, rows["start_not_success"]


def test_l1_reach_fails_with_radius_040(rig):
    """Shown to fail: goals up to 0.40 m without the reachability filter are not all reachable."""
    bad = GoalBounds(radius_max_m=0.40, box=REACH.goal_bounds.box, margin_m=0.0, z_min_m=0.0)
    rows = _l1(rig, bad.sample_np)
    assert not rows["fk_coverage"].passed


def test_l1_travel_budget_fails_with_a_short_horizon(rig):
    cfg, cal, fk = rig
    rows = gates.l1_reach_cpu(cfg, cal, fk, sample_goals=_fk_goals(rig, 400), success_radius_m=REACH.success_radius_m,
                              episode_steps=5, n_joint=20_000, n_goal=100)  # fmt: skip
    assert not {r.name: r for r in rows}["travel_budget"].passed


def test_l1_start_not_success_fails_without_the_exclusion(rig, monkeypatch):
    """Shown to fail: goals at the start pose are trivial episodes unless the env's near-start rule removes them."""
    monkeypatch.setattr(gates, "exclude_near_start", lambda goals, alt, tip, d: goals)
    rows = _l1(rig, _fk_goals(rig, 3))
    assert not rows["start_not_success"].passed


@pytest.mark.parametrize("case", ["real", "broken_radius", "nan_accepting"])
def test_l2_adversarial_table(case):
    pred = {
        "real": None,
        "broken_radius": lambda a, d: np.linalg.norm(a - d, axis=-1) <= 0.03,
        "nan_accepting": lambda a, d: ~(np.linalg.norm(a - d, axis=-1) > REACH.success_radius_m),
    }[case]
    rows = {r.name: r for r in gates.l2_reach_cpu(REACH, predicate=pred, n=20_000)}
    if case == "real":
        assert rows["adversarial"].passed and rows["independent_predicate"].passed
    else:
        assert not rows["adversarial"].passed


def test_l2_independent_predicate_agrees_and_its_twin_fails():
    rows = {r.name: r for r in gates.l2_reach_cpu(REACH, n=100_000)}
    assert rows["independent_predicate"].passed
    rows = {r.name: r for r in gates.l2_reach_cpu(REACH, predicate=lambda a, d: np.linalg.norm(a - d, axis=-1) <= 0.0151,
                                                   n=100_000)}  # fmt: skip
    assert not rows["independent_predicate"].passed


def test_independent_predicate_nan():
    assert not gates.independent_reach_success(np.full((1, 3), np.nan), np.zeros((1, 3)), 0.015)[0]


def test_cli_refuses_die_task():
    assert gates.main(["--task", "die_to_zone"]) == 2


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_scripted_reach_do_nothing_random_gpu():
    rows = {r.name: r for r in gates.l_reach_gpu(512, calibration=CAL)}
    assert rows["scripted_reach"].passed, rows["scripted_reach"]
    assert rows["do_nothing"].passed, rows["do_nothing"]
    assert rows["random_policy"].passed, rows["random_policy"]
