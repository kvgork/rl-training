"""L1/L2 reach and die_to_zone gates (MVP M3, sim plan S4.2; die: plans/die-to-zone-plan.md D1), each with a
shown-to-fail twin."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from so101_rl.env.tasks import DIE_TO_ZONE, REACH, GoalBounds
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
        # inside the gate's IK limits (2 % shrunk): a goal only reachable at a joint stop is not "well posed"
        lo, hi = np.asarray(cfg.joints.min_ticks), np.asarray(cfg.joints.max_ticks)
        m = ((hi - lo) * 0.02).astype(np.int64)
        return fk_tip_batch(fk, np.clip(home + off, lo + m, hi - m))

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


def test_cli_refuses_unknown_task():
    assert gates.main(["--task", "pick_and_place"]) == 2


# ------------------------------------------------------------------------------------------------ die_to_zone (D1)
def _zone_at(xy):
    """DIE_TO_ZONE with the zone moved to ``xy`` (the shown-to-fail twin of the start / do-nothing gates)."""
    return dataclasses.replace(DIE_TO_ZONE, zone=dataclasses.replace(DIE_TO_ZONE.zone, center_xy_m=xy))


def test_l1_die_start_not_success_and_its_twin():
    assert gates.l1_die_cpu(DIE_TO_ZONE, n=2000)[0].passed
    # shown to fail: a zone whose edge reaches into the start well holds some starts (the worst case finds it)
    well = np.asarray(DIE_TO_ZONE.die_start.well_center_xy_m)
    row = gates.l1_die_cpu(_zone_at(tuple(well + [DIE_TO_ZONE.zone.half_edge_m + 0.005, 0.0])), n=2000)[0]
    assert not row.passed, row


def test_l2_die_passes_with_the_task_predicate():
    rows = {r.name: r for r in gates.l2_die_cpu(DIE_TO_ZONE, n=20_000)}
    assert set(rows) == {"adversarial", "independent_predicate", "independent_settle"}
    assert all(r.passed for r in rows.values()), rows


@pytest.mark.parametrize("case", ["euclidean", "ignores_visibility", "no_height_bound", "nan_accepting"])
def test_l2_die_adversarial_twins(case):
    h, mh = DIE_TO_ZONE.zone.half_edge_m, DIE_TO_ZONE.zone.max_height_m

    def euclid(a, d, v):
        return (np.linalg.norm(a[..., :2] - d[..., :2], axis=-1) <= h) & (a[..., 2] - d[..., 2] <= mh) & v

    def square(a, d):
        return np.all(np.abs(a[..., :2] - d[..., :2]) <= h, axis=-1)

    pred = {
        "euclidean": euclid,  # a disc, not the zone square: the inside corner fails
        "ignores_visibility": lambda a, d, v: square(a, d) & (a[..., 2] - d[..., 2] <= mh),
        "no_height_bound": lambda a, d, v: square(a, d) & v,  # a die held 20 mm up would count
        "nan_accepting": lambda a, d, v: ~np.any(np.abs(a[..., :2] - d[..., :2]) > h, axis=-1) & v,
    }[case]
    rows = {r.name: r for r in gates.l2_die_cpu(DIE_TO_ZONE, predicate=pred, n=20_000)}
    assert not rows["adversarial"].passed, rows["adversarial"]


def test_l2_die_independent_twins():
    """Shown to fail: a 0.1 mm wider square fails ``independent_predicate``; a debounce that ignores the settle rule
    fails ``independent_settle``."""
    from so101_rl.env.tasks import success_np

    wide = dataclasses.replace(DIE_TO_ZONE, zone=dataclasses.replace(DIE_TO_ZONE.zone, half_edge_m=0.0301))
    rows = {r.name: r for r in gates.l2_die_cpu(DIE_TO_ZONE, predicate=lambda a, d, v: success_np(a, d, wide, visible=v),
                                                 n=100_000)}  # fmt: skip
    assert not rows["independent_predicate"].passed

    def no_settle(p, c, ach, prev):
        new = np.where(p, c + 1, 0)
        return new >= DIE_TO_ZONE.success_debounce_steps, new

    rows = {r.name: r for r in gates.l2_die_cpu(DIE_TO_ZONE, debounce=no_settle, n=1000)}
    assert not rows["independent_settle"].passed and rows["independent_predicate"].passed


def test_independent_die_predicate_nan_and_visibility():
    g = DIE_TO_ZONE.zone.goal_xyz()[None]
    h, mh = DIE_TO_ZONE.zone.half_edge_m, DIE_TO_ZONE.zone.max_height_m
    assert gates.independent_die_success(g, g, np.array([True]), h, mh)[0]
    assert not gates.independent_die_success(g, g, np.array([False]), h, mh)[0]
    assert not gates.independent_die_success(np.full((1, 3), np.nan), g, np.array([True]), h, mh)[0]
    assert not gates.independent_die_success(g, np.full((1, 3), np.nan), np.array([True]), h, mh)[0]


def test_die_do_nothing_cpu_and_its_twin(rig):
    cfg, cal, _ = rig
    row = gates.l2_die_do_nothing_cpu(cfg, cal, DIE_TO_ZONE, nworld=2)[0]
    assert row.passed, row
    # shown to fail: with the zone on the start well, the resting die succeeds without any action
    row = gates.l2_die_do_nothing_cpu(cfg, cal, _zone_at(DIE_TO_ZONE.die_start.well_center_xy_m), nworld=2)[0]
    assert not row.passed, row


def test_cli_die_runs_on_cpu(tmp_path, monkeypatch):
    """``sim-gates --task die_to_zone`` on CPU: every CPU row passes; the batched-env rows FAIL as not run (no CUDA),
    so the CLI exits 1, exactly like reach without a GPU."""
    import json

    monkeypatch.setattr(gates.torch.cuda, "is_available", lambda: False)
    out = tmp_path / "die.json"
    assert gates.main(["--task", "die_to_zone", "--calibration", CAL, "--nworld", "2", "--out", str(out)]) == 1
    doc = json.loads(out.read_text())
    rows = {r["name"]: r for r in doc["rows"]}
    assert doc["task"] == "die_to_zone" and not doc["passed"]
    cpu = ("start_not_success", "adversarial", "independent_predicate", "independent_settle", "do_nothing")
    assert all(rows[n]["passed"] for n in cpu), rows
    assert not rows["do_nothing_batched"]["passed"] and "CUDA unavailable" in rows["do_nothing_batched"]["value"]
    assert not rows["random_policy"]["passed"]


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_die_do_nothing_random_gpu():
    rows = {r.name: r for r in gates.l_die_gpu(256, calibration=CAL)}
    assert rows["do_nothing_batched"].passed, rows["do_nothing_batched"]
    assert rows["random_policy"].passed, rows["random_policy"]


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_scripted_reach_do_nothing_random_gpu():
    rows = {r.name: r for r in gates.l_reach_gpu(512, calibration=CAL)}
    assert rows["scripted_reach"].passed, rows["scripted_reach"]
    assert rows["do_nothing"].passed, rows["do_nothing"]
    assert rows["random_policy"].passed, rows["random_policy"]
