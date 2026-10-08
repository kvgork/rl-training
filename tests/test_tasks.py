"""FS4 task specs: one reward / success definition for sim and real (sim-first track S0.4; SU0 review fixes:
modular non-positive reward, 16 mm height bound, settled die, plan §10 2026-10-06)."""

import numpy as np
import pytest
import torch

from so101_rl.env.obs import DIE_OBS, REACH_OBS
from so101_rl.env.tasks import (
    DIE_HALF_EDGE_M,
    DIE_TO_ZONE,
    REACH,
    debounce_np,
    debounce_torch,
    reward_np,
    reward_torch,
    success_np,
    success_torch,
)


def test_constants():
    assert (REACH.name, REACH.obs, REACH.episode_steps, REACH.gamma) == ("reach", REACH_OBS, 100, 0.97)
    assert (REACH.success_radius_m, REACH.success_debounce_steps, REACH.settle_m) == (0.015, 1, None)
    assert REACH.zone is None and REACH.die_start is None and REACH.start_pose == "home"
    assert (DIE_TO_ZONE.name, DIE_TO_ZONE.obs, DIE_TO_ZONE.episode_steps, DIE_TO_ZONE.gamma) == (
        "die_to_zone", DIE_OBS, 100, 0.98)
    assert DIE_TO_ZONE.success_debounce_steps == 2 and DIE_TO_ZONE.reset_settle_steps == 0
    # plan §10 2026-10-06: 16 mm above the goal, settled within 3 mm (both from so101.yaml tasks.die_to_zone)
    assert DIE_TO_ZONE.zone.max_height_m == pytest.approx(0.016)
    assert DIE_TO_ZONE.settle_m == pytest.approx(0.003)
    for spec in (REACH, DIE_TO_ZONE):
        d, s, tm = spec.reward.term("distance"), spec.reward.term("success"), spec.reward.term("time")
        assert (d.weight, d.scale_m, d.zero_on_success, s.weight, tm.weight) == (1.0, 0.30, True, 1.0, 0.05)
        assert [t.name for t in spec.reward.terms] == ["distance", "success", "time"]
    assert DIE_TO_ZONE.die_start.well_center_xy_m == (0.20, -0.05) and DIE_TO_ZONE.die_start.sigma_xy_m == 0.005
    assert DIE_TO_ZONE.die_start.measured_on_hardware is False  # NEEDS H9


def test_box_and_zone_match_config_and_scene(cfg):
    from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M, ZONE_EDGE_M, load_model

    for spec in (REACH, DIE_TO_ZONE):
        assert spec.goal_bounds.box == cfg.box
    _, data = load_model()
    zone = DIE_TO_ZONE.zone
    np.testing.assert_allclose(zone.center_xy_m, data.site("zone").xpos[:2], atol=1e-9)
    assert zone.half_edge_m * 2 == pytest.approx(ZONE_EDGE_M)
    assert zone.z_table_m == pytest.approx(TABLE_Z_M) == cfg.safe_state.table_z_m
    assert DIE_HALF_EDGE_M * 2 == pytest.approx(DIE_EDGE_M)
    np.testing.assert_allclose(DIE_TO_ZONE.die_start.well_center_xy_m, data.body("die").xpos[:2], atol=1e-9)


def _cases(rng, n, spec):
    if spec is REACH:
        desired = spec.goal_bounds.sample_np(n, rng)
        achieved = desired + rng.normal(0, 0.02, size=(n, 3))
    else:
        desired = np.tile(spec.zone.goal_xyz(), (n, 1))
        achieved = desired + rng.normal(0, 0.03, size=(n, 3)) * np.array([1, 1, 0.3])
    visible = rng.random(n) < 0.7
    return achieved, desired, visible


@pytest.mark.parametrize("spec", [REACH, DIE_TO_ZONE], ids=lambda s: s.name)
def test_np_torch_reward_identical(spec):
    rng = np.random.default_rng(1)
    a, d, v = _cases(rng, 20000, spec)
    rn = reward_np(a, d, spec, visible=v)
    rt = reward_torch(torch.as_tensor(a), torch.as_tensor(d), spec, visible=torch.as_tensor(v))
    assert rn.dtype == np.float32 and rt.dtype == torch.float32
    assert np.max(np.abs(rn - rt.numpy())) < 1e-6
    sn = success_np(a, d, spec, visible=v)
    st = success_torch(torch.as_tensor(a), torch.as_tensor(d), spec, visible=torch.as_tensor(v))
    np.testing.assert_array_equal(sn, st.numpy())
    assert 0.05 < sn.mean() < 0.95  # both outcomes are exercised


def test_reward_definition():
    """Default reward: -dist/0.30 - 1 while the predicate is false, plus time -0.05 on EVERY step (success included)."""
    d = np.array([0.2, 0.0, 0.1])
    assert reward_np(d + [0.03, 0, 0], d, REACH) == pytest.approx(-0.1 - 1.0 - 0.05)
    assert reward_np(d + [0.01, 0, 0], d, REACH) == pytest.approx(-0.05)  # inside the 15 mm radius
    goal = DIE_TO_ZONE.zone.goal_xyz()
    assert reward_np(goal, goal, DIE_TO_ZONE, visible=np.True_) == pytest.approx(-0.05)
    assert reward_np(goal + [0.02, 0, 0], goal, DIE_TO_ZONE, visible=np.True_) == pytest.approx(-0.05)  # in zone
    assert reward_np(goal + [0.031, 0, 0], goal, DIE_TO_ZONE, visible=np.True_) == pytest.approx(-0.031 / 0.3 - 1 - 0.05)
    # vectorised over a batch (SB3 HER calls compute_reward on [batch, 3])
    assert reward_np(np.tile(d, (7, 1)), np.tile(d, (7, 1)), REACH).shape == (7,)


def test_success_requires_visible():
    """An invisible die inside the zone is UNKNOWN, never SUCCESS; the predicate fails safe."""
    goal = DIE_TO_ZONE.zone.goal_xyz()
    assert success_np(goal, goal, DIE_TO_ZONE, visible=np.True_)
    assert not success_np(goal, goal, DIE_TO_ZONE, visible=np.False_)
    assert not success_torch(torch.as_tensor(goal), torch.as_tensor(goal), DIE_TO_ZONE, visible=torch.tensor(False))
    # an invisible step is an unsuccessful step: the success term is -1 (dense part from the held position)
    assert reward_np(goal, goal, DIE_TO_ZONE, visible=np.False_) == pytest.approx(-1.05)
    with pytest.raises(ValueError, match="visible"):
        success_np(goal, goal, DIE_TO_ZONE)  # forgetting visibility is an error, not "visible"
    with pytest.raises(ValueError, match="visible"):
        reward_torch(torch.as_tensor(goal), torch.as_tensor(goal), DIE_TO_ZONE)


def test_die_zone_is_a_square_on_the_table():
    z = DIE_TO_ZONE.zone
    goal = z.goal_xyz()
    v = np.True_
    inside_corner = goal + [z.half_edge_m - 1e-4, -(z.half_edge_m - 1e-4), 0]
    outside_x = goal + [z.half_edge_m + 1e-3, 0, 0]
    held_above = goal + [0, 0, 0.05]  # die held in the air over the zone (L2 adversarial state)
    cases = {
        "inside_corner": (inside_corner, True),
        "on_x_edge": (goal + [z.half_edge_m - 1e-6, 0, 0], True),  # inclusive edge (minus float32 rounding)
        "outside_x": (outside_x, False),
        "outside_y": (goal + [0, -(z.half_edge_m + 1e-3), 0], False),
        "held_above": (held_above, False),
        # plan §10 2026-10-06: the centroid at most 16 mm above the goal (D435 depth noise); ~2 cm up fails
        "at_15mm": (goal + [0, 0, 0.015], True),
        "at_16mm_minus": (goal + [0, 0, 0.016 - 1e-5], True),
        "at_17mm": (goal + [0, 0, 0.017], False),
        "held_25mm_up": (goal + [0, 0, 0.025], False),
        "pushed_into_table": (goal - [0, 0, 0.003], True),
    }
    pts = np.array([p for p, _ in cases.values()])
    want = np.array([w for _, w in cases.values()])
    vis = np.ones(len(pts), bool)
    np.testing.assert_array_equal(success_np(pts, np.tile(goal, (len(pts), 1)), DIE_TO_ZONE, visible=vis), want)
    st = success_torch(torch.as_tensor(pts, dtype=torch.float32), torch.as_tensor(goal, dtype=torch.float32).expand(len(pts), 3),
                       DIE_TO_ZONE, visible=torch.ones(len(pts), dtype=torch.bool))
    np.testing.assert_array_equal(st.numpy(), want)
    assert success_np(inside_corner, goal, DIE_TO_ZONE, visible=v)


def test_height_bound_is_relative_to_desired():
    """The 16 mm bound is measured from ``desired`` (HER-relabelled goals), not from the table."""
    lifted_goal = DIE_TO_ZONE.zone.goal_xyz() + [0, 0, 0.05]
    pts = lifted_goal + np.array([[0, 0, 0.015], [0, 0, 0.017]])
    np.testing.assert_array_equal(
        success_np(pts, np.tile(lifted_goal, (2, 1)), DIE_TO_ZONE, visible=np.ones(2, bool)), [True, False])


def test_reach_radius_boundary_np_and_torch():
    g = np.array([0.2, 0.0, 0.1])
    pts = g + np.array([[0.0149, 0, 0], [0.0151, 0, 0], [0, 0.0150 - 1e-9, 0]])
    want = [True, False, True]
    np.testing.assert_array_equal(success_np(pts, np.tile(g, (3, 1)), REACH), want)
    np.testing.assert_array_equal(success_torch(torch.as_tensor(pts), torch.as_tensor(g).expand(3, 3), REACH).numpy(), want)


def test_her_relabelled_die_goal_is_consistent():
    """HER relabels desired := an achieved die position (possibly lifted): achieved == desired must succeed when
    the stored transition saw the die, and never when it did not; visibility comes from the stored info."""
    from so101_rl.env.tasks import visible_from_info

    lifted = np.array([[0.15, 0.05, 0.06], [0.25, -0.1, -0.012]])
    infos = [{"tracker_visible": True}, {"tracker_visible": False}]
    vis = visible_from_info(infos, DIE_TO_ZONE)
    np.testing.assert_array_equal(vis, [True, False])
    np.testing.assert_array_equal(success_np(lifted, lifted, DIE_TO_ZONE, visible=vis), [True, False])
    r = reward_np(lifted, lifted, DIE_TO_ZONE, visible=vis)
    np.testing.assert_allclose(r, [-0.05, -1.05])
    assert visible_from_info(infos, REACH) is None
    with pytest.raises(KeyError):
        visible_from_info([{}], DIE_TO_ZONE)  # a missing flag is never "visible"


def test_die_reward_is_3d_distance():
    goal = DIE_TO_ZONE.zone.goal_xyz()
    a = goal + np.array([0.06, 0.08, 0.0])  # 0.1 m away, outside the zone
    assert reward_np(a, goal, DIE_TO_ZONE, visible=np.True_) == pytest.approx(-0.1 / 0.3 - 1.05)
    assert float(reward_torch(torch.as_tensor(a), torch.as_tensor(goal), DIE_TO_ZONE, visible=torch.tensor(True))) == (
        pytest.approx(-0.1 / 0.3 - 1.05))


def test_debounce():
    """Die task needs 2 consecutive true steps; a gap resets the count. Reach succeeds on the first."""
    pred = np.array([True, False, True, True, True, False, True])
    count = np.zeros((), int)
    die, reach = [], []
    rc = np.zeros((), int)
    still = np.zeros(3)  # a die at rest: the settle rule is satisfied
    for p in pred:
        s, count = debounce_np(p, count, DIE_TO_ZONE, achieved=still, prev_achieved=still)
        die.append(bool(s))
        r, rc = debounce_np(p, rc, REACH)
        reach.append(bool(r))
    assert die == [False, False, False, True, True, False, False]
    assert reach == pred.tolist()
    z3 = torch.zeros(3, 3)
    st, ct = debounce_torch(torch.tensor([True, True, False]), torch.tensor([0, 1, 5]), DIE_TO_ZONE,
                            achieved=z3, prev_achieved=z3)
    assert st.tolist() == [False, True, False] and ct.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="settled"):
        debounce_np(np.True_, np.zeros((), int), DIE_TO_ZONE)  # forgetting the positions is an error


def _run_die(track: np.ndarray, visible: np.ndarray | None = None) -> list[bool]:
    """Feed a tracker trajectory through predicate + settle debounce (numpy); returns per-step termination."""
    goal = DIE_TO_ZONE.zone.goal_xyz()
    visible = np.ones(len(track), bool) if visible is None else visible
    count, prev, out = np.zeros((), int), track[0], []
    for p, v in zip(track, visible, strict=True):
        pred = success_np(p, goal, DIE_TO_ZONE, visible=np.bool_(v))
        done, count = debounce_np(pred, count, DIE_TO_ZONE, achieved=p, prev_achieved=prev)
        out.append(bool(done))
        prev = p
    return out


def test_settle_slide_through_at_5cm_per_s_fails():
    """L2: a die sliding through the zone at 0.05 m/s (5 mm per 10 Hz policy step) is in the zone on many
    consecutive visible steps but never settled: no success."""
    goal = DIE_TO_ZONE.zone.goal_xyz()
    xs = np.arange(-0.05, 0.05 + 1e-9, 0.005)
    track = goal + np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    in_zone = success_np(track, np.tile(goal, (len(track), 1)), DIE_TO_ZONE, visible=np.ones(len(track), bool))
    assert in_zone.sum() >= 10  # the predicate alone would have accepted it
    assert not any(_run_die(track))


def test_settle_nudge_out_and_jitter():
    goal = DIE_TO_ZONE.zone.goal_xyz()
    e = DIE_TO_ZONE.zone.half_edge_m
    # in the zone one step, nudged out the next: never 2 consecutive in-zone steps
    assert not any(_run_die(np.array([goal + [e - 0.001, 0, 0], goal + [e + 0.002, 0, 0], goal + [e + 0.004, 0, 0]])))
    # nudged by 4 mm inside the zone: the run restarts, success only once it stays put
    track = np.array([goal, goal + [0.004, 0, 0], goal + [0.004, 0, 0]])
    assert _run_die(track) == [False, False, True]
    # 2.9 mm of tracker jitter is settled; 3.1 mm is not
    assert _run_die(np.array([goal, goal + [0.0029, 0, 0]])) == [False, True]
    assert _run_die(np.array([goal, goal + [0.0031, 0, 0]])) == [False, False]
    # invisible on the second step: no success even though the held position did not move
    assert not any(_run_die(np.array([goal, goal]), visible=np.array([True, False])))


def test_settle_debounce_np_torch_agree():
    rng = np.random.default_rng(5)
    n, t = 512, 30
    goal = DIE_TO_ZONE.zone.goal_xyz()
    steps = rng.normal(0, 0.003, size=(t, n, 3)) * np.array([1, 1, 0.2])
    track = goal + np.cumsum(steps, axis=0) * 0.5
    vis = rng.random((t, n)) < 0.85
    cn, ct = np.zeros(n, int), torch.zeros(n, dtype=torch.int64)
    prev = track[0]
    total = 0
    for k in range(t):
        pn = success_np(track[k], np.tile(goal, (n, 1)), DIE_TO_ZONE, visible=vis[k])
        sn, cn = debounce_np(pn, cn, DIE_TO_ZONE, achieved=track[k], prev_achieved=prev)
        pt = success_torch(torch.as_tensor(track[k]), torch.as_tensor(goal).expand(n, 3), DIE_TO_ZONE,
                           visible=torch.as_tensor(vis[k]))
        st, ct = debounce_torch(pt, ct, DIE_TO_ZONE, achieved=torch.as_tensor(track[k]),
                                prev_achieved=torch.as_tensor(prev))
        np.testing.assert_array_equal(sn, st.numpy())
        np.testing.assert_array_equal(cn, ct.numpy())
        total += int(sn.sum())
        prev = track[k]
    assert total > 0  # both outcomes exercised


def test_goal_bounds_inside_box_and_radius(cfg):
    gb = REACH.goal_bounds
    g = gb.sample_np(100_000, np.random.default_rng(2))
    assert g.shape == (100_000, 3)
    m = gb.margin_m
    assert np.all(g[:, 0] >= cfg.box.x[0] + m) and np.all(g[:, 0] <= cfg.box.x[1] - m)
    assert np.all(g[:, 1] >= cfg.box.y[0] + m) and np.all(g[:, 1] <= cfg.box.y[1] - m)
    assert np.all(g[:, 2] >= max(cfg.box.z[0] + m, gb.z_min_m)) and np.all(g[:, 2] <= cfg.box.z[1] - m)
    assert np.all(np.linalg.norm(g, axis=1) <= gb.radius_max_m)
    assert gb.contains_np(g).all()
    # the bounds are not degenerate: samples spread over the region
    assert np.ptp(g[:, 0]) > 0.15 and np.ptp(g[:, 1]) > 0.3 and np.ptp(g[:, 2]) > 0.15
    assert not gb.contains_np(np.array([0.33, 0.0, 0.1]))  # inside the box, but within the margin of x = 0.34
