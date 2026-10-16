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


# ---- trip penalty (plan §10 2026-10-08 "Trip penalty": a policy-caused trip is terminal with a fixed penalty)
def test_trip_penalty_is_worse_than_any_trip_free_episode():
    import dataclasses

    from so101_rl.env.reward import RewardSpecError, validate_trip_penalty, worst_step_reward

    for t in (REACH, DIE_TO_ZONE):
        worst = worst_step_reward(t.reward, max_dist_m=t.max_dist_m)
        # box diagonal (+ 5 mm BOX_PRESENT tolerance per face) / 0.30 + success 1 + time 0.05
        assert worst == pytest.approx(-(t.max_dist_m / 0.30 + 1.05))
        assert t.trip_penalty < 0 and t.trip_penalty <= t.episode_steps * worst
        # the undiscounted bound also beats the worst DISCOUNTED continuation, for every gamma < 1
        assert t.trip_penalty < worst / (1 - t.gamma)
    assert REACH.max_dist_m == pytest.approx(np.linalg.norm([0.27, 0.45, 0.2895]))  # z: 0.0005..0.28 at the measured table
    with pytest.raises(RewardSpecError, match="not worse"):
        dataclasses.replace(REACH, trip_penalty=-10.0)
    with pytest.raises(RewardSpecError, match="< 0"):
        validate_trip_penalty(REACH.reward, 0.0, episode_steps=1, max_dist_m=0.0)
    with pytest.raises(RewardSpecError, match="< 0"):
        validate_trip_penalty(REACH.reward, float("nan"), episode_steps=1, max_dist_m=0.0)
    # a shorter episode only relaxes the bound; a longer one needs a larger penalty
    dataclasses.replace(REACH, episode_steps=10)
    with pytest.raises(RewardSpecError, match="not worse"):
        dataclasses.replace(REACH, episode_steps=200)


def test_step_reward_penalises_terminal_trips_only_np_and_torch_agree():
    from so101_rl.env.tasks import TERMINAL_TRIPS, step_reward_np, step_reward_torch, tripped_from_info
    from so101_rl.safety.types import TripCause

    assert TERMINAL_TRIPS == int(TripCause.NAN | TripCause.RANGE | TripCause.BOX_PRESENT | TripCause.STUCK)
    d = np.array([0.2, 0.0, 0.1])
    trips = [0, int(TripCause.NAN), int(TripCause.RANGE), int(TripCause.BOX_PRESENT), int(TripCause.STUCK),
             int(TripCause.THERMAL), int(TripCause.JOINT_LIMIT | TripCause.TRACKING),
             int(TripCause.STALE_CMD | TripCause.STUCK)]  # fmt: skip
    ach = np.tile(d, (len(trips), 1))  # every step sits exactly on its goal: the success predicate holds
    infos = [{"trips": t} for t in trips]
    want_trip = np.array([False, True, True, True, True, False, False, True])
    np.testing.assert_array_equal(tripped_from_info(infos), want_trip)
    r = step_reward_np(ach, ach, REACH, info=infos)  # HER: desired = achieved
    np.testing.assert_array_equal(r, np.where(want_trip, np.float32(REACH.trip_penalty), reward_np(ach, ach, REACH)))
    assert r.dtype == np.float32 and (r <= 0).all()
    rt = step_reward_torch(torch.as_tensor(ach), torch.as_tensor(ach), REACH, trips=torch.tensor(trips))
    np.testing.assert_array_equal(rt.numpy(), r)
    assert float(step_reward_np(d, d, REACH, info={"trips": int(TripCause.NAN)})) == REACH.trip_penalty
    with pytest.raises(KeyError):
        step_reward_np(d, d, REACH, info={})  # a missing trip field is never "no trip"


# ------------------------------------------------------------------------------------------------ start curriculum (D2a)
def _tray_outer_xy(data, model):
    """(x_lo, x_hi, y_lo, y_hi) of the tray's outer rim faces, from the scene's axis-aligned rim boxes."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("tray_rim_"):
            c, h = data.geom_xpos[g][:2], model.geom_size[g][:2]
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo[0], hi[0], lo[1], hi[1]


def _rect_gap(a, b):
    """Separation of two xy rectangles (x_lo, x_hi, y_lo, y_hi); <= 0 when they overlap."""
    dx = max(b[0] - a[1], a[0] - b[1])
    dy = max(b[2] - a[3], a[2] - b[3])
    return max(dx, dy)


def test_die_start_stages_s0_to_s3():
    """Plan §10 2026-10-16 "Die start curriculum (attempt 2, D2a)": S0..S2 on the open table, S3 the (unchanged) well,
    the default and exit-gate stage."""
    ds = DIE_TO_ZONE.die_start
    assert ds.n_stages == 4 and ds.well_stage == 3
    assert [ds.stage_name(k) for k in range(4)] == ["table_near", "table_far", "tray_rim", "well"]
    assert (ds.well_center_xy_m, ds.sigma_xy_m, ds.yaw) == ((0.20, -0.05), 0.005, "uniform")
    with pytest.raises(ValueError):
        ds.stage_name(4)


def test_die_start_region_needs_room_for_every_yaw():
    from so101_rl.env.tasks import DIE_MAX_FOOTPRINT_HALF_M, DieStartRegion

    DieStartRegion("ok", ((0.0, 2 * DIE_MAX_FOOTPRINT_HALF_M, 0.0, 0.05),))
    with pytest.raises(ValueError, match="too small"):
        DieStartRegion("narrow", ((0.0, 2 * DIE_MAX_FOOTPRINT_HALF_M - 1e-4, 0.0, 0.05),))
    with pytest.raises(ValueError):
        DieStartRegion("empty", ())


def test_die_start_table_stages_are_on_the_open_table_and_reachable(cfg):
    """Every footprint rectangle: inside the safety box (xy), >= 2 mm clear of the tray's outer rim, >= 10 mm clear
    of the zone square, >= 10 mm clear of the fiducial site (a die on it would hide the marker), its corners 0.12..0.26 m from the base (where the jaws reach a resting die,
    tests/test_die_reach_feasibility.py) at a bearing of at most 35 deg."""
    from so101_rl.sim.model import load_model

    model, data = load_model()
    tray = _tray_outer_xy(data, model)
    assert tray == pytest.approx((0.167, 0.233, -0.083, -0.017), abs=1e-6)
    z = DIE_TO_ZONE.zone
    zone = (z.center_xy_m[0] - z.half_edge_m, z.center_xy_m[0] + z.half_edge_m,
            z.center_xy_m[1] - z.half_edge_m, z.center_xy_m[1] + z.half_edge_m)  # fmt: skip
    box = cfg.box
    for reg in DIE_TO_ZONE.die_start.table_stages:
        for r in reg.rects_xy_m:
            assert box.x[0] <= r[0] < r[1] <= box.x[1] and box.y[0] <= r[2] < r[3] <= box.y[1], (reg.name, r)
            assert _rect_gap(r, tray) >= 0.002 - 1e-9, (reg.name, r)
            assert _rect_gap(r, zone) >= 0.010, (reg.name, r)
            assert _rect_gap(r, _site_xy(data, model, "fiducial")) >= 0.010, (reg.name, r)
            corners = np.array([(x, y) for x in r[:2] for y in r[2:]])
            radius = np.hypot(corners[:, 0], corners[:, 1])
            assert radius.min() >= 0.12 and radius.max() <= 0.26, (reg.name, r)
            assert np.degrees(np.abs(np.arctan2(corners[:, 1], corners[:, 0]))).max() <= 35.0, (reg.name, r)


def _dist_to_zone(xy):
    z = DIE_TO_ZONE.zone
    d = np.clip(np.abs(xy - np.asarray(z.center_xy_m)) - z.half_edge_m, 0.0, None)
    return np.hypot(d[:, 0], d[:, 1])


@pytest.mark.parametrize(("stage", "lo", "hi"), [(0, 0.019, 0.061), (1, 0.075, 0.121), (2, 0.018, 0.116)])
def test_die_start_stage_distances_to_the_zone(stage, lo, hi):
    """Die centre to the zone square: S0 2-6 cm (a straight -y push), S1 7.6-12 cm (the same lane farther out), S2 (at
    the rim) 1.9-11.5 cm. Sampled with the env's rule (the numpy twin ``rl.gates._die_starts``), so the yaw-dependent
    clipping is included."""
    from so101_rl.rl.gates import _die_starts

    xyz, _ = _die_starts(DIE_TO_ZONE, 20_000, np.random.default_rng(0), 0.0115, stage)
    d = _dist_to_zone(xyz[:, :2])
    assert lo <= d.min() and d.max() <= hi, (d.min(), d.max())
    if stage in (0, 1):  # in front of the zone: within its x span, so the push is straight in -y
        z = DIE_TO_ZONE.zone
        assert (np.abs(xyz[:, 0] - z.center_xy_m[0]) <= z.half_edge_m).all()


def test_zone_is_clear_of_the_tray_and_the_fiducial():
    """Plan §10 2026-10-16 "Die zone moved into the camera view": the zone square is >= 17 mm from the tray's outer
    rim and >= 5 mm from the fiducial site."""
    from so101_rl.sim.model import load_model

    model, data = load_model()
    z = DIE_TO_ZONE.zone
    zone = (z.center_xy_m[0] - z.half_edge_m, z.center_xy_m[0] + z.half_edge_m,
            z.center_xy_m[1] - z.half_edge_m, z.center_xy_m[1] + z.half_edge_m)  # fmt: skip
    assert _rect_gap(zone, _tray_outer_xy(data, model)) >= 0.017 - 1e-9
    assert _rect_gap(zone, _site_xy(data, model, "fiducial")) >= 0.005 - 1e-9


def _site_xy(data, model, name):
    c, h = data.site(name).xpos[:2], model.site(name).size[:2]
    return c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1]


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_die_start_positions_are_in_the_camera_view(stage):
    """Every die start is in the calibrated camera's view (``calib/camera.yaml``, the batched env's
    ``SimDieTracker.in_view``: colour + depth image, half-blob border margin), sampled with the env's rule and, for
    the table stages, on a grid over each rectangle shrunk by the smallest footprint half extent (occlusion by the
    arm is not part of this check). The well is in view too; its rim hides the die from the oblique camera."""
    import torch

    from so101_rl.env.tasks import DIE_HALF_EDGE_M
    from so101_rl.perception.camera_calib import load_camera_calib
    from so101_rl.rl.gates import _die_starts
    from so101_rl.sim.sim_tracker import NO_NOISE, CameraModel, SimDieTracker

    xyz, _ = _die_starts(DIE_TO_ZONE, 5000, np.random.default_rng(1), 0.0115, stage)
    if stage < DIE_TO_ZONE.die_start.well_stage:
        h = DIE_HALF_EDGE_M
        for x0, x1, y0, y1 in DIE_TO_ZONE.die_start.table_stages[stage].rects_xy_m:
            gx, gy = np.meshgrid(np.linspace(x0 + h, x1 - h, 9), np.linspace(y0 + h, y1 - h, 9))
            xyz = np.vstack([xyz, np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, xyz[0, 2])])])
    calib = load_camera_calib()
    tr = SimDieTracker(CameraModel.from_calib(calib, "color"), NO_NOISE, len(xyz), device="cpu", seed=0,
                       depth_cam=CameraModel.from_calib(calib, "depth"))  # fmt: skip
    ok = tr.in_view(torch.as_tensor(xyz, dtype=torch.float32)).numpy()
    assert ok.all(), xyz[~ok][:5]


@pytest.mark.parametrize("stage", [0, 1])
def test_s0_s1_starts_are_unoccluded_with_the_arm_at_home(stage):
    """S0 and S1 (the easy push stages) start with the die SEEN: with the arm at its home start pose, the sim tracker's
    occlusion rule (jaws, arm links and the tray, ``SimDieTracker.occluded``) hides none of them. The placeholder
    tray rim shadows x >= 0.15 for y -0.02..-0.125 from the oblique camera; S0 stays at x <= 0.149. Shown to fail: a
    die in that shadow, next to the tray at (0.16, -0.06)."""
    import mujoco
    import torch

    from so101_rl.perception.camera_calib import load_camera_calib
    from so101_rl.rl.gates import _die_starts
    from so101_rl.sim.model import load_model
    from so101_rl.sim.sim_tracker import NO_NOISE, CameraModel, Occluders, SimDieTracker

    model, data = load_model()
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    occ = Occluders(model, "cpu")
    xyz, _ = _die_starts(DIE_TO_ZONE, 500, np.random.default_rng(2), 0.0115, stage)
    xyz = np.vstack([xyz, [0.16, -0.06, xyz[0, 2]]])  # the shown-to-fail point (last)
    n = len(xyz)
    calib = load_camera_calib()
    tr = SimDieTracker(CameraModel.from_calib(calib, "color"), NO_NOISE, n, device="cpu", seed=0,
                       depth_cam=CameraModel.from_calib(calib, "depth"))  # fmt: skip
    bi = [model.body(b).id for b in occ.bodies]
    c, h, rot = occ.compute(torch.as_tensor(data.xpos[bi], dtype=torch.float32)[None].expand(n, -1, -1),
                            torch.as_tensor(data.xquat[bi], dtype=torch.float32)[None].expand(n, -1, -1))  # fmt: skip
    tip = torch.as_tensor(data.site("gripper_tip").xpos, dtype=torch.float32)[None].expand(n, -1)
    hidden = tr.occluded(torch.as_tensor(xyz, dtype=torch.float32), tip, c, h, rot).numpy()
    assert not hidden[:-1].any(), xyz[:-1][hidden[:-1]][:5]
    assert hidden[-1]
