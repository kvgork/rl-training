"""Loader checks of the safe-state geometry (review B1 a/b, B5, S10): FK(home) in the box, FK(rest) above the table,
the exact rest <-> home transit replayed tick by tick, and bound_s long enough for the slowest ramp."""

import numpy as np
import pytest
import yaml

from so101_rl.config import DEFAULT_CONFIG_PATH, load_config
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.transit import TransitPlanner, ramp_toward, simulate_transit

KIN = DEFAULT_CONFIG_PATH.with_name("so101_kinematics.yaml")


def _variant(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["kinematics_path"] = str(KIN)
    mutate(raw)
    p = tmp_path / "so101.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def _rejects(tmp_path, calibration, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_config(_variant(tmp_path, mutate), calibration=calibration)


@pytest.fixture
def fk(calibration):
    return So101FK(KinematicChain.from_yaml(KIN), calibration)


def tip(fk, ticks):
    return fk.gripper_xyz_from_ticks(np.asarray(ticks), np.zeros(3))


# ---- the shipped configuration ----------------------------------------------------------------------------------
def test_shipped_home_is_inside_the_box_with_margin(cfg, fk):
    xyz = tip(fk, cfg.safe_state.home_ticks)
    m = cfg.safe_state.home_box_margin_m
    assert m >= 0.010
    assert fk.in_box(xyz, cfg.box, margin=-m)
    np.testing.assert_allclose(xyz, [0.200, 0.0, 0.150], atol=0.002)  # the review's placeholder


def test_shipped_rest_is_the_measured_pose_above_the_table(cfg, fk):
    assert cfg.safe_state.rest_ticks.tolist() == [2050, 2200, 2301, 2745, 2040, 2065]
    xyz = tip(fk, cfg.safe_state.rest_ticks)
    np.testing.assert_allclose(xyz, [0.208, -0.001, -0.012], atol=0.002)
    assert not fk.in_box(xyz, cfg.box)  # rest is below the box floor: it lies on the table
    assert xyz[2] >= cfg.safe_state.table_z_m + cfg.safe_state.table_clearance_m


def test_shipped_flag_stays_unverified():
    assert yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["verified_on_hardware"] is False
    assert "NEEDS H5" in DEFAULT_CONFIG_PATH.read_text()


@pytest.mark.parametrize("direction", ["rest->home", "home->rest"])
def test_shipped_transit_stays_above_the_table_every_tick(cfg, fk, direction):
    planner = TransitPlanner(fk, cfg.safe_state)
    home, rest = cfg.safe_state.home_ticks, cfg.safe_state.rest_ticks
    a, b = (rest, home) if direction == "rest->home" else (home, rest)
    n, z_min, reached = simulate_transit(planner, a, b, 2000)
    assert reached and n > 0
    assert z_min >= planner.floor_z


def test_shipped_bound_covers_limit_to_home_plus_home_to_rest(cfg, fk):
    """B5, recomputed independently of the loader."""
    ss, hz = cfg.safe_state, cfg.rates.servo_hz
    home = np.asarray(ss.home_ticks)
    worst = max(np.abs(cfg.joints.min_ticks - home).max(), np.abs(cfg.joints.max_ticks - home).max())
    n_rest, _, _ = simulate_transit(TransitPlanner(fk, ss), home, ss.rest_ticks, 2000)
    assert ss.bound_s >= worst / (ss.ramp_ticks_per_tick * hz) + n_rest / hz + 1.0


# ---- (a) home / rest pose checks -------------------------------------------------------------------------------
def test_home_outside_the_box_is_rejected(tmp_path, calibration):
    # the old placeholder: q = 0 on every joint, tip at x = 0.39 m
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].update(home_ticks=[2047] * 6), "FK.home_ticks")


def test_home_inside_but_closer_than_the_margin_is_rejected(tmp_path, calibration, fk, cfg):
    home = np.asarray(cfg.safe_state.home_ticks)
    z = tip(fk, home)[2]
    # move the box floor up to 5 mm below the home tip: inside, but not 10 mm inside
    _rejects(tmp_path, calibration, lambda r: r["box"].update(z=[float(z) - 0.005, 0.28]), "inside the box")


def test_rest_below_the_table_floor_is_rejected(tmp_path, calibration):
    def mut(raw):
        raw["safe_state"]["rest_ticks"][1] += 300  # shoulder further down: tip 4 cm into the table

    _rejects(tmp_path, calibration, mut, "FK.rest_ticks. z")


def test_rest_outside_the_box_footprint_is_rejected(tmp_path, calibration):
    def mut(raw):
        raw["safe_state"]["rest_ticks"][0] = 2735  # pan to the calibration limit: the tip swings out in y
        raw["box"]["y"] = [-0.05, 0.05]

    _rejects(tmp_path, calibration, mut, "footprint")


# ---- (b) the exact transit -------------------------------------------------------------------------------------
def test_transit_check_rejects_a_table_that_the_ramp_would_hit(tmp_path, calibration):
    # raise the table to 1 mm below the rest tip: rest itself is still "above", but the clearance is not
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].update(table_z_m=-0.012), "table floor")


def test_planner_prevents_a_dip_that_the_naive_ramp_makes(cfg, fk):
    """The independent per-joint ramp (the old code) dips below a floor that the planner respects."""
    from dataclasses import replace

    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    home = np.full(6, 2047, dtype=np.int32)  # the old placeholder home
    cur = rest.copy()
    out = np.empty_like(cur)
    naive_min = 1.0
    while not (cur == home).all():
        ramp_toward(cur, home, 8, out)
        cur[...] = out
        naive_min = min(naive_min, tip(fk, cur)[2])
    z_rest = tip(fk, rest)[2]
    assert naive_min > z_rest  # the naive ramp from this rest happens not to dip: build a floor it crosses
    floor_cfg = replace(cfg.safe_state, table_z_m=float(z_rest) - 0.002 - 0.005)  # floor 2 mm below rest
    planner = TransitPlanner(fk, floor_cfg)
    _n, z_min, reached = simulate_transit(planner, rest, home, 2000)
    assert reached and z_min >= planner.floor_z


def test_lift_first_inside_the_zone(cfg, fk):
    """From rest, no single tick of the planner lowers the tip while some joint is still lifting it."""
    planner = TransitPlanner(fk, cfg.safe_state)
    cur = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32).copy()
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    nxt = np.empty_like(cur)
    z = tip(fk, cur)[2]
    ticks_in_zone = 0
    while z < planner.zone_top_z:
        ticks_in_zone += 1
        assert ticks_in_zone < 500
        planner.next_target(cur, home, nxt)
        z_next = tip(fk, nxt)[2]
        assert z_next >= z - 1e-9
        cur[...] = nxt
        z = z_next
    assert ticks_in_zone > 5


# ---- B5: bound ---------------------------------------------------------------------------------------------------
def test_five_second_bound_is_rejected_at_eight_ticks_per_tick(tmp_path, calibration):
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].update(bound_s=5.0), "bound_s")


def test_a_faster_ramp_allows_a_shorter_bound(tmp_path, calibration):
    cfg = load_config(
        _variant(tmp_path, lambda r: r["safe_state"].update(bound_s=10.0, ramp_ticks_per_tick=16)), calibration=calibration
    )
    assert cfg.safe_state.bound_s == 10.0


def test_bound_check_uses_the_calibration_limits(tmp_path, calibration):
    """A wider calibration (full 0..4095 on every joint) needs a longer bound; 20 s still covers it."""
    from dataclasses import replace

    wide = {n: replace(c, range_min=0, range_max=4095) for n, c in calibration.items()}
    load_config(calibration=wide)  # shipped 20 s
    _rejects(tmp_path, wide, lambda r: r["safe_state"].update(bound_s=17.0), "bound_s")


# ---- S10 / misc bounds ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, 11, 10**6])
def test_max_trips_per_hour_is_bounded(tmp_path, calibration, bad):
    _rejects(tmp_path, calibration, lambda r: r["thermal"].update(max_trips_per_hour=bad), "max_trips_per_hour")


@pytest.mark.parametrize("bad", [-0.001, 0.05])
def test_present_tolerance_is_bounded(tmp_path, calibration, bad):
    _rejects(tmp_path, calibration, lambda r: r["box"].update(present_tolerance_m=bad), "present_tolerance_m")


@pytest.mark.parametrize("key", ["table_z_m", "table_clearance_m", "transit_zone_m", "home_box_margin_m", "rest_tolerance_ticks"])
def test_new_safe_state_keys_have_no_hidden_default(tmp_path, calibration, key):
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].pop(key), key)


def test_present_tolerance_has_no_hidden_default(tmp_path, calibration):
    _rejects(tmp_path, calibration, lambda r: r["box"].pop("present_tolerance_m"), "present_tolerance_m")


def test_geometry_cache_does_not_mask_a_changed_value(tmp_path, calibration):
    load_config(calibration=calibration)
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].update(bound_s=5.0), "bound_s")
    load_config(calibration=calibration)  # and back
