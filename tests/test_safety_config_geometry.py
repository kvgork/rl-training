"""Safe-state geometry checks (review B1 a/b, B5, S10), now in ``safety.config_check.validate_geometry``: FK(home) in
the box, FK(rest) above the table, the exact rest <-> home transit replayed tick by tick, and bound_s long enough for
the slowest ramp. ``load_config`` itself only does the pure range/consistency checks."""

from dataclasses import replace

import numpy as np
import pytest
import yaml

from so101_rl.config import DEFAULT_CONFIG_PATH, load_config
from so101_rl.safety.config_check import geometry_problems, validate_geometry
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
    """load_config accepts the variant (it is pure) and validate_geometry rejects it; or load_config rejects it."""
    with pytest.raises(ValueError, match=match):
        cfg = load_config(_variant(tmp_path, mutate), calibration=calibration)
        validate_geometry(cfg, calibration)


def _problems(tmp_path, calibration, mutate):
    return geometry_problems(load_config(_variant(tmp_path, mutate), calibration=calibration), calibration)


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
    # plan §10 2026-10-10: rest measured on the arm (joint-probe, torque off) after the zero adjust; the table
    # measured by camera-calibrate at z = -0.0045 m. The folded rest pose holds the tip ~26 mm above the table.
    assert cfg.safe_state.rest_ticks.tolist() == [2151, 828, 3071, 2854, 2203, 2010]
    xyz = tip(fk, cfg.safe_state.rest_ticks)
    np.testing.assert_allclose(xyz, [0.167, -0.022, 0.022], atol=0.002)
    assert cfg.box.z[0] == pytest.approx(cfg.safe_state.table_z_m + 0.005) == pytest.approx(0.0005)
    assert fk.in_box(xyz, cfg.box)
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
    # move the box top down to 5 mm above the home tip: inside, but not 10 mm inside
    _rejects(tmp_path, calibration, lambda r: r["box"].update(z_max_m=float(z) + 0.005), "inside the box")


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
def test_transit_stall_is_rejected_on_its_own(tmp_path, calibration):
    """Synthetic rest pose from which the planner has to hold every joint on the way up to home. Rest is above the
    floor and inside the footprint, so only the stall check can reject this (exactly one problem is reported)."""

    def mut(raw):
        raw["safe_state"]["rest_ticks"] = [2050, 2264, 2481, 2993, 2040, 2065]
        raw["safe_state"]["table_z_m"] = 0.0025
        raw["box"]["floor_above_table_m"] = 0.0075  # box floor z[0] = table + this = 0.010
        raw["safe_state"]["start_min_tip_z_m"] = 0.0125  # >= table + 1 cm (loader rule)

    probs = _problems(tmp_path, calibration, mut)
    assert len(probs) == 1
    assert "rest -> home transit stalls" in probs[0]
    _rejects(tmp_path, calibration, mut, "transit stalls")


def test_transit_dip_is_rejected(tmp_path, calibration):
    """Raise the table 5 mm above the rest tip: the replay starts below the floor (plain transit would sit there /
    dip), and says so with the dip message, not only the rest-floor one. (The box floor follows the table.)"""

    def mut(raw):
        raw["safe_state"]["table_z_m"] = 0.02  # rest tip z ~0.022; the box floor follows: table + 5 mm = 0.025
        raw["safe_state"]["start_min_tip_z_m"] = 0.03  # the loader keeps the start floor >= table + 1 cm

    probs = _problems(tmp_path, calibration, mut)
    assert any("rest -> home transit dips" in p for p in probs)
    assert any("FK(rest_ticks) z" in p for p in probs)  # the rest-floor check fires too, but is a different message
    _rejects(tmp_path, calibration, mut, "transit dips")


# Synthetic pair for which the plain per-joint ramp rest -> home crosses a floor that the rest pose is above.
DIP_REST = [2050, 1951, 2017, 2964, 2040, 2065]
DIP_TABLE_Z = 0.08446094929348767  # floor = this + 5 mm clearance; the plain ramp bottoms out 2 mm below it


def test_planner_prevents_a_dip_that_the_naive_ramp_makes(cfg, fk):
    rest = np.asarray(DIP_REST, dtype=np.int32)
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    ss = replace(cfg.safe_state, rest_ticks=rest, table_z_m=DIP_TABLE_Z)
    planner = TransitPlanner(fk, ss)
    assert tip(fk, rest)[2] > planner.floor_z  # rest itself is above the floor
    cur, out, naive_min = rest.copy(), np.empty_like(rest), 1.0
    while not (cur == home).all():
        ramp_toward(cur, home, ss.ramp_ticks_per_tick, out)
        cur[...] = out
        naive_min = min(naive_min, tip(fk, cur)[2])
    assert naive_min < planner.floor_z - 1e-3  # non-vacuous: the independent ramp DOES dip
    for a, b in ((rest, home), (home, rest)):
        _n, z_min, reached = simulate_transit(planner, a, b, 2000)
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
    validate_geometry(cfg, calibration)


def test_bound_check_uses_the_calibration_limits(tmp_path, calibration):
    """A wider calibration (full 0..4095 on every joint) needs a longer bound; 20 s still covers it."""
    wide = {n: replace(c, range_min=0, range_max=4095) for n, c in calibration.items()}
    validate_geometry(load_config(calibration=wide), wide)  # shipped 20 s
    _rejects(tmp_path, wide, lambda r: r["safe_state"].update(bound_s=15.0), "bound_s")


# ---- S10 / misc bounds ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, 11, 10**6])
def test_max_trips_per_hour_is_bounded(tmp_path, calibration, bad):
    _rejects(tmp_path, calibration, lambda r: r["thermal"].update(max_trips_per_hour=bad), "max_trips_per_hour")


@pytest.mark.parametrize("bad", [-0.001, 0.05])
def test_present_tolerance_is_bounded(tmp_path, calibration, bad):
    _rejects(tmp_path, calibration, lambda r: r["box"].update(present_tolerance_m=bad), "present_tolerance_m")


@pytest.mark.parametrize(
    "key",
    ["table_z_m", "table_clearance_m", "transit_zone_m", "home_box_margin_m", "rest_tolerance_ticks", "start_min_tip_z_m"],
)
def test_new_safe_state_keys_have_no_hidden_default(tmp_path, calibration, key):
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].pop(key), key)


@pytest.mark.parametrize("bad", [-0.0101, -0.015, -0.03, 0.28, 0.5, "0.01", float("nan")])
def test_start_min_tip_z_is_range_checked(tmp_path, calibration, bad):
    """Less than 1 cm above the table (z < -0.010; e.g. the running box floor -0.015), at/above the box top, or not
    a finite number: refused."""
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].update(start_min_tip_z_m=bad), "start_min_tip_z_m")


def test_start_min_tip_z_is_loaded(cfg):
    assert cfg.safe_state.start_min_tip_z_m == pytest.approx(0.010)


def test_present_tolerance_has_no_hidden_default(tmp_path, calibration):
    _rejects(tmp_path, calibration, lambda r: r["box"].pop("present_tolerance_m"), "present_tolerance_m")


def test_geometry_cache_does_not_mask_a_changed_value(tmp_path, calibration):
    validate_geometry(load_config(calibration=calibration), calibration)
    _rejects(tmp_path, calibration, lambda r: r["safe_state"].update(bound_s=5.0), "bound_s")
    validate_geometry(load_config(calibration=calibration), calibration)  # and back


def test_shipped_config_passes_validate_geometry(cfg, calibration):
    validate_geometry(cfg, calibration)
    validate_geometry(cfg)  # calibration read from cfg.calibration_path (drive_mode only)


def test_explicit_chain_bypasses_the_cache(cfg, calibration):
    validate_geometry(cfg, calibration, KinematicChain.from_yaml(cfg.kinematics_path))


# ---- table_z_m sanity range, in the (pure) loader --------------------------------------------------------------
@pytest.mark.parametrize("bad", [-0.25, 0.03, 0.05])
def test_table_z_is_range_checked_by_the_loader(tmp_path, calibration, bad):
    with pytest.raises(ValueError, match="table_z_m"):
        load_config(_variant(tmp_path, lambda r: r["safe_state"].update(table_z_m=bad)), calibration=calibration)


@pytest.mark.parametrize("bad", [0.0, -0.001, 0.06])
def test_box_floor_offset_is_range_checked_by_the_loader(tmp_path, calibration, bad):
    with pytest.raises(ValueError, match="floor_above_table_m"):
        load_config(_variant(tmp_path, lambda r: r["box"].update(floor_above_table_m=bad)), calibration=calibration)


def test_both_box_floors_follow_the_table(tmp_path, calibration):
    """safe_state.table_z_m is the one table height: the tip floor (table + 5 mm) and the gripper floor (table + 1 mm)
    are derived from it, never written twice (the recalibration what-if of 2026-10-10: table at -5 mm)."""
    cfg = load_config(_variant(tmp_path, lambda r: r["safe_state"].update(table_z_m=-0.005)), calibration=calibration)
    assert cfg.box.z == (0.0, 0.28)
    assert cfg.box.gripper_floor_z_m == pytest.approx(-0.004)


# ---- dependency direction: config is a leaf ----------------------------------------------------------------------
def test_config_package_does_not_import_safety():
    import ast
    from pathlib import Path

    import so101_rl.config as config_pkg

    offenders = []
    for f in Path(config_pkg.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [("." * node.level) + (node.module or "")] + [a.name for a in node.names]
            if any("safety" in n.split(".") or n.endswith("safety") for n in names):
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, offenders
