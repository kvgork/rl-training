import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from so101_rl.config import (
    DEFAULT_CONFIG_PATH,
    dump_env,
    is_verified_on_hardware,
    load_calibration,
    load_config,
    rad_to_ticks,
    ticks_to_rad,
)

CALIB = Path(__file__).parent / "data" / "so101_follower.calib.json"


@pytest.fixture
def calib():
    return load_calibration(CALIB)


@pytest.fixture
def cfg(calib):
    return load_config(calibration=calib)


def _variant(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    # the loader FK-checks home/rest, so the copied yaml must still find the kinematic chain
    raw["kinematics_path"] = str(DEFAULT_CONFIG_PATH.with_name("so101_kinematics.yaml"))
    mutate(raw)
    p = tmp_path / "so101.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_defaults_match_plan(cfg):
    assert cfg.action.max_delta_ticks.tolist() == [45, 45, 45, 45, 45, 60]
    assert cfg.action.tracking_error_ticks.tolist() == [120, 120, 120, 120, 120, 400]
    assert cfg.action.max_delta_ticks.dtype == np.int32
    assert cfg.thermal.trip_c == 60.0 and cfg.thermal.resume_c == 50.0
    assert cfg.rates.servo_hz == 30.0 and cfg.rates.stale_cmd_ms == 150.0
    # review B5: 5 s cannot cover limit -> home -> rest at 8 ticks/tick; the loader computes the minimum
    assert cfg.safe_state.bound_s == 20.0 and cfg.safe_state.ramp_ticks_per_tick == 8


def test_frozen_dataclasses_reject_mutation(cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.port = "/dev/null"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.thermal.trip_c = 99.0
    with pytest.raises(ValueError):
        cfg.action.max_delta_ticks[0] = 1  # arrays are read-only too


def test_limits_intersection_narrowest_wins(cfg, calib):
    # wrist_roll calibration is the full turn -> config limits (0..4095) win there
    assert cfg.joints.min_ticks[4] == 0 and cfg.joints.max_ticks[4] == 4095
    # shoulder_pan: calibration is narrower than the config
    assert cfg.joints.min_ticks[0] == calib["shoulder_pan"].range_min == 1426
    assert cfg.joints.max_ticks[0] == calib["shoulder_pan"].range_max == 2735


def test_limits_config_narrower_than_calibration(tmp_path, calib):
    def mut(raw):
        raw["joints"]["min_ticks"] = [1500, 0, 0, 0, 100, 0]
        raw["joints"]["max_ticks"] = [4095, 4095, 4095, 4095, 3000, 4095]

    cfg = load_config(_variant(tmp_path, mut), calibration=calib)
    assert cfg.joints.min_ticks[0] == 1500 and cfg.joints.max_ticks[4] == 3000


def test_empty_intersection_rejected(tmp_path, calib):
    def mut(raw):
        raw["joints"]["min_ticks"] = [3000, 0, 0, 0, 0, 0]  # above shoulder_pan range_max

    with pytest.raises(ValueError, match="intersection"):
        load_config(_variant(tmp_path, mut), calibration=calib)


def test_nan_in_yaml_rejected(tmp_path, calib):
    p = tmp_path / "so101.yaml"
    p.write_text(DEFAULT_CONFIG_PATH.read_text().replace("trip_c: 60.0", "trip_c: .nan"))
    with pytest.raises(ValueError):
        load_config(p, calibration=calib)


def test_trip_above_firmware_limit_rejected(tmp_path, calib):
    with pytest.raises(ValueError, match="trip_c"):
        load_config(_variant(tmp_path, lambda r: r["thermal"].update(trip_c=75)), calibration=calib)


def test_bad_motor_id_rejected(tmp_path):
    raw = json.loads(CALIB.read_text())
    raw["gripper"]["id"] = 9
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="ids"):
        load_calibration(p)


def test_missing_joint_rejected(tmp_path):
    raw = json.loads(CALIB.read_text())
    del raw["wrist_roll"]
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing"):
        load_calibration(p)


def test_ticks_rad_roundtrip_and_center(calib):
    ticks = np.array([1500, 2047, 2600, 1800, 2047, 3000], dtype=np.int32)
    q = ticks_to_rad(ticks, calib)
    assert q.shape == (6,)
    assert q[1] == 0.0
    back = rad_to_ticks(q, calib)
    assert back.dtype == np.int32 and np.array_equal(back, ticks)
    # not clipped
    assert rad_to_ticks(np.full(6, 4.0), calib).max() > 4095


def test_drive_mode_flips_sign(calib):
    flipped = dict(calib)
    flipped["elbow_flex"] = dataclasses.replace(calib["elbow_flex"], drive_mode=1)
    t = np.full(6, 2447)
    assert ticks_to_rad(t, flipped)[2] == -ticks_to_rad(t, calib)[2]


def test_kinematics_path_and_flag(cfg):
    assert cfg.kinematics_path.name == "so101_kinematics.yaml"
    assert cfg.kinematics_path.parent == DEFAULT_CONFIG_PATH.parent
    assert is_verified_on_hardware() is False


def test_home_rest_within_limits(cfg):
    for t in (cfg.safe_state.home_ticks, cfg.safe_state.rest_ticks):
        assert np.all(t >= cfg.joints.min_ticks) and np.all(t <= cfg.joints.max_ticks)


def test_dump_env(cfg, monkeypatch):
    monkeypatch.setenv("SO101_FOO", "1")
    monkeypatch.setenv("REQUIRE_RT", "1")
    monkeypatch.setenv("UNRELATED_VAR", "x")
    d = dump_env(cfg)
    assert d["env"] == {"REQUIRE_RT": "1", "SO101_FOO": "1"}
    json.dumps(d)  # fully serialisable
    assert d["config"]["thermal"]["trip_c"] == 60.0


def test_load_from_calibration_path(tmp_path):
    p = _variant(tmp_path, lambda r: r.update(calibration_path=str(CALIB)))
    cfg = load_config(p)
    assert cfg.calibration_path == CALIB


# ---- strict validation (review fixes) -------------------------------------------------------------


def _rejects(tmp_path, calib, mutate, match=None):
    with pytest.raises((ValueError, TypeError), match=match):
        load_config(_variant(tmp_path, mutate), calibration=calib)


def test_typo_key_rejected(tmp_path, calib):
    def mut(raw):
        raw["action"]["max_delta_tick"] = raw["action"].pop("max_delta_ticks")

    _rejects(tmp_path, calib, mut, "max_delta_tick")


def test_removed_trip_c_rejected(tmp_path, calib):
    _rejects(tmp_path, calib, lambda r: r["thermal"].pop("trip_c"), "trip_c")


@pytest.mark.parametrize(
    "section", ["joints", "action", "safe_state", "box", "thermal", "rates", "registers", "sim_warp", "tasks"]
)
def test_unknown_and_missing_key_in_every_section(tmp_path, calib, section):
    _rejects(tmp_path, calib, lambda r: r[section].update(bogus_key=1), "bogus_key")
    _rejects(tmp_path, calib, lambda r: r[section].pop(next(iter(r[section]))), "missing")


def test_unknown_and_missing_top_level_key(tmp_path, calib):
    _rejects(tmp_path, calib, lambda r: r.update(extra_top=1), "extra_top")
    _rejects(tmp_path, calib, lambda r: r.pop("heartbeat_port"), "heartbeat_port")
    _rejects(tmp_path, calib, lambda r: r.pop("verified_on_hardware"), "verified_on_hardware")


def test_interp_ticks_has_no_hidden_default(tmp_path, calib):
    _rejects(tmp_path, calib, lambda r: r["action"].pop("interp_ticks"), "interp_ticks")


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("safe_state", "bound_s"),
        ("safe_state", "ramp_ticks_per_tick"),
        ("rates", "stale_cmd_ms"),
        ("rates", "servo_hz"),
        ("rates", "policy_hz"),
        ("thermal", "resume_dwell_s"),
        ("thermal", "duty_on_s"),
        ("thermal", "duty_off_s"),
        ("thermal", "stale_temp_s"),
    ],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_scalars_rejected(tmp_path, calib, section, key, bad):
    _rejects(tmp_path, calib, lambda r: r[section].update({key: bad}))


@pytest.mark.parametrize("key", ["max_delta_ticks", "tracking_error_ticks", "max_ticks_per_tick"])
@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_action_vector_any_joint_rejected(tmp_path, calib, key, bad):
    def mut(raw):
        raw["action"][key][3] = bad

    _rejects(tmp_path, calib, mut, key)


def test_velocity_cap_must_reach_max_delta(tmp_path, calib):
    # 5 ticks/tick * (30/10) = 15 < 45
    _rejects(tmp_path, calib, lambda r: r["action"].update(max_ticks_per_tick=[5] * 6), "max_ticks_per_tick")
    # exactly enough is accepted
    load_config(_variant(tmp_path, lambda r: r["action"].update(max_ticks_per_tick=[15, 15, 15, 15, 15, 20])), calibration=calib)


def test_servo_hz_must_be_integer_multiple_of_policy_hz(tmp_path, calib):
    _rejects(tmp_path, calib, lambda r: r["rates"].update(servo_hz=25.0), "multiple")
    _rejects(tmp_path, calib, lambda r: r["rates"].update(servo_hz=5.0), "multiple")


@pytest.mark.parametrize(
    ("reg", "bad"),
    [
        ("p_coefficient", 255),
        ("p_coefficient", -1),
        ("i_coefficient", 300),
        ("d_coefficient", 1000),
        ("maximum_acceleration", 255),
        ("acceleration", -3),
        ("protection_current", 0),
        ("protection_current", 4000),
        ("overload_torque", 101),
        ("overload_torque", 0),
        ("over_current_protection_time", 255),
        ("return_delay_time", 255),
        ("operating_mode", 1),
    ],
)
def test_register_out_of_range_rejected(tmp_path, calib, reg, bad):
    _rejects(tmp_path, calib, lambda r: r["registers"].update({reg: bad}), reg)


@pytest.mark.parametrize("bad", [45.5, "45", True, None])
def test_tick_values_must_be_integral(tmp_path, calib, bad):
    def mut(raw):
        raw["action"]["max_delta_ticks"][0] = bad

    _rejects(tmp_path, calib, mut, "max_delta_ticks")


@pytest.mark.parametrize("bad", [2.5, "3", True])
def test_interp_ticks_must_be_integral(tmp_path, calib, bad):
    _rejects(tmp_path, calib, lambda r: r["action"].update(interp_ticks=bad), "interp_ticks")


def test_integral_float_ticks_accepted(tmp_path, calib):
    cfg = load_config(_variant(tmp_path, lambda r: r["action"].update(interp_ticks=3.0)), calibration=calib)
    assert cfg.action.interp_ticks == 3


def test_scalar_is_not_broadcast_to_vector(tmp_path, calib):
    _rejects(tmp_path, calib, lambda r: r["action"].update(max_delta_ticks=45), "max_delta_ticks")


def _calib_file(tmp_path, mutate):
    raw = json.loads(CALIB.read_text())
    mutate(raw)
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(raw))
    return p


@pytest.mark.parametrize("bad", [2, -1, True, "0", 0.5])
def test_calibration_drive_mode_must_be_0_or_1(tmp_path, bad):
    p = _calib_file(tmp_path, lambda r: r["elbow_flex"].update(drive_mode=bad))
    with pytest.raises(ValueError, match="drive_mode"):
        load_calibration(p)


@pytest.mark.parametrize("bad", [2048, -2048, 100000])
def test_calibration_homing_offset_range(tmp_path, bad):
    p = _calib_file(tmp_path, lambda r: r["elbow_flex"].update(homing_offset=bad))
    with pytest.raises(ValueError, match="homing_offset"):
        load_calibration(p)


@pytest.mark.parametrize("field", ["id", "homing_offset", "range_min", "range_max"])
@pytest.mark.parametrize("bad", [True, "5", 5.5])
def test_calibration_int_fields_reject_bool_str_fractional(tmp_path, field, bad):
    p = _calib_file(tmp_path, lambda r: r["elbow_flex"].update({field: bad}))
    with pytest.raises(ValueError, match=field):
        load_calibration(p)


def test_calibration_unknown_and_missing_field_rejected(tmp_path):
    p = _calib_file(tmp_path, lambda r: r["elbow_flex"].update(extra=1))
    with pytest.raises(ValueError, match="extra"):
        load_calibration(p)
    p = _calib_file(tmp_path, lambda r: r["elbow_flex"].pop("drive_mode"))
    with pytest.raises(ValueError, match="drive_mode"):
        load_calibration(p)


def test_calibration_argument_is_validated_too(tmp_path, calib):
    bad = dict(calib)
    bad["gripper"] = dataclasses.replace(calib["gripper"], drive_mode=7)
    with pytest.raises(ValueError, match="drive_mode"):
        load_config(calibration=bad)


# ---- FS1 sim_warp (sim-first track S0.3) ------------------------------------------------------------


def test_sim_warp_defaults(cfg):
    from so101_rl.config import SimWarpConfig

    w = cfg.sim_warp
    assert isinstance(w, SimWarpConfig)
    assert (w.njmax, w.nconmax_per_world, w.nworld_default, w.overflow_check_every_steps) == (256, 32, 1024, 10)
    assert w.overflow_policy == "fail"
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.njmax = 64  # type: ignore[misc]


def test_sim_warp_section_strict(tmp_path, calib):
    _rejects(tmp_path, calib, lambda r: r.pop("sim_warp"), "sim_warp")
    _rejects(tmp_path, calib, lambda r: r["sim_warp"].pop("njmax"), "njmax")
    _rejects(tmp_path, calib, lambda r: r["sim_warp"].pop("overflow_policy"), "overflow_policy")
    _rejects(tmp_path, calib, lambda r: r["sim_warp"].update(nconmax=32), "nconmax")
    _rejects(tmp_path, calib, lambda r: r["sim_warp"].update(overflow_policy="warn"), "overflow_policy")
    _rejects(tmp_path, calib, lambda r: r["sim_warp"].update(overflow_policy="ignore"), "overflow_policy")


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("njmax", 63),  # below MJWarp's own default, which already overflows here
        ("njmax", 0),
        ("nconmax_per_world", 0),
        ("nworld_default", 0),
        ("overflow_check_every_steps", 0),
        ("njmax", 256.5),
        ("njmax", True),
        ("nworld_default", "1024"),
    ],
)
def test_sim_warp_range_and_type_checked(tmp_path, calib, key, bad):
    _rejects(tmp_path, calib, lambda r: r["sim_warp"].update({key: bad}), key)


def test_sim_warp_lower_bounds_accepted(tmp_path, calib):
    def mut(r):
        r["sim_warp"].update(njmax=64, nconmax_per_world=1, nworld_default=1, overflow_check_every_steps=1)

    w = load_config(_variant(tmp_path, mut), calibration=calib).sim_warp
    assert (w.njmax, w.nconmax_per_world, w.nworld_default, w.overflow_check_every_steps) == (64, 1, 1, 1)


# ---- FS4 tasks section (SU0 review fixes, plan §10 2026-10-06) -----------------------------------------------


def test_tasks_defaults(cfg):
    die = cfg.tasks.die_to_zone
    assert die.max_height_m == 0.016 and die.settle_mm == 3.0
    assert cfg.tasks.reach.max_height_m is None and cfg.tasks.reach.settle_mm is None
    for t in (cfg.tasks.reach, die):
        assert [x.name for x in t.reward.terms] == ["distance", "success"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r["tasks"]["die_to_zone"]["reward"].update(bonus={"weight": 1.0}), "unknown reward term"),
        (lambda r: r["tasks"]["reach"]["reward"]["success"].pop("weight"), "weight"),
        (lambda r: r["tasks"]["reach"]["reward"]["success"].update(weight=-1.0), "weight"),
        (lambda r: r["tasks"]["reach"]["reward"]["success"].update(scale_m=0.3), "scale_m"),
        (lambda r: r["tasks"]["reach"]["reward"]["distance"].pop("zero_on_success"), "zero_on_success"),
        (lambda r: r["tasks"]["reach"]["reward"]["distance"].update(zero_on_success="yes"), "zero_on_success"),
        (lambda r: r["tasks"]["reach"]["reward"]["distance"].update(scale_m=0.0), "scale_m"),
        (lambda r: r["tasks"]["reach"].update(reward={}), "reward"),
        (lambda r: r["tasks"]["reach"].update(settle_mm=3.0), "settle_mm"),  # reach has no settle rule
        (lambda r: r["tasks"]["die_to_zone"].pop("settle_mm"), "settle_mm"),
        (lambda r: r["tasks"]["die_to_zone"].pop("max_height_m"), "max_height_m"),
        (lambda r: r["tasks"]["die_to_zone"].update(max_height_m=0.025), "max_height_m"),  # 2.5 cm up must fail
        (lambda r: r["tasks"]["die_to_zone"].update(settle_mm=0.0), "settle_mm"),
        (lambda r: r["tasks"]["die_to_zone"].update(settle_mm=True), "settle_mm"),
        (lambda r: r["tasks"].update(stack={}), "stack"),
        (lambda r: r.pop("tasks"), "tasks"),
    ],
)
def test_tasks_section_strict(tmp_path, calib, mutate, match):
    _rejects(tmp_path, calib, mutate, match)


def test_load_tasks_config_reads_only_the_section(tmp_path):
    from so101_rl.config import load_tasks_config

    p = tmp_path / "t.yaml"
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    p.write_text(yaml.safe_dump({"tasks": raw["tasks"]}))
    assert load_tasks_config(p) == load_tasks_config()
    p.write_text(yaml.safe_dump({"other": 1}))
    with pytest.raises(ValueError, match="tasks"):
        load_tasks_config(p)
