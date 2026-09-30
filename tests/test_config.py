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
    assert cfg.safe_state.bound_s == 5.0


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
