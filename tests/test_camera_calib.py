"""calib/camera.yaml (camera parity C.1): strict schema, the device read on a fake D435, plane-constrained poses."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from so101_rl.perception import camera_intrinsics
from so101_rl.perception.camera_calib import (
    CAMERA_YAML,
    CameraNotCalibrated,
    CameraPose,
    camera_calib_doc,
    load_camera_calib,
    parse_camera_calib,
    plane_constrained_pose,
    provisional_pose,
    save_camera_calib,
)


@pytest.fixture
def calib():
    return load_camera_calib()


def test_committed_file_loads_with_device_intrinsics(calib):
    assert calib.serial == "819312071240"
    c, d = calib.color, calib.depth
    assert (c.width, c.height, c.fps, d.width, d.height) == (640, 480, 30, 640, 480)
    # the factory values measured 2026-10-06 (plan C "What is measured already"), to the reported precision
    assert (round(c.fx, 2), round(c.fy, 2), round(c.cx, 2), round(c.cy, 2)) == (617.17, 617.57, 327.95, 236.12)
    assert (round(d.fx, 2), round(d.cx, 2), round(d.cy, 2)) == (388.33, 326.99, 240.87)
    assert calib.depth_scale == pytest.approx(0.001)
    np.testing.assert_allclose(calib.T_color_depth[:3, 3] * 1000, [14.76, 0.13, 0.13], atol=0.01)
    assert calib.pose is not None and calib.table_plane is not None


def test_roundtrip_and_strictness(calib, tmp_path):
    p = tmp_path / "camera.yaml"
    save_camera_calib(calib, p)
    back = load_camera_calib(p)
    np.testing.assert_allclose(back.T_base_color, calib.T_base_color)
    assert back.color == calib.color and back.depth == calib.depth
    doc = camera_calib_doc(calib)
    for mutate, match in (
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d.pop("serial"), "missing keys"),
        (lambda d: d.update(version=2), "version"),
        (lambda d: d["color"]["distortion"].update(coeffs=[0.1, 0, 0, 0, 0]), "non-zero distortion"),
        (lambda d: d["color"].update(cx=700.0), "cx"),
        (lambda d: d["color"].update(fx=True), "number"),
        (lambda d: d["T_base_color"].update(method="guess"), "method"),
        (lambda d: d["T_base_color"]["matrix"][0].__setitem__(0, 2.0), "rotation"),
        (lambda d: d["depth_to_color"][0].__setitem__(3, 0.5), "baseline"),
        (lambda d: d["table_plane"].update(normal_color=[0, 0, 1.0]), "point up"),
    ):
        bad = copy.deepcopy(doc)
        mutate(bad)
        with pytest.raises((ValueError, TypeError), match=match):
            parse_camera_calib(bad)


def test_calibrated_pose_needs_a_residual_within_the_gate(calib):
    doc = camera_calib_doc(calib)
    doc["T_base_color"].update(method="arm-shape", residual_mm=None)
    with pytest.raises(ValueError, match="residual_mm"):
        parse_camera_calib(doc)
    doc["T_base_color"]["residual_mm"] = 5.5
    with pytest.raises(ValueError, match="gate"):
        parse_camera_calib(doc)
    doc["T_base_color"]["residual_mm"] = 2.0
    assert parse_camera_calib(doc).pose.calibrated


def test_require_calibrated_and_missing_pose(calib, tmp_path):
    p = tmp_path / "c.yaml"
    save_camera_calib(calib.with_pose(CameraPose(calib.T_base_color, "provisional", "x", None, 150, 30)), p)
    with pytest.raises(CameraNotCalibrated):
        load_camera_calib(p, require_calibrated=True)
    doc = camera_calib_doc(calib)
    doc["T_base_color"] = None
    nopose = parse_camera_calib(doc)
    with pytest.raises(CameraNotCalibrated, match="camera-calibrate"):
        _ = nopose.T_base_color
    with pytest.raises(FileNotFoundError, match="camera-intrinsics"):
        load_camera_calib(tmp_path / "missing.yaml")


def test_dr_ranges_floor_and_uncertainty(calib):
    prov = calib.with_pose(CameraPose(calib.T_base_color, "provisional", "x", None, 150.0, 30.0))
    t, r = prov.dr_ranges()
    assert t == pytest.approx(0.005) and r == pytest.approx(math.radians(1.5))  # provisional: the floor only
    tight = calib.with_pose(CameraPose(calib.T_base_color, "arm-shape", "x", 2.0, 1.0, 0.2))
    assert tight.dr_ranges() == pytest.approx((0.005, math.radians(1.5)))
    wide = calib.with_pose(CameraPose(calib.T_base_color, "arm-shape", "x", 2.0, 9.0, 3.0))
    assert wide.dr_ranges() == pytest.approx((0.009, math.radians(3.0)))


def test_plane_constrained_pose_fixes_height_and_tilt():
    up = np.array([-0.110, -0.892, -0.439])
    up /= np.linalg.norm(up)
    for x, y, yaw in ((0.0, 0.0, 0.0), (0.2, -0.3, 1.0), (-0.1, 0.4, -2.5)):
        T = plane_constrained_pose(up, 0.3457, -0.02, x, y, yaw)
        assert T[2, 3] == pytest.approx(0.3257)
        np.testing.assert_allclose(T[:3, :3] @ up, [0, 0, 1], atol=1e-12)  # the camera's up is base +z
        tilt = math.degrees(math.acos(-T[2, 2]))  # optical axis vs straight down
        assert tilt == pytest.approx(64.0, abs=0.1)
    T = provisional_pose(up, 0.3457, -0.02, np.array([0.165, -0.27]), np.array([0.0, -1.0]))
    z = T[:3, 2]
    s = 0.3457 / -z[2]
    np.testing.assert_allclose(T[:3, 3] + s * z, [0.165, -0.27, -0.02], atol=1e-12)  # the axis meets the look-at
    assert z[0] == pytest.approx(0.0, abs=1e-12) and z[1] < 0


# ---- camera-intrinsics on a fake device ----------------------------------------------------------------------
def _fake_rs(serial="819312071240", rot_colmajor=None):
    def intr(w, h, fx, fy, cx, cy, model):
        return SimpleNamespace(width=w, height=h, fx=fx, fy=fy, ppx=cx, ppy=cy, model=model, coeffs=[0.0] * 5)

    rot = rot_colmajor or [1, 0, 0, 0, 1, 0, 0, 0, 1]
    started = []

    class VSP:
        def __init__(self, i, fps):
            self._i, self._fps = i, fps

        def get_intrinsics(self):
            return self._i

        def fps(self):
            return self._fps

        def as_video_stream_profile(self):
            return self

        def get_extrinsics_to(self, other):
            return SimpleNamespace(rotation=rot, translation=[0.01476, 0.00013, 0.00013])

    color = VSP(intr(640, 480, 617.1, 617.5, 327.9, 236.1, "distortion.inverse_brown_conrady"), 30)
    depth = VSP(intr(640, 480, 388.3, 388.3, 326.9, 240.8, "distortion.brown_conrady"), 30)
    dev = SimpleNamespace(
        get_info=lambda k: {"sn": serial, "fw": "5.11.1.100"}[k],
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001),
    )

    class Pipeline:
        def start(self, cfg):
            started.append(cfg.streams)
            return SimpleNamespace(get_device=lambda: dev, get_stream=lambda s: color if s == "color" else depth)

        def stop(self):
            started.append("stopped")

    class Config:
        def __init__(self):
            self.streams = []

        def enable_device(self, s):
            self.serial = s

        def enable_stream(self, *a):
            self.streams.append(a)

    rs = SimpleNamespace(
        config=Config, pipeline=Pipeline, stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"), camera_info=SimpleNamespace(serial_number="sn", firmware_version="fw"),
    )  # fmt: skip
    return rs, started


def test_camera_intrinsics_on_a_fake_device(tmp_path, calib):
    # a 90 deg rotation about z, column-major as librealsense reports it: R = [[0,-1,0],[1,0,0],[0,0,1]]
    rs, started = _fake_rs(rot_colmajor=[0, 1, 0, -1, 0, 0, 0, 0, 1])
    out = tmp_path / "camera.yaml"
    assert camera_intrinsics.main(["--serial", "819312071240", "--out", str(out)], rs_module=rs) == 0
    c = load_camera_calib(out)
    assert c.color.fx == pytest.approx(617.1) and c.depth.cx == pytest.approx(326.9)
    assert c.color.distortion_model == "inverse_brown_conrady"
    np.testing.assert_allclose(c.T_color_depth[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert c.pose is None and c.table_plane is None  # a fresh file has no pose
    assert started[-1] == "stopped" and len(started[0]) == 2  # streams only, pipeline stopped
    # a second run keeps the pose of an existing file of the same camera
    save_camera_calib(c.with_pose(calib.pose, calib.table_plane), out)
    rs, _ = _fake_rs()
    assert camera_intrinsics.main(["--serial", "819312071240", "--out", str(out)], rs_module=rs) == 0
    again = load_camera_calib(out)
    np.testing.assert_allclose(again.T_base_color, calib.T_base_color)
    np.testing.assert_allclose(again.T_color_depth[:3, :3], np.eye(3))


def test_camera_intrinsics_refuses_another_serial(tmp_path):
    rs, _ = _fake_rs(serial="000")
    with pytest.raises(RuntimeError, match="reports 000"):
        camera_intrinsics.main(["--serial", "819312071240", "--out", str(tmp_path / "c.yaml")], rs_module=rs)


def test_committed_file_is_generated_format():
    doc = yaml.safe_load(CAMERA_YAML.read_text())
    assert parse_camera_calib(doc).serial == "819312071240"
