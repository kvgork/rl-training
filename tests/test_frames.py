import numpy as np
import pytest
import yaml

from so101_rl.perception.camera import Intrinsics
from so101_rl.perception.frames import (
    HandEye,
    cam_to_base,
    fit_hand_eye,
    load_hand_eye,
    pixel_to_cam,
    save_hand_eye,
)


def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def _truth():
    T = np.eye(4)
    T[:3, :3] = _rot(2, 25) @ _rot(0, 160) @ _rot(1, 7)
    T[:3, 3] = [0.21, -0.03, 0.55]
    return T


def _points(noise_mm, n=12, seed=0):
    rng = np.random.default_rng(seed)
    p_base = rng.uniform([0.08, -0.2, 0.02], [0.34, 0.2, 0.25], (n, 3))
    T = _truth()
    p_cam = (p_base - T[:3, 3]) @ T[:3, :3]  # inverse rotation
    p_cam = p_cam + rng.normal(0, noise_mm / 1000 / np.sqrt(3), p_cam.shape)
    return p_base, p_cam, T


def test_fit_recovers_transform_with_noise():
    p_base, p_cam, T = _points(1.0)
    T_fit, res = fit_hand_eye(p_base, p_cam)
    assert res < 2.0
    assert np.allclose(T_fit[:3, :3], T[:3, :3], atol=1e-2)
    assert np.allclose(T_fit[:3, 3], T[:3, 3], atol=3e-3)
    assert np.allclose(T_fit[:3, :3] @ T_fit[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(T_fit[:3, :3]), 1.0)


def test_fit_exact_data_recovers_to_1e3():
    p_base, p_cam, T = _points(0.0)
    T_fit, res = fit_hand_eye(p_base, p_cam)
    assert res < 1e-6
    assert np.allclose(T_fit, T, atol=1e-3)


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_hand_eye(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        fit_hand_eye(np.zeros((2, 3)), np.zeros((2, 3)))
    bad = np.zeros((5, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_hand_eye(bad, np.zeros((5, 3)))


def test_fit_never_returns_a_reflection():
    p_cam = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    p_base = p_cam * np.array([-1, 1, 1])  # mirrored: no proper rotation fits exactly
    T, res = fit_hand_eye(p_base, p_cam)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    assert res > 1.0  # residual reveals the bad data


def test_pixel_cam_base_roundtrip():
    intr = Intrinsics(600.0, 610.0, 320.0, 240.0, 640, 480, 0.001)
    p = pixel_to_cam(380.0, 300.0, 0.5, intr)
    assert np.allclose(p, [60 * 0.5 / 600, 60 * 0.5 / 610, 0.5])
    T = _truth()
    assert np.allclose(cam_to_base(p, T), T[:3, :3] @ p + T[:3, 3])


def _write(path, T, residual, n):
    save_hand_eye(HandEye(T, residual, n, "2026-09-30", "819312071240"), path)


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "hand_eye.yaml"
    _write(p, _truth(), 1.2, 12)
    he = load_hand_eye(p)
    assert np.allclose(he.T_base_cam, _truth())
    assert (he.residual_mm, he.n_poses, he.camera_serial) == (1.2, 12, "819312071240")


def test_load_refuses_high_residual(tmp_path):
    p = tmp_path / "hand_eye.yaml"
    _write(p, _truth(), 3.5, 12)
    with pytest.raises(ValueError, match="residual"):
        load_hand_eye(p)


def test_load_refuses_few_poses(tmp_path):
    p = tmp_path / "hand_eye.yaml"
    _write(p, _truth(), 1.0, 9)
    with pytest.raises(ValueError, match="poses"):
        load_hand_eye(p)


def test_load_refuses_malformed(tmp_path):
    p = tmp_path / "hand_eye.yaml"
    _write(p, _truth(), 1.0, 12)
    doc = yaml.safe_load(p.read_text())
    del doc["camera_serial"]
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="missing"):
        load_hand_eye(p)
    doc["camera_serial"] = "x"
    doc["T_base_cam"][0][0] = 2.0  # not a rotation
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="rotation"):
        load_hand_eye(p)
    with pytest.raises(FileNotFoundError):
        load_hand_eye(tmp_path / "nope.yaml")
