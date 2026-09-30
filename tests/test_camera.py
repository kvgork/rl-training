import sys
import time

import numpy as np
import pytest

from so101_rl.perception.camera import Camera, D435Camera, FakeCamera, Frame, Intrinsics

INTR = Intrinsics(620.0, 620.0, 320.0, 240.0, 64, 48, 0.001)


def _fake():
    rgb = np.full((48, 64, 3), 7, np.uint8)
    depth = np.full((48, 64), 0.6, np.float32)
    return FakeCamera(rgb, depth, INTR), rgb, depth


def test_fake_camera_returns_frames_and_intrinsics():
    cam, rgb, depth = _fake()
    assert isinstance(cam, Camera)
    assert cam.start() == INTR
    rgb_out = np.zeros_like(rgb)
    d_out = np.zeros_like(depth)
    f = cam.read(rgb_out, d_out)
    assert isinstance(f, Frame) and f.ok and f.t_mono_ns > 0
    assert np.array_equal(rgb_out, rgb) and np.array_equal(d_out, depth)
    cam.stop()
    with pytest.raises(RuntimeError):
        cam.read(rgb_out, d_out)


def test_fake_camera_timeout_and_replacement():
    cam, rgb, depth = _fake()
    cam.start()
    cam.fail_next = 1
    rgb_out, d_out = np.zeros_like(rgb), np.zeros_like(depth)
    assert not cam.read(rgb_out, d_out).ok
    assert not rgb_out.any()  # buffers untouched on failure
    cam.set_frames(rgb + 1, depth)
    assert cam.read(rgb_out, d_out).ok and rgb_out[0, 0, 0] == 8
    with pytest.raises(ValueError):
        cam.set_frames(rgb, depth[:-1])


def test_d435_defaults_and_selection_by_serial():
    cam = D435Camera()
    assert cam.serial == "819312071240"
    assert (cam.width, cam.height, cam.fps, cam.warmup_frames) == (640, 480, 30, 45)
    with pytest.raises(RuntimeError):
        cam.read(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.float32))
    cam.stop()  # stop before start is a no-op


def test_d435_start_without_sdk_is_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)  # makes `import pyrealsense2` raise ImportError
    cam = D435Camera()  # constructing must not need the SDK
    with pytest.raises(ImportError, match="pyrealsense2"):
        cam.start()


@pytest.mark.hardware
def test_d435_hardware_fps_and_depth():
    cam = D435Camera()
    intr = cam.start()  # 45 warmup frames inside
    try:
        assert intr.depth_scale > 0 and (intr.width, intr.height) == (640, 480)
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.zeros((480, 640), np.float32)
        t0 = time.monotonic()
        n_ok = 0
        for _ in range(30):
            n_ok += cam.read(rgb, depth).ok
        fps = 30 / (time.monotonic() - t0)
        assert n_ok == 30
        assert fps >= 25.0, fps
        centre = float(np.median(depth[235:245, 315:325]))
        assert 0.3 <= centre <= 1.5, centre
    finally:
        cam.stop()
