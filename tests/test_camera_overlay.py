"""camera-overlay (C.4): real frame | sim render | blend, and the arm-mask IoU gate (>= 0.8), on synthetic frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from camera_synth import perturbed, render_rgbd, with_pose

from so101_rl.config import load_calibration
from so101_rl.perception import camera_overlay
from so101_rl.perception.camera_calib import load_camera_calib, save_camera_calib
from so101_rl.servo.probe import ProbeRefused
from so101_rl.sim.arm_cloud import qpos_from_ticks
from so101_rl.sim.model import TABLE_Z_M, load_model

CAL_PATH = Path(__file__).parent / "data" / "so101_follower.calib.json"
CAL = load_calibration(CAL_PATH)
TICKS = np.array([2050, 1500, 2500, 2700, 2040, 2300], dtype=np.int32)


@pytest.fixture(scope="module")
def real():
    calib = load_camera_calib()
    model = load_model(camera=False)[0]
    q = qpos_from_ticks(model, TICKS, CAL)
    rgb, depth = render_rgbd(calib, q, noise_m=0.0015, seed=3)  # the "real" frame: same camera, clutter, noise
    return calib, q, rgb, depth


def test_iou_passes_when_the_sim_camera_is_the_real_one(real):
    calib, q, rgb, depth = real
    res = camera_overlay.overlay(calib, rgb, depth, q, TABLE_Z_M)
    print(f"IoU (same camera) {res.iou:.3f}, real {res.real_px} px, sim {res.sim_px} px")
    assert res.iou >= camera_overlay.MIN_IOU
    assert res.image.shape == (480, 3 * 640, 3)


def test_iou_fails_with_a_wrong_camera_pose_or_wrong_angles(real):
    calib, q, rgb, depth = real
    wrong_pose = with_pose(calib, perturbed(calib.T_base_color, dx=0.02, dy=-0.015, dyaw_deg=4.0))
    r1 = camera_overlay.overlay(wrong_pose, rgb, depth, q, TABLE_Z_M)
    q2 = q.copy()
    q2[1] += 0.25
    r2 = camera_overlay.overlay(calib, rgb, depth, q2, TABLE_Z_M)
    print(f"IoU wrong pose {r1.iou:.3f}, wrong shoulder {r2.iou:.3f}")
    assert r1.iou < camera_overlay.MIN_IOU and r2.iou < camera_overlay.MIN_IOU


def test_overlay_cli_writes_png_and_json(tmp_path, real):
    calib, _, rgb, depth = real
    cam_yaml = tmp_path / "camera.yaml"
    save_camera_calib(calib, cam_yaml)
    out = tmp_path / "ov"
    rc = camera_overlay.main(["--camera-yaml", str(cam_yaml), "--out-dir", str(out), "--calibration", str(CAL_PATH)],
                             probe=lambda: TICKS, camera=lambda c, n: (rgb, np.stack([depth] * n)))  # fmt: skip
    assert rc == 0
    rep = json.loads(next(out.glob("*.json")).read_text())
    assert rep["iou"] >= 0.8 and Path(rep["png"]).exists()
    rc = camera_overlay.main(["--camera-yaml", str(cam_yaml), "--out-dir", str(out), "--calibration", str(CAL_PATH)],
                             probe=lambda: (_ for _ in ()).throw(ProbeRefused("silent")), camera=None)  # fmt: skip
    assert rc == 3
