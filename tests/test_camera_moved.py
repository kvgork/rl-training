"""H8 camera-moved check (``pixi run camera-moved-check``, ``perception/camera_moved.py``) on synthetic frames: the
fiducial warped by camera.yaml's homography and the table plane's depth (``tests/fiducial_synth.py``)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from fiducial_synth import SequenceCamera, marker_frame, plane_depth, predicted_corners

from so101_rl.perception import camera_moved as cm
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.fiducial import FiducialCheck


@pytest.fixture(scope="module")
def calib():
    return load_camera_calib()


@pytest.fixture()
def fid_yaml(calib, tmp_path):
    expected = FiducialCheck(0, predicted_corners(calib)).detect_corners(marker_frame(calib))
    p = tmp_path / "fiducial.yaml"
    p.write_text(yaml.safe_dump({"dictionary": "DICT_4X4_50", "marker_id": 0, "tol_px": 2.0,
                                 "camera_serial": calib.serial, "expected_corners_px": expected.tolist()}))  # fmt: skip
    return p


def _args(fid_yaml, out, *extra):
    return ["--fiducial-yaml", str(fid_yaml), "--out", str(out), "--max-frames", "12", "--plane-every", "4",
            "--duration-s", "30", *extra]  # fmt: skip


def test_single_check_passes_and_fails(calib, fid_yaml, tmp_path):
    depth = plane_depth(calib)
    good = SequenceCamera(calib, lambda: (marker_frame(calib), depth))
    out = tmp_path / "ok.json"
    assert cm.main(_args(fid_yaml, out, "--label", "baseline"), camera=good) == 0
    doc = json.loads(out.read_text())
    assert doc["ok"] and doc["pass"] and doc["label"] == "baseline" and doc["err_px"] < 0.5
    assert doc["plane"]["ok"] and doc["plane"]["checks"] == 3 and abs(doc["plane"]["dh_mm"]) < 0.5
    moved = SequenceCamera(calib, lambda: (marker_frame(calib, shift_px=(2.5, 0.5)), depth))
    out2 = tmp_path / "moved.json"
    assert cm.main(_args(fid_yaml, out2), camera=moved) == 1
    doc = json.loads(out2.read_text())
    assert not doc["ok"] and doc["err_px"] > 2.0 and doc["plane"]["ok"]  # the plane cannot see an image shift


def test_procedure_baseline_nudged_restored(calib, fid_yaml, tmp_path):
    depth = plane_depth(calib)
    shift = [(0.0, 0.0)]
    cam = SequenceCamera(calib, lambda: (marker_frame(calib, shift_px=shift[0]), depth))
    prompts = []

    def prompt(msg):
        prompts.append(msg)
        shift[0] = (2.0, 1.5) if len(prompts) == 1 else (0.0, 0.0)  # nudge, then put back

    out = tmp_path / "proc.json"
    assert cm.main(_args(fid_yaml, out, "--procedure"), camera=cam, prompt=prompt) == 0
    doc = json.loads(out.read_text())
    assert len(prompts) == 2 and doc["pass"]
    assert doc["baseline_err_px"] < 0.5 and doc["nudged_err_px"] > 2.0 and doc["restored_err_px"] <= 2.0
    assert not doc["nudged"]["ok"] and doc["baseline"]["ok"] and doc["restored"]["ok"]


def test_procedure_fails_when_the_nudge_is_not_seen(calib, fid_yaml, tmp_path):
    depth = plane_depth(calib)
    cam = SequenceCamera(calib, lambda: (marker_frame(calib), depth))
    out = tmp_path / "proc.json"
    assert cm.main(_args(fid_yaml, out, "--procedure"), camera=cam, prompt=lambda _m: None) == 1
    assert not json.loads(out.read_text())["pass"]


def test_lifted_camera_fails_the_plane_half(calib, fid_yaml, tmp_path):
    depth = plane_depth(calib) + np.float32(0.02)
    cam = SequenceCamera(calib, lambda: (marker_frame(calib), depth))
    out = tmp_path / "lift.json"
    assert cm.main(_args(fid_yaml, out), camera=cam) == 1
    doc = json.loads(out.read_text())
    assert doc["fiducial"]["ok"] and not doc["plane"]["ok"]


def test_missing_fiducial_yaml_is_refused(calib, tmp_path):
    cam = SequenceCamera(calib, lambda: (marker_frame(calib), plane_depth(calib)))
    assert cm.main(_args(tmp_path / "none.yaml", tmp_path / "x.json"), camera=cam) == 2
    assert not cam.started
