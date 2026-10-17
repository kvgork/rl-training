import cv2
import numpy as np
import pytest

from so101_rl.perception.fiducial import FiducialCheck

MARKER_ID = 7
SIZE = 120


def _render(x0=200, y0=150, dx=0, dy=0, marker_id=MARKER_ID):
    img = np.full((480, 640, 3), 255, np.uint8)
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, SIZE)
    x, y = x0 + dx, y0 + dy
    img[y - 20 : y + SIZE + 20, x - 20 : x + SIZE + 20] = 255  # quiet zone
    img[y : y + SIZE, x : x + SIZE] = m[..., None]
    return img


def _expected(x0=200, y0=150):
    return np.array([[x0, y0], [x0 + SIZE, y0], [x0 + SIZE, y0 + SIZE], [x0, y0 + SIZE]], np.float64)


def test_marker_at_expected_corners_ok():
    chk = FiducialCheck(MARKER_ID, _expected())
    rep = chk.check(_render())
    assert rep.found and rep.ok and rep.reproj_err_px < 1.5


def test_calibrated_expected_from_detection_ok_then_shift_fails():
    base = _render()
    expected = FiducialCheck(MARKER_ID, _expected()).detect_corners(base)
    chk = FiducialCheck(MARKER_ID, expected, tol_px=2.0)
    assert chk.check(base).ok
    rep = chk.check(_render(dx=3))
    assert rep.found and not rep.ok and 2.5 < rep.reproj_err_px < 3.5


def test_shift_3px_against_rendered_corners_fails():
    chk = FiducialCheck(MARKER_ID, _expected(), tol_px=1.5)
    assert chk.check(_render()).ok
    assert not chk.check(_render(dx=3)).ok


def test_marker_absent_is_not_ok():
    chk = FiducialCheck(MARKER_ID, _expected())
    rep = chk.check(np.full((480, 640, 3), 255, np.uint8))
    assert not rep.ok and not rep.found and rep.reproj_err_px == float("inf")


def test_other_marker_id_is_not_found():
    chk = FiducialCheck(MARKER_ID, _expected())
    rep = chk.check(_render(marker_id=3))
    assert not rep.found and not rep.ok


def test_from_yaml(tmp_path):
    p = tmp_path / "fiducial.yaml"
    corners = _expected().tolist()
    p.write_text(
        f"marker_id: {MARKER_ID}\ndictionary: DICT_4X4_50\ntol_px: 2.0\nexpected_corners_px: {corners}\n"
    )
    chk = FiducialCheck.from_yaml(p)
    assert chk.check(_render()).ok
    p.write_text(p.read_text().replace("DICT_4X4_50", "NOT_A_DICT"))
    with pytest.raises(AttributeError):
        FiducialCheck.from_yaml(p)


# ------------------------------------------------------------------------------------------------ H8: the 50 mm marker
# (plan §10 2026-10-17 "H8 software"): capture CLI, geometry, the MJCF site


@pytest.fixture(scope="module")
def calib():
    from so101_rl.perception.camera_calib import load_camera_calib

    return load_camera_calib()


def test_the_mjcf_site_is_the_documented_marker():
    from so101_rl.perception import fiducial as f
    from so101_rl.sim.model import TABLE_Z_M, load_model

    model, data = load_model()
    sid = model.site("fiducial").id
    assert data.site_xpos[sid][:2] == pytest.approx(f.XY_BASE, abs=1e-9)
    assert data.site_xpos[sid][2] == pytest.approx(TABLE_Z_M + 0.0005, abs=1e-9)  # flat on the table
    assert model.site_size[sid][:2] == pytest.approx([f.SIZE_MM / 2000] * 2, abs=1e-9)
    assert (f.DICTIONARY, f.MARKER_ID, f.TOL_PX) == ("DICT_4X4_50", 0, 2.0)


def test_the_marker_is_in_view_and_about_58_px(calib):
    """50 mm at (0.09, 0.05): fully inside the calibrated image with margin, ~58 px a side (30 mm at the old site was
    ~25 px, ~4 px per cell)."""
    from fiducial_synth import predicted_corners

    px = predicted_corners(calib)
    c = calib.color
    assert (px[:, 0] > 20).all() and (px[:, 0] < c.width - 20).all() and (px[:, 1] > 5).all() and (px[:, 1] < c.height - 5).all()
    sides = np.linalg.norm(np.roll(px, -1, axis=0) - px, axis=1)
    assert 45 < sides.mean() < 75, sides


def test_synthetic_frame_is_detected_at_the_predicted_corners(calib):
    from fiducial_synth import marker_frame, predicted_corners

    from so101_rl.perception.fiducial import FiducialCheck

    chk = FiducialCheck(0, predicted_corners(calib))
    rep = chk.check(marker_frame(calib))
    assert rep.found and rep.reproj_err_px < 0.5, rep


def _capture(calib, tmp_path, frames, *extra):
    from fiducial_synth import SequenceCamera, plane_depth

    from so101_rl.perception import fiducial as f

    depth = plane_depth(calib)
    it = iter(frames)
    cam = SequenceCamera(calib, lambda: (next(it), depth))
    out = tmp_path / "fiducial.yaml"
    code = f.main(["--out", str(out), "--frames", str(len(frames)), *extra], camera=cam)
    return code, out, cam


def test_capture_writes_the_yaml_the_gate_reads(calib, tmp_path):
    import yaml
    from fiducial_synth import marker_frame, predicted_corners

    from so101_rl.perception.fiducial import FiducialCheck

    img = marker_frame(calib)
    code, out, cam = _capture(calib, tmp_path, [img] * 10)
    assert code == 0 and not cam.started  # stopped after the capture
    doc = yaml.safe_load(out.read_text())
    assert doc["dictionary"] == "DICT_4X4_50" and doc["marker_id"] == 0 and doc["tol_px"] == 2.0
    assert doc["size_mm"] == 50.0 and doc["xy_base"] == [0.09, 0.05] and doc["camera_serial"] == calib.serial
    assert doc["camera_yaml_sha256"] == calib.source_sha256 and doc["found_frames"] == 10
    assert doc["prior_err_px"] < 0.5 and doc["jitter_px"] < 1e-6
    assert np.abs(np.array(doc["expected_corners_px"]) - predicted_corners(calib)).max() < 1.0  # warped, sub-pixel
    chk = FiducialCheck.from_yaml(out)
    assert chk.camera_serial == calib.serial and chk.check(img).ok
    rep = chk.check(marker_frame(calib, shift_px=(3.0, 0.0)))  # a moved camera
    assert rep.found and not rep.ok and 2.5 < rep.reproj_err_px < 3.5


def test_capture_refuses_a_missing_marker_or_jitter(calib, tmp_path):
    from fiducial_synth import marker_frame

    good, gone = marker_frame(calib), marker_frame(calib, present=False)
    code, out, _ = _capture(calib, tmp_path, [good] * 5 + [gone] * 5)  # found in 50 % < 90 %
    assert code == 1 and not out.exists()
    shaky = [marker_frame(calib, shift_px=(1.5 * (i % 2), 0.0)) for i in range(10)]  # +-0.75 px corner jitter
    code, out, _ = _capture(calib, tmp_path, shaky)
    assert code == 1 and not out.exists()


def test_capture_refuses_another_camera(calib, tmp_path):
    from fiducial_synth import SequenceCamera, marker_frame, plane_depth

    from so101_rl.perception import fiducial as f

    cam = SequenceCamera(calib, lambda: (marker_frame(calib), plane_depth(calib)))
    cam.device_serial = "000000000000"
    assert f.main(["--out", str(tmp_path / "x.yaml")], camera=cam) == 3


def test_prior_error_reports_a_rotated_marker():
    from so101_rl.perception.fiducial import prior_error_px

    pred = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float64)
    err, k = prior_error_px(np.roll(pred, 1, axis=0), pred)
    assert err == pytest.approx(0.0) and k == 3
