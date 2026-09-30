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
