import cv2
import numpy as np
import pytest

from so101_rl.perception.camera import Intrinsics
from so101_rl.perception.die_tracker import HsvDepthDieTracker

Z_TABLE = 0.614
Z_DIE = 0.600
INTR = Intrinsics(fx=Z_TABLE / 0.00099, fy=Z_TABLE / 0.00099, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001)
RED = (200, 20, 20)
GREY = (120, 120, 120)


def _scene():
    rgb = np.full((480, 640, 3), GREY, np.uint8)
    depth = np.full((480, 640), Z_TABLE, np.float32)
    return rgb, depth


def _square(rgb, depth, x0, y0, size=16, colour=RED, z=Z_DIE):
    rgb[y0 : y0 + size, x0 : x0 + size] = colour
    depth[y0 : y0 + size, x0 : x0 + size] = z


def _tracker(**kw):
    return HsvDepthDieTracker.from_yaml(INTR) if not kw else HsvDepthDieTracker(
        np.array([170, 120, 70]), np.array([10, 255, 255]), INTR, **kw
    )


def _expected_xyz(x0, y0, size=16, z=Z_DIE):
    u, v = x0 + (size - 1) / 2, y0 + (size - 1) / 2
    return np.array([(u - INTR.cx) * z / INTR.fx, (v - INTR.cy) * z / INTR.fy, z])


def test_die_position_within_2mm():
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300)
    det = _tracker().detect(rgb, depth)
    assert det is not None
    assert np.linalg.norm(det.xyz_cam - _expected_xyz(400, 300)) < 0.002
    assert det.area_px == 256
    assert 0.5 < det.confidence <= 1.0


def test_no_red_returns_none():
    rgb, depth = _scene()
    assert _tracker().detect(rgb, depth) is None


def test_half_occluded_die_returns_none():
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300)
    # grey object, nearer than the die, hides the right half
    rgb[290:330, 408:440] = GREY
    depth[290:330, 408:440] = 0.45
    assert _tracker().detect(rgb, depth) is None


def test_fingertip_lookalike_ignored_and_die_chosen():
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300)
    _square(rgb, depth, 100, 100, size=5)  # 25 px < min_area
    det = _tracker().detect(rgb, depth)
    assert det is not None and abs(det.u - 407.5) < 0.5
    # the look-alike alone is not a detection
    rgb2, depth2 = _scene()
    _square(rgb2, depth2, 100, 100, size=5)
    assert _tracker().detect(rgb2, depth2) is None


def test_two_valid_blobs_choose_largest():
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300, size=16)
    _square(rgb, depth, 100, 100, size=10)
    det = _tracker().detect(rgb, depth)
    assert det is not None and det.area_px == 256


def test_inconsistent_depth_returns_none():
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300)
    rng = np.random.default_rng(0)
    depth[300:316, 400:416] = Z_DIE + rng.uniform(-0.05, 0.05, (16, 16)).astype(np.float32)
    assert _tracker().detect(rgb, depth) is None


def test_missing_depth_returns_none():
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300, z=0.0)
    assert _tracker().detect(rgb, depth) is None


@pytest.mark.parametrize("hue", [2, 5, 10, 170, 175, 179])
def test_hue_wraparound(hue):
    hsv = np.array([[[hue, 220, 200]]], np.uint8)
    colour = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300, colour=colour)
    assert _tracker().detect(rgb, depth) is not None


@pytest.mark.parametrize("hue", [30, 60, 120])
def test_non_red_hues_rejected(hue):
    hsv = np.array([[[hue, 220, 200]]], np.uint8)
    colour = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
    rgb, depth = _scene()
    _square(rgb, depth, 400, 300, colour=colour)
    assert _tracker().detect(rgb, depth) is None


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        HsvDepthDieTracker(np.array([170, 120, 70]), np.array([10, 255, 255]), INTR, min_area_px=0)


def test_accept_skips_a_larger_blob_off_the_table():
    """Plan §10 2026-10-17: a larger red blob off the table (a book on the bed) must not hide the die."""
    rgb, depth = _scene()
    _square(rgb, depth, 100, 100, size=30)  # big, "off the table" (left half of the image here)
    _square(rgb, depth, 400, 300, size=16)  # the die
    tr = _tracker()
    on_table = lambda det: det.u > 320  # noqa: E731 - the test's stand-in for the base-frame workspace check
    det = tr.detect(rgb, depth, accept=on_table)
    assert det is not None and det.area_px == 256
    assert tr.rejected == 1
    assert tr.detect(rgb, depth).area_px == 900  # without accept: the original largest-blob rule


def test_accept_skips_unmeasurable_blobs_and_returns_none_when_nothing_is_accepted():
    rgb, depth = _scene()
    _square(rgb, depth, 0, 100, size=30)  # cut by the image border: skipped, not a stop
    _square(rgb, depth, 400, 300, size=16)
    det = _tracker().detect(rgb, depth, accept=lambda d: True)
    assert det is not None and det.area_px == 256
    assert _tracker().detect(rgb, depth, accept=lambda d: False) is None


def test_table_workspace_contains_the_zone_and_the_well_but_not_the_bed():
    from so101_rl.perception.die_tracker import table_workspace

    lo, hi = table_workspace()
    for p in ([0.15, -0.13, 0.0035], [0.20, -0.05, 0.0035], [0.174, 0.033, 0.001]):
        assert np.all(np.asarray(p) >= lo) and np.all(np.asarray(p) <= hi), p
    assert not np.all(np.asarray([0.36, 0.60, 0.3]) <= hi)
