"""``hand-eye --execute`` (H8; plan §10 2026-10-17 "H8 software"): target selection, the marker FK, the camera.yaml
comparison, the refusals, and the whole path against ``servo --backend mujoco`` with the sim camera (MuJoCo renders
following the servo's ticks, a red sphere on the tip). Never a serial device, never the D435."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from so101_rl.perception import calibrate_hand_eye as he
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.frames import HandEye, load_hand_eye

ROOT = Path(__file__).parent.parent
CALIB = ROOT / "tests" / "data" / "so101_follower.calib.json"
# the 3 mm reach export (plan §10 2026-10-16); runs/ is not in git: SO101_REACH_POLICY points elsewhere (a worktree)
REACH_POLICY = Path(os.environ.get("SO101_REACH_POLICY", ROOT / "runs/sac-sim/reach-3mm-20261015-201549/policy.pt"))
SIM_MARKER_OFFSET = ("0", "0.012", "-0.01")  # clear of the fixed jaw: sim detections within 0.4 mm of the truth


@pytest.fixture(scope="module")
def calib():
    return load_camera_calib()


def test_candidate_targets_are_reachable_in_view_and_in_the_box(calib):
    from so101_rl.config.loader import load_geometry
    from so101_rl.env.tasks import REACH

    box = load_geometry()[1]
    pts = he.candidate_targets(REACH.goal_bounds, box, calib.T_base_color, calib.color_intrinsics())
    assert len(pts) >= 40
    assert REACH.goal_bounds.contains_np(pts).all()
    assert (pts[:, 2] >= he.Z_RANGE_M[0] - 1e-9).all() and (pts[:, 2] <= he.Z_RANGE_M[1] + 1e-9).all()
    for k, (lo, hi) in enumerate((box.x, box.y, box.z)):
        assert (pts[:, k] >= lo + he.BOX_MARGIN_M - 1e-9).all() and (pts[:, k] <= hi - he.BOX_MARGIN_M + 1e-9).all()
    uv = calib.color.project((pts - calib.T_base_color[:3, 3]) @ calib.T_base_color[:3, :3])
    m = he.VIEW_MARGIN_PX
    assert (uv >= m - 1e-6).all() and (uv[:, 0] <= calib.color.width - m).all() and (uv[:, 1] <= calib.color.height - m).all()
    chosen = pts[he.farthest_point_subset(pts, 12)]  # spread: no two of the first 12 closer than 4 cm
    d = np.linalg.norm(chosen[:, None] - chosen[None], axis=2) + np.eye(12)
    assert d.min() > 0.04


def test_marker_fk_is_the_safety_fk_tip_plus_the_gripper_frame_offset():
    from so101_rl.config import load_calibration
    from so101_rl.config.loader import DEFAULT_CONFIG_PATH
    from so101_rl.perception.arm_mask import ARM_BODIES, ArmGeometry, joint_angles
    from so101_rl.safety.fk import KinematicChain, So101FK

    cal = load_calibration(CALIB)
    chain = KinematicChain.from_yaml(DEFAULT_CONFIG_PATH.with_name("so101_kinematics.yaml"))
    fk = So101FK(chain, cal)
    ticks = np.array([2100, 1800, 2300, 2500, 1900, 2048], np.float64)
    tip = he.MarkerFK(fk, cal)(ticks)
    assert tip == pytest.approx(fk.gripper_xyz_from_ticks(ticks, np.empty(3)))
    T = ArmGeometry().body_frames(joint_angles(ticks, cal))[ARM_BODIES.index("gripper")]
    assert (T @ np.append(chain.tip_T[:3, 3], 1))[:3] == pytest.approx(tip, abs=1e-9)  # the two FKs agree
    off = np.array([-0.01, 0.0, -0.004])
    assert he.MarkerFK(fk, cal, off)(ticks) == pytest.approx(tip + T[:3, :3] @ off, abs=1e-9)


def _pose(T, method="arm-shape", residual=4.5, trans=0.2, rot=0.05):
    from so101_rl.perception.camera_calib import CameraPose

    return CameraPose(T, method, "2026-10-10", residual, trans, rot)


def test_comparison_agree_keeps_the_lower_residual_disagree_investigates(calib):
    from so101_rl.sim.d435 import _axis_angle

    T = calib.T_base_color
    p_cam = np.random.default_rng(0).uniform([-0.1, -0.1, 0.4], [0.1, 0.1, 0.6], (12, 3))
    same = he.compare_with_camera_yaml(HandEye(T.copy(), 1.0, 12, "d", "s"), (0.5, 0.1), _pose(T), p_cam)
    assert same["agree"] and same["keep"] == "hand-eye" and "fold it into calib/camera.yaml" in same["recommendation"]
    worse = he.compare_with_camera_yaml(HandEye(T.copy(), 4.8, 12, "d", "s"), (0.5, 0.1), _pose(T), p_cam)
    assert worse["agree"] and worse["keep"] == "arm-shape"
    T2 = T.copy()
    T2[:3, :3] = T[:3, :3] @ _axis_angle(np.radians([0.0, 1.0, 0.0]))
    T2[:3, 3] += [0.004, 0.0, 0.0]
    off = he.compare_with_camera_yaml(HandEye(T2, 1.0, 12, "d", "s"), (0.5, 0.1), _pose(T), p_cam)
    assert not off["agree"] and off["keep"] is None and "investigate" in off["recommendation"]
    assert off["rotation_diff_deg"] == pytest.approx(1.0, abs=1e-6) and off["translation_diff_mm"] == pytest.approx(4.0)
    assert off["marker_point_displacement_mm"]["mean"] > 4.0


def test_bootstrap_uncertainty_shrinks_with_noise():
    rng = np.random.default_rng(1)
    p_cam = rng.uniform([-0.1, -0.1, 0.4], [0.1, 0.1, 0.6], (12, 3))
    T = np.eye(4)
    T[:3, 3] = [0.3, 0.3, 0.3]
    p_base = p_cam + T[:3, 3]
    clean = he.bootstrap_uncertainty(p_base, p_cam)
    noisy = he.bootstrap_uncertainty(p_base + rng.normal(0, 0.002, p_base.shape), p_cam)
    assert clean[0] < 1e-6 and clean[1] < 1e-4 and noisy[0] > 0.5 and noisy[1] > 0.05


def test_execute_refuses_without_a_servo_before_anything_moves(tmp_path, capsys):
    """No servo on the shm: refused (exit 2) after the camera opened (sim renders), nothing written."""
    from test_deploy_policy import _export

    code = he.main(["--execute", "--camera-backend", "sim", "--policy", str(_export(tmp_path / "p.pt")),
                    "--shm-name", "so101_he_none", "--wait-servo-s", "0", "--run-dir", str(tmp_path / "run"),
                    "--out", str(tmp_path / "hand_eye.yaml")])  # fmt: skip
    assert code == he.EXIT_REFUSED, capsys.readouterr().err
    assert "no servo process owns shm" in capsys.readouterr().err
    assert not (tmp_path / "hand_eye.yaml").exists() and not (tmp_path / "run").exists()


def test_execute_refuses_too_few_poses_and_a_missing_policy(tmp_path):
    assert he.main(["--execute", "--n-poses", "9"]) == he.EXIT_REFUSED
    assert he.main(["--execute", "--policy", str(tmp_path / "missing.pt"), "--wait-servo-s", "0"]) == he.EXIT_REFUSED


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_execute_against_the_mujoco_servo_recovers_the_camera(tmp_path, capsys):
    """The whole H8 path in sim: servo --backend mujoco, the 3 mm reach policy, SimRenderCamera following the servo's
    ticks with a 6 mm red sphere next to the tip (``--marker-offset-m``; ``--marker-radius-m`` defaults to the sphere's
    radius on the sim backend). The fit recovers the calibrated camera (the sim camera IS camera.yaml's pose; seen
    2026-10-17: 10/10 poses used, residual 0.13 mm, 0.011 deg / 0.20 mm off) and the comparison says AGREE."""
    if not REACH_POLICY.is_file():
        pytest.skip(f"the 3 mm reach export is not on this host ({REACH_POLICY}); set SO101_REACH_POLICY")
    from test_deploy_policy import ServoChild

    from so101_rl.perception.arm_registration import rotation_angle_deg

    calib = load_camera_calib()
    servo = ServoChild(tmp_path, "mujoco")
    out, run = tmp_path / "hand_eye.yaml", tmp_path / "run"
    try:
        code = he.main(["--execute", "--camera-backend", "sim", "--policy", str(REACH_POLICY), "--shm-name", servo.name,
                        "--wait-servo-s", "120", "--n-poses", "10", "--hold-s", "0.6", "--settle-s", "0.3",
                        "--marker-offset-m", *SIM_MARKER_OFFSET, "--run-dir", str(run), "--out", str(out)])  # fmt: skip
    finally:
        assert servo.stop() == 0
    err = capsys.readouterr().err
    print(err)
    assert code == 0, err
    fit = load_hand_eye(out)
    T = calib.T_base_color
    assert fit.n_poses == 10 and fit.residual_mm < 3.0
    assert rotation_angle_deg(fit.T_base_cam[:3, :3] @ T[:3, :3].T) < 0.2
    assert np.linalg.norm(fit.T_base_cam[:3, 3] - T[:3, 3]) * 1000 < 2.0
    report = json.loads((run / "report.json").read_text())
    poses = json.loads((run / "poses.json").read_text())
    assert report["written"] == str(out) and report["servo"]["backend"] == "mujoco" and not report["servo"]["real"]
    assert sum(p["used"] for p in poses) == 10 and all(p["fk_spread_mm"] <= 1.0 for p in poses if p["used"])
    assert report["comparison"]["agree"] and report["comparison"]["keep"] == "hand-eye"
    assert "stale-command stop" in err


class _Ctl:
    """Stand-in TipController: reach returns at once, hold reports fixed ticks (or reach raises ServoLeftRun)."""

    def __init__(self, ticks_seq=None, trip_at=None):
        self.ticks_seq = ticks_seq or [np.full(6, 2048.0)]
        self.trip_at = trip_at
        self.reaches = 0

    def reach(self, xyz):
        from so101_rl.rl.tip_control import ServoLeftRun

        self.reaches += 1
        if self.reaches == self.trip_at:
            raise ServoLeftRun("trip")
        return {"success": True, "final_error_m": 0.002, "steps": 20}

    def hold(self, seconds, on_step=None):
        info = None
        for t in self.ticks_seq:
            info = {"present_ticks": t}
            if on_step is not None:
                on_step(info)
        return info


class _Det:
    def __init__(self, xyz):
        self.xyz_cam = xyz


class _Grabber:
    """Stand-in FrameGrabber: a window of ``frames`` frames with the tracker's surface points ``xyz`` (filtered by the
    rig's accept, as the real grabber's tracker call is)."""

    def __init__(self, xyz, frames):
        self.xyz, self.frames, self.error = xyz, frames, None

    def begin(self, accept):
        self.accept = accept

    def end(self):
        w = he._Window(self.accept)
        w.frames = self.frames
        w.xyz = [x for x in self.xyz if self.accept(_Det(x))]
        return w


def _marker_fk(ticks):
    return np.array([0.0, 0.0, 0.5]) + (ticks[0] - 2048) * 1e-3  # the marker at camera z = 0.5 m (T = identity)


def _rig(xyz, frames, ctl=None, **kw):
    return he.ArmCameraRig(ctl or _Ctl(), _Grabber(xyz, frames), _marker_fk, np.eye(4), log=lambda m: None, **kw)


def test_rig_averages_and_gates_a_pose():
    p = np.array([0.0, 0.0, 0.5])
    rig = _rig([p + [0.0005, 0, 0], p - [0.0005, 0, 0]] * 9, 18)
    fk_xyz, cam = rig.measure(np.zeros(3))
    assert fk_xyz == pytest.approx(p) and cam == pytest.approx(p, abs=1e-9) and rig.records[0]["used"]
    assert _rig([p] * 12, 18).measure(np.zeros(3)) is None  # 67 % < 70 %
    far = p + [0.03, 0, 0]  # 30 mm from the FK marker: another red blob, not the marker
    assert _rig([far] * 18, 18).measure(np.zeros(3)) is None
    shaky = _rig([p + [0.003, 0, 0], p - [0.003, 0, 0]] * 9, 18)
    assert shaky.measure(np.zeros(3)) is None and "spread" in shaky.records[0]["skip_reason"]
    moving = _rig([p] * 18, 18, ctl=_Ctl([np.full(6, 2048.0), np.full(6, 2051.0)]))
    assert moving.measure(np.zeros(3)) is None and "moved" in moving.records[0]["skip_reason"]


def test_rig_round_marker_centre_is_behind_the_seen_surface():
    p = np.array([0.0, 0.0, 0.5])
    rig = _rig([p - [0, 0, 0.006]] * 18, 18, marker_radius_m=0.006)
    _, cam = rig.measure(np.zeros(3))
    assert cam == pytest.approx(p, abs=1e-9)


def test_collect_stops_on_a_trip_and_needs_ten_poses():
    from so101_rl.rl.tip_control import ServoLeftRun

    p = np.array([0.0, 0.0, 0.5])
    cands = np.random.default_rng(0).uniform(-0.1, 0.1, (30, 3))
    with pytest.raises(ServoLeftRun):
        he.collect_correspondences(_rig([p] * 18, 18, ctl=_Ctl(trip_at=3)), cands, n_poses=10)
    with pytest.raises(RuntimeError, match="only 0 usable poses"):
        he.collect_correspondences(_rig([], 18), cands, n_poses=10)
    pb, pc = he.collect_correspondences(_rig([p] * 18, 18), cands, n_poses=10)
    assert len(pb) == len(pc) == 10
