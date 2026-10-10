"""Does ``camera-calibrate --poses 2`` tell a tilted base from joint-zero errors? (2026-10-10, rig hypotheses (a)/(b))

Synthetic D435 frames from the camera pose of ``calib/camera.yaml``, the arm at the rest pose and at the rest pose
with the base turned 60 deg (the hand-moved pose ``camera-calibrate`` asks for). Two worlds:

* **tilted base**: the base pitched 3.5 deg about base y relative to the table (table and camera rotate together about
  the base origin), FK exact, table through the assumed height at the base;
* **joint offsets**: a level base, the table 15 mm higher than assumed (z = -5 mm), and the arm's true angles
  shoulder_lift -5 deg / wrist_flex -2.5 deg off the probed ones.

``diagnostics.tilt_vs_offsets`` fits the camera pose (6 DOF) and the joint-zero offsets together and must name the
case; with the first pose alone it must say ``ambiguous`` (the level-base alternative then explains the tilt with a
~ -5 deg shoulder_lift error, the signature of the 2026-10-09 rig runs).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from camera_synth import render_rig, rot_y, with_pose

from so101_rl.config import load_calibration
from so101_rl.perception import camera_calibrate
from so101_rl.perception.arm_registration import MAX_SEPARATION_VIF, OFFSET_JOINTS, ArmShapeRegistration
from so101_rl.perception.camera_calib import load_camera_calib
from so101_rl.perception.camera_calibrate import Capture, calibrate
from so101_rl.sim.arm_cloud import qpos_from_ticks
from so101_rl.sim.model import TABLE_Z_M, load_model

CAL = load_calibration(Path(__file__).parent / "data" / "so101_follower.calib.json")
REST = np.array([2050, 2200, 2301, 2745, 2040, 2065], dtype=np.int32)
TURNED = REST.copy()
TURNED[0] += round(camera_calibrate.SECOND_POSE_TURN_DEG * 4096 / 360)
POSES = (REST, TURNED)
TILT_DEG = 3.5
SL_DEG, WF_DEG = -5.0, -2.5
TABLE_DZ_M = 0.015


def _captures(case: str) -> list[Capture]:
    c0 = load_camera_calib()
    model = load_model(camera=False)[0]
    R, dz, delta = np.eye(3), 0.0, np.zeros(len(OFFSET_JOINTS))
    if case == "tilt":
        R = rot_y(TILT_DEG)
    else:
        dz = TABLE_DZ_M
        delta[OFFSET_JOINTS.index("shoulder_lift")] = math.radians(SL_DEG)
        delta[OFFSET_JOINTS.index("wrist_flex")] = math.radians(WF_DEG)
    T = np.eye(4)
    T[:3, :3] = R @ c0.T_base_color[:3, :3]
    T[:3, 3] = R @ c0.T_base_color[:3, 3] + (0.0, 0.0, dz)
    adr = [model.jnt_qposadr[model.joint(n).id] for n in OFFSET_JOINTS]
    caps = []
    for k, ticks in enumerate(POSES):
        q = qpos_from_ticks(model, ticks, CAL)
        q[adr] += delta  # the arm really stands at probed + delta
        rgb, frames = render_rig(with_pose(c0, T), q, R_world=R, table_dz_m=dz, seed=k)
        caps.append(Capture(rgb, frames, ticks))
    return caps


@pytest.fixture(scope="module", params=["tilt", "joints"])
def fitted(request):
    c0 = load_camera_calib()
    model = load_model(camera=False)[0]
    res, plane, obs = calibrate(model, CAL, c0, _captures(request.param), TABLE_Z_M, bootstrap=0, workspace_plane=False)
    return request.param, res, plane, obs, c0


@pytest.mark.slow
def test_two_poses_name_the_case(fitted):
    case, res, *_ = fitted
    t = res.diagnostics["tilt_vs_offsets"]
    print(camera_calibrate.tilt_vs_offsets_line(res))
    assert t["separation_vif"] <= MAX_SEPARATION_VIF
    o = t["offsets_deg"]
    if case == "tilt":
        assert t["verdict"] == "tilted_base"
        assert t["tilt_about_base_y_deg"] == pytest.approx(TILT_DEG, abs=0.3)
        assert abs(t["tilt_about_base_x_deg"]) < 0.3
        assert abs(t["table_z_minus_assumed_mm"]) < 2.0
        assert all(abs(o[j]) < 0.5 for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))
    else:
        assert t["verdict"] == "joint_offsets"
        assert t["tilt_deg"] < 0.3
        assert t["table_z_minus_assumed_mm"] == pytest.approx(1000 * TABLE_DZ_M, abs=2.0)
        assert o["shoulder_lift"] == pytest.approx(SL_DEG, abs=0.4)
        assert o["wrist_flex"] == pytest.approx(WF_DEG, abs=0.4)
        assert abs(o["elbow_flex"]) < 0.4
    assert all(s > 0 for s in t["offsets_sigma_deg"].values()) and t["tilt_sigma_deg"]["y"] > 0
    # the full model explains the frames at least as well as either single explanation
    assert t["rms_mm"] <= min(res.rms_mm, res.diagnostics["level_base_alternative"]["rms_mm"]) + 0.05


@pytest.mark.slow
def test_one_pose_is_ambiguous(fitted):
    """The first (rest) pose alone: the tilt about base y and a shoulder_lift offset trade (VIF ~3), so the verdict
    must be ``ambiguous``, never a confident wrong name."""
    case, res, plane, obs, c0 = fitted
    reg = ArmShapeRegistration(obs[:1], plane, c0.color, TABLE_Z_M)
    t = reg.tilt_vs_offsets(res.T_base_color)
    assert t is not None and t["verdict"] == "ambiguous", t
    assert t["separation_vif"] > MAX_SEPARATION_VIF
    if case == "tilt":
        # the rig's signature: a level base with a ~ -5 deg shoulder_lift error fits a 3.5 deg tilt almost as well
        alt = reg.level_base_alternative(np.array(res.seeds[0]["T"]))  # from the plane-constrained (level) pose
        assert alt is not None and alt["offsets_deg"]["shoulder_lift"] < -3.0


def test_hand_pose_prompt_asks_for_the_separating_turn():
    p = camera_calibrate.hand_pose_prompt(1, 2)
    assert "pose 2/2" in p and "turn ONLY the base about 60 deg" in p and "at least 45 deg" in p and "torque stays off" in p
    assert "folded" not in p and "stays put by itself" in p and "including the gripper" in p and "out of view" in p
    assert "other way" in camera_calibrate.hand_pose_prompt(2, 3)
