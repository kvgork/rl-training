"""``So101FK`` against MuJoCo on the RL scene (plan Task 3.5).

``tests/test_fk.py`` already checks the FK chain against the upstream MJCF's ``gripperframe`` with joint angles in
radians. This test closes the loop the servo actually uses: random in-range **ticks** -> ``So101FK`` (through the
calibration's ``drive_mode`` signs) vs ticks -> ``MujocoBus.ticks_to_q`` -> ``mj_kinematics`` on the compiled
``so101_rl_scene.xml`` -> the ``gripper_tip`` site that ``sim.model.load_model`` adds.
"""

import dataclasses

import mujoco
import numpy as np

from so101_rl.config import JOINT_NAMES
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

N_POSES = 500


def _max_err(cfg, calibration, seed=0):
    model, data = load_model()
    bus = MujocoBus(model, data, calibration, cfg)
    fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
    site = model.site("gripper_tip").id
    rng = np.random.default_rng(seed)
    lo, hi = cfg.joints.min_ticks, cfg.joints.max_ticks
    xyz = np.zeros(3)
    worst = 0.0
    for _ in range(N_POSES):
        ticks = rng.integers(lo, hi + 1).astype(np.int32)
        data.qpos[bus.qpos_adr] = bus.ticks_to_q(ticks)
        mujoco.mj_kinematics(model, data)
        fk.gripper_xyz_from_ticks(ticks, xyz)
        worst = max(worst, float(np.abs(xyz - data.site_xpos[site]).max()))
    return worst


def test_fk_matches_mujoco_gripper_tip_on_500_random_tick_poses(cfg, calibration):
    assert _max_err(cfg, calibration) < 1e-6


def test_fk_matches_mujoco_with_a_flipped_drive_mode(cfg, calibration):
    flipped = {n: dataclasses.replace(c, drive_mode=1 - c.drive_mode) if n in JOINT_NAMES[:3] else c
               for n, c in calibration.items()}
    assert _max_err(cfg, flipped, seed=1) < 1e-6


def test_the_comparison_is_not_vacuous(cfg, calibration):
    """A wrong sign on one joint must show up as a large error."""
    model, data = load_model()
    bus = MujocoBus(model, data, calibration, cfg)
    wrong = {n: dataclasses.replace(c, drive_mode=1 - c.drive_mode) if n == "shoulder_lift" else c
             for n, c in calibration.items()}
    fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), wrong)
    ticks = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    data.qpos[bus.qpos_adr] = bus.ticks_to_q(ticks)
    mujoco.mj_kinematics(model, data)
    err = np.abs(fk.gripper_xyz_from_ticks(ticks, np.zeros(3)) - data.site_xpos[model.site("gripper_tip").id]).max()
    assert err > 0.01
