# calib/

Committed calibration artifacts for the perception stack. Both files are produced on the real rig (H8) and are
**refused at load time when they fail their own gates**; there is no fallback to defaults.

## `hand_eye.yaml`

Written by `python -m so101_rl.perception.calibrate_hand_eye --execute`, read by
`so101_rl.perception.frames.load_hand_eye`.

```yaml
T_base_cam:            # 4x4, camera frame -> robot base frame, metres; proper rotation, last row [0, 0, 0, 1]
  - [r00, r01, r02, tx]
  - [r10, r11, r12, ty]
  - [r20, r21, r22, tz]
  - [0.0, 0.0, 0.0, 1.0]
residual_mm: 1.2       # rms distance between FK gripper-tip points and camera points after the Umeyama fit
n_poses: 12            # number of (FK xyz, camera xyz) correspondences used
date: "2026-09-30"     # ISO date of the calibration
camera_serial: "819312071240"
```

`load_hand_eye` raises `ValueError` if `residual_mm > 3.0`, `n_poses < 10`, a key is missing, or `T_base_cam` is not a
finite rigid transform.

## `fiducial.yaml`

Read by `so101_rl.perception.fiducial.FiducialCheck.from_yaml`. The marker is fixed to the table inside the
camera's view. `expected_corners_px` is what the detector reports right after the hand-eye calibration.

```yaml
marker_id: 7
dictionary: DICT_4X4_50        # name of an attribute of cv2.aruco
tol_px: 2.0                    # mean corner distance above this (or marker not found) = camera moved, halt training
expected_corners_px:           # TL, TR, BR, BL as returned by cv2.aruco.ArucoDetector
  - [200.0, 150.0]
  - [320.0, 150.0]
  - [320.0, 270.0]
  - [200.0, 270.0]
```

## `sim_params.yaml`

Simulator parameters (FS7 of `plans/sim-track-implementation-plan.md`), read by
`so101_rl.sim.sysid.params.SimParams.from_yaml` / `read_sim_params_yaml` and by `compare-real --sim-params`.
Release 0 (shipped now) is the MJCF prior, regenerated with
`python -m so101_rl.sim.sysid.params --write-prior calib/sim_params.yaml`; release 1 is written by
`sysid-fit` from the HS1/HS2 excitation runs (HS3), and only when the held-out gate passes.

```yaml
version: 1                 # schema version (SIM_PARAMS_VERSION); any other value is refused
meta:
  version: 0               # parameter release: 0 = MJCF prior, >= 1 = fitted on the real arm (HS3)
  date: "2026-10-06"
  git_sha: "..."
  mjcf_sha256: "..."       # sha256 of so101_rl_scene.xml + assets/so101/upstream/*.xml
  excitation_run: "runs/sysid/<ts>"
  fit_rms_pct: {joint: pct}        # training RMS, % of calibrated range
  heldout_rms_pct: null | {joint: pct}   # compare-real definition (open-loop replay); null = never verified
  spread: {"kp.shoulder_pan": 0.12, ..., kp: max}   # log-space spread; > 0.5 = unidentified (wide DR in S3)
  backend: mjcf-prior | warp | cpu
  notes: [...]             # optional
params:                    # per-joint lists in JOINT_NAMES order
  kp: [6]; kv: [6]; damping: [6]; frictionloss: [6]; armature: [6]; forcerange: [6]
  delay_ticks: 0           # whole servo ticks of command-to-motion delay, 0..6
  deadband_ticks: [6]      # target hysteresis (D-BACKLASH), 0 = none
  die_mass_kg: 0.004
  die_friction: 1.0
  table_friction: 0.8
```

`from_yaml` refuses the file unless every joint has `heldout_rms_pct <= 5` (`compare_real.MAX_RMS_PCT`); pass
`allow_unverified=True` to use the release-0 prior (reach training on v0; die-to-zone waits for release >= 1, plan
D-DIE-PARAMS). Keys are strict: a missing or unknown key is an error.
