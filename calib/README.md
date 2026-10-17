# calib/

Committed calibration artifacts for the perception stack. They are produced on the real rig and are
**refused at load time when they fail their own gates**; there is no fallback to defaults.

## `camera.yaml`

The ONE source of the D435 model (camera parity unit SUC, plan §10 2026-10-06 "Camera parity"): the sim cameras
`d435_color` / `d435_depth`, the sim tracker, the rollout videos, the real tracker and `perception.frames` all read
it (`so101_rl.perception.camera_calib.load_camera_calib`; `tests/test_camera_single_source.py` fails on hard-coded
intrinsics or poses). Written by `pixi run camera-intrinsics --serial 819312071240 --out calib/camera.yaml`
(device block; keeps an existing pose) and by `pixi run camera-calibrate --method arm-shape` (pose + table plane,
only when its gate passes). Frames: RealSense optical (x right, y down, z forward), metres.

```yaml
version: 1
serial: "819312071240"
firmware: 5.11.1.100
color: {width: 640, height: 480, fps: 30, fx: .., fy: .., cx: .., cy: .., distortion: {model: inverse_brown_conrady, coeffs: [0, 0, 0, 0, 0]}}
depth: {width: 640, height: 480, fps: 30, fx: .., fy: .., cx: .., cy: .., distortion: {...}, depth_scale: 0.001}
depth_to_color: [[4x4]]       # p_color = T @ p_depth (librealsense get_extrinsics_to; its rotation is column-major)
T_base_color:                 # null = never calibrated
  matrix: [[4x4]]             # colour frame -> robot base frame
  method: arm-shape | hand-eye | provisional
  date: "YYYY-MM-DD"
  residual_mm: 1.8            # null only for provisional; a calibrated pose above 5 mm is refused
  uncertainty: {trans_mm: .., rot_deg: ..}   # 2 sigma bootstrap; the camera-pose DR uses max(this, 5 mm / 1.5 deg)
  notes: "..."
table_plane:                  # null = not measured
  normal_color: [3]           # unit, pointing UP (towards the camera), colour frame
  height_m: 0.3457            # camera centre above the table
  inlier_fraction: 0.35
  n_frames: 5
  date: "2026-10-06"
```

Strict: unknown or missing keys are errors; non-zero distortion coefficients are refused (every consumer is
pinhole). **Current state (2026-10-07):** device intrinsics; a PROVISIONAL pose (height and tilt from the
2026-10-06 table plane, yaw and x/y read off a colour frame; see its `notes`) until `camera-calibrate` passes on the
rig. To regenerate the provisional pose: `python -m so101_rl.perception.camera_calib --provisional ...` (all
numbers are arguments; it refuses to replace a calibrated pose).

## `distractors.yaml`

Red/orange blobs the HSV die tracker would take for the die (C.5), written by `pixi run camera-distractors` (a
colour frame WITHOUT the die; `--from-npz` for a saved capture). Read by
`so101_rl.sim.sim_tracker.DistractorModel.from_yaml` (the sim tracker's false-positive model).

```yaml
version: 1
date: "2026-10-07"
source: "camera-distractors ..., what else was in the frame"
camera_pose_method: provisional     # the pose used for xyz_base (a provisional pose makes them provisional)
hsv: {hsv_lo: [..], hsv_hi: [..], min_area_px: 40, max_area_px: 4000}   # perception/hsv_red_die.yaml at the time
distractors:
  - {label: blob0, u: 357.5, v: 300.4, area_px: 306, xyz_base: [x, y, z], xyz_source: depth | table-ray}
```

## `hand_eye.yaml`

Written by `pixi run -e cpu hand-eye --execute --attended` (`perception/calibrate_hand_eye.py`, H8), read by
`so101_rl.perception.frames.load_hand_eye`. The run always writes `runs/hw/hand-eye-<ts>/` (`hand_eye.yaml`,
`poses.json` per pose, `report.json` with the fit, the per-pose residuals and the comparison with `camera.yaml`'s
pose); it writes `calib/hand_eye.yaml` only when residual <= 3 mm and n >= 10. It never writes `camera.yaml`:
folding an agreeing, lower-residual hand-eye pose into it (method `hand-eye`) is a manual, reviewed step.

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

Written by `pixi run fiducial-capture` (`perception/fiducial.py`, H8: arm at rest or out of view, no servo), read by
`so101_rl.perception.fiducial.FiducialCheck.from_yaml` (the perception process's camera-moved gate when the file
exists, and `pixi run camera-moved-check`). The marker: ArUco `DICT_4X4_50` id 0, a 50 mm black square printed from
`scripts/make_fiducial_print.py`, glued flat with its centre at base (0.09, 0.05), sides along the base x / y axes,
top edge towards +x (plan §10 2026-10-17 "H8 software"; the MJCF `fiducial` site is the same square).

```yaml
dictionary: DICT_4X4_50        # name of an attribute of cv2.aruco
marker_id: 0
tol_px: 2.0                    # mean corner distance above this = camera (or marker) moved; not found = no evidence
size_mm: 50.0
xy_base: [0.09, 0.05]          # marker centre, base frame (informational: the gate compares pixels)
expected_corners_px:           # TL, TR, BR, BL as returned by cv2.aruco.ArucoDetector, mean over the capture frames
  - [487.1, 462.0]
  - [451.3, 416.6]
  - [505.8, 398.4]
  - [545.4, 440.9]
date: "2026-10-17"
camera_serial: "819312071240"  # the perception process refuses a fiducial recorded with another camera
camera_yaml_sha256: "..."      # calib/camera.yaml at capture time (informational)
frames: 30                     # capture statistics: frames read, frames with the marker,
found_frames: 30
jitter_px: 0.05                # worst corner's rms distance from its mean (capture refused above 0.5 px)
prior_err_px: 1.2              # mean distance to the corners camera.yaml's pose predicts for this marker (informational)
```

## `sim_params.yaml`

Simulator parameters (FS7 of `plans/sim-track-implementation-plan.md`), read by
`so101_rl.sim.sysid.params.SimParams.from_yaml` / `read_sim_params_yaml` and by `compare-real --sim-params`.
Release 0 (shipped now) is the MJCF prior, regenerated with
`python -m so101_rl.sim.sysid.params --write-prior calib/sim_params.yaml`; release 1 is written by
`sysid-fit` from the HS1/HS2 excitation runs (HS3), and only when the held-out gate passes.

```yaml
version: 2                 # schema version (SIM_PARAMS_VERSION; 2 since 2026-10-06); any other value is refused
meta:
  version: 0               # parameter release: 0 = MJCF prior, >= 1 = fitted on the real arm (HS3)
  date: "2026-10-06"
  git_sha: "..."
  mjcf_sha256: "..."       # sha256 of so101_rl_scene.xml + assets/so101/upstream/*.xml
  excitation_run: "runs/sysid/<ts>"
  fit_rms_pct: {joint: pct}        # training RMS, % of calibrated range
  heldout_rms_pct: null | {joint: pct}   # compare-real definition (open-loop replay); null = never verified
  spread: {"servo_group.shoulder_pan": 0.12, ..., servo_group: max}   # log-space spread; > 0.5 = unidentified
  backend: mjcf-prior | warp | cpu
  unidentified: [frictionloss, ...]       # never to be read as precise values (frictionloss: always)
  notes: [...]             # optional
params:                    # absolute values, per-joint lists in JOINT_NAMES order (what the simulator gets)
  kp: [6]; kv: [6]; damping: [6]; frictionloss: [6]; armature: [6]; forcerange: [6]
  delay_ticks: 0           # whole servo ticks of command-to-motion delay, 0..6
  deadband_ticks: [6]      # target hysteresis (D-BACKLASH), 0 = none
  die_mass_kg: 0.004
  die_friction: 1.0
  table_friction: 0.8
servo_group:               # kp, kv, armature = prior x scale: ONE ratio group per joint (plan §10 2026-10-06)
  prior: {kp: [6], kv: [6], armature: [6]}
  scale: [6]               # the fitted factor (1 = the prior); checked to reproduce params.kp/kv/armature
dr:                        # domain-randomisation ranges for S3 (plan §10 2026-10-06)
  servo_group_rel: 0.4     # the group x U(1 - 0.4, 1 + 0.4) per joint and episode, around the fit
  frictionloss_range: [0.0, 0.21]   # N m, uniform, every joint (frictionloss is not identified)
  frictionloss_source: "0 .. 4 x the MJCF prior 0.052 N m = 7 % of the STS3215 12 V stall torque 2.94 N m ..."
```

`from_yaml` refuses the file unless every joint has `heldout_rms_pct <= 1` (`compare_real.MAX_RMS_PCT`, 1 % since
plan §10 2026-10-06; it was 5 %); pass `allow_unverified=True` to use the release-0 prior (reach training on v0;
die-to-zone waits for release >= 1, plan D-DIE-PARAMS). Keys are strict: a missing or unknown key is an error, and a
`servo_group` that does not reproduce `params` is refused.

**Semantics (schema 2).** The sysid fits one scale per joint applied to kp, kv and armature jointly (through the
clamp chain they act almost only through their ratios) plus the integer delay; frictionloss is invisible under
±1-tick noise, stays at the prior and is listed in `meta.unidentified`. The DR ranges travel with the release: the
group ±40 % around the fit and frictionloss over `[0, 0.21]` N m. That upper bound is a choice bounded by two
sourced numbers, not a measurement: 4 x the upstream MJCF prior (0.052 N m, class `sts3215` of TheRobotStudio's
`so101_new_calib.xml`) = 7 % of the STS3215 stall torque at 12 V (30 kg cm = 2.94 N m, the class `forcerange`).
