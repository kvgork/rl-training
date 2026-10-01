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
