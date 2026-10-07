# Camera parity unit (SUC): make the sim camera identical to the real D435

The user asked for this on 2026-10-06: *"make sure the camera setup in sim is the same as the one in real"*. The decisions are in plan §10, row "2026-10-06 | Camera parity". The pose comes from camera-only calibration (table plane plus arm-shape registration), and the current oblique mount is kept. H8 hand-eye calibration refines the pose later.

**Base:** `main` after sim wave 1 is merged.

**Hardware rules:**
- No arm motion.
- The camera may be read freely.
- Servo angles may be read only through a read-only probe in `servo/` (see C.2). Only the servo package touches the bus (CLAUDE.md).
- Never write a register, and never enable torque.

## What is measured already (2026-10-06, device and camera only)

**Device:** D435, serial 819312071240, firmware 5.11.1.100.

**Colour stream:**

| Setting | Value |
|---|---|
| Resolution | 640×480 @ 30 Hz, rgb8 |
| Focal length | fx 617.17, fy 617.57 |
| Principal point | cx 327.95, cy 236.12 |
| Distortion | inverse Brown–Conrady, all coefficients 0 |

**Depth stream:**

| Setting | Value |
|---|---|
| Resolution | 640×480 @ 30 Hz, z16 |
| Focal length | fx = fy = 388.33 |
| Principal point | cx 326.99, cy 240.87 |
| Distortion | 0 |
| Scale | 0.001 m per unit |

**Offset between sensors:** the depth-to-colour translation is (14.76, 0.13, 0.13) mm.

**Table plane** (RANSAC on aligned depth, 5 frames):
- normal in the camera frame: (0.110, 0.892, 0.439);
- camera height above the table: **345.7 mm** (±4 mm across frames);
- optical axis **64° from vertical**: the camera looks across the table, not down at it;
- inlier fraction about 35 %, because the scene is cluttered.

**Scene:**
- a window with strong daylight behind the arm;
- clutter, such as the bed and a mouse;
- red servo cables and an orange tip marker near the workspace, which are **false-positive risks for the red-die HSV tracker**;
- the arm fills much of the frame, so the die will often be hidden behind it.

## Tasks

### C.1 Single source of truth: `calib/camera.yaml` (SW plus a camera read)
- **Schema:** a strict-loaded config with:
  - `serial`;
  - `color` and `depth` blocks (width, height, fps, fx, fy, cx, cy, distortion model and coefficients);
  - `depth_to_color`;
  - `T_base_color` as a 4×4 matrix plus `{method, date, residual_mm, uncertainty: {trans_mm, rot_deg}}`;
  - `table_plane`.
- **Command:** `pixi run camera-intrinsics --serial 819312071240 --out calib/camera.yaml` reads the factory intrinsics with pyrealsense2, streaming only. A test runs it on a fake device.
- **Every consumer reads this file:** the sim camera (C.3), `SimDieTracker`, `viz/video.py` (replacing `default_synthetic_camera_pose`), the real `HsvDepthDieTracker` and `perception/frames`. A grep test checks that no hard-coded intrinsics or camera pose remain.

### C.2 Camera-only extrinsics: table plane and arm-shape registration (SW; reads the camera and the servo angles)
1. **Read-only joint probe** (`servo/probe.py`).
   - Ping and block-read the 6 STS3215s and return the present ticks. It structurally refuses any write: no write method is exposed, and a FakeBus test proves zero writes.
   - It refuses if the servo process holds the port lock.
2. **Table plane.** RANSAC on several aligned depth frames (port `fit_table_plane` from LIT `_object_pose_depth.py`, credited). This fixes camera height, roll and pitch relative to the table, whose base-frame z is known (`table_z_m`).
3. **Arm-shape registration.**
   - Render the arm's collision and visual meshes at the probed joint angles with MuJoCo (`load_model`, FK), and sample a base-frame point cloud from the surfaces the camera can see.
   - Segment the arm in the depth cloud: the points above the table plane, inside a generous box around the arm's expected projection.
   - Point-to-plane ICP, initialised from several yaw and x/y seeds and constrained by the plane fit (search the remaining 3 DOF first, then refine all 6).
   - Repeat over K ≥ 3 arm poses **only if the user moves the arm by hand with torque off**, which is optional and attended. One pose is the minimum.
4. **Output and gate.**
   - Write `T_base_color` with its residual and a bootstrap uncertainty.
   - **Gate:** RMS ICP residual ≤ 5 mm; the plane-implied height and tilt agree with the 6-DOF result within 5 mm and 2°; projecting the FK gripper tip into the colour image lands on the arm's tip pixels within 8 px.
   - Otherwise refuse, and keep the previous file.
5. **Command:** `pixi run camera-calibrate --method arm-shape [--poses K]`. It writes `runs/camera-calib/<ts>/` (point clouds, overlay PNGs) and updates `calib/camera.yaml` only when the gate passes.

### C.3 Sim camera equals the real camera (SW + GPU)
- **Scene cameras:**
  - **Colour camera:** add `d435_color` to the scene from `camera.yaml`, using MuJoCo camera intrinsics (`resolution`, `focalpixel`, `principalpixel` and `sensorsize`, or the equivalent supported by MuJoCo 3.14; verify in the installed source). Pose = `T_base_color`.
  - **Depth camera:** add `d435_depth` at `T_base_color · inv(depth_to_color)`, with its own intrinsics.
  - **Coordinate convention:** MuJoCo cameras look along −z with +y up, while the RealSense optical frame looks along +z with +y down. Convert explicitly, and test the conversion.
- **Consumers:**
  - `SimDieTracker` projects and ray-casts with the depth/colour camera models from `camera.yaml`, so occlusion is computed from the real viewpoint.
  - The videos' "overhead" view becomes the `d435_color` view. Rename "overhead" to "d435" in user-facing text, since the real camera is oblique.
- **Domain randomisation (S3 priority 1):** the camera pose is randomised around `T_base_color` within the calibration uncertainty, at least ±5 mm and ±1.5° until H8. The intrinsics are not randomised, because they are factory-calibrated.

### C.4 Parity tests (SW + GPU)
- **Projection parity:** for 200 random 3D points in the workspace, the pinhole projection with the `camera.yaml` intrinsics and the MuJoCo-rendered pixel of a small marker sphere agree to within **0.5 px**, for colour and for depth.
- **Depth parity:** the rendered sim depth at known table and die points equals the analytic depth within 1 mm.
- **Tracker parity:** for the same scene rendered in sim, `HsvDepthDieTracker` (the real code, run on the sim RGB-D render) and `SimDieTracker` agree on the die centroid within 3 mm and on visibility in at least 18 of 20 configurations. This extends the existing test from the synthetic pose to the calibrated pose.
- **Real-vs-sim overlay (artifact for the user):** `pixi run camera-overlay` captures one real D435 frame, probes the joint angles, renders the sim from `d435_color` at those angles, and writes the real frame, the sim render and a 50 % blend side by side to `runs/camera-overlay/<ts>.png`. The arm silhouettes should coincide. It also reports the IoU of the real and sim arm masks: **gate ≥ 0.8**.

### C.5 Scene realism for the tracker (SW)
- **Distractors:** the sim tracker's false-positive model includes red and orange distractors at the measured image regions (servo cables, the orange tip marker). Train-time DR adds random red distractor blobs.
- **Lighting:** a back-lit lighting model (a bright window behind the scene), for the pixel world-model track only.
- **Recommendation to the user:** remove the orange tip marker, or tape over it, if it isn't needed. Better still, route the red cables out of the workspace.

### C.6 H8 refinement (hardware-gated; plan only)
- **Port the LIT hand-eye method into `perception/calibrate_hand_eye --execute`:**
  - constant orientation;
  - tip found by motion;
  - Kabsch/Umeyama fit over at least 8 poses;
  - expected accuracy 5–15 mm.
- **Run it through ServoCore,** attended, only after H2/H3 pass.
- **Combine with the arm-shape result:**
  - If the two agree within the stated uncertainty, keep the one with the lower residual and shrink the DR range.
  - If they disagree, investigate before training any die task.

## Acceptance
- `pixi run camera-calibrate` passes its gate on the real rig, and `calib/camera.yaml` is committed.
- `pixi run camera-overlay` reaches IoU ≥ 0.8, and the user has checked the overlay PNG.
- The projection parity of 0.5 px or better and the tracker parity tests pass.
- No hard-coded camera pose or intrinsics remain (the grep test).
- ARCHITECTURE §4/§5/§7 and plan §10 are updated.

**Estimate:** about 8 agent-hours, plus about 15 minutes of the user's time for the overlay check (and an optional hand-moved second arm pose).
