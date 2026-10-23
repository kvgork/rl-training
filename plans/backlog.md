# Backlog

Ideas and requested work that is not yet planned in detail. When an item starts, it gets its own plan in `plans/`
and a decision-log row (`so101-self-learning-rl-plan.md` §10).

## Open

### Dashboard: training data + sim vs live camera (requested 2026-10-23)

One local web dashboard that shows:

- **Training data:** the sim training runs (`runs/sac-sim/*`, `runs/lab/*`): `metrics.csv` curves (success, return,
  DR / start stage, critic loss, eval Wilson bounds), review triggers, eval results and renders; real-arm runs
  (`runs/deploy/*`, `runs/feasibility/*`, `runs/sac-her*`): episodes, success per rung, final errors, trips.
- **Sim vs live camera view:** the live D435 colour (+ depth) stream next to a MuJoCo render of the same scene from
  the calibrated camera (`calib/camera.yaml`), with the arm posed from the servo's live joint state and the die from
  the perception object table; overlays: the arm mask, die detections (on-table / rejected), the zone, the fiducial
  check, and the per-pixel difference, so sim/real gaps (camera calibration, joint-zero errors, occlusion) are visible
  at a glance.

Constraints from the plan and lessons:

- Read-only: never writes the servo shm or touches the serial bus; reads the servo state and object table shm only.
- Must not starve the servo loop (no real-time scheduling on the host): cap thread pools, render at a low rate
  (e.g. 2–5 Hz), and never start it while the arm is moving (start it before the servo, like perception).
- The camera has one owner: the live view should come from the perception process (e.g. a frame shared through shm
  or a small HTTP endpoint), not a second camera open.

Existing pieces to reuse: `pixi run dashboard` (TensorBoard over `runs/`), `viz/video.py` and the eval renders,
`perception/camera_overlay.py`, `perception/sim_camera.py` (renders from the calibrated camera with the arm set from
joint ticks), `perception/arm_mask.py`.
