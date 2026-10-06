# SO-101 assets

- `upstream/` is the pinned, unmodified MuJoCo model from the upstream repo (see `UPSTREAM.json`, `LICENSE`, `NOTICE`;
  sha256 of every file is test-enforced). Never edit it; regenerate with `pixi run fetch-assets`.
- `so101_rl_scene.xml` is our override: it `<include>`s `upstream/so101_new_calib.xml` and adds
  - the table (plane at `TABLE_Z_M = -0.02`, see `src/so101_rl/sim/model.py`),
  - a 16 mm die (4 g, free joint), a 60 x 60 mm target `zone` site, a 30 mm `fiducial` site (ArUco-sized),
  - the `base` site (world origin) and the `rest` / `home` keyframes,
  - default classes `jaw_col` (primitive jaw collision geoms), `die` and `tray`,
  - the `overhead` camera: the nominal D435 pose of `sim/sim_tracker.py` `CameraModel` (0.614 m above the table at
    (0.20, 0), straight down, image x = base +x, image down = base -y, `fovy` from fy = 615 px at 640 x 480),
  - a static start-well `tray` body around `die_start` (0.20, -0.05) (decision D-TRAY, sim track S1.3): a flat
    23 x 23 mm well, four 50 deg funnel plates to +-30 mm and a 26 mm rim. Placeholder until H9 measures the real
    tray; `pixi run tray-check --drops 100` (`sim/tray_check.py`) is its sim twin (100/100 drops return to within
    10 mm of the well; 18/100 with the tray disabled). The funnel was 20 deg first: dice came to rest leaning on
    the slope edge (45 % returned), 40 deg returned 92.5 %.
  - a later `<compiler meshdir="upstream/assets">`, because upstream's `meshdir="assets"` is relative to the main file,
  - `<option timestep="1/510 s">` (decision D-TIMESTEP, 2026-10-05): `MujocoBus` runs `round((1/servo_hz)/dt) = 17`
    substeps per servo tick, so one tick is exactly 1/30 s of sim time and one policy step exactly 100 ms (with
    upstream's default 0.002 s a tick was 34 ms, a 2 % time-scale error). The CPU backend, the parity rig and the
    batched MuJoCo Warp env all use the same 17 substeps.

## MuJoCo Warp compatibility (S0.2)

- `jaw_col` has `margin="0"` (was 0.0005): MuJoCo Warp 3.14 rejects box-box / box-mesh contact pairs with a non-zero
  margin (`NotImplementedError`), and the CPU and GPU engines must simulate the same model. Spike:
  `plans/reviews/2026-10-05-mjwarp-feasibility-spike.md`.
- The 10 arm-link collision meshes (9k–27k vertices) run unmodified on MuJoCo Warp; no simplification is needed.
- Euler integrator, Newton solver and the pyramidal cone (MuJoCo defaults) are all supported by MuJoCo Warp.

## Always load through `so101_rl.sim.model.load_model()`

MJCF cannot add geoms or sites to a body that comes from an `<include>`, so `load_model()` compiles the scene through
`MjSpec` and adds, in code:

- four primitive box collision geoms (`jaw_fixed_tip`, `jaw_fixed_base`, `jaw_moving_tip`, `jaw_moving_base`, class
  `jaw_col`, sizes fitted to the jaw mesh vertices) on the fixed and moving jaw,
- `contype = conaffinity = 0` on the upstream mesh collision geoms of both jaws (plan sec. 6: no mesh-mesh grasp contact),
- the `gripper_tip` site (same frame as upstream `gripperframe`, the site `so101_kinematics.yaml` is generated from).

Collision bits: arm mesh geoms `1/1`, jaw primitives `2/2`, die `2/7`, table `3/3`, tray `4/0`. The die therefore
touches the table, the jaw primitives, the arm meshes and the tray; the jaw primitives touch only the die and the
table; the tray touches only the die. Deviation from the sim-track plan (tray "collides with the die and the
jaws"): the jaw boxes at the measured `rest` pose sweep through the tray area in the S0 CPU/Warp parity gate
(`tests/test_warp_model.py::test_step_matches_cpu_300_steps`, sinusoid around rest), and the jaw-box vs tray-box
contacts there diverged 7e-3 rad between CPU MuJoCo and MuJoCo Warp (gate: 1e-6). Until H9 the jaws pass through
the tray; revisit with the real tray geometry. The tray geoms have `priority="1"`, so a die/tray contact uses the
tray's friction (0.15) instead of the max of the pair.

## Placeholders (NEEDS H5)

`rest` and `home` keyframes are consistent with `config/so101.yaml` `safe_state` (`rest_ticks`: measured gravity-stable
pose, resting on the table; `home_ticks`: in-box placeholder, NEEDS H5; qpos and ctrl both encode the ticks through the
calibration). `tests/test_sim_model.py` fails if keyframes and config drift apart. The arm sags from `home` when torque
is off; it does not from `rest`.
Jaw box sizes and contact parameters (friction 1.2, condim 4) are first guesses until grasps are compared with hardware.
