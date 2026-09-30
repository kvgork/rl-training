# SO-101 assets

- `upstream/` is the pinned, unmodified MuJoCo model from the upstream repo (see `UPSTREAM.json`, `LICENSE`, `NOTICE`;
  sha256 of every file is test-enforced). Never edit it; regenerate with `pixi run fetch-assets`.
- `so101_rl_scene.xml` is our override: it `<include>`s `upstream/so101_new_calib.xml` and adds
  - the table (plane at `TABLE_Z_M = -0.02`, see `src/so101_rl/sim/model.py`),
  - a 16 mm die (4 g, free joint), a 60 x 60 mm target `zone` site, a 30 mm `fiducial` site (ArUco-sized),
  - the `base` site (world origin) and the `rest` / `home` keyframes,
  - default classes `jaw_col` (primitive jaw collision geoms) and `die`.
  - a later `<compiler meshdir="upstream/assets">`, because upstream's `meshdir="assets"` is relative to the main file.

## Always load through `so101_rl.sim.load_model()`

MJCF cannot add geoms or sites to a body that comes from an `<include>`, so `load_model()` compiles the scene through
`MjSpec` and adds, in code:

- four primitive box collision geoms (`jaw_fixed_tip`, `jaw_fixed_base`, `jaw_moving_tip`, `jaw_moving_base`, class
  `jaw_col`, sizes fitted to the jaw mesh vertices) on the fixed and moving jaw,
- `contype = conaffinity = 0` on the upstream mesh collision geoms of both jaws (plan sec. 6: no mesh-mesh grasp contact),
- the `gripper_tip` site (same frame as upstream `gripperframe`, the site `so101_kinematics.yaml` is generated from).

Collision bits: arm mesh geoms `1/1`, jaw primitives `2/2`, die `2/3`, table `3/3`. The die therefore touches the table,
the jaw primitives and the arm meshes, and the jaw primitives touch only the die and the table.

## Placeholders (NEEDS H5)

`rest` and `home` keyframes are placeholders consistent with `config/so101.yaml` `safe_state` (tick 2047 on every joint:
0 rad on the arm joints, 2.4 % open on the gripper with the current calibration). They are gravity-stable in sim only;
the real rest pose is measured in H5. `tests/test_sim_model.py` fails if keyframes and config drift apart.
Jaw box sizes and contact parameters (friction 1.2, condim 4) are first guesses until grasps are compared with hardware.
