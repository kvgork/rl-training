# rl-trainer (so101_rl)

- The plan is `so101-self-learning-rl-plan.md`. Its §10 decision log is the source of truth. Do not re-litigate a logged decision without the user.
- The architecture doc is `docs/ARCHITECTURE.md`. Update it in the **same commit** as any change to modules, processes, interfaces, safety-check order or pixi features (see its §10).
- Implementation plans live in `plans/`.
- `safety/` must never import `rl`, `wm` or `env`. Only the servo process touches the serial bus.
- Hardware-dependent code needs a fake backend, so that `pixi run test` passes without the arm.
