# Die-to-zone implementation plan (Phase 4, first target task)

Drafted 2026-10-15 after the reach skill reached its ceiling at 5 mm (plan §10 "AWR round 2: no gain"). The user chose "harder reach target, then the next skill"; this plan covers the next skill. The task is plan Phase 4: move the red 16 mm die into the marked 60 mm zone; push and place both count.

## Where it stands (read-only survey, 2026-10-15)

**Already built:**
- `DIE_TO_ZONE` task (`env/tasks.py`): reward, success predicate, settle debounce and visibility rule. `die_start` is still a placeholder (`measured_on_hardware=False`).
- Batched GPU env die support: placement, zone goal, simulated tracker with occluders (`env/batched_env.py`).
- MJCF die, zone and placeholder tray. The sim tray-return check passes 100/100 (`runs/tray/20261007-183111.json`).
- Jaw-to-die feasibility on the meshes (`tests/test_die_reach_feasibility.py`).
- The `HsvDepthDieTracker` library and `calib/camera.yaml` (arm-shape pose, 4.5 mm residual).
- The `So101GoalEnv` die path and `ObjectTableShm`.

**Missing:**
- Every entry point accepts reach only: `eval_sim.TASKS`, `TASK_GAMMA` / `train-sim-sac`, the lab trainers, `sim-gates`, `deploy-policy` and `train-reach-sac`.
- No die sim runs exist.
- No perception process writes the object table.
- Hardware steps not done: H8 hand-eye with fiducial, H9 physical tray, HS2/HS3 gripper and die sysid (`sim_params` v1).
- The real-arm feasibility gate: minimum graspable height, reach to the well and the zone, and scripted IK seeds at ≥ 30 %.
- No Tier 1 watchdog. Every real run so far used the no-watchdog variant.

## Decisions needed from the user before D2

1. **Training on the `sim_params` v0 prior.**
   - S4.6 forbids training the die task on v0, because gripper and die contact were never identified.
   - Either wait for HS1–HS3, which need H2/H3 hardware, or waive the rule for a first sim run whose policy is never deployed. The waiver is recorded in plan §10, and the run is repeated on v1 before HS6.
2. **The start-well occlusion parity gap** (`tests/test_die_convention.py` strict xfail).
   - From the home pose, the real tracker sees the die in the well as hidden; the sim occluders do not.
   - Options: fix the sim occluders before training (camera-parity plan), or accept the gap for the first sim run and record that in §10.
3. **Die start distribution.** Use the placeholder well until H9 has been measured (recorded), or build and measure the tray first.

## Units

| # | Unit | Kind | Depends on | Notes |
|---|---|---|---|---|
| D1 | Die task through every sim entry point: `eval_sim.TASKS`, `TASK_GAMMA`, `train-sim-sac --task die_to_zone`, the lab `fastsac_sim` task list, and `sim-gates` for the die (`do_nothing`, `start_not_success`, independent predicate) | SW | none | Claude can do this unattended. ARCHITECTURE row; tests with a fake/toy env as for reach. |
| D2 | S4.6 die pre-training to the exit gate: `--nworld 2048 --budget-min 480`, gate 0.95 / Wilson 0.93 at DR stage 4 over 1000 episodes | GPU run | D1; decisions 1–3 | Budget cap 30 GPU-h over ≤ 3 attempts. Coordinate the GPU with the rl-growth-probe session. Renders reviewed for predicate gaming; push vs place recorded. |
| D3 | Perception process: D435 → `HsvDepthDieTracker` → fiducial / camera-moved check → `ObjectTableShm` | SW + HW check | camera rig | Needs a fake backend for the tests (CLAUDE.md). |
| D4 | Hardware: H8 hand-eye ≤ 3 mm with the fiducial, H9 tray ≥ 19/20 returns, and the Phase 4 feasibility gate (minimum graspable height, reach to the well and the zone, scripted IK seeds ≥ 30 % with an FK residual check) | HW (user) | rig parts | The user builds the tray; Claude can drive the measured steps once D3 exists. |
| D5 | `deploy-policy` and `train-reach-sac` die support (tracker visibility, tray reset, halt when the tracker is UNKNOWN > 50 % of steps) | SW | D3 | Fake-backend tests. |
| D6 | HS6 zero-shot die evaluation (30 episodes, ladder) and the Phase 1 verifier's false-accept check ≤ 2 % | HW run | D2 on v1 params, D4, D5 | Attended until the Tier 1 watchdog exists. |
| D7 | HS7 real die fine-tuning (AWR route: frozen stochastic collection, then offline AWR; plan §10 2026-10-14/15) | HW run | D6 | Phase 4 milestone: > 70 % over N = 50. |

## Lessons carried over from reach (plan §10 2026-10-14 and 2026-10-15)

- Do not follow a critic's action gradient on the real arm without the offline sim replay gate. Prefer critic-free AWR on frozen-actor stochastic data.
- Judge every candidate by trips as well as success.
- No CPU-heavy jobs while the arm runs: SCHED_FIFO is unavailable on this host.
- Keep run scripts in the repo or the run directory, not the session scratchpad.
- Check the base policy's headroom before spending arm time on more collection.

## Attempt 2 design (after attempt 1 failed, `plans/reviews/2026-10-16-s4-die-run.md`)

Attempt 1 never produced a success: the die starts in a recessed well it cannot be pushed out of, and the critic
settled on ending episodes with a STUCK trip (genuine trips: a link pressed into the table with a constant
command). The user accepted the recommendation: C (STUCK check: done, trips genuine), then A + B.

| # | Unit | What | Acceptance |
|---|---|---|---|
| D2a | **Start curriculum (A)** | `DieStartSpec` gains staged start sets: S0 the open table between the tray and the zone (push distance 2-6 cm to the zone edge), S1 farther (6-12 cm), S2 just outside the tray rim, S3 the well (the target). The curriculum advances like the DR stages (eval success >= the stage threshold) and is recorded in `metrics.csv`; the exit gate stays at the well start (S3) and DR stage 4. Gates (`start_not_success`, `do_nothing`) run per start stage. | `sim-gates --task die_to_zone` passes for every start stage; a short CPU smoke run advances S0 -> S1 on a toy; ARCHITECTURE row; plan §10 row (the start set is part of the task spec) |
| D2b | **Scripted seeds (B)** | `rl/scripted.py`: a batched tick-space controller for the die (approach above the die, open, descend, close, lift over the rim, move above the zone, lower, open), using the existing FK/IK-free joint-delta action space (a damped least-squares IK step on `safety/fk.py` per policy step). Its success rate is measured in sim per start stage; RLPD-style mixing: a fixed share (e.g. 25 %) of each FastSAC batch from a demo replay filled with its episodes. | scripted success >= 30 % at the well start in sim (the plan's seed bar); mixing ratio a config key; tests on the toy env |
| D2c | Attempt 2 run | D2a + D2b, `sim_params` v0 under the same waiver | as S4.6 |

Order: D2a first (small, may already crack the push part), D2b next; D2c only with both.
