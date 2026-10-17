# so101_rl — reinforcement learning for a real SO-101 arm

Goal-conditioned reinforcement learning for one [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm (Feetech
STS3215 servos, Intel RealSense D435 camera). Policies are trained in GPU simulation (MuJoCo Warp, FastSAC), then
deployed zero-shot to the real arm and improved with real data where that helps. A safety layer that does not depend
on the policy guards every real-arm motion.

**Where it stands (October 2026)**

| Task | Simulation | Real arm |
|---|---|---|
| Reach, 3 mm success radius | exit gate 0.953 (Wilson LB 0.938) | **70/70 zero-shot**, error p50 2.6 mm |
| Reach, 5 mm | gate passed | 69/70 after one offline AWR round (50/50 at full speed) |
| Die-to-zone (pick the 16 mm die out of a tray, place it in a 60 mm zone) | **0.984** from the tray well (curriculum + scripted demos) | not yet: needs H8/H9 and calibrated sim parameters (see Next steps) |

The decisions behind every number are in the plan's decision log (`so101-self-learning-rl-plan.md` §10).

## How it fits together

```
 GPU sim (MuJoCo Warp, 1024-2048 worlds)          real rig
 ┌──────────────────────────────┐        ┌──────────────────────────────────────────┐
 │ train-sim-sac  (FastSAC, DR) │ export │ deploy-policy / train-reach-sac (10 Hz)  │
 │ eval-sim, sim-gates          │ ─────► │        │ joint deltas (shared memory)    │
 └──────────────────────────────┘        │ servo process (30 Hz safety loop) ──► bus│
                                         │        ▲ object table (shared memory)    │
                                         │ perception process (D435 → die tracker)  │
                                         └──────────────────────────────────────────┘
```

- **Servo process** (`src/so101_rl/servo/`, `safety/`): the only code that touches the serial bus. A 30 Hz loop with
  joint and workspace-box clamps, tracking limits, a stuck-action monitor, a thermal monitor (median-filtered), the
  firmware protection bits and a state machine. Policies only send joint deltas.
- **Perception process** (`src/so101_rl/perception/`): D435 colour + depth → HSV/depth die tracker (on-table blobs only,
  the arm's own pixels masked) → base-frame die position in a shared-memory object table, with a camera-moved check.
- **Simulation and training** (`src/so101_rl/sim/`, `env/`, `rl/`): batched MuJoCo Warp env with domain randomisation,
  FastSAC, sim gates, evaluation, policy export; offline AWR scripts for real-data fine-tuning.
- **Lab** (`src/so101_rl/lab/`): an MCP server and CLI so an AI client can run sim training jobs.

Full detail: `docs/ARCHITECTURE.md`.

## Repository layout

| Path | What |
|---|---|
| `src/so101_rl/` | the package: `servo/`, `safety/`, `perception/`, `sim/`, `env/`, `rl/`, `lab/`, `config/` |
| `assets/so101/` | MJCF scene (arm, table, die, tray, zone, fiducial) |
| `calib/` | committed calibration: `camera.yaml`, `sim_params.yaml`, `distractors.yaml` (+ `fiducial.yaml`, `hand_eye.yaml` after H8) |
| `scripts/` | generators and one-off tools (`point_tip.py`, `finetune_*.py`, `make_fiducial_print.py`, ...) |
| `tests/` | the test suite (no hardware or GPU needed for the default selection) |
| `docs/ARCHITECTURE.md` | architecture, interfaces, versions (changelog at the end) |
| `docs/HARDWARE_CHECKLIST.md` | the hardware steps (H-series) and real-arm runs (HS-series) with their results |
| `so101-self-learning-rl-plan.md` | the plan; §10 is the decision log and the source of truth |
| `plans/` | implementation plans and run reviews (`plans/reviews/`) |
| `runs/` | run outputs (not committed) |

## Setup

Everything runs through [pixi](https://pixi.sh). Environments:

- `default`: CUDA torch + MuJoCo Warp, for GPU training (`pixi run <task>`);
- `cpu`: the same with CPU torch, for tests and hardware-free runs (`pixi run -e cpu <task>`);
- `robot`: perception + CPU torch, for the servo host process (`pixi run -e robot <task>`).

```bash
pixi install            # default env
pixi run -e cpu test    # the suite without hardware or GPU
pixi run test-gpu       # GPU tests
pixi task list          # every task
```

Version pins and their reasons (lerobot 0.6.1 needs torch < 2.12 and numpy < 2.3, etc.) are in
`docs/ARCHITECTURE.md` §8.

## Common workflows

**Train and evaluate in simulation**

```bash
pixi run sim-gates --task reach                       # L1/L2 gates before any training
pixi run train-sim-sac --task reach --nworld 1024 --success-radius-mm 3 --budget-min 90
pixi run eval-sim --ckpt runs/sac-sim/<run>/ckpt_last.pt --task reach --episodes 1000 --dr-stage 4 --gate
pixi run export-policy --ckpt runs/sac-sim/<run>/ckpt_last.pt --out runs/sac-sim/<run>/policy.pt
```

Die-to-zone uses `--task die_to_zone` with a config yaml (curriculum, demo mixing; see
`runs/sac-sim/die-v0-attempt2.yaml` and `plans/die-to-zone-plan.md`).

**Run a policy on the real arm** (attended; start the client first, then the servo)

```bash
pixi run -e cpu deploy-policy --policy <export> --protocol zero-shot --seed 7 --attended --accept-no-watchdog
pixi run -e robot servo --backend feetech --no-mcu --tick-log runs/deploy/ticks-<ts>.bin
```

The speed ladder is 10 episodes at 0.5, 10 at 0.75, then 50 at full speed, with Wilson intervals and a fine-tune
verdict. `--accept-no-watchdog` is the documented exception until the watchdog MCU (H3) exists.

**Hardware-free checks**

```bash
pixi run -e cpu servo-fake --max-ticks 300            # servo process on the fake backend
pixi run -e cpu soak-fake --minutes 0.1               # fake-backend soak (servo + random client)
pixi run perception --backend sim --duration-s 5      # perception process on MuJoCo renders
```

**Real-rig tools**: `perception` (camera → object table), `scripts/point_tip.py` (hold the gripper tip over a
point, e.g. to mark the zone), `fiducial-capture`, `hand-eye --execute`, `camera-moved-check` (H8), `calib-check`,
`zero-adjust`, `camera-calibrate`. Procedures are in `docs/HARDWARE_CHECKLIST.md`.

## Safety rules for the real arm

- Only the servo process touches the bus; every policy goes through the 30 Hz safety loop.
- Real runs are attended (hand on the servo power switch) until the e-stop and watchdog hardware (H2/H3) exist.
- Do not run CPU-heavy jobs (the test suite, training threads, starting Python/torch processes) while the arm moves:
  the host has no real-time scheduling, and a starved servo loop stops the arm.
- Start perception and other clients before the servo / arm moves.
- A die policy trained on unverified sim parameters is refused on the real arm by `deploy-policy`.
- Fine-tune on real data with critic-free updates (offline AWR); a critic-gradient fine-tune collapsed twice
  (plan §10 2026-10-14).

## Next steps with the arm

1. **H8: hand-eye calibration and table fiducial.** Software is ready (`docs/HARDWARE_CHECKLIST.md` H8).
   - Print `runs/hw/fiducial/aruco_4x4_50_id0_50mm_A4.pdf` (`pixi run python scripts/make_fiducial_print.py`)
     at 100 %; glue it centred at base (0.09, 0.05), measured from the base by hand.
   - Put a 12–16 mm red sticker on the gripper tip; clear the table of red objects.
   - Run `fiducial-capture` (arm at rest), then `hand-eye --execute --attended` (the arm visits ~12 points; it
     writes `calib/hand_eye.yaml` only at residual ≤ 3 mm and compares with the arm-shape camera pose), then
     `camera-moved-check --procedure` (nudge the camera ~2 mm: the check must fail; put it back). Remove the
     sticker afterwards.
2. **Phase 4 feasibility gate on the real arm.** The scripted grasp–lift–place controller (49–50/50 in sim) runs on
   the real arm with the die placed by hand, to measure reach to the start area and the zone, the lowest graspable
   height and a scripted success rate ≥ 30 %. Needs a real-arm runner (not built yet).
3. **H9: the start-well tray** (≥ 19/20 returns), or `deploy-policy --manual-die-reset` for first tests.
4. **H2/H3: e-stop and watchdog MCU.** They unlock attended-free runs and the sysid runs HS1–HS3, which produce
   `calib/sim_params.yaml` v1.
5. **Retrain die-to-zone on v1 sim parameters**, then **HS6**: the first real die evaluation (ladder 10/10/30).

Progress on each step is recorded in `docs/HARDWARE_CHECKLIST.md` (results table) and the plan's decision log.
