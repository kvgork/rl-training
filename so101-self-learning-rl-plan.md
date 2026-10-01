# SO-101 Self-Learning Robot Arm — Project Plan

> **Version 2 (2026-09-30).** This version adds evidence from the Second Brain vault and existing code to the v1 plan (`so101-self-learning-rl-plan.v1.md`).
>
> Markers:
> - **[v2]** marks new or changed content.
> - **[v2-conflict]** marks a place where vault evidence contradicts v1. Each conflict is an open decision in §8 until it is resolved.
>
> The sources are listed in §11. Vault paths are relative to `~/Documents/Vaults/Local`, and `LIT` means `/mnt/data/workspaces/lerobot-isaac-training`.

## 1. Goal

Build a robot system that learns to manipulate objects mainly through reinforcement learning. The system should:

- perceive and label its workspace with a camera,
- accept high-level commands such as "move cup",
- reason from the goal down to actuator signals:
  *move cup → move gripper to cup → move arm → joint commands*,
- learn its own body by exploring (moving joints and observing the effect),
- reuse learned skills across different tasks.

RL is the main learning engine. Other techniques (simulation, demonstrations, foundation models) are only used to make learning faster or more efficient.

**[v2] Purpose, in three stages.** The project moves through these stages in order, and each has its own bar:

| Stage | Purpose | Covers | Bar |
|---|---|---|---|
| A. Research | Understand self-learning RL on the real arm | Phases 0–3 | Insight and measured results. A negative result that is well documented counts. |
| B. Demo | Make "move cup to X" work end to end | Phases 4–6 v1 | It works reliably with little human time. |
| C. Platform | Build a reusable skill and world-model platform | Phase 6 v2 and later | New tasks need no new motor learning, and new skills do not forget old ones. |

**[v2] Hard constraints** (confirmed by the user on 2026-09-30):

1. **RL stays the core learner.** Imitation learning, VLAs and scripted controllers may accelerate learning, but they are never the final learner.
2. **Minimize human time.** Unattended training is a priority, so avoid manual resets and demos.
3. **Use one consumer GPU.** All training fits the local RTX 3080 with 10 GB, with no cloud.
4. **Reuse existing code opportunistically.** Use it where it fits, but don't spend effort making it fit.
5. **[v2] Ship it as a pixi package.** The project is its own pixi workspace and package at `/mnt/data/workspaces/rl-trainer`, with a locked `pixi.lock` and named tasks for every entry point (for example `pixi run babble` and `pixi run train-reach`). It does not depend on the `LIT` environments.

## 2. Hardware and Compute

| Component | Details |
|---|---|
| Robot | SO-101 follower arm with Feetech STS3215 servos (5 DOF + gripper). **[v2]** It is on `/dev/ttyACM0` at 12 V and calibrated. The calibration file is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`. |
| Camera | Intel RealSense D435 (RGB + depth), fixed, pointed at the work area. **[v2]** Serial 819312071240. The colour stream is `/dev/video4`, but indices change on every USB replug. It needs about 45 warmup frames. At 0.614 m, 1 px ≈ 0.99 mm. It has only ever been used as an RGB overhead view at 640×480 and 30 Hz, never for depth in a policy. |
| Training compute | **[v2]** One RTX 3080 with 10 GB. **The driver is currently not loaded.** `nvidia-smi` fails and `/dev/nvidia*` is missing, so it needs a reboot or a DKMS fix, which is a Phase 0 blocker. One vault note calls this machine a laptop, so check its thermals and power. |
| ~~Optional: leader arm~~ | **[v2-conflict]** The leader arm is **permanently gone**, so the plan cannot rely on teleop demos. Demos must be scripted (IK), generated in sim, or taken from existing datasets (`LIT/datasets/so101-pickplace-new`, 50 episodes). |
| **[v2]** Objects on the rig | A white cup with green leaves, a red 16 mm die and a white arm. |
| **[v2]** Missing | A **hardware e-stop**. Today only cutting power stops the arm, and Ctrl-C cannot interrupt a blocked serial write (`LIT/SESSION_STATE.md`). |

## 2A. [v2] Lessons From Prior Work (Second Brain)

All earlier RL on the SO-101 ran in **simulation** (Isaac Lab, DreamerV3 via sheeprl). No note records RL on the real arm. The only success on the real arm was ACT behaviour cloning on 48 teleop demos, which the missing leader arm now rules out.

| # | Lesson | Evidence | Impact on this plan |
|---|---|---|---|
| L1 | **Check that a task is physically solvable before tuning rewards.** About six multi-hour runs chased tasks that could not be done: an object at 0.51 m against a 0.346 m maximum reach, and a 4 mm cube instead of the 16 mm die. | `04-Archive/2026-06-09-staged-reward-tuning-completed.md`, `2026-06-10-sim-rl-manipulation-lessons.md` | Add a feasibility gate before every skill phase: targets within about 0.30 m of the base, objects under about 3 cm. |
| L2 | **Success predicates get gamed.** One predicate fired on reach instead of place, the agent slid the die into the bin, and a perception predicate had false accepts that passed as a green test suite. | `04-Archive/2026-06-28-so101-carryplace-rl-campaign.md`, `2026-09-10-grill-review-so101-safety-fixes.md` | Validate each reward and success signal against labelled ground truth *before* training, including a check for false accepts. |
| L3 | **Pure RL (DreamerV3) never discovered placing, and pure BC (ACT) scored 0/20 in sim.** A scripted IK controller reached 67–80%, so the direction changed to **residual RL on a scripted base**. | `04-Archive/2026-06-28-so101-carryplace-rl-campaign.md` | Consider residual RL (a scripted IK base plus an RL correction) as the path for grasp and transport. It is still RL. |
| L4 | **Action clamps are a hazard.** Five days were lost to a [-1,1] clamp that only worked when an env var was exported. The replay buffer must store the *post-clamp* action. | `04-Archive/2026-07-24-residual-rl-clamp-saga-lessons.md`, `05-Wiki/log.md` | The safety layer fails loudly, has one source of truth for its config, and logs the action that was actually applied. Diff launch environments. |
| L5 | **Resets are the bottleneck for autonomy.** A 1% reset-failure rate forces heavy supervision, and extra human demos beat autonomous data (Mirchandani, CoRL 2024). No published system runs multi-hour unattended RL on a real arm. | `04-Archive/2026-09-10-claude-driven-so101-recording-research.md`, `05-Wiki/concepts/Autonomous-Robot-Data-Collection.md` | This conflicts with "minimal human time". Design resets in from Phase 0 with a bounded tray and reset-free skills first, instead of waiting until Phase 5. |
| L6 | **Pure sim pretraining did not transfer** (community HIL-SERL SO-101 result). Use sim for augmentation or warm starts only. The vault's SO-101 sim was never validated against the real arm. | `05-Wiki/entities/SO-101.md`, `05-Wiki/entities/HIL-SERL.md` | Keep the plan's "sim light" stance, and add a sim-vs-real validation step. |
| L7 | **Real-arm IK in LeRobot 0.6.0 (placo) never reports failure.** Its default orientation weight gives 54.8 mm of error, and one call runs only one solve step. | `05-Wiki/concepts/SO-101-Inverse-Kinematics-(Placo-LeRobot).md` | End-effector actions or a scripted base need an FK residual check on every command. |
| L8 | **Naive motor babbling does not scale; goal babbling replaced it.** The world model needs coverage: 200–500 or more exploratory episodes. | `05-Wiki/research/2026-08-08-onyx-per-joint-composition.md`, `LIT/plans/2026-06-23-world-model-training-plan.md` | Phase 2 uses goal babbling (random targets in the workspace), with random joint motion only as a seed. |
| L9 | **Learned options rarely beat flat policies on real robots, and Option-Critic collapses.** Hierarchy *inside* a world model has support (Director, HWM). | `05-Wiki/research/2026-06-21-hierarchical-planning-world-models-robotics.md` | Phase 6 v2 (RL over skills) is the weakest part. It moves to Stage C as research. |
| L10 | **The LLM planner works as "SayCan-lite".** It means an API model emitting discrete skill IDs at 1 Hz or slower, with precondition gates and replanning. **The skill layer is what limits it** (0% → 13% → 67% from better skills alone). Local models of 4B or less collapse on multi-turn plans. | `05-Wiki/research/2026-08-02-llm-robot-control.md` | Use an API LLM for v1. Spend the effort on skill quality, not the planner. |
| L11 | **With one D435, the die-in-cup SUCCESS state is invisible** because the cup rim occludes it. Auto-labelling can emit FAILURE or UNKNOWN but never SUCCESS. There is no room for the planned overhead camera. | `LIT/plans/2026-09-12-overhead-verification-camera-plan.md` | Choose first tasks whose success the camera can observe, such as a block on a pad instead of a die in a cup. Use depth, and gripper load as a second signal. |
| L12 | **Pipeline hazards.** Autoresearch loops mask silent failures, so check output files and not exit codes. Detach long jobs with `setsid`. The LeRobot 0.6.0 upgrade broke frame caching. | `04-Archive/2026-07-19-gpu-campaign-3day-lessons.md` | The logging and evaluation harness checks its artifacts, not return codes. |
| L13 | **DreamerV3 on the 3080** ran at 0.5–1.2 steps/s in Isaac. Offline training on real data ran at 0.27 steps/s (about 29 h for 30k steps). Reconstruction loss is a poor control metric. DayDreamer learned pick-and-place on a real arm from pixels in 8–10 h. | `05-Wiki/entities/TD-MPC2.md`, `04-Archive/2026-05-14-world-models-swap-matrix.md` | Gate world-model work on task success, not on loss. Budget the GPU time in advance. |

## 3. Architecture: Hierarchical RL

Three layers. Each layer reasons about its own level of abstraction, and the lower layers are shared by all tasks.

```
 Command: "move cup"
        │
┌───────▼─────────────────────────────────────┐
│ TASK LEVEL (why)                             │
│ Goal + scene graph → sequence of skills      │
│ reach(cup) → grasp(cup) → transport(xyz)     │
│ → release()                                  │
│ v1: LLM planner (API)  v2: learned (Stage C) │
└───────┬─────────────────────────────────────┘
        │ skill + goal parameters
┌───────▼─────────────────────────────────────┐
│ SKILL LEVEL (what)                           │
│ Goal-conditioned RL policies:                │
│ reach · grasp · transport · release · push   │
│ [v2] pure RL (TD-MPC2 + HER); SAC baseline   │
│ Task-independent → transferable              │
└───────┬─────────────────────────────────────┘
        │ joint actions
┌───────▼─────────────────────────────────────┐
│ SAFETY LAYER [v2: explicit, policy-independent]│
│ clamp → limits → thermal → e-stop; logs the  │
│ applied (post-clamp) action                  │
└───────┬─────────────────────────────────────┘
┌───────▼─────────────────────────────────────┐
│ MOTOR LEVEL (how)                            │
│ World model learned from goal babbling:      │
│ state + action → next state                  │
│ Shared by all skills                         │
└───────┬─────────────────────────────────────┘
        │
   Servos (SO-101)
```

Perception runs alongside all layers and supplies the object table, gripper state and reward signals. **[v2]** Every reward or success signal that perception produces has a measured false-accept rate before it is used for training (L2).

### Why this transfers between tasks

- **Skills are goal-conditioned.** `reach(xyz)` works for any target, not just "the cup".
- **The world model is task-independent.** Every new skill starts with a robot that already knows how its body moves.
- **The task level only composes skills.** New tasks such as "stack block" or "clear table" need new plans, not new motor learning.
- **[v2] Skills are isolated modules.** In Stage C, per-skill adapters (for example LoRA, about 8 MB each, under 2% backward forgetting with on-policy RL) keep new skills from overwriting old ones (`2026-05-16-extending-neural-nets` research, arXiv 2603.11653).

## 4. Techniques That Speed Up Learning

| Technique | What it does | Where it is used |
|---|---|---|
| Hindsight Experience Replay (HER) | Relabels failed attempts as successes toward the position actually reached | All goal-conditioned skills. **[v2]** It helps reach most. Grasp and lift needed a shaping term such as grip × ee_height. HER is not built into TD-MPC2, so it must be implemented. |
| Model-based RL (TD-MPC2, with DreamerV3 as an alternative) | Trains largely inside the learned world model, so fewer real movements are needed | Skill level. **[v2]** TD-MPC2 has never been run by the user and is not installed anywhere. DreamerV3 has run in the user's setup, but it failed at placing (L3, L13). |
| MuJoCo pretraining | Short warm start with domain randomization; most training happens on the real arm | Reach, and possibly grasp. **[v2]** Use the official SO-101 MJCF (TheRobotStudio/SO-ARM100), then system-identify it. Randomize in this priority order: camera pose, then dynamics, then textures. |
| ~~Teleop demos (RLPD-style)~~ **[v2] Scripted or prior demos (RLPD-style)** | Seeds the replay buffer with 10–20 demos | Grasp. **[v2-conflict]** With no leader arm, the demos come from a scripted IK controller, from sim, or from `so101-pickplace-new`. |
| **[v2] Residual RL (fallback only)** | RL learns a correction on top of a scripted IK controller | **Not the main path**: the user decided on pure RL everywhere (see §10). It is kept as a documented fallback in case pure RL stalls. A scripted controller reached 67–80% in sim (L3). |
| **[v2] SAC + HER baseline** | A cheap model-free baseline trained on the same reach task | Phase 3. It shows whether the TD-MPC2 world model actually helps. |
| **[v2] Goal babbling** | Reaches random targets instead of random joint noise | World-model data (L8) |
| Foundation models | SAM2 and Grounding DINO for perception, an LLM for the initial task planner | Perception and task level. **[v2]** There are no VRAM or latency numbers for SAM2 or Grounding DINO on the 3080, and they compete with training for the GPU. Run perception at a low rate (object table at 5 Hz or faster, from cached masks plus tracking). |
| Self-resets | The robot resets the scene itself once basic skills work | **[v2]** Designed in from Phase 0 (a bounded tray and reset-free skills first). Full scene reset stays in Phase 5 (L5). |

## 5. Roadmap

**[v2]** Each phase has a feasibility gate (L1) before any training, and a milestone that is checked by a script rather than by eye.

**[v2] Budgets and review triggers.** A trigger opens a manual review and never switches the method automatically, which matches the §10 decision to judge the fallback per case.

| Phase | Real-arm budget | GPU budget | Review trigger |
|---|---|---|---|
| 2 | 10 h of babbling | 48 h | The probe error has not improved over 3 checkpoints |
| 3 | 8 h of RL per algorithm (about 190k steps at 10 Hz under the duty cycle) | 48 h | Success below 10% after 40k steps, or no drop in mean final distance over 3 checkpoints. Joint-space reach is expected to need 30k–100k steps. Use episodes of 50 steps or fewer and gamma of about 0.95. |
| 4 | 15 h of RL | 72 h | Zero verified successes after 30k real steps |

**[v2] Latency budget at 10 Hz (100 ms per step), to be measured in Phase 0:**

**Process layout.** The **30 Hz servo and safety loop is its own OS process**, and it owns both the serial port and the heartbeat. It talks to the 10 Hz policy process through shared memory. The separation is needed because Python's GIL would otherwise let MPPI starve the safety tick.

The servo process runs with:
- SCHED_FIFO,
- `gc.disable()` or `gc.freeze()` with no allocation per tick,
- absolute-deadline scheduling (`next += period`),
- a log entry for every tick overrun.

If the policy is late, the servo loop holds the last target and never extrapolates. After N consecutive misses it trips to the safe state.

| Loop | Item | Target |
|---|---|---|
| 30 Hz servo | One raw `sync_read` (address 56, length 8: position, speed, load, voltage, temperature) | 3–6 ms |
| 30 Hz servo | Clamp and FK on the cached read (no second LeRobot read) | < 1 ms |
| 30 Hz servo | `sync_write`, heartbeat, async log enqueue | ~1 ms |
| 10 Hz policy | Camera frame + encoder | ≤ 20 ms |
| 10 Hz policy | MPPI plan | ≤ 50 ms |

**Checks and modelling:**
- Set `Return_Delay_Time = 0`.
- **Soak gate:** p99.9 tick overrun below 5 ms, and zero heartbeat gaps longer than 200 ms, under full camera, MPPI and logging load.
- Measure the end-to-end latency from observation to command (expected 100–150 ms) and model it as an action delay in the MuJoCo env.

Perception runs asynchronously. Online TD-MPC2 updates run between episodes or in a separate process with a VRAM cap.

**[v2] VRAM is not assumed to fit.** "Fits in 10 GB" is a hypothesis until Phase 2 measures pixel TD-MPC2, the image replay buffer and the perception models running together. Perception and training use per-process memory caps (`torch.cuda.set_per_process_memory_fraction`), and an OOM must fail loudly rather than return rc=0.

### [v2] Stage A critical path (lean minimum)

Only these items block the first real-arm RL result:

1. The safety layer with the deadman relay.
2. A colour + depth die tracker.
3. A state-based world model.
4. SAC + HER on reach.
5. TD-MPC2 + HER on reach.

These are developed **in parallel, off the critical path**, and attach when their own gates pass:

- **Pixel world model.** "Pixels from the start" is kept, but it does not block Phase 3.
- **SAM2 and Grounding DINO**, in Stage B.
- **Full MuJoCo sysid.** In Stage A, sim covers kinematics, safety-box validation and the HER unit tests.
- **Stage B/C extras:** native extensions, the ROS2 bridge and `pip install` support.

### Phase 0: Foundation — Stage A

- **[v2] Preconditions (blockers):**
  - Restore the NVIDIA driver, so that `nvidia-smi` reports an RTX 3080 with 10 GB. **[v2] Fallback:** if DKMS fails on kernel 7.0.0-34, boot the previous kernel or use another driver branch. Also resolve the laptop-vs-desktop question and check GPU thermals under multi-hour load.
  - Fit a **hardware e-stop** (a power-cut button in the 12 V line) before any unattended motion.
  - Adopt the **tiered autonomy model** (§10). This replaces the current "consent per session, hand on the power switch" rule:
    - **Tier 0 (attended):** the default for all of Stage A until the soak test passes.
    - **Tier 1 (unattended):** allowed only when all of these hold:
      - **Power path (one topology):**
        - **Relay and heartbeat:**
          - An **off-host watchdog MCU** (RP2040 or Arduino class, about 50 lines) drives a **12 V automotive relay** with normally-open 30/40 A DC contacts. The coil is switched by a transistor with a flyback diode.
          - The relay is energised only while heartbeat pulses arrive, and opens after a 1 s monotonic timeout. An MCU reset or a lost USB link also opens it.
          - The servo loop sends a heartbeat **every tick (≤ 100 ms)**, never from a timer thread, and only after that tick's `sync_read` + `sync_write` succeeded. A live process with a dead bus therefore still drops power.
        - **Parts:**
          - A bidirectional TVS diode (15–16 V) sits across the servo-side bus.
          - The **manual e-stop is rated DC-13 at or above the supply's continuous current**. The alternative is to put it in the relay's low-current control circuit while keeping a mechanical break in the load path.
          - A plain USB or GPIO relay is *not* a deadman, and AC-only SSRs latch on DC, so neither is used.
        - **Trip latching:** deadman and e-stop trips latch in the MCU and need a physical reset.
        - **Supply:** the PSU is at least 8–10 A at 12 V, so that one stalled servo cannot brown out the bus.
      - **Safe-state sequence for non-emergency trips** (thermal, stuck action, NaN, out-of-distribution, lost camera):
        - **Sequence:** ramp home at low speed, set the arm down in the rest pose, then turn torque off. Cutting power is reserved for emergencies and deadman loss.
        - **Bounds:** the sequence is **bounded to 5 s**. After that it falls through to torque-off in place, and finally to the deadman.
        - **Rest pose:** it rests on the base or the table, so turning torque off causes no motion. Verify this by turning torque off from that pose.
        - **Process exits:** SIGINT, exceptions and disconnects all go through this sequence. Set `disable_torque_on_disconnect` explicitly, because LeRobot's default drops the arm wherever it is.
      - **Sag map:** limp falls are sampled *on hardware*, from the safety-box corners and the highest poses, with foam underneath. Each is recorded with the camera. MuJoCo does not model the gearbox's back-drive friction. None of the falls may touch the D435 mount.
      - **Soak tests:** the 30-minute soak passes, and later a 2-hour goal-directed soak at the real duty cycle passes before a multi-hour run.
    - A human confirms every promotion from one tier to the next.
- **[v2]** Scaffold the pixi package: a `pyproject.toml` with a `[tool.pixi]` section and one package at `src/so101_rl`. `pip install` support comes later (Stage B/C). lerobot comes from PyPI. **TD-MPC2 is not on PyPI** (`pypi.org/pypi/tdmpc2` returns 404), and LeRobot 0.6.0 ships TD-MPC v1 (`policies/tdmpc`), not TD-MPC2, so **TD-MPC2 is vendored**: `nicklashansen/tdmpc2` has no setup.py or pyproject, so it cannot be a git dependency. It is MIT-licensed. Copy `tdmpc2/` into `src/so101_rl/_vendor/tdmpc2/`, keep its LICENSE, and declare `torchrl` and `tensordict` as direct dependencies.
  - **Dependency channels:**
    - **conda-forge:** `pytorch-gpu`, `torchvision`, `torchcodec` and `cuda-version`, pinned to the 3080 driver. Never mix in the legacy `pytorch` channel.
    - **PyPI:** `lerobot[feetech,grpcio-dep]`, without the `intelrealsense` extra because it pins `pyrealsense2<2.57`. `pyrealsense2` is a direct PyPI dependency.
    - The `LIT` lock is not evidence of compatibility, because torch and lerobot were pip-installed there by hand.
  - **Solver spike first**, before the layout is committed:
    - Target python 3.12, `pytorch-gpu` 2.7.1 (or 2.8–2.11 with the matching `torchrl` and `tensordict`), and numpy 2.2.x. TD-MPC2 itself pins python 3.11, numpy 1.24 and torch 2.7.1, so python 3.12 and numpy 2 are untested there.
    - Run a TD-MPC2 smoke train (state observations, 1k steps) under numpy 2.
  - **Pin lerobot** to 0.6.x. 0.6.1 is the latest release. Use CUDA pinned to the 3080 driver, features for `sim`, `perception` and `train`, and tasks for each entry point. `pixi install` and `pixi run test` must pass on a clean checkout.
- Use LeRobot drivers for the SO-101 (lerobot 0.6.0 is already in the `LIT` pixi envs). A ROS2 bridge is an option.
- Create **one Gym-style environment interface** shared by sim and real.
- Add a safety layer: joint limits, a Cartesian safety box, velocity and torque limits, and an emergency stop.
  - **[v2] Action units.** The vendored modules use three conventions ([-1,1], normalized -100..100 and `action_scale`), so this project uses **one convention** in sim and real: the delta is applied to the *last target*, and a tracking-error clamp keeps |target − present| ≤ E. The 30 Hz loop enforces the clamp, so a blocked arm cannot wind up.
    - **Test 1:** intercept `sync_write` and assert the same Goal_Position delta in ticks (0.088° per tick) for a unit action in both backends.
    - **Test 2:** block the arm (an obstacle in sim, a stop on hardware) and assert that |target − present| never exceeds E.
    - **Test 3:** an out-of-range command is rejected before serialisation, and 0 ≤ goal ≤ 4095 always holds.
  - **[v2] Reuse:** `LIT/src/robot-data-runner/src/robot_data_runner/safety_limits.py` (clamp_action and NaN handling), `safety.py` (same folder) (safe_connect, stuck-action watchdog, SIGINT handler) and `lerobot_isaac_deploy/arm_motor_writer.py` (joint floor and ramped_home). All three already run on this arm.
  - **[v2] Known issues:**
    - LeRobot #1342 (a negative Homing_Offset written as unsigned) is handled by sign-magnitude encoding in 0.6.0. The connect-time read-back check stays, to guard against calibration drift and partial writes.
    - Pin `--robot.id=so101_follower`, and ignore the legacy `robots/so101_follower/` directory and the `None.json` files.
    - `use_degrees=True` is the 0.6.0 default. Set it explicitly and leave `max_relative_target=None`, because when it is set, `send_action` does a second full-bus read on every step. The clamp lives in this project's safety layer instead.
    - LeRobot's `RobotEnv.step` sends *absolute* joint targets, so joint deltas need a delta-to-absolute step in the servo process.
    - `Over_Current_Protection_Time` is never written.
- Monitor servo temperature and load, and cool down automatically, because the STS3215s overheat under continuous RL.
  - **[v2] Cooldown scheduler (new code, because none exists in `safety.py` or `safety_limits.py`):**
    - A fixed duty schedule measured in Phase 0, with a starting point of 20 minutes on and 10 off. "Off" means torque off in the rest pose, never holding.
    - The **software trip is at about 60 °C**, below the firmware's 70 °C cutoff so the firmware never wins the race and drops the arm. Resume is at 45–50 °C or below, with hysteresis, checked every servo tick.
    - Any firmware Protection status bit counts as a trip.
    - Write `Protection_Current`, `Overload_Torque` and `Over_Current_Protection_Time` in `configure()`. Prove the fold-back by stalling one joint against a stop.
    - A small fan on the servos.
  - **[v2] Post-trip behaviour:**
    - The interrupted episode is marked `truncated` in the buffer and never counts as a failure.
    - **Resume policy by cause:**
      - *Thermal:* auto-resumes after cooldown. After 3 trips in one hour it stops and waits for a human.
      - *NaN, stuck action or OOD:* resumes only from a fresh process and a checkpoint, or waits for a human.
      - *Deadman or e-stop:* only a human resumes, with a physical reset.
    - **On every re-energise:**
      1. Wait until all 6 IDs answer.
      2. Run the calibration register check.
      3. Write `Goal_Position = Present_Position` for every joint *before* `Torque_Enable`. This prevents a snap from a stale goal.
      4. Run `ramped_home`.
      5. Test the stale-goal case on purpose, with foam underneath.
  - **[v2] Calibration check on every connect:** read the motor registers back and compare them with the calibration file, which guards against LeRobot #1342 uint16 corruption. Refuse to move on a mismatch.
  - **[v2]** A joint-current and thermal watchdog exists (0.50 s trip), but there is no duty-cycle data, and holding still is worse for heat than moving. Measure temperature against duty cycle in Phase 0.
- Calibrate the D435 (intrinsics, plus extrinsics to the robot base).
  - **[v2] Hand-eye is load-bearing:** goals come from the camera frame and the gripper position from FK in the base frame. Produce and commit a calibration artifact (`calib/hand_eye.yaml`) with a residual below 3 mm on 10 or more poses.
  - **[v2] Camera-moved gate:** a fiducial fixed to the table in view is checked every episode. If its reprojection error is above 2 px, training halts.
  - **[v2] Reuse:** `LIT/scripts/calibrate_hand_eye.py` and `_object_pose_depth.py` (±3 mm over 3 runs). Whether hand-eye calibration was ever finished is not recorded.
- Build a data logging pipeline for all episodes (states, actions, images, rewards). **[v2]** It logs the *applied* action (L4) and checks its output files (L12). **[v2]** Budget disk space: images at 30 Hz and 640×480 run to tens of GB per day. Store downscaled frames for world-model training, keep full frames only for evaluation and failure episodes, and delete anything older than N days.
- Set up the MuJoCo model of the SO-101, matched to the real joint limits. **[v2]** Adopt the **official MJCF** from `TheRobotStudio/SO-ARM100/Simulation/SO101/` (`so101_new_calib.xml`, `scene.xml`) instead of building one from the URDF. Then system-identify it (see `05-Wiki/concepts/Sim-Calibration-Loop.md`) and validate it against logged real trajectories (L6). MuJoCo's contact fidelity for this arm is unproven, because GAUGE did not evaluate standalone MuJoCo.

**Milestone:** the same script runs a random policy in both sim and real, safely, with logging. **[v2]** Add these checks:
- A 30-minute unattended random-motion soak run with no thermal trip. It logs the maximum temperature per servo, with zero trips and zero calibration mismatches.
- MuJoCo sysid: the RMS joint error against 10 or more logged real trajectories is 5% or less.
- **Power-path e-stop:** the bus voltage falls below 1 V within 50 ms of the button press. This is measured with the watchdog MCU's ADC or a scope, because the encoder log simply ends when power is cut.
- **Software safe-stop:** after a thermal, stuck or NaN trip, Goal_Position is frozen and joint velocity falls below the threshold within 100 ms, measured from the encoder log.
- **Deadman:** `kill -9` on the servo process, unplugging the bus adapter's USB, and rebooting the host each drop power within 1.5 s.
- The e-stop is a latching contactor in the 12 V feed. A de-energised arm falls, so run the test with a hand or foam block under the arm.
- **Every gate is shown to fail on a deliberately broken input** (the style of `05-Wiki/concepts/Arm-Bring-Up-Ladder.md`), so a green check cannot pass with nothing behind it.
- `nvidia-smi` healthy.

### Phase 1: Perception — Stage A

- **[v2] Stage A:** an HSV colour threshold plus depth tracks the red die at 30 Hz with almost no VRAM, and feeds the object table and verifier. The SAM2 and Grounding DINO steps below move to **Stage B**, when open-vocabulary objects such as the cup are needed.
- Take a snapshot and segment it with SAM2.
- Label the segments, either manually (click and name) or automatically with Grounding DINO.
- Build an **object table**: label, mask, 3D centroid (from depth), and approximate size.
- Track the gripper with an ArUco marker at first. Replace it later with learned or segmented tracking, or **[v2]** FK plus hand-eye calibration, which already exists (`_arm_kinematics.py`).
- Provide a scene graph output for the task level, for example: `cup at (x,y,z), block at (...), gripper at (...)`.
- **[v2]** Build a **binary success verifier** per skill on top of the object table (reuse `binary_success_verifier` from the harness). Measure its false-accept and false-reject rates on 50 or more labelled snapshots (L2, L11). **[v2]** The test set must include *in-motion, arm-in-frame* negatives: the gripper hovering over the zone, the gripper covering the die, shadows, and red-fingertip look-alikes. The gripper hides the die while grasping, so the verifier returns UNKNOWN, never SUCCESS, when the die is occluded. Validate gripper load (STS3215 `Present_Load`) against labelled grasps before it is used as a signal.

**Milestone:** reliable object table updated at more than 5 Hz, with 3D positions accurate to about 1 cm. **[v2]** Ground truth is an ArUco-tagged die on a printed grid at 20 or more poses, with a 95th-percentile error of 1 cm or less. The verifier's false-accept rate is 2% or less. **[v2]** Also measure the verifier's false-accept rate and the GPU memory perception uses while training runs.

### Phase 2: Motor Babbling and World Model — Stage A

- Explore safely. **[v2]** Use goal babbling (random Cartesian targets inside the safety box) seeded with single-joint random motion (L8).
- Collect 1–2 hours of real data, plus extra sim data. **[v2]** That is 200–500 or more episodes, and it doubles as the thermal duty-cycle study.
- Train a dynamics model: ~~`(joint state, action) → (next joint state, gripper position)`, later extended with a latent visual state~~ **[v2] from pixels from the start** (§10). The inputs are D435 RGB (and optionally depth) at a low resolution such as 64×64 or 84×84, plus proprioception, encoded to a latent. The model is `(latent, action) → next latent`. Decoded or probed gripper and die positions are still validated against FK and the object table.
  - **[v2] Pixel-WM safeguards:**
    - Measure VRAM with the encoder running at the same time as perception.
    - Keep a state-based world model as a debugging baseline, to separate encoder problems from dynamics problems.
    - Rewards still come from the object table and verifier, not from pixels.
    - **TD-MPC2 pixel mode needs code changes:**
      - The encoder uses *one* modality (`state` or `rgb`), and its conv stack is hard-wired to 64×64 input and to DMControl.
      - Pixels plus proprioception therefore needs a fused encoder: an RGB conv branch and a state MLP, concatenated, then SimNorm. The conv stack must be parametrized for 96×96.
    - **RAM budget as well as VRAM:** a pixel replay buffer of 1M steps is about 37 GB, which spills to system RAM. This host has 31 GB total and 18 GB free. Keep `buffer_size` at about 300k or less and store uint8 frames.
    - **Resolution check:** at 64×64 one pixel covers about 10 mm, so the 1–2 cm probe milestone sits at the floor. Use 96×96 or larger, or a crop around the workspace, and measure.
    - **Cross-session gate:** before a pixel world model may drive MPPI on the real arm, it must pass the probe milestone on held-out data from a *different day and lighting*.
    - **Data budget:** DayDreamer needed 8–10 h from pixels. 1–2 h is the first checkpoint, not the expected end.
    - Camera pose is fixed and randomized first in sim (L6).
    - Run an **out-of-distribution monitor** on encoder features that halts on a novel scene or state (`05-Wiki/concepts/Safe-RL-for-Robot-Manipulation.md`).
- Validate by comparing predicted and actual trajectories on held-out data.
- Optionally use active exploration (curiosity: go where prediction error is high; Plan2Explore).

**Milestone:** the world model predicts gripper position over a 1-second horizon within about 1–2 cm. **[v2]** This is measured through a probe on the pixel latent, checked against FK.

### Phase 3: Reach Skill — Stage A

- Goal-conditioned policy `reach(xyz)`. **[v2] Order:** SAC + HER runs first on the real arm, for a fast first result. TD-MPC2 + HER, the main algorithm, runs second, starting from the state-based world model, and is compared against it.
- **[v2]** Train a SAC + HER baseline on the same task, with the same budget of real steps. Compare the steps each one needs to reach 90% success.
- Reward: negative distance from gripper to goal, with a success bonus.
- Warm-start briefly in MuJoCo, then train on the real arm.
- **[v2] Implementation facts** (checked in the LeRobot 0.6.0 source and the TD-MPC2 repo):
  - **SAC:**
    - SAC lives in `lerobot/rl/algorithms/sac/`, not `policies/sac`, and is tied to the HIL-SERL actor/learner gRPC split.
    - LeRobot 0.6.0 has no HER or goal-relabelling hook.
    - Baseline plan: SB3 `HerReplayBuffer` (TQC or SAC) on a goal env with dictionary observations. LeRobot is used only as the driver.
  - **TD-MPC2:**
    - For the state path, append goal xyz to the `state` observation, which makes goal-conditioning free.
    - HER is a relabel in `Buffer._prepare_batch` that recomputes the reward from the FK gripper position against the relabelled goal.
    - Budget about 1–2 weeks, including the sim tests.
- **[v2] HER integration test first:** TD-MPC2 is multitask, not goal-conditioned, so the reward head, Q ensemble and policy prior must all be goal-conditioned and relabelled consistently. Both TD-MPC2 + HER and SAC + HER must reach 90% or more on MuJoCo reach before any real-arm episode. Frame conversions (camera to base) get their own unit tests.
- **[v2] Goal distribution:** radius 0.30 m or less, inside the safety box, inside the camera field of view, and above the measured kinematic floor.
- **[v2]** Actions are **clamped per-step joint deltas** (§10) for every skill. Reach needs no scene reset, so it is the first unattended real-arm RL experiment and the test for constraint 2.

**Milestone:** reaches arbitrary points in the workspace within about 1–2 cm, at a success rate above 90%. **[v2]** The eval uses N=100 goals from the goal distribution above, and the lower 95% bound must exceed 85% (never gate on n=1). An *intervention* is any human touch, restart or reset, logged per hour.

### Phase 4: Grasp Skill — Stage B

- Goal-conditioned on the object: `grasp(object_id)`.
- **[v2] First target task: move the red 16 mm die into a marked zone on the table.** Both push and pick-place count as success. The zone is visible from above, which avoids the cup-rim occlusion (L11). The binary verifier checks the die's centroid against the zone polygon.
- Reward: ~~the object mask rises in depth, and gripper load increases~~ **[v2] task-aligned:** the negative distance from the die centroid to the zone centre, plus the binary zone verifier as a success bonus. Push and place solutions both earn it. The lift shaping (grip × ee_height) belongs only to a separate `grasp` skill.
- **[v2] Reset mechanism:** a **gravity-return tray** (tilted, with a start well) returns the die to a start distribution every episode, and is tested in Phase 0. Without it the die ends up in a corner and the state distribution collapses.
- **[v2] Feasibility gate:** measure the real arm's minimum graspable height and its reach to the start well and the zone before training. Scripted IK seeds must reach 30% or more success on the real arm before they go into the buffer, otherwise they are not used.
- ~~Bootstrap with 10–20 teleop demos~~ **[v2]** The main path is pure RL (TD-MPC2 + HER + shaped reward). Scripted IK grasps may seed the replay buffer as RLPD-style demos. Residual RL is a fallback only (see §10).
- Reuse the reach skill and the world model.

**Milestone:** ~~stable grasp of 2–3 different objects at a success rate above 70%~~ **[v2]** die-to-zone above 70% success over N=50 episodes. A second and third object are named and pass the L1 gate before they are added.

### Phase 5: Transport, Release and Push — Stage B

- `transport(xyz)`: reach while holding an object.
- `release()`: open the gripper and check that the object is placed.
- `push(object, direction)`: non-prehensile manipulation.
- Enable self-resets so the robot puts objects back itself, allowing unattended training. **[v2]** Consider a forward-backward reset (the SERL pattern) and measure the reset failure rate.

**Milestone:** full "move cup to X" works when the skills are chained by a hand-written sequence. **[v2]** First check that the cup passes the L1 gate (the gripper must span it). If it does not, the Stage B demo object becomes a graspable substitute.

### Phase 6: Task Level — Stage B (v1) / Stage C (v2)

- **v1, LLM planner:** scene graph plus command → skill sequence with preconditions and effects. Perception checks each step, and the planner replans on failure. **[v2]** Use an API model with discrete skill IDs at 1 Hz or slower (L10).
- Log every plan, skill call and outcome.
- **v2, learned high-level policy:** RL over skills (options framework), with the logged LLM plans as a prior. **[v2]** This is Stage C research, because learned options rarely beat flat policies on real robots (L9). Hierarchy inside the world model (Director, HWM) is an alternative.

**Milestone:** new commands made only of existing skills work without new motor training. **[v2]** A held-out list of 10 commands is fixed before Phase 6 starts, and success is measured by the verifier.

## 6. Software Stack (Initial Choice)

| Area | Tool |
|---|---|
| Robot interface | LeRobot. **[v2]** Pure Python with no ROS2 (§10). lerobot 0.6.0 is in the `LIT` pixi envs. |
| Native extensions | **[v2]** Optional C++ (nanobind or scikit-build-core) or Rust (PyO3 or maturin), added only after profiling identifies a hot path |
| Packaging and environments | **[v2] Decided: a pixi package** (user requirement). It is its own workspace, separate from the `LIT` pixi.toml, which is fragile because gymnasium conflicts with sheeprl. Heavy optional stacks (perception, sim, LLM) go in pixi features and environments so the core env stays small. The layout is still open (§8). |
| Simulation | MuJoCo. **[v2] Decided** (§10). It supersedes the Isaac choice in the May 2026 vault plans for this project. The model is the **official MJCF** (onshape-to-robot, class `sts3215` actuators). Its damping, armature and friction come from Open Duck Mini and are not identified for the SO-101, so treat kp, kv, damping and armature as sysid targets. Add primitive jaw collision geoms (its mesh-to-mesh contact is slow), a rest-pose keyframe, and a test for the gripper mapping between 0–100 and radians. |
| RL | TD-MPC2 (main), with HER implemented in the replay buffer. **[v2]** Not installed and never run. The **SAC + HER baseline** is decided. DreamerV3 (sheeprl) is the tested alternative. |
| Segmentation | SAM2 **[v2] (Stage B; Stage A uses an HSV colour threshold + depth)** |
| Detection and labeling | Grounding DINO **[v2] (Stage B)** |
| Camera | pyrealsense2. **[v2]** Version 2.57.7 is installed. |
| Task planner v1 | LLM (local or API) with structured output. **[v2]** API recommended (L10). |
| Logging | LeRobot dataset format, or HDF5 |
| Experiment tracking | Weights & Biases or TensorBoard |

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Servo wear or overheating | Temperature monitoring, duty cycle limits, action smoothing. **[v2]** Measure duty cycle in Phase 0 and Phase 2. |
| Manual resets slow down training | Self-resets from phase 5; a tray or bounded workspace. **[v2]** Design for resets from Phase 0, and start with reset-free skills. |
| Sim-to-real gap | Keep sim light; domain randomization; most training on the real arm. **[v2]** Validate the sim against real trajectories. |
| Sparse rewards for grasping | HER, demos, dense shaped rewards from perception. **[v2]** Add residual RL on a scripted base. |
| Perception errors corrupt rewards | Validate the object table; use the ArUco marker as ground truth at first. **[v2]** Measure the verifier's false-accept rate before training. |
| Unsafe exploration | Hard safety layer independent of the policy |
| **[v2]** No hardware e-stop | This blocks unattended runs. Fit a power-cut button in Phase 0. |
| **[v2]** Reward or predicate gaming | Adversarial predicate tests. Watch rollout videos, not just metrics. |
| **[v2]** Clamp or config drift between sim and real | One config source of truth, loud failures, and logging of the applied action (L4) |
| **[v2]** RL cannot discover multi-step tasks from scratch | Use a residual-RL fallback path (L3), and document negative results as Stage A output. |
| **[v2]** Perception and training compete for GPU memory | Low-rate perception, cached masks, and measured VRAM |
| **[v2]** Placo IK fails silently | Check the FK residual on every command (L7). |

## 8. Open Decisions

- [ ] Task level: when to switch from the LLM planner to the learned high-level policy, and what the criteria are. **[v2] Deferred to Stage C.**
- [x] ROS2 in the loop or pure Python with LeRobot: **pure Python, with C++ or Rust for hot paths only** (§10)
- [x] Action space: **clamped joint deltas** (§10)
- [x] Visual latent state in the world model (pixels) or state-based only (object table): **pixels from the start** (§10)
- [x] Which objects to use for the first grasp experiments: **the red 16 mm die, moved into a marked zone** (§10)
- [x] Local or API LLM for the planner: **API** (default, §10)
- [x] Control frequency: **the policy runs at 10 Hz, and the safety and servo loop at 30 Hz or faster with interpolation** (§10)
- [x] **[v2]** Simulator: **MuJoCo** (§10)
- [x] **[v2]** Main RL algorithm: **TD-MPC2, with a SAC + HER baseline** (§10)
- [x] **[v2]** Skill learning: **pure RL everywhere** (§10). A fallback to residual RL is decided per case.
- [x] **[v2]** Demo source: **scripted IK seeds** (§10)
- [x] **[v2]** Autonomy and consent model: **a hardware e-stop, then tiered autonomy** (§10)
- [x] **[v2]** First real task: **die to zone** (§10)
- [x] **[v2]** Relation to `01-Projects/hil-digital-twin-package-plan.md`: **this plan supersedes it**. Reuse its pixi layout (D1), safety stack (D4) and TD-MPC2 shared-replay design (D13).
- [x] **[v2]** Relation to `lerobot-isaac-training`: **vendor** the few safety and calibration modules into this package, with no dependency on `LIT` (default, §10)
- [x] **[v2]** Pixi package layout: **`pyproject.toml` with `[tool.pixi]`** (§10)

## 9. Next Steps

1. Detail Phase 0: the environment interface, safety layer and logging format.
2. Detail Phase 2: babbling strategy and world model architecture.
3. Choose the first test objects and set up the workspace.
4. **[v2]** Resolve the §8 decisions in the interactive session, and record them in §10.

## 10. [v2] Decision Log

Each decision is recorded here once it is made, so it is not argued again.

| Date | Decision | Choice | Reason |
|---|---|---|---|
| 2026-09-30 | Deliverable | Refine this plan in place (v1 kept as `.v1.md`) | User answer |
| 2026-09-30 | Purpose | Research, then demo, then platform (§1 stages) | User answer |
| 2026-09-30 | Constraints | RL core, minimal human time, single GPU, opportunistic reuse | User answer |
| 2026-09-30 | Packaging | A pixi package with its own workspace, lock file and tasks | User requirement |
| 2026-09-30 | Skill learning | Pure RL everywhere. Residual RL stays a documented fallback only. | User chose research purity over the vault evidence (L3). The risk is accepted. |
| 2026-09-30 | Residual fallback | Decided per case, with no automatic gate | User answer. A stalled skill gets a manual review. |
| 2026-09-30 | Action space | Clamped per-step joint deltas | This avoids placo's silent IK failures (L7) and matches the existing safety clamps. The world model learns the kinematics. |
| 2026-09-30 | First task | The red 16 mm die into a marked zone, by push or place | Success is visible to one D435 (L11), and the die is already on the rig. |
| 2026-09-30 | Pixi layout | `pyproject.toml` with `[tool.pixi]`, one `src/so101_rl` package, features core, sim, perception and train | User answer |
| 2026-09-30 | Runtime language | Pure Python. C++ (scikit-build-core or nanobind) or Rust (maturin or PyO3) only for hot paths that profiling identifies. | User answer. The pixi package then needs a compiled build backend. Start pure Python. |
| 2026-09-30 | World-model state | Pixels from the start (low-res RGB and optional depth, plus proprioception) | User answer. This goes against the recommendation, so the risks are VRAM, data volume and debuggability. They are mitigated by a state-based debugging baseline. |
| 2026-09-30 | Control rate | The policy runs at 10 Hz, and the safety and servo loop at 30 Hz or faster with interpolation | Leaves budget for MPPI planning. |
| 2026-09-30 | Demos | Scripted IK seeds for die-to-zone in the replay buffer (RLPD-style), with an FK residual check | Needs no human time. |
| 2026-09-30 | Planner LLM | An API model (default, not asked) | L10: local models of 4B or less collapse. |
| 2026-09-30 | LIT reuse | Vendor the safety, calibration and hand-eye modules, with no dependency on `LIT` (default, not asked) | Keeps the pixi package self-contained. |
| 2026-09-30 | Stage A scope | Lean critical path: safety + relay, colour die tracker, state WM, SAC then TD-MPC2 on reach. The pixel WM, SAM2/GDINO and full sysid run in parallel. | Critic finding (Step 4.4), user agreed |
| 2026-09-30 | Stage A perception | HSV colour + depth. SAM2 and Grounding DINO move to Stage B. | Saves VRAM, and one red die needs nothing more |
| 2026-09-30 | RL order | SAC + HER first on the real arm, then TD-MPC2 + HER | Gets a first real result faster. Goal-conditioning TD-MPC2 is custom work. |
| 2026-09-30 | Overlap | This plan supersedes `01-Projects/hil-digital-twin-package-plan.md` and reuses its D1, D4 and D13 assets | User answer |
| 2026-10-01 | Thermal cooldown | On a thermal trip the arm goes through the safe-state sequence to the rest pose, then torque off. It never holds pose while energised. It resumes only through a full re-energise. | User answer to the wave-1 review, which found the Phase 0 plan and code contradicting the main plan |
| 2026-10-01 | MCU reset | The watchdog MCU boots LATCHED (reason MCU_RESET), so only the physical reset button re-enables power after any MCU reset | User answer to the wave-1 review |
| 2026-09-30 | RL algorithm | TD-MPC2 as main, plus a SAC + HER baseline on reach | Tests whether the world model helps. Fitting in 10 GB is a hypothesis to be measured in Phase 2. TD-MPC2 is **vendored** (MIT licence, no packaging files in the repo). |
| 2026-09-30 | Simulator | MuJoCo, using the official TheRobotStudio MJCF, system-identified | Light on VRAM, good contacts. Supersedes the Isaac choice from the earlier plans for this project. |
| 2026-09-30 | Autonomy | Hardware e-stop, then tiers: attended until the 30-minute soak test passes, then unattended with a watchdog that cuts power | Resolves the conflict between minimal human time and the current consent rule. |

## 11. [v2] Sources

**Vault notes** (paths relative to `~/Documents/Vaults/Local`):

- *Archive, experiments and lessons:*
  - `04-Archive/2026-06-28-so101-carryplace-rl-campaign.md`
  - `04-Archive/2026-07-24-residual-rl-clamp-saga-lessons.md`
  - `04-Archive/2026-06-10-sim-rl-manipulation-lessons.md`
  - `04-Archive/2026-06-09-staged-reward-tuning-completed.md`
  - `04-Archive/2026-07-19-gpu-campaign-3day-lessons.md`
  - `04-Archive/2026-07-04-act-real-first-hw-success.md`
  - `04-Archive/2026-09-10-grill-review-so101-safety-fixes.md`
  - `04-Archive/2026-09-10-claude-driven-so101-recording-research.md`
  - `04-Archive/2026-07-11-vla-jepa-rtx3080-finetune.md`
  - `04-Archive/2026-05-14-world-models-swap-matrix.md`
- *Wiki entities and concepts:*
  - `05-Wiki/entities/SO-101.md`
  - `05-Wiki/entities/TD-MPC2.md`
  - `05-Wiki/entities/HIL-SERL.md`
  - `05-Wiki/concepts/Arm-Safety-Envelope.md`
  - `05-Wiki/concepts/SO-101-Inverse-Kinematics-(Placo-LeRobot).md`
  - `05-Wiki/concepts/Autonomous-Robot-Data-Collection.md`
  - `05-Wiki/concepts/Arm-Bring-Up-Ladder.md`
  - `05-Wiki/concepts/Safe-RL-for-Robot-Manipulation.md`
  - `05-Wiki/concepts/Sim-Calibration-Loop.md`
  - `05-Wiki/entities/MuJoCo.md`
  - `05-Wiki/research/2026-09-20-ai-controlling-a-robotic-arm.md`
- *Wiki research pages:*
  - `05-Wiki/research/2026-06-21-hierarchical-planning-world-models-robotics.md`
  - `05-Wiki/research/2026-06-21-rl-reward-coupled-world-model-training.md`
  - `05-Wiki/research/2026-06-23-world-model-pipeline-practical-plan.md`
  - `05-Wiki/research/2026-08-02-llm-robot-control.md`
  - `05-Wiki/research/2026-08-08-onyx-per-joint-composition.md`
- *Project plans:*
  - `01-Projects/lerobot-isaac-hil-serl-plan.md`
  - `01-Projects/lerobot-isaac-sim-to-real-plan.md`
  - `01-Projects/hil-digital-twin-package-plan.md`. This project overlaps with it: both use pixi, the real SO-101 with the D435, and a TD-MPC2 track (see §8).

**Code:**

- `LIT/SESSION_STATE.md`
- `LIT/plans/2026-06-23-world-model-training-plan.md`
- `LIT/plans/2026-09-12-overhead-verification-camera-plan.md`
- `/mnt/data/workspaces/robot_data_runner`
- `~/tools/claude_code/skills/binary_success_verifier`

**Papers:**

| Area | Papers |
|---|---|
| Model-based RL | TD-MPC2 (2310.16828), DreamerV3 (2301.04104), DayDreamer (2206.14176) |
| Hierarchy | Director (2206.04114), HWM (2604.03208, provisional) |
| Goal-conditioned RL and HER | HER (1707.01495), HIRO (1805.08296) |
| Human-in-the-loop RL and VLA | HIL-SERL (2410.21845, verified), ConRFT (2502.05450), LoRA forgetting (2603.11653) |
| Autonomy and resets | Mirchandani, CoRL 2024 (2411.01813) |
| LLM planning | SayCan (2204.01691) |
| Simulation | ManiSkill3 (2410.00425) |
