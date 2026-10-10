# SO-101 Self-Learning Robot Arm — Project Plan

> **Version 3 (2026-10-05).**
> - Version 2 (2026-09-30) added evidence from the Second Brain vault and existing code to the v1 plan (`so101-self-learning-rl-plan.v1.md`).
> - Version 3 makes **GPU sim-first pre-training** the main learning path: §5 "Sim-first track", and §10 rows dated 2026-10-05.
>
> Markers:
> - **[v2]** marks new or changed content.
> - **[v2-conflict]** marks a place where vault evidence contradicts v1. Each conflict is an open decision in §8 until it is resolved.
> - **[v3]** marks the sim-first changes (2026-10-05).
> - **[v4]** marks the AI training connector (2026-10-07, §12).
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
| Training compute | **[v2]** One RTX 3080 with 10 GB, plus 16 CPU cores and 31 GB RAM. **[v3] The driver works as of 2026-10-05** (610.57, CUDA 12.6 in the `default` pixi env; a matmul is verified). One vault note calls this machine a laptop, so check its thermals and power under multi-hour GPU sim load. |
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
| L6 | ~~**Pure sim pretraining did not transfer** (community HIL-SERL SO-101 result). Use sim for augmentation or warm starts only.~~ **[v3] Revisited 2026-10-05.** That was a single anecdote. Squint (arXiv 2602.21203, 2026) trained visual SAC on 8 SO-101 tasks in ManiSkill3 in about 15 min on one RTX 3090 and transferred them **zero-shot** to a real SO-101, with 73 of 80 trials succeeding. Its recipe: 10 Hz joint-delta actions, 1024 parallel envs, broad domain randomisation, and a wrist camera. The vault's SO-101 sim has still never been validated against the real arm. | `05-Wiki/entities/SO-101.md`, `05-Wiki/entities/HIL-SERL.md`, arXiv 2602.21203 | **[v3] Sim-first pre-training is now the main path** (§5 "Sim-first track", §10). Sim-vs-real validation and sysid become gates, not extras. |
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
| **[v3] GPU sim pre-training (MuJoCo Warp)** | Trains each skill to high success in 1024–2048 parallel randomised sims on the 3080, so the real arm only fine-tunes | **Every Stage A/B skill** (§5 "Sim-first track"). Uses the official SO-101 MJCF, system-identified. Randomisation priority: camera and tracker pose, then servo dynamics and delay, then contact and object properties, then lighting. |
| ~~Teleop demos (RLPD-style)~~ **[v2] Scripted or prior demos (RLPD-style)** | Seeds the replay buffer with 10–20 demos | Grasp. **[v2-conflict]** With no leader arm, the demos come from a scripted IK controller, from sim, or from `so101-pickplace-new`. |
| **[v2] Residual RL (fallback only)** | RL learns a correction on top of a scripted IK controller | **Not the main path**: the user decided on pure RL everywhere (see §10). It is kept as a documented fallback in case pure RL stalls. A scripted controller reached 67–80% in sim (L3). |
| **[v3] Massively parallel SAC (FastSAC/Squint-style)** | Off-policy SAC tuned for thousands of parallel envs: true clamp-unit action bounds, low initial entropy, a moderate update-to-data ratio | **The sim pre-training learner.** It also stays as the real-arm baseline (SB3 SAC + HER) that shows whether TD-MPC2's world model helps. |
| **[v3] Offline TD-MPC2 pre-training from sim rollouts** | TD-MPC2's world model and critic are pre-trained on logged sim SAC rollouts, so the real-arm TD-MPC2 starts informed | Before every real-arm TD-MPC2 run |
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
3. **[v3] The GPU sim pre-training track for reach** (S0–S3 below): a batched MuJoCo Warp env with servo parity, the sysid gate, and SAC reach at ≥ 95 % in randomised sim.
4. A state-based world model, pre-trained from sim rollouts.
5. Reach on the real arm, starting from the sim policy: zero-shot evaluation first, then fine-tuning with SAC + HER. TD-MPC2 + HER is pre-trained from sim rollouts and compared against it.

These are developed **in parallel, off the critical path**, and attach when their own gates pass:

- **Pixel world model.** "Pixels from the start" is kept, but it does not block Phase 3.
- **SAM2 and Grounding DINO**, in Stage B.
- ~~**Full MuJoCo sysid.** In Stage A, sim covers kinematics, safety-box validation and the HER unit tests.~~ **[v3]** Sysid is now on the critical path, as gate S2.

### [v3] Sim-first track (decided 2026-10-05)

Each skill is pre-trained in GPU sim to high success, deployed on the real arm, then fine-tuned with real RL. The real arm is used for evaluation, sysid data and fine-tuning, not for learning from scratch.

**S0: GPU sim backend.**
- **Version pin:** one MuJoCo version in *every* pixi env, so the physics we train on is the physics we test on. On 2026-10-05 `cpu` had mujoco 3.13 / torch 2.11 while `default` had mujoco 3.3.2 / torch 2.7.1. Add `mujoco-warp` + `warp-lang` (conda-forge or PyPI, torch interop through `wp.to_torch`, no JAX) to the `sim` feature.
- **Collision meshes:** replace the MJCF's link collision meshes with primitives or convex hulls of about 32 vertices or fewer, like the jaw boxes we already add.
- **Integrator:** check the MJCF integrator against MuJoCo Warp's supported list. Its PyPI metadata lists IMPLICITFAST as unsupported.
- **Measure, as gates:**
  - steps/s against N parallel envs on our scene;
  - VRAM;
  - contact-buffer overflow (`nconmax`);
  - parity against CPU MuJoCo, meaning the same trajectory within tolerance.

**S0 spike result (2026-10-05, measured).** Full report: `plans/reviews/2026-10-05-mjwarp-feasibility-spike.md`.
- **Works:** MuJoCo Warp 3.14 with mujoco 3.14 and warp-lang 1.17, all from conda-forge, runs our full scene (repo `load_model`) on the 3080 with no blockers.
- **Mesh collisions:** supported. The 10 arm-link collision meshes have 9k–27k vertices each and needed no simplification.
- **Required changes:**
  - `margin=0` on the `jaw_col` default, because MJWarp rejects non-zero margins on box pairs.
  - `njmax ≥ 256`, because the default 64 overflows, and overflow only prints a warning.
  - `nconmax = 32` per world is enough.

  | N envs | env-steps/s | VRAM |
  |---|---|---|
  | 256 | 300k | 0.4 GB |
  | 1024 | 745k | 0.8 GB |
  | 2048 | 1.09M | 1.3 GB |
  | 4096 | 1.68M | 2.4 GB |

  For comparison, one CPU core runs about 77k steps/s.
- **Parity with CPU MuJoCo:** at most 5e-7 rad on the arm joints over 300 steps, with matching contact counts. Grasp-level contact parity has **not been tested yet**.
- **Zero-copy torch interop:** confirmed, via `wp.to_torch`.
- **Pin to verify in the repo:** `mujoco 3.14.*` and `mujoco-warp 3.14.*` in both envs. The spike resolved torch 2.12 and numpy 2.5, but **lerobot 0.6.1 needs torch < 2.12 and numpy < 2.3**, so the repo pin must be `pytorch-gpu 2.11.*` (CUDA 12.x) with numpy 2.2. Re-run `test_sim_model.py` under mujoco 3.14 before adopting it.

**S1: Batched servo-parity env.** A torch-batched goal env on MuJoCo Warp that reproduces the ServoCore tick semantics exactly:
- 3 servo ticks per policy step;
- the same clamp chain in tick units (delta clamp, joint limits, velocity cap, tracking-error clamp, box);
- the calibration mapping;
- an action-delay FIFO;
- the applied post-clamp action as the logged action.

A **parity test** runs the same action sequence through `ServoCore` + `MujocoBus` and through the batched env, and requires identical goal sequences and matching trajectories within tolerance. A policy trained in sim must not see any rule the real servo loop wouldn't apply.

**S2: Sysid gate.**
- **Data:** log real multisine and random-policy trajectories through ServoCore. This needs the Phase 0 e-stop and deadman, and runs Tier 0 (attended).
- **Fit:** fit kp, kv, damping, armature, friction and delay as parallel worlds with CMA-ES or Optuna. The gripper is the likeliest misfit, and BAM / Rhoban STS3215 data can serve as a prior.
- **Pass:** held-out `compare-real` within 5 % RMS of each joint's range.
- **Use:** the fitted values become the **centre** of the randomisation ranges.

**S3: Randomisation set**, in priority order. Each range is widened only after the task reaches ≥ 70 % success in sim.
1. D435 / tracker pose and noise. The simulated die tracker reproduces the real one's noise, delay and occlusion, including UNKNOWN when the gripper covers the die.
2. Servo kp, kv, friction, damping and armature (±25 % around the sysid fit), plus a servo delay of 2–5 ticks.
3. Backlash and proprio noise.
4. Die mass, size and friction, and gripper closing speed.
5. Lighting. This only matters for the pixel world-model track.

**S4: Per-skill pre-training.**
- **Learner and inputs:**
  - Massively parallel SAC with 1024–2048 envs.
  - The policy sees state only: noisy joints, the last action, the goal, and the die from the simulated tracker.
  - The critic is privileged and sees the true sim state.
- **Gates:**
  - **Feasibility and predicate gates first** (L1, L2): the task must be reachable, and the success predicate must not be gameable, both checked in sim before any reward tuning.
  - **Exit gate:** ≥ 95 % success over 1000 randomised episodes, with no predicate gaming visible in rendered rollouts.
- **Wall-clock estimates on the 3080:** these are research estimates, not measurements.

  | Skill | Estimate |
  |---|---|
  | Reach | 10–30 min |
  | Die push to zone | 1–4 h |
  | Place | 4–12 h, and may need scripted-IK seeds (L3) |

**S5: Real deployment and fine-tuning.**
1. Zero-shot evaluation on the real arm, with N episodes and a confidence bound.
2. Fine-tuning with SAC + HER on real data.
3. TD-MPC2: its world model and critic are pre-trained offline from sim rollouts, then it fine-tunes on real data and is compared against SAC.
4. Record the sim → real success gap per skill. It is the Stage A research result.

**Pixels.** The pixel world model stays on its parallel track. MuJoCo Warp's batch renderer gives 64–96 px overhead frames at about 256 envs. A 1M-step pixel replay buffer does not fit in VRAM, so keep it in host RAM at 300k steps or fewer.
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
        - **Bounds:** the sequence is **bounded by `bound_s` (20 s; see §10, 2026-10-01)**. After that it falls through to torque-off in place, and finally to the deadman.
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
- **[v3]** GPU sim: S0 (backend and version pin) and S1 (servo-parity batched env) of the sim-first track belong to Phase 0. S2 sysid data collection follows once the e-stop and deadman pass.
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

- Goal-conditioned policy `reach(xyz)`. **[v3] Order:**
  1. Sim pre-training with massively parallel SAC (S4).
  2. Zero-shot real evaluation.
  3. Real fine-tuning with SAC + HER.
  4. TD-MPC2 + HER, pre-trained offline from sim rollouts, fine-tuned on the real arm and compared against SAC.

  ~~SAC + HER first on the real arm from scratch.~~
- **[v2]** Train a SAC + HER baseline on the same task, with the same budget of real steps. Compare the steps each one needs to reach 90% success.
- Reward: negative distance from gripper to goal, with a success bonus.
- ~~Warm-start briefly in MuJoCo, then train on the real arm.~~ **[v3]** Pre-train in GPU sim to ≥ 95 %, then fine-tune on the real arm (sim-first track).
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
- **[v3]** Pre-train die-to-zone in GPU sim first (S4, push and place, with the simulated tracker including occlusion). Real fine-tuning starts from the sim policy. The gravity-return tray is modelled in sim, too.

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
| RL | **[v3]** **Sim:** massively parallel SAC (FastSAC/Squint-style) on MuJoCo Warp. **Real:** SAC + HER (SB3) fine-tuning, and TD-MPC2 + HER (vendored), pre-trained offline from sim rollouts. DreamerV3 (sheeprl) is the tested alternative. |
| GPU simulation | **[v3]** MuJoCo Warp (`mujoco-warp`, `warp-lang`) with torch interop, no JAX, in the `sim` feature. It is alpha, so its throughput and parity are gates (S0). Fallback: ManiSkill3, which has SO-101 tasks via Squint but runs on PhysX and would need a port. Rejected: Isaac Lab (needs 16 GB or more) and MJX/Playground (JAX next to torch). |
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
| Sim-to-real gap | ~~Keep sim light; most training on the real arm.~~ **[v3]** Sysid gate S2, with fitted values as the randomisation centre; the S1 servo-parity env, so sim applies exactly the real clamp and tick rules; a simulated tracker with real noise, delay and occlusion; and real fine-tuning. The sim → real gap is measured per skill. |
| **[v3]** MuJoCo Warp is alpha (nondeterministic GPU execution, some solvers and integrators unsupported) | Gate S0 measures throughput and parity against CPU MuJoCo before any training. ManiSkill3 is the fallback. |
| **[v3]** The overhead camera and push contacts are not covered by the Squint evidence (it used a wrist camera) | Simulate the overhead tracker with occlusion. Treat die push and place transfer as the open research question, and measure it. |
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
- [x] **[v2]** Simulator: **MuJoCo** (§10). **[v3]** GPU: **MuJoCo Warp**, sim-first pre-training (§10).
- [x] **[v2]** Main RL algorithm: **TD-MPC2, with a SAC + HER baseline** (§10). **[v3]** Sim: massively parallel SAC; real: SAC + HER fine-tuning, and TD-MPC2 pre-trained from sim rollouts (§10).
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
| 2026-10-01 | Safe-state bound | Keep the slow 240 ticks/s ramp (about 21°/s) with `bound_s` = 20 s. The loader computes the minimum from the calibration. This replaces the earlier 5 s figure, which cannot be met at that speed. | User answer. The worst case (joint limit → home → rest) is 15.33 s with the 1 s margin, computed by `safety.config_check` for the shipped calibration. |
| 2026-10-02 | Heartbeat miss policy | In normal running, latch when the MCU reports latched or after 5 consecutive missed heartbeat acknowledgements (about 150 ms, inside the MCU's 1 s deadline). Re-energise stays strict: one miss aborts it. | User answer to the U3 review. One lost USB echo must not end an unattended run. |
| 2026-10-02 | State record `read_ok` | The F3 shared-memory state record gets a `read_ok` flag, so readers can tell a fresh reading from a repeated one | User answer to the U3 review |
| 2026-10-02 | Stop during re-energise | A stop (Ctrl-C, SIGTERM, SIGHUP, SIGQUIT, STOP_REQ) during the re-energise ramp hands over to the time-bounded safe-state sequence back to rest, with heartbeats. A stop before torque-on never energises. A second stop signal, or the e-stop, cuts torque or power immediately. | User answer to the U3 fix report. Keeps the rule that every exit goes through the safe-state sequence. |
| 2026-10-05 | Sim role | **Sim-first pre-training**: each skill is trained to ≥ 95 % in GPU sim with heavy randomisation, deployed zero- or few-shot on the real arm, then fine-tuned with real RL. Supersedes "sim light, warm-start only" (L6 revisited). | User request plus research: Squint, arXiv 2602.21203, zero-shot SO-101 transfer with 91 % success |
| 2026-10-05 | GPU simulator | MuJoCo Warp on the RTX 3080, in the same engine family as our MJCF and MujocoBus. One MuJoCo version pinned across all pixi envs. ManiSkill3 is the fallback. | User answer. Isaac Lab doesn't fit 10 GB, and MJX would need JAX next to torch. |
| 2026-10-05 | Sim/real algorithm split | Massively parallel SAC in sim. Its rollouts pre-train TD-MPC2's world model and critic offline, then SAC + HER and TD-MPC2 fine-tune on the real arm. | User answer. TD-MPC2's MPPI is too slow at 1000+ envs. |
| 2026-10-05 | Pre-training observations | State plus a simulated overhead tracker (noise, delay, occlusion), with a privileged critic. Works with the current D435. Pixels stay on the parallel world-model track, and no wrist camera is added. | User answer |
| 2026-10-05 | Die success definition | Reward and success use the **tracker-reported** die position, held at the last known value while hidden. Success needs the die visible in the zone for 2 consecutive steps. The same rule applies in sim and on the real arm, and the sim's true position goes only to the critic. | User answer (sim-track plan D-REWARD) |
| 2026-10-05 | Zero-shot → fine-tune gate | Zero-shot evaluation uses 50 reach / 30 die episodes, with a speed ladder of 0.5 → 0.75 → 1.0. Real fine-tuning starts only if the 95 % Wilson lower bound is at least 60 %. Otherwise one budgeted sysid/DR revision loop comes first. | User answer (D-ZEROSHOT) |
| 2026-10-05 | Sim SAC implementation | A small self-written SAC with a GPU ring replay. The actor has SB3's shape, so its weights transfer to SB3 for real fine-tuning. The critics are privileged and are not transferred. | User answer (D-SAC, D-ACTOR-SHAPE) |
| 2026-10-05 | GPU budgets | Reach ≤ 6 GPU-h and die ≤ 30 GPU-h, with at most 3 attempts each. Training pauses above 85 °C GPU. Die waits for `sim_params` v1, which uses real gripper and die data. | User answer (budgets, D-DIE-PARAMS) |
| 2026-10-05 | Sim-track technical defaults | The remaining 15 decisions in `plans/sim-track-implementation-plan.md` ("Decisions…") are accepted as stated: D-CHAIN, D-PIN/D-CUDA, D-TIMESTEP 1/510 s, D-CONFIG, D-OBS, D-TRIPS, D-TRAY, D-SYSID, D-BACKLASH, D-DR-FREQ, D-EXIT-GATE, D-TDMPC2-LOOP, D-SUPERSEDE, D-DEPS. | Accepted by default after review; the user confirmed the four project-shaping ones |
| 2026-10-05 | Seeing training in the sim | Three ways to watch training:<br>• a **live 3D viewer**: a separate process showing selected parallel worlds (0, best, worst), which can be attached or detached during a run;<br>• **rollout videos** every N minutes and at every checkpoint, from the D435 overhead pose and a free camera;<br>• a **live TensorBoard dashboard** with curves, SAC internals, GPU health and the embedded videos.<br>Visualisation must never slow training by more than 5 %, and a crash in it never stops training. Real-arm replay is not included. | User request; sim-track plan unit SUV (frozen interface FS11) |
| 2026-10-05 | Die success height bound | ~~16 mm, measured relative to the goal~~ superseded on 2026-10-06 (see below) | Found while building SU0 |
| 2026-10-06 | Reward structure | The reward is **modular**: a sum of named, configurable terms (dense distance, success, optional penalties), each computed from a single step so HER relabelling works, and selected and weighted per task in config. A validator enforces that **no non-terminal step earns a positive reward** for any configured combination. The default die reward is −distance − 1 when not in the zone and 0 in the zone, so finishing is always best. | User answer to the SU0 review. The original +bonus-per-step reward paid the policy never to finish (dip in and out, or hide the die). |
| 2026-10-06 | Die success height | The die centroid may be at most **16 mm above the goal**. This replaces the code's 8 mm, which was too tight for D435 depth noise. A die held clearly in the air (≥ about 2 cm) still fails. | User answer to the SU0 review |
| 2026-10-06 | Die must be settled | Die success requires the die to be **visible in the zone for 2 consecutive steps and to move ≤ 3 mm between them**, by tracker position. A slide-through or a nudge-out does not count. | User answer to the SU0 review. Matches the plan's L2 gaming table. |
| 2026-10-06 | Workspace box floor | Lower the Cartesian box floor from z = 0.010 m to **table + 5 mm (z = −0.015 m)** everywhere. At 0.010 m (3 cm above the table) the jaws could never reach a die on the table, so die-to-zone was impossible. The measured rest pose (tip z = −0.012) is now inside the box. | User answer, after SU1 found the blocker while building the grasp check |
| 2026-10-06 | Layer B parity tolerance | Keep the 1e-4 rad trajectory bound. Accept 3 documented xfail sequences (mesh self-contact at the shoulder_lift limit, and sustained ±force-limit commands; max 1.7e-3 rad), with targets identical on every tick | User answer |
| 2026-10-06 | Sysid gate | The held-out sim-vs-real error at 30 Hz must be **≤ 1 %** of each joint's range, replacing 5 %. The 5 % gate could not detect a halved servo gain (≈ 0.6 %) or a 3-tick delay (≈ 1.5 %); the measured noise floor is 0.03–0.07 %. It may be revisited after the first real-arm fit. | User answer to the SU2 findings |
| 2026-10-06 | Servo parameter identification | Fit servo kp, kv and armature as **one ratio group** plus the integer delay, which is identifiable. Randomise the group by ±40 % around the fit, and frictionloss across its full plausible range. Unidentifiable parameters must not be reported as precise. | User answer to the SU2 findings (kp/kv/armature act only as ratios under the clamp chain; friction is invisible under ±1-tick noise) |
| 2026-10-06 | Friction DR ceiling | frictionloss is randomised in 0–0.21 N·m (4× the MJCF prior, about 7 % of stall torque). Revisit if the real arm shows sticking. | User answer |
| 2026-10-06 | Re-energise start rule | Re-energise starts only from the measured rest pose (within tolerance) or from a pose with the tip **≥ 1 cm above the table** (the old 0.010 m floor). The lower box floor (−0.015 m) applies only while running. A fallen arm still needs a human. | User answer to the integration review: the lower floor had silently widened the start rule |
| 2026-10-06 | Gripper floor check | The box floor applies to the **lowest point of the gripper** (several FK points on the gripper body), not only the jaw tip, so no part of the gripper is driven into the table. The safety pipeline and the batched sim chain must implement it identically. | User answer to the integration review: tip-only allowed pressing about 3 mm into the table |
| 2026-10-07 | Gripper floor clearance | Supersedes the 5 mm clearance for the gripper-point check, which made the 16 mm die ungraspable: the jaw centres sit about 11 mm above the lowest gripper vertex, and the best margin was +0.05 mm on the points and −1.1 mm on the mesh. The **lowest gripper/wrist point must stay ≥ 1 mm above the table**. Its BOX_PRESENT tolerance for these points is **2 mm**, so nothing presses more than about 1 mm into the table. The jaw-tip floor stays at −0.015 m. | User answer after the executor stopped on the feasibility test |
| 2026-10-07 | Re-energise start floor value | The start floor is the old box-floor value, tip **z ≥ 0.010 m in the base frame (3 cm above the table)**, as shipped in `safe_state.start_min_tip_z_m`. The rest-pose tolerance branch is unchanged. | User clarification. The earlier row said "≥ 1 cm above the table"; this row supersedes it. |
| 2026-10-07 | AI connector (P1) | `so101_rl.lab` (a Python API) plus a thin stdio MCP server with 12 tools in LA1. Training jobs run as detached processes under a `systemd --user` supervisor and outlive the MCP session. A read-only local web dashboard (127.0.0.1) supersedes "no web UI beyond TensorBoard". The trainer interface is general in shape; only FastSAC, SB3 SAC+HER and TD-MPC2 are implemented. Done = full unattended sim runs, plus unattended real-hardware runs after Tier 1 approval. | User request and answers; design `plans/ai-connector-design.md` v2 |
| 2026-10-07 | Signed human records (P2) | Approvals (`APPROVE <run_id> <episodes>x<speed>`), S4-exit checklist ticks, needs-human acks and the Tier 1 record are all Ed25519-signed. They are made only by root-owned scripts outside the repo (`so101-approve`, `so101-gate-sign`, `so101-ack`) against a root-owned `allowed_signers`. The connector verifies records and never creates them. The S4 exit gate: the AI reviews first, and the human signs later; nothing reaches the real arm unsigned. | User answers |
| 2026-10-07 | Servo park / idle / resume (P3, SV1) | Between Tier 1 runs the arm holds at `home`. A servo-side idle timer (`park_timeout_s`, default 600) rests the arm and turns torque off. `RESUME_REQ` triggers the full existing re-energise. Every existing safety obligation is kept. This is a servo work item with its own review. Supersedes "rest + torque off between runs". | User answer to the design critic |
| 2026-10-07 | Firmware arm timeout (P4, FW1) | The watchdog goes DISARMED → LATCHED after 120 s without ARM (new reason 8). A button press therefore means "arm within 2 minutes". The servo's `--arm-timeout-s` is 110 s. At Tier 1 the connector may start the servo itself, inside that window. | User answer to the design critic |
| 2026-10-07 | Tier 1 session rules (P5) | Session cap ramp: 60 → 120 → 240 → 480 min, raised only after clean sessions. A clean session has no faults beyond at most 1 auto-resumed thermal trip. Also required:<br>• a power-cycle and relay-weld check between sessions;<br>• quiet hours 23:00–07:00;<br>• a minimum 2-minute gap between runs;<br>• a policy allow-list;<br>• at most 240 arm-minutes per day, with the day ending at local midnight;<br>• a ≥ 120-minute strict soak, verified at promotion and at every admission. | User answers |
| 2026-10-07 | Budget accounting (P6) | Probes (≤ 10 min, ≤ 1 GPU-h pooled per task) are not attempts. GPU-h is counted as progress-holding time, with a stall cap. The ledger is derived from the run dirs, has an `import-run` CLI, and halts when attempts shrink. | User answer |
| 2026-10-07 | Review triggers via connector (P7) | Sim attempts: the AI is the first reviewer and may pause and resume at most twice, then the run stops. Probes continue. Real runs stop and need a signed ack. The method is never switched automatically. | User answer |
| 2026-10-07 | Thermal fail-closed (P8) | GPU core: pause at 85 °C, resume at 75 °C (as logged before). GPU memory junction: pause at 95 °C, resume at 85 °C, when observable via NVML. Six empty samples pause all jobs and raise a needs-human item. | User answer |
| 2026-10-07 | Proposals inbox (P9) | The AI proposes plan changes only as files in `plans/proposals/`. A human merges them into §10. | User answer |
| 2026-10-07 | Authority enforcement (P10) | Real-arm authority is enforced with **two dedicated Linux accounts**, which supersedes the "same-user" P2 assumption.<br>• `so101-approver` holds the signing key in a 0700 home. Signing happens only from a login on a Linux text console, never via su or sudo from the user's terminal. Shims run with a scrubbed environment (`env -i`, `python3 -I`, absolute `ssh-keygen`, no SSH_AUTH_SOCK, ASKPASS or DISPLAY).<br>• `so101-servo` is the sole owner of the arm and MCU serial devices (udev; the user is removed from dialout). It runs a root-owned, human-reviewed snapshot of `servo/`, `safety/` and config as a system unit, and it verifies the signed records itself before ARM, before RESUME_REQ and before the first RUN from a new client. It keeps the consumed-nonce ledger and the authoritative session id.<br>The editable lab/supervisor/MCP can then only fail to schedule; they cannot move the arm unsigned. Claude Code permission rules and env flags count only as guard rails against honest mistakes. | Security review (2 blockers: terminal hijack and same-user verifier bypass); user answer |
| 2026-10-07 | Approval hardening (P11) | Approvals carry a per-invocation **challenge word**, `APPROVE <run_id> <episodes>x<speed> <CHAL>`, and every request field is strictly validated: ASCII grammar, typed values, and rejection of anything with control or Unicode tricks. The machine fields and a highlighted diff sit directly above the prompt, and AI text is escaped and boxed. **Anti-rollback:** `seq` and `supersedes_sha256` on tier and gate records, a current pointer and revocation list in the approver-owned record store, sha256 of the policy, soak reports and hw artifacts inside the signed body, an explicit `type` field, one signature namespace per record type, and a namespace-restricted allowed_signers file. Files are read once (O_NOFOLLOW) and signed over the exact bytes shown. A hardware security key is not adopted for now. | User answer to the security review |
| 2026-10-07 | SV1 scope amendment (P3a) | Every re-energise inside a session lands in **HOLD(parked)** at home, so the stale-command fuse cannot trip before a client attaches.<br>• PARK is accepted only within `park_tolerance_ticks` of home and with `block_ok`.<br>• Parked HOLD runs the present-pose box check, including the gripper points (1 mm / 2 mm), and clamps the re-write.<br>• `RESUME_REQ` needs an admitted client and no fresh-process or human fault.<br>• `--exit-on-terminal` makes the servo exit on USER_STOP, on fresh-process or human trips and on `--max-ticks`, so the relay opens.<br>• Command dedupe is monotonic, with a `client_id`.<br>• `park_timeout_s` stays 600 until HS8 passes (< 50 °C after a 10-min hold following a 20-min run); then a default near `min_gap_min` is proposed. | Safety panel (3 blockers); design v3 |
| 2026-10-07 | Clean session amendment (P5a) | The clean-session verdict has two phases: `servo_clean` at exit, then `weld_ok` at the next ARM (first MCU reason ∈ {1, 6, 8}, never 7). A parked thermal trip that is resumed on request counts toward the ≤ 1 allowance. A cap raise is a new signed tier record. | Safety panel; design v3 |
| 2026-10-07 | Session lifecycle and units (P12) | Jobs and perception run as transient `systemd --user` units, so a supervisor restart never kills them. The servo runs as the system unit `so101-servo@<session>` from the snapshot, with the pinned argv in the root-owned unit. A session closes by STOP_REQ plus `--exit-on-terminal`; the fallback is `systemctl stop` (one SIGTERM, then SIGKILL at bound_s + 10). The button is requested only once the servo is waiting (servo wait ≤ 110 s; FW1 120 s runs from DISARMED entry). | Safety panel; design v3 |
| 2026-10-07 | Device identities (P13) | `port` and `heartbeat_port` are `/dev/serial/by-id/` paths owned by `so101-servo`. A session whose devices are unresolvable or wrongly owned is refused before any button press is asked for. | Safety panel; design v3 |
| 2026-10-07 | Servo-side Tier 1 rules (P14) | The servo snapshot enforces **all** Tier 1 rules from its own ledger: the signed records, the session cap, the policy allow-list, quiet hours (23:00–07:00), the 2-min gap between runs and 240 arm-min per day. The connector's copies are only early warnings. | User answer (U7) |
| 2026-10-07 | Connector build-plan defaults D1–D9 (decided overnight; open for review) | These are the defaults recommended in `plans/connector-implementation-plan.md` and taken overnight, while the user was away and had asked to keep going.<br>• **D1:** the servo authority check runs synchronously in the tick (heartbeat first, 120 ms ssh-keygen timeout, warm-up at BOOT, 1 s denial back-off). It moves to an async worker only if HS8 shows heartbeat gaps > 200 ms.<br>• **D2:** exit code 5 = terminal trip under `--exit-on-terminal`, 6 = LATCHED.<br>• **D3:** `so101_rl.authority` is a stdlib-only package, installed verbatim in the snapshot.<br>• **D4:** the snapshot env is a root-run `pixi install --locked -e robot` in `/opt/so101_rl/<sha>/`.<br>• **D5:** LA4 and LA5 are split into -SW and -HW parts.<br>• **D6:** shm layout v3 (STATE 232 → 248 B; old tick logs become unreadable).<br>• **D7:** the firmware line budget may grow to 100.<br>• **D8:** MCP ↔ supervisor communication is files only, and the server runs in the `cpu` env.<br>• **D9:** a new HS9 (a 120-min strict soak with ≥ 2 parked segments). By-id paths live only in the pinned unit argv. | Implementation plan; master defaults while the user was away |
| 2026-10-08 | MVP scope | **Keep the package small.** The MVP is: an AI trains a **reach** policy in GPU sim through a minimal MCP connector (6 tools, run dirs, transient systemd units, budgets and thermal limits, TensorBoard only), followed by an **attended Tier 0 zero-shot reach eval** on the real arm through the existing servo process, not through the connector. Everything else is deferred (not cancelled), including die-to-zone, TD-MPC2, real fine-tuning, the web dashboard, the supervisor daemon, SV1, OS1/P10/P11/P14 and Tier 1. Existing code is frozen, not deleted. Ordered steps: `plans/mvp-plan.md`. | User request and answers |
| 2026-10-08 | Reach episode length and goal set | The reach episode length goes from 50 to **100 policy steps** (10 s at 10 Hz), and the shared reachable-goal table (`env/goals.py`, used by both the sim and the real env) is **trimmed by one 1 cm voxel layer**, so every sampled goal is reachable. This lets the L1 `fk_coverage` and `travel_budget` gates (with the 2× margin) pass. Measured before the change: coverage 99.42 % within 1 cm; worst goal 42.8 full-speed steps; scripted reach 99.8 %. | User answer after M3's L1 gates failed |
| 2026-10-08 | STUCK emulated in batched sim (changes D-TRIPS) | The batched sim (`BatchedServoChain` / `BatchedGoalEnv`) now emulates the servo's **stuck-action monitor** with the exact `safety/stuck.py` predicate: 90 consecutive 30 Hz ticks (3 s) of the same non-zero sanitized, rounded *commanded* delta (every component within 0.5 tick) with no joint moving; Moving = `|qvel| > 1e-3 rad/s` per joint, as `MujocoBus` reports it. STUCK truncates the episode like the other emulated trips (NAN, RANGE, BOX_PRESENT: `truncated`, never a success, bootstrapped by FastSAC) and is counted in the trip stats and in `eval.json` `trip_causes`. THERMAL and the protection bits stay un-emulated. Then retrain reach (M6 attempt 2). | User decision "Simulate stuck + retrain": the M7 CPU-servo zero-shot run (`deploy-policy --protocol zero-shot` on `servo --backend mujoco`) found a deterministic wall press ending in STUCK on 1/83 goals (seed 7, goal #32), which halts the ladder; in the batched sim that episode was an ordinary failure. |
| 2026-10-06 | TensorBoard videos | Add `moviepy` so rollout videos show inline in the dashboard | User answer |
| 2026-10-06 | Camera parity | The sim camera must equal the real D435. Intrinsics come from the device (colour 640×480, fx 617.17, fy 617.57, cx 327.95, cy 236.12; depth fx 388.33, 14.8 mm from colour; no distortion). Extrinsics come from a **camera-only calibration**: a table-plane fit (measured on 2026-10-06: height 346 mm, axis 64° from vertical) plus registration of the arm's FK shape (servo angles read read-only) against the depth point cloud. H8 hand-eye later refines it. **The current oblique mount is kept**, and the sim models its occlusion. One file, `calib/camera.yaml`, is the source for the sim camera, the sim tracker and the videos. | User answers. No camera-to-base calibration existed in the LeRobot workspace (`extrinsics.json` was never written). |
| 2026-09-30 | RL algorithm | TD-MPC2 as main, plus a SAC + HER baseline on reach | Tests whether the world model helps. Fitting in 10 GB is a hypothesis to be measured in Phase 2. TD-MPC2 is **vendored** (MIT licence, no packaging files in the repo). |
| 2026-09-30 | Simulator | MuJoCo, using the official TheRobotStudio MJCF, system-identified | Light on VRAM, good contacts. Supersedes the Isaac choice from the earlier plans for this project. |
| 2026-09-30 | Autonomy | Hardware e-stop, then tiers: attended until the 30-minute soak test passes, then unattended with a watchdog that cuts power | Resolves the conflict between minimal human time and the current consent rule. |
| 2026-10-08 | Zero-shot ladder advance rule | The zero-shot speed ladder (0.5 → 0.75 → 1.0) climbs to the next rung only if the finished rung had **zero trips and at least 80 % successes** (`ceil(0.8 n)`: 8 of 10 at the defaults); otherwise the ladder stops (exit 1, `halt_reason: rung_below_threshold`). The same rule in sim and on the arm. The full-speed rung has no advance check: its verdict is `finetune_eligible` (D-ZEROSHOT). | User decision |
| 2026-10-08 | Zero-shot resume after trip | A trip still halts the ladder. The operator restarts the servo and reruns `deploy-policy --resume runs/deploy/<ts>`: the finished episodes are kept, the run continues the same seeded goal list (pre-drawn from the seed, independent of measurements) at the next unfinished episode of the same rung, appends to the same run dir and rewrites `summary.json` over all episodes, recording each resume. A resume with a different policy, seed, protocol, rung counts or servo real/sim class is refused. | User decision |
| 2026-10-08 | Deploy reset: lift then return | The reset between deploy episodes (`ShmBackend.reset`, the same code on the real arm and on `servo --backend mujoco` / `fake`) no longer ramps straight to the start pose in joint space: it first moves the tip on a straight line **away from every box wall it is near and up** to a safe height inside the box (the start pose's tip height, x / y at least 3 cm inside the box), then returns to the start pose. All of it is RUN deltas through the servo's full safety pipeline (FK + the normal command path; nothing is bypassed), and a STUCK or other trip during the reset still halts the ladder. Reason: in M7 the STUCK trip of the seed-7 wall goal fired in the reset ramp after the episode (the box clamp held the tip at the wall while the ramp kept pushing), not in the policy episode. | User decision |
| 2026-10-08 | Trip penalty (terminal, fixed negative) (changes D-TRIPS) | A **policy-caused trip** (NAN, RANGE, BOX_PRESENT, STUCK: the trips the batched sim emulates; `env/tasks.py` `TERMINAL_TRIPS`) is now a **terminal** step (`terminated`, so FastSAC and SB3 do not bootstrap it; `truncated` stays set as before) that pays a **fixed negative reward** `tasks.<name>.trip_penalty` = **−310** (config) instead of the step reward, in the batched sim and in `So101GoalEnv` alike. The penalty is worse than any trip-free outcome: the validator requires it ≤ `episode_steps` × the worst step reward, with the distance at the workspace-box diagonal plus the BOX_PRESENT tolerance (0.607 m): 100 × −(0.607/0.30 + 1 + 0.05) = −307.3; the bound is undiscounted, so it holds for any γ, and rewards stay non-positive. The penalty is read from the stored transition's trip bits, so HER relabelling never turns a trip into a success. Other faults (THERMAL, protection bits, STALE_CMD, servo left RUN, stalled read) are not the policy's doing: no penalty, truncated only. Success is reported separately (`info["success"]` / `info["is_success"]`). Then retrain reach (M6 attempt 3, the last of the M6 budget). | User decision "Add penalty + retrain": until now a trip only truncated the episode and FastSAC bootstrapped through it, so the learner never saw a cost for tripping. |
| 2026-10-10 | Joint zeros set numerically from the 2-pose camera fit | Joint-zero errors are fixed **numerically** instead of redoing lerobot's hand-placed calibration: `pixi run zero-adjust --from-report runs/camera-calib/<ts>` shifts `Homing_Offset` (and `Min/Max_Position_Limit` by the same ticks, so the physical range is unchanged) of each arm joint whose `camera-calibrate --poses 2` offset (`diagnostics.tilt_vs_offsets`, verdict `joint_offsets`) is ≥ 0.5°, so the reading equals the true angle; the calibration JSON is updated to match (backup kept). Measured 2026-10-10 (run 20261010-084545, consistent with 20261010-083836): shoulder_lift −4.94°, elbow_flex −0.16° (left alone), wrist_flex +1.00°, base tilt 0.45°, table z −5.1 mm. Accepted after the adjust: a new `camera-calibrate --poses 2` shows every \|offset\| < 1°. The gripper is never adjusted; wrist_roll only when given explicitly. | User decision. The hand-placed lerobot middle pose missed the MJCF zero by ~3–5° (shoulder_lift), which the two-pose fit now separates from a base tilt (VIF 1.8); a numerical shift is repeatable and does not depend on placing the arm by eye again. |

## 12. [v4] AI training connector (2026-10-07)

An AI client (e.g. Claude Code) can drive training through the `so101_rl.lab` Python API and its thin MCP server.

**What the AI can do on its own:**
- create, run, monitor, pause, evaluate, gate and export **sim** training runs, fully unattended, within the GPU budgets and thermal limits;
- run **real-hardware** jobs only under the autonomy tiers:
  - **Tier 0:** a signed human approval per run.
  - **Tier 1:** after the signed promotion and the session rules, the AI may run real-arm jobs and start the servo inside a human-opened arming window.

**What never changes:** the connector never bypasses ServoCore, the hardware gates or the safety layer.

The trainer interface is generic in shape (datasets, offline training, export), so imitation, world-model and VLA trainers can be added later without changing the connector.

**Sources:** the design is `plans/ai-connector-design.md` (v2), and the decisions are the §10 rows dated 2026-10-07.

**Build order:**

| Phase | Contents | Depends on |
|---|---|---|
| LA0–LA1 | Core lab, trainers, supervisor, 12-tool MCP, dashboard | Buildable now |
| LA2 | Training pipelines | SU4 |
| LA3 | Real-arm deployment | SU5 / SR2 |
| LA4 | Real-arm jobs via the connector | LA3, SV1, FW1, PE1 and the H-track |
| LA5 | Tier 1 unattended | H13, the 120-minute soak and a clean attended session |

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

**[v3] Sim pre-training sources:**

- Squint, arXiv 2602.21203 (zero-shot SO-101 transfer)
- MuJoCo Warp: https://github.com/google-deepmind/mujoco_warp and https://pypi.org/project/mujoco-warp/
- GPUSimBench, arXiv 2607.13059
- FastSAC / SAC on massive sim: https://araffin.github.io/post/sac-massive-sim/
- SimLauncher, arXiv 2507.04452
- BAM actuator models: https://github.com/Rhoban/bam
- so101-magic (mjlab + fitted motor model): https://github.com/bigfoot1144/so101-magic

**Vault:**

- `05-Wiki/concepts/Domain-Randomization-(SO-101-MuJoCo).md`
- `Sim-to-Real-Transfer.md`
- `Actuator-System-Identification.md`

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
