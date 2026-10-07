# Sim-first pre-training track (S0–S5) — Implementation Plan

> **Version 1 (2026-10-05).** Source plan: `so101-self-learning-rl-plan.md` v3 (§1, §2A L1/L2/L3/L4/L6-revisited/L12/L13, §4, §5 "Stage A critical path" and "Sim-first track" incl. the S0 spike result, Phases 0–4, §6, §7, §10 — the 2026-10-05 rows are the sim decisions, the 2026-10-01/02 safety decisions are binding). Architecture: `docs/ARCHITECTURE.md` v0.11. Measured facts: `plans/reviews/2026-10-05-mjwarp-feasibility-spike.md`. Phase 0 interfaces: `plans/phase0-implementation-plan.md` F1–F11 (as-built annotations win where they differ; `docs/ARCHITECTURE.md` §5–§6 is authoritative over both).
>
> **Scope discipline (anti-hypertrophy):** S0–S5 of the sim-first track plus the remaining Phase 0 units U8/U9/U10 that they absorb or depend on. No pixel policy (the pixel world model stays a parallel track; only its hooks are named), no Stage B/C, no ROS2, no C++/Rust, no ManiSkill3 port unless the S0 gate fails, no new hardware beyond the Phase 0 H-track.
>
> **Hardware rule for every task in this file:** nothing here opens `/dev/ttyACM0` or any serial device except the tasks marked **HW** in the HS-track table, which run attended (Tier 0) after the named H-track prerequisites. Every SW task runs on the fake, CPU-MuJoCo or MuJoCo-Warp backends and `pixi run test` stays green without the arm.

## Overview

| Item | Value |
|---|---|
| Objective | A GPU-batched MuJoCo Warp environment that applies exactly the real servo loop's tick rules, a system-identified and randomised simulator, massively parallel SAC pre-training of `reach` and `die-to-zone` to ≥ 95 % in randomised sim, and the software to deploy, evaluate (zero-shot, with a confidence bound) and fine-tune those policies on the real arm through the existing `ServoCore`. |
| Deliverables | Version pin (`pyproject.toml`, `pixi.lock`), `src/so101_rl/sim/{warp_model,batched_servo,sim_tracker,domain_rand,tray_check}.py`, `src/so101_rl/sim/sysid/`, `src/so101_rl/env/{obs,tasks,batched_env,backend,goal_env,goals,random_policy}.py`, `src/so101_rl/rl/{fast_sac,replay_gpu,gates,train_sim_sac,eval_sim,rollout_logger,render,policy_export,curriculum,deploy_policy,sb3_bridge,tdmpc2_offline,gap_report,her,tdmpc2_env,cfg,train_sac_her,train_tdmpc2}.py`, `src/so101_rl/wm/*`, `calib/sim_params.yaml`, `scripts/bench_warp.py`, `scripts/check_milestone_phase0.py`, tests, pixi tasks, `docs/ARCHITECTURE.md` updates. |
| Domain | GPU robot simulation (MuJoCo Warp + torch), sim-to-real RL, system identification, real-time servo parity |
| Technologies | python 3.12, numpy 2.2, torch 2.11 (CUDA 12.x on `default`, CPU on `cpu`), mujoco 3.14 + mujoco-warp 3.14 + warp-lang ≥ 1.17 (conda-forge), stable-baselines3 2.9, torchrl 0.14 / tensordict 0.14, vendored TD-MPC2, h5py, CMA-ES (`cma`, PyPI) or Optuna |
| Unit of this plan | Master passed no `Unit Id` / `Owns Files`; this track is treated as its own unit set. The tree is partitioned into the sub-units in `execution_shape` (file ownership is exclusive within a wave; hand-offs across waves are annotated). |
| Preconditions (done) | H1: the driver works (610.57, CUDA 12.6 in `default`), MJWarp spike passed on this host. Wave 0–2 of Phase 0 built (U0–U7, U3). |
| Preconditions (pending, hardware-gated tasks only) | H2 power path + e-stop, H3 watchdog MCU, H5 rest pose, H7 calibration read-back, H8 hand-eye (die task), H9 tray (die task), H11 safe-stop timing, H13 Tier 1 (unattended fine-tuning). |

### Environment and code facts checked on 2026-10-05

- `pixi.lock` today: `cpu` = mujoco-python 3.13.0, pytorch 2.11.0 (cpu), numpy 2.2.6; `default` = mujoco-python **3.3.2**, pytorch **2.7.1** (cuda126), cuda-version 12.8; `robot` = pytorch 2.11 cpu, no mujoco. The two sim-capable envs disagree on MuJoCo *and* torch, which is exactly what S0 fixes.
- Spike (`plans/reviews/2026-10-05-mjwarp-feasibility-spike.md`): mujoco 3.14.0 + mujoco-warp 3.14.0 + warp-lang 1.17.0 (cuda129) solve from conda-forge in one go; the full scene (`load_model`, all 10 collision meshes) runs; **`margin=0` required on the `jaw_col` default**; **`njmax ≥ 256`** (default 64 overflows; overflow is only a `wp.printf` warning); `nconmax = 32`/world; 745k env-steps/s and 0.8 GB at N=1024, 1.09M and 1.3 GB at 2048; CPU parity 5e-7 rad (arm) over 300 steps with matching contact counts; grasp-level contact parity **not yet tested**; `wp.to_torch` is zero-copy. **lerobot 0.6.1 needs torch < 2.12 and numpy < 2.3**, so the repo pin is torch 2.11 / numpy 2.2 (the spike solved 2.12 / 2.5 — not usable here).
- MJWarp API shape (checked in the 3.5 package available locally; **re-verify names against 3.14 at implementation**): `put_model(mjm) -> Model`, `make_data(mjm, nworld=, nconmax=, njmax=, naconmax=)`, `put_data`, `get_data_into`, `reset_data(m, d, reset: wp.array | None)`, `step(m, d)`. `Data.nefc` has shape `(nworld,)`, `Data.nacon` shape `(1,)`, `Data.njmax`, `Data.naconmax`. Model fields `body_mass`, `dof_damping`, `dof_armature`, `dof_frictionloss`, `geom_friction`, `actuator_gainprm`, `actuator_biasprm` carry a leading `*` (per-world) dimension — per-world sysid / DR is supported in the engine. Overflow handling: `forward.py` prints `nefc overflow` / `broadphase overflow` / `narrowphase overflow` and continues — hence our own detector that **fails**.
- Upstream MJCF (`assets/so101/upstream/so101_new_calib.xml`): no `<option>` (defaults: Euler, dt 0.002, Newton, pyramidal cone), class `sts3215`: joint `damping 0.60, frictionloss 0.052, armature 0.028`, `<position kp="998.22" kv="2.731" forcerange="-2.94 2.94">`, per-actuator `forcerange ±3.35`, `ctrlrange` per joint (gripper `-0.17453 1.74533`). The gripper jaw joint has its own tiny damping/armature. These are the sysid starting values.
- `MujocoBus.step_ticks(1)` runs `round((1/30)/0.002) = 17` `mj_step`s = 34.0 ms of sim time per 33.3 ms servo tick (2 % slow). The parity rig and the batched env must use the **same** substep count; see decision D-TIMESTEP.
- `ServoCore.tick()` order and the `SafetyPipeline` chain are as documented in `docs/ARCHITECTURE.md` §5–§6 and `safety/pipeline.py`: `new_command` (sanitize → rint → goal += Δ → joint limits) at 10 Hz; per tick `interpolate(prev_target → goal, ticks_remaining)` → joint limits → velocity cap (20/tick) → tracking clamp |target − present| ≤ E with the **goal pulled back** → box via FK of the target (BOX = revert to `prev_target` and goal, BOX_PRESENT = trip) → thermal → stuck → `check_range`; `applied_delta_ticks = goal − goal_at_cmd_start`. Commands are applied on the first tick after receipt with `interp_left = 3`. In `SimBackend`/`Rig` (tests/test_action_units.py) each servo tick is `core.tick()` then `bus.step_ticks(1)`.
- `MujocoBus.read_block`: `position = clip(rint(sign·q/rad_per_tick + 2047), 0, 4095)` (gripper via the percent map), `velocity = rint(qd · ticks_per_rad · sign)`, `moving = |qd| > 1e-3`. `write_goal`: clip to the Min/Max_Position_Limit registers (= calibration range) → optional delay FIFO (`action_delay_ticks`) → `ctrl = ticks_to_q(goal)` clipped to `ctrlrange`.
- `HsvDepthDieTracker.detect` returns `None` on: no blob in `[min_area, max_area]`, < 50 % valid depth in the 5×5 window, depth spread > `depth_window_m` (0.02), or > 15 % of the surrounding ring nearer than the blob by more than `depth_window_m` (occlusion). The object table carries `visible` + `t_mono_ns` per object (F9).
- `LocalIpc` already exists in `servo/shm.py` (U3), so Phase 0 task 8.1's `env/local_ipc.py` is unnecessary.
- `pixi.toml` task `random-policy-sim` points at `so101_rl.env.random_policy` while Phase 0 Task 10.1 planned `scripts/run_random_policy.py`; this plan resolves it in favour of the module (SR3).
- `torch.cuda.is_available()` gates the `gpu` marker (`tests/conftest.py`); `pixi run test` excludes `gpu` and `hardware`. There is no GPU test task yet.

## Specialist Guidance Summary

**simulation-to-hardware-bridge** (S1–S3, S5):
- Re-implement the clamp chain in torch, vectorised, in int64 tick space with the same rounding rule; `ServoCore` + `SafetyPipeline` stay the single source of truth for semantics, the batched code lives in `sim/` + `env/`, never in `safety/`. Parity in two layers: (A) identical present-tick streams into both chains → bit-identical goals, targets, applied deltas and per-stage clamp flags, with adversarial sequences (saturating, sign flips, joint limits, box boundary, E wind-up recovery, delay FIFO on/off); (B) closed loop f64 CPU vs f32 Warp → exact goals while present ticks agree, report the ±1-tick divergence near rounding boundaries, trajectory ≤ 1e-4 rad for contact-free motion and ≤ 5e-4 rad with contacts (measured drift 5e-7 is far inside), over ~200 seeds × 300 steps.
- Pitfalls: the Feetech dead-band and speed/accel registers make the real servo track goals less crisply than an ideal kp/kv actuator; keep `forcerange` realistic; sag under load interacts with E; present position is quantised and one serial tick late — model both. Keep identical substep counts on CPU and Warp.
- Sysid order: delay → kp → kv *or* damping (not both) → frictionloss → armature → forcerange/dead-band → gripper separately (E = 400, own kp, contact with the die). Excite through the real clamp chain with multisines 0.2–4 Hz (below the 5 Hz Nyquist of the 10 Hz command), chirps (armature), steps of varied size (delay, dead-band), slow ramps (friction), random-policy segments; log the clamp-active fraction and keep it partial (fully clamped data is blind to servo dynamics). Hold out whole trajectories *and* whole excitation types; report CMA-ES covariance and drop unidentifiable parameters into wide DR ranges. Synthetic recovery test incl. a deliberately mismatched model.
- DR around the sysid posterior (±2–3 σ), delay 0–2 extra ticks per episode, forcerange ±10 %, die mass/friction/size ±1 mm, zone/tray ±5 mm, camera extrinsic a few mm / ~1°, per-joint tick offsets, per-episode backlash. Tracker: σ 2–6 mm (z heavier), slowly varying bias + per-frame jitter, 1–3 frames latency at 30 Hz, bursty 0–10 % drops, occlusion by projecting die and jaws into the camera and returning UNKNOWN below a visible-fraction threshold; the policy must see the validity flag plus a held last-known position with age.
- Zero-shot: ≥ 30 episodes per task, Wilson 95 % interval (30/30 → lower bound 88.6 %), fixed goal seeds interleaved sim/real, halt on any safety event or 3 consecutive failures. Gap metrics: success delta, final-error distribution, per-joint tracking-error distributions (KS/Wasserstein), clamp-active fraction per stage, replay-in-sim RMS, tracker UNKNOWN fraction, online delay estimate by cross-correlation. Freeze and version the obs/reward interface between sim and real.

**lerobot-specialist** (S4/S5 learners):
- Minimal self-written SAC (~300–400 lines) on the batched tensors rather than torchrl's SAC: the asymmetric critic, separate `terminated`/`truncated`, n-step across auto-reset and CUDA-graph capture all need custom code. Actor 3×256 (reach) / 3×512 (die), **SB3-compatible shape (ReLU, no LayerNorm) so the actor copies into SB3's `latent_pi`/`mu`/`log_std`**; twin critics 3×512 with LayerNorm on `concat(priv_obs, action)`. Batch 4096/8192, 8 gradient steps per vector step (sweep 4/8/16), GPU ring replay 2–4 M, fp32. γ 0.97 (50 steps) / 0.98 (100 steps); **bootstrap through truncation and store the pre-reset final obs** (the most common bug). n-step 3 for the sparse die reward. α auto-tuned from 0.01–0.05 with target entropy −3 (fallback −6), log-α clamped. Fixed analytic obs normalisation, ~~rewards in [−1, 1] per step, success bonus ≤ 5~~ (superseded by FS4: non-positive modular reward, every step < 0, finishing best; no success bonus), tanh-Gaussian with exact [−1, 1] bounds (scaling to ticks only in the env). Randomise the initial episode phase per env to decorrelate synchronous resets.
- Throughput: 745k mj-steps/s ÷ 51 ≈ 14k policy-steps/s raw; expect 5–10k env-steps/s with learning; reach 2–5 M steps (10–20 min), die 20–50 M (1–3 h) — to be measured in a spike. Bottlenecks: python/kernel-launch overhead in the update (torch.compile reduce-overhead or manual CUDA graphs), Warp graph coexisting with torch streams.
- Gates: L1 = FK over ≥ 1e5 joint samples covers the goal distribution with margin, a scripted IK/Jacobian controller (quantised ticks, obs noise on) achieves the predicate in ≥ 95 % within the horizon, travel budget ≥ 2× needed; L2 = adversarial-state table (hover over zone, die moving, die held in the air, die slid out after success → hold K steps, wrong height, tracker-invisible), random/do-nothing false-accept rates (< 0.1 % / 0), a second independent predicate implementation, post-training gaming audit on rendered top/shortest episodes.
- TD-MPC2 offline: TensorDict chunks `[episodes, T+1]` with obs/action/reward/terminated/truncated/episode; 1–5 M transitions mixing ~30 % random/noisy, 40 % mid-training, 30 % near-final; custom single-task loop using `TDMPC2.update()` (the vendored `OfflineTrainer` asserts mt30/mt80), goal in obs, no task embedding; warm-start the real run with the checkpoint, small `seed_steps`, 10–20 % sim data kept in the buffer, lower lr.
- SB3 fine-tuning: copy the actor only; re-initialise critics and warm them up with the actor frozen for the first 2–5k steps while the sim actor fills the buffer; HER `n_sampled_goal=4, "future"`, `ent_coef="auto_0.01"`, lr 1e-4, keep the frozen sim actor as the eval baseline with a checkpoint-and-rollback rule.
- Review triggers: UTD < 0.005 grads/transition, α < 1e-4 or entropy > 2 below target for 100k steps, |Q| > 2 × r_max/(1−γ), action saturation > 50 %, Q-loss monotone growth; compare against the scripted upper bound and random lower bound every 1 M steps.

## How the sim track supersedes or absorbs Phase 0 U8 / U9 / U10

| Phase 0 item | Status under this plan | Where |
|---|---|---|
| U8 Task 8.1 `SimBackend` (in-process ServoCore + MujocoBus) | **Kept, demoted.** It is the parity reference rig, the CPU sanity env for SB3 smoke tests and the end-to-end check of exported policies through the real servo path (`servo --backend mujoco` + `ShmBackend`). It is **not** the training env. | SR1 (`env/backend.py`) |
| U8 Task 8.1 `ShmBackend` | **Kept unchanged** — the real-arm backend used by S5. | SR1 |
| U8 Task 8.1 `env/local_ipc.py` | **Dropped** — `servo.shm.LocalIpc` exists since U3. | — |
| U8 Task 8.2 `So101ReachEnv` with the F7 15-D observation | **Superseded by `So101GoalEnv(backend, cfg, task: TaskSpec)` with the FS3 observation layout** (adds the last applied action, the die block for the die task). F7's action rule (`delta_ticks = rint(action × max_delta_ticks)`, the only scaling in the code base) is unchanged. `compute_reward` is delegated to `env/tasks.py` so sim and real use one definition. | SU0 (interfaces), SR1 (env) |
| U8 goal sampler `env/goals.py` | **Kept**, now parametrised by `TaskSpec.goal_bounds`; the same bounds drive the batched env's torch sampler (cross-checked by a test). | SU0 / SR1 |
| U9 Task 9.1 HER relabel + TD-MPC2 goal-env adapter | **Kept** (needed for real fine-tuning). Adapter obs = FS3 policy input (goal appended). | SR2 |
| U9 Task 9.2 `train_sac_her.py`, `train_tdmpc2.py` | **Absorbed into S5:** both gain `--init-from` (sim actor / offline TD-MPC2 checkpoint), the actor-frozen critic warm-up and the baseline eval. The Phase 3 "≥ 90 % on MuJoCo before any real episode" gate is **replaced** by the S4 exit gate (≥ 95 % over 1000 DR episodes in the batched env) **plus** the exported policy passing the CPU `servo --backend mujoco` end-to-end eval. | SU5 |
| U9 Task 9.3 state world model + dataset | **Kept**; first training data = S4 sim rollouts (plan §5 critical path item 4), real data later. `EpisodeDataset` gains a TensorDict-chunk reader. | SR2 |
| U10 Task 10.1 random-policy script | **Kept** as `src/so101_rl/env/random_policy.py` (matches the existing pixi tasks). | SR3 |
| U10 Task 10.2 `check_milestone_phase0.py` | **Kept**, with three new rows: MuJoCo version pinned across envs, parity test green on GPU, `calib/sim_params.yaml` present with held-out RMS ≤ 5 %. | SR3 |
| H12 sysid check "recorded, not fixed" | **Superseded**: S2 fits the parameters (HS1–HS3) and `compare-real --sim-params calib/sim_params.yaml` is the gate. | SU2, HS3 |
| Phase 3 order (plan §5) | Sim SAC → zero-shot → SAC+HER → TD-MPC2 offline-pretrained; implemented as S4 → HS4 → HS5. | SU4, SU5, HS |

## Frozen Interfaces (FS-series)

A unit may add behaviour but may not change a listed signature, dtype, layout version, enum value or ordering without updating this section **and** `docs/ARCHITECTURE.md` §5/§7 in the same change. F1–F11 of the Phase 0 plan remain in force except where an FS entry says it supersedes them.

### FS1. Sim configuration — `config/schema.py`, `config/loader.py`, `config/so101.yaml` (extends F1)

```python
@dataclass(frozen=True)
class SimWarpConfig:
    njmax: int = 256                 # per world; spike: default 64 overflows (peak nefc 188 measured at S0.3)
    nconmax_per_world: int = 32      # spike: CPU max ncon 18
    nworld_default: int = 1024
    overflow_check_every_steps: int = 10   # policy steps between device-side counter reads (a sync each)
    overflow_policy: Literal["fail"] = "fail"   # no "warn" value exists on purpose

class So101Config:                   # F1 + one field
    ...
    sim_warp: SimWarpConfig
```

`so101.yaml` gains a `sim_warp:` section with exactly those keys (the loader is strict: unknown or missing keys are errors; `njmax ≥ 64`, `nconmax_per_world ≥ 1`, `nworld_default ≥ 1` are range-checked). `tests/test_config_single_source.py` is extended with `njmax`/`nconmax_per_world` literals (no hard-coding outside `config/`).

### FS2. MuJoCo Warp model wrapper — `src/so101_rl/sim/warp_model.py`

```python
class ContactBufferOverflow(RuntimeError): ...          # njmax / nconmax / naconmax exceeded in any world

@dataclass(frozen=True)
class WarpSizes: nworld: int; njmax: int; nconmax: int; naconmax: int

class WarpSim:
    """One MJWarp Model + Data for N worlds of the RL scene, torch views zero-copy on cuda."""
    def __init__(self, model: mujoco.MjModel, cfg: SimWarpConfig, nworld: int, *, device: str = "cuda:0", capture_graph: bool = True) -> None
    model: mujoco.MjModel; m: "mujoco_warp.Model"; d: "mujoco_warp.Data"; sizes: WarpSizes; device: torch.device
    @property
    def qpos(self) -> torch.Tensor      # [N, nq] float32, zero-copy view (wp.to_torch)
    @property
    def qvel(self) -> torch.Tensor      # [N, nv]
    @property
    def ctrl(self) -> torch.Tensor      # [N, nu]
    @property
    def actuator_force(self) -> torch.Tensor   # [N, nu]
    def step(self, nsteps: int) -> None                       # nsteps mj_steps for every world (captured CUDA graph per nsteps value)
    def check_overflow(self) -> None                          # reads d.nefc[:] and d.nacon; raises ContactBufferOverflow; also called by close()
    def reset_worlds(self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> None   # mask bool[N]; mj_forward equivalent for those worlds
    def site_xpos(self, site: str) -> torch.Tensor            # [N, 3] world frame (base frame == world frame in the scene)
    def body_xpos(self, body: str) -> torch.Tensor            # [N, 3]
    def body_xquat(self, body: str) -> torch.Tensor           # [N, 4] wxyz
    def contact_count(self, geoms_a: Sequence[int], geoms_b: Sequence[int]) -> torch.Tensor   # int32[N]: contacts between the two geom sets this step
    def set_world_params(self, params: "BatchedSimParams", *, validate: bool = True) -> None   # per-world dof_damping/armature/frictionloss, actuator_gainprm/biasprm, geom_friction, body_mass (FS7); validate rejects non-finite, kp/forcerange/die_mass <= 0, others < 0 before writing (host inputs: no device sync; GPU inputs: one sync); in-place writes reach already-captured graphs
    def copy_world_to(self, world: int, data: mujoco.MjData) -> None # for CPU rendering / debugging
    def close(self) -> None
    @property
    def vram_bytes(self) -> int
```

`so101_rl.sim` keeps importing lazily: `import so101_rl.sim` must not import `mujoco_warp` or `warp`. As built (SU0 review, 2026-10-06): `stat.meaninertia` is made per-world by the wrapper (outside `batch_sizes`); `with WarpSim(...)` never masks an in-flight exception (the final overflow check becomes a note); runs are not bitwise deterministic (shared atomic contact pool).

### FS3. Policy observation layout — `src/so101_rl/env/obs.py` (supersedes F7's `observation` layout)

```python
OBS_LAYOUT_VERSION: Final[int] = 1

@dataclass(frozen=True)
class PolicyObsLayout:
    q_norm: slice            # 6: (present_ticks - CENTER_TICK) / 2048, all six joints (gripper included, same formula)
    dq_norm: slice           # 6: velocity_ticks_per_tick / max_ticks_per_tick, clipped to [-2, 2]
    last_applied: slice      # 6: applied_delta_ticks / max_delta_ticks (post-clamp, L4), [-1, 1]
    tip_xyz: slice           # 3: So101FK tip from PRESENT ticks, base frame, metres
    die_xyz: slice | None    # 3: last-known die centroid, base frame (zeros until first seen)   [die tasks]
    die_visible: slice | None# 1: 1.0 if the tracker saw the die this policy step, else 0.0       [die tasks]
    die_age: slice | None    # 1: min(frames_since_seen, 30) / 30                                [die tasks]
    dim: int                 # 21 (reach) or 26 (die)

REACH_OBS: Final[PolicyObsLayout]   # dim 21
DIE_OBS: Final[PolicyObsLayout]     # dim 26
GOAL_DIM: Final[int] = 3
def policy_input_dim(layout: PolicyObsLayout) -> int        # layout.dim + GOAL_DIM (desired_goal appended; TD-MPC2 "state")

def build_policy_obs_np(layout, *, present_ticks: np.ndarray, velocity: np.ndarray, applied_delta_ticks: np.ndarray, tip_xyz: np.ndarray,
                        cfg: So101Config, die: DieObs | None, out: np.ndarray) -> np.ndarray
def build_policy_obs_torch(layout, *, present_ticks: torch.Tensor, velocity: torch.Tensor, applied_delta_ticks: torch.Tensor, tip_xyz: torch.Tensor,
                           cfg: So101Config, die: DieObsT | None, out: torch.Tensor) -> torch.Tensor   # same semantics, [N, dim]

@dataclass(frozen=True)
class PrivObsLayout:         # critic only, sim only
    q_true: slice            # 6  joint angles (rad), unquantised
    dq_true: slice           # 6  rad/s
    tip_xyz_true: slice      # 3
    die_pos: slice           # 3  true die centre
    die_quat: slice          # 4  wxyz
    die_vel: slice           # 3  linear velocity
    die_in_gripper: slice    # 1  jaw_col geoms touching the die this step
    die_on_table: slice      # 1
    dr_params: slice         # 33: per joint (kp, kv, damping, frictionloss, armature) × 6 normalised by the centre, delay_ticks/5, die_mass/centre, die_friction/centre
    tracker_delay: slice     # 1  frames / 3
    dim: int                 # 61
PRIV_OBS: Final[PrivObsLayout]
```

Gymnasium `Dict` contract (F7) is kept: `observation` = this layout, `achieved_goal` = tip xyz (reach) or last-known die xyz (die), `desired_goal` = goal xyz (reach) or zone centre (die). A unit test feeds one state to the numpy and the torch builders and asserts equality to 1e-6.

### FS4. Task specification — `src/so101_rl/env/tasks.py`, `src/so101_rl/env/reward.py`, `tasks:` in `so101.yaml`

Revised 2026-10-06 (SU0 review; plan §10 rows "Reward structure", "Die success height", "Die must be settled").

```python
# config/schema.py (parsed strictly by config.loader.parse_tasks; So101Config.tasks; load_tasks_config() reads only this section)
@dataclass(frozen=True)
class RewardTermSpec: name: str; weight: float; scale_m: float | None = None; zero_on_success: bool = False
@dataclass(frozen=True)
class RewardSpec: terms: tuple[RewardTermSpec, ...]          # def term(name) -> RewardTermSpec | None
@dataclass(frozen=True)
class TaskConfig: reward: RewardSpec; max_height_m: float | None = None; settle_mm: float | None = None
@dataclass(frozen=True)
class TasksConfig: reach: TaskConfig; die_to_zone: TaskConfig
```

```yaml
tasks:                     # strict: exactly these keys; term names from REWARD_TERM_PARAMS; weight in [0, 100],
  reach:                   # scale_m in [0.001, 10], max_height_m in [0.001, 0.020], settle_mm in [0.1, 10]
    reward:
      distance: {weight: 1.0, scale_m: 0.30, zero_on_success: true}
      success: {weight: 1.0}
  die_to_zone:
    max_height_m: 0.016    # centroid at most this far above the goal
    settle_mm: 3.0         # tracker displacement between the 2 debounce steps
    reward:
      distance: {weight: 1.0, scale_m: 0.30, zero_on_success: true}
      success: {weight: 1.0}
```

```python
# env/reward.py — terms (unweighted, every one <= 0): distance = -dist/scale_m (0 on success steps if zero_on_success),
# success = -(1 - predicate), clamp = -info["action_clamped"], action_l2 = -mean(info["applied_action"]**2)
REWARD_TERMS: Mapping[str, RewardTerm]               # name -> (inf, sup <= 0, numpy fn, torch fn)
class RewardSpecError(ValueError): ...
def validate_reward_spec(spec: RewardSpec) -> RewardBounds   # max_step = 0, max_unsuccessful_step = -w_success
def evaluate_np(spec, *, dist, success, info=None) -> np.ndarray          # float32; terms_np(...) -> dict per term
def evaluate_torch(spec, *, dist, success, info=None) -> torch.Tensor     # terms_torch(...)

# env/tasks.py
@dataclass(frozen=True)
class GoalBounds:            # base frame, metres
    radius_max_m: float = 0.30          # from the base
    box: WorkspaceBox                   # cfg.box, shrunk by margin_m on every face
    margin_m: float = 0.02
    z_min_m: float = 0.03               # kinematic floor; H-track may raise it

@dataclass(frozen=True, kw_only=True)
class ZoneSpec: center_xy_m: tuple[float, float]; half_edge_m: float = 0.030; z_table_m: float = -0.02; max_height_m: float  # 0.016 (config)

@dataclass(frozen=True)
class TaskSpec:
    name: Literal["reach", "die_to_zone"]
    obs: PolicyObsLayout
    episode_steps: int                  # 50 / 100
    gamma: float                        # 0.97 / 0.98 (training hint, not an env property)
    success_radius_m: float             # reach 0.015; die: centroid inside the zone square
    success_debounce_steps: int         # reach 1; die 2 (consecutive steps with the predicate true)
    settle_m: float | None              # die 0.003 (config settle_mm / 1000): max tracker displacement between counted steps; reach None
    reward: RewardSpec                  # config tasks.<name>.reward; __post_init__ runs validate_reward_spec
    goal_bounds: GoalBounds
    zone: ZoneSpec | None
    die_start: DieStartSpec | None      # well centre xy, sigma_xy_m, yaw uniform; measured by H9, placeholder until then
    start_pose: Literal["home", "random_in_box"]
    reset_settle_steps: int = 0         # die task: physics steps after placing the die (0 = place at rest)

def task_from_config(name: str, tasks: TasksConfig) -> TaskSpec
REACH: Final[TaskSpec]; DIE_TO_ZONE: Final[TaskSpec]   # built from the shipped so101.yaml at import
def reward_np(achieved: np.ndarray, desired: np.ndarray, spec: TaskSpec, *, visible: np.ndarray | None = None, info=None) -> np.ndarray     # vectorised (SB3 HER), <= 0
def reward_torch(achieved: torch.Tensor, desired: torch.Tensor, spec: TaskSpec, *, visible: torch.Tensor | None = None, info: Mapping[str, Tensor] | None = None) -> torch.Tensor
def reward_terms_np(...) -> dict[str, np.ndarray]; def reward_terms_torch(...) -> dict[str, torch.Tensor]   # per-term, for logging (FS11)
def success_np(achieved, desired, spec, *, visible=None) -> np.ndarray      # bool; visible=False never succeeds (UNKNOWN, L11)
def success_torch(achieved, desired, spec, *, visible=None) -> torch.Tensor
def debounce_np(predicate, count, spec, *, achieved=None, prev_achieved=None) -> tuple[np.ndarray, np.ndarray]   # (success, new_count); die needs the positions
def debounce_torch(predicate, count, spec, *, achieved=None, prev_achieved=None) -> tuple[torch.Tensor, torch.Tensor]
```

**Reward.** A sum of named, weighted terms, each a pure instantaneous function of one step's `(achieved, desired, visible, info)` — the instantaneous success predicate, never the debounce/settle state — so HER relabels it. Default (both tasks): `-dist/0.30 * (1 - pred) - 1 * (1 - pred) - 0.05`: −0.05 on success steps, ≤ −1.05 on every other step. The `time` term (constant −1 × weight 0.05, the only term not zeroed on success steps, instantaneous so HER-safe) makes **every** non-terminal step strictly negative, in-zone steps included. `validate_reward_spec` proves for any configured combination that **every step earns ≤ −w_time < 0** and unsuccessful steps ≤ −(w_success + w_time) (known terms once each, every term ≤ 0, weights finite ≥ 0, `success` and `time` weight > 0, `distance.scale_m` > 0); the old `+bonus × predicate` paid the policy not to finish (dip in and out of the zone, hide the die). **Success predicate:** reach `|tip − goal| ≤ 0.015`; die: visible AND xy in the zone square around `desired` AND centroid ≤ `max_height_m` (16 mm) above `desired` (D435 depth noise; a die held ~2 cm up fails). **Termination:** `success_debounce_steps` consecutive predicate steps; die also needs ≤ `settle_m` (3 mm) tracker displacement between them (a slide at 0.05 m/s = 5 mm/step or a nudge out never terminates). A die kept moving inside the zone forever earns −w_time per step, so it is strictly worse than finishing (the terminal state is 0): finishing is always best.

`reward_*` use the **tracker-reported** die position (last-known when invisible), so sim and real rewards are the same function of the same inputs; the privileged truth is critic-only. `tests/test_tasks.py::test_np_torch_reward_identical` pins the two implementations; `tests/test_reward.py` pins the validator (incl. a hypothesis property test over random specs and states) and that finishing beats every gaming policy in discounted return.

### FS5. Batched servo chain — `src/so101_rl/sim/batched_servo.py`

```python
class BatchedServoChain:
    """torch re-implementation of SafetyPipeline.new_command/apply for N worlds, int64 tick math, same rounding (rint)."""
    def __init__(self, cfg: So101Config, calibration: dict[str, MotorCalibration], chain: KinematicChain, nworld: int, *, device: torch.device) -> None
    goal: torch.Tensor; prev_target: torch.Tensor          # int64[N,6] (same roles as SafetyPipeline.goal / prev_target)
    def resync(self, present_ticks: torch.Tensor, mask: torch.Tensor | None = None) -> None
    def new_command(self, delta_ticks: torch.Tensor) -> torch.Tensor      # float[N,6] -> nan_mask bool[N]; NaN rows: delta zeroed, TripCause.NAN flagged (sanitize_delta semantics)
    def apply(self, present_ticks: torch.Tensor, ticks_remaining: int, out: ChainOut) -> ChainOut
    def fk_tip(self, ticks: torch.Tensor, out: torch.Tensor) -> torch.Tensor   # batched So101FK (5 Rz) from the same KinematicChain yaml
    clamp_active_fraction: dict[str, float]                 # running stats per stage: joint_limit, velocity, tracking, box

@dataclass
class ChainOut:
    target_ticks: torch.Tensor        # int64[N,6]  what is written this tick
    goal_ticks: torch.Tensor          # int64[N,6]
    applied_delta_ticks: torch.Tensor # float32[N,6] goal - goal_at_cmd_start (the LOGGED action)
    clamped_mask: torch.Tensor        # bool[N,6]
    trips: torch.Tensor               # int64[N] bitmask of TripCause (NAN, JOINT_LIMIT, TRACKING, BOX, BOX_PRESENT, RANGE)
```

Stage order and semantics are those of `safety/pipeline.py` (interpolate → joint limits → velocity → tracking with goal pull-back → box via FK of target: present outside ⇒ BOX_PRESENT trip and target = prev_target; target outside ⇒ revert target and goal to prev_target → range check). Thermal, protection bits and stuck are **not** emulated (decision D-TRIPS). `sim/` may import `safety.types.TripCause` and `safety.fk.KinematicChain` (constants only; recorded in ARCHITECTURE §4 dependency arrows).

### FS6. Simulated die tracker — `src/so101_rl/sim/sim_tracker.py`

```python
@dataclass(frozen=True)
class CameraModel:            # the D435 as mounted; defaults from calib/hand_eye.yaml when present, else the scene's nominal pose
    T_base_cam: np.ndarray    # 4x4
    fx: float; fy: float; cx: float; cy: float; width: int = 640; height: int = 480
    frame_hz: float = 30.0

@dataclass(frozen=True)
class TrackerNoise:
    sigma_xy_m: float = 0.0015; sigma_z_m: float = 0.003
    bias_max_m: float = 0.003            # per-episode constant offset (hand-eye residual <= 3 mm)
    rot_bias_max_rad: float = 0.0175     # per-episode small rigid error of the camera pose (~1 deg)
    latency_frames: tuple[int, int] = (1, 3)   # per-episode integer draw
    dropout_p: float = 0.02; dropout_burst_frames: tuple[int, int] = (1, 5)
    occlusion_radius_m: float = 0.020    # tip-to-die horizontal distance below which the gripper covers the die when tip is above it
    occlusion_fraction: float = 0.5      # visible-fraction threshold of the projected die disc not covered by the projected jaw boxes

@dataclass
class TrackerOut:
    xyz: torch.Tensor          # float32[N,3] last-known base-frame position (held while invisible)
    visible: torch.Tensor      # bool[N]  seen in THIS policy step's latest frame
    age_frames: torch.Tensor   # int64[N] frames since last seen (0 when visible)
    confidence: torch.Tensor   # float32[N]

class SimDieTracker:
    def __init__(self, cam: CameraModel, noise: TrackerNoise, nworld: int, *, device: torch.device, seed: int) -> None
    def reset(self, mask: torch.Tensor, die_xyz_true: torch.Tensor) -> None           # redraw per-episode bias, rotation, latency, start the FIFO at "never seen"
    def update(self, die_xyz_true: torch.Tensor, tip_xyz: torch.Tensor, jaw_centers: torch.Tensor, jaw_half: torch.Tensor,
               frames: int, out: TrackerOut) -> TrackerOut                             # advance `frames` camera frames (3 per policy step), apply occlusion/FOV/dropout -> None-semantics
    def occluded(self, die_xyz_true, tip_xyz, jaw_centers, jaw_half) -> torch.Tensor  # bool[N]; the rule the real tracker's ring test approximates
```

Semantics pinned to `HsvDepthDieTracker`: invisible ⇒ the policy sees `visible = 0`, the **held** last position and a growing age; `success_*` never fires while invisible (UNKNOWN is never SUCCESS); the die outside the camera FOV or off the table region is invisible; a per-frame dropout models depth holes / missed blobs.

### FS7. Sim parameters and sysid — `src/so101_rl/sim/sysid/`

```python
# params.py
SIM_PARAMS_VERSION: Final[int] = 1
@dataclass(frozen=True, eq=False)
class SimParams:                           # float64 arrays of length 6 (joint order JOINT_NAMES) unless noted
    kp: np.ndarray; kv: np.ndarray; damping: np.ndarray; frictionloss: np.ndarray; armature: np.ndarray
    forcerange: np.ndarray                 # [6] symmetric |F| max (fixed from the MJCF unless `fit_forcerange`)
    delay_ticks: int                       # whole servo ticks of command-to-motion delay (one value for the bus)
    deadband_ticks: np.ndarray             # [6] int, 0 = none (target hysteresis model, decision D-BACKLASH)
    die_mass_kg: float = 0.004; die_friction: float = 1.0; table_friction: float = 0.8
    @classmethod
    def from_mjcf(cls, model: mujoco.MjModel) -> SimParams            # the upstream sts3215 values (the version-0 prior)
    @classmethod
    def from_yaml(cls, path: Path = Path("calib/sim_params.yaml")) -> SimParams   # refuses version != SIM_PARAMS_VERSION or heldout_rms_pct > 5 unless allow_unverified=True
    def to_yaml(self, path: Path, *, meta: SimParamsMeta) -> None
    def apply_cpu(self, model: mujoco.MjModel) -> None                # sets actuator gainprm/biasprm (kp, kv), dof_* and die/table geom params on the CPU model
    def as_vector(self, names: Sequence[str]) -> np.ndarray; @classmethod def from_vector(...)

@dataclass(frozen=True)
class SimParamsMeta: version: int; date: str; git_sha: str; mjcf_sha256: str; excitation_run: str; fit_rms_pct: dict[str, float]; heldout_rms_pct: dict[str, float]; spread: dict[str, float]; backend: str

class BatchedSimParams:                    # per-world tensors for WarpSim.set_world_params / DR
    def __init__(self, nworld: int, device: torch.device) -> None
    kp, kv, damping, frictionloss, armature: torch.Tensor      # float32[N,6]
    forcerange: torch.Tensor; die_mass: torch.Tensor; die_friction: torch.Tensor; table_friction: torch.Tensor; delay_ticks: torch.Tensor (int64[N]); deadband_ticks: torch.Tensor (int64[N,6])
    @classmethod
    def broadcast(cls, p: SimParams, nworld: int, device) -> BatchedSimParams
    def to_gainprm_biasprm(self) -> tuple[torch.Tensor, torch.Tensor]  # MuJoCo position actuator: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv

# excitation.py
@dataclass(frozen=True)
class ExcitationPlan: segments: tuple[Segment, ...]; policy_hz: float; seed: int    # Segment(kind: Literal["multisine","chirp","step","ramp","random","gripper"], joints, duration_s, amplitude_ticks, params)
def default_plan(cfg: So101Config, *, with_gripper_die: bool) -> ExcitationPlan     # ~10 min arm time; amplitudes keep the clamp-active fraction within [0.05, 0.4]
def generate_deltas(plan: ExcitationPlan, cfg: So101Config) -> np.ndarray          # float32[T,6] 10 Hz deltas in ticks (pre-clamp); fed to ServoCore through the normal command path
def check_plan_in_box(plan, cfg, calibration, chain, start_ticks) -> PlanCheck        # kinematic preview of the clamped path: never leaves the box, never crosses the table

# dataset.py
class ExcitationDataset:                    # 30 Hz tick-log records (TICK_LOG_DTYPE present/target per tick) + the F8 episodes of one run
    @classmethod
    def from_run(cls, run_dir: Path) -> ExcitationDataset
    segments: list[TickSegment]            # TickSegment(target_ticks int32[T,6], present_ticks int32[T,6], kind: str, t0_ns: int)
    def split(self, *, heldout_kinds: set[str], heldout_fraction_by_time: float = 0.2, seed: int) -> tuple[ExcitationDataset, ExcitationDataset]

# replay.py
def replay_segment_cpu(seg: TickSegment, model, data, calibration, cfg, params: SimParams, *, shooting_window_ticks: int = 60) -> np.ndarray   # sim present ticks int32[T,6]; re-initialised from the real state every window ("multiple shooting")
def replay_segments_warp(segs: Sequence[TickSegment], ws: WarpSim, candidates: BatchedSimParams, ...) -> torch.Tensor   # [n_candidates, T, 6] one candidate parameter set per world
def rms_pct(sim_present: np.ndarray, real_present: np.ndarray, calibration) -> np.ndarray   # per joint, % of calibrated range (same definition as compare_real)

# fit.py
@dataclass
class FitResult: params: SimParams; train_rms_pct: dict[str, float]; heldout_rms_pct: dict[str, float]; spread: dict[str, float]; evals: int; wall_s: float; backend: str; history: list[tuple[int, float]]
def fit_sim_params(ds_train: ExcitationDataset, ds_heldout: ExcitationDataset, *, prior: SimParams, fit: Sequence[str] = ("kp","kv","frictionloss","armature","delay_ticks"), fixed: Sequence[str] = ("damping","forcerange"),
                   backend: Literal["warp","cpu"] = "warp", popsize: int = 256, generations: int = 60, seed: int = 0, log_dir: Path | None = None) -> FitResult
#   delay_ticks: outer integer grid 0..6; the continuous parameters: CMA-ES (`cma`) in log-space with the prior as the mean; spread = final sigma per parameter / |mean|
#   cpu backend: multiprocessing pool over 16 cores (fallback when per-world MJWarp fields fail); identical objective
```

`compare_real.py` (U4) gains `--sim-params PATH` and `--tick-log` (30 Hz replay); the gate value `MAX_RMS_PCT = 5.0` is unchanged. `calib/sim_params.yaml` schema: `{version, meta: SimParamsMeta, params: {kp: [...], ...}}`; committed like `calib/hand_eye.yaml`; version 0 (= MJCF prior, `heldout_rms_pct: null`, `allow_unverified`) ships with SU2 so the batched env always has a centre.

> **As built / changed 2026-10-06 (integration, plan §10 "Sysid gate" and "Servo parameter identification"; the text above is kept as written):** `MAX_RMS_PCT = 1.0`. `SIM_PARAMS_VERSION = 2`: the yaml adds `servo_group: {prior: {kp, kv, armature}, scale: [6]}` (checked to reproduce `params`), `dr: {servo_group_rel: 0.40, frictionloss_range: [0, 0.21], frictionloss_source}` and `meta.unidentified`; new `ServoGroup`, `DrRanges`, `SimParamsFile`, `read_sim_params_file` (`read_sim_params_yaml` still returns `(params, meta)`). `fit_sim_params` defaults: `fit=("servo_group", "delay_ticks")` (one per-joint scale on kp, kv, armature jointly), `fixed=("damping", "forcerange", "frictionloss")`; frictionloss is always reported unidentified; `FitResult.servo_group` / `.group_scale`. 1-D blocks use a deterministic shrinking grid instead of CMA-ES (cma does not support 1-D). FS8 (SU3) must draw the servo DR from `dr` (group ±40 % around the fit, frictionloss over its range) instead of `servo_rel` on separate fields.

### FS8. Domain randomisation and curriculum — `src/so101_rl/sim/domain_rand.py`, `src/so101_rl/sim/dr_stages.yaml`, `src/so101_rl/rl/curriculum.py`

```python
@dataclass(frozen=True)
class DrStage:                 # one row of dr_stages.yaml (stage 0 = nominal, no randomisation)
    index: int; name: str
    tracker: TrackerNoise | None            # stage 1: tracker pose/noise/latency/occlusion (plan S3 priority 1)
    servo_rel: float                        # stage 2: ±fraction around the sysid centre for kp, kv, frictionloss, armature (0.25); damping if fitted
    delay_ticks: tuple[int, int]            # stage 2: per-episode draw added to the fitted delay (2..5 total per plan)
    backlash_ticks: tuple[int, int]         # stage 3: deadband_ticks per joint (0..6)
    proprio_noise_ticks: int                # stage 3: uniform integer noise on present ticks seen by the POLICY (±2)
    die_mass_rel: float; die_size_mm: float; die_friction_range: tuple[float, float]; gripper_speed_rel: float   # stage 4
    lighting: bool = False                  # stage 5: hook only (pixel track); no effect on the state policy

class DrSpec:
    stages: tuple[DrStage, ...]
    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_DR_YAML) -> DrSpec

@dataclass
class EpisodeDr: params: BatchedSimParams; proprio_noise_ticks: torch.Tensor; tracker_bias: torch.Tensor; tracker_rot: torch.Tensor; tracker_latency: torch.Tensor; die_half_edge: torch.Tensor; zone_offset_xy: torch.Tensor

class DomainRandomizer:
    def __init__(self, spec: DrSpec, centre: SimParams, nworld: int, *, stage: int = 0, seed: int, device: torch.device) -> None
    stage: int; max_stage: int
    def set_stage(self, k: int) -> None
    def sample_episode(self, mask: torch.Tensor, out: EpisodeDr) -> EpisodeDr   # per-episode (physics) draws for the worlds in `mask`
    def frame_noise(self) -> TrackerNoise                                         # per-frame draws are the tracker's job; this returns the active stage's noise spec
    def describe(self) -> dict                                                    # for checkpoints and logs

class CurriculumController:
    def __init__(self, randomizer: DomainRandomizer, *, threshold: float = 0.70, window_episodes: int = 1000, min_episodes_per_stage: int = 2000) -> None
    def observe(self, success: torch.Tensor, done: torch.Tensor) -> bool          # True when it advanced a stage this call
    history: list[tuple[int, int, float]]                                          # (env_step, stage, success_rate)
```

Physics parameters are drawn **per episode per world**; tracker noise is drawn **per frame**; the pose bias per episode (decision D-DR-FREQ). Widening only after ≥ 70 % success over the window at the current stage (plan §5 S3).

### FS9. Batched goal environment — `src/so101_rl/env/batched_env.py`

```python
@dataclass
class BatchedStep:
    obs: torch.Tensor; priv: torch.Tensor; achieved: torch.Tensor; desired: torch.Tensor      # [N, ...]
    reward: torch.Tensor; terminated: torch.Tensor; truncated: torch.Tensor                 # [N] float / bool / bool
    final_obs: torch.Tensor; final_priv: torch.Tensor; final_achieved: torch.Tensor          # the pre-reset observation of the worlds that reset this step (others: copies of obs)
    info: dict[str, torch.Tensor]   # "applied_delta_ticks" f32[N,6], "clamped_mask" bool[N,6], "trips" i64[N], "success" bool[N], "tracker_visible" bool[N], "episode_step" i64[N], "dr_stage" i64[N]

class BatchedGoalEnv:
    def __init__(self, cfg: So101Config, calibration: dict[str, MotorCalibration], task: TaskSpec, nworld: int, *,
                 sim_params: SimParams, randomizer: DomainRandomizer | None = None, tracker_noise: TrackerNoise | None = None,
                 model: mujoco.MjModel | None = None, seed: int = 0, device: str = "cuda:0") -> None
    nworld: int; task: TaskSpec; obs_dim: int; priv_dim: int; act_dim: int = 6; device: torch.device
    ws: WarpSim; chain: BatchedServoChain; tracker: SimDieTracker | None
    def reset(self, mask: torch.Tensor | None = None, *, phase_randomise: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]   # obs, priv, achieved, desired
    def step(self, action: torch.Tensor) -> BatchedStep       # action float32[N,6] in [-1,1]; delta_ticks = rint(action * max_delta_ticks) (F7 rule); 3 servo ticks: chain.apply -> ctrl -> ws.step(substeps); auto-reset on done
    def render_worlds(self, worlds: Sequence[int], *, width: int = 320, height: int = 240, camera: str = "overhead") -> np.ndarray   # uint8[k,H,W,3] via mujoco.Renderer on a CPU MjData copy (pixel-track hook; not used by the policy)
    def close(self) -> None
    @property
    def stats(self) -> dict[str, float]  # clamp-active fractions, tracker visible fraction, trips per 1000 steps, resets, overflow checks
```

Episode semantics (F7): `terminated` = success (after debounce), `truncated` = step limit **or** any emulated trip (NAN, RANGE, BOX_PRESENT); `final_*` hold the pre-reset state so a learner bootstraps correctly through truncation. `phase_randomise` starts each world at a random `episode_step` offset on the first reset (decorrelated auto-resets).

> **As built 2026-10-06 (SU1 + integration):** observation timing = the real arm's: obs / priv / achieved / reward use the 3rd tick's state record, whose `present` was read after two ticks of the new command (Layer B pins it). Reach goals come from the real env's `env/goals.py` `BoxGoalSampler` (`sample_torch`; the same FK-reachability table object as `So101GoalEnv`), not an env-local voxel grid. The scene gained the start-well tray (die-only collisions) and the `overhead` camera; die/jaw contacts use `solref` 0.004. `randomizer` still raises (SU3 hand-off).

### FS10. Learner, replay, export, rollout log — `src/so101_rl/rl/`

```python
# fast_sac.py
@dataclass
class SacConfig:
    actor_hidden: tuple[int, ...] = (256, 256, 256)     # SB3-compatible: ReLU, no LayerNorm (decision D-ACTOR-SHAPE)
    critic_hidden: tuple[int, ...] = (512, 512, 512); critic_layernorm: bool = True; n_critics: int = 2
    batch_size: int = 4096; grad_steps_per_env_step: int = 8; lr_actor: float = 3e-4; lr_critic: float = 3e-4; lr_alpha: float = 3e-4
    gamma: float = 0.97; tau: float = 0.005; n_step: int = 1; alpha_init: float = 0.02; target_entropy: float = -3.0; log_alpha_bounds: tuple[float, float] = (-9.2, 0.0)
    log_std_bounds: tuple[float, float] = (-5.0, 2.0); grad_clip_critic: float = 10.0; warmup_env_steps: int = 20; compile: bool = False; seed: int = 0
class Actor(torch.nn.Module):           # latent_pi: Sequential(Linear, ReLU, ...), mu: Linear, log_std: Linear  (state_dict keys match SB3 SACPolicy.actor)
class FastSAC:
    def __init__(self, obs_dim: int, priv_dim: int, act_dim: int, cfg: SacConfig, *, device: torch.device) -> None
    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor      # [N,6] in [-1,1]
    def update(self, batch: "Batch") -> dict[str, float]                                  # one gradient step on critics, actor, alpha; returns metrics
    def state_dict(self) -> dict; def load_state_dict(self, d: dict) -> None

# replay_gpu.py
@dataclass
class Batch: obs, priv, act, rew, next_obs, next_priv, terminated, truncated, n_step_discount: torch.Tensor
class GpuReplay:
    def __init__(self, capacity: int, obs_dim: int, priv_dim: int, act_dim: int, *, device: torch.device, n_step: int = 1, gamma: float = 0.97) -> None
    def add(self, obs, priv, act, rew, next_obs, next_priv, terminated, truncated) -> None   # [N, ...] one vector step; next_obs = final_obs for resetting worlds
    def sample(self, batch_size: int, *, generator: torch.Generator) -> Batch
    size: int; capacity: int; vram_bytes: int

# policy_export.py
EXPORT_VERSION: Final[int] = 1
@dataclass
class ExportedPolicy: actor: Actor; obs_layout_version: int; task_name: str; obs_dim: int; act_dim: int; sim_params_version: int; dr_stage: int; git_sha: str; sac_cfg: dict; sim_success_rate: float
def export_policy(sac: FastSAC, task: TaskSpec, *, sim_params_meta: SimParamsMeta, dr_stage: int, sim_success_rate: float, path: Path) -> Path   # torch.save of a plain dict (no pickled classes)
def load_policy(path: Path, *, device: str = "cpu") -> ExportedPolicy       # refuses obs_layout_version != OBS_LAYOUT_VERSION

# rollout_logger.py  (TD-MPC2 offline pre-training input and the state-WM dataset)
class RolloutLogger:
    def __init__(self, root: Path, *, episode_steps: int, obs_dim: int, act_dim: int, chunk_episodes: int = 1000, keep_priv: bool = False) -> None
    def add(self, step: BatchedStep, obs_before: torch.Tensor, action_applied_norm: torch.Tensor) -> None   # action = applied_delta_ticks / max_delta_ticks (L4), not the policy sample
    def flush(self) -> Path | None         # writes chunk_<k>.pt: TensorDict batch [E, T+1] with keys obs f32, action f32 (row 0 = NaN like TD-MPC2's to_td), reward f32 (row 0 = NaN), terminated bool, truncated bool, episode i64, policy_tag i64
    def close(self) -> None
def check_chunk(path: Path) -> ArtifactReport     # shapes, no NaN beyond row 0, |action| <= 1, episode ids contiguous (L12: every gate shown to fail on a broken file)
```

### FS11. Deployment, fine-tuning bridge, gap report — `src/so101_rl/rl/`

```python
# deploy_policy.py — `pixi run deploy-policy --policy PATH --backend {shm,sim} --task {reach,die_to_zone} --episodes N --seed S --run-id ID [--attended] [--deterministic] [--speed-scale 0.5]`
#   shm: ShmBackend (F7) + ObjectTableShm reader (F9) + FiducialCheck; sim: ShmBackend against a `servo --backend mujoco` child process (the SAME code path, MujocoBus instead of FeetechBus)
#   builds FS3 obs with build_policy_obs_np, runs the exported actor at 10 Hz, logs F8 episodes (applied action), refuses --backend shm without --attended until runs/hw/tier.json says tier >= 1
#   --speed-scale s multiplies the policy action before the env's tick scaling (the clamp chain is untouched); zero-shot ladder starts at 0.5
#   writes runs/<run-id>/eval.json: {n, successes, wilson95: [lo, hi], final_err_m: {p50, p90}, interventions, trips: {...}}
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]

# sb3_bridge.py
def load_actor_into_sb3(model: "stable_baselines3.SAC", exported: ExportedPolicy) -> None   # asserts net_arch/activation match, copies latent_pi/mu/log_std; critics untouched
class FrozenActorWarmup(stable_baselines3.common.callbacks.BaseCallback): def __init__(self, steps: int) -> None   # actor lr = 0 for `steps` gradient steps, critics learn
class SimBaselineEval(BaseCallback): def __init__(self, exported: ExportedPolicy, env, *, every_steps: int, n_episodes: int, rollback_margin: float) -> None

# tdmpc2_offline.py — `pixi run pretrain-tdmpc2 --data-dir runs/sac-sim/<run>/rollouts --task reach --steps 200000 --out runs/tdmpc2-offline/<ts>/`
def pretrain_tdmpc2(data_dir: Path, task: TaskSpec, *, steps: int, batch_size: int = 512, horizon: int = 3, device: str, out: Path, seed: int = 0) -> Path
#   custom single-task loop: make_tdmpc2_cfg({task: f"so101-{task.name}", obs: "state", obs_shape: {"state": (policy_input_dim,)}, action_dim: 6, episode_length: T+1, multitask: False, ...}); Buffer.load(td) per chunk; agent.update(buffer) `steps` times; writes model.pt (+ optimizer, scale) and metrics.csv; exits non-zero if metrics.csv has < 3 rows (L12)

# gap_report.py — `pixi run gap-report --real runs/<real-eval> --sim runs/<sim-eval> --out runs/gap/<ts>.json`
@dataclass
class GapReport: task: str; policy: str; n_real: int; n_sim: int; success_real: float; wilson95_real: tuple[float, float]; success_sim: float; gap: float
                 final_err_real_m: dict; final_err_sim_m: dict; tracking_err_ks: dict[str, float]; clamp_active_real: dict[str, float]; clamp_active_sim: dict[str, float]
                 delay_est_ticks: float; replay_in_sim_rms_pct: dict[str, float]; tracker_unknown_frac_real: float; tracker_unknown_frac_sim: float; interventions_per_hour: float
```

### FS12. Real-arm goal env (supersedes F7's `So101ReachEnv`) — `src/so101_rl/env/goal_env.py`, `env/backend.py`, `env/goals.py`

```python
class EnvBackend(Protocol):            # F7 unchanged: reset / send_delta / read_state / wait_policy_tick / fk
class ShmBackend(EnvBackend): def __init__(self, cfg: So101Config, clock: Clock = RealClock()) -> None
class SimBackend(EnvBackend): def __init__(self, cfg: So101Config, calibration, *, action_delay_ticks: int = 0, sim_params: SimParams | None = None, seed: int = 0, start: Literal["home","rest"] = "home") -> None   # in-process ServoCore + MujocoBus + LocalIpc + FakeHeartbeatLink + FakeClock; 3 core.tick()/step_ticks(1) per policy step
class So101GoalEnv(gymnasium.Env):
    def __init__(self, backend: EnvBackend, cfg: So101Config, task: TaskSpec, *, goal_sampler: GoalSampler | None = None, object_table: ObjectTableShm | None = None, render_mode: str | None = None) -> None
    observation_space = Dict(observation: Box(task.obs.dim), achieved_goal: Box(3), desired_goal: Box(3)); action_space = Box(-1, 1, (6,))
    def reset(...); def step(action) -> (obs, reward, terminated, truncated, info)   # info keys as F7 plus "tracker_visible"; truncated on step limit or any fault_bits (trip -> info["trip"])
    def compute_reward(self, achieved_goal, desired_goal, info) -> np.ndarray         # = tasks.reward_np (vectorised for SB3 HER)
class BoxGoalSampler(GoalSampler): def __init__(self, bounds: GoalBounds, *, fk: So101FK, limits: JointLimits, seed: int) -> None   # rejection sampling + FK-reachability table
gymnasium.register("So101Reach-v0" -> So101GoalEnv(task=REACH)); gymnasium.register("So101DieToZone-v0" -> So101GoalEnv(task=DIE_TO_ZONE))
```

> **As built 2026-10-06 (SR1 + integration):** registered ids default to the CPU `SimBackend`, never the arm. `BoxGoalSampler` gained a torch front end (`sample_torch`, `reachable_torch`, `contains_torch`, `.keys`) and is the one sampler of both envs. The object table's die position is the **centroid** (`HsvDepthDieTracker.centroid_base`: top-face detection minus half an edge along the table normal).

## Parity test definition (frozen; `tests/test_batched_parity.py`)

**Rig.** `cfg` + `calibration` from `tests/data`. CPU reference: `build_servo_core(cfg, calibration, MujocoBus(load_model(), ...), LocalIpc(), FakeHeartbeatLink(FakeMcu(clock)), FakeClock)` booted like `tests/test_action_units.py::Rig` (button + ARM + re-energise to home). Batched: `BatchedGoalEnv(cfg, calibration, REACH, nworld=4, sim_params=SimParams.from_mjcf(model), randomizer=None, tracker_noise=None, seed=0)` reset to the same `home` keyframe with the chain resynced to the same present ticks, delay 0, no noise.

**Action corpus** (seeded, 200 policy steps × 20 sequences, stored as a generator so the test stays small): (a) uniform `U(-1, 1)` deltas; (b) saturating ±1 bursts and sign flips every 3 steps; (c) a ramp that drives `shoulder_lift` into its joint limit and holds; (d) a ramp that drives the tip into the box wall at x = 0.34 (BOX clamp active) and down to the box floor; (e) alternating ±1 on all joints for 30 steps (velocity cap and tracking clamp active); (f) NaN rows at 3 known steps.

**Layer A — chain parity (deterministic, CPU, no GPU needed, runs in `pixi run test`).** Record `present_ticks` per tick from the CPU rig for every sequence; feed the recorded stream to a fresh `SafetyPipeline` (numpy) and to `BatchedServoChain` (torch, CPU device, N=1 and N=4 with world 2 getting the sequence). Assert for every tick: `target_ticks`, `goal_ticks`, `clamped_mask`, the clamp bits of `trips`, and per policy step `applied_delta_ticks` are **bit-identical**; the NaN rows produce `TripCause.NAN` with zero delta in both. Expected: 20 × 600 ticks, 0 mismatches.

**Layer B — closed-loop parity (`@pytest.mark.gpu`, `pixi run test-gpu`).** Run both rigs closed loop on the same sequences. Assert: (1) on every tick where `present_cpu == present_warp` on all 6 joints, `target_cpu == target_warp` exactly; (2) overall `max |target_cpu − target_warp| ≤ 1` tick and the fraction of (tick, joint) pairs that differ is ≤ 0.1 %; (3) arm `qpos` max |Δ| ≤ 1e-4 rad over the 600 ticks of each contact-free sequence (a, b, e), ≤ 5e-4 rad for the box-floor sequence (d) where the jaw boxes touch the table; (4) `contact_count(jaw_col, table)` equal per tick in (d). Report (not assert) the per-tick histogram of present differences into `runs/parity/<ts>.json`.

**Layer C — grasp contact parity (`gpu`, the spike's open item).** Start the die between the open jaws at the home-lowered pose, close the gripper over 20 steps, lift 10 steps, hold 20 steps, on CPU and Warp. Assert the die stays in the gripper in both (die-to-tip distance < 0.02 m at the end) and die position |Δ| ≤ 2e-3 m, with `contact_count(jaw_col, die)` ≥ 1 on the same ticks (±1 tick tolerance for first contact). Failure here is a **gate failure of S0**, not of S1: it means the primitive-jaw contact model differs between engines and the fallback (ManiSkill3 port) is on the table (plan §7).

Every layer has a "shown to fail" twin: Layer A with `max_ticks_per_tick` perturbed by +1 in the torch chain only (→ mismatches), Layer B with the batched env's substep count off by one (→ trajectory tolerance exceeded), Layer C with the die mass × 10 on one side.

## Tasks

Legend: **SW** = software, runs now on fake / CPU MuJoCo / MuJoCo Warp; **GPU** = SW task that needs the 3080 (`pixi run test-gpu`, `default` env); **HW** = hardware/human, Tier 0 attended, prerequisites named. Acceptance commands run from the repo root. `Est.` = agent-hours (SW/GPU implementation) + GPU-hours (compute) + human-hours (HW). Unless stated, acceptance is in the `cpu` environment: `pixi run -e cpu <task>`; GPU acceptance uses the `default` environment: `pixi run <task>`.

---

### Unit SU0 — S0: version pin, Warp backend, interface freeze (wave 0, sequential)

#### Task S0.1 (SW): One MuJoCo and one torch across envs; `pixi install` for every env; full suite green on `cpu` AND `default`
- **Files:** `pyproject.toml`, `pixi.lock`, `README.md` (Environment table), `tests/test_env_sanity.py`
- **Specialist:** pixi-workspace-maintainer (worker) for lock health
- **Dependencies:** None (H1 done)
- **Steps:**
  1. `[tool.pixi.feature.sim.dependencies]`: `mujoco-python = "3.14.*"`, `mujoco-warp = "3.14.*"`, `warp-lang = ">=1.17"` (all conda-forge). Remove `mujoco-python = ">=3.3"`.
  2. Core `pytorch = "2.11.*"` (was `>=2.7,<2.12`); `torchvision = ">=0.26,<0.27"`; `numpy = "2.2.*"` unchanged. `[tool.pixi.feature.gpu.dependencies]`: `pytorch-gpu = "2.11.*"`, `cuda-version = "12.9.*"` (the spike's warp build is cuda129; **decision D-CUDA** — if the solver cannot pair torch 2.11 with 12.9, fall back to `12.8.*`/`12.6.*` and record it). Keep `[tool.pixi.system-requirements] cuda = "12"`.
  3. Add the `cma` PyPI dependency to the `train` feature (sysid). Do **not** add `optuna` unless S2.4 needs it (anti-hypertrophy).
  4. New pixi tasks (as built after the SU0 review: only tasks whose module exists are defined — `test-gpu`, `ci-gpu`, `bench-warp`; the rest are listed as planned in a `pyproject.toml` comment and land with their module, guarded by `tests/test_pixi_tasks.py`; `sim-parity` will be plain `pytest tests/test_batched_parity.py`, the `-m "gpu or not gpu"` filter was a no-op): `test-gpu = 'pytest -m "gpu and not hardware"'`, `ci-gpu = { depends-on = ["lint", "test", "test-gpu"] }`, `bench-warp = "python scripts/bench_warp.py"`, `sim-parity = 'pytest tests/test_batched_parity.py -m "gpu or not gpu"'`, `tray-check = "python -m so101_rl.sim.tray_check"`, `sysid-excite = "python -m so101_rl.sim.sysid.run_excitation"`, `sysid-fit = "python -m so101_rl.sim.sysid.fit"`, `sim-gates = "python -m so101_rl.rl.gates"`, `train-sim-sac = "python -m so101_rl.rl.train_sim_sac"`, `eval-sim = "python -m so101_rl.rl.eval_sim"`, `export-policy = "python -m so101_rl.rl.policy_export"`, `deploy-policy = "python -m so101_rl.rl.deploy_policy"`, `pretrain-tdmpc2 = "python -m so101_rl.rl.tdmpc2_offline"`, `gap-report = "python -m so101_rl.rl.gap_report"`. Keep every existing task.
  5. `pixi install -e cpu && pixi install -e default && pixi install -e robot`. Record the solved versions per env in `README.md` (replace the 2026-09-30 table; keep the pin rationale incl. "lerobot 0.6.1 needs torch < 2.12, numpy < 2.3").
  6. `tests/test_env_sanity.py` gains: `test_mujoco_version_pinned` (`mujoco.__version__.startswith("3.14")` whenever `mujoco` imports), `test_mujoco_warp_importable` (`import mujoco_warp, warp` succeeds in any env that has the `sim` feature; on a host without CUDA `warp` must still import), `test_torch_version_pinned` (`2.11.`), `test_single_torch` unchanged.
- **Tests:** the four above plus the whole existing suite in both envs.
- **Acceptance:** `pixi install -e cpu && pixi install -e default && pixi install -e robot` → exit 0. `pixi run -e cpu test` → all pass (≥ the current count), `pixi run test` (default env) → all pass; `pixi run -e cpu python -c "import mujoco, torch, numpy, mujoco_warp; print(mujoco.__version__, torch.__version__, numpy.__version__)"` → `3.14.x 2.11.y 2.2.z`; `pixi run python -c "import torch; assert torch.cuda.is_available()"` → exit 0; `pixi list -e default | grep -E '^(mujoco|pytorch|numpy|warp)'` shows one torch (conda) and mujoco 3.14.
- **Est.:** 1.5 agent-h (solver iterations dominate).

#### Task S0.2 (SW): `jaw_col margin=0`, exact tick timestep, CPU tests re-run under mujoco 3.14
- **Files:** `assets/so101/so101_rl_scene.xml`, `assets/so101/README.md`, `tests/test_sim_model.py`
- **Specialist:** simulation-to-hardware-bridge
- **Dependencies:** S0.1
- **Steps:**
  1. `margin="0"` on the `jaw_col` default (MJWarp rejects non-zero margins on box pairs). Comment why.
  2. **Decision D-TIMESTEP (default: adopt):** add `<option timestep="0.00196078431"/>` (= 1/(30·17) s) so `MujocoBus.steps_per_tick × timestep == 1/servo_hz` exactly and one policy step is 100.0 ms of sim time instead of 102 ms. If the user prefers to keep 0.002, skip and record the 2 % scale in `assets/so101/README.md`.
  3. Re-run `tests/test_sim_model.py`, `test_mujoco_bus.py`, `test_action_units.py`, `test_reenergize_backends.py`, `test_compare_real.py`, `test_fk_vs_mujoco.py`, `test_servo_process.py` (mujoco backend) under 3.14 in both envs; fix any tolerance drift (report, do not loosen silently).
- **Tests:** `test_sim_model.py::test_jaw_col_margin_is_zero`, `::test_tick_is_integer_substeps` (`abs(steps_per_tick * timestep - 1/servo_hz) < 1e-9`), `::test_rest_keyframe_stable_300_steps` (existing, must still pass with the new dt).
- **Acceptance:** `pixi run -e cpu pytest tests/test_sim_model.py tests/test_mujoco_bus.py tests/test_action_units.py tests/test_reenergize_backends.py tests/test_compare_real.py tests/test_fk_vs_mujoco.py -q` → pass; same under `pixi run pytest ... -q`.
- **Est.:** 1 agent-h.

#### Task S0.3 (GPU): `WarpSim` wrapper with sizes from config and a FAILING overflow detector; benchmark script
- **Files:** `src/so101_rl/sim/warp_model.py`, `src/so101_rl/config/schema.py`, `src/so101_rl/config/loader.py`, `src/so101_rl/config/so101.yaml`, `scripts/bench_warp.py`, `tests/test_warp_model.py`, `tests/test_config.py` (sim_warp rows), `tests/test_config_single_source.py`
- **Specialist:** cuda-performance-engineer (graph capture, sync points); simulation-to-hardware-bridge
- **Dependencies:** S0.2
- **Steps:**
  1. FS1: `SimWarpConfig`, `sim_warp:` section, strict loading.
  2. FS2 `WarpSim`: `put_model(load_model()[0])`, `make_data(mjm, nworld=N, njmax=cfg.njmax, nconmax=cfg.nconmax_per_world)` (API names per the 3.14 package — check `inspect.signature` at implementation and record in the module docstring), torch views via `wp.to_torch`, `step(nsteps)` captured once per distinct `nsteps` as a CUDA graph (`wp.ScopedCapture`), `check_overflow()` reads `d.nefc` (`[N]`) and `d.nacon` (`[1]`) with one device→host copy and raises `ContactBufferOverflow(f"world {w}: nefc {n} ≥ njmax {njmax}")` / for `nacon ≥ naconmax`. `BatchedGoalEnv` calls it every `overflow_check_every_steps` policy steps and on `close()`; `WarpSim.close()` also calls it so a run cannot end with an undetected overflow.
  3. `set_world_params` writes the per-world `*` fields (`dof_damping[N, nv]`, `dof_armature`, `dof_frictionloss`, `actuator_gainprm[N, nu, :]`, `actuator_biasprm`, `geom_friction[N, ngeom]`, `body_mass[N, nbody]`) — expanding the field to a leading world dimension if the model was created unbatched. **If 3.14 refuses per-world expansion of any of these fields, the task records which, and S2/S3 use the CPU/chunked fallbacks named in FS7/FS8.**
  4. `scripts/bench_warp.py --nworld 256,1024,2048 --steps 2000 --njmax 256 --nconmax 32 --out runs/bench/warp-<ts>.json`: random ctrl within `ctrlrange` resampled every 50 steps (as in the spike), reports env-steps/s, VRAM delta (`torch.cuda.max_memory_allocated` + Warp's), max `nefc` / `nacon` seen (sampled every step, not every 50th), and runs `check_overflow()` at the end.
- **Tests (`@pytest.mark.gpu`):** `test_warp_model.py::test_sizes_from_config` (sizes equal the yaml), `::test_qpos_zero_copy` (`qpos.data_ptr()` equals the warp pointer; writing through torch is visible to warp), `::test_step_matches_cpu_300_steps` (sinusoid around rest: max |Δqpos| arm ≤ 1e-6 rad vs CPU, the spike's number with margin), `::test_overflow_detector_fails` (`njmax=16` → `ContactBufferOverflow` within 100 steps of random ctrl — the gate shown to fail), `::test_overflow_detector_quiet_at_256` (no raise over 2000 steps at N=256), `::test_set_world_params_per_world` (two worlds with kp 998 vs 300: step responses differ in the expected direction), `::test_close_checks_overflow`. CPU-marked: `test_config.py::test_sim_warp_section_strict` (missing key → error; `overflow_policy: warn` → error).
- **Acceptance:** `pixi run pytest tests/test_warp_model.py -q` → all pass; `pixi run -e cpu pytest tests/test_config.py tests/test_config_single_source.py -q` → pass; `pixi run bench-warp --nworld 1024 --steps 2000` → prints `env_steps_per_s ≥ 500000`, `vram_mb ≤ 1500`, `overflow: none`, writes `runs/bench/warp-<ts>.json`. Exit 0 = gate PASS, 1 = FAIL, 2 = NOT EVALUATED (N=1024 missing, `--no-parity`, `--njmax`/`--nconmax` overrides, or `--steps` < 2000; SU0 review). **S0 gate values** (plan §5 S0): throughput ≥ 500k env-steps/s at N=1024, VRAM ≤ 1.5 GB at 1024, parity ≤ 1e-6 rad over 300 steps, zero overflow at njmax 256 / nconmax 32.
- **Est.:** 4 agent-h + 0.3 GPU-h.

#### Task S0.4 (SW): Interface freeze — obs layout, task specs, goal bounds, ARCHITECTURE sync
- **Files:** `src/so101_rl/env/obs.py`, `src/so101_rl/env/tasks.py`, `src/so101_rl/env/__init__.py`, `tests/test_obs_layout.py`, `tests/test_tasks.py`, `docs/ARCHITECTURE.md` (§4 new modules and the `sim → safety.types/fk` and `env/batched_env → sim` arrows, §5 FS pointers, §7 status "S0 built", §8 pins and tasks, §10 row 0.12), `plans/phase0-implementation-plan.md` (status note: U8/U9/U10 superseded per this plan's table)
- **Specialist:** lerobot-specialist (SB3 Dict-obs contract); testing-specialist
- **Dependencies:** S0.1
- **Steps:** implement FS3 and FS4 exactly (both numpy and torch builders; torch on CPU in tests); `REACH` and `DIE_TO_ZONE` constants with `die_start` placeholder `(0.20, -0.05)`, `sigma_xy 0.005`, flagged `measured_on_hardware: false` for H9; zone from the scene site (`0.22, 0.10`, half edge 0.030).
- **Tests:** `test_obs_layout.py::test_np_and_torch_builders_identical` (random states → |Δ| < 1e-6), `::test_dims` (21 / 26, policy input 24 / 29), `::test_layout_version_constant`; `test_tasks.py::test_np_torch_reward_identical`, `::test_success_requires_visible` (invisible die inside the zone → no success; the predicate shown to fail-safe), `::test_debounce` (die task needs 2 consecutive steps), `::test_goal_bounds_inside_box_and_radius` (1e5 samples).
- **Acceptance:** `pixi run -e cpu pytest tests/test_obs_layout.py tests/test_tasks.py -q` → pass; `pixi run -e cpu pytest tests/test_architecture_sync.py -q` → pass; `grep -c "sim_warp" docs/ARCHITECTURE.md` ≥ 1; version table has a 0.12 row.
- **Est.:** 2 agent-h.

**Commit after S0.4:** `S0: pin mujoco 3.14 / torch 2.11 across envs, MuJoCo Warp backend with failing overflow detector, obs/task interface freeze`.

---

### Unit SU1 — S1: batched servo-parity env (wave 1; parallel with SU2, SR1)

#### Task S1.1 (SW): `BatchedServoChain` + Layer A parity
- **Files:** `src/so101_rl/sim/batched_servo.py`, `tests/test_batched_servo.py`, `tests/test_batched_parity.py` (Layer A part)
- **Specialist:** real-time-control-specialist (clamp semantics); simulation-to-hardware-bridge
- **Dependencies:** SU0
- **Steps:** FS5; int64 everywhere; `torch.round` (half-to-even) matches `np.rint`; `new_command` reproduces `sanitize_delta` (any non-finite entry zeroes the whole row and flags NAN), `rint`, `goal += Δ`, joint-limit clip and `_cmd_mask`; `apply` reproduces the nine stages incl. the `_note` bookkeeping (command-level mask ORed into the first tick's mask); batched FK with the same `KinematicChain` (`parent_T`, `tip_T`) as `So101FK`; `in_box` with `present_tolerance_m` for the present test and exact for the target; the RANGE check leaves the target as-is and sets the bit (pipeline semantics). No `safety/` code is imported except `TripCause` and `KinematicChain`.
- **Tests:** `test_batched_servo.py::test_fk_matches_so101fk` (500 random q, |Δ| < 1e-9 m), `::test_interp_lands_on_goal_in_3_ticks`, `::test_tracking_pullback_bounds_goal` (blocked present → |goal − present| ≤ E every tick), `::test_box_revert`, `::test_nan_row_zeroed_whole_row`, `::test_no_cross_world_leak` (world k's clamps never change world j); `test_batched_parity.py::test_layer_a_chain_bit_identical[seq0..19]`, `::test_layer_a_shown_to_fail_with_perturbed_vmax`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_batched_servo.py tests/test_batched_parity.py -q -m "not gpu"` → pass (≥ 28 tests).
- **Est.:** 3 agent-h. `parallelizable-via-worktree: true`

#### Task S1.2 (SW): Simulated die tracker
- **Files:** `src/so101_rl/sim/sim_tracker.py`, `tests/test_sim_tracker.py`
- **Specialist:** robotics-vision-navigator (projection, occlusion); simulation-to-hardware-bridge
- **Dependencies:** SU0
- **Steps:** FS6. Camera model from `calib/hand_eye.yaml` when it exists (`load_hand_eye`), else a nominal overhead pose at z = 0.614 m looking down at (0.20, 0, −0.02) with the D435 640×480 intrinsics (fx ≈ fy ≈ 615 px, so 1 px ≈ 1 mm at that range, plan §2). Occlusion: project the die centre and the four `jaw_col` boxes (centres + half sizes, FS2 `body_xpos`/`body_xquat`) into the image; the die is invisible when the fraction of its projected disc (radius 8 mm) covered by the jaw rectangles exceeds `occlusion_fraction`, **or** the tip is above the die within `occlusion_radius_m` horizontally (cheap fallback, same result in practice); also invisible when outside the FOV or below the table. Latency: a per-world ring of `max_latency` frames; `update(frames=3)` advances 3 camera frames per policy step and emits the frame `latency` back. Dropout bursts. Noise: per-frame Gaussian in the camera frame (σ_z along the optical axis), then a per-episode rigid bias (`tracker_bias`, `tracker_rot`) applied in the base frame. Held last value + age when invisible; `visible` means "seen in the latest emitted frame".
- **Tests:** `test_sim_tracker.py::test_visible_die_noise_stats` (10k frames: σ within 10 %), `::test_occluded_when_tip_over_die` (tip 5 mm above the die → invisible), `::test_not_occluded_when_tip_away`, `::test_latency_frames_exact` (a die jump appears `latency` frames later), `::test_dropout_burst_holds_last_and_ages`, `::test_outside_fov_invisible`, `::test_semantics_match_real_tracker_on_synthetic_frames` (renders a synthetic RGB+depth frame of the sim scene for 20 configurations with `mujoco.Renderer`, runs `HsvDepthDieTracker.detect` on it and compares visible/invisible with the sim tracker's verdict: agreement ≥ 18/20; the 2 allowed disagreements are logged). CPU only (torch CPU device).
- **Acceptance:** `pixi run -e cpu pytest tests/test_sim_tracker.py -q` → pass.
- **Est.:** 3 agent-h. `parallelizable-via-worktree: true`

#### Task S1.3 (GPU): `BatchedGoalEnv` (reach + die), resets, tray geometry, Layer B/C parity
- **Files:** `src/so101_rl/env/batched_env.py`, `assets/so101/so101_rl_scene.xml` (tray walls + overhead camera; hand-off from SU0 — sequential, SU0 is merged), `assets/so101/README.md`, `src/so101_rl/sim/tray_check.py`, `tests/test_batched_env.py`, `tests/test_batched_parity.py` (Layers B and C), `tests/test_tray_check.py`
- **Specialist:** simulation-to-hardware-bridge; cuda-performance-engineer (sync points, graphs)
- **Dependencies:** S1.1, S1.2, S0.3
- **Steps:**
  1. FS9. Per policy step: `delta = rint(a × max_delta)`; `chain.new_command`; for `rem in (3, 2, 1)`: present = `q_to_ticks(ws.qpos)` (+ proprio noise for the *policy* copy only; the chain sees the true quantised ticks, like the real loop), `chain.apply(present, rem)`, write `ctrl` through the per-world delay FIFO (`delay_ticks` per world, FS7) with the register clip to the calibration range and the `ctrlrange` clip, `ws.step(steps_per_tick)`. Then tracker update (3 frames), FS3 obs (policy) and priv, reward/success via `tasks`, debounce counters, `terminated`/`truncated`, `final_*`, auto-reset of done worlds.
  2. Resets: reach — `start_pose="home"` + uniform ±30 ticks on the arm joints (never outside the box: FK-checked, resample); goal from `GoalBounds` via a torch rejection sampler whose bounds equal `BoxGoalSampler` (FS12) — cross-test. Die — die placed at rest on the table at `die_start` (xy ~ N(well centre, σ), yaw uniform, z = table + 8 mm, zero velocity), `reset_settle_steps` physics steps if > 0 (default 0; decision D-TRAY); zone centre fixed ± `zone_offset_xy` (DR); the chain is resynced to present on every reset (`resync_target` semantics).
  3. Scene: a static `tray` body — four inclined box geoms forming a shallow funnel around the start well plus a low rim around the work area (contype 3 so the die and jaws collide, the arm meshes do not); an `overhead` camera matching `CameraModel`'s nominal pose; **upstream files untouched**. `sim/tray_check.py` (`pixi run tray-check --drops 100`): drops the die from random points inside the rim on the CPU model, settles 2 s, reports the fraction that ends within 2 σ of the well centre and writes `runs/tray/<ts>.json` — the sim twin of H9 (≥ 95 % expected; the real tray's measured distribution replaces `die_start` after H9).
  4. `render_worlds` via `copy_world_to` + `mujoco.Renderer` (`overhead` camera). `stats`.
  5. Overflow check cadence from config; `close()` checks.
- **Tests:** CPU-markable parts use `nworld=2` on a CPU torch device with a `WarpSim` stub? **No** — the env needs MJWarp; all env tests are `@pytest.mark.gpu` except the pure-torch reset/goal sampler helpers. `test_batched_env.py::test_reset_obs_shapes_and_layout`, `::test_step_advances_51_substeps`, `::test_action_scaling_is_f7_rule` (+1 on joint j → applied `max_delta_ticks[j]` after one step from home, other joints 0 — Test 1 of plan §5 "Action units" on the batched backend), `::test_truncation_sets_final_obs`, `::test_auto_reset_phase_randomised`, `::test_scripted_reach_succeeds` (a Jacobian-free proportional controller in tick space toward the goal reaches success on ≥ 90 % of 256 worlds — kinematic sanity, not RL), `::test_die_task_success_requires_visible_in_zone`, `::test_trip_on_nan_action_truncates`, `::test_goal_sampler_matches_numpy_bounds`; `test_batched_parity.py::test_layer_b_closed_loop[seq]`, `::test_layer_b_shown_to_fail_substeps`, `::test_layer_c_grasp_contact_parity`, `::test_layer_c_shown_to_fail_die_mass`; `test_tray_check.py::test_sim_tray_returns_die` (CPU, 20 drops ≥ 18 returns), `::test_tray_check_fails_without_tray` (model with the tray removed → return rate < 50 %, reported as FAIL).
- **Acceptance:** `pixi run pytest tests/test_batched_env.py tests/test_batched_parity.py -q` → pass (Layer C passing closes the spike's open item; a Layer C failure halts the track for a user decision); `pixi run -e cpu pytest tests/test_tray_check.py -q` → pass; `pixi run tray-check --drops 100` → `return_rate ≥ 0.95`, writes `runs/tray/<ts>.json`; `pixi run python -m so101_rl.env.batched_env --task reach --nworld 1024 --steps 500 --bench` → prints policy-steps/s ≥ 5000 (env only, random actions) and `clamp_active` stats.
- **Est.:** 6 agent-h + 0.5 GPU-h.

#### Task S1.4 (SW): SU1 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§4 `sim/batched_servo.py`, `sim/sim_tracker.py`, `sim/tray_check.py`, `env/batched_env.py`; §5 FS5/FS6/FS9 behaviour incl. the parity definition pointer; §7 status "S1 built, parity green"; §10 row), `README.md` (entry points: `tray-check`, `sim-parity`)
- **Est.:** 0.3 agent-h.

**Commit after S1.4:** `S1: batched servo-parity env on MuJoCo Warp (chain, tracker, tray, parity layers A-C)`.

---

### Unit SU2 — S2: sysid software (wave 1; parallel with SU1, SR1)

#### Task S2.1 (SW): `SimParams`, `calib/sim_params.yaml` v0, `compare_real --sim-params`
- **Files:** `src/so101_rl/sim/sysid/__init__.py`, `src/so101_rl/sim/sysid/params.py`, `calib/sim_params.yaml` (version 0 = MJCF prior, `heldout_rms_pct: null`), `calib/README.md` (schema row), `src/so101_rl/sim/compare_real.py` (hand-off from U4: `--sim-params`, `--tick-log`), `tests/test_sim_params.py`, `tests/test_compare_real.py` (new cases)
- **Specialist:** simulation-to-hardware-bridge
- **Dependencies:** SU0
- **Steps:** FS7 `SimParams`/`BatchedSimParams`; `apply_cpu` sets `actuator_gainprm[:, 0] = kp`, `actuator_biasprm[:, 1] = -kp`, `actuator_biasprm[:, 2] = -kv`, `dof_damping/armature/frictionloss`, `actuator_forcerange`, die/table `geom_friction`, die `body_mass`; `from_mjcf` reads them back (round trip). `compare_real --sim-params` applies them before the replay; `--tick-log` replays the 30 Hz `TICK_LOG_DTYPE` log (target per tick → present per tick).
- **Tests:** `test_sim_params.py::test_mjcf_roundtrip`, `::test_yaml_roundtrip_and_version_check` (version 2 file → refused), `::test_unverified_refused_by_default` (`heldout_rms_pct: null` needs `allow_unverified`), `::test_gainprm_biasprm_mapping` (a step response with kp doubled settles faster — direction check); `test_compare_real.py::test_sim_params_change_the_error` (a sim-generated log with kp×0.5 is > 5 % under the prior and ≈ 0 % with the matching params — the gate discriminates), `::test_tick_log_replay`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_sim_params.py tests/test_compare_real.py -q` → pass; `pixi run -e cpu compare-real --help` lists `--sim-params`.
- **Est.:** 2 agent-h. `parallelizable-via-worktree: true`

#### Task S2.2 (SW): Excitation plan generator and the (hardware-gated) runner, dry-run on sim
- **Files:** `src/so101_rl/sim/sysid/excitation.py`, `src/so101_rl/sim/sysid/run_excitation.py`, `src/so101_rl/sim/sysid/dataset.py`, `tests/test_excitation.py`
- **Specialist:** real-time-control-specialist (shm client, tick log); simulation-to-hardware-bridge (excitation design)
- **Dependencies:** SU0 (uses `servo/soak.py`'s random-policy shm client pattern and `servo/main.py --backend mujoco` for the dry run)
- **Steps:** FS7 `ExcitationPlan`/`generate_deltas`/`check_plan_in_box`: per joint solo multisines (3–5 components in 0.2–4 Hz, amplitudes so the velocity/tracking clamps are active 5–40 % of ticks — measured in the dry run and reported), chirps 0.1→3 Hz, steps ±10/±20/±45 ticks with 1 s holds, slow ramps ±5 ticks/step, 60 s random-policy, and (with `--with-gripper-die`) gripper open/close cycles at 3 speeds; all-joint combined segment last. `run_excitation.py`: `--backend {sim,real} --plan default --out runs/sysid/<ts>/ [--attended] [--with-gripper-die]`; `sim` spawns `servo --backend mujoco --tick-log ... --shm-name sysid-<pid>` as a child (like `soak.py`) and drives it over `ShmBackend`-equivalent raw shm writes at 10 Hz; `real` refuses without `--attended` and without `runs/hw/{estop,deadman,rest-pose,calib-check}-*.json` present (H2, H3, H5, H7 artifacts); logs F8 episodes per segment (`EpisodeLogger`) and copies the tick log; writes `plan.json` with the segment table and the measured clamp-active fraction per segment; exits non-zero if any artifact check fails (L12). **Never opens a serial device in `sim` mode.**
- **Tests:** `test_excitation.py::test_plan_in_box_kinematic_preview`, `::test_plan_out_of_box_rejected` (amplitude ×10 → `PlanCheck.ok False`), `::test_multisine_band_below_nyquist`, `::test_dry_run_sim_writes_dataset` (`@pytest.mark.slow`: 20 s plan through the mujoco servo child → `ExcitationDataset.from_run` has ≥ 3 segments, clamp-active in [0, 0.6]), `::test_real_refused_without_attended_and_artifacts` (monkeypatched argv, asserts exit 2 and that no `serial.Serial` was constructed), `::test_split_by_time_and_kind`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_excitation.py -q -m "not slow"` → pass; `pixi run -e cpu sysid-excite --backend sim --plan short --out runs/sysid/dry` → exit 0, `runs/sysid/dry/{episodes/*.h5, tick_log.bin, plan.json}` exist and `plan.json` reports per-segment `clamp_active_frac`.
- **Est.:** 3 agent-h. `parallelizable-via-worktree: true`

#### Task S2.3 (SW + GPU): Replay (multiple shooting) and the CMA-ES fit with Warp and CPU backends
- **Files:** `src/so101_rl/sim/sysid/replay.py`, `src/so101_rl/sim/sysid/fit.py`, `tests/test_sysid_fit.py`
- **Specialist:** simulation-to-hardware-bridge; performance-optimization-specialist (CPU pool)
- **Dependencies:** S2.1, S2.2, S0.3 (`set_world_params`)
- **Steps:** FS7 `replay_*`, `rms_pct`, `fit_sim_params`. Objective = mean over segments and joints of `rms_pct`, plus 0.2 × worst joint; multiple shooting re-initialises `qpos/qvel` from the real present every 60 ticks (2 s) so friction/delay errors do not integrate into a drift; the delay is an outer integer grid (0..6 ticks) with the FIFO pre-filled from the real log; CMA-ES (`cma.CMAEvolutionStrategy`) over `log(kp, kv, frictionloss, armature)` with σ₀ = 0.3 and the prior mean, `popsize` = number of worlds per batch (Warp) or 32 (CPU pool of 16 processes, each with its own `MjModel`/`MjData`); `fixed` parameters are held at the prior; `spread` = final σ_i (log space); parameters with `spread > 0.5` are reported as *unidentified* (they get wide DR ranges in S3 instead of a point estimate). `pixi run sysid-fit --run runs/sysid/<ts> --backend warp --out calib/sim_params.yaml --heldout-kinds chirp,random` writes the yaml only when `heldout_rms_pct ≤ 5` for every joint (else writes `runs/sysid/<ts>/sim_params.unverified.yaml` and exits 1).
- **Tests:** `test_sysid_fit.py::test_synthetic_recovery_cpu` (`@pytest.mark.slow`: generate "real" data by running a 40 s plan through `SimBackend`/`MujocoBus` with truth = prior × {kp 0.7, kv 1.4, frictionloss 2.0, armature 1.0} and `delay_ticks = 3`, plus ±1 tick quantisation noise; fit from the prior with popsize 8 × 25 generations on the CPU backend; assert recovered kp and kv within ±10 %, frictionloss within ±25 %, delay exactly 3, held-out RMS < 1 %), `::test_fit_with_delay_fixed_zero_fails_gate` (the gate shown to fail: same data, `fixed=("delay_ticks",)` → held-out RMS > 5 % on at least one joint), `::test_unidentifiable_pair_flagged` (fit kp and damping together on step-free sinusoid data → `spread["damping"] > 0.5`), `::test_multiple_shooting_reinit`, `@pytest.mark.gpu ::test_warp_and_cpu_objectives_agree` (same candidate set → |Δ objective| < 1e-3 pct).
- **Acceptance:** `pixi run -e cpu pytest tests/test_sysid_fit.py -q -m "not gpu"` → pass (the slow one < 10 min); `pixi run pytest tests/test_sysid_fit.py -q -m gpu` → pass; `pixi run sysid-fit --run runs/sysid/dry --backend cpu --out runs/sysid/dry/fit.yaml --popsize 8 --generations 10` → exit 0 and a yaml with `fit_rms_pct` ≈ 0 for the self-generated data.
- **Est.:** 5 agent-h + 1 GPU-h.

#### Task S2.4 (SW): SU2 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§4 `sim/sysid/`, `calib/sim_params.yaml`; §7 sysid gate as built; §10 row), `README.md` (entry points `sysid-excite`, `sysid-fit`), `docs/HARDWARE_CHECKLIST.md` (H12 now points at HS1–HS3; append-only)
- **Est.:** 0.3 agent-h.

**Commit after S2.4:** `S2: sysid software — excitation plans, 30 Hz replay, CMA-ES fit (Warp/CPU), versioned sim_params`.

---

### Unit SR1 — Real-arm goal env (Phase 0 U8 remainder, wave 1; parallel with SU1, SU2)

#### Task R1.1 (SW): `SimBackend` (reference rig) and `ShmBackend`
- **Files:** `src/so101_rl/env/backend.py`, `tests/test_env_backend.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** SU0 (U3, U4 done)
- **Steps:** FS12 backends as Phase 0 Task 8.1 minus `local_ipc.py`; `SimBackend` boots the core like `tests/test_action_units.py::Rig` (button + ARM + re-energise), accepts `sim_params` (`apply_cpu`) and `action_delay_ticks`; `ShmBackend.reset()` issues HOLD then ≤ `max_ticks_per_tick`-sized deltas toward `q0` and waits for `|target − present| < 5` ticks (never a jump).
- **Tests:** as Phase 0 8.1 plus `::test_sim_backend_applies_sim_params`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_env_backend.py -q` → pass.
- **Est.:** 2.5 agent-h. `parallelizable-via-worktree: true`

#### Task R1.2 (SW): `So101GoalEnv`, `BoxGoalSampler`, registration, checkers
- **Files:** `src/so101_rl/env/goal_env.py`, `src/so101_rl/env/goals.py`, `src/so101_rl/env/__init__.py` (registrations; hand-off from SU0), `tests/test_goal_env.py`, `tests/test_goals.py`
- **Specialist:** lerobot-specialist (SB3 HER contract); testing-specialist
- **Dependencies:** R1.1, S0.4
- **Steps:** FS12; obs via `build_policy_obs_np` (FS3); die block from `ObjectTableShm` when given (`visible`, age from `t_mono_ns`), zeros/invisible otherwise; `compute_reward` = `tasks.reward_np`; `truncated` on any `fault_bits` with `info["trip"]`.
- **Tests:** `gymnasium.utils.env_checker.check_env` and `stable_baselines3.common.env_checker.check_env` on `SimBackend` for both tasks; `::test_obs_equals_torch_builder_on_same_state`; scripted move-toward-goal reaches `is_success` on ≥ 8/10 goals; NaN action → `truncated`, `info["trip"] & NAN`, target unchanged; `info["applied_delta_ticks"]` equals the state record; `test_goals.py::test_sampler_bounds_equal_batched_env_bounds` (shares `GoalBounds` with the torch sampler; 1e5 samples all inside both).
- **Acceptance:** `pixi run -e cpu pytest tests/test_goal_env.py tests/test_goals.py -q` → pass.
- **Est.:** 3 agent-h.

#### Task R1.3 (SW): SR1 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§4 `env/`, §5 env row: FS3 shapes, truncation semantics; §10 row)
- **Est.:** 0.2 agent-h.

**Commit after R1.3:** `Goal-conditioned real/sim env with the FS3 observation layout (U8 remainder)`.

---

### Unit SUV — Training visualisation: live viewer, rollout videos, live dashboard (wave 1; parallel with SU1, SU2, SR1)

Added 2026-10-05 at the user's request: *"make sure it is possible to see the model training in the sim"*. The user chose all three of the live 3D viewer, rollout videos and a live dashboard. Real-arm replay is not included.

**Frozen interface FS11 (`so101_rl.viz`).** `rl/train_sim_sac.py` (S4.3) and the S1 env debugging tools call only this interface:

```python
# viz/monitor.py
@dataclass(frozen=True)
class VizConfig:
    live_viewer: bool = False           # --viewer
    viewer_worlds: tuple[str, ...] = ("0", "best", "worst")   # world ids or selectors
    viewer_hz: float = 30.0
    video_every_min: float = 10.0       # 0 disables
    video_worlds: int = 4
    video_seconds: float = 10.0
    tensorboard: bool = True
class TrainingMonitor:                  # one per training run; never raises into the training loop
    def __init__(self, cfg: VizConfig, sim: WarpSim, run_dir: Path, *, task: str, render_cfg: So101Config) -> None
    def on_step(self, env_step: int, *, success: torch.Tensor, ep_return: torch.Tensor) -> None   # O(1), no host sync
    def on_metrics(self, env_step: int, scalars: Mapping[str, float]) -> None                     # TensorBoard scalars
    def maybe_record(self, env_step: int, policy: Callable[[torch.Tensor], torch.Tensor]) -> Path | None   # rollout video when due
    def close(self) -> None
```

**Viz rules:**
- **Training speed:** visualisation must never slow training by more than **5 %**, measured.
- **Failure handling:** a crash in the viewer or the video writer is logged once, and the visualisation is disabled for the rest of the run, but training continues. This is the one place where swallowing an error is allowed. It applies only because the error is visualisation-only, and the swallowed error is reported in `metrics.csv` as `viz_disabled=1`.
- **Dependencies:** `viz/` imports `sim/` and `env/` read-only, and `rl/` imports `viz/`. `safety/` never imports it.

#### Task V.1 (SW + GPU): Live 3D viewer of selected parallel worlds
- **Files:** `src/so101_rl/viz/__init__.py`, `src/so101_rl/viz/live_viewer.py`, `tests/test_live_viewer.py`
- **Steps:**
  - A **separate process**, so the GUI never holds the training process's GIL. It runs `mujoco.viewer.launch_passive` on a CPU `MjModel` built by the same `load_model()` and receives the qpos and qvel of the selected worlds over a shared-memory seqlock (reuse `servo/shm.py` `SeqlockBlock`) at `viewer_hz`.
  - The training side copies only the selected worlds' rows, as one small device-to-host copy every `1/viewer_hz` s, and never waits on the viewer.
  - World selectors:
    - `0`: a fixed world;
    - `best` / `worst`: by the current episode return, re-picked at each episode boundary;
    - integers.
  - On-screen overlay:
    - the world id;
    - the DR stage;
    - the episode step;
    - the success flag;
    - the goal as a marker site, and the zone for die tasks;
    - the tracker's last-seen die position as a ghost marker, so occlusion is visible.
  - The viewer can be toggled with `pixi run viz-attach --run runs/...` against a training run that is already going, and closing the window never stops training.
  - Headless (no `DISPLAY`): the viewer refuses with a clear message, and videos still work.
- **Tests:** `test_viewer_payload_roundtrip` (shm), `test_selector_best_worst`, `test_viewer_absent_display_is_clean_error`, and `@gpu test_viewer_overhead_under_5pct`. The last one measures training steps/s with and without the publisher, with no window opened (publisher only).
- **Acceptance:** `pixi run -e default viz-demo --nworld 1024` opens a window showing worlds 0, best and worst under a random policy on WarpSim, run manually on `DISPLAY=:1`. The automated tests pass.

#### Task V.2 (SW + GPU): Rollout videos per checkpoint
- **Files:** `src/so101_rl/viz/video.py`, `tests/test_video.py`
- **Steps:**
  - Record `video_worlds` worlds for `video_seconds` with the current deterministic policy every `video_every_min`, and at every checkpoint.
  - **Two cameras:**
    1. the **overhead D435 pose**, the same camera as the real arm, so sim and real views match;
    2. a free 3/4 view.
  - Render through MuJoCo Warp's batch renderer when available, otherwise through CPU `mujoco.Renderer` on copied states. The CPU fallback is always available.
  - Overlay text: step, DR stage, return, success. Encode the frames to `runs/<run>/videos/<env_step>_<world>.mp4` with ffmpeg, which is already in the env.
  - Supersedes S4.3's `rl/render.py`. The gaming-audit renders (8 random, 4 highest-return, 4 shortest episodes) use this module.
- **Tests:** `test_video_written_cpu_renderer` (a stub env, CPU), `test_overlay_text_present`, `test_overhead_camera_matches_d435_extrinsics` (the camera pose equals the config's D435 pose).
- **Acceptance:** `pixi run -e default viz-demo --record 10 --out runs/viz-demo/` writes at least 2 mp4s that play.

#### Task V.3 (SW): Live TensorBoard dashboard
- **Files:** `src/so101_rl/viz/monitor.py` (FS11), `tests/test_monitor.py`
- **Steps:**
  - `TrainingMonitor` writes TensorBoard scalars. Training curves:
    - success rate, current and eval;
    - mean return;
    - episode length;
    - DR stage.
  - SAC internals:
    - α;
    - entropy;
    - Q mean and max;
    - critic loss;
    - action saturation;
    - clamp-active fractions;
    - tracker visible fraction.
  - System health:
    - env-steps/s;
    - grads/s;
    - GPU temperature, memory and utilisation (NVML, which `warp-lang` already depends on);
    - review triggers as text.
  - It embeds the V.2 videos (`add_video`), and `metrics.csv` stays the source of truth.
  - `pixi run dashboard --logdir runs/` starts TensorBoard on localhost.
  - Add `tensorboard` (conda-forge) to the `train` feature.
- **Tests:** `test_scalars_written`, `test_video_embedded`, and `test_monitor_never_raises_into_training`, which injects a writer failure and asserts training continues with `viz_disabled=1` logged.
- **Acceptance:** `pixi run -e cpu pytest tests/test_monitor.py tests/test_video.py tests/test_live_viewer.py -q -m "not gpu"` passes. `pixi run dashboard` serves the demo run's scalars and videos.

**Est.:** about 6 agent-hours in total, with `parallelizable-via-worktree: true`.

### Unit SU3 — S3: domain randomisation and curriculum (wave 2; parallel with SU4, SR2)

#### Task S3.1 (SW + GPU): `DrSpec`, `DomainRandomizer`, stage yaml, env hooks
- **Files:** `src/so101_rl/sim/domain_rand.py`, `src/so101_rl/sim/dr_stages.yaml`, `src/so101_rl/env/batched_env.py` (hand-off from SU1: the `randomizer` hooks only — `sample_episode` on reset, `set_world_params`, proprio noise, tracker bias/latency, die size/zone offset), `tests/test_domain_rand.py`
- **Specialist:** simulation-to-hardware-bridge
- **Dependencies:** SU1, SU2 (`SimParams`, `BatchedSimParams`)
- **Steps:** FS8. `dr_stages.yaml` stages 0–5 exactly in the plan's priority order with the specialist's ranges: 1 tracker (σ 1.5/3 mm → up to 3/6 mm, bias 3 mm, rot 1°, latency 1–3, dropout 2–10 %, occlusion radius ±20 %); 2 servo ±25 % around the centre (or ±2σ of the fit spread when larger), delay +0..+3 ticks on top of the fitted delay so the total spans 2–5; 3 backlash 0–6 ticks hysteresis + proprio ±2 ticks; 4 die mass ±40 %, half edge ±0.5 mm, friction 0.5–1.5×, gripper kp/kv ±40 % (closing speed), zone ±5 mm; 5 lighting flag only. Die size changes need per-world `geom_size` — if MJWarp 3.14 does not batch `geom_size`, approximate with mass/friction only and record it. Backlash as target hysteresis: the ctrl written to the actuator moves only when the goal leaves a ±`deadband` band around the last written value (decision D-BACKLASH).
- **Tests:** `test_domain_rand.py::test_stage0_is_identity`, `::test_ranges_within_spec` (1e4 draws per stage), `::test_per_episode_params_change_only_on_reset`, `::test_servo_params_centred_on_sim_params_yaml`, `::test_unidentified_params_get_wide_ranges` (spread 0.8 → range ≥ ±50 %), `@gpu ::test_env_applies_dr_per_world` (two worlds, kp high vs low → different step responses in one episode), `@gpu ::test_backlash_hysteresis`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_domain_rand.py -q -m "not gpu"` → pass; `pixi run pytest tests/test_domain_rand.py -q -m gpu` → pass.
- **Est.:** 3 agent-h. `parallelizable-via-worktree: true` (with SU4 — SU4 codes against the FS8 `DomainRandomizer` protocol and a no-op stage-0 randomizer until SU3 lands)

#### Task S3.2 (SW): `CurriculumController` + ARCHITECTURE sync
- **Files:** `src/so101_rl/rl/curriculum.py`, `tests/test_curriculum.py`, `docs/ARCHITECTURE.md` (§4, §7 DR stages table, §10 row)
- **Dependencies:** S3.1
- **Tests:** `::test_advances_at_70pct_after_window`, `::test_never_advances_below_threshold`, `::test_min_episodes_per_stage`, `::test_history_recorded`, `::test_stops_at_max_stage`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_curriculum.py -q` → pass.
- **Est.:** 1 agent-h.

**Commit after S3.2:** `S3: domain randomisation stages around the sysid centre, curriculum controller`.

---

### Unit SU4 — S4: massively parallel SAC, gates, eval, rollout logging (wave 2 software; wave 3 training runs)

#### Task S4.1 (SW): GPU replay and `FastSAC` with unit tests on a toy batched env
- **Files:** `src/so101_rl/rl/replay_gpu.py`, `src/so101_rl/rl/fast_sac.py`, `src/so101_rl/rl/toy_batched_env.py` (torch point-mass goal env with the same `BatchedStep` contract, CPU-capable), `tests/test_replay_gpu.py`, `tests/test_fast_sac.py`
- **Specialist:** lerobot-specialist (SAC recipe); cuda-performance-engineer (compile / graphs)
- **Dependencies:** SU0 (FS3/FS9 contracts), independent of SU1 for the toy env
- **Steps:** FS10. Actor exactly SB3's shape (`latent_pi` Sequential of Linear+ReLU, `mu`, `log_std` heads; tanh-Gaussian with the log-prob correction); critics with LayerNorm; α auto-tuning with clamped log-α; targets `r + γ^n (1 − terminated) · (min Q' − α log π')` — `truncated` **does not** zero the bootstrap; n-step returns computed in the replay across the time axis with episode-boundary masks; `grad_steps_per_env_step` updates per vector step; optional `torch.compile(mode="reduce-overhead")` on the update (off by default; a flag). All tensors on `device`, no host syncs inside `update` except the metrics every 100 updates. `rl/` imports nothing from `safety/`.
- **Tests:** `test_replay_gpu.py::test_ring_overwrite`, `::test_final_obs_used_for_reset_worlds`, `::test_n_step_masks_episode_boundaries`; `test_fast_sac.py::test_tanh_logprob_matches_numerical`, `::test_truncation_bootstraps` (constant-reward toy: value at the last step equals mid-episode value within 5 %), `::test_terminal_does_not_bootstrap`, `::test_actor_state_dict_keys_match_sb3` (constructs `stable_baselines3.SAC("MlpPolicy", ...)` on a dummy Box env with `net_arch=[256,256,256]` and asserts `set(actor.state_dict()) == set(sb3.actor.state_dict())` with equal shapes), `::test_learns_toy_reach_cpu` (`@slow`: 64 toy worlds on CPU, success ≥ 90 % within 3 min), `::test_alpha_bounds`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_replay_gpu.py tests/test_fast_sac.py -q -m "not slow"` → pass; `pixi run -e cpu pytest tests/test_fast_sac.py -q -m slow` → pass < 5 min.
- **Est.:** 5 agent-h. `parallelizable-via-worktree: true`

#### Task S4.2 (SW + GPU): Feasibility (L1) and predicate (L2) gates in sim — before any reward tuning
- **Files:** `src/so101_rl/rl/gates.py`, `src/so101_rl/rl/scripted.py` (tick-space proportional / Jacobian-pseudo-inverse controllers for reach and push), `tests/test_gates.py`
- **Specialist:** testing-specialist; kinematics-specialist (reachability)
- **Dependencies:** SU1 (batched env), S0.4
- **Steps:** `pixi run sim-gates --task {reach,die_to_zone} --out runs/gates/<task>-<ts>.json` runs and prints a PASS/FAIL table:
  - **L1 reach:** FK over 2e5 random in-limit joint vectors → nearest-FK distance ≤ 0.01 m for 100 % of 1e4 sampled goals and ≥ `success_radius` margin to the joint limits; travel budget `max_delta × episode_steps ≥ 2 ×` the largest goal-to-start joint distance; scripted controller (tick-quantised, obs noise stage 1) succeeds ≥ 95 % of 1024 worlds within 50 steps.
  - **L1 die:** start well and the whole zone reachable at graspable height (tip z ≤ table + 0.03 m within 0.01 m) and the pushing approach (tip at table + 0.012 m on the far side of the die) reachable; scripted push controller achieves the predicate ≥ 30 % (the Phase 4 scripted-seed bar) and the scripted grasp-and-place ≥ 30 %; die start distribution never inside the zone and never under the arm's home footprint.
  - **L2 (both):** the adversarial-state table (tip hovering over the zone without the die; die moving through the zone at > 0.05 m/s; die held in the air above the zone (> 16 mm above the goal, e.g. 25 mm); die slid out one step after success; die at zone edge ± 1 mm; die in the zone but tracker-invisible; wrong height) → no success in every case except the genuine ones. Static states are judged by the instantaneous predicate `success_*`; the motion cases (moving through at > 0.05 m/s, nudged/slid out) by the episode-level success `debounce_*` with the settle rule (2 consecutive visible in-zone steps and ≤ 3 mm tracker displacement between them, plan §10 2026-10-06), since the per-step predicate cannot see motion; random-policy false-accept over 1e5 episode-steps < 0.1 % and do-nothing = 0; an independent second predicate implementation (plain numpy, written separately in `gates.py`) agrees with `tasks.success_np` on 1e5 random states.
  Each gate is also a pytest; each has a broken-input twin (e.g. a goal sampler radius of 0.40 m fails L1; a predicate without the visibility test fails L2).
- **Tests:** `test_gates.py::test_l1_reach_passes`, `::test_l1_reach_fails_with_radius_040`, `::test_l1_die_passes`, `::test_l2_adversarial_table[case]`, `::test_l2_random_false_accept`, `::test_l2_independent_predicate_agrees`, `::test_l2_fails_without_visibility_rule`, `@gpu ::test_scripted_reach_ge_95pct`, `@gpu ::test_scripted_push_ge_30pct`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_gates.py -q -m "not gpu"` → pass; `pixi run sim-gates --task reach` → exit 0, all rows PASS; `pixi run sim-gates --task die_to_zone` → exit 0 (if the scripted push bar fails, the task is **not** trained: report to the user — L1).
- **Est.:** 4 agent-h + 0.3 GPU-h.

#### Task S4.3 (SW + GPU): Training loop, checkpoints, rollout logging, renders, sim eval + exit gate, policy export
- **Files:** `src/so101_rl/rl/train_sim_sac.py`, `src/so101_rl/rl/rollout_logger.py`, `src/so101_rl/rl/eval_sim.py`, `src/so101_rl/rl/policy_export.py`, `tests/test_train_sim_sac.py`, `tests/test_rollout_logger.py`, `tests/test_eval_sim.py`, `tests/test_policy_export.py`
- **Specialist:** lerobot-specialist; cuda-performance-engineer
- **Dependencies:** S4.1, S4.2, SU1, **SUV** (the training loop creates a `viz.TrainingMonitor` from `--viewer/--video-every/--tensorboard` flags and calls `on_step`, `on_metrics`, `maybe_record`; renders go through `viz/video.py`, which supersedes `rl/render.py`); SU3 optional at start (stage 0)
- **Steps:**
  1. `pixi run train-sim-sac --task reach --nworld 1024 --budget-min 60 --dr-start-stage 0 --dr-max-stage 4 --out runs/sac-sim/<task>-<ts>/ [--sim-params calib/sim_params.yaml] [--seed] [--resume CKPT]`: `torch.cuda.set_per_process_memory_fraction(0.6)`; env + randomizer + curriculum; warm-up random actions; loop with `grad_steps_per_env_step` updates; every `eval_every` env-steps a deterministic eval on 1000 episodes at the **current** DR stage (`eval_sim.evaluate`), logging to `metrics.csv` (env-steps/s, grads/s, success, final error p50/p90, α, entropy, Q mean/max, critic loss, action saturation fraction, clamp-active fractions, tracker visible fraction, dr stage, VRAM); checkpoint `ckpt_<step>.pt` every eval and `ckpt_last.pt`; `RolloutLogger` writes rollouts (mix as per the specialist: a `policy_tag` per chunk: 0 random/warm-up, 1 mid-training, 2 final — the logger samples every k-th vector step so the total stays 1–5 M transitions); an OOM is **not** caught (fails loudly, L12); exits non-zero if `metrics.csv` has < 3 rows or no checkpoint exists.
  2. **Review triggers** (printed and written to `review.json`, never auto-switching the method — plan §5): UTD < 0.005 grads/transition; α < 1e-4 or entropy > target + 2 for 100k steps; |Q| > 2 · r_max/(1 − γ); critic loss rising over 3 evals; saturation > 50 %; reach success < 50 % after 1 GPU-h or no improvement over 3 evals; die success < 10 % after 4 GPU-h; curriculum stuck at a stage for > 2 GPU-h.
  3. `eval_sim.py`: `pixi run eval-sim --ckpt PATH --task T --episodes 1000 --dr-stage 4 --render 16 --out runs/eval-sim/<ts>/` → `eval.json` (success rate, Wilson 95 %, final error, trips, clamp stats, tracker stats) + `renders/*.mp4` (16 rollouts: 8 random, 4 highest-return, 4 shortest — the gaming audit) + `review_checklist.md` (predicate-gaming signs to look for: hovering, sliding through, releasing then re-touching, die at the zone edge). **Exit gate** (plan §5 S4): `success ≥ 0.95` over 1000 episodes at the final DR stage with the lower Wilson bound ≥ 0.93 **and** a human ticks the rendered-rollout checklist; `eval-sim --gate` returns exit 0 only when the numeric part passes.
  4. `policy_export.py`: FS10 export/load; `pixi run export-policy --ckpt ... --out runs/sac-sim/<run>/policy_export.pt`.
- **Tests:** `test_train_sim_sac.py::test_smoke_toy_env_cpu` (`@slow`: 200 steps on the toy batched env write metrics, ckpt, rollouts), `::test_exit_nonzero_without_artifacts` (budget 0 → exit 1), `@gpu ::test_smoke_reach_200_steps` (writes everything, no NaN, VRAM < 4 GB); `test_rollout_logger.py::test_chunk_layout_matches_tdmpc2_to_td` (row 0 NaN action/reward, shapes `[E, T+1]`), `::test_action_is_applied_not_sampled`, `::test_check_chunk_fails_on_nan_mid_episode`; `test_eval_sim.py::test_wilson`, `::test_gate_fails_at_94pct`, `::test_renders_written` (CPU renderer on a stub env); `test_policy_export.py::test_roundtrip`, `::test_refuses_wrong_layout_version`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_rollout_logger.py tests/test_eval_sim.py tests/test_policy_export.py tests/test_train_sim_sac.py -q -m "not gpu"` → pass; `pixi run pytest tests/test_train_sim_sac.py -q -m gpu` → pass; `pixi run train-sim-sac --task reach --nworld 1024 --budget-min 2 --out runs/sac-sim/smoke` → exit 0 with `metrics.csv` (≥ 3 rows), `ckpt_last.pt`, `rollouts/chunk_0000.pt`, and reports env-steps/s ≥ 4000 (incl. learning).
- **Est.:** 6 agent-h + 0.5 GPU-h.

#### Task S4.4 (SW): SU4 ARCHITECTURE sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§3: the trainer process for sim pre-training with its VRAM cap; §4 `rl/` modules; §7 "S4 built"; §8 tasks; §10 row), `README.md` (entry points)
- **Est.:** 0.3 agent-h.

**Commit after S4.4:** `S4: massively parallel SAC on the batched env, L1/L2 gates, sim eval + exit gate, rollout logging, policy export`.

#### Task S4.5 (GPU RUN): Reach pre-training to the exit gate
- **Files (artifacts, gitignored):** `runs/gates/reach-*.json`, `runs/sac-sim/reach-<ts>/`, `runs/eval-sim/reach-<ts>/`; **committed:** `plans/reviews/<date>-s4-reach-run.md` (1 page: config, curve, stages, eval, checklist ticks, renders reviewed)
- **Dependencies:** S4.3, SU3, `calib/sim_params.yaml` (v0 prior is acceptable for the first run; the run is **repeated** after HS3 delivers v1 — the v1 run is the one deployed)
- **Steps:** `pixi run sim-gates --task reach` → `pixi run train-sim-sac --task reach --nworld 1024 --budget-min 90` → `pixi run eval-sim --ckpt runs/sac-sim/reach-<ts>/ckpt_last.pt --task reach --episodes 1000 --dr-stage 4 --render 16 --gate` → human reviews `renders/` + `review_checklist.md` → `pixi run export-policy`. If a review trigger fires, stop and write the review note; at most 3 attempts (seed/hyper-parameter changes recorded) before a user decision.
- **Acceptance:** `eval.json: success ≥ 0.95, wilson_lo ≥ 0.93` at stage 4; `policy_export.pt` written; review note committed.
- **Est.:** 1.5 agent-h + 2–4 GPU-h (budget cap 6 GPU-h over all attempts). Wall-clock estimate per the plan: 10–30 min per attempt of pure training; expect 30–60 min with evals.

#### Task S4.6 (GPU RUN): Die-to-zone pre-training to the exit gate
- **Files:** `runs/gates/die_to_zone-*.json`, `runs/sac-sim/die-<ts>/`, `runs/eval-sim/die-<ts>/`, `plans/reviews/<date>-s4-die-run.md`
- **Dependencies:** S4.5 passed, HS3 (sim_params v1 incl. the gripper/die segments — the die task is **not** trained on the v0 prior), H9 measured `die_start` distribution preferred (else the placeholder, recorded)
- **Steps:** as S4.5 with `--task die_to_zone --nworld 2048 --budget-min 480` (VRAM 1.3 GB env + replay 2 M ≈ 1.5 GB — within the cap); review triggers as in S4.3; the push-vs-place behaviour is recorded in the review note (both count, plan Phase 4).
- **Acceptance:** `success ≥ 0.95, wilson_lo ≥ 0.93` over 1000 episodes at stage 4; renders reviewed (no predicate gaming); `policy_export.pt`.
- **Est.:** 2 agent-h + 8–24 GPU-h (budget cap 30 GPU-h across ≤ 3 attempts; then a user review of the residual-RL fallback per §10).

---

### Unit SR2 — HER, TD-MPC2 adapter, state world model (Phase 0 U9 remainder, wave 2; parallel)

Tasks 9.1 and 9.3 of `plans/phase0-implementation-plan.md` apply **unchanged** except: `TdMpc2GoalEnv` exposes `obs_shape["state"] == (policy_input_dim(task.obs),)` (24 reach / 29 die, FS3) instead of `(18,)`; `wm/dataset.py: EpisodeDataset` gains `from_rollout_chunks(dir)` reading FS10 chunks (`check_chunk` on every file first, L12) so the state WM trains on S4 rollouts first (plan §5 critical path item 4). Files, tests and acceptance as in Phase 0 (`tests/test_her.py`, `tests/test_tdmpc2_env.py`, `tests/test_state_wm.py` + `::test_dataset_from_rollout_chunks`). Task 9.2 moved to SU5.

- **Est.:** 4 agent-h total. `parallelizable-via-worktree: true`
- **Commit after SR2:** `U9 remainder: HER relabel, TD-MPC2 goal-env adapter (FS3 state), state WM on sim rollouts`.

---

### Unit SU5 — S5: deployment, zero-shot eval, fine-tuning bridges, gap report (wave 3 software)

#### Task S5.1 (SW): `deploy_policy` — the same script against `servo --backend mujoco` and the real arm
- **Files:** `src/so101_rl/rl/deploy_policy.py`, `tests/test_deploy_policy.py`
- **Specialist:** real-time-control-specialist (shm client timing); simulation-to-hardware-bridge (ladder)
- **Dependencies:** SR1, S4.3 (export), U5 object table, U6 logger
- **Steps:** FS11. 10 Hz loop on `ShmBackend.wait_policy_tick`; FS3 obs via `build_policy_obs_np` from the state record (+ FK) and the object table (die task); actor on CPU (`torch.set_num_threads(1)`; inference < 2 ms, measured and logged); action × `--speed-scale` (ladder: 0.5 → 0.75 → 1.0, each step requires the previous eval's lower Wilson bound ≥ 0.6 or a human override flag `--force-speed`); goals from the same `BoxGoalSampler` with a fixed `--seed` so sim and real evaluate the **same goal list** (the sim twin: `eval-sim --goals-from runs/<real-run>/goals.json`); `FiducialCheck` every episode in `shm` mode (halt on failure, plan Phase 0 camera-moved gate); episodes logged with `EpisodeLogger` (applied action), `check_episode` on every file at the end; halt rules: any trip other than clamps, 3 consecutive failures in the first 10 episodes, tracker UNKNOWN > 50 % of steps in the die task; `eval.json` with the Wilson interval. **`--backend sim` launches `python -m so101_rl.servo.main --backend mujoco --shm-name deploy-<pid> --calibration tests/data/...` as a child and never touches serial; `--backend shm` requires `--attended` and `runs/hw/tier.json` (tier ≥ 0 attended; tier ≥ 1 for `--episodes > 50` or without `--attended`).**
- **Tests:** `test_deploy_policy.py::test_sim_backend_end_to_end` (`@slow`: a scripted "actor" stub that moves toward the goal, 3 episodes through the mujoco servo child → 3 `.h5`, `eval.json` with `n == 3`), `::test_refuses_shm_without_attended` (no `serial.Serial` constructed — monkeypatched to raise), `::test_speed_ladder_blocks_without_prior_eval`, `::test_same_goal_list_as_eval_sim`, `::test_obs_from_state_record_matches_batched_builder`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_deploy_policy.py -q` → pass; `pixi run -e cpu deploy-policy --policy runs/sac-sim/smoke/policy_export.pt --backend sim --task reach --episodes 5 --seed 1 --run-id deploy-sim-smoke` → exit 0, `runs/deploy-sim-smoke/{episodes/*.h5, eval.json, goals.json}`; the exported **reach** policy from S4.5 through this path on the CPU `servo --backend mujoco` process reaches ≥ 90 % over 50 episodes (`eval.json`) — the end-to-end proof that the policy survives the real servo code path (replaces the Phase 3 "≥ 90 % in MuJoCo" gate).
- **Est.:** 4 agent-h.

#### Task S5.2 (SW): SB3 SAC + HER fine-tuning from the sim actor; TD-MPC2 offline pre-training and warm-start
- **Files:** `src/so101_rl/rl/sb3_bridge.py`, `src/so101_rl/rl/train_sac_her.py`, `src/so101_rl/rl/tdmpc2_offline.py`, `src/so101_rl/rl/train_tdmpc2.py`, `tests/test_sb3_bridge.py`, `tests/test_train_entrypoints.py`, `tests/test_tdmpc2_offline.py`
- **Specialist:** lerobot-specialist
- **Dependencies:** SR1, SR2, S4.3, S5.1
- **Steps:** FS11. `train-reach-sac --init-from policy_export.pt --backend {sim,shm} --total-steps N --warmup-frozen-actor 2000 --baseline-eval-every 1000`: SB3 `SAC("MultiInputPolicy", env, policy_kwargs=dict(net_arch=[256,256,256]), replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"), ent_coef="auto_0.01", learning_rate=1e-4, learning_starts=2000, batch_size=256, gamma=task.gamma, tau=0.005)`; `load_actor_into_sb3`; `FrozenActorWarmup(2000)`; `SimBaselineEval` keeps the frozen sim actor as the baseline and rolls back the checkpoint if fine-tuned success < baseline − 0.15 over two consecutive evals; `handle_timeout_termination` left on (truncation); artifacts `runs/sac-her/<ts>/{model.zip, monitor.csv, baseline.json}`, exit non-zero if missing. `pretrain-tdmpc2`: custom single-task loop (vendored `OfflineTrainer` is **not** used, decision D-TDMPC2-LOOP), `Buffer.load` per chunk, `agent.update(buffer)` for `--steps`, validation = reward/value prediction error on a held-out chunk + a 50-episode MPC rollout on `SimBackend` (CPU, slow) reported in `metrics.csv`. `train-reach-tdmpc2 --init-from runs/tdmpc2-offline/<ts>/model.pt --backend {sim,shm} --seed-steps 500 --sim-replay-fraction 0.15 --lr 1e-4`.
- **Tests:** `test_sb3_bridge.py::test_actor_copy_exact` (sim actor and SB3 actor give identical deterministic actions on 100 obs), `::test_frozen_warmup_keeps_actor`, `::test_rollback_rule`; `test_train_entrypoints.py` (`@slow`: 300-step runs of both entry points on `SimBackend` write their artifacts; `--total-steps 0` → non-zero — L12); `test_tdmpc2_offline.py::test_batch_shape_matches_online_trainer` (one sampled batch has exactly the shapes `OnlineTrainer` would feed `agent.update`), `::test_pretrain_200_updates_cpu` (`@slow`, metrics + model written), `::test_exit_nonzero_on_empty_dir`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_sb3_bridge.py tests/test_tdmpc2_offline.py -q -m "not slow"` → pass; `pixi run -e cpu train-reach-sac --backend sim --init-from runs/sac-sim/smoke/policy_export.pt --total-steps 300` → exit 0 with artifacts; `pixi run -e cpu pretrain-tdmpc2 --data-dir runs/sac-sim/smoke/rollouts --task reach --steps 200 --device cpu` → exit 0, `model.pt` + `metrics.csv` (≥ 3 rows); `pixi run -e cpu train-reach-tdmpc2 --backend sim --init-from runs/tdmpc2-offline/<ts>/model.pt --steps 300` → exit 0.
- **Est.:** 6 agent-h (+ 1 GPU-h for a 200k-update offline pre-train on the real reach rollouts, run in HS4 prep).

#### Task S5.3 (SW): Gap report + ARCHITECTURE sync
- **Files:** `src/so101_rl/rl/gap_report.py`, `tests/test_gap_report.py`, `docs/ARCHITECTURE.md` (§3 policy process as built: `deploy_policy` at 10 Hz over shm, obs construction, speed ladder; §4 `rl/`; §7 "S5 built"; §10 row), `README.md`
- **Dependencies:** S5.1, S5.2
- **Steps:** FS11 `GapReport` from a real `deploy-policy` run and its sim twin: Wilson intervals, final-error quantiles, per-joint tracking-error KS statistics (from the tick logs: `|target − present|` distributions), clamp-active fractions per stage (sim from `BatchedGoalEnv.stats`, real from `clamped_mask` in the state records), delay estimate by cross-correlating target and present per joint, `replay_in_sim_rms_pct` (the real episodes replayed through `compare_real --sim-params`), tracker UNKNOWN fractions, interventions per hour (from the run's `interventions.json` written by the attendant via `deploy-policy --note`). Markdown summary next to the json.
- **Tests:** `test_gap_report.py::test_report_from_synthetic_runs`, `::test_delay_estimate_recovers_3_ticks`, `::test_fails_on_missing_sim_twin`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_gap_report.py -q` → pass; `pixi run -e cpu gap-report --real runs/deploy-sim-smoke --sim runs/eval-sim/<ts> --out runs/gap/smoke.json` → exit 0 (sim-vs-sim gap ≈ 0).
- **Est.:** 2 agent-h.

**Commit after S5.3:** `S5: policy deployment over shm, zero-shot eval protocol, SB3/TD-MPC2 fine-tuning bridges, sim-to-real gap report`.

---

### Unit SR3 — Milestone scripts (Phase 0 U10 remainder, wave 3; parallel with SU5)

Phase 0 Tasks 10.1 and 10.2 apply with these changes: 10.1 lives at `src/so101_rl/env/random_policy.py` (`--backend {sim,real}`; `real` requires `--attended` + tier) matching the existing pixi tasks; 10.2 `scripts/check_milestone_phase0.py` adds rows `mujoco-version-pinned` (parses `pixi list --json -e cpu` and `-e default`: same mujoco 3.14.x and torch 2.11.x), `parity-green` (latest `runs/parity/*.json` with zero Layer A mismatches and Layer B/C within tolerance — NOT-RUN until the GPU test ran), `sim-params-verified` (`calib/sim_params.yaml` version ≥ 1 with every `heldout_rms_pct ≤ 5`), and reads `runs/gates/*.json` for L1/L2. Each new row has a broken-fixture test.

- **Files:** `src/so101_rl/env/random_policy.py`, `scripts/check_milestone_phase0.py`, `tests/test_random_policy.py`, `tests/test_check_milestone.py`
- **Acceptance:** `pixi run -e cpu random-policy-sim --episodes 3` → exit 0, 3 `.h5` all ok; `pixi run -e cpu pytest tests/test_check_milestone.py -q` → pass; `pixi run -e cpu check-milestone` → exit 1 with NOT-RUN rows for hardware items (expected).
- **Est.:** 2.5 agent-h. `parallelizable-via-worktree: true`
- **Commit after SR3:** `Phase 0 milestone script and gate checker, with sim-track rows`.

---

### Hardware-gated sim-track tasks (HS-series; all **HW**, Tier 0 attended unless stated)

Each has a procedure section appended to `docs/HARDWARE_CHECKLIST.md` (append-only, by SU2's and SU5's sync tasks) and produces artifacts under `runs/`. The H-ids are the Phase 0 hardware tasks (`docs/HARDWARE_CHECKLIST.md`).

| Id | Task | Needs (hardware) | Needs (software) | Procedure (abridged) | Artifact / acceptance | Est. (human + GPU) |
|---|---|---|---|---|---|---|
| **HS1** | Reach-space excitation run on the real arm | H2 (power path + e-stop test), H3 (deadman), H5 (rest pose), H7 (calibration read-back); H6 recommended | S2.2 | Foam under the arm, hand on the e-stop. `REQUIRE_RT=1 pixi run servo --backend feetech --heartbeat-port ... --tick-log runs/sysid/<ts>/tick_log.bin` in one terminal; `pixi run sysid-excite --backend real --attended --plan default --out runs/sysid/<ts>` in another. Watch temperatures (the 60 °C trip ends the segment; resume after cooldown). | `runs/sysid/<ts>/{episodes/*.h5, tick_log.bin, plan.json}`; every `check_episode` ok; `plan.json` clamp-active in [0.05, 0.4] for the multisine/chirp segments (else lower the amplitudes and repeat) | 1.5 h |
| **HS2** | Gripper + die excitation | HS1 done; H9 (tray) optional | S2.2 `--with-gripper-die` | Die in the start well; gripper open/close cycles at three speeds, closing on the die and on air; 10 push strokes with a scripted tick-space controller (`rl/scripted.py`, speed scale 0.5) | `runs/sysid/<ts>-grip/` as above | 1 h |
| **HS3** | Sysid fit → `calib/sim_params.yaml` v1 (software after HS1/HS2) | — | S2.3 | `pixi run sysid-fit --run runs/sysid/<ts> --backend warp --heldout-kinds chirp,random --out calib/sim_params.yaml`; inspect `spread`; if any joint's held-out RMS > 5 %: widen `fit` (add `damping`, `forcerange`), repeat once; if still failing, record it and treat that joint's parameters as unidentified (wide DR) — a documented negative result counts (plan §1 Stage A bar). Commit the yaml. Re-run S4.5 on v1. | `calib/sim_params.yaml` v1 with `heldout_rms_pct ≤ 5` per joint; `runs/hw/sysid-<ts>.json` (H12's artifact, now from the fitted model) | 0.5 h + 1–2 GPU-h |
| **HS4** | Zero-shot reach evaluation | H2, H3, H5, H7, H11 (safe-stop timing); H10 soak recommended | S4.5 (exit gate on v1 params), S5.1 | Speed ladder 0.5 → 0.75 → 1.0, 10 episodes each, then `--episodes 50 --speed-scale 1.0 --seed 7`; goals from `BoxGoalSampler(seed 7)`, the same list as `eval-sim --goals-from`; halt rules per S5.1; attendant notes interventions. Then `pixi run gap-report`. | `runs/deploy-reach-<ts>/eval.json` with `n = 50` and the Wilson interval; `runs/gap/reach-<ts>.json`. **Proceed to HS5 when `wilson_lo ≥ 0.60`; below that, the gap report drives a sysid/DR revision (S3 stage ranges or a second excitation run), not real fine-tuning.** | 2 h |
| **HS5** | Real fine-tuning of reach: SAC + HER, then TD-MPC2 (offline-pretrained) | HS4; Tier 0 sessions of ≤ 20 min on / 10 min off until H13 (Tier 1) allows unattended runs | S5.2; `pretrain-tdmpc2` run on the reach rollouts (1 GPU-h) | `pixi run train-reach-sac --backend shm --init-from ... --total-steps 60000 --attended`; baseline eval every 1000 steps (N = 10, same goals) with rollback; then `train-reach-tdmpc2 --backend shm --init-from runs/tdmpc2-offline/...`. Budget per plan §5: 8 h of RL per algorithm; review trigger: success below 10 % after 40k steps or no drop in mean final distance over 3 checkpoints. | `runs/sac-her/<ts>/`, `runs/tdmpc2/<ts>/`; the Phase 3 milestone eval (N = 100 goals, lower 95 % bound > 85 %) run with `deploy-policy` on the fine-tuned actors; `runs/gap/` updated — the **Stage A research result** (sim → real gap per algorithm) | 2 × 8 h arm time (mostly unattended after H13) + 2 h human |
| **HS6** | Zero-shot die-to-zone evaluation | HS4 passed; H8 (hand-eye ≤ 3 mm, fiducial), H9 (tray ≥ 19/20), Phase 1 verifier false-accept ≤ 2 % on labelled snapshots (L2) | S4.6 exit gate, S5.1 die mode, `die_start` updated from H9 | Ladder as HS4 but `--episodes 30`; the attendant resets nothing (the tray does); halt if tracker UNKNOWN > 50 % of steps | `runs/deploy-die-<ts>/eval.json` (n = 30, Wilson), `runs/gap/die-<ts>.json`; record push vs place counts | 2 h |
| **HS7** | Real fine-tuning of die-to-zone | HS6; H13 (Tier 1) | S5.2 with `task=die_to_zone` | As HS5 with the Phase 4 budget (15 h, review trigger: zero verified successes after 30k real steps) | Phase 4 milestone: > 70 % over N = 50 | 15 h arm time + 3 h human |

Rough hardware/human total for the sim track: **≈ 12 attended human-hours** plus ≈ 30 h of mostly unattended arm time (after H13), on top of the Phase 0 H-track's ≈ 20 h.

## Budgets and review triggers (summary)

| Item | Budget | Trigger → manual review (never an automatic method switch, plan §5) |
|---|---|---|
| S0 gate | 0.5 GPU-h | throughput < 500k env-steps/s at N=1024, VRAM > 1.5 GB, parity > 1e-6 rad, any overflow at njmax 256, or Layer C grasp parity fails → ManiSkill3 fallback discussion |
| S2 fit | 2 GPU-h per attempt, 2 attempts | held-out RMS > 5 % on any joint after the second attempt → record as a negative result, widen DR |
| S4 reach | ≤ 6 GPU-h total, ≤ 3 attempts | success < 50 % after 1 GPU-h; no improvement over 3 evals; any S4.3 learner trigger |
| S4 die-to-zone | ≤ 30 GPU-h total, ≤ 3 attempts | success < 10 % after 4 GPU-h; curriculum stuck > 2 GPU-h at a stage; learner triggers → then the residual-RL fallback is *discussed* (§10) |
| HS4/HS6 zero-shot | 2 h arm time each | `wilson_lo < 0.60` → sysid/DR revision loop (one iteration budgeted: HS1 repeat 1.5 h + 2 GPU-h), not fine-tuning |
| HS5/HS7 fine-tuning | 8 h per algorithm (reach), 15 h (die) | plan §5 Phase 3/4 triggers |
| Thermals | — | GPU > 85 °C or host throttling during multi-hour sim load (plan §2 laptop question) → pause and log |

## Execution Strategy

- **Wave 0 (sequential):** SU0 S0.1 → S0.2 → S0.3 → S0.4. Everything else codes against SU0's frozen files (FS1–FS4) and the installed pins.
- **Wave 1 (parallel, 3 units):** SU1 (batched env), SU2 (sysid software), SR1 (real env). No shared files; `docs/ARCHITECTURE.md` only in each unit's final sync task (append-only).
- **Wave 2 (parallel, 3 units):** SU3 (DR), SU4 software (S4.1–S4.4; codes against the FS8 protocol with a stage-0 no-op randomizer until SU3 merges), SR2 (HER/TD-MPC2 adapter/WM).
- **Wave 3:** S4.5 reach run (GPU) ‖ SU5 software ‖ SR3. S4.6 die run after HS3 delivers `sim_params.yaml` v1.
- **HS track (interleaved):** HS1/HS2 as soon as SU2 and H2/H3/H5/H7 are done; HS3 right after; S4.5 re-run on v1; HS4 after SU5 + S4.5; HS5 after HS4 (Tier 0 sessions, Tier 1 after H13); S4.6 → HS6 → HS7.
- **Always-parallel:** quality review + test generation per unit (executor's standard step). GPU tests run via `pixi run test-gpu` on this host only; CI without a GPU runs `pixi run -e cpu ci`.

Rough software effort: **≈ 70 agent-hours** (SU0 8.5, SU1 12.5, SU2 10.5, SR1 6, SU3 4, SU4 software 15.5 + runs 3.5, SR2 4, SU5 12, SR3 2.5), **≈ 35–45 GPU-hours** (S0/S1/S2 gates ≈ 3, reach ≤ 6, die ≤ 30, offline TD-MPC2 ≈ 1, sysid fits ≈ 3), **≈ 12 attended human-hours + ≈ 30 h arm time**. Critical path to the first real result (HS4): SU0 → SU1 → SU4 software → S4.5 → SU5.1 → HS4 ≈ 8.5 + 12.5 + 15.5 + 1.5 + 4 ≈ 42 agent-h of that, plus HS1–HS3 for the v1 parameters.

## Git Commits

One commit per unit (listed after each unit), plus `S4 run notes` commits for the review notes, `Sysid: sim_params v1` after HS3 (edits `calib/sim_params.yaml`, `docs/HARDWARE_CHECKLIST.md` results table), `Phase 3 result: reach sim→real` after HS5 (review note + `so101-self-learning-rl-plan.md` §5 budgets table "measured" column — the master applies plan edits). Every commit that changes a module, process, interface, safety-check order or pixi feature touches `docs/ARCHITECTURE.md` in the same commit (`tests/test_architecture_sync.py`). The master commits; executors stage per task.

## How to Run

Single-session run (default):
1. Open this plan in the executor and proceed wave by wave, unit by unit; run each task's acceptance command before moving on; GPU acceptance in the `default` env on this host.

Parallel run (waves 1–3 carry `parallelizable-via-worktree: true`):
1. Spawn one worktree per parallel unit:
   ```bash
   /worktree create feature/sim-su1-batched-env
   /worktree create feature/sim-su2-sysid
   /worktree create feature/sim-sr1-real-env
   ```
   and for wave 2: `feature/sim-su3-dr`, `feature/sim-su4-sac`, `feature/sim-sr2-her-tdmpc2`; wave 3: `feature/sim-su5-deploy`, `feature/sim-sr3-milestone`.
2. Open a terminal per worktree, `cd` into the printed path, run `claude`, execute that unit's tasks (file sets are exclusive within a wave; `docs/ARCHITECTURE.md` edits are append-only in the final sync task).
3. Merge each wave into `main` before starting the next; the GPU runs (S4.5, S4.6) run from `main`.

This section is suggestive — the user owns the decision. See `commands/worktree.md` and `skills/worktree/SKILL.md`.

## execution_shape

```yaml
execution_shape:
  plan: plans/sim-track-implementation-plan.md
  phase: "Sim-first pre-training track S0-S5 (+ Phase 0 U8/U9/U10 remainder)"
  units:
    - id: SU0
      name: s0-version-pin-warp-backend-interface-freeze
      wave: 0
      depends_on: []
      parallelizable_via_worktree: false
      tasks: [S0.1, S0.2, S0.3, S0.4]
      owns_files:
        - pyproject.toml
        - pixi.lock
        - README.md
        - assets/so101/so101_rl_scene.xml          # margin=0, timestep (hand-off to SU1 in wave 1 for the tray/camera)
        - assets/so101/README.md
        - src/so101_rl/config/schema.py             # + SimWarpConfig (FS1)
        - src/so101_rl/config/loader.py
        - src/so101_rl/config/so101.yaml            # + sim_warp section only (H-track measured values untouched)
        - src/so101_rl/sim/warp_model.py
        - src/so101_rl/env/obs.py
        - src/so101_rl/env/tasks.py
        - src/so101_rl/env/__init__.py              # hand-off to SR1 for the gymnasium registrations
        - scripts/bench_warp.py
        - tests/test_env_sanity.py
        - tests/test_sim_model.py
        - tests/test_config.py
        - tests/test_config_single_source.py
        - tests/test_warp_model.py
        - tests/test_obs_layout.py
        - tests/test_tasks.py
        - plans/phase0-implementation-plan.md       # status note only (supersession table)
    - id: SU1
      name: s1-batched-servo-parity-env
      wave: 1
      depends_on: [SU0]
      parallelizable_via_worktree: true
      tasks: [S1.1, S1.2, S1.3, S1.4]
      owns_files:
        - src/so101_rl/sim/batched_servo.py
        - src/so101_rl/sim/sim_tracker.py
        - src/so101_rl/sim/tray_check.py
        - src/so101_rl/env/batched_env.py           # hand-off to SU3 in wave 2 for the DR hooks
        - assets/so101/so101_rl_scene.xml          # wave-1 hand-off from SU0: tray walls + overhead camera
        - assets/so101/README.md
        - tests/test_batched_servo.py
        - tests/test_sim_tracker.py
        - tests/test_batched_env.py
        - tests/test_batched_parity.py
        - tests/test_tray_check.py
    - id: SU2
      name: s2-sysid-software
      wave: 1
      depends_on: [SU0]
      parallelizable_via_worktree: true
      tasks: [S2.1, S2.2, S2.3, S2.4]
      owns_files:
        - src/so101_rl/sim/sysid/**
        - src/so101_rl/sim/compare_real.py          # hand-off from U4: --sim-params, --tick-log
        - calib/sim_params.yaml                     # version 0 prior; HS3 writes version 1
        - calib/README.md
        - tests/test_sim_params.py
        - tests/test_excitation.py
        - tests/test_sysid_fit.py
        - tests/test_compare_real.py
    - id: SR1
      name: u8-remainder-real-goal-env
      wave: 1
      depends_on: [SU0]
      parallelizable_via_worktree: true
      tasks: [R1.1, R1.2, R1.3]
      owns_files:
        - src/so101_rl/env/backend.py
        - src/so101_rl/env/goal_env.py
        - src/so101_rl/env/goals.py
        - src/so101_rl/env/__init__.py              # wave-1 hand-off from SU0: registrations
        - tests/test_env_backend.py
        - tests/test_goal_env.py
        - tests/test_goals.py
    - id: SUV
      name: training-visualisation
      wave: 1
      depends_on: [SU0]
      parallelizable_via_worktree: true
      tasks: [V.1, V.2, V.3]
      owns_files:
        - src/so101_rl/viz/__init__.py
        - src/so101_rl/viz/live_viewer.py
        - src/so101_rl/viz/video.py
        - src/so101_rl/viz/monitor.py
        - tests/test_live_viewer.py
        - tests/test_video.py
        - tests/test_monitor.py
      # pyproject (tensorboard dep; viz-demo, viz-attach, dashboard tasks) and ARCHITECTURE (§3 viewer process, §4 viz/) are shared writes, applied after the join
    - id: SU3
      name: s3-domain-randomisation-curriculum
      wave: 2
      depends_on: [SU1, SU2]
      parallelizable_via_worktree: true
      tasks: [S3.1, S3.2]
      owns_files:
        - src/so101_rl/sim/domain_rand.py
        - src/so101_rl/sim/dr_stages.yaml
        - src/so101_rl/rl/curriculum.py
        - src/so101_rl/env/batched_env.py           # wave-2 hand-off from SU1: randomizer hooks only
        - tests/test_domain_rand.py
        - tests/test_curriculum.py
    - id: SU4
      name: s4-massively-parallel-sac-gates-eval
      wave: 2
      depends_on: [SU1]
      parallelizable_via_worktree: true
      tasks: [S4.1, S4.2, S4.3, S4.4]
      owns_files:
        - src/so101_rl/rl/replay_gpu.py
        - src/so101_rl/rl/fast_sac.py
        - src/so101_rl/rl/toy_batched_env.py
        - src/so101_rl/rl/gates.py
        - src/so101_rl/rl/scripted.py
        - src/so101_rl/rl/train_sim_sac.py
        - src/so101_rl/rl/rollout_logger.py
        - src/so101_rl/rl/eval_sim.py
        - src/so101_rl/rl/policy_export.py
        - tests/test_replay_gpu.py
        - tests/test_fast_sac.py
        - tests/test_gates.py
        - tests/test_train_sim_sac.py
        - tests/test_rollout_logger.py
        - tests/test_eval_sim.py
        - tests/test_policy_export.py
    - id: SR2
      name: u9-remainder-her-tdmpc2-adapter-state-wm
      wave: 2
      depends_on: [SR1]
      parallelizable_via_worktree: true
      tasks: ["9.1", "9.3"]
      owns_files:
        - src/so101_rl/rl/her.py
        - src/so101_rl/rl/tdmpc2_env.py
        - src/so101_rl/rl/cfg.py
        - src/so101_rl/wm/state_wm.py
        - src/so101_rl/wm/dataset.py
        - src/so101_rl/wm/train_state_wm.py
        - tests/test_her.py
        - tests/test_tdmpc2_env.py
        - tests/test_state_wm.py
    - id: SU4-RUNS
      name: s4-pretraining-runs
      wave: 3
      depends_on: [SU3, SU4]
      parallelizable_via_worktree: false           # GPU runs from main
      tasks: [S4.5, S4.6]
      owns_files:
        - plans/reviews/*-s4-reach-run.md
        - plans/reviews/*-s4-die-run.md
        - runs/gates/**                              # gitignored
        - runs/sac-sim/**
        - runs/eval-sim/**
    - id: SU5
      name: s5-deployment-finetuning-gap
      wave: 3
      depends_on: [SR1, SR2, SU4]
      parallelizable_via_worktree: true
      tasks: [S5.1, S5.2, S5.3]
      owns_files:
        - src/so101_rl/rl/deploy_policy.py
        - src/so101_rl/rl/sb3_bridge.py
        - src/so101_rl/rl/train_sac_her.py
        - src/so101_rl/rl/tdmpc2_offline.py
        - src/so101_rl/rl/train_tdmpc2.py
        - src/so101_rl/rl/gap_report.py
        - tests/test_deploy_policy.py
        - tests/test_sb3_bridge.py
        - tests/test_train_entrypoints.py
        - tests/test_tdmpc2_offline.py
        - tests/test_gap_report.py
    - id: SR3
      name: u10-remainder-milestone-scripts
      wave: 3
      depends_on: [SR1]
      parallelizable_via_worktree: true
      tasks: ["10.1", "10.2"]
      owns_files:
        - src/so101_rl/env/random_policy.py
        - scripts/check_milestone_phase0.py
        - tests/test_random_policy.py
        - tests/test_check_milestone.py
    - id: HS
      name: hardware-gated-sim-track
      wave: "1..4 (interleaved; HS1-HS3 after SU2 + H2/H3/H5/H7; HS4 after SU5 + S4.5; HS5 after HS4 (+H13 for unattended); HS6 after S4.6 + H8/H9; HS7 last)"
      depends_on: [SU2, SU5, SU4-RUNS]
      parallelizable_via_worktree: false
      tasks: [HS1, HS2, HS3, HS4, HS5, HS6, HS7]
      owns_files:
        - calib/sim_params.yaml                     # version 1+ (HS3); never concurrently with SU2
        - src/so101_rl/env/tasks.py                 # ONLY die_start measured values after H9; never concurrently with a software unit
        - runs/sysid/**                             # gitignored
        - runs/deploy-*/**
        - runs/gap/**
        - runs/hw/sysid-*.json
  waves:
    - {wave: 0, units: [SU0]}
    - {wave: 1, units: [SU1, SU2, SR1, SUV]}
    - {wave: 2, units: [SU3, SU4, SR2]}
    - {wave: 3, units: [SU4-RUNS, SU5, SR3]}
  shared_writes:
    - path: docs/ARCHITECTURE.md
      rule: "Updated at the END of every unit (tasks S0.4, S1.4, S2.4, R1.3, S3.2, S4.4, S5.3; SR2/SR3 add a §4/§8 line). Append-only edits to §4 lists, §5/§7 tables and the §10 changelog; bump the version line. Never edited inside a wave-1/2/3 task other than the unit's final sync task."
    - path: pyproject.toml
      rule: "Owned by SU0 in wave 0, which pre-registers every pixi task named in this plan. Later units that need another dependency or task report it back; the integrator applies it between waves (same rule as Phase 0)."
    - path: README.md
      rule: "Environment table rewritten by S0.1. Later units append one line under 'Entry points' per new pixi task in their final sync task."
    - path: so101-self-learning-rl-plan.md
      rule: "The master orchestrator records confirmed decisions from this plan's 'Decisions to confirm' in §10 and updates the §5 budget table's measured values after S4.5/HS4/HS5. No unit edits it."
    - path: plans/sim-track-implementation-plan.md
      rule: "This file. Status notes are appended by the integrator after each wave (as the Phase 0 plan did); frozen-interface changes are edited here together with docs/ARCHITECTURE.md."
    - path: docs/HARDWARE_CHECKLIST.md
      rule: "SU2 (S2.4) and SU5 (S5.3) append the HS1-HS7 procedure sections; the HS track appends a results-table row per artifact."
```

## Decisions made in this plan that the user may want to confirm

1. **D-CHAIN:** the batched env **re-implements** the clamp chain in torch (`sim/batched_servo.py`) instead of vectorising `safety/pipeline.py`; parity is enforced by the frozen three-layer test. Rationale: the safety module stays small, allocation-free and policy-independent; 3072 numpy pipeline calls per policy step would cost ~150 ms vs < 2 ms batched; the chain is elementwise integer math + 5 batched rotations.
2. **D-PIN / D-CUDA:** `mujoco-python 3.14.*`, `mujoco-warp 3.14.*`, `warp-lang ≥ 1.17` in the `sim` feature; `pytorch 2.11.*` in both envs (`pytorch-gpu 2.11.*`, `cuda-version 12.9.*` — fall back to 12.8/12.6 if the solver refuses); `numpy 2.2.*`; `cma` added to `train`. `pixi run test` must be green on `cpu` **and** `default`; new `test-gpu` task.
3. **D-TIMESTEP (default: adopt):** scene `timestep = 1/510 s` so one servo tick is exactly 17 substeps = 1/30 s (today 17 × 2 ms = 34 ms, a 2 % time-scale error in every sim).
4. **D-CONFIG:** a `sim_warp:` section in `config/so101.yaml` and `SimWarpConfig` on `So101Config` (F1 extension), with `overflow_policy` having only the value `fail`.
5. **D-OBS (FS3):** the policy observation adds the **last applied (post-clamp) action** and, for die tasks, the tracker block (last-known xyz, visible flag, age); 21 / 26 dims (+3 goal) superseding F7's 15-D layout; `So101ReachEnv` becomes `So101GoalEnv(task)`. Reward/success definitions live once in `env/tasks.py` for sim and real.
6. **D-REWARD:** dense reward and the success predicate use the **tracker-reported** die position (held last-known when invisible); success needs the die **visible** in the zone for 2 consecutive steps; the privileged truth goes only to the critic.
7. **D-TRIPS:** in sim, NAN / RANGE / BOX_PRESENT trips truncate the episode (as on the real arm); THERMAL, STUCK and protection bits are **not** emulated.
8. **D-TRAY:** the tray is modelled as static funnel/rim geometry in the scene; resets place the die at rest from the measured start-well distribution (no per-reset drop-and-settle); `tray-check` is the sim twin of H9.
9. **D-SYSID:** fit `kp, kv, frictionloss, armature, delay_ticks` (delay as an integer grid), hold `damping` and `forcerange` at the MJCF values unless the held-out gate fails; held-out split by time **and** by excitation kind; parameters with CMA-ES spread > 0.5 are declared unidentified and get wide DR ranges; `calib/sim_params.yaml` is a committed, versioned artifact (v0 = MJCF prior shipped now); CPU multiprocessing is the fallback backend.
10. **D-BACKLASH:** backlash/dead-band is approximated as a per-episode hysteresis band on the commanded actuator target (0–6 ticks), not as a gear model.
11. **D-DR-FREQ:** physics parameters are redrawn per episode per world; tracker noise per frame; tracker pose bias per episode. Stage widening at ≥ 70 % success over 1000 episodes with ≥ 2000 episodes per stage.
12. **D-SAC:** a self-written minimal SAC (`rl/fast_sac.py`) with a GPU ring replay, instead of torchrl's SAC; **D-ACTOR-SHAPE:** the sim actor has SB3's `net_arch=[256,256,256]` ReLU shape so its weights copy into SB3's `MultiInputPolicy` actor for real fine-tuning; critics are asymmetric (privileged) and not transferred.
13. **D-EXIT-GATE:** S4 exit = success ≥ 95 % over 1000 episodes at the final DR stage with Wilson lower bound ≥ 93 % **plus** a human-ticked rendered-rollout checklist; the Phase 3 "≥ 90 % in MuJoCo" gate is replaced by this plus the exported policy reaching ≥ 90 % through the CPU `servo --backend mujoco` + `ShmBackend` path.
14. **D-ZEROSHOT:** N = 50 (reach) / 30 (die) episodes, Wilson 95 % interval, speed ladder 0.5 → 0.75 → 1.0; **proceed to real fine-tuning only if the lower bound ≥ 0.60**, otherwise one budgeted sysid/DR revision loop first.
15. **D-TDMPC2-LOOP:** offline pre-training uses a custom single-task loop over `TDMPC2.update()` and `Buffer.load`, not the vendored multitask `OfflineTrainer`.
16. **D-DIE-PARAMS:** die-to-zone is **not** trained on the v0 (MJCF prior) parameters; it waits for `sim_params.yaml` v1 incl. the gripper/die excitation (HS2/HS3). Reach is trained on v0 first and re-run on v1.
17. **D-SUPERSEDE:** the U8/U9/U10 table above (SimBackend demoted to reference rig, `local_ipc.py` dropped, 9.2 absorbed into S5, `random_policy` as a module, H12 superseded by HS1–HS3).
18. **D-DEPS:** dependency-direction additions recorded in ARCHITECTURE §4: `sim → safety.types, safety.fk (constants only)`, `env/batched_env → sim`, `rl → env`; `rl/` and `env/` never import `safety/` clamps; `safety/` still imports nothing from `rl`, `wm`, `env`.
19. **Budgets:** reach ≤ 6 GPU-h, die ≤ 30 GPU-h (≤ 3 attempts each), zero-shot 2 h arm time each, one sysid revision loop; thermal pause at GPU > 85 °C.

## Gaps and notes for the master orchestrator

- No MuJoCo-Warp or massively-parallel-RL specialist agent exists in the harness; `simulation-to-hardware-bridge`, `lerobot-specialist`, `cuda-performance-engineer` and `real-time-control-specialist` were assigned. Both consulted specialists answered from memory without checking upstream APIs; **every MJWarp 3.14 API name in FS2 must be verified in S0.3 against the installed package** (the local 3.5 package was used for the shape of the API).
- MJWarp per-world batching of `geom_size` (die size DR) is unverified; S3.1 records the outcome.
- The `robot` environment has no `sim` feature and is unaffected by the pin, but it must still solve with `pytorch 2.11.*` (it does today).
- `pixi.lock` regeneration is the only step that needs network access; everything else is offline.
- The spike's grasp-level contact parity (Layer C) is the single most likely S0/S1 gate failure; the fallback (ManiSkill3) is named in plan §6 but **not** planned here (anti-hypertrophy) — it would need its own plan.
- `plans/phase0-implementation-plan.md` keeps its F7 and F11 text; S0.4 adds a status note pointing at this plan's supersession table rather than rewriting them.

## As-built notes: sim wave 1 integration (2026-10-06)

Appended by the integrator (branch `integration/sim-wave1`: main + SU1, SU2, SR1, SUV). Nothing above is rewritten
except the additive "As built" notes under FS7, FS9 and FS12.

- **SU1 (S1.1–S1.3):** `sim/batched_servo.py` (Layer A bit-identical, 20 × 600 ticks), `sim/sim_tracker.py`
  (occluders = jaw boxes + arm-link mesh bounding boxes; 19/20 agreement with the real tracker on rendered frames),
  `env/batched_env.py`, the tray + overhead camera, `sim/tray_check.py` (100/100). Layer B: targets identical on all
  sequences; 3 trajectory xfails accepted (plan §10 2026-10-06). Layer C passes; since the integration it runs on the
  shipped config (box floor −0.015). S1.4 (ARCHITECTURE sync) was applied by the integrator (v0.15).
- **SU2 (S2.1–S2.3):** `sim/sysid/` and `compare_real --sim-params/--tick-log` as planned. Findings that led to the
  2026-10-06 decisions: kp/kv/armature act almost only as ratios through the clamp chain; frictionloss is invisible
  under ±1-tick noise; the 5 % gate could not see a 3-tick delay. Integration: gate 1 %, ratio-group fit + delay,
  `sim_params.yaml` schema 2 (see FS7 note); synthetic recovery: delay exact, group scale within 10 % per joint,
  held-out ≤ 0.07 %; the delay-0 twin fails the gate at 1.5 %. S2.4 applied by the integrator (ARCHITECTURE,
  README, `docs/HARDWARE_CHECKLIST.md` HS1–HS3, H12 superseded note).
- **SR1 (R1.1–R1.2):** `env/backend.py`, `env/goal_env.py`, `env/goals.py`, registrations. Integration: one goal
  sampler for both envs (FS12 note). R1.3 applied by the integrator.
- **SUV (V.1–V.3):** `viz/{monitor,live_viewer,video,demo}.py`. Integration: `tensorboard` (conda-forge) in the
  `train` feature, `viz/dashboard.py` (the `dashboard` task wraps `tensorboard.main` so the task targets a module of
  this package), tasks `viz-demo` (`MUJOCO_GL=egl`), `viz-attach`, `dashboard`. The Warp batch renderer is not used
  (CPU `mujoco.Renderer`; recorded deviation of V.2).
- **Decisions applied (plan §10 2026-10-06):** box floor = table + 5 mm (z −0.015 m) in `so101.yaml` and the
  `WorkspaceBox` default; the rest tip is inside the box; L1 feasibility test `tests/test_die_reach_feasibility.py`
  (jaws reach a resting die at the well and the zone inside the box; none with the old floor). Die position
  convention: centroid everywhere (`tests/test_die_convention.py`). Import-path guard in `tests/conftest.py` (the
  shared editable install pointed at another worktree).
- **Pixi tasks added** (removed from the planned list): `sim-parity`, `tray-check`, `sysid-excite`, `sysid-fit`,
  `viz-demo`, `viz-attach`, `dashboard`.
- **Open:** the `check_milestone_phase0.py` row "held-out RMS ≤ 5 %" (SR3) must use `MAX_RMS_PCT` (1 %); FS8
  `servo_rel` must follow `dr` from `sim_params.yaml`; `frictionloss_range` upper bound is a bounded choice to revisit
  after HS3; `tests/test_live_viewer.py::test_viewer_overhead_under_5pct` is timing-noisy (measured −10 % … +6 % on
  repeated runs of one tree).
