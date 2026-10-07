# AI training connector (`so101_rl.lab`) — design

> **Version 1 (2026-10-07). Design only; nothing in this file is implemented.** The user asked: *"Update this plan/package so that it can be used by an AI to train various models. Make it like a connector or MCP. For now focus on RL models; other models should be possible in the future."*
>
> **Binding user answers (2026-10-07):**
> - Plan first (design + plan/architecture deltas + implementation plan), then build in reviewed waves.
> - **Definition of done.** The connector runs **full unattended SIM training end to end** (start, monitor, review triggers, evaluate, gate, export) without a human in the loop, inside the GPU budgets and thermal limits; and, once the Tier 1 promotion is approved, **unattended REAL-HARDWARE runs** (sysid excitation, zero-shot evaluation, real fine-tuning) under the autonomy-tier rules and the hardware gates. Trips, thermal events, an MCU latch, review triggers and budget exhaustion are handled with nobody watching; real-arm runs end in the safe state and wait for a human whenever the tier rules require it.
> - AI authority: SIM runs are free within the GPU budgets; REAL-ARM runs follow the existing tiers. Tier 0 (attended): a human is present and approves each real-arm run. Tier 1 (unattended): only after the soak gates pass and a human approves the promotion; the AI may then start real-arm runs itself within the tier rules. Never bypass ServoCore, the e-stop, the deadman MCU or the consent/arm button.
> - **Tier 0 approval = a local terminal command** `pixi run approve <run-id>` that shows the run summary, requires a typed confirmation, and writes a **single-use, time-limited approval record** the connector verifies. The AI and the MCP server must never be able to create one.
> - **Alerts and visibility for unattended runs = a local web dashboard** (127.0.0.1, single user, read-mostly, small). This supersedes the earlier "no web UI beyond TensorBoard".
> - **MCP transport = stdio**, launched by the client from `.mcp.json`. Training jobs are separate background processes that outlive the MCP session (closing Claude Code never kills a run), so the job-runner state lives outside the MCP process (a run registry on disk plus a lightweight supervisor).
> - Interface: a thin MCP server (standard `mcp` Python SDK 2.3.0, conda-forge) over a Python API other frameworks can import.
> - Models: a trainer interface that is **general in shape** (datasets, offline training, export are generic concepts so imitation, world models and VLA fit later) but **implemented and tested only for FastSAC (sim), SB3 SAC + HER and TD-MPC2 (offline pre-train + fine-tune)**. No stub plugins for other model families.
> - Constraints: do not modify `safety/`, the servo process semantics, or any logged §10 decision; the connector respects the dependency rules; `docs/ARCHITECTURE.md` is updated in the same commit as any module change during the build.
>
> **Sources of truth it obeys:** `so101-self-learning-rl-plan.md` §10 (all rows, incl. 2026-10-05 GPU budgets reach ≤ 6 GPU-h / die ≤ 30 GPU-h with ≤ 3 attempts, the 85 °C pause, the zero-shot gate Wilson LB ≥ 0.60, review triggers never auto-switching methods, the reward/success definitions, D-EXIT-GATE's human-ticked checklist), `docs/ARCHITECTURE.md` v0.15 (§3 processes, §4 dependency direction, §5 interfaces, §6 safety order and resume table, §8 pixi), `plans/sim-track-implementation-plan.md` (FS1–FS12, SU3/SU4/SU5/SR2/SR3, S4.3 `metrics.csv` / `review.json` / `eval.json`, FS10 export, HS1–HS7), `plans/phase0-implementation-plan.md` (H1–H13, `runs/hw/tier.json`), `plans/camera-parity-plan.md` (`calib/camera.yaml`).
>
> **Scope discipline:** one machine, one user, one GPU, stdio, no auth beyond the local user, no cloud. The web dashboard is the one UI addition and it is read-only. Every mechanism below exists because a plan rule needs enforcing in code, or because unattended operation needs it.
>
> **Hardware rule:** nothing in this design opens `/dev/ttyACM0` or any serial device. The connector contains no bus code; on the real arm it attaches to a ServoCore process over shared memory (§1.4, §5.3).

## 0. The design in one paragraph

A new top-of-stack package `so101_rl.lab` is the **Python API**: a filesystem-backed run registry under `runs/`, a trainer-plugin registry, a task/env registry over `env/tasks.py`, dataset and policy registries, a gate/approval/authority service, a budget ledger, an audit log, a **job runner** (a detached, single-instance supervisor that drains a queue, holds the one GPU lease, enforces GPU-hour budgets, the 85 °C pause and the review-trigger policy, chains train → evaluate → gate → export pipelines, and spawns one subprocess per run attempt), a **needs-a-human queue**, and a read-only **web dashboard** on 127.0.0.1. The **MCP server** (`so101_rl.lab.mcp_server`, FastMCP, stdio) is a stateless adapter over that API: typed tools in three side-effect classes (read-only / sim-compute / real-arm), resources for runs, metrics, videos, checkpoints, configs and the plan's §10 decisions, and a few prompts. Trainers are plugins with a strict, versioned JSON-Schema config, declared capabilities (sim/real/offline, observation layout, GPU need), `train / resume / evaluate / export`, and one metrics contract (`metrics.csv` + TensorBoard through `viz`). Authority lives in code: SIM and offline jobs run freely inside the budget ledger; REAL-ARM jobs need the hardware-gate artifacts, the tier record and, at Tier 0, a human approval **signed in a separate terminal with a passphrase-protected key the server does not hold**, bound to the operator-started servo session. The connector cannot write safety config, calibration files, the decision log, the tier record or gate signatures; it reports back to plan §10 only through human-confirmed proposals.

## 1. Layering

### 1.1 Layers and dependency direction

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP client: Claude Code (stdio subprocess per session)   │ Browser (dashboard)│
└──────────────┬───────────────────────────────────────────┴───────┬───────────┘
               │ JSON-RPC over stdio                               │ GET http://127.0.0.1:6007
┌──────────────▼─────────────────────────────┐  ┌──────────────────▼───────────┐
│ so101_rl.lab.mcp_server  THIN: typing,     │  │ so101_rl.lab.dashboard        │
│ paging, response shaping, audit row/call   │  │ read-only JSON + one HTML page│
└──────────────┬─────────────────────────────┘  └──────────────────┬───────────┘
               │ plain Python calls                                │ read-only calls
┌──────────────▼───────────────────────────────────────────────────▼───────────┐
│ so101_rl.lab  (importable by any framework; no MCP dependency)               │
│  registry/   runs, policies, datasets, sweeps, pipelines (runs/<id>/…)       │
│  trainers/   TrainerPlugin protocol, plugin registry, config validation      │
│  tasks.py    task/env registry over env.tasks (+ gate status per task)       │
│  authority/  tiers, signed approvals, hw-gate artifact checks, needs-human   │
│  budgets.py  GPU-h / arm-time ledger, attempts, probes                       │
│  jobs/       supervisor (queue, leases, thermal, review policy, pipelines),  │
│              job entry (one process per attempt)                             │
│  audit.py    append-only runs/audit.jsonl + per-run events.jsonl             │
│  provenance.py  git sha, config hash, sim_params/camera/calib versions       │
│  cli.py      approve, ack, gate-sign, halt, status, audit (human, TTY)       │
│  plugins/    fastsac_sim, sb3_sac_her, tdmpc2_offline, tdmpc2_finetune, fake │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │ imports (lazily, inside job processes)
┌──────────────▼───────────────────────────────────────────────────────────────┐
│ rl/  wm/  env/  sim/  viz/  logging/  config/   (existing; unchanged roles)  │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │ shm (ShmBackend) / child process (servo --backend mujoco|fake)
┌──────────────▼──────────────┐   ┌───────────────────────────────────────────┐
│ servo (ServoCore process)   │──►│ safety  (imports nothing above it)        │
│ owns the serial bus         │   └───────────────────────────────────────────┘
└─────────────────────────────┘
```

Dependency arrows to add to ARCHITECTURE §4 (the arrows point at what a module may import):

```
lab ──► rl, wm, env, sim, viz, logging, config          (top of the stack; nothing imports lab)
lab ──► servo.shm (attach read-only), servo.main (spawn a --backend mujoco|fake child only)
lab ──► safety.types (TripCause / state names for status decoding; constants only, like sim/ and env/)
lab.mcp_server, lab.dashboard ──► lab (only); they never import torch, mujoco, warp, serial or lerobot (test-enforced)
```

Rules, test-enforced like the existing D-DEPS tests: `lab/` never imports `safety.pipeline`, `safety.clamp`, `safety.safe_state`, `servo.feetech_bus`, `servo.core`, or `pyserial`; no module under `lab/` constructs a `serial.Serial` (the same monkeypatched check as `tests/test_excitation.py::test_real_refused_without_attended_and_artifacts`). `safety/` keeps importing none of `rl`, `wm`, `env` (and none of `lab`). **Nothing in `safety/` or `servo/` changes for this design**: the connector uses the existing shm command path (`CmdMode.STOP_REQ`, RUN deltas through `ShmBackend`), the existing state record and summary, and the existing resume table of ARCHITECTURE §6 layer 5.

### 1.2 Why a Python API with a thin MCP on top

- Other frameworks (a pytest harness, an autoresearch loop, a notebook, the dashboard) import `so101_rl.lab` directly. The MCP layer adds only typing, paging and audit.
- Every rule the AI must obey (budgets, tiers, gates, allow-lists) is enforced **once**, in `lab`, so a caller that bypasses MCP gets exactly the same refusals (vault: `05-Wiki/concepts/Mechanical-Guarantees-vs-Prompt-Prose.md`; `05-Wiki/concepts/Arm-Safety-Envelope.md` §1, "a guard is only a guard if the model cannot reach the thing that enforces it").
- The MCP server process stays light (no torch, no MuJoCo), starts in well under a second, and can be killed and restarted without touching running jobs (§3).

### 1.3 Pattern choice

The vault's default for simulator MCP servers is Pattern 1, in-process sim (`05-Wiki/concepts/MCP-Server-Pattern-Choice.md`). Here it is wrong: training runs for hours, must outlive the Claude Code session that started it, and the server must never hold the GPU. The hybrid that note names ("in-process for fast operations + spawn for heavy operations like training", the `isaac_lab__train_start` example) is what this design uses: the MCP server is stateless over the filesystem; all compute runs in detached job processes under a supervisor (§3). There is deliberately **no `execute_script` tool** and no tool that takes a shell command, an executable path or a Python snippet (`05-Wiki/concepts/Hardware-Test-Harness-Pattern.md`, rule 1).

### 1.4 Where real-arm work goes through

The connector contains no bus code. Real-arm jobs use `env.backend.ShmBackend` against a **ServoCore process** (`servo --backend feetech`), exactly as `deploy-policy --backend shm`, `train-reach-sac --backend shm` and `sysid-excite --backend real` do today. The connector launches ServoCore processes itself only for the sim twin and the fakes (`servo --backend mujoco|fake`). The real-arm servo session is **human-started** at both tiers: the deadman MCU boots and trips LATCHED, the relay is held only by the heartbeat, and heartbeats stop whenever a servo process exits, so **every** real-arm power-up needs the physical button regardless of which process launches `servo` (ARCHITECTURE §3 start-up, §6 layer 2). Unattended Tier 1 work therefore means *runs inside a human-started session*, bounded by the tier rules (§5.4, §5.8).

## 2. Trainer plugin interface

### 2.1 Contract (`so101_rl/lab/trainers/api.py`; frozen dataclasses + a Protocol; the FL-series is this design's frozen-interface list)

```python
# FL1. Capabilities and plan
Domain = Literal["sim", "real", "offline"]

@dataclass(frozen=True)
class Capabilities:
    domains: frozenset[Domain]                 # fastsac_sim {"sim"}; sb3_sac_her {"sim","real"}; tdmpc2_offline {"offline"}; tdmpc2_finetune {"sim","real"}
    obs_layouts: tuple[str, ...]               # ("fs3/1",) = env.obs OBS_LAYOUT_VERSION 1; free-form strings so later layouts need no enum change
    action_space: str                          # "joint_delta_6" (F7 rule; the only one in Stage A/B)
    inputs: tuple[str, ...]                    # dataset kinds consumed: (), ("rollout_chunks",) …
    outputs: tuple[str, ...]                   # artifact kinds produced: ("policy_export",), ("tdmpc2_checkpoint",) …
    gpu: Literal["required", "optional", "none"]
    vram_gb_estimate: float
    resumable: bool
    evaluates: bool
    exports: bool
    role: Literal["core_rl", "accelerator"]    # plan §1 constraint 1: RL is the core learner; a later BC/VLA plugin is tagged accelerator
    env_name: Literal["default", "cpu", "robot"]   # the pixi environment the job runs in

@dataclass(frozen=True)
class JobPlan:                                 # computed BEFORE anything runs; returned by start_run(dry_run=True)
    domain: Domain
    leases: frozenset[Literal["gpu", "arm", "cpu"]]
    budget_key: str                            # ledger key, e.g. "s4.reach", "s4.die_to_zone", "sysid.fit", "s5.reach.finetune.sb3"
    budget_class: Literal["attempt", "probe"]  # §9 proposal 3: probes do not count against the ≤ 3 attempts
    gpu_h_estimate: float
    arm_min_estimate: float                    # 0 for sim/offline
    requires: tuple[str, ...]                  # gate / artifact ids the authority checks (e.g. "gates.L1.reach", "sim_params.version>=1", "hw.H2", "policy.sim_gate_passed")
    resumable_fields: tuple[str, ...]          # JSON pointers the AI may override on resume (e.g. "/dr/max_stage", "/budget/gpu_h")
    unattended_ok: bool                        # the plugin's own loop is unattended-safe (checkpoints on SIGTERM, honours pause/stop) — all four shipped plugins: True

# FL2. The plugin
class TrainerPlugin(Protocol):
    name: str                                  # "fastsac_sim" | "sb3_sac_her" | "tdmpc2_offline" | "tdmpc2_finetune" | "fake"
    version: str                               # plugin semver; recorded in run.json
    config_version: int                        # bumps on any incompatible schema change; the schema pins it with const
    capabilities: Capabilities
    def config_schema(self) -> dict            # JSON Schema 2020-12, additionalProperties: false everywhere, every bound explicit
    def validate(self, cfg: dict) -> dict      # jsonschema + cross-field checks → canonical dict (sorted keys, defaults filled); raises ConfigError(path, message, hint)
    def plan(self, cfg: dict, task: TaskSpec, init_from: ArtifactRef | None) -> JobPlan
    def train(self, ctx: RunContext) -> None   # in the job process; blocking; honours ctx.should_stop()/should_pause(); metrics via ctx; checkpoints on SIGTERM
    def resume(self, ctx: RunContext) -> None  # from ctx.run_dir / ckpt_last.pt; new attempt index
    def evaluate(self, ctx: RunContext, eval_cfg: dict) -> EvalReport   # writes eval.json (FS11 shape: n, successes, wilson95, final_err_m, trips, …)
    def export(self, ctx: RunContext) -> ArtifactRef                     # writes the policy artifact (FS10 export for the SAC family)
    def artifacts(self, run_dir: Path) -> ArtifactManifest               # declared outputs, each with a checker (L12: check the artifact, not the exit code)

# FL3. What the job runner gives a plugin
class RunContext:
    run_id: str; attempt: int; run_dir: Path; cfg: Mapping; task: TaskSpec; init_from: ArtifactRef | None
    device: str; vram_fraction: float          # torch.cuda.set_per_process_memory_fraction value from lab/policy.yaml
    budget: BudgetView                         # gpu_h_total, gpu_h_remaining(), hard_deadline_monotonic (the job enforces this itself too, §3.8)
    monitor: ScalarMonitor                     # metrics.csv + TensorBoard scalars (viz.TrainingMonitor when a WarpSim exists; §2.3)
    def should_stop(self) -> bool              # control/stop exists or SIGTERM seen
    def should_pause(self) -> bool             # control/pause exists (thermal, review, AI)
    def wait_if_paused(self) -> None           # blocks between steps (sim) or between episodes (real)
    def checkpoint(self, path: Path, *, step: int) -> None    # registers in artifacts; updates status.json progress
    def event(self, kind: str, **payload) -> None             # events.jsonl
    def review_trigger(self, name: str, **details) -> None    # review.json + event; the SUPERVISOR applies the run's review policy (never the plugin)
    def servo(self) -> ServoView | None        # real runs: read-only view of the servo state record + summary (state, trip bits, session id)
```

Strictness mirrors `config/loader.py`: unknown keys are errors, no hidden defaults beyond those the schema states, integers reject bools, every numeric field has `minimum`/`maximum`, enums are closed. The schema is served to the AI as a resource (`so101://trainers/{name}/schema`), and `validate()` errors come back as tool-execution errors with the JSON-pointer path and a hint (§4.5), so the AI self-corrects without reading code.

Not configurable through any plugin schema, by construction: thermal thresholds, budget caps, halt rules, the speed ladder, the tier rules, `tensorboard` (always on), the `metrics.csv` contract columns, the DR *ranges* (`sim/dr_stages.yaml` is repo-owned; a run may choose `dr.start_stage`/`dr.max_stage` and the curriculum thresholds within schema bounds). Those live in the repo-owned, strictly loaded `src/so101_rl/lab/policy.yaml` (§5.5).

**Shape vs implementation.** The shape (datasets in, artifacts out, offline domain, `role`, free-form layout strings) is what lets imitation, world-model and VLA plugins register later (§7). Only the four RL plugins below plus the `fake` test double are implemented and tested. **No stub plugins for other model families are created** (binding user decision).

### 2.2 The four RL plugins

| Plugin | Wraps (sim-track unit) | Domains | Config (schema top-level keys) | evaluate | export | Budget key / class |
|---|---|---|---|---|---|---|
| `fastsac_sim` | `rl/train_sim_sac.py` (S4.3), `rl/eval_sim.py`, `rl/policy_export.py`, `sim/domain_rand.py` + `rl/curriculum.py` (SU3), `viz` (FS11) | sim | `task`, `nworld` (256–4096), `sac` (= `SacConfig` fields with bounds), `dr {start_stage, max_stage}`, `curriculum {threshold, window_episodes, min_episodes_per_stage}`, `budget {gpu_h}`, `eval_every`, `rollouts {enabled, target_transitions}`, `viz {video_every_min, video_worlds}`, `seed`, `sim_params` (ref, default `calib/sim_params.yaml`), `review_policy` (§3.6) | `eval_sim.evaluate` (episodes, dr_stage, render n, goals_from) → `eval.json` + `renders/` + `review_checklist.md`; `gate="S4_exit"` applies ≥ 0.95 / Wilson LB ≥ 0.93 numerically | FS10 `export_policy` → `policy_export.pt` | `s4.<task>` / attempt (probe when `budget.gpu_h ≤ 0.17`) |
| `sb3_sac_her` | `rl/train_sac_her.py`, `rl/sb3_bridge.py` (S5.2), `rl/deploy_policy.py` (S5.1) | sim (CPU `SimBackend` rig, or `ShmBackend` against a `servo --backend mujoco` child), real (`ShmBackend` against the operator's `servo --backend feetech`) | `task`, `backend {sim, shm}`, `init_from` (policy ref), `total_steps`, `warmup_frozen_actor`, `baseline_eval {every, n, rollback_margin}`, `sb3 {learning_rate, batch_size, ent_coef, n_sampled_goal, tau}`, `budget {arm_min, gpu_h}`, `seed` | `deploy_policy` protocol (same goal list as the sim twin, Wilson interval, speed ladder, halt rules) | SB3 `model.zip` **and** the actor re-exported in the FS10 `policy_export.pt` format (the actor is SB3-shaped by D-ACTOR-SHAPE) | `s5.<task>.finetune.sb3` / attempt; real runs debit the arm-time ledger (8 h per algorithm for reach, 15 h for die, plan §5) |
| `tdmpc2_offline` | `rl/tdmpc2_offline.py` (S5.2, D-TDMPC2-LOOP) | offline | `task`, `dataset` (rollout_chunks ref), `steps`, `batch_size`, `horizon`, `budget {gpu_h}`, `seed` | the CPU MPC rollout on `SimBackend` (n configurable) + held-out reward/value error | `model.pt` (+ optimizer, scale) as artifact kind `tdmpc2_checkpoint` | `s5.<task>.tdmpc2_offline` / attempt |
| `tdmpc2_finetune` | `rl/train_tdmpc2.py`, `rl/tdmpc2_env.py`, `rl/her.py` (SR2/S5.2) | sim, real | `task`, `backend`, `init_from` (tdmpc2_checkpoint ref), `seed_steps`, `sim_replay_fraction`, `lr`, `total_steps`, `budget {arm_min, gpu_h}` | `deploy_policy` with the TD-MPC2 actor behind the FS10/FS11 policy path | `tdmpc2_checkpoint` | `s5.<task>.finetune.tdmpc2` / attempt |
| `fake` (tests only) | nothing | sim, offline, real (fake servo) | `duration_s`, `rows`, `behaviour: ok \| crash_at_s \| hang_at_s \| trigger_review_at_s \| exceed_budget \| oom \| servo_trip:<cause>` | synthetic `eval.json` | a dummy artifact | `fake` / probe |

Plan rules each plugin enforces or surfaces, unchanged: S4 training is refused until `runs/gates/<task>-*.json` reports L1 and L2 PASS (S4.2: gates before any reward tuning); die-to-zone training is refused on `sim_params.yaml` version 0 (D-DIE-PARAMS); the S4.3 review triggers are written to `review.json` exactly as planned and the supervisor applies the run's review policy (§3.6); the 85 °C pause is the supervisor's (§3.5).

### 2.3 Metrics and artifact contract (one for every plugin)

| Artifact | Writer | Contract |
|---|---|---|
| `metrics.csv` | the plugin, through `ctx.monitor` | Source of truth (FS11). Required columns: `env_step` (or `update` for offline), `wall_s`, `gpu_h_used`, `viz_disabled`; then the plugin's scalars. Appended atomically per row; never rewritten. |
| TensorBoard `tensorboard/` | `ctx.monitor` | Same scalars; videos embedded when rendered. `pixi run dashboard` (TensorBoard) unchanged. |
| `review.json` | the plugin via `ctx.review_trigger` | `{triggers: [{name, at_step, at_wall_s, details}], plugin, config_version}` (S4.3 shape plus the `name` the supervisor matches). |
| `ckpt_<step>.pt`, `ckpt_last.pt` | plugin via `ctx.checkpoint` | A checkpoint at every eval and on SIGTERM; `ckpt_last.pt` written as `.tmp` then renamed (the `episode_logger` pattern). |
| `eval.json` (+ `renders/`, `review_checklist.md`) | `evaluate()` | FS11 shape; `gate.json` next to it when a gate was requested (§6.3). |
| `policy_export.pt` / `model.pt` / `model.zip` | `export()` | Registered as a **policy** (`runs/policies/<policy_id>.json`, §6) with `EXPORT_VERSION`, `OBS_LAYOUT_VERSION`, task, sim success, provenance, gate statuses. |
| `rollouts/chunk_*.pt` | `fastsac_sim` (FS10 `RolloutLogger`) | Registered as a **dataset** of kind `rollout_chunks` (§7.1) after `check_chunk` passes on every file. |
| `artifacts.json` | the job runner at exit, from `plugin.artifacts()` | Each declared artifact with its checker result. A run whose checker fails is `failed` even when the process returned 0 (vault: `04-Archive/2026-07-19-gpu-campaign-3day-lessons.md`, the silent-failure stack). |

`ScalarMonitor`: FS11's `TrainingMonitor` needs a `WarpSim`. The design extracts its scalar/TensorBoard/`metrics.csv` part into a base class `viz.monitor.ScalarMonitor(run_dir, *, csv_fields)` that `TrainingMonitor` extends; non-sim plugins get the base. A small `viz/` refactor, recorded in ARCHITECTURE §4/§5 when built (open decision D14). `viz/` is not `safety/` or `servo/`, so this is within the constraints.

Plugin discovery: a static table in `lab/plugins/__init__.py`, plus the Python entry-point group `so101_rl.trainers` so a later package registers itself without editing the connector. Plugin modules are imported **lazily in the job process**; the MCP server and the dashboard read only `name`, `version`, `capabilities` and `config_schema()` from a light `manifest.py` per plugin (no torch import; test-enforced).

## 3. Run lifecycle and job runner

### 3.1 Process model

```mermaid
flowchart LR
  CC[Claude Code] -->|stdio, .mcp.json| MCP[lab-mcp<br/>stateless over runs/]
  BR[Browser] -->|GET 127.0.0.1:6007| DASH[lab-dashboard<br/>read-only]
  MCP -->|writes run.json + queue entry;<br/>starts the supervisor if absent| FS[(runs/)]
  DASH -->|reads| FS
  SUP[lab-supervisor<br/>detached, single instance<br/>flock runs/.supervisor.lock] -->|queue, leases, ledger, pipelines| FS
  SUP -->|pixi run -e <env> python -m so101_rl.lab.job --run runs/<id>| JOB[job process<br/>one per run attempt, detached]
  JOB -->|metrics.csv, status.json, events.jsonl, ckpts| FS
  JOB -.->|sim only: spawns| SERVO_SIM[servo --backend mujoco|fake]
  JOB -.->|real: ShmBackend attach| SERVO_REAL[servo --backend feetech<br/>human-started, owns the bus]
  SUP -->|nvidia-smi every 10 s| GPU[(RTX 3080)]
  HUMAN[human at the e-stop] -->|pixi run approve <id>  (separate terminal, signs)| FS
```

| Process | Lifetime | Owns | Must never |
|---|---|---|---|
| `lab-mcp` | the Claude Code session | nothing durable; reads `runs/`, writes run requests, notes, proposals, audit rows | import torch / mujoco / serial; hold the GPU; run a job in-process; write `calib/`, `config/`, the plan, `tier.json`, any `approval*` file |
| `lab-supervisor` | from the first queued job until the queue is empty and nothing runs (then exits); restarted lazily by any `lab` call that needs it; **single instance** by `flock` (the `servo/shm.py` pattern) | the queue, the leases (`gpu` ×1, `arm` ×1, `cpu` ×2), ledger writes, the thermal sampler, the stale/heartbeat detector, pipeline chaining, the needs-human queue, desktop notifications | start a real-arm job without a verified approval (Tier 0) or outside the tier rules (Tier 1); exceed a lease; auto-switch a method; resume a crashed run on its own; sign anything |
| job (`so101_rl.lab.job`) | one run attempt; detached (`start_new_session=True`) | its `run_dir`; its CUDA context with `set_per_process_memory_fraction(vram_fraction)`; its own hard budget deadline | write outside its `run_dir` (except registering datasets/policies through `lab`); open a serial device; outlive its hard deadline |
| `lab-dashboard` | started by the supervisor when the first job is admitted (`dashboard.autostart`), or by `pixi run lab-dashboard`; single instance; stays up while runs exist | nothing; GET-only | bind anything but 127.0.0.1; accept POST; write any file |
| approval CLI (`pixi run approve`) | one human interaction in a terminal | `runs/<id>/approval.json` (signed) | run without a TTY; sign a request whose hash changed; sign without the passphrase |

All lab processes are detached from Claude Code, so a closed session, a Claude Code restart or an MCP server crash never touches a running job (binding decision 3). The vault's `setsid` lesson applies: the supervisor records each job's real pid and `/proc/<pid>/cmdline`, and judges liveness by pid + cmdline, never by log greps (`05-Wiki/concepts/Autonomous-ML-Training-Loop.md`, "supervisor-first autonomy").

### 3.2 Run identity and layout

- `run_id = <task>-<plugin>-<YYYYMMDD-HHMMSS>-<4 hex>`, e.g. `reach-fastsac_sim-20261012-1403-7a2f`. Human-readable, sortable, unique.
- `runs/<run_id>/`: `run.json` (immutable request + provenance + pipeline), `status.json` (mutable, rewritten atomically), `events.jsonl`, `control/` (`stop`, `pause`, `pause_reason`), `heartbeat` (mtime touched every 10 s by a thread in the job), `metrics.csv`, `tensorboard/`, `videos/`, `ckpt_*.pt`, `review.json`, `eval.json`, `gate.json`, `artifacts.json`, `summary.json`, `notes.md`, `attempts/<k>.json` + `attempts/<k>.log`, `approval_request.json`, `approval.json`, `approval.consumed`, plus plugin artifacts.
- `runs/queue/<run_id>` (empty files; order = mtime), `runs/ledger.json` (under `flock`), `runs/policies/<policy_id>.json`, `runs/datasets/<dataset_id>.json`, `runs/sweeps/<sweep_id>.json`, `runs/needs_human/<item_id>.json`, `runs/audit.jsonl`, `runs/HALT`, `runs/supervisor/{status.json,log}`, `runs/dashboard/{pid,log}`.
- The run index is derived by scanning `runs/*/run.json` (cached in-process, invalidated by mtime). No database; SQLite is the named fallback if runs exceed a few thousand.

`runs/` stays gitignored; review notes and proposals that must survive go to the committed inbox (§6.5, open decision D11).

### 3.3 State machine

```
            request_real_run ──► awaiting_approval ──approve (signed)──► approved ─┐
                                      │ cancel / reject / expire                   │
                                      ▼                                            ▼
start_run ────────────────────────────────────────────────────────────────────► queued ──► starting ──► running ──┬──► finished ──► [pipeline: evaluate → gate → export]
                                                                                   ▲                    │  ▲       ├──► failed (artifact check or exception)
                                                                                   │                    ▼  │       ├──► stopped (graceful; reasons: ai, review, halt_rule, servo_*, thermal_gpu_timeout, idle)
                                                                        resume_run ┘                 paused       ├──► killed
                                                                        (new attempt)              (thermal |      ├──► crashed (pid gone, no exit record)
                                                                                                    review |       └──► budget_exhausted
                                                                                                    ai_pause |
                                                                                                    servo_wait)
```

Terminal: `finished`, `failed`, `stopped`, `killed`, `crashed`, `budget_exhausted`, `rejected`, `expired`. `resume_run` is allowed from `paused`, `stopped`, `crashed`, `failed` when `ckpt_last.pt` exists and the plugin is `resumable`; it creates attempt k+1 and re-queues. A paused run **releases its leases** (the GPU goes to the next queued job) and keeps its queue position; its lease clock is stopped. The supervisor never resumes anything by itself; it only unpauses the pauses it caused (thermal, servo wait).

### 3.4 Scheduling and leases

- One GPU, so `gpu` is exclusive: one GPU-class job at a time (sim training, offline pre-training, sysid fit, sim eval). `arm` is exclusive and implies `gpu` when the plugin needs it. `cpu` ×2 covers CPU evals and the fake plugin. A job declares its leases in `JobPlan`.
- FIFO within a class; `arm` jobs first. The AI reorders only by `pause_run`/`stop_run` of the running job (explicit, audited).
- Perception and training share the GPU (plan §5 "VRAM is not assumed to fit"): every job gets `vram_fraction` from `policy.yaml` (default 0.6 for sim training, S4.3); an OOM is never caught (fails loudly, L12).
- Admission also requires `disk_free_gb ≥ policy.min_disk_free_gb` (default 20) and no `runs/HALT`.

### 3.5 Budgets and the thermal pause, enforced by the supervisor (and by the job itself)

- **Ledger** (`runs/ledger.json`): per `budget_key`: `gpu_h_cap`, `gpu_h_used`, `attempts_cap` (3), `attempts_used`, `probe_gpu_h_cap`, `probe_gpu_h_used`, `arm_min_cap`, `arm_min_used`, and the run ids that debited it. Caps come from `policy.yaml` and are the plan's: `s4.reach` 6 GPU-h, `s4.die_to_zone` 30 GPU-h, `sysid.fit` 2 GPU-h × 2 attempts, `s5.<task>.tdmpc2_offline` ≈ 1 GPU-h, arm time 8 h per algorithm for reach and 15 h for die, zero-shot 2 h each. Raising a cap is not a tool; it is a §10 decision (the AI may `propose_decision`).
- **Admission:** refused when `gpu_h_estimate > remaining` or `attempts_used == attempts_cap` (`BUDGET_EXCEEDED` / `ATTEMPTS_EXHAUSTED`, with the remaining amounts and the human path). A run's own `budget.gpu_h` is clipped to the remaining cap and becomes its hard wall. Pipeline evals debit the same key (minutes).
- **Hard wall, twice:** GPU-h is wall-clock while holding the `gpu` lease minus paused time, sampled every 10 s by the supervisor. At the wall it writes `control/stop`, sends SIGTERM, waits `graceful_stop_s` (120 s; the plugin checkpoints), then SIGKILLs → `budget_exhausted`; the ledger is debited for actual use. **The job enforces the same deadline itself** (`ctx.budget.hard_deadline_monotonic`, checked between steps), so a dead supervisor cannot let a run overspend. Probes (`budget_class: probe`, ≤ 10 min each, ≤ 1 GPU-h per task in total) debit the probe pool and do **not** count as attempts (§9 proposal 3).
- **Thermal (plan §10 2026-10-05: pause above 85 °C):** `nvidia-smi --query-gpu=temperature.gpu` every 10 s (the `viz.monitor` call). Above 85 °C the supervisor writes `control/pause` (`pause_reason: thermal`); a sim job pauses at its next step, the lease clock stops; below 75 °C (hysteresis) the file is removed and the job continues. A sim job that has not acknowledged within 60 s gets SIGSTOP/SIGCONT as a fallback (safe: no hardware). A **real-arm** job is never SIGSTOPped: it pauses at an episode boundary with the arm parked (§5.8); if the GPU stays hot longer than `real_pause_max_min` (20) the run ends cleanly (`stopped: thermal_gpu_timeout`) rather than keep the arm energised. Every thermal event is an event in `get_run` and on the dashboard.

### 3.6 Review triggers: surfaced and policed, never acted on

S4.3 writes `review.json` when a trigger fires (UTD < 0.005, α or entropy out of range, |Q| blow-up, rising critic loss, saturation > 50 %, success < 50 % after 1 GPU-h, no improvement over 3 evals, curriculum stuck > 2 GPU-h; die < 10 % after 4 GPU-h). Phase 3/4 real-arm triggers (success < 10 % after 40k steps, no drop in final distance over 3 checkpoints, zero verified successes after 30k real steps) are written by the S5.2 entry points the same way. The supervisor applies the run's `review_policy`, chosen at `start_run` from a closed set and defaulted by `policy.yaml`:

| Run class | Allowed `review_policy` | Default | What happens unattended |
|---|---|---|---|
| sim `attempt` | `pause` \| `stop` \| `continue` | `pause` | `pause`: leases released, the next queued job starts, event `review_wait`, dashboard "needs review"; the AI later calls `resume_run(note=…)` (a written rationale is required), `stop_run`, or `propose_decision`. A second identical trigger on the same attempt pauses again; a third stops it (`stopped: review_repeat`). `stop`: checkpoint and stop at the first trigger. `continue`: flag only. |
| sim `probe` / sweep child | `continue` \| `stop` | `continue` | flagged in `get_run`/`get_sweep` |
| **real** (any) | `stop` only | `stop` | the episode finishes, the arm is parked (§5.8), the run is `stopped: review:<name>` and a needs-human item `review_real` is raised; a resume under Tier 1 needs a human `ack` first (real arm time is the Stage A resource and plan §5 calls these *manual* reviews). |

No trigger ever changes the method, the task, the reward or the plugin (plan §5: "a trigger opens a manual review and never switches the method automatically"; §10 2026-09-30 "Residual fallback: decided per case, with no automatic gate"). The residual-RL fallback is reachable only as a `propose_decision` for the human. Letting the AI be the *first* reviewer of a sim trigger is proposed as §10 row 4 and needs the user's confirmation; until then the default is `stop`.

### 3.7 Resume after crash, idempotency, provenance

- **Crash detection:** on every supervisor tick, a `running` job whose pid is gone (or whose cmdline no longer matches) with no exit record becomes `crashed`; a job whose `heartbeat` mtime is older than 120 s is hung → SIGTERM then SIGKILL → `crashed`. Progress stalls (no new `metrics.csv` row for `stall_flag_min`, default 30, above the longest legitimate quiet gap of a 1000-episode eval) only raise an event, never a kill (vault: stale-kill thresholds must exceed the longest legitimate quiet gap).
- **Supervisor restart:** on start the supervisor re-adopts `running` jobs from `status.json` by pid + cmdline, re-reads leases from them, and marks the rest `crashed`. A supervisor crash therefore loses nothing: jobs continue, their own deadlines hold, and admission resumes when any `lab` call restarts it (or the dashboard's "supervisor down" banner tells the human).
- **Resume:** `resume_run` → attempt k+1, `plugin.resume(ctx)` from `ckpt_last.pt`; overrides only for the plugin's `resumable_fields`, recorded in `attempts/<k+1>.json` and `run.json.overrides[]`. A crashed or tripped real-arm run is never resumed unattended (§5.8).
- **Idempotency:** `start_run(idempotency_key)` — a repeated call with the same key returns the existing run (no new attempt). Keys are stored in `run.json` and indexed on scan.
- **Provenance** (`run.json.provenance`): `git {sha, branch, dirty, diff_sha256}`, `config_sha256` (canonical JSON of the validated config), `plugin {name, version, config_version}`, `obs_layout_version`, `export_version`, `so101_yaml_sha256`, `calibration_sha256`, `sim_params {version, sha256, heldout_rms_pct}`, `camera_yaml_sha256` (when present), `hand_eye_sha256`, `dr_stages_sha256`, `policy_yaml_sha256`, `pixi {env, lock_sha256}`, `host {hostname, gpu_name, driver}`, `init_from` (ids + sha256), `created_by` (MCP session id or CLI user), `idempotency_key`. Copied into every policy and dataset manifest derived from the run.

### 3.8 Unattended operation end to end

**Pipelines.** `start_run(..., pipeline: {evaluate: {episodes, dr_stage, render, gate: "S4_exit"} | null, export: bool})` records the chain in `run.json`; when training reaches `finished` with its artifact checks green, the supervisor queues the eval job (same `budget_key`, `budget_class: eval`), then runs `export()` and registers the policy. `request_real_run(kind="zero_shot_eval")` chains the sim twin (`eval-sim --goals-from`) and `gap-report` as sim jobs after the real job. The AI (or nobody) need not be present for the chain to complete.

**The S4 exit gate stays half human by §10 (D-EXIT-GATE).** The unattended pipeline ends with `gate.json.numeric.pass`, the renders, the `review_checklist.md` (with the AI's pre-screen notes if it ran `get_rollout_frames` and `write_note`), the exported policy with status `sim_gate_numeric_passed`, and a needs-human item `gate_checklist`. The human's `pixi run lab gate-sign` flips it to `sim_gate_passed`. Nothing in this design asks to change that decision.

**When nobody is watching (sim).** A trigger → the run's review policy (§3.6); a thermal event → pause/unpause (§3.5); budget exhaustion → checkpoint + `budget_exhausted` and the next queued job starts; a crash → `crashed`, next job starts, the AI resumes later; a hung job → killed → `crashed`; disk low → no new admissions, event; host reboot → every `running` run is `crashed` on the next supervisor start; the queue is not auto-drained after a reboot until any `lab` call restarts the supervisor (D2 keeps it lazy). Every one of these produces an event, a dashboard row and, for attempt-class terminal states and needs-human items, a desktop notification (`notify-send`, when available; `policy.yaml notifications.desktop: true`).

**When nobody is watching (real arm).** §5.8 is the full failure matrix. The short form: anything the servo classes as needing a human ends the run cleanly, raises a needs-human item, blocks every further real-arm admission, and leaves the arm in the servo's own safe state; anything the servo handles itself (a thermal trip with automatic re-energise) pauses the run (`servo_wait`) and continues when the servo is back in RUN; the session ends by itself after `idle_min` or `max_session_min` through `STOP_REQ` (safe-state → torque off → process exit → MCU timeout latch), so an unattended session never stays energised without work.

**The needs-human queue.** `runs/needs_human/<item_id>.json` = `{kind, run_id?, reason, since, how_to_clear}`; kinds: `servo_latched` (physical reset), `servo_needs_human` (4th thermal trip in an hour, protection bit, calibration mismatch, user stop), `servo_trip_fresh_process` (NaN / stuck / OOD / RANGE / BOX_PRESENT / N stale commands: the servo is in TORQUE_OFF and a fresh process means the button), `review_real`, `gate_checklist`, `approval_pending`, `proposal_pending`, `hw_artifact_stale`, `disk_low`, `supervisor_restarted` (informational). The servo-class items are cleared only by `pixi run lab ack <item_id>` (TTY) after the human has looked at the arm; while any servo-class item is open, `request_real_run` returns `HARDWARE_UNAVAILABLE`.

## 4. MCP surface

### 4.1 Conventions

- Server name `so101-lab`; Claude Code exposes tools as `mcp__so101-lab__<tool>`, so tool names carry no extra prefix. Names are `verb_noun`, `[a-z_]`.
- Every tool declares `outputSchema` (FastMCP derives it from the return type) and annotations: read-only tools `readOnlyHint: true, destructiveHint: false, idempotentHint: true`; sim-compute tools `readOnlyHint: false, destructiveHint: false`; real-arm tools `destructiveHint: true, openWorldHint: true`; `stop_run(mode="kill")` is `destructiveHint: true`.
- `response_format: "concise" | "detailed"` wherever lists or records are returned (default `concise`: no hashes, paths or provenance). Lists are paged (`cursor`, `page_size ≤ 50`, default 20). Default responses stay under ~2k tokens; a hard cap of 8k tokens (`truncated: true` + a resource pointer) stays inside Claude Code's 25k-token MCP output limit (`05-Wiki/concepts/MCP-Tool-Design-and-Optimization.md`).
- **Raw logs are never returned**: `tail_run_log` gives ≤ 200 lines and the path; metrics are downsampled; videos come back as paths plus extracted PNG frames the AI can open with its own `Read` tool. (The vault measured ~16.5k tokens per 10-s episode of verbose stdout against ~160 tokens for the run JSON: `04-Archive/2026-09-10-claude-driven-so101-recording-research.md`.)
- Real-arm tools are **registered only when the server starts with `SO101_LAB_REAL_ARM=1`** (the hardware-test-mcp `HW_TEST_READONLY` convention; vault rule "read-only first, motion last"). Their refusals are enforced in `lab.authority` regardless of the flag.

### 4.2 Tools

`RunRef = {run_id, task, trainer, state, attempt, budget_class, gpu_h_used, progress: {step, pct_of_budget, eta_min}, pipeline_stage?}`.

**Read-only**

| Tool | Input | Output | Notes |
|---|---|---|---|
| `list_runs` | `{task?, trainer?, state?: [..], since?: iso, sweep_id?, cursor?, page_size?, response_format?}` | `{runs: RunRef[], next_cursor?}` | newest first |
| `get_run` | `{run_id, response_format?}` | `{run: RunRef, config_summary, budget: {key, class, gpu_h_cap, gpu_h_used, attempts}, latest_metrics: {step, ≤ 12 keys}, best, review: {policy, triggers[], waiting, pause_reason}, servo?: {state, session_id, trips}, events_tail: ≤ 10, artifacts: [{kind, path, ok}], pipeline: {stages, states}, next_actions: [..], provenance? (detailed)}` | `next_actions` lists exactly what is allowed from this state, with the reason when blocked |
| `get_metrics` | `{run_id, keys?: [..], last_n?: 50, downsample?: "auto" \| int, since_step?}` | `{columns, rows: ≤ 200, n_total, step_range}` | `auto` = uniform stride to ≤ 200 rows |
| `compare_runs` | `{run_ids: ≤ 8, keys: ≤ 8, at: "final" \| "best" \| step}` | `{table: [{run_id, config_diff, values}], common_config}` | from `summary.json` |
| `tail_run_log` | `{run_id, lines?: ≤ 200, grep?}` | `{lines, path}` | the attempt's stdout/stderr |
| `list_trainers` | `{}` | `{trainers: [{name, version, config_version, domains, inputs, outputs, gpu, role, resumable, unattended_ok}]}` | schema via resource |
| `list_tasks` | `{}` | `{tasks: [{name, obs_dim, episode_steps, gates: {L1, L2, S4_exit, S5_zero_shot}: pass \| numeric_pass \| fail \| not_run, budgets: {...remaining}, sim_params_version_required, measured_on_hardware}]}` | from `env.tasks`, `runs/gates/*.json`, the ledger, the policy registry |
| `list_policies` / `get_policy` | `{task?, status?}` / `{policy_id}` | `{policy_id, task, trainer, run_id, export_version, obs_layout_version, sim_success, sim_gate: {numeric_pass, human_pass, by}, zero_shot: {n, wilson95, pass}, real_evals[], status, provenance?}` | the single place a policy's gate status lives |
| `list_datasets` / `get_dataset` | `{kind?}` / `{dataset_id}` | `{dataset_id, kind, source_runs, n_episodes \| n_transitions, obs_layout_version, checks: ok, path}` | §7.1 |
| `get_budget` | `{task?}` | the ledger rows | |
| `get_system_status` | `{}` | `{gpu: {temp_c, mem_used_gb, util_pct}, supervisor: {alive, pid, running_run_ids, queue}, leases, arm: {servo_session: {alive, backend, state, rt_active, session_id} \| null, tier, tier_rules?, hw_artifacts: {H2, H3, H5, H7, H11, H8, H9: fresh \| stale \| missing}}, needs_human: count, disk_free_gb, halt: bool, dashboard_url}` | servo state from the shm record and the summary; never from the bus |
| `get_events` | `{since_cursor?, kinds?: [..], run_id?}` | `{events: ≤ 100, next_cursor}` | review triggers, state changes, approvals, trips, thermal, MCU |
| `get_needs_human` | `{}` | `{items: [{item_id, kind, run_id, reason, since, how_to_clear}]}` | the AI sees what blocks it and what to tell the user |
| `get_plan_decisions` | `{query?, since?}` | `{decisions: [{date, decision, choice, reason}]}` | parsed read-only from plan §10 |
| `check_gates` | `{task, stage: "L1" \| "L2" \| "S4_exit" \| "S5_zero_shot" \| "tier1", policy_id?}` | `{stage, pass, evidence: [{artifact, value, threshold, ok}], missing: [..]}` | evaluates artifacts only; runs nothing |
| `get_approval` | `{run_id}` | `{state, request_sha256, approved_by?, expires_at?, consumed?, servo_session?, limits?}` | |
| `get_sweep` | `{sweep_id}` | `{sweep_id, runs: RunRef[], best, axes, budget}` | |
| `get_rollout_frames` | `{run_id, video?: name, n?: ≤ 8}` | `{frames: [{t_s, png_path}], video_path}` | ffmpeg frame extraction into `runs/<id>/frames/` |

**Sim-compute** (GPU time and disk; never real motion)

| Tool | Input | Output | Enforced before anything runs |
|---|---|---|---|
| `start_run` | `{trainer, task, config, init_from?: policy_id \| run_id, dataset?: dataset_id, budget_class?: "attempt" \| "probe", review_policy?, pipeline?: {evaluate?, export?}, idempotency_key, notes?, dry_run?: false}` | `{run_id, state, plan: JobPlan, queue_position, est_start_min}` | schema validation; `plan.domain ∈ {sim, offline}` (a real-domain config → `TIER_FORBIDDEN`, "use request_real_run"); L1/L2 PASS for S4 training; the `sim_params` version rule; budget and attempts; `review_policy` ∈ the class's allowed set; disk; not `HALT` |
| `resume_run` | `{run_id, overrides?: {json_pointer: value}, note}` | `{run_id, attempt, state}` | overrides ⊆ `resumable_fields`; `note` required after a review pause; real runs: no open servo-class needs-human item |
| `pause_run` / `unpause_run` | `{run_id, reason}` | `{run_id, state}` | cooperative; recorded |
| `stop_run` | `{run_id, mode: "graceful" \| "kill", reason}` | `{run_id, state}` | `graceful` = control/stop + SIGTERM + 120 s grace; `kill` = SIGKILL after 5 s |
| `evaluate_policy` | `{policy_id \| run_id, episodes: ≤ 2000, dr_stage, render?: ≤ 16, goals_from?: run_id, gate?: "S4_exit", idempotency_key}` | `{eval_run_id}` | sim only; `gate` writes `gate.json` (numeric verdict); the human part is §6.3 |
| `export_policy` | `{run_id, ckpt?: "last" \| step}` | `{policy_id}` | refuses on `OBS_LAYOUT_VERSION` mismatch |
| `run_gates` | `{task, stage: "L1" \| "L2", idempotency_key}` | `{run_id}` | `pixi run sim-gates` as a job → `runs/gates/<task>-<ts>.json` |
| `start_sweep` | `{trainer, task, base_config, axes: {json_pointer: [values]} (≤ 3), seeds?, max_runs: ≤ 12, strategy: "grid" \| "random", gpu_h_total, idempotency_key}` | `{sweep_id, run_ids}` | all children `probe` unless the per-run share exceeds the probe size, then refused with the attempt arithmetic; axes must be schema paths |
| `register_dataset` | `{from_run_id, kind: "rollout_chunks" \| "episodes_h5", name?}` | `{dataset_id}` | checker on every file first |
| `write_note` | `{run_id \| "lab", text ≤ 4k}` | `{path}` | appends to `runs/<id>/notes.md` |
| `propose_decision` | `{title, proposal, evidence_run_ids, suggested_plan_row: {decision, choice, reason}}` | `{proposal_id, path}` | writes a draft in the proposals inbox (§6.5); never edits the plan |

**Real-arm** (registered only with `SO101_LAB_REAL_ARM=1`; §5)

| Tool | Input | Output | Enforced |
|---|---|---|---|
| `request_real_run` | `{kind: "zero_shot_eval" \| "finetune" \| "sysid_excitation" \| "random_policy", task, policy_id? \| trainer+config?, episodes?: ≤ 100, speed_scale?: 0.5 \| 0.75 \| 1.0, arm_min_cap: ≤ 120, idempotency_key, notes}` | `{run_id, state: "awaiting_approval" \| "queued", approval_request: {sha256, how_to_approve}}` | policy status (`sim_gate_passed` incl. the human tick for zero-shot; `zero_shot_passed` for fine-tune, D-ZEROSHOT); task prerequisites (die: `sim_params ≥ 1`, H8/H9 artifacts, verifier false-accept artifact); hardware-gate artifacts fresh; a `feetech` servo session in RUN/HOLD; arm-time ledger; speed-ladder order; no open servo-class needs-human item. Tier 0 → `awaiting_approval`; Tier 1 → `queued` only inside the tier rules (§5.4) |
| `cancel_real_request` | `{run_id}` | `{state}` | |
| (shared) `stop_run`, `get_run`, `get_approval` | | | `stop_run` on a real job = `CmdMode.STOP_REQ` through the job → the servo's safe-state sequence; **not** a safety stop (§5.7) |

There is no tool that moves a joint, writes a register, changes a speed, edits a calibration, approves, acknowledges, or starts the feetech servo. Real-arm motion exists only as whole, pre-defined procedures (`deploy_policy`, `train_sac_her`, `train_tdmpc2`, `sysid-excite`, `random_policy`) running through ServoCore with their own halt rules.

### 4.3 Resources

| URI | Content | Size discipline |
|---|---|---|
| `so101://runs` | concise run list | ≤ 50 rows |
| `so101://runs/{run_id}` | `run.json` + `status.json` merged | small; **subscribable** (`notifications/resources/updated` on every `status.json` change; the server polls the file every 2 s) |
| `so101://runs/{run_id}/metrics.csv` | the raw file | > 2 MB → path only |
| `so101://runs/{run_id}/{review,eval,gate,artifacts,summary}.json`, `/notes.md`, `/review_checklist.md` | the files | small |
| `so101://runs/{run_id}/videos/{name}` | `{path, duration_s, frames_available}` | never bytes |
| `so101://runs/{run_id}/checkpoints` | `[{step, path, size_mb, sha256}]` | never bytes |
| `so101://policies/{id}`, `so101://datasets/{id}`, `so101://sweeps/{id}` | manifests | small |
| `so101://trainers/{name}/schema`, `/readme` | the JSON Schema; a one-page usage note | ≤ 20 kB / ≤ 8 kB |
| `so101://tasks/{name}` | `TaskSpec` as JSON + gate status | small |
| `so101://plan/decisions`, `so101://plan/budgets` | parsed §10; the §5 budget table + ledger | ~30 kB; small |
| `so101://config/so101.yaml`, `so101://config/dr_stages.yaml`, `so101://calib/sim_params`, `so101://calib/camera` | read-only views (calib: `meta` only) | small |
| `so101://lab/policy`, `so101://lab/status`, `so101://lab/needs_human`, `so101://lab/audit?since=` | the enforced rules; system status; the queue; the audit tail | small |

### 4.4 Prompts

| Prompt | Arguments | Assembles |
|---|---|---|
| `design_experiment` | `task, goal, gpu_h_available?` | the task resource, trainer schemas, remaining budgets, L1/L2 status, the binding §10 rows (budgets, reward, exit gate, no auto method switch), the §5 wall-clock estimates, the L1–L13 lessons, and a required output shape: hypothesis, config diff vs defaults, budget class, review policy, pipeline, stop rule |
| `diagnose_stalled_run` | `run_id` | `get_run`, the last 50 rows of the SAC internals (α, entropy, Q, critic loss, saturation, clamp-active, tracker-visible), `review.json`, the trigger glossary, the known failure modes (L2 gaming, L4 clamps, L12 silent failures); asks for ranked hypotheses and a next action from `next_actions` |
| `review_before_real` | `policy_id` | the policy manifest, `gate.json`, `eval.json`, `check_gates(S5 prerequisites)`, the HS4 procedure, the D-ZEROSHOT ladder, the tier rules; output = the `request_real_run` text or the blocking reasons |
| `write_review_note` | `run_id` | the S4.5 one-page template filled from resources; the AI writes `runs/<id>/review_note.md` via `write_note`; a human copies it to `plans/reviews/` |
| `propose_plan_decision` | `topic, run_ids` | the §10 row format, the evidence summaries, the rule that a proposal is a draft until the human adds the row |

### 4.5 Error model

Domain failures are **tool-execution errors** (`isError: true`) with a structured body, never JSON-RPC protocol errors (`05-Wiki/concepts/MCP-Server-Development.md`, "Schema & error conventions"):

```json
{"code": "BUDGET_EXCEEDED", "message": "s4.die_to_zone: 4.2 GPU-h remaining, run needs 8.0",
 "hint": "Lower config.budget.gpu_h to <= 4.2, use budget_class=probe (<= 0.17 GPU-h), or propose_decision to raise the cap (plan §10 2026-10-05 GPU budgets).",
 "details": {"budget_key": "s4.die_to_zone", "remaining_gpu_h": 4.2, "attempts_used": 2, "attempts_cap": 3}}
```

Codes: `CONFIG_INVALID` (`details.path`, `details.schema_rule`), `UNKNOWN_TRAINER`, `UNKNOWN_TASK`, `BUDGET_EXCEEDED`, `ATTEMPTS_EXHAUSTED`, `GATE_FAILED` (`stage`, `missing[]`, `evidence[]`), `PREREQUISITE_MISSING`, `TIER_FORBIDDEN`, `APPROVAL_REQUIRED` (`how_to_approve`), `APPROVAL_INVALID` (expired / consumed / hash mismatch / bad signature / session changed), `LEASE_BUSY` (`running_run_id`, `eta_min`), `RUN_NOT_FOUND`, `STATE_CONFLICT` (`state`, `allowed_actions`), `SUPERVISOR_DOWN`, `HARDWARE_UNAVAILABLE` (servo session absent / not feetech / not RUN-HOLD / needs-human item open), `HALTED`, `DISK_LOW`, `RESPONSE_TOO_LARGE`.

### 4.6 Long-running calls and cancellation

- **Start → poll.** Every compute tool returns within a second with a `run_id`. Progress: `get_run` (`progress.eta_min`), subscriptions on `so101://runs/{id}`, and `wait_run(run_id, until: "state_change" | "terminal" | "eval" | "pipeline_done", timeout_s ≤ 300)`, a blocking convenience that returns early on the condition (cap below the client's tool timeout; configurable).
- **MCP Tasks** (SEP-1686; experimental in spec 2025-11-25, an Extension in the 2026-07-28 RC): not relied on. If the installed SDK 2.3.0 and the client both support it at build time, `start_run` may additionally return a task handle; the run_id path is the contract.
- **Cancellation.** A client `notifications/cancelled` on `wait_run` stops the wait and nothing else. Stopping a job is always the explicit, audited `stop_run`. The server never cancels a job because a session ended.

## 5. Safety and authority model

Principle (vault `Arm-Safety-Envelope.md` §1 and §6; `Agentic-Safety.md` "conservative default"): everything the AI may do is decided by **code below the tool boundary**, and everything the arm may physically do is decided by the layers below that (ServoCore clamps, the thermal trip, the safe-state sequence, the deadman MCU, the e-stop). Nothing in this section lives in a prompt. The connector **adds no safety layer and removes none**; it is a client of the policy process.

### 5.1 The authority table (`lab/authority/decide.py`; each row has a shown-to-fail test)

| Action | SIM / offline | REAL, Tier 0 (attended) | REAL, Tier 1 (unattended) |
|---|---|---|---|
| Start a training / eval / gate job | allowed when: config valid, L1+L2 PASS for S4 training, `sim_params` rule, budget + attempts, disk, not `HALT` | n/a | n/a |
| `request_real_run` | refused (`TIER_FORBIDDEN`) when the plan's domain is sim/offline | → `awaiting_approval`; needs: HW artifacts fresh (H2 `estop`, H3 `deadman`, H5 `rest-pose`, H7 `calib-check`, H11 `safe-stop`; die tasks add H8 `hand_eye.yaml` ≤ 3 mm + `fiducial.yaml` and H9 `tray`), `tier.json` tier ≥ 0, policy status, arm-time ledger, speed-ladder order, a `feetech` servo session in RUN/HOLD, no open servo-class needs-human item. Then a **signed** human approval (§5.2) | as Tier 0 **plus** `tier.json.tier == 1` with the H13 fields, the soak report ids and a `rules` block (§5.4); admitted **without** a per-run approval only when the request fits the rules; otherwise it falls back to `awaiting_approval` |
| Start the approved real job | — | the supervisor verifies the approval signature, expiry, nonce (single use), request hash and servo session id **at start**; the job verifies again before its first command | re-validates the tier rules, the session and the needs-human queue at start and before every episode batch |
| `stop_run`, `pause_run` | allowed | allowed (STOP_REQ → safe state) | allowed |
| `resume_run` of a real run | — | a new approval | admitted under the rules only when the previous attempt ended `finished`/`stopped` for a non-servo reason and no needs-human item is open; after any trip, a human `ack` first |
| Edit `config/so101.yaml`, `calib/*`, `sim/dr_stages.yaml`, `lab/policy.yaml`, `runs/hw/tier.json`, `runs/hw/*.json`, the plan, `docs/ARCHITECTURE.md`, `.mcp.json`; create an approval, an `ack`, a gate signature, a §10 row, a tier promotion | **never**: no tool exists, and a `WriteGuard` in `lab.registry` lets the MCP and supervisor processes write only under `runs/<run_id>/` (excluding `approval*`), `runs/{queue,policies,datasets,sweeps,needs_human,audit.jsonl,ledger.json,supervisor,dashboard}` and the proposals inbox | never | never |
| Disable visualisation, raise a limit, change a threshold, skip a gate | never (not in any schema) | never | never |

### 5.2 Tier 0 approval: `pixi run approve <run-id>` (signed; the server cannot forge it)

Requirements (binding): a local terminal command that shows the run summary, needs a typed confirmation, writes a **single-use, time-limited** approval record the connector verifies, and **the AI/MCP server must never be able to create one**.

Design:

1. **A human-only signing key.** One-time setup, `pixi run lab init-approver`: generates a passphrase-protected Ed25519 key `~/.config/so101_rl/approval/id_ed25519` (`ssh-keygen -t ed25519`), prints its public key, and prints the one `sudo` command that installs the trust root `/etc/so101_rl/approval_allowed_signers` (root-owned, 0644, the OpenSSH `allowed_signers` format with principal `so101-approver`). **The passphrase lives in the human's head**; the private key file is useless without it.
2. **The request.** `request_real_run` writes `runs/<id>/approval_request.json` = `{run_id, kind, task, policy_id + sha256, config_sha256, episodes, speed_scale, arm_min_cap, hw_artifacts: {id: path, mtime, sha256}, servo_session_required: true}` and returns `how_to_approve: "pixi run approve <run_id>"`.
3. **The command** (`so101_rl.lab.cli approve`, a separate process; no MCP tool calls it): refuses without a controlling TTY (`os.ctermid()` / `isatty`); re-runs every gate check (fresh artifacts, `tier.json`, a `feetech` servo session alive in RUN/HOLD with `rt_active` reported, calibration sha equal to the shipped one, no open needs-human item); prints the request summary **and the diff against the last approved request of the same kind**; asks for the typed phrase `APPROVE <run_id>`; then builds `approval.json` = `{run_id, request_sha256, approved_by: $USER, approved_at, expires_at (+2 h), nonce (128-bit random), servo_session_id (servo pid + ARM timestamp from the summary/shm), limits: {episodes, arm_min_cap, speed_scale_max}}` and signs it with `ssh-keygen -Y sign -f ~/.config/so101_rl/approval/id_ed25519 -n so101-approval`. `ssh-keygen` reads the passphrase from `/dev/tty` itself. The signature is written next to the record (`approval.json.sig`), both mode 0444.
4. **Verification** (`lab.authority.approvals.verify`, run by the supervisor at admission and by the job before its first command): `ssh-keygen -Y verify -f /etc/so101_rl/approval_allowed_signers -I so101-approver -n so101-approval -s approval.json.sig < approval.json`; `request_sha256` equals the current request file's hash; `expires_at` in the future; `servo_session_id` equals the live session; and the nonce has not been consumed. Consumption is atomic: the supervisor creates `runs/<id>/approval.consumed` with `O_EXCL` when the job starts; a resume needs a new approval.
5. **Why the server cannot create one.** (a) Signing needs the passphrase, which no process on the host holds; (b) the trust root is root-owned, so neither the MCP server, the supervisor nor an agent shell running as the user can substitute a key it generated; (c) the `WriteGuard` denies `approval*` paths to the lab processes and no tool writes them; (d) the CLI refuses without a TTY. (a)+(b) are the mechanical part; (c)+(d) are hygiene. If the user prefers not to use `sudo` for the trust root, the fallback is a user-owned `allowed_signers` protected only by (c), (d) and the Claude Code deny rules — weaker, and stated so (open decision D4).
6. **Threat model, stated plainly.** This protects against the AI misusing the tool surface as intended and against prompt-injected attempts through it. It does not protect against an agent that edits the connector's source and restarts it; that is the domain of code review (human-reviewed commits, the ARCHITECTURE sync rule) and of the physical layer (button, e-stop, deadman, bounded envelope), which hold regardless.

Two client-side levers are layered on top: Claude Code's permission system set to **always ask** for `mcp__so101-lab__request_real_run` (never allow-listed), and the `_meta["anthropic/requiresUserInteraction"]` annotation the vault recorded as an un-skippable human prompt (`Mechanical-Guarantees-vs-Prompt-Prose.md`, 2026-08-02 row), to be verified against the installed Claude Code at build time. MCP elicitation is not relied on.

### 5.3 The servo session

A **servo session** is one `servo --backend feetech` process from its ARM to its exit: `session_id = <pid>:<arm_timestamp>` read from the servo summary/shm state record. The human starts it (terminal: `REQUIRE_RT=1 pixi run servo --backend feetech --heartbeat-port … --tick-log runs/servo/<ts>/tick_log.bin`) and presses the MCU button. The lab attaches read-only to the shm segment for status and lets jobs attach as shm clients. Approvals and Tier 1 admissions are bound to the session id; a new session voids them. Sessions end through `STOP_REQ` (the existing `CmdMode`) → safe-state sequence → torque off → process exit → MCU heartbeat timeout → LATCHED; the next session needs the button. The lab never starts, signals or kills the feetech servo process other than through `STOP_REQ` over shm.

### 5.4 The Tier 1 promotion record and its rules

H13 produces `runs/hw/tier.json {tier: 1, by, date, milestone_sha}`. The connector additionally requires, for unattended admission, a human-written `rules` block (H13's procedure gains one step; the connector never writes the file):

```json
{"tier": 1, "by": "koen", "date": "...", "milestone_sha": "...",
 "soak_reports": ["runs/soak-<ts>/report.json", "runs/soak-<ts2>/report.json"],
 "rules": {"tasks": ["reach"], "kinds": ["sysid_excitation", "zero_shot_eval", "finetune"],
           "trainers": ["sb3_sac_her", "tdmpc2_finetune"],
           "max_episodes_per_run": 100, "max_session_min": 480, "idle_min": 10, "max_arm_min_per_day": 240,
           "speed_scale_max": 1.0, "require_rt_active": true, "require_camera_moved_gate": true,
           "valid_until": "2026-12-31"}}
```

A request outside the rules is not refused; it becomes `awaiting_approval` (the Tier 0 path). The servo's own protections (thermal duty 20 min on / 10 min off with torque off at rest, the 60 °C trip, at most 3 automatic thermal resumes per hour, every trip class of ARCHITECTURE §6 layer 5) are untouched; the lab only counts arm time and applies the rules.

### 5.5 `lab/policy.yaml` (repo-owned, strictly loaded, versioned)

Every number the supervisor and the authority enforce: the budget caps per `budget_key` (plan §10), `probe {max_gpu_h_each: 0.17, pool_gpu_h_per_task: 1.0}`, `thermal {pause_c: 85, resume_c: 75, sample_s: 10, sigstop_after_s: 60, real_pause_max_min: 20}`, `heartbeat {stale_s: 120}`, `stall_flag_min: 30`, `graceful_stop_s: 120`, `min_disk_free_gb: 20`, `vram_fraction {sim_training: 0.6, offline: 0.6, real: 0.3}`, `review {sim_attempt: pause, sim_probe: continue, real: stop, max_pauses_per_attempt: 2}`, `approval {expiry_h: 2, phrase: "APPROVE {run_id}", allowed_signers: "/etc/so101_rl/approval_allowed_signers"}`, `hw_artifacts {max_age_days: 90, required: {…per kind}}`, `session {park_pose: "home"}`, `notifications {desktop: true}`, `dashboard {autostart: true, port: 6007}`, `response {default_rows: 50, max_rows: 200, max_tokens: 8000}`, `wait_run {max_s: 300}`. Changing it is a code change through review (and an ARCHITECTURE §6 entry when a safety-relevant value moves).

### 5.6 Audit log

`runs/audit.jsonl`, append-only, one row per MCP tool call, CLI action and supervisor decision: `{ts, actor: {kind: mcp|cli|supervisor|job, session_id|user|pid}, action, args_sha256, args_redacted (≤ 1 kB), run_id?, result: ok|<error_code>, duration_ms}`. Read-only calls are logged too. Per run, `events.jsonl` carries the same rows filtered plus the job's events (trips with their `TripCause` names, servo state changes, thermal, pauses, checkpoints, triggers). The audit tail is a resource and `pixi run lab audit --since` prints it.

### 5.7 Kill switches, honestly labelled

- **Compute kill:** `stop_run(kill)` and `runs/HALT` (`pixi run lab halt`): nothing new starts, running jobs stop gracefully, a real-arm job sends `STOP_REQ` so the servo goes through the safe-state sequence. This protects GPU hours and the schedule.
- **Arm stop:** the e-stop and the deadman MCU. An MCP `stop_run` on a real job goes through the same lossy software path as the job itself and is counted as **zero** safety value ("never expose `stop()` as the safety stop", `Arm-Safety-Envelope.md`).

### 5.8 Unattended real-arm operation: the failure matrix

What the job (`deploy_policy` / the S5.2 entry points behind the plugin) and the supervisor do for each event, using only the servo's existing behaviour (ARCHITECTURE §3, §6 layer 5):

| Event | What the servo does (unchanged) | What the connector does | Run state → needs-human |
|---|---|---|---|
| Thermal trip (60 °C) | SAFE_STOPPING → rest → TORQUE_OFF → dwell → CHECK → automatic re-energise → RUN; at most 3 per hour, the 4th waits for a human | the episode is `truncated` (never a failure, plan §5); `ShmBackend` raises `ServoNotRunning` outside RUN/HOLD → the job enters `paused: servo_wait`, releases nothing (the arm lease stays), stops the arm-time clock, polls the state record; when RUN returns it re-creates the env (`reset` = HOLD then bounded deltas, never a jump) and continues | `paused` → `running`. After the servo's 4th trip (it does not re-energise) the job ends cleanly: `stopped: servo_needs_human` → item `servo_needs_human` |
| NaN / stuck action / OOD / RANGE / BOX_PRESENT / N stale commands | SAFE_STOPPING → TORQUE_OFF; resume needs a fresh process + checkpoint | the episode is `truncated`; `info["trip"]` names the cause; the job writes its checkpoint and exits cleanly (the servo is already in the safe state, no STOP_REQ needed) | `stopped: servo_trip:<cause>` → item `servo_trip_fresh_process` (a fresh servo process means the button). All real admissions blocked until `ack`. |
| Protection bit / calibration mismatch / user stop (incl. the human's own STOP_REQ) | SAFE_STOPPING → TORQUE_OFF; a human | as above | `stopped: servo_needs_human` → item |
| Heartbeat failure / MCU latched / deadman / e-stop pressed | LATCHED (absorbing; relay open) | `ServoNotRunning`; the job exits cleanly; the session is marked `latched`; every approval and Tier 1 admission bound to it is void | `stopped: servo_latched` → item `servo_latched` (physical reset). Real queue blocked. |
| Job's own halt rule (S5.1: non-clamp trip, 3 consecutive failures in the first 10 episodes, tracker UNKNOWN > 50 %, camera-moved fiducial gate) | nothing (RUN/HOLD) | the job parks the arm (§5.9) and exits | `stopped: halt_rule:<name>`. Fiducial → item `hw_artifact_stale` (camera moved); others are events the AI reviews |
| Real-arm review trigger (plan §5 Phase 3/4) | nothing | `review_policy` for real is `stop`: finish the episode, park, exit | `stopped: review:<name>` → item `review_real`; resume needs `ack` + a rationale |
| Arm-time budget exhausted (supervisor wall, or the job's own deadline) | nothing | finish the current episode, park, exit; ledger debited | `budget_exhausted` |
| GPU > 85 °C during a real run | nothing | pause at the episode boundary with the arm parked; after `real_pause_max_min` end the run | `paused` → `running` or `stopped: thermal_gpu_timeout` |
| Job process crash / `kill -9` | the servo holds the last target; after N consecutive misses: SAFE_STOPPING → TORQUE_OFF (stale commands → fresh process) | the supervisor sees the dead pid → `crashed`; reads the servo state → `servo_trip_fresh_process` | `crashed` → item. Real queue blocked. |
| Supervisor crash | nothing | the job continues under its own deadline and halt rules; on restart the supervisor re-adopts it by pid/cmdline | event `supervisor_restarted` |
| Host reboot / power loss | heartbeats stop → relay open → LATCHED | on the next start every `running` run is `crashed`; the servo session is gone | items `servo_latched` + `crashed` runs |
| Session idle (`idle_min`) or `max_session_min` reached, or `max_arm_min_per_day` reached | — | the supervisor sends `STOP_REQ` through a tiny shm client (the soak harness's client pattern) → safe-state → torque off → exit → MCU latch | event `session_ended:<reason>`; no item (this is the intended end) |

Every row ends with the arm in a servo-defined safe state, a terminal or paused run state, an audit row, a dashboard entry and (for items) a desktop notification. No row resumes real motion without either the servo's own automatic thermal resume (which the plan already allows) or a human act.

### 5.9 Parking between runs (Tier 1)

Within a session, between two real runs the arm is **parked**: the job's last act is a bounded-delta move to the `home` pose (the pose the servo itself ramps to on re-energise; in-box, H5-verified) and the servo is left in HOLD, energised; the servo's duty scheduler and 60 °C trip govern it. The session then either serves the next admitted run or ends after `idle_min` (default 10) via `STOP_REQ`, so an unattended arm never stays energised without work for long. Whether the park pose should be `rest`-adjacent instead (torque against gravity is lower, but it is below the re-energise start rule's 1 cm floor unless exactly at rest) is open decision D17; `home` is the default because the servo already validates it.

### 5.10 What the AI can never do (summary for the plan)

Edit safety or servo config; write calibration files or `sim_params.yaml`; change budgets, thermal thresholds, halt rules, the speed ladder or tier rules; create an approval, an `ack` or a gate signature; promote a tier or write a §10 row; disable TensorBoard/metrics; start the feetech servo or touch a register; move the arm outside a whole, admitted procedure; run arbitrary code or shell through the connector; receive raw logs or video bytes through the connector; resume a tripped real-arm run without a human.

## 6. Experiment management for an AI user

### 6.1 Comparison and summaries

Every run gets `summary.json` at exit (from `metrics.csv` + `eval.json`): final and best success, final error p50/p90, GPU-h, env-steps/s, DR stage reached, triggers fired, verdicts. `compare_runs` and `get_sweep` read only summaries (~600 tokens for 8 runs). Config diffs are computed against the plugin defaults and against each other.

### 6.2 Sweeps within budgets

`start_sweep` expands ≤ 3 axes (JSON-pointer paths into the plugin schema; an unknown key is `CONFIG_INVALID`) into ≤ 12 child runs, `probe` class by default, sharing `gpu_h_total` from the task's probe pool. A sweep needing attempt-class runs is refused with the arithmetic; the AI then promotes the best probes as attempts via `start_run(init_from=<probe run>)` where the plugin supports warm starts. Random strategy is seeded; the manifest reproduces the sweep.

### 6.3 Curriculum / DR stage control and the gates as first-class actions

- Stage control is configuration (`dr.start_stage`, `dr.max_stage`, `curriculum.*` within FS8 bounds); `resume_run(overrides={"/dr/max_stage": 3})` re-stages a paused run as a new attempt. Live mutation of a running job is deliberately absent.
- **L1/L2:** `run_gates(task, stage)` → `runs/gates/<task>-<ts>.json`; `start_run` for S4 training refuses without PASS.
- **S4 exit gate:** `evaluate_policy(..., gate="S4_exit", episodes=1000, dr_stage=4, render=16)` (or the pipeline) → `gate.json {numeric: {success, wilson_lo, pass}, human: {pass: null}}`. The human watches `renders/` and `review_checklist.md`, then `pixi run lab gate-sign <eval_run_id>` (TTY, typed confirmation) sets `human.pass`, `by`, `at`. The policy manifest flips to `sim_gate_passed` only with both (D-EXIT-GATE). `get_rollout_frames` lets the AI pre-screen for gaming signs and add *suggestions* to the checklist via `write_note`; the tick stays human.
- **S5 zero-shot gate:** `request_real_run(kind="zero_shot_eval")` runs the HS4/HS6 protocol (ladder 0.5 → 0.75 → 1.0, 10 episodes each, then N = 50/30 with the fixed goal seed), then the chained sim twin and `gap-report`. The lab sets `zero_shot: {wilson95, pass: wilson_lo ≥ 0.60}`; `request_real_run(kind="finetune")` requires `pass`, else `GATE_FAILED` with D-ZEROSHOT's hint ("one budgeted sysid/DR revision loop first") and the gap report path.
- **Tier 1 check:** `check_gates(stage="tier1")` reports `tier.json`, its `rules`, the soak reports.

### 6.4 The AI's notebook

`write_note` appends timestamped Markdown to `runs/<id>/notes.md` or `runs/lab-notes.md`; notes are resources, so a later session reads the previous session's reasoning (`Harness-Engineering.md` §2: state in files).

### 6.5 Reporting back to plan §10

`propose_decision` writes `plans/proposals/<YYYY-MM-DD>-<slug>.md` (the only repo-tree path the connector may create files in; open decision D11): title, the proposed §10 row, the evidence (run ids with summaries and provenance), alternatives, `status: proposed`. The human adds the row to §10 (citing the file) or marks it `rejected`. The connector has no code path that writes `so101-self-learning-rl-plan.md`.

## 7. Future model types: generic in shape now, nothing built

### 7.1 Datasets

Generic now: `runs/datasets/<id>.json` manifests `{dataset_id, kind, source_runs, paths, n_episodes | n_transitions, obs_layout_version, action_convention: "joint_delta_6_applied" (L4), checks: {checker, ok, at}, provenance}`; kinds `rollout_chunks` (FS10 chunks, `check_chunk`), `episodes_h5` (F8 episodes, `check_episode`), `sysid_tick_logs` (SU2). `register_dataset` runs the checker on every file (L12) first.

Deferred, plugin-side when a plugin needs it: kind `lerobot` (LeRobotDataset v3 directory; `lerobot` 0.6.1 is already a dependency) with a converter `episodes_h5 → lerobot`, kind `pixel_episodes`, quality filters. The manifest already has the fields (`obs_layout_version` becomes e.g. `"lerobot/v3-features"` with a feature list in `details`).

### 7.2 How later plugins would fit (described, not built)

| Later plugin | Would declare | Plugin-side work (not connector) |
|---|---|---|
| ACT / diffusion BC | `domains={"offline"}`, `inputs=("lerobot",)`, `outputs=("lerobot_policy",)`, `role="accelerator"` | a policy adapter so `deploy_policy` runs a non-FS3 policy: `rl/policy_export.py` v2 gains `adapter: "fs3_mlp" | "lerobot_policy"` and `deploy_policy` builds the observation the adapter asks for |
| state / pixel world model | `inputs=("rollout_chunks" | "episodes_h5",)`, `outputs=("world_model",)`, `evaluates=True` (probe error over a 1-s horizon vs FK), `exports=False` | `wm/train_state_wm.py` (SR2) behind the plugin; the cross-session gate as an `evaluate()` mode over two datasets |
| VLA fine-tune | `gpu="required"`, a high `vram_gb_estimate`, `inputs=("lerobot",)`, `role="accelerator"` | the lerobot training entry point behind the plugin; VRAM cap from `policy.yaml` |

Connector change for any of these: none. Plan §1 constraint 1 (RL stays the core learner) stays visible: `list_runs`/`compare_runs` show `role`, and `design_experiment` states it.

## 8. Dashboard, packaging, registration, testing, observability

### 8.1 The local web dashboard (`so101_rl.lab.dashboard`)

Binding: a local web page for alerts and visibility of unattended runs; 127.0.0.1 only; single user; read-mostly; small.

- **Server:** stdlib `http.server.ThreadingHTTPServer` bound to `127.0.0.1:6007` (TensorBoard keeps 6006). GET only: any other method → 405; there are no handlers that write. Endpoints: `/` (one static HTML page, vanilla JS, no framework, no build step), `/api/status.json`, `/api/runs.json?state=&task=`, `/api/runs/<id>.json`, `/api/runs/<id>/metrics.json?last_n=`, `/api/events.json?since=`, `/api/needs_human.json`, `/api/approvals.json`, `/api/ledger.json`, and `/files/<run_id>/<path>` for videos, frames and renders (resolved path must stay under `runs/`; `..` rejected). All data comes from `lab.registry` read-only calls; the server imports no torch.
- **Page** (refresh every 5 s): a banner for **needs a human** (count, kinds, `how_to_clear` text, flashing when a servo-class item is open), a **pending approvals** table (display only; it shows the exact `pixi run approve <run_id>` line), the **runs** table (state, task, trainer, attempt, progress %, GPU-h used / cap, ETA, pipeline stage, last review trigger, servo state for real runs), a **live metrics summary** per running run (last row of `metrics.csv`, sparkline of success), **events** (review triggers, trips with `TripCause` names, thermal pauses, MCU/latch, approvals consumed, session start/end), **system** (GPU temperature/memory, supervisor alive, leases, queue, disk, halt flag, servo session), and links: TensorBoard (`http://127.0.0.1:6006/#scalars`), each run's videos and renders (served from `/files/`), `review_checklist.md`.
- **Alerts:** the browser `Notification` API for new needs-human items (permission asked once); the supervisor also sends `notify-send` (Linux desktop, zero dependencies) for needs-human items and attempt-class terminal states when `notifications.desktop` is on.
- **Lifecycle:** autostarted by the supervisor when the first job is admitted (single instance by `flock`; `dashboard.autostart: true`), or `pixi run lab-dashboard`; it stays up until no run exists for an hour. A crash of the dashboard never affects jobs (separate process).

### 8.2 Pixi

- New feature `lab`: `mcp = "2.3.*"` (conda-forge), `jsonschema` (conda-forge). Environments: `default = [sim, perception, train, gpu, lab]`, `cpu = [sim, perception, train, cpu, lab]`; `robot` unchanged. OpenSSH (`ssh-keygen`) is a host requirement for approvals (present on every Linux host; checked by `lab init-approver`).
- Tasks (each in the same commit as its module; `tests/test_pixi_tasks.py`): `lab-mcp = "python -m so101_rl.lab.mcp_server"` (run in `cpu`), `lab-supervisor = "python -m so101_rl.lab.supervisor"`, `lab-dashboard = "python -m so101_rl.lab.dashboard"`, **`approve = "python -m so101_rl.lab.cli approve"`** (binding name), `lab = "python -m so101_rl.lab.cli"` (`status`, `ack`, `gate-sign`, `halt`, `unhalt`, `audit`, `proposals`, `init-approver`), `test-lab = "pytest tests/test_lab_*.py"`.
- Jobs are spawned as `pixi run -e <plugin.env_name> python -m so101_rl.lab.job --run runs/<id>` so activation (`PYTHONNOUSERSITE=1`, `MUJOCO_GL=egl`) is pixi's (open decision D3).

### 8.3 Claude Code registration (`.mcp.json` at the repo root; committed; real-arm tools off by default)

```json
{
  "mcpServers": {
    "so101-lab": {
      "command": "pixi",
      "args": ["run", "-e", "cpu", "lab-mcp"],
      "env": { "SO101_LAB_ROOT": ".", "SO101_LAB_REAL_ARM": "0", "SO101_LAB_LOG": "runs/lab-mcp.log" }
    }
  }
}
```

Recommended companion settings in `.claude/settings.json` (not part of the package): `permissions.ask` for `mcp__so101-lab__request_real_run` and `mcp__so101-lab__stop_run`; `permissions.deny` for `Edit`/`Write` on `calib/**`, `src/so101_rl/config/so101.yaml`, `src/so101_rl/lab/policy.yaml`, `runs/hw/**`, `firmware/**`, `/etc/so101_rl/**` in the experimenter persona's project settings — mechanical controls that do not depend on prompts (`Harness-Engineering.md` §7).

### 8.4 Testing strategy (all without hardware; `pixi run -e cpu test-lab` is part of `test`)

| Test file | Proves |
|---|---|
| `tests/test_lab_contract.py` | MCP contract through the SDK's in-memory client/server pair: `tools/list` is the read-only + sim-compute set (the real-arm set only with `SO101_LAB_REAL_ARM=1`); every tool has `outputSchema` + annotations; every input schema has `additionalProperties: false`; descriptions ≤ 600 chars; `tools/list` JSON ≤ 12 kB; each read-only tool on a fixture `runs/` tree returns ≤ 8k tokens at `concise`; every error path returns `isError` with a known code |
| `tests/test_lab_registry.py` | manifests round-trip; index rebuild; idempotency keys; `WriteGuard` refuses every forbidden path incl. `approval*` (shown to fail) |
| `tests/test_lab_trainers.py` | each plugin's `config_schema()` is valid 2020-12 and strict; `validate()` rejects unknown keys, bools-as-ints, out-of-range values with a JSON pointer; manifests import without torch |
| `tests/test_lab_supervisor.py` | with the `fake` plugin, a `FakeClock` and a fake `nvidia-smi` source: FIFO + leases; budget wall → checkpoint + `budget_exhausted` + ledger debit; attempts cap; probe pool; thermal pause/unpause at 85/75 °C with the lease clock stopped and the next job starting; heartbeat-stale kill; crash detection by pid/cmdline; supervisor restart re-adopts a running fake job; `HALT`; review policies per class incl. the pause-releases-lease and third-trigger-stops rules; pipeline chaining train → eval → export; disk-low refusal; no auto-resume |
| `tests/test_lab_authority.py` | the §5.1 table row by row with refusal twins: missing/stale HW artifact; Tier 0 without approval; expired / consumed / hash-mismatched / badly signed approval; new servo session voids approvals; Tier 1 rules in and out; open needs-human item blocks admission; policy status gates; `sim_params` version; L1/L2 missing |
| `tests/test_lab_approve.py` | `approve` refuses without a TTY and with a wrong phrase; under a pty with a test key and a test `allowed_signers` it writes a verifiable record; a record signed by a different key fails verification; the MCP server process has no code path that creates an approval (AST: no writer under `lab/mcp_server.py` or `lab/jobs/`); `gate-sign` and `ack` behave the same way |
| `tests/test_lab_job.py` | the job entry runs the `fake` plugin end to end (metrics, ckpt, events, artifacts.json; rc 0 with a failed checker → `failed`); SIGTERM → checkpoint + `stopped`; the job's own deadline fires with the supervisor absent; resume creates attempt 2 |
| `tests/test_lab_deps.py` | AST: `lab/` imports none of `safety.pipeline`, `safety.clamp`, `safety.safe_state`, `servo.feetech_bus`, `servo.core`, `serial`; `lab/mcp_server.py` and `lab/dashboard.py` import none of torch/mujoco/warp/lerobot; no `serial.Serial` is ever constructed by any lab test |
| `tests/test_lab_real_fake.py` | the real-arm path against `servo --backend fake` (FakeBus with thermal and fault injection, FakeMcu): request → approve (pty) → the job attaches over shm and runs `deploy_policy` with a scripted actor; `STOP_REQ` ends it through the safe state; injected thermal trip → `paused: servo_wait` → auto-resume → run continues; injected NaN → `stopped: servo_trip` + needs-human item + real admissions blocked until `ack`; FakeMcu latch → `servo_latched`; a Tier 1 `tier.json` with rules admits without approval and the session ends by `idle_min` through `STOP_REQ` |
| `tests/test_lab_dashboard.py` | binds 127.0.0.1 only (a non-loopback bind is refused); POST → 405; `/files/../` rejected; every `/api/*` renders from the fixture tree; the HTML contains no form |
| `tests/test_lab_plugins_smoke.py` (`@slow` / `@gpu`) | `fastsac_sim` 200 steps on the toy batched env (CPU) and 200 steps on the GPU through the lab (the S4.3 smoke acceptance); `tdmpc2_offline` 200 updates on a smoke rollout set; `sb3_sac_her --backend sim` 300 steps; `tdmpc2_finetune --backend sim` 300 steps |
| `tests/test_architecture_sync.py` | unchanged: `lab/` must appear in ARCHITECTURE §4 |

Eval-driven check (`MCP-Tool-Design-and-Optimization.md`): after wave L1, three scripted multi-step sessions (design → start → diagnose → export → propose) are run and their tool calls, tokens and errors recorded in `plans/reviews/<date>-lab-eval.md`; tools are consolidated only from that evidence.

### 8.5 Observability

The MCP server, supervisor and dashboard log to stderr (stdout is the MCP wire) and to rotating files under `runs/` (10 MB × 3); `runs/audit.jsonl`; `runs/supervisor/status.json`; every job's stdout/stderr in `runs/<id>/attempts/<k>.log`; TensorBoard unchanged; the dashboard renders all of it. Nothing else.

## 9. Deltas to propose and the phased build

### 9.1 `so101-self-learning-rl-plan.md`

**New section §12 "AI training connector (`so101_rl.lab`)"** (about one page): purpose; the definition of done (unattended sim end to end; unattended real runs under Tier 1); the authority model (SIM free within budgets; REAL by tier; the signed per-run Tier 0 approval; the Tier 1 `rules` block; what the AI can never do); the plugin model (RL now, imitation/WM/VLA later by shape only); review-trigger handling (never a method switch); the dashboard; proposals as the only path into §10; a pointer to this file.

**§10 rows to confirm** (dated 2026-10-07; each needs the user's yes; none changes a logged decision — rows 3 and 4 add interpretations the user must confirm):

| # | Decision | Proposed choice | Reason |
|---|---|---|---|
| 1 | AI connector | A Python API `so101_rl.lab` with a thin stdio MCP server over it; jobs are detached processes under a lazily started single-instance supervisor; state on disk under `runs/`; local single-user; no cloud | User request 2026-10-07; jobs must outlive the Claude Code session |
| 2 | AI authority | SIM/offline jobs free within the §10 budgets; REAL runs by tier: Tier 0 one signed human approval per run (`pixi run approve`, passphrase-protected key, single-use, time-limited, bound to the servo session); Tier 1 admission by a human-written `rules` block in `tier.json`. The connector never writes safety config, calibration, the plan, the tier record, approvals or gate signatures. | User answers 2026-10-07 |
| 3 | Budget accounting | A **probe** (≤ 10 min GPU, ≤ 1 GPU-h per task in total) does not count as one of the ≤ 3 attempts; an **attempt** is any longer run; pipeline evals debit the same cap. Caps change only by a §10 row. | Lets the AI sweep cheaply without eroding the attempt rule, which was written for full pre-training runs |
| 4 | Review-trigger handling | Sim attempt-class runs **pause** on a trigger (leases released) and wait for the AI's written rationale (resume / stop / propose); probes continue and are flagged; a third identical trigger stops the attempt; **real-arm** triggers always stop the run and wait for a human `ack`. No automatic method, task or reward change ever. | Keeps "a trigger opens a manual review" literal for the real arm and lets the AI be the first reviewer in sim |
| 5 | Real-arm servo session | The human starts `servo --backend feetech` and presses the button at both tiers; connector jobs attach over shm; sessions end through `STOP_REQ` after `idle_min`/`max_session_min`; the connector launches ServoCore only on `mujoco`/`fake` | The MCU needs a physical button on every power-up; "only the servo process touches the bus" stays trivially true |
| 6 | Dashboard | A read-only local web dashboard (127.0.0.1:6007, stdlib HTTP, one static page) for runs, metrics summary, triggers, trips/thermal/MCU events, needs-human items and pending approvals (display only), with links to TensorBoard and videos. **Supersedes** "no web UI beyond TensorBoard" (2026-10-07 earlier answer). | User answer 2026-10-07 |
| 7 | Proposals inbox | `plans/proposals/<date>-<slug>.md`, the only repo path the connector may create files in; a §10 row cites the file when accepted | Proposals must survive `runs/` cleanup and be reviewable in git |
| 8 | Packaging | pixi feature `lab` (`mcp 2.3.*`, `jsonschema`) in `default` and `cpu`; the server and dashboard run in `cpu`; jobs spawn with `pixi run -e <env>`; `ssh-keygen` for approval signatures | ARCHITECTURE §8 rule; the server never needs CUDA |
| 9 | Plugin set | `fastsac_sim`, `sb3_sac_her`, `tdmpc2_offline`, `tdmpc2_finetune` (+ `fake` for tests); the interface is general in shape; no stub plugins for other families | User answer 2026-10-07 |
| 10 | Trainer configs | JSON Schema 2020-12, `additionalProperties: false`, every bound explicit, `config_version` const per plugin; validated with `jsonschema` into frozen dataclasses | Same strictness as `config/loader.py`; the schema doubles as the AI's documentation |
| 11 | Real-arm tool visibility | Real-arm tools are registered only when the server starts with `SO101_LAB_REAL_ARM=1`; refusals are enforced in code regardless | Read-only first, motion last |
| 12 | Safety wording | The connector adds no safety layer and removes none; its `stop_run` is not a safety stop; all trips resolve through the servo's existing resume table and a human `ack` | Vault: a model-reachable stop counts as zero |

Also: the §5 budget table gains a "ledger key" column so the plan's numbers and `lab/policy.yaml` are one table; H13's procedure gains "write the `rules` block"; `docs/HARDWARE_CHECKLIST.md` gains the approver-key setup and the `approve`/`ack` procedures (append-only).

### 9.2 `docs/ARCHITECTURE.md` (v0.16 with the first lab unit; updated in the same commit as every later lab module change)

- **§3 process view:** rows for `lab-mcp`, `lab-supervisor`, job, `lab-dashboard`, the approval CLI (the table in §3.1), and the sentence that `lab-mcp` lives with the Claude Code session while jobs are detached.
- **§4 module view:** `lab/` and its subpackages; the dependency arrows of §1.1; the `WriteGuard` paths.
- **§5 interfaces:** FL1–FL3; the file schemas (`run.json`, `status.json`, `events.jsonl`, `approval_request.json`, `approval.json` + `.sig`, `gate.json`, `summary.json`, `ledger.json`, `needs_human/*.json`, the `tier.json` `rules` extension); `ScalarMonitor`; the servo session id.
- **§6 safety:** a new item 7 "Connector authority" after the autonomy tiers: the §5.1 table, the §5.8 failure matrix, where each rule is enforced (code, signed file, CLI, physical), and the row-12 sentence.
- **§8 packaging:** the `lab` feature, environments, tasks, `.mcp.json`, the host requirement `ssh-keygen`.
- **§9/§10:** the decision summary and the 0.16 row.

### 9.3 Phased build (units, waves, what they need from the sim track)

| Unit | Wave | Builds | Depends on | Can start |
|---|---|---|---|---|
| **LA0** core | L0 | `lab/registry`, `provenance`, `audit`, `budgets` + `policy.yaml`, `trainers/api` + registry + `fake` plugin, `jobs/supervisor` + `jobs/job` (leases, thermal, review policies, pipelines, needs-human queue, re-adoption, self-enforced deadline), `authority` for SIM rules, `WriteGuard`; `test_lab_{registry,trainers,supervisor,job,deps}` | nothing from SU3–SU5 (the `fake` plugin, plus a `vizdemo` wrapper over `viz.demo` as the first real GPU job) | **now** |
| **LA1** MCP + dashboard | L0 | `lab/mcp_server.py` (read-only + sim-compute tools, resources, prompts, errors, `wait_run`, subscriptions), `lab/dashboard.py` + the static page, `.mcp.json`, tasks, `test_lab_contract.py`, `test_lab_dashboard.py`, ARCHITECTURE v0.16, plan §12 + §10 rows | LA0 | now (parallel after FL1–FL3 freeze) |
| **LA2** FastSAC unattended | L1 | `plugins/fastsac_sim` (train/resume/evaluate/export/artifacts), `run_gates`, `evaluate_policy(gate=S4_exit)` + `lab gate-sign`, `start_sweep`, review-trigger surfacing, `ScalarMonitor` extraction, `register_dataset(rollout_chunks)`, `get_plan_decisions`; **acceptance A1–A4** | SU3 (FS8), SU4 S4.1–S4.3 | after SU4 software; S4.5/S4.6 are then driven through the lab so the budgets are enforced on the real runs |
| **LA3** offline + fine-tune (sim) | L2 | `plugins/tdmpc2_offline`, `plugins/sb3_sac_her` (`backend: sim`), `plugins/tdmpc2_finetune` (sim); `test_lab_plugins_smoke.py` | SR2, SU5 S5.1–S5.2 | after SU5 software |
| **LA4** real arm, Tier 0 | L3 | real-arm tools behind `SO101_LAB_REAL_ARM`, `approval_request`, `lab init-approver`, `approve` (signing), verification, servo-session binding, HW-artifact checks, arm-time ledger, `request_real_run(kind=…)`, the §5.8 matrix on `servo --backend fake`, `ack`, `gap.json` ingestion; `test_lab_real_fake.py`, `test_lab_approve.py`, authority real rows; **acceptance A5** | LA3; hardware H2, H3, H5, H7, H11 for first real use | software now-ish; first real use with HS1/HS4 (attended) |
| **LA5** Tier 1 unattended | L4 | `tier.json` `rules` reader, unattended admission, parking + session end (`idle_min`, `max_session_min`, `max_arm_min_per_day`), per-episode-batch re-validation; **acceptance A6** | H13 (tier 1), H10 + the 2-hour soak | after H13 |
| Deferred | — | LeRobot dataset kind + converter, policy adapter v2, imitation/WM/VLA plugins, MCP Tasks, HTTP transport, SQLite index | a plugin that needs them | when needed |

Effort (agent-hours, rough): LA0 12, LA1 10, LA2 8, LA3 6, LA4 10, LA5 3 → ≈ 49 h, plus ≈ 1 GPU-h of smoke runs. Every unit ends with the ARCHITECTURE sync **in the same commit** and a review note under `plans/reviews/`.

## 10. Acceptance (definition of done, testable)

| # | Criterion | How it is shown |
|---|---|---|
| **A1** Unattended sim run end to end | `start_run(fastsac_sim, reach, pipeline={evaluate: {1000 episodes, dr_stage 4, render 16, gate S4_exit}, export: true}, review_policy=pause)` is issued, **then Claude Code is closed**. Later: the run is `finished` within its budget; `eval.json`, `gate.json.numeric`, `renders/`, `review_checklist.md`, `policy_export.pt` exist and pass their checkers; the policy is registered `sim_gate_numeric_passed`; a `gate_checklist` needs-human item exists; the ledger shows the debit; no human action occurred; the audit log has every step. | GPU acceptance: the S4.5 reach run driven through the lab. CI twin: `test_lab_supervisor.py::test_pipeline_train_eval_export_fake` with the `fake` plugin and the MCP server process killed mid-run (`test_lab_contract.py::test_jobs_survive_server_exit`). |
| **A2** Review triggers unattended | A trigger on an attempt-class sim run pauses it, releases the GPU, starts the next queued job, raises `review_wait`; `resume_run(note)` continues from `ckpt_last.pt`; a third identical trigger stops the attempt. A probe continues and is flagged. No config, method, task or reward changes anywhere. | `test_lab_supervisor.py::test_review_policies` (`fake` plugin `trigger_review_at_s`); S4.3's `review.json` on the GPU smoke. |
| **A3** Budgets and thermal | The budget wall ends a run with a checkpoint and `budget_exhausted`, debits the ledger, and the 4th attempt is refused; a fake 86 °C reading pauses the job, stops the lease clock, resumes at 75 °C; a job with the supervisor killed still stops at its own deadline. | `test_lab_supervisor.py::test_budget_wall`, `::test_attempts_cap`, `::test_thermal_pause_unpause`; `test_lab_job.py::test_self_deadline_without_supervisor`. |
| **A4** Crash recovery and session independence | `kill -9` of a job → `crashed`, next job starts, `resume_run` works; `kill -9` of the supervisor → jobs continue; the restarted supervisor re-adopts them; closing Claude Code never changes any run state. | `test_lab_supervisor.py::test_crash_detection`, `::test_readopt_after_restart`; `test_lab_contract.py::test_jobs_survive_server_exit`. |
| **A5** Tier 0 signed approval | `request_real_run` → `awaiting_approval`; the MCP server and supervisor cannot write `approval*` (WriteGuard, AST); `pixi run approve <id>` under a pty with the test key writes a record that verifies; expired, consumed, hash-changed, wrong-key and new-session records are refused; the job starts only after verification and consumes the nonce; `resume_run` needs a new approval. On the real arm (attended): HS1 sysid excitation and HS4 zero-shot are run through `request_real_run` + `approve` and produce their planned artifacts. | `test_lab_approve.py`, `test_lab_authority.py`, `test_lab_real_fake.py::test_tier0_flow`; HS1/HS4 results in `plans/reviews/`. |
| **A6** Tier 1 unattended real runs | With `tier.json` tier 1 + `rules` and a human-started session: two real runs are admitted and run back to back with no approval, parked between them; an injected thermal trip pauses and auto-continues; an injected NaN trip ends the run in the servo's safe state, raises `servo_trip_fresh_process`, blocks further real admissions until `ack`; a FakeMcu latch yields `servo_latched`; the session ends by `idle_min` through `STOP_REQ`. On the real arm after H13: HS5 reach fine-tuning sessions run unattended inside the rules, with the arm in rest/torque-off at every session end and every trip resolved per the matrix. | `test_lab_real_fake.py::test_tier1_*`; HS5 session logs and the `servo` summaries in `plans/reviews/`. |
| **A7** Dashboard | Serves on 127.0.0.1:6007 only; refuses other binds and POST; shows runs/status/metrics/triggers/events/needs-human/pending approvals with the `approve` command text, TensorBoard and video links; no write control exists. | `test_lab_dashboard.py`; a screenshot in the LA1 review note. |
| **A8** Constraints respected | No file under `src/so101_rl/safety/` or `src/so101_rl/servo/` changes in any lab commit (reviewer `git diff --stat` check recorded in each review note); `test_lab_deps.py` green; every lab commit touches `docs/ARCHITECTURE.md` (`test_architecture_sync.py`); no §10 row is edited by the connector or by the build (the plan changes only through the user's confirmed rows). | Review notes per unit; the tests named. |
| **A9** Token discipline | `tools/list` ≤ 12 kB; every default response ≤ 8k tokens; no raw log or video bytes through MCP; the three scripted eval sessions complete within recorded budgets. | `test_lab_contract.py`; `plans/reviews/<date>-lab-eval.md`. |

## 11. Open decisions for the user (each with the recommended default)

| # | Decision | Options | Recommended default and why |
|---|---|---|---|
| D1 | Package name | `so101_rl.lab` / `so101_rl.platform` / `so101_rl.connector` | **`lab`**: short, names the thing, avoids the stdlib name `platform`; `connector` is only the MCP half |
| D2 | Who owns the queue | lazily started detached single-instance supervisor (exits when idle) / the MCP server process / an always-on daemon (systemd user unit) | **lazy supervisor**: jobs and the queue survive Claude Code restarts; nothing runs when nothing is queued; one `flock`, no systemd. Caveat: after a host reboot nothing restarts it until a `lab` call or `pixi run lab status`; a systemd user unit is the upgrade if that bites |
| D3 | How jobs are spawned | `pixi run -e <env> python -m …` / the env's `bin/python` directly | **`pixi run -e`**: activation stays pixi's and matches the "named task per entry point" rule |
| D4 | Approval trust root | root-owned `/etc/so101_rl/approval_allowed_signers` (one `sudo` at setup) / user-owned file guarded by `WriteGuard` + Claude Code deny rules | **root-owned**: it is the only way a same-user agent cannot substitute its own key; the passphrase-protected Ed25519 key via `ssh-keygen -Y` needs no Python dependency. Alternative signer: the `cryptography` package with an encrypted PEM (same guarantees, one more dependency) |
| D5 | Who starts the real-arm servo session | the human (both tiers), jobs attach / the supervisor launches `servo --backend feetech` under Tier 1 | **the human**: the MCU needs a physical button press on every power-up regardless of who launches the process; a lab-launched servo would only move the terminal command, not the human act |
| D6 | Budget accounting | probes ≠ attempts (≤ 10 min, ≤ 1 GPU-h pool per task) / every run is an attempt | **probes ≠ attempts** (§10 row 3); otherwise a 4-run sweep is impossible |
| D7 | Review-trigger policy | sim attempts pause, probes continue, real stops (row 4) / always stop / always continue | **the split**: the plan's "manual review" stays literal for the real arm; in sim the AI is the first reviewer with a written rationale and a hard cap of three |
| D8 | Config schema technology | hand-written JSON Schema + `jsonschema` + frozen dataclasses / pydantic models emitting the schema | **JSON Schema + dataclasses**: matches `config/loader.py`; pydantic (already present via `mcp`) is the fallback if hand-written schemas become tedious |
| D9 | Real-arm tool visibility | hidden unless `SO101_LAB_REAL_ARM=1` / always listed, always refusing | **hidden**: the AI cannot be talked into asking; the flag is flipped by the person who starts the servo |
| D10 | Long-running call pattern | poll + `wait_run ≤ 300 s` + resource subscriptions / MCP Tasks / progress notifications | **poll + wait_run + subscriptions**: works with today's client; Tasks when the SDK and client both support it |
| D11 | Proposals and review-note location | `plans/proposals/` (committed inbox, the only repo path the connector writes) / `runs/proposals/` (gitignored) | **`plans/proposals/`**: proposals must survive and be reviewed in git; the connector still never edits an existing repo file |
| D12 | Thermal pause mechanism | cooperative `control/pause` + SIGSTOP fallback for sim jobs; episode-boundary pause then timeout for real jobs / SIGSTOP always / kill | **cooperative + fallback**: SIGSTOP on a real-arm policy process would stall the shm client mid-episode (safe, but it trips the servo for no reason) |
| D13 | GPU sharing between a sim job and a real fine-tune | exclusive `gpu` lease / concurrent with VRAM caps | **exclusive**: 10 GB with perception alongside is already a hypothesis (plan §5); real jobs get a clean GPU |
| D14 | Metrics writer for non-sim plugins | extract `ScalarMonitor` from `viz.monitor` / a new `lab.metrics` writer | **extract**: one `metrics.csv`/TensorBoard implementation (FS11 stays the contract); a small `viz/` refactor recorded in ARCHITECTURE |
| D15 | Plan decisions resource | parse the §10 Markdown table read-only / a separate `decisions.yaml` | **parse**: the plan stays the single source of truth |
| D16 | Where the Tier 1 `rules` live | extend `runs/hw/tier.json` (H13 writes it) / a separate `runs/hw/tier_rules.yaml` | **extend `tier.json`**: one human-signed record, one check |
| D17 | Park pose between Tier 1 runs | `home` in HOLD (energised; the servo's duty and 60 °C trip govern) / drive to `rest` and let the duty scheduler torque off / end the session after every run | **`home` in HOLD with `idle_min` = 10**: the servo already validates `home`; torque-off at rest between runs would need the rest pose exactly (the 1 cm re-energise floor) and is better left to the servo's own duty cycle; ending the session per run would need a button press per run, which defeats Tier 1 |
| D18 | Dashboard process | separate process autostarted by the supervisor / served from the supervisor process / manual start only | **separate, autostarted**: crash isolation from the queue, and an unattended run always has its page up |
| D19 | Desktop notifications | `notify-send` from the supervisor for needs-human and attempt terminal states / none (dashboard only) | **on**: zero dependencies, off by one `policy.yaml` flag |

## 12. Sources consulted

Repository: `CLAUDE.md`; `so101-self-learning-rl-plan.md` (§1, §2A, §5 incl. the sim-first track and budget table, §8, §10 all rows, §11); `docs/ARCHITECTURE.md` v0.15 (§3–§8, §10); `plans/sim-track-implementation-plan.md` (FS1–FS12, SU0–SU5, SR1–SR3, SUV, HS1–HS7, budgets and review triggers, D-CHAIN … D-DEPS, as-built notes); `plans/phase0-implementation-plan.md` (H1–H13, the tier record); `plans/camera-parity-plan.md`; `pyproject.toml`; `README.md`; `tests/{conftest,test_architecture_sync,test_pixi_tasks}.py`; `src/so101_rl/{config/schema.py,logging/checks.py,viz/monitor.py}`; `docs/HARDWARE_CHECKLIST.md` (tier.json, attended rules). House MCP pattern: `~/tools/claude_code/mcp/servers/hardware-test-mcp/{README.md,server.py}` (typed tools, `READONLY` default, no `execute_script`, structured errors, soft imports, stderr-only logging).

Second Brain (`~/Documents/Vaults/Local`): `05-Wiki/concepts/MCP-Server-Development.md` (spec status, primitives, Tasks, elicitation, tool-execution errors, stdio rules); `05-Wiki/concepts/MCP-Tool-Design-and-Optimization.md` (consolidation, `response_format`, 25k-token cap, annotations, progressive disclosure); `05-Wiki/concepts/MCP-Server-Architecture.md` and `MCP-Server-Pattern-Choice.md` (job-id pattern, in-process vs spawn hybrid, never `execute_script`); `05-Wiki/concepts/Hardware-Test-Harness-Pattern.md` (typed bounded tools, summaries not blobs, read-only first, PreToolUse hook); `05-Wiki/concepts/Arm-Safety-Envelope.md` (placement rule, `stop()` is theatre, physical boundary); `05-Wiki/concepts/Agentic-Safety.md` (conservative default, approval fatigue); `05-Wiki/concepts/Mechanical-Guarantees-vs-Prompt-Prose.md` (`requiresUserInteraction`, env-var gates are conventions); `05-Wiki/concepts/Harness-Engineering.md` (state in files, verification as commands, enforce don't request, loop stop conditions); `05-Wiki/concepts/Agentic-Loops.md` (bounded loops, protected evaluation); `05-Wiki/concepts/Autonomous-ML-Training-Loop.md` and `04-Archive/2026-07-19-gpu-campaign-3day-lessons.md` (supervisor-first autonomy, stale-kill thresholds, `setsid` dead parent, pid + cmdline liveness, the silent-failure stack, check artifacts not exit codes); `05-Wiki/concepts/Autoresearch-ML-Loop-Integration.md` and `05-Wiki/source-library/lib-autoresearch.md` (program config, budget per trial, keep-or-reset); `04-Archive/2026-09-10-claude-driven-so101-recording-research.md` (token cost per episode; TTY gates are not security; reset reliability); `05-Wiki/research/2026-09-12-review-agents-need-read-only-tools.md` (read-only tool types for reviewers).
