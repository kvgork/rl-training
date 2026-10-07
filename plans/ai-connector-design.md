# AI training connector (`so101_rl.lab`) — design

> **Version 2 (2026-10-07).** v1 (commit 481cd2f) was reviewed by the plan critic (REFINE); the user then made the decisions listed under "Binding user answers" (items 8–11). This version folds in every critic finding except where a user decision overrides it (each such case is stated). **Design only; nothing in this file is implemented.** The user asked: *"Update this plan/package so that it can be used by an AI to train various models. Make it like a connector or MCP. For now focus on RL models; other models should be possible in the future."*
>
> **Binding user answers (2026-10-07):**
> 1. Plan first (design + plan/architecture deltas + implementation plan), then build in reviewed waves.
> 2. **Definition of done.** Full **unattended SIM** training end to end (start, monitor, review triggers, evaluate, gate, export) without a human in the loop, inside the GPU budgets and thermal limits; and, after the Tier 1 promotion, **unattended REAL-HARDWARE** runs (sysid excitation, zero-shot evaluation, real fine-tuning) under the tier rules and hardware gates. Trips, thermal events, an MCU latch, review triggers and budget exhaustion are handled with nobody watching; real-arm runs end in the safe state and wait for a human whenever the tier rules require it.
> 3. AI authority: SIM free within the GPU budgets; REAL by the autonomy tiers (Tier 0 attended: a human present who approves each real-arm run; Tier 1 unattended: after the soak gates and a human-approved promotion, the AI may start real-arm runs inside the tier rules). Never bypass ServoCore, the e-stop, the deadman MCU or the consent/arm button.
> 4. Interface: a thin MCP server (standard `mcp` Python SDK 2.3.0, conda-forge) over a Python API other frameworks can import. **Transport = stdio**, launched by the client from `.mcp.json`; training jobs are separate background processes that outlive the MCP session; the job-runner state lives outside the MCP process.
> 5. Models: a trainer interface **general in shape** (datasets, offline training, export are generic concepts) but **implemented and tested only for FastSAC (sim), SB3 SAC + HER and TD-MPC2 (offline pre-train + fine-tune)**. No stub plugins for other families.
> 6. **Alerts and visibility for unattended runs = a local web dashboard** (127.0.0.1, single user, read-mostly, small). Supersedes "no web UI beyond TensorBoard".
> 7. Constraints: do not modify `safety/`, the servo process semantics, or any logged §10 decision **except through the two servo/firmware work items the user decided below (SV1, FW1)**; the connector respects the dependency rules; `docs/ARCHITECTURE.md` is updated in the same commit as any module change.
> 8. **Between Tier 1 runs: hold at `home` + idle torque-off.** A NEW servo-side idle timer rests the arm and turns torque off after N idle minutes (configurable), and a NEW servo-side resume path lets an admitted run request a full checked re-energise from rest + torque-off (the thermal-resume sequence). This is a servo change, specified as work item **SV1** (§5.9) with every existing safety obligation kept. The v1 "rest + torque off between runs" is superseded.
> 9. **Firmware arm timeout (FW1, §5.10):** the watchdog goes DISARMED → LATCHED after 120 s without ARM. A button press means "arm within 2 minutes"; unattended Tier 1 runs happen inside a human-opened session that stays powered across runs (via item 8).
> 10. **Root-owned approval tools:** `approve`, `gate-sign`, `ack` are small root-owned scripts installed outside the repo (one `sudo` at setup) that use the root-owned `allowed_signers`; repo/editable code is never executed in the signing terminal. The S4-exit human tick and needs-human acknowledgements are **signed records** (same key); no plain JSON field satisfies a human gate. (This supersedes the earlier `pixi run approve` wording: the command is `so101-approve <run-id>`.)
> 11. **Supervisor as a `systemd --user` service**, and a **Tier 1 session cap ramp**: 60 min first, power-cycle + relay-weld check between sessions, the human raises the cap 60 → 120 → 240 → 480 after clean sessions; quiet hours; a minimum gap between runs; a policy allow-list; a per-day arm-minute cap with a defined day boundary.
>
> **Sources of truth it obeys:** `so101-self-learning-rl-plan.md` §10 (all rows; incl. the 2026-10-05 GPU budgets, the 85 °C pause, Wilson LB ≥ 0.60, review triggers never auto-switching methods, D-EXIT-GATE's human checklist), `docs/ARCHITECTURE.md` v0.15, `plans/sim-track-implementation-plan.md` (FS1–FS12, SU3–SU5, SR2/SR3, HS1–HS7), `plans/phase0-implementation-plan.md` (H1–H13), `plans/camera-parity-plan.md`, `firmware/watchdog_mcu/PROTOCOL.md`, `src/so101_rl/safety/state_machine.py` (the transition table), `src/so101_rl/servo/shm.py` (`CmdMode`), `src/so101_rl/servo/main.py` (the servo CLI).
>
> **Hardware rule:** nothing in this design opens `/dev/ttyACM0` or any serial device except the servo process the supervisor launches with a fixed, test-pinned argv (§5.6); the connector itself contains no bus code.

## 0. The design in one paragraph

A new top-of-stack package `so101_rl.lab` is the **Python API**: a run registry under `runs/` (manifests, `status.json`, `events.jsonl`, `usage.json`), a trainer-plugin registry (four RL plugins), a task registry over `env/tasks.py`, an authority service (tiers, signed human records, hardware-gate checks, a needs-human queue), a budget ledger derived from the run directories, an audit log, a **job runner** (a `systemd --user` supervisor that drains a queue, holds the single GPU lease, enforces GPU-hour budgets by progress-holding time, a fail-closed thermal policy, review-trigger policies, pipelines train → evaluate → gate → export, and spawns one detached process per run segment), a **real-arm session manager** (starts the feetech servo and the perception process with pinned argv, binds approvals and Tier 1 admissions to the session, enforces the session cap ramp), and a read-only **web dashboard** on 127.0.0.1. The **MCP server** (`lab.mcp_server`, FastMCP, stdio) is a stateless adapter with **12 tools**, resources for the registries and the plan's §10 decisions, and three prompts. Authority lives in code: SIM/offline jobs run freely inside the ledger; REAL-ARM jobs need the hardware-gate artifacts, the signed tier record and, at Tier 0, a per-run approval **signed in a separate terminal by a root-owned tool with a passphrase-protected key the server does not hold**. Two small, explicitly decided changes outside the connector make unattended sessions possible: a servo-side park/idle/resume path (SV1) and a firmware arm timeout (FW1). The connector adds no safety layer and removes none.

## 1. Layering

### 1.1 Layers and dependency direction

```
┌─────────────────────────────────────────────────────────────┬────────────────────────────┐
│ MCP client: Claude Code (stdio subprocess per session)      │ Browser (dashboard)        │
└──────────────┬──────────────────────────────────────────────┴──────────┬─────────────────┘
               │ JSON-RPC over stdio                                     │ GET 127.0.0.1:6007
┌──────────────▼──────────────────────────────┐  ┌──────────────────────▼─────────────────┐
│ so101_rl.lab.mcp_server  (12 tools, thin)   │  │ so101_rl.lab.dashboard (read-only)      │
└──────────────┬──────────────────────────────┘  └──────────────────────┬─────────────────┘
               │ plain Python calls                                     │ read-only calls
┌──────────────▼────────────────────────────────────────────────────────▼─────────────────┐
│ so101_rl.lab  (importable by any framework; no MCP dependency)                          │
│  registry/   runs, policies, datasets, pipelines; idempotency; WriteGuard; locks        │
│  trainers/   TrainerPlugin protocol, in-repo plugin table, config validation            │
│  tasks.py    task registry over env.tasks (+ gate status per task)                      │
│  authority/  tiers, signed-record verification, hw-gate checks, needs-human queue       │
│  budgets.py  ledger derived from runs/*/usage.json (cache runs/ledger.json)             │
│  jobs/       supervisor (systemd --user), job entry, gpu_health (pynvml), stall ladder  │
│  session/    real-arm session manager: servo + perception launch, cap ramp, cmd owner   │
│  audit.py    runs/audit.jsonl + per-run events.jsonl                                    │
│  provenance.py                                                                          │
│  cli.py      status, halt/unhalt, audit, import-run, init-approver (human, non-signing) │
│  plugins/    fastsac_sim, sb3_sac_her, tdmpc2_offline, tdmpc2_finetune, fake            │
└──────────────┬──────────────────────────────────────────────────────────────────────────┘
               │ imports (lazily, inside job processes)
┌──────────────▼──────────────────────────────────────────────────────────────────────────┐
│ rl/  wm/  env/  sim/  viz/  logging/  config/  perception/   (existing roles)           │
└──────────────┬──────────────────────────────────────────────────────────────────────────┘
               │ shm (ShmBackend / object table) ; child processes with pinned argv
┌──────────────▼──────────────┐   ┌───────────────────────────────────────────────────────┐
│ servo (ServoCore process)   │──►│ safety  (imports nothing above it)                    │
│ owns the serial bus         │   └───────────────────────────────────────────────────────┘
└─────────────────────────────┘
   root-owned, outside the repo: /usr/local/bin/so101-{approve,gate-sign,ack}  (signing)
```

Dependency arrows to add to ARCHITECTURE §4:

```
lab ──► rl, wm, env, sim, viz, logging, config, perception (process launch only)   (top of the stack; nothing imports lab)
lab ──► servo.shm (attach read-only; command writes only by an attached job or the session manager under the cmd-owner lock)
lab ──► servo.main (spawn with the pinned argv of §5.6; never imported for bus access)
lab ──► safety.types (TripCause / SmState names; constants only)
lab.mcp_server, lab.dashboard ──► lab only; never torch, mujoco, warp, serial, lerobot (test-enforced)
```

Test-enforced rules: `lab/` never imports `safety.pipeline`, `safety.clamp`, `safety.safe_state`, `servo.feetech_bus`, `servo.core`, or `pyserial`; no `serial.Serial` is constructed by anything under `lab/`. `safety/` imports none of `rl`, `wm`, `env`, `lab`. **`safety/`, `servo/` and `firmware/` change only in the SV1 and FW1 commits** (guard in §8.6).

### 1.2 Why a Python API with a thin MCP on top

Other frameworks import `so101_rl.lab` directly; the MCP layer adds typing, paging and audit. Every rule the AI must obey is enforced once, in `lab`, so a caller that bypasses MCP gets the same refusals (vault `05-Wiki/concepts/Mechanical-Guarantees-vs-Prompt-Prose.md`; `Arm-Safety-Envelope.md` §1). The MCP server stays light (no torch, no MuJoCo) and can be restarted without touching jobs.

### 1.3 Pattern choice

The vault's default (Pattern 1, in-process sim; `05-Wiki/concepts/MCP-Server-Pattern-Choice.md`) is wrong here: training runs for hours and must outlive the session. The hybrid that note describes (`isaac_lab__train_start` spawning a training process) is used: a stateless server, detached jobs under a supervisor. No `execute_script`, no shell, no path-to-execute tool (`05-Wiki/concepts/Hardware-Test-Harness-Pattern.md`, rule 1).

### 1.4 Where real-arm work goes through

Real-arm jobs use `env.backend.ShmBackend` against a **ServoCore process** (`servo --backend feetech`), as `deploy-policy --backend shm`, `train-reach-sac --backend shm` and `sysid-excite --backend real` do. The servo process for a real-arm **session** is launched by the session manager with a fixed argv (§5.6); the relay still closes only after the human presses the MCU button, and with FW1 the press must be followed by ARM within 120 s. A session therefore always begins with a human act and, with SV1, stays powered across runs (parked at `home`, resting with torque off when idle, re-energised through the full checked sequence on request).

## 2. Trainer plugin interface

### 2.1 Contract (`so101_rl/lab/trainers/api.py`; frozen dataclasses + a Protocol; FL-series = this design's frozen interfaces)

```python
# FL1. Capabilities and plan
Domain = Literal["sim", "real", "offline"]

@dataclass(frozen=True)
class Capabilities:
    domains: frozenset[Domain]
    obs_layouts: tuple[str, ...]               # ("fs3/1",) today; free-form strings so later layouts need no enum change
    action_space: str                          # "joint_delta_6" (F7 rule)
    inputs: tuple[str, ...]                    # dataset kinds consumed: (), ("rollout_chunks",) …
    outputs: tuple[str, ...]                   # artifact kinds: ("policy_export",), ("tdmpc2_checkpoint",) …
    gpu: Literal["required", "optional", "none"]
    vram_gb_estimate: float
    resumable: bool; evaluates: bool; exports: bool
    role: Literal["core_rl", "accelerator"]    # plan §1 constraint 1 stays visible
    env_name: Literal["default", "cpu", "robot"]

@dataclass(frozen=True)
class JobPlan:                                 # computed before anything runs; start_run(dry_run=True) returns it
    domain: Domain
    leases: frozenset[Literal["gpu", "arm", "cpu"]]
    budget_key: str                            # "s4.reach", "s4.die_to_zone", "sysid.fit", "s5.reach.finetune.sb3", …
    budget_class: Literal["attempt", "probe", "eval"]
    gpu_h_estimate: float; arm_min_estimate: float
    disk_gb_estimate: float                    # per-run quota check
    requires: tuple[str, ...]                  # gate / artifact ids for the authority
    resumable_fields: tuple[str, ...]          # JSON pointers the AI may override on resume
    unattended_ok: bool                        # checkpoints on SIGTERM, honours pause/stop, progress-coupled heartbeat

# FL2. The plugin
class TrainerPlugin(Protocol):
    name: str; version: str; config_version: int; capabilities: Capabilities
    def config_schema(self) -> dict            # JSON Schema 2020-12, additionalProperties: false, every bound explicit, const config_version
    def validate(self, cfg: dict) -> dict      # jsonschema + cross-field checks → canonical dict; ConfigError(path, message, hint)
    def plan(self, cfg: dict, task: TaskSpec, init_from: ArtifactRef | None) -> JobPlan
    def train(self, ctx: RunContext) -> None   # blocking; honours ctx.should_stop()/should_pause(); heartbeats from inside its loops
    def resume(self, ctx: RunContext) -> None  # from ckpt_last.pt; new segment
    def evaluate(self, ctx: RunContext, eval_cfg: dict) -> EvalReport   # eval.json (FS11 shape)
    def export(self, ctx: RunContext) -> ArtifactRef
    def artifacts(self, run_dir: Path) -> ArtifactManifest               # declared outputs + a checker each (L12)

# FL3. What the job runner gives a plugin
class RunContext:
    run_id: str; segment: int; run_dir: Path; cfg: Mapping; task: TaskSpec; init_from: ArtifactRef | None
    device: str; vram_fraction: float
    budget: BudgetView                         # gpu_h_total, gpu_h_remaining(), hard_deadline_monotonic (the job enforces it itself)
    monitor: ScalarMonitor                     # metrics.csv + TensorBoard (viz.TrainingMonitor when a WarpSim exists)
    def heartbeat(self, *, step: int | None = None, phase: str = "train") -> None   # PROGRESS-COUPLED: called from inside the train/eval loops, never from a thread
    def should_stop(self) -> bool; def should_pause(self) -> bool
    def checkpoint(self, path: Path, *, step: int) -> None
    def event(self, kind: str, **payload) -> None
    def review_trigger(self, name: str, **details) -> None    # review.json + event; the supervisor applies the run's review policy
    def servo(self) -> ServoView | None        # real runs: read-only view of the state record + summary (state, trips, session id, parked)
    def record_usage(self) -> None             # writes usage.json atomically (gpu-h by progress-holding time, arm-min, disk); every 60 s and at exit
```

Strictness mirrors `config/loader.py`. The schema is served as a resource and `validate()` errors are tool-execution errors with a JSON pointer and a hint. Not configurable through any plugin schema: thermal thresholds, budget caps, halt rules, the speed ladder, tier rules, `tensorboard` (always on), the `metrics.csv` contract, DR ranges (`sim/dr_stages.yaml` is repo-owned). Those live in `src/so101_rl/lab/policy.yaml` (§5.5).

**Shape vs implementation.** The shape (datasets in, artifacts out, `offline`, `role`, free-form layout strings) lets imitation, world-model and VLA plugins register later (§7). Only the four RL plugins and the `fake` test double exist. Plugin discovery is a **static in-repo table** (`lab/plugins/__init__.py`); entry-point discovery was dropped (critic: unnecessary surface).

### 2.2 The four RL plugins

| Plugin | Wraps | Domains | Config (top-level keys) | evaluate | export | Budget key / class |
|---|---|---|---|---|---|---|
| `fastsac_sim` | `rl/train_sim_sac.py`, `rl/eval_sim.py`, `rl/policy_export.py` (S4.3), `sim/domain_rand.py` + `rl/curriculum.py` (SU3), `viz` (FS11) | sim | `task`, `nworld`, `sac` (`SacConfig` fields, bounded), `dr {start_stage, max_stage}`, `curriculum {…}`, `budget {gpu_h}`, `eval_every`, `rollouts {enabled, target_transitions}`, `viz {video_every_min, video_worlds}`, `seed`, `sim_params`, `review_policy` | `eval_sim.evaluate` → `eval.json`, `renders/`, `review_checklist.md`; `gate="S4_exit"` numeric part | FS10 `policy_export.pt` | `s4.<task>` / attempt (probe when `budget.gpu_h ≤ 0.17`) |
| `sb3_sac_her` | `rl/train_sac_her.py`, `rl/sb3_bridge.py`, `rl/deploy_policy.py` (S5.1/S5.2) | sim (`SimBackend`, or `ShmBackend` vs a `servo --backend mujoco` child), real (`ShmBackend` vs the session's feetech servo) | `task`, `backend {sim, shm}`, `init_from`, `total_steps`, `warmup_frozen_actor`, `baseline_eval {…}`, `sb3 {…}`, `budget {arm_min, gpu_h}`, `seed` | `deploy_policy` protocol | `model.zip` + the actor re-exported as `policy_export.pt` (D-ACTOR-SHAPE) | `s5.<task>.finetune.sb3` / attempt; real runs debit arm time |
| `tdmpc2_offline` | `rl/tdmpc2_offline.py` (D-TDMPC2-LOOP) | offline | `task`, `dataset {id \| from_run}` (registers the rollout set if `from_run`), `steps`, `batch_size`, `horizon`, `budget {gpu_h}`, `seed` | CPU MPC rollout on `SimBackend` + held-out errors | `tdmpc2_checkpoint` | `s5.<task>.tdmpc2_offline` / attempt |
| `tdmpc2_finetune` | `rl/train_tdmpc2.py`, `rl/tdmpc2_env.py`, `rl/her.py` | sim, real | `task`, `backend`, `init_from`, `seed_steps`, `sim_replay_fraction`, `lr`, `total_steps`, `budget {arm_min, gpu_h}` | `deploy_policy` | `tdmpc2_checkpoint` | `s5.<task>.finetune.tdmpc2` / attempt |
| `fake` (tests) | nothing | sim, offline, real (fake servo) | `duration_s`, `rows`, `behaviour: ok \| crash_at_s \| hang_at_s \| stall_at_s \| trigger_review_at_s \| exceed_budget \| oom \| disk_fill \| servo_trip:<cause>` | synthetic | dummy | `fake` / probe |

Plan rules enforced per plugin (unchanged): L1/L2 PASS before S4 training; die-to-zone refused on `sim_params` v0 (D-DIE-PARAMS); S4.3 review triggers → `review.json`; the thermal pause is the supervisor's. The real-arm entry points gain a **`--tier1 <tier.json>` mode** (a `rl/` and `sim/sysid` change, not `safety`/`servo`): it verifies the signed tier record through `lab.authority` and replaces the `--attended` flag for unattended runs, so `--attended` is never passed when nobody attends.

### 2.3 Metrics and artifact contract

| Artifact | Writer | Contract |
|---|---|---|
| `metrics.csv` | plugin via `ctx.monitor` | Source of truth. Required columns `env_step`/`update`, `wall_s`, `gpu_h_used`, `viz_disabled`; appended atomically; never rewritten. |
| TensorBoard | `ctx.monitor` | Same scalars; `pixi run dashboard` unchanged. |
| `review.json` | `ctx.review_trigger` | `{triggers: [{name, at_step, at_wall_s, details}], plugin, config_version}`. |
| `ckpt_<step>.pt`, `ckpt_last.pt` | `ctx.checkpoint` | At every eval and on SIGTERM; `.tmp` + rename. |
| `eval.json`, `renders/`, `review_checklist.md`, `gate.json` | `evaluate()` | FS11 shape; `gate.json` holds the numeric verdict only; the human part is the signed `gate_sign.json` (§5.3). |
| `policy_export.pt` / `model.pt` / `model.zip` | `export()` | registered as a policy (`runs/policies/<id>.json`) with gate statuses derived from signed records. |
| `rollouts/chunk_*.pt` | FS10 `RolloutLogger` | registered as a `rollout_chunks` dataset after `check_chunk` on every file. |
| `usage.json` | the job (`ctx.record_usage`) | `{gpu_h, arm_min, disk_gb, segments: [{pid, start, end, progress_s, stalled_s}]}`; the ledger is folded from these. |
| `artifacts.json` | the job runner at exit | each declared artifact + checker result; a failed checker → `failed` even with rc 0 (vault: the silent-failure stack). |

`ScalarMonitor`: the scalar/TensorBoard/`metrics.csv` part of FS11's `TrainingMonitor` is extracted into `viz.monitor.ScalarMonitor`; `TrainingMonitor` extends it (a `viz/` refactor, recorded in ARCHITECTURE when built).

## 3. Run lifecycle and job runner

### 3.1 Process model

```mermaid
flowchart LR
  CC[Claude Code] -->|stdio, .mcp.json| MCP[lab-mcp<br/>stateless over runs/]
  BR[Browser] -->|GET 127.0.0.1:6007| DASH[lab-dashboard<br/>systemd --user, read-only]
  MCP -->|run.json + queue entry under flock| FS[(runs/)]
  DASH -->|reads| FS
  SUP[lab-supervisor<br/>systemd --user service<br/>single instance] -->|queue, leases, ledger, pipelines, thermal| FS
  SUP -->|pixi run -e env python -m so101_rl.lab.job| JOB[job process<br/>one per run segment, detached]
  JOB -->|metrics, status, events, usage, ckpts| FS
  JOB -.->|sim only: spawns| SERVO_SIM[servo --backend mujoco|fake]
  SUP -->|session manager: pinned argv| SERVO_REAL[servo --backend feetech]
  SUP -->|session manager| PERC[perception process<br/>D435 tracker, fiducial, object table]
  JOB -.->|ShmBackend attach, cmd-owner lock| SERVO_REAL
  SUP -->|pynvml every 10 s| GPU[(RTX 3080)]
  HUMAN[human at the e-stop] -->|so101-approve / gate-sign / ack  (root-owned, separate terminal)| FS
  HUMAN -->|MCU button within 120 s of ARM wait| SERVO_REAL
```

| Process | Lifetime | Owns | Must never |
|---|---|---|---|
| `lab-mcp` | the Claude Code session | nothing durable; writes run requests, notes, proposals, audit rows under `flock` | import torch/mujoco/serial; hold the GPU; run a job in-process; write `calib/`, `config/`, the plan, `tier.json`, any signed record |
| `lab-supervisor` | `systemd --user` service (`Restart=on-failure`, `loginctl enable-linger`, `WorkingDirectory` = repo, `ExecStart=pixi run -e cpu lab-supervisor`); single instance by `flock` | queue, leases (`gpu` ×1, `arm` ×1, `cpu` ×2), ledger cache, `gpu_health`, stall ladder, pipelines, needs-human queue, the real-arm session manager, notifications | start a real-arm job without a verified signed approval (Tier 0) or outside the signed tier rules (Tier 1); exceed a lease; auto-switch a method; resume a tripped real run on its own; sign anything; write the shm command block while a job holds the cmd-owner lock |
| job (`so101_rl.lab.job`) | one run **segment** (one process lifetime); detached (`start_new_session=True`); records **its own pid** (`os.getpid()`, not the pixi wrapper's) in `status.json` | its `run_dir`; its CUDA context (`set_per_process_memory_fraction`); its own hard deadline and disk quota | write outside `run_dir` except registering datasets/policies through `lab`; open a serial device; outlive its deadline |
| `lab-dashboard` | `systemd --user` service; GET-only on 127.0.0.1:6007 | nothing | bind elsewhere; accept POST; write any file |
| servo (real session) | one session: from ARM to exit, bounded by `--max-ticks` (§5.6) | the serial bus, every safety check, the heartbeat | be started with `--no-mcu`; outlive `--max-ticks` |
| perception (real session) | the session | the D435, the object table, the fiducial check | share a CUDA context with training without a cap (Stage A tracker is CPU) |
| `so101-approve` / `so101-gate-sign` / `so101-ack` | one human interaction | the signed record | run without a TTY; execute repo code; sign without the passphrase |

### 3.2 Run identity and layout

- `run_id = <task>-<plugin>-<YYYYMMDD-HHMMSS>-<4 hex>`.
- `runs/<run_id>/`: `run.json` (immutable request + provenance + pipeline), `status.json` (atomic rewrite: state, segment, pid, boot_id, leases, pause_reason, progress, last heartbeat step/phase), `events.jsonl`, `control/` (`stop`, `pause`, `end_session`), `heartbeat` (written by `ctx.heartbeat` with `{step, phase, mono_ns}`), `usage.json`, `metrics.csv`, `tensorboard/`, `videos/`, `ckpt_*.pt`, `review.json`, `eval.json`, `gate.json`, `gate_sign.json` + `.sig`, `artifacts.json`, `summary.json`, `notes.md`, `segments/<k>.json` + `.log`, `approval_request.json`, `approval.json` + `.sig`, `approval.consumed`.
- `runs/queue/<run_id>` (mtime order; `ttl` in the file), `runs/idem/<sha256(key)>` (O_EXCL, holds the run_id: multi-client idempotency), `runs/.registry.lock` (`flock` around every registry mutation, so two MCP processes cannot double-create), `runs/ledger.json` (**cache**, §3.5), `runs/policies/`, `runs/datasets/`, `runs/needs_human/<item_id>.json` (+ `.ack.json`/`.sig`), `runs/audit.jsonl`, `runs/HALT`, `runs/supervisor/status.json`, `runs/servo/<session_id>/{argv.json, tick_log.bin, summary.json, stdout.log, stderr.log, cmd_owner.lock}`, `runs/perception/<session_id>/`, `runs/sessions.jsonl` (session history for the cap ramp and the per-day arm-minute cap).
- Index derived by scanning `runs/*/run.json` (cached by mtime). `runs/` stays gitignored; proposals go to the committed inbox (§6.4).

### 3.3 State machine (a "pause" is a checkpoint plus process exit; no paused process holds VRAM)

```
request_real_run → awaiting_approval ─approve(signed)→ approved ─┐          (cancel / reject / expire)
start_run ───────────────────────────────────────────────────────┴→ queued → starting → running ─┬→ finished → [pipeline: evaluate → gate(numeric) → export]
                                                                      ▲                   │  ▲   ├→ failed (artifact check or exception)
                                                      resume_run /    │                   ▼  │   ├→ stopped (reasons: ai, review, halt_rule, servo_*, thermal_gpu_timeout, disk, stall, idle)
                                                      auto-unpause ───┘                paused   ├→ killed
                                                      (new segment)            (thermal | review | ai | servo_wait | gpu_health | host_slept)
                                                                                                 ├→ crashed (pid gone / boot_id changed / heartbeat stale → killed)
                                                                                                 └→ budget_exhausted
```

- `paused` = the job checkpointed and **exited**; `status.json.pause_reason` says why. Continuing starts a new segment (`plugin.resume`). **Thermal, `gpu_health` and `servo_wait` pauses are supervisor-owned**: the supervisor unpauses them automatically when the condition clears; they do **not** release the GPU lease (nothing else is admitted while the GPU is hot or its health is unknown). **Review and AI pauses release the leases** and keep the queue position; only the AI continues them.
- Terminal: `finished`, `failed`, `stopped`, `killed`, `crashed`, `budget_exhausted`, `rejected`, `expired`. `resume_run` from `paused`/`stopped`/`crashed`/`failed` when `ckpt_last.pt` exists and the plugin is `resumable`.

### 3.4 Scheduling, leases, queue hygiene

- One GPU: `gpu` exclusive (sim training, offline, sysid fit, sim eval); `arm` exclusive and implies `gpu` when the plugin needs it; `cpu` ×2. FIFO within a class, `arm` first. The AI reorders only via `control_run`.
- Admission requires: no `HALT`; `disk_free_gb ≥ min_disk_free_gb` (20) and the run's `disk_gb_estimate` fits; the ledger (§3.5); for real jobs the session state (§5.7).
- **Queue hygiene:** entries carry a `ttl` (`queue_ttl_h`, default 24 for sim/offline, 4 for real); expired entries become `expired`. After a **reboot** (`boot_id` changed) every queued real-arm entry is voided (its servo session no longer exists); after a **suspend** (`CLOCK_BOOTTIME − CLOCK_MONOTONIC` jumped by > 5 s between supervisor ticks) the supervisor emits `host_slept`, revalidates the servo session (a suspend stops heartbeats, so the MCU has latched: §5.8) and voids every approval (wall-clock expiry plus the session binding). Heartbeat ages use `CLOCK_MONOTONIC`, which does not advance in suspend, so a sleep never looks like a stall.

### 3.5 Budgets: the ledger is derived, GPU-h is progress-holding time, and the job enforces its own wall

- **Derived ledger.** `budgets.fold(runs/*/usage.json, runs/*/run.json)` is the truth; `runs/ledger.json` is a cache with `derived_sha256` and a **monotonic high-water mark per key** (`attempts_seen`). The supervisor recomputes at every admission; if the derived attempts for a key are *fewer* than the high-water mark (a run directory vanished) or any `usage.json`/`run.json` is unparseable, the supervisor writes `HALT`, raises `ledger_corrupt`/`ledger_regressed`, and admits nothing until a human `so101-ack`s. Writes are atomic (`.tmp` + rename). **`pixi run lab import-run <dir> --budget-key <key>`** (human CLI) registers a run executed outside the lab (e.g. `pixi run train-sim-sac` by hand) so the ≤ 3-attempt rule cannot be bypassed by running outside the connector; `check_milestone_phase0.py` gains a row that every `runs/sac-sim/*` directory is registered.
- **Caps** come from `policy.yaml` and are the plan's: `s4.reach` 6 GPU-h, `s4.die_to_zone` 30 GPU-h, ≤ 3 attempts each; `sysid.fit` 2 GPU-h × 2; offline TD-MPC2 ≈ 1 GPU-h; arm time 8 h per algorithm (reach), 15 h (die); zero-shot 2 h each. Raising a cap is a §10 decision (`propose_decision`).
- **Accounting.** GPU-h = time holding the `gpu` lease **while progressing** (the progress-coupled heartbeat advanced within `stall_grace_s`), plus stalled time up to `stall_cap_min` (30) per segment. The job writes `usage.json` every 60 s and at exit; the supervisor's `status.json` samples are the cross-check. Probes (≤ 10 min GPU each, ≤ 1 GPU-h per task pool) do not count as attempts (§9 proposal P6, to confirm).
- **Hard wall, twice.** The supervisor writes `control/stop` + SIGTERM at the wall (120 s grace, then SIGKILL → `budget_exhausted`); the job stops itself at `hard_deadline_monotonic` as well, so a dead supervisor cannot let a run overspend.
- **Stall ladder** (progress heartbeat not advancing): event at `stall_flag_min` (30) → graceful stop at `stall_stop_min` (45) → kill at `stall_kill_min` (50) → `stopped: stall`/`crashed`. Long evals keep the heartbeat alive by calling `ctx.heartbeat(phase="eval")` inside their loop. A process whose heartbeat file is older than `heartbeat_dead_s` (120) with the pid alive is hung → SIGTERM/SIGKILL → `crashed`.
- **Disk.** Running check every 60 s: below `min_disk_free_gb`, or a run over its `disk_quota_gb` (plugin default; `fastsac_sim` 50 GB incl. videos/rollouts), → graceful stop (`stopped: disk`) + `disk_low` item; new admissions blocked while low.

### 3.6 Thermal: fail-closed, via NVML

- Sampler: `pynvml` (`nvidia-ml-py`, declared in `train` and `lab`) every 10 s: GPU temperature, memory used, utilisation, and the memory-junction temperature through `nvmlDeviceGetFieldValues(NVML_FI_DEV_MEMORY_TEMP)` when the device reports it (GDDR6X usually does; if unsupported, logged once and documented as unobserved in `runs/supervisor/status.json`).
- Rules (plan §10 2026-10-05: pause above 85 °C): GPU ≥ 85 °C or memory junction ≥ `mem_pause_c` (95, proposed) → **pause every GPU job** (checkpoint + exit; the `gpu` lease is held; nothing new is admitted); resume automatically at GPU ≤ 75 °C and memory ≤ 85 °C held for 60 s (new segment). **Fail closed:** `N = 6` consecutive failed samples (60 s with no reading) → the same pause for all GPU jobs plus a needs-human item `gpu_health_unknown`; the item auto-clears when readings return and the pause lifts; if unknown for > 30 min it stays until a human `ack`. A sim job that does not exit within `pause_grace_s` (60) after `control/pause` is SIGTERMed (checkpoint on SIGTERM is the plugin contract). A **real** job pauses at an episode boundary with the arm parked (§5.9); if the GPU stays hot past `real_pause_max_min` (20) the run ends (`stopped: thermal_gpu_timeout`) so the arm is not held energised waiting.

### 3.7 Review triggers: surfaced and policed, never acted on

| Run class | Allowed `review_policy` | Default | Unattended behaviour |
|---|---|---|---|
| sim `attempt` | `pause` \| `stop` \| `continue` | `pause` (P7, to confirm; `stop` until then) | `pause`: checkpoint + exit, leases released, event `review_wait`, dashboard "needs review"; the AI continues with `control_run(resume, note=…)` (rationale required). Second identical trigger pauses again; third stops the attempt (`stopped: review_repeat`). |
| sim `probe` / eval | `continue` \| `stop` | `continue` | flagged |
| **real** (any) | `stop` only | `stop` | finish the episode, park (§5.9), exit; `stopped: review:<name>`; needs-human item `review_real`; resuming needs a signed `ack` + a rationale |

No trigger changes the method, task, reward or plugin (plan §5; §10 2026-09-30). The residual-RL fallback is only reachable as `propose_decision`. **Vault note:** the vault's recipe-designer pattern auto-advances curricula/recipes on plateaus (`05-Wiki/concepts/Autonomous-ML-Training-Loop.md`); here plan §10 wins: the only automatic curriculum motion is FS8's stage widening *inside* a run, which the plan itself defines.

### 3.8 Unattended operation end to end

- **Pipelines.** `start_run(..., pipeline: {evaluate: {episodes, dr_stage, render, gate: "S4_exit"} | null, export: bool})`; on `finished` with green artifacts the supervisor queues the eval (same `budget_key`, class `eval`), then exports and registers the policy. `request_real_run(kind="zero_shot_eval")` chains the sim twin and `gap-report` as sim jobs.
- **D-EXIT-GATE stays half human.** The unattended pipeline ends with `gate.json.numeric.pass`, renders, `review_checklist.md`, the policy at `sim_gate_numeric_passed`, and a `gate_checklist` item; the human's **signed** `so101-gate-sign` makes it `sim_gate_passed`.
- **Nobody watching (sim):** triggers → §3.7; thermal/health → §3.6; budget → §3.5; crash/hang/stall → `crashed`/`stopped`, next job starts, the AI resumes later; disk → §3.5; suspend → `host_slept` + revalidation; reboot → `crashed` runs, voided real queue, the supervisor is back automatically (systemd) and resumes admission of sim jobs. Every event: an event row, a dashboard row, and for attempt-class terminal states and needs-human items a desktop notification.
- **Nobody watching (real arm):** §5.8 is the matrix; the short form: anything the servo classes as human-needed ends the run cleanly, raises an item, blocks real admissions, and leaves the arm in the servo's safe state; the servo's own automatic paths (thermal resume, SV1 idle rest and resume-on-request) continue; the session ends by its cap or by the SV1 idle timer and the subsequent heartbeat-timeout latch.
- **Needs-human queue:** `runs/needs_human/<item_id>.json` `{kind, run_id?, session_id?, reason, since, how_to_clear}`; kinds `servo_latched`, `servo_needs_human`, `servo_trip_fresh_process`, `arm_refused` (ARM refused: weld, undervolt, bus not dead), `rt_unavailable`, `review_real`, `gate_checklist`, `approval_pending`, `session_open_request` (press the button), `gpu_health_unknown`, `ledger_corrupt`, `ledger_regressed`, `disk_low`, `proposal_pending`, `hw_artifact_stale`, `supervisor_restarted`, `host_slept`. Servo-class, ledger and `gpu_health_unknown` (> 30 min) items clear only by a **signed** `so101-ack`.

## 4. MCP surface

### 4.1 Conventions

Server `so101-lab`; tool names `verb_noun`. Every tool: `outputSchema`, annotations (read-only `readOnlyHint: true, idempotentHint: true`; compute `destructiveHint: false`; real-arm and `control_run(stop)` `destructiveHint: true`). `response_format: concise | detailed`; paging (`cursor`, `page_size ≤ 50`); ≤ ~2k tokens by default, hard cap 8k with `truncated: true` (Claude Code caps MCP output at 25k). **Raw logs and video bytes never cross MCP.** **Untrusted text is marked as data:** notes, proposals, `review_checklist.md`, log tails and any AI-written field come back as `{"data": …, "trust": "untrusted:ai-written" | "untrusted:log"}` and the prompts say so. Real-arm tools are registered only when the server starts with `SO101_LAB_REAL_ARM=1` (the hardware-test-mcp `HW_TEST_READONLY` convention; vault "read-only first, motion last"); refusals are enforced in `lab.authority` regardless.

### 4.2 Tools — LA1 ships exactly these 12

| # | Tool | Class | Input → output |
|---|---|---|---|
| 1 | `get_status` | read | `{}` → system (GPU temps incl. memory, health state, supervisor alive/pid, leases, queue), arm session (`alive, backend, sm_state, parked, session_id, cap_min, elapsed_min, tier, rules_summary`), hw artifacts freshness, **needs-human items**, budgets remaining per key, `HALT`, `dashboard_url` |
| 2 | `find_runs` | read | `{task?, trainer?, state?: [..], since?, cursor?, page_size?}` → `{runs: RunRef[], next_cursor?}` |
| 3 | `get_run` | read | `{run_id, response_format?, metrics_last_n?: 20, wait?: {until: "state_change" \| "terminal" \| "eval" \| "pipeline_done", timeout_s ≤ 300}}` → run, budget, latest/best metrics, review state, servo view, approval state, pipeline, artifacts, `next_actions` (what is allowed now and why not otherwise), events tail; with `wait` it blocks until the condition or the timeout |
| 4 | `get_metrics` | read | `{run_id, keys?, last_n?: 50, downsample?: "auto" \| int, since_step?}` → `{columns, rows ≤ 200, n_total}` |
| 5 | `check_gates` | read | `{task, stage: "L1" \| "L2" \| "S4_exit" \| "S5_zero_shot" \| "tier1", policy_id?}` → `{pass, evidence[], missing[]}` (artifacts only; runs nothing) |
| 6 | `start_run` | sim-compute | `{trainer, task, config, init_from?, budget_class?, review_policy?, pipeline?, idempotency_key, notes?, dry_run?}` → `{run_id, state, plan, queue_position, est_start_min}`; refuses real-domain configs (`TIER_FORBIDDEN`) |
| 7 | `control_run` | sim-compute (`stop` destructive) | `{run_id, action: "pause" \| "unpause" \| "stop" \| "kill" \| "resume" \| "cancel", note, overrides?}` → `{run_id, state, segment}`; `resume` after a review pause requires `note`; `overrides ⊆ resumable_fields`; `cancel` for queued/awaiting entries |
| 8 | `evaluate_policy` | sim-compute | `{policy_id \| run_id, episodes ≤ 2000, dr_stage, render ≤ 16, goals_from?, gate?: "S4_exit", idempotency_key}` → `{eval_run_id}` |
| 9 | `export_policy` | sim-compute | `{run_id, ckpt?}` → `{policy_id}` |
| 10 | `run_gates` | sim-compute | `{task, stage: "L1" \| "L2", idempotency_key}` → `{run_id}` |
| 11 | `write_note` | disk | `{run_id \| "lab", text ≤ 4k}` → `{path}` |
| 12 | `propose_decision` | disk | `{title, proposal, evidence_run_ids, suggested_plan_row}` → `{proposal_id, path}` |

LA4 adds `request_real_run` (`{kind: "zero_shot_eval" \| "finetune" \| "sysid_excitation" \| "random_policy", task, policy_id? \| trainer+config?, episodes ≤ 100, speed_scale ∈ {0.5, 0.75, 1.0}, arm_min_cap ≤ 120, idempotency_key, notes}` → `{run_id, state, approval_request: {sha256, how_to_approve: "so101-approve <run_id>"}}`) behind the flag; `control_run(cancel)` covers requests. **Deferred until transcript evidence asks for them** (critic): `compare_runs`, `start_sweep`, `get_rollout_frames`, `tail_run_log`, `get_sweep`. Registries, events, plan decisions and the needs-human list are **resources**, not tools.

### 4.3 Resources

`so101://runs` (concise list), `so101://runs/{id}` (subscribable; `status.json` polled every 2 s), `so101://runs/{id}/{metrics.csv (path only > 2 MB), review.json, eval.json, gate.json, gate_sign.json, artifacts.json, summary.json, notes.md, review_checklist.md}`, `so101://runs/{id}/videos/{name}` (`{path, duration_s}`), `so101://runs/{id}/checkpoints` (list), `so101://policies`, `so101://policies/{id}`, `so101://datasets`, `so101://datasets/{id}`, `so101://trainers`, `so101://trainers/{name}/schema`, `so101://trainers/{name}/readme`, `so101://tasks`, `so101://tasks/{name}`, `so101://plan/decisions` (parsed §10, read-only), `so101://plan/budgets`, `so101://config/so101.yaml`, `so101://config/dr_stages.yaml`, `so101://calib/sim_params` (meta), `so101://calib/camera` (meta), `so101://lab/policy`, `so101://lab/status`, `so101://lab/needs_human`, `so101://lab/events?since=`, `so101://lab/sessions`, `so101://lab/audit?since=`.

### 4.4 Prompts (three)

`design_experiment(task, goal)`: task resource, trainer schemas, remaining budgets, L1/L2 status, binding §10 rows, §5 wall-clock estimates, L1–L13, required output shape (hypothesis, config diff, budget class, review policy, pipeline, stop rule). `diagnose_stalled_run(run_id)`: `get_run` + 50 rows of SAC internals + `review.json` + the trigger glossary + the known failure modes (L2 gaming, L4 clamps, L12 silent failures). `review_before_real(policy_id)`: policy manifest, `gate_sign.json`, `eval.json`, `check_gates(S5…)`, the HS4 protocol, the D-ZEROSHOT ladder, the tier rules; output = the `request_real_run` text or the blocking reasons.

### 4.5 Error model

Tool-execution errors (`isError: true`) with `{code, message, hint, details}`; codes `CONFIG_INVALID`, `UNKNOWN_TRAINER`, `UNKNOWN_TASK`, `BUDGET_EXCEEDED`, `ATTEMPTS_EXHAUSTED`, `GATE_FAILED`, `PREREQUISITE_MISSING`, `TIER_FORBIDDEN`, `APPROVAL_REQUIRED`, `APPROVAL_INVALID` (expired / consumed / hash mismatch / bad signature / session changed), `LEASE_BUSY`, `RUN_NOT_FOUND`, `STATE_CONFLICT`, `SUPERVISOR_DOWN`, `HARDWARE_UNAVAILABLE`, `SESSION_RULE` (quiet hours, min gap, cap, allow-list, per-day cap), `HALTED`, `DISK_LOW`, `RESPONSE_TOO_LARGE`. Never JSON-RPC protocol errors for domain failures.

### 4.6 Long-running calls and cancellation

Start → poll (`get_run`, `get_run(wait=…)` ≤ 300 s, resource subscriptions). MCP Tasks (SEP-1686) not relied on. A client `notifications/cancelled` stops a wait and nothing else; stopping a job is always the audited `control_run(stop)`.

## 5. Safety and authority model

Principle: everything the AI may do is decided by code below the tool boundary; everything the arm may do is decided by the layers below that (ServoCore, the MCU, the e-stop). The connector adds no safety layer and removes none.

### 5.1 The authority table (`lab/authority/decide.py`; each row with a shown-to-fail test)

| Action | SIM / offline | REAL, Tier 0 | REAL, Tier 1 |
|---|---|---|---|
| Start a training / eval / gate job | config valid; L1+L2 PASS for S4; `sim_params` rule; ledger; disk; not `HALT`; GPU health known | n/a | n/a |
| `request_real_run` | `TIER_FORBIDDEN` for sim/offline plans | → `awaiting_approval`; needs: hw artifacts fresh (H2, H3, H5, H7, H11; die adds H8, H9, the verifier false-accept artifact), signed `tier.json` tier ≥ 0, policy status, arm-time ledger, speed-ladder order, an open session (§5.7) in RUN/HOLD/parked, no open servo-class item; then a **signed approval** | as Tier 0 **plus** signed `tier.json.tier == 1` with soak ids verified (§5.4) and a `rules` block; admitted without a per-run approval only inside the rules (allow-list, cap, quiet hours, min gap, per-day minutes); otherwise `awaiting_approval` |
| Start the approved/admitted real job | — | supervisor verifies the signature, expiry, nonce, request hash, session id at start; the job verifies again before its first command | re-verifies the rules, the session, the soak report and the needs-human queue at start and before every episode batch |
| `control_run(stop/pause)` | allowed | allowed (STOP_REQ via the job) | allowed |
| Resume a real run | — | a new signed approval | only after a clean end and no open item; after any trip, a signed `ack` first |
| Write `config/so101.yaml`, `calib/*`, `sim/dr_stages.yaml`, `lab/policy.yaml`, `runs/hw/*`, the plan, `docs/ARCHITECTURE.md`, `.mcp.json`; create an approval, ack, gate signature, §10 row or tier record | **never** (no tool; `WriteGuard` allows the lab processes only `runs/<id>/` minus signed records, `runs/{queue,idem,policies,datasets,needs_human (items only),audit.jsonl,ledger.json,supervisor,servo,perception,sessions.jsonl}` and the proposals inbox) | never | never |

### 5.2 Signed human records: approvals (`so101-approve <run-id>`)

Requirements (binding): local terminal command, run summary, typed confirmation, single-use, time-limited, verified by the connector; the AI/MCP server can never create one; root-owned tools outside the repo; repo code never runs in the signing terminal.

1. **Setup (one `sudo`).** `pixi run lab init-approver` generates a passphrase-protected Ed25519 key `~/.config/so101_rl/approval/id_ed25519` (`ssh-keygen -t ed25519`) and prints the install command. `sudo so101-lab-install` (a script the user reviews first) writes, root-owned, `0755`/`0644`: `/usr/local/lib/so101_rl/{approve,gate_sign,ack}.py` (stdlib-only Python 3, no imports from the repo, ~150 lines each), `/usr/local/bin/so101-{approve,gate-sign,ack}` (shell shims calling `/usr/bin/python3 -I`), and `/etc/so101_rl/approval_allowed_signers` (OpenSSH `allowed_signers`, principal `so101-approver`, the public key). Their sha256 sums are recorded in `docs/HARDWARE_CHECKLIST.md` for the human to verify with `sha256sum`. **The passphrase lives in the human's head.**
2. **Request.** `request_real_run` writes `runs/<id>/approval_request.json` = machine fields `{run_id, kind, task, trainer, policy_id, policy_sha256, config_sha256, episodes, speed_scale, arm_min_cap, hw_artifacts: {id: path, mtime, sha256}, session_id_required, ai_notes (string, untrusted)}`.
3. **Command** (`so101-approve <run_id>`): refuses without a controlling TTY (`os.isatty(0)` and `/dev/tty` opens); reads the request; re-checks what it can without repo code (artifact files exist and are fresh by mtime, `tier.json` + `.sig` verify, the servo summary/pid file shows a live session); renders **machine fields first**, then the diff against the last approved request of the same kind, then — in a boxed section headed **"AI-written text (untrusted)"**, ≤ 20 lines, or omitted with `--no-ai-text` — the `ai_notes`; asks for the typed phrase **`APPROVE <run_id> <episodes>x<speed_scale>`** (the key parameters must be retyped, so a misread summary cannot be confirmed by reflex); builds `approval.json` `{run_id, request_sha256, approved_by, approved_at, expires_at (+2 h), nonce (128-bit), session_id, limits: {episodes, arm_min_cap, speed_scale_max}}` and signs it: `ssh-keygen -Y sign -f ~/.config/so101_rl/approval/id_ed25519 -n so101-approval` (passphrase prompted on `/dev/tty`). Writes `approval.json` + `.sig` mode `0444`.
4. **Verification** (`lab.authority.records.verify`, supervisor at admission + job before its first command): `ssh-keygen -Y verify -f /etc/so101_rl/approval_allowed_signers -I so101-approver -n so101-approval -s approval.json.sig`; hash equals the current request; not expired; `session_id` equals the live session; nonce unconsumed (`approval.consumed` created `O_EXCL` at start). A resume needs a new approval.
5. **What this protects against, and what not.** It protects against a same-user agent (including one driving a shell) **forging** a record (needs the passphrase), **substituting** the trust root or **altering the signing screen** (both root-owned), and against the lab processes writing records (no code path; `WriteGuard`). It does **not** protect against: an agent editing the *verifier* or the job code in the repo and restarting the supervisor (that is code review's job: human-reviewed commits, the protected-path guard of §8.6, and the ARCHITECTURE sync rule); an agent with `sudo`; a human approving a misread summary (mitigated by machine-fields-first and the retyped parameters); or anything physical. The layers that hold regardless are the button (now time-bound by FW1), the e-stop, the deadman and ServoCore's envelope.

Client-side, Claude Code's permission system is set to **always ask** for `mcp__so101-lab__request_real_run`; the `_meta["anthropic/requiresUserInteraction"]` annotation (vault 2026-08-02) is used if the installed client honours it. Elicitation is not relied on.

### 5.3 Signed gate ticks and acknowledgements (`so101-gate-sign`, `so101-ack`)

- **`so101-gate-sign <eval_run_id>`** renders `eval.json` (machine fields: n, successes, Wilson, final error, trips) and the path of `renders/` and `review_checklist.md` (checklist text shown under the untrusted banner), asks for **`SIGN <eval_run_id> <success_pct>`**, and writes `gate_sign.json` `{eval_run_id, gate: "S4_exit", eval_sha256, checklist_sha256, success, wilson_lo, signed_by, at}` + `.sig`. **`so101-gate-sign --tier runs/hw/tier.json`** signs the tier record (promotion and every session-cap raise are gates: `tier.json.sig`). A policy is `sim_gate_passed` only when `gate.json.numeric.pass` **and** a verifying `gate_sign.json` exist; `tier.json` without a verifying `.sig` is treated as tier 0 with no rules.
- **`so101-ack <item_id>`** renders the item and the servo summary (final `sm_state`, `fault_bits` names, MCU status, tick-log tail stats), asks for **`ACK <item_id> <kind>`**, and writes `<item_id>.ack.json` `{item_id, kind, run_id, session_id, servo_summary_sha256, note, signed_by, at}` + `.sig`. An item is cleared only by a verifying ack.
- Same key, same trust root, same TTY rule, same threat model as §5.2.

### 5.4 The Tier 1 record, its rules, and the session cap ramp

`runs/hw/tier.json` (H13, now **signed**) carries:

```json
{"tier": 1, "by": "koen", "date": "...", "milestone_sha": "...",
 "soak_reports": {"attended_30min": "runs/soak-<ts>/report.json", "goal_directed_120min": "runs/soak-<ts2>/report.json"},
 "rules": {"tasks": ["reach"], "kinds": ["sysid_excitation", "zero_shot_eval", "finetune"],
           "trainers": ["sb3_sac_her", "tdmpc2_finetune"], "policies": ["<policy_id>", "..."],
           "session_cap_min": 60, "clean_sessions_at_cap": 0,
           "max_episodes_per_run": 100, "min_gap_min": 2, "quiet_hours": ["23:00", "07:00"],
           "max_arm_min_per_day": 240, "day_boundary": "local-midnight",
           "speed_scale_max": 1.0, "require_rt_active": true, "require_camera_moved_gate": true,
           "valid_until": "2026-12-31"}}
```

- **Verified at promotion and at every unattended admission:** `check_soak_report(strict=True)` on the `goal_directed_120min` report (≥ 120 min, goal-directed, p99.9 jitter < 5 ms, no heartbeat gap > 200 ms, no HOLD entry, no thermal trip, no calibration mismatch, `rt_active`), produced by a servo build whose `so101.yaml` sha and calibration sha equal the current ones; a mismatch makes the record invalid (needs a new soak and re-signing).
- **Session cap ramp (binding).** `session_cap_min` starts at **60**; the human may raise it to 120, then 240, then 480, each time only after `clean_sessions_at_cap ≥ 1` at the current cap, by editing and re-signing `tier.json`. A **clean session** (proposed definition, P4): ended by its cap, by the SV1 idle timer + heartbeat-timeout latch, or by an admitted run's clean end; no `_FRESH_PROCESS_CAUSES` or `_HUMAN_CAUSES` trip; thermal trips ≤ 1 and all auto-resumed; no `LATCHED` other than the end-of-session timeout latch; the servo summary's jitter within the soak gate; the post-session **power-cycle + weld check** passed (§5.7). `runs/sessions.jsonl` records every session's verdict; the lab counts `clean_sessions_at_cap` from it and refuses a raise that skips a rung.
- **Allow-list and limits** are enforced by `lab.authority` with `SESSION_RULE` errors: tasks/kinds/trainers/policies; `quiet_hours` (local time; no run *starts* inside them; a running run finishes its episode and parks at the boundary); `min_gap_min` between consecutive real runs; `max_arm_min_per_day` where arm minutes = minutes the servo spent energised (RUN/HOLD incl. parked) summed over sessions whose start falls in the local calendar day (`day_boundary: local-midnight`, host timezone); `speed_scale_max`.

### 5.5 `lab/policy.yaml` (repo-owned, strictly loaded)

Budget caps per key; `probe {max_gpu_h_each: 0.17, pool_gpu_h_per_task: 1.0}`; `thermal {gpu_pause_c: 85, gpu_resume_c: 75, mem_pause_c: 95, mem_resume_c: 85, sample_s: 10, fail_closed_samples: 6, pause_grace_s: 60, real_pause_max_min: 20, unknown_needs_human_min: 30}`; `heartbeat {dead_s: 120}`; `stall {flag_min: 30, stop_min: 45, kill_min: 50, cap_min: 30, grace_s: 90}`; `graceful_stop_s: 120`; `disk {min_free_gb: 20, check_s: 60}`; `queue_ttl_h {sim: 24, real: 4}`; `vram_fraction {sim_training: 0.6, offline: 0.6, real: 0.3}`; `review {sim_attempt: stop (pause after P7), sim_probe: continue, real: stop, max_pauses_per_attempt: 2}`; `approval {expiry_h: 2, allowed_signers: /etc/so101_rl/approval_allowed_signers}`; `hw_artifacts {max_age_days: 90, required: {…}}`; `session {arm_timeout_s: 110, perception_restart_max: 1, weld_check: true}`; `notifications {desktop: true}`; `dashboard {port: 6007}`; `response {max_tokens: 8000}`.

### 5.6 The connector-started feetech servo: pinned argv, clean environment, pre-checks

The session manager builds the argv with a pure function `session.servo_argv(session)` whose output is **pinned by a test** (exact tokens; never `--no-mcu`; always `--require-rt`, `--arm-timeout-s`, `--max-ticks`, `--tick-log`, `--summary`):

```
pixi run -e robot python -m so101_rl.servo.main --backend feetech
  --config <repo so101.yaml> --calibration <live calibration path from config>
  --heartbeat-port <from config> --port <from config> --shm-name <from config>
  --require-rt --arm-timeout-s 110
  --max-ticks <session_cap_min × 60 × servo_hz + 2 × bound_s × servo_hz>
  --tick-log runs/servo/<session_id>/tick_log.bin --summary runs/servo/<session_id>/summary.json
```

- `--arm-timeout-s 110` < FW1's 120 s: if no button press arrives, the servo exits 2 ("MCU never armed") before the MCU would latch on its own; the session request expires with item `session_open_request` closed as `timed_out`.
- `--max-ticks` is a **supervisor-independent session bound**: the servo exits through its normal shutdown (safe-state sequence) at the cap even if the supervisor is dead; the margin covers two bounded sequences.
- **Pre-checks** before launching: `resource.getrlimit(RLIMIT_RTPRIO)` soft ≥ 50 and `RLIMIT_MEMLOCK` ≥ 512 MB in the supervisor's environment (the service unit sets `LimitRTPRIO=50`, `LimitMEMLOCK=infinity`); otherwise no launch and item `rt_unavailable`. `REQUIRE_RT=1` in the child environment.
- **Clean environment:** `env = {PATH, HOME, LANG, PYTHONNOUSERSITE=1, REQUIRE_RT=1, SO101_SERVO_CPU (from policy)}`; stdin `/dev/null`; stdout/stderr to files; `start_new_session=True`; the child's pid and `/proc/<pid>/cmdline` recorded in `runs/servo/<session_id>/pid.json`; the servo's own single-instance locks stay authoritative.
- **Perception process** (die tasks; reach and sysid do not need it): launched by the session manager alongside the servo: `pixi run -e robot python -m so101_rl.perception.main --serial <from camera.yaml> --camera calib/camera.yaml --hand-eye calib/hand_eye.yaml --fiducial calib/fiducial.yaml --shm-name <object table>`. Work item **PE1**: `perception/main.py` does not exist yet (U5 built the modules); it publishes the object table and the fiducial verdict at ≥ 5 Hz, logs to `runs/perception/<session_id>/`, and is restarted at most once per session on crash; a die-task job refuses to start without an object table younger than 1 s and a fiducial OK. `deploy_policy`'s per-episode fiducial gate is unchanged.

### 5.7 Sessions: opening, the single writer, closing, the power-cycle check

- **Opening.** A real request that is admitted (Tier 1) or approved (Tier 0) while no session exists makes the session manager launch perception (if needed) and the servo; the servo waits in BOOT for the button (its operator message every 10 s); the dashboard and a desktop notification show item `session_open_request` ("press the MCU button within 110 s; session cap N min"). The human presses; ARM; one good read/write/ACKED heartbeat within 0.8 s; re-energise; RUN. `session_id = <servo pid>:<ARM timestamp>` from the summary/state record. A refused ARM (weld reason 7, undervolt, servo-side bus not dead) → exit 2 → item `arm_refused`.
- **Single writer on the shm command block.** `runs/servo/<session_id>/cmd_owner.lock` (`flock`): the attached job holds it while it writes commands; the session manager writes only when it holds the lock (no job attached) — for `PARK` right after a run detaches without parking, and for `STOP_REQ` at session end. A supervisor-initiated end while a job is attached goes through `control/end_session`: the job finishes its episode, parks, sends `STOP_REQ`, releases the lock.
- **Between runs.** The job's last act is `PARK` (SV1): the servo holds at the latched `home` target, energised; after `idle.park_timeout_s` (default 600) the servo runs the safe-state sequence to rest and turns torque off (fault `IDLE_TIMEOUT`); the next admitted job sends `RESUME_REQ` and the servo performs the **full** re-energise (ping, calibration registers, register write/read-back, start-pose rule, goal = present, torque on, ramp home) before RUN.
- **Closing.** At `session_cap_min` (the manager, via the lock holder) or when `--max-ticks` fires (servo-side), or when a human/fresh-process trip occurs: safe-state → torque off → process exit → heartbeats stop → MCU LATCHED (reason 1) → relay open. **Power-cycle + weld check:** the manager reads the final servo summary (last MCU status LATCHED, relay off) and, before the *next* session is allowed, requires that the MCU's ARM precondition "servo-side bus dead (< 1 V)" held (a weld would have latched reason 7 or made ARM refuse) — i.e. the next ARM succeeding *is* the weld check passing, and a refused ARM is item `arm_refused`. `runs/sessions.jsonl` gets the verdict (§5.4 clean-session definition).

### 5.8 Unattended real-arm operation: the failure matrix

| Event | Servo (unchanged, except SV1 rows) | Connector | Run state → item |
|---|---|---|---|
| Thermal trip (60 °C) during a run | SAFE_STOPPING → rest → TORQUE_OFF → dwell → CHECK → full re-energise → RUN; ≤ 3/h, then a human | episode `truncated`; `ServoNotRunning` → `paused: servo_wait` (arm lease held; arm-time clock stopped); on RUN the job re-creates the env (HOLD, bounded deltas) and continues as a new segment | `paused` → `running`; after the 4th trip the job exits: `stopped: servo_needs_human` → item |
| Thermal trip while **parked** (SV1) | SAFE_STOPPING → rest → TORQUE_OFF, fault THERMAL \| IDLE_TIMEOUT; **no automatic re-energise while parked**; the next `RESUME_REQ` re-energises only after the dwell and inside the hourly budget | nothing to do; the next admitted job's `RESUME_REQ` may be refused until the dwell passes (the job waits ≤ `resume_wait_min`, then `stopped: servo_wait_timeout`) | — |
| Idle timeout (SV1) | parked HOLD → SAFE_STOPPING → rest → TORQUE_OFF (IDLE_TIMEOUT); heartbeat and goal = present continue | event `session_idle_rest`; the next job sends `RESUME_REQ` | — |
| NaN / stuck / OOD / RANGE / BOX_PRESENT / N stale commands | SAFE_STOPPING → TORQUE_OFF; fresh process | episode `truncated`; checkpoint; exit cleanly | `stopped: servo_trip:<cause>` → `servo_trip_fresh_process`; real admissions blocked until a signed ack; the session is over (a fresh process = a new session = the button) |
| Protection bit / calibration mismatch / user stop | SAFE_STOPPING → TORQUE_OFF; human | as above | `stopped: servo_needs_human` → item |
| Heartbeat failure / MCU latched / deadman / e-stop | LATCHED | `ServoNotRunning`; exit; session marked `latched`; approvals and admissions bound to it void | `stopped: servo_latched` → item (physical reset) |
| Job halt rule (S5.1) | — | park, exit | `stopped: halt_rule:<name>`; fiducial → `hw_artifact_stale` |
| Real review trigger | — | `stop`: finish episode, park, exit | `stopped: review:<name>` → `review_real` |
| Arm-time budget / per-day cap / quiet hour reached | — | finish episode, park, exit | `budget_exhausted` / `stopped: session_rule` |
| GPU ≥ 85 °C or health unknown during a real run | — | pause at episode boundary, parked; after `real_pause_max_min` end the run | `paused` → `running` or `stopped: thermal_gpu_timeout` |
| Job crash / `kill -9` while attached | holds last target; after 5 misses SAFE_STOPPING → TORQUE_OFF (STALE_CMD, fresh process) | dead pid → `crashed`; servo state read → fresh-process item | `crashed` → `servo_trip_fresh_process` |
| Supervisor crash | — | systemd restarts it; it re-adopts jobs by pid/cmdline/boot_id; the servo's `--max-ticks` bounds the session regardless | event `supervisor_restarted` |
| Host suspend | heartbeats stop → LATCHED | `host_slept`: session marked latched, approvals void, runs `crashed`/revalidated | items `servo_latched`, `host_slept` |
| Host reboot / power loss | LATCHED | every `running` run `crashed`; real queue voided | items |
| Session cap (`--max-ticks` or the manager) | shutdown → safe-state → torque off → exit → LATCHED (timeout) | records the session; weld check at the next ARM | event `session_ended:cap` |

Every row ends with the arm in a servo-defined safe state, a terminal or paused run state, an audit row, a dashboard entry and (for items) a desktop notification. No row restarts real motion without the servo's own automatic paths or a human act (button, signed ack, signed approval).

### 5.9 Work item SV1 — servo-side park, idle rest and resume-on-request (the one `safety/`/`servo/` change; user decision 8)

**Why.** Today a client that stops commanding trips STALE_CMD after 5 missed windows (SAFE_STOPPING → TORQUE_OFF, resume = a fresh process), so an arm cannot wait between runs inside one session. SV1 adds an explicit *parked* mode so waiting is a declared state, not a missed-command fault, and a resume path that reuses the full checked re-energise.

**Interface changes (frozen once built; ARCHITECTURE §5/§6 in the same commit).**
- `servo/shm.py`: `CmdMode.PARK = 4` ("I am detaching; hold the current latched target, start the idle timer") and `CmdMode.RESUME_REQ = 5` ("an admitted client asks for a re-energise from an idle rest"). F3 layout unchanged (the mode field already holds a byte); command-block version unchanged, documented.
- `safety/types.py`: `TripCause.IDLE_TIMEOUT` (new bit), `ResumePolicy.ON_REQUEST`, `RESUME_BY_CAUSE[IDLE_TIMEOUT] = ON_REQUEST`. `SmState` unchanged (frozen F5 enum; parked is a flag on HOLD, idle rest is TORQUE_OFF with cause IDLE_TIMEOUT).
- `config/schema.py` + `so101.yaml`: strict `idle:` section `{park_timeout_s: 600 (30..3600), resume_requires_dwell: true (const)}`; `tests/test_config_single_source.py` extended.
- State record (F3) layout **v3**: `parked` (u1) and `resume_refused_count` (u1) in the remaining alignment padding; size/stride/other offsets unchanged; `attach` refuses v2. Tick log gets the same two fields.

**State-machine edges (added to the table of `safety/state_machine.py`; first-match order kept; every existing row unchanged).**

| from | condition | to | resume |
|---|---|---|---|
| RUN / HOLD | `PARK` command received, `block_ok` | HOLD, `parked = True` (target = latched last RUN target) | — |
| HOLD(parked) | the stale-command and `consecutive_misses` rows **do not apply**; HOLD duty unchanged (re-write latched target, heartbeat every tick) | HOLD(parked) | — |
| HOLD(parked) | fresh RUN command, age ≤ `stale_cmd_ms`, `block_ok` | RUN (resync), `parked = False` | — (existing HOLD → RUN row) |
| HOLD(parked) | `parked_ns ≥ park_timeout_s` | SAFE_STOPPING (to rest, bounded by `bound_s`), fault `IDLE_TIMEOUT` | on request |
| HOLD(parked) | THERMAL in trips | SAFE_STOPPING → TORQUE_OFF, fault THERMAL \| IDLE_TIMEOUT; **`may_resume_thermal` does not re-energise while `parked`** | on request, after the dwell, inside `max_trips_per_hour` |
| TORQUE_OFF | faults ⊆ {IDLE_TIMEOUT, THERMAL, STALE_CMD-from-parked}; THERMAL (if set) has `may_resume_thermal`; `RESUME_REQ` received; `block_ok`; thermal trips this hour ≤ max | CHECK (`reenergize_requested`) → core runs the **full** `reenergize` (ping all, calibration registers, Lock = 0 / Torque_Enable == 0 check, write + read back every register, block read with the start rule *rest within tolerance or tip ≥ 1 cm above the table*, goal = present, read back, torque on, ramp home through the transit zone) → `mark_energized` → RUN (resync on the first good read) | — |
| TORQUE_OFF | `RESUME_REQ` with any `_FRESH_PROCESS_CAUSES` / `_HUMAN_CAUSES` fault, or in LATCHED / BOOT / CHECK / SAFE_STOPPING | ignored; `resume_refused_count += 1`; one stderr line per second at most | fresh process / human (unchanged) |

**Obligations kept (each with a test):** the heartbeat is sent every tick in every state including parked HOLD and idle TORQUE_OFF (TORQUE_OFF already writes goal = present and heartbeats); `resync(present)` on the step that enters RUN; the re-energise is the existing `reenergize()` with `should_abort`, the strict ACKED heartbeat, `hand_over_on_abort`, and `UnsafeStartPose` → `notify_reenergize_failed` → TORQUE_OFF (human); the safe-state sequence stays bounded by `bound_s`; a stop signal or STOP_REQ during a parked wait or a resume goes through the existing paths; `tick()` still allocates nothing (tracemalloc test); the thermal trip count is still the machine's; the batched sim chain is untouched (D-CHAIN: parking is a state-machine matter, not a clamp).

**Tests.** `tests/test_state_machine.py`: parked HOLD never trips STALE_CMD over 10 000 ticks without commands; timeout → SAFE_STOPPING → TORQUE_OFF with IDLE_TIMEOUT; `RESUME_REQ` → CHECK exactly once and only with `block_ok`; refused for NAN/USER_STOP/LATCHED; thermal-while-parked does not auto-re-energise, `RESUME_REQ` honoured only after the dwell and within the hourly budget. `tests/test_servo_core.py` / `test_reenergize_backends.py`: on FakeBus and MujocoBus, PARK → wait → rest reached and torque off with ≤ 5 ticks of motion after torque-off (the H5 criterion, simulated) → `RESUME_REQ` → full re-energise → RUN; heartbeat gap never > 200 ms across a 15-minute fake session with two park/resume cycles; a `RESUME_REQ` from a pose below the 1 cm floor is refused (`UnsafeStartPose`). `tests/test_soak.py`: the soak harness gains an optional parked segment so the Tier 1 120-minute soak can include park/resume cycles. **Hardware (attended, Tier 0, before any Tier 1 use):** HS8 — a 60-minute parked session on the real arm with three park → idle-rest → resume cycles, foam under the arm; artifact `runs/hw/park-resume-<ts>.json {cycles, moved_ticks_after_torque_off ≤ 5, resume_ok}`.

**ARCHITECTURE deltas:** §3 "When the policy is late" gains "unless parked"; §5 commands (`PARK`, `RESUME_REQ`), state record v3; §6 layer 5 table gains the IDLE_TIMEOUT row and the parked-thermal rule; §10 row. Plan §10 row P3.

### 5.10 Work item FW1 — firmware arm timeout (user decision 9)

- **Firmware (`firmware/watchdog_mcu/main.py` Core):** in DISARMED, if no accepted ARM arrives within **120 000 ms** (wrap-safe `ticks_diff`, `>=`) of entering DISARMED (the button release that cleared the latch), go LATCHED with new reason **8 = ARM_TIMEOUT** (the status byte's 4-bit reason field has room). A later button press returns to DISARMED and restarts the window. No other transition changes.
- **FakeMcu** byte-identical; `test_vectors.json` gains: 119 999 ms → DISARMED; 120 000 ms → LATCHED reason 8; ARM at 119 000 ms → ARMED; button after the timeout → DISARMED. `PROTOCOL.md`: the transition row, the reason code, the example status byte (`0x82`). Host `McuStatus` reason decoding gains `ARM_TIMEOUT`; `servo/main.py --arm-timeout-s` documents that it should stay below 120 s (the session manager uses 110). `docs/HARDWARE_CHECKLIST.md` H3 gains the timeout test (press, wait 2 min, verify LATCHED reason 8). ARCHITECTURE §6 layer 2 and the trip-reason table. Plan §10 row P4.
- **Meaning.** A button press is consent for at most two minutes; a session the human opens stays powered across runs through SV1, and an unattended arm never sits DISARMED-but-unarmed indefinitely.

### 5.11 Audit log and kill switches

`runs/audit.jsonl`, append-only: every MCP tool call, CLI action, signed record verification, supervisor and session-manager decision (`admitted`, `refused:<code>`, `paused:<reason>`, `stopped:<reason>`, `session_opened/closed`, `ark`-level MCU events), with `{ts, actor, action, args_sha256, args_redacted ≤ 1 kB, run_id?, session_id?, result, duration_ms}`. Compute kill: `control_run(kill)` and `runs/HALT` (`pixi run lab halt`): nothing new starts, running jobs stop gracefully, a real job sends `STOP_REQ`. Arm stop: the e-stop and the deadman; an MCP stop on a real job is counted as **zero** safety value (`Arm-Safety-Envelope.md`).

### 5.12 What the AI can never do

Edit safety, servo or firmware code or config; write calibration files or `sim_params.yaml`; change budgets, thermal thresholds, halt rules, the speed ladder or tier rules; create an approval, an ack, a gate signature, a tier record or a §10 row; disable TensorBoard/metrics; start the feetech servo with a different argv or `--no-mcu`; write the shm command block without the cmd-owner lock; move the arm outside a whole, admitted procedure; run arbitrary code or shell through the connector; receive raw logs or video bytes through the connector; resume a tripped real-arm run without a signed ack; lower the attempts count (the ledger high-water mark halts the lab).

## 6. Experiment management for an AI user

**Summaries.** `summary.json` at exit (final/best success, final error p50/p90, GPU-h by progress, env-steps/s, DR stage, triggers, verdicts); `get_run` shows it; comparisons are done by the AI from summaries via resources until transcript evidence justifies a `compare_runs` tool.

**Stage control.** `dr.start_stage`, `dr.max_stage`, `curriculum.*` within FS8 bounds; `control_run(resume, overrides={"/dr/max_stage": 3})` re-stages a paused run as a new segment. No live mutation.

**Gates as first-class actions.** `run_gates(task, L1|L2)` → `runs/gates/*.json`; `evaluate_policy(gate="S4_exit")` → `gate.json` numeric; the human's signed `so101-gate-sign` → `sim_gate_passed`; `request_real_run(kind="zero_shot_eval")` runs the HS4/HS6 protocol and chains the sim twin + `gap-report`; `zero_shot.pass = wilson_lo ≥ 0.60` gates `kind="finetune"` (D-ZEROSHOT); `check_gates(tier1)` shows the signed tier record and the 120-minute soak verification.

**Notebook and proposals.** `write_note` → `runs/<id>/notes.md` (returned as untrusted data). `propose_decision` → `plans/proposals/<date>-<slug>.md` (the only repo-tree path the connector creates files in; P9): title, the proposed §10 row, evidence (run ids + summaries + provenance), alternatives, `status: proposed`; a §10 row citing the file marks it accepted. No code path writes the plan.

## 7. Future model types: generic in shape now, nothing built

Datasets: `runs/datasets/<id>.json` `{dataset_id, kind, source_runs, paths, n, obs_layout_version, action_convention: "joint_delta_6_applied", checks, provenance}`; kinds now `rollout_chunks` (`check_chunk`), `episodes_h5` (`check_episode`), `sysid_tick_logs`. Deferred, plugin-side: kind `lerobot` (LeRobotDataset v3; `lerobot` 0.6.1 is a dependency) with an `episodes_h5 → lerobot` converter; `pixel_episodes`. Later plugins (ACT/diffusion BC: `offline`, `inputs=("lerobot",)`, `role="accelerator"`, needs a policy adapter in `rl/policy_export.py` v2 + `deploy_policy`; state/pixel world model: `outputs=("world_model",)`, `evaluates` probe error; VLA fine-tune: `gpu="required"`, `inputs=("lerobot",)`) register in the in-repo table without connector changes. Plan §1 constraint 1 stays visible through `role`.

## 8. Dashboard, packaging, registration, testing, observability

### 8.1 The local web dashboard (`so101_rl.lab.dashboard`)

stdlib `ThreadingHTTPServer` on `127.0.0.1:6007`, GET only (405 otherwise), one static HTML page (vanilla JS, 5 s polling) over `/api/{status,runs,runs/<id>,runs/<id>/metrics,events,needs_human,approvals,sessions,ledger}.json` and `/files/<run_id>/<path>` (resolved under `runs/`, `..` rejected). Page: **needs-a-human banner** (kinds, `how_to_clear`, the exact `so101-approve <run_id>` / `so101-ack <item_id>` lines; flashing for servo-class items), **session panel** (state, parked, cap/elapsed, arm minutes today, next quiet hour, "press the button" prompt with its countdown), **runs** (state, progress, GPU-h by progress, ETA, pipeline stage, last trigger, servo state), **live metrics summary** (last row + sparkline), **events** (triggers, trips with `TripCause` names, thermal incl. memory temperature, MCU/latch, approvals consumed, session open/close), **system** (GPU temps, health state, supervisor, leases, queue, disk, halt), links to TensorBoard and each run's videos/renders/checklist. Alerts: browser `Notification` API for new items; `notify-send` from the supervisor (needs `DBUS_SESSION_BUS_ADDRESS` in the unit environment; degrades silently). A `systemd --user` unit, read-only, crash-isolated from jobs.

### 8.2 Pixi, services, declared dependencies

- Feature `lab`: `mcp = "2.3.*"`, `jsonschema`, `nvidia-ml-py` (all conda-forge); `ffmpeg` (conda-forge) declared explicitly in `train` (viz already depends on it implicitly). Environments: `default = [sim, perception, train, gpu, lab]`, `cpu = [sim, perception, train, cpu, lab]`; `robot` unchanged (the servo and perception processes run there).
- Host requirements (documented, checked by `lab init-approver` / `lab status`): OpenSSH ≥ 8.0 (`ssh-keygen -Y`), `/usr/bin/python3` for the root-owned tools, `libnotify` (`notify-send`, optional), systemd user sessions with `loginctl enable-linger $USER`, `LimitRTPRIO`/`LimitMEMLOCK` in the servo path.
- Tasks: `lab-mcp`, `lab-supervisor`, `lab-dashboard`, `lab` (`status`, `halt`, `unhalt`, `audit`, `import-run`, `init-approver`, `install-units`), `test-lab`. **No `approve`/`gate-sign`/`ack` pixi tasks** (they are root-owned commands outside the repo). `lab install-units` writes `~/.config/systemd/user/so101-lab-{supervisor,dashboard}.service` and prints the `systemctl --user enable --now` lines.
- Jobs spawn as `pixi run -e <plugin.env_name> python -m so101_rl.lab.job --run runs/<id>`; the job records `os.getpid()`.

### 8.3 Claude Code registration (`.mcp.json`, committed; real-arm tools off by default)

```json
{"mcpServers": {"so101-lab": {"command": "pixi", "args": ["run", "-e", "cpu", "lab-mcp"],
  "env": {"SO101_LAB_ROOT": ".", "SO101_LAB_REAL_ARM": "0", "SO101_LAB_LOG": "runs/lab-mcp.log"}}}}
```

Recommended `.claude/settings.json`: `permissions.ask` for `mcp__so101-lab__request_real_run` and `mcp__so101-lab__control_run`; `permissions.deny` for `Edit`/`Write` on `calib/**`, `src/so101_rl/config/so101.yaml`, `src/so101_rl/lab/policy.yaml`, `src/so101_rl/safety/**`, `src/so101_rl/servo/**`, `firmware/**`, `runs/hw/**`.

### 8.4 Testing strategy (no hardware; `pixi run -e cpu test-lab` is part of `test`)

| Test file | Proves |
|---|---|
| `test_lab_contract.py` | the 12 tools and their schemas/annotations via the SDK's in-memory client; `tools/list` ≤ 12 kB; concise responses ≤ 8k tokens on a fixture tree; every error code; real-arm tools absent without the flag; untrusted-data envelopes on notes/log tails; a job survives the server process being killed |
| `test_lab_registry.py` | manifests, index rebuild, idempotency under two concurrent creators (`runs/idem` O_EXCL + `flock`), `WriteGuard` refusals incl. signed-record paths |
| `test_lab_trainers.py` | strict schemas; manifests import without torch |
| `test_lab_budgets.py` | ledger folded from `usage.json`; cache invalidation; high-water-mark regression → HALT + item; corrupt file → HALT; `import-run`; probe pool; GPU-h by progress with the stall cap |
| `test_lab_supervisor.py` | `fake` plugin + `FakeClock` + fake NVML: FIFO/leases; pause = exit (no process); thermal pause holds the gpu lease and auto-resumes; memory-temp pause; fail-closed on 6 empty samples + item; stall ladder; heartbeat-dead kill; crash detection by pid/cmdline/boot_id; suspend detection (`host_slept`) and approval voiding; queue TTL; reboot voids real entries; pipelines; disk quota + low-disk graceful stop; review policies; HALT; no auto-resume of review pauses |
| `test_lab_authority.py` | the §5.1 table with refusal twins; signed-record verification (good/bad key, expired, consumed, hash mismatch, session changed, unsigned tier.json = tier 0); session rules (quiet hours, min gap, cap, per-day minutes across local midnight, allow-list); 120-min soak verification incl. sha mismatch |
| `test_lab_records_tools.py` | the three root-owned scripts run from a temp "install" with a test key under a pty: TTY refusal; phrase with parameters; machine-fields-first rendering (golden output); `--no-ai-text`; signatures verify with the test `allowed_signers`; they import nothing from the repo (`-I`, `sys.modules` check) |
| `test_lab_job.py` | fake plugin end to end; rc 0 + failed checker → `failed`; SIGTERM → checkpoint + `stopped`; own deadline fires with the supervisor dead; `usage.json` cadence; own pid recorded |
| `test_lab_session.py` | pinned servo argv (golden; never `--no-mcu`; `--max-ticks` arithmetic); rt pre-check failure → item; `servo --backend fake` + FakeMcu session: open request → fake button → ARM → run attaches under the cmd-owner lock → PARK → idle rest → `RESUME_REQ` → second run → cap → `STOP_REQ` → latch; weld-refused ARM → item; perception stub lifecycle; single-writer violation refused |
| `test_lab_real_fake.py` | the §5.8 matrix on `servo --backend fake` with fault injection: thermal during run (wait + continue), thermal while parked (no auto re-energise), NaN (item, blocked until a signed ack), latch (item), halt rule, review stop, per-day cap, quiet hours, suspend simulation |
| `test_lab_dashboard.py` | loopback-only bind; 405 on POST; path traversal rejected; every endpoint renders from the fixture tree; no form in the HTML |
| `test_lab_deps.py` | AST import rules; no `serial.Serial` anywhere under `lab/` |
| `test_lab_plugins_smoke.py` (`@slow`/`@gpu`) | the four plugins' short runs through the lab (the S4.3 / S5.2 smoke acceptances) |
| SV1: `test_state_machine.py`, `test_servo_core.py`, `test_reenergize_backends.py`, `test_soak.py`, `test_shm.py` (v3) | §5.9 |
| FW1: `test_firmware_vectors.py`, `test_heartbeat.py` | §5.10 |
| `test_architecture_sync.py`, `test_pixi_tasks.py` | unchanged guards |

### 8.5 Observability

stderr + rotating files under `runs/` for the three services; `runs/audit.jsonl`; `runs/supervisor/status.json`; per-segment logs; servo tick logs and summaries per session; TensorBoard; the dashboard.

### 8.6 Protected-path guard (CI + pre-commit)

`scripts/check_protected_paths.py` fails `pixi run ci` (and a pre-commit hook) when the diff against `main` touches `src/so101_rl/safety/**`, `src/so101_rl/servo/**`, `firmware/**` or `src/so101_rl/config/so101.yaml` unless every such commit carries a trailer `Work-Item: SV1` or `Work-Item: FW1` and that id is listed in the human-edited `docs/protected_work_items.yaml` with its scope (files allowed). The review note of each lab unit records `git diff --stat main -- src/so101_rl/safety src/so101_rl/servo firmware` as empty (or exactly SV1/FW1).

### 8.7 Vault librarian notes (recorded)

- The default-off real-arm tool class (§4.1) is the same convention as hardware-test-mcp's `HW_TEST_READONLY=1` default (`~/tools/claude_code/mcp/servers/hardware-test-mcp/README.md`): motion tools absent or inert until an operator flips a flag at the bench.
- The vault's autoresearch/recipe-designer loops advance curricula and swap recipes automatically on plateaus (`05-Wiki/concepts/Autonomous-ML-Training-Loop.md`, `Autoresearch-ML-Loop-Integration.md`). This design does not: plan §10 (2026-09-30 "Residual fallback", §5 budgets and triggers) wins; only FS8's in-run DR stage widening is automatic, and it is the plan's own rule.

## 9. Deltas to propose and the phased build

### 9.1 `so101-self-learning-rl-plan.md`

**New §12 "AI training connector (`so101_rl.lab`)"** (one page): purpose; definition of done; the authority model (signed human records; Tier 0 per-run approvals; Tier 1 rules, cap ramp, soak verification); SV1/FW1 as the two decided safety-side changes; review-trigger handling; the dashboard; proposals as the only path into §10; pointer to this file.

**Proposed §10 rows (project-shaping; implementation facts go to ARCHITECTURE §9 instead):**

| # | Decision | Proposed choice | Reason |
|---|---|---|---|
| P1 | AI connector | Python API `so101_rl.lab` + thin stdio MCP (12 tools); jobs are detached processes under a `systemd --user` supervisor with state on disk; a read-only local web dashboard (127.0.0.1) **supersedes** "no web UI beyond TensorBoard" | User answers 2026-10-07 |
| P2 | Human records are signed | Tier 0 approvals, S4-exit gate ticks, needs-human acks and the tier record are Ed25519-signed by root-owned tools (`so101-approve`, `so101-gate-sign`, `so101-ack`) with a passphrase-protected key and a root-owned trust root; the connector verifies and can never create them | User decision 2026-10-07 |
| P3 | Servo park / idle rest / resume-on-request (SV1) | `CmdMode.PARK`/`RESUME_REQ`, `TripCause.IDLE_TIMEOUT`, `idle.park_timeout_s` (default 600): hold at `home` between runs, rest + torque off when idle, full checked re-energise on request; parked thermal trips never auto-re-energise. Supersedes "rest + torque off between runs". | User decision 2026-10-07; without it a session cannot wait between runs |
| P4 | Firmware arm timeout (FW1) | DISARMED → LATCHED after 120 s without ARM (reason 8); the button is a two-minute consent; sessions stay powered across runs via P3 | User decision 2026-10-07 |
| P5 | Tier 1 session rules | Human-opened sessions; cap ramp 60 → 120 → 240 → 480 min after a clean session at each rung (clean = no fresh-process/human trip, ≤ 1 auto-resumed thermal trip, cap/idle end, weld check passed); power-cycle + weld check between sessions; quiet hours (default 23:00–07:00 local), min gap 2 min, allow-list (tasks/kinds/trainers/policies), per-day arm-minute cap (default 240, local-midnight boundary); a ≥ 120-min strict goal-directed soak verified at promotion **and** at every unattended admission | User decision 2026-10-07 + critic |
| P6 | Budget accounting | Probes (≤ 10 min GPU, ≤ 1 GPU-h per task pool) do not count as attempts; GPU-h is progress-holding time with a 30-min stall cap; the ledger is derived from run directories, runs made outside the lab are imported by a human CLI, and a shrinking attempt count halts the lab | Keeps the ≤ 3-attempt rule meaningful and unbypassable |
| P7 | Review-trigger handling | Sim attempt-class runs pause (checkpoint + exit, GPU released) and wait for the AI's written rationale, max two pauses then stop; probes continue; real-arm triggers always stop and need a signed ack. Never a method/task/reward change. | Keeps "manual review" literal for the arm; lets the AI be the first reviewer in sim |
| P8 | Thermal fail-closed | NVML sampling; GPU 85/75 °C as logged plus memory junction 95/85 °C when observable; six empty samples pause every GPU job and raise a needs-human item | Critic; the 3080 is a laptop-class thermal question (plan §2) |
| P9 | Proposals inbox | `plans/proposals/<date>-<slug>.md`, the only repo path the connector creates files in; a §10 row citing the file accepts it | Proposals must survive `runs/` cleanup |

Also: §5 budget table gains a "ledger key" column; H13 gains the `rules` block, the 120-min soak reference and signing; `docs/HARDWARE_CHECKLIST.md` gains the approver setup, the sha256 of the root-owned tools, HS8 (park/resume), and the FW1 test.

### 9.2 `docs/ARCHITECTURE.md` (v0.16 with the first lab unit; same-commit rule for every later change)

§3: rows for `lab-mcp`, `lab-supervisor` (systemd --user), job, `lab-dashboard`, the real-arm session (servo + perception, pinned argv, cmd-owner lock), the root-owned tools; "When the policy is late … unless parked". §4: `lab/`; dependency arrows (§1.1); `WriteGuard` paths; `perception/main.py`. §5: FL1–FL3; file schemas (`run.json`, `status.json`, `usage.json`, `events.jsonl`, `approval_request.json`, signed records, `gate.json`, `summary.json`, `ledger.json` cache, `needs_human/*`, `sessions.jsonl`, `tier.json` `rules` + `.sig`); `PARK`/`RESUME_REQ`, state record v3; `ScalarMonitor`. §6: layer 2 (FW1 reason 8), layer 5 (IDLE_TIMEOUT row, parked-thermal rule), new item 7 "Connector authority" (§5.1 table, §5.8 matrix, enforcement locations, "adds no safety layer"). §8: `lab` feature, `nvidia-ml-py`, `ffmpeg`, host requirements, services, tasks, `.mcp.json`. §9: the implementation defaults (JSON-Schema configs, hidden real-arm tools, exclusive GPU lease, cooperative pause, poll + wait, parse §10, `tier.json` holds the rules, static plugin table). §10: the 0.16 row.

### 9.3 Phased build (effort re-estimated, roughly 2× v1; agent-hours)

| Unit | Wave | Builds | Depends on | Est. |
|---|---|---|---|---|
| **LA0** core | L0 | registry + locks + idempotency + `WriteGuard`; provenance; audit; derived ledger + `import-run`; `trainers/api` + table + `fake`; supervisor (leases, pipelines, thermal/NVML fail-closed, stall ladder, suspend/reboot detection, queue TTL, disk, review policies, needs-human queue, notifications) + job entry (own pid, usage, own deadline); `policy.yaml`; SIM authority; systemd units | nothing from SU3–SU5 (`fake` + a `vizdemo` wrapper over `viz.demo`) | 26 |
| **LA1** MCP + dashboard | L0 | the 12 tools, resources, 3 prompts, errors, untrusted-data envelopes; dashboard; `.mcp.json`; contract tests; ARCHITECTURE v0.16; plan §12 + rows | LA0 | 20 |
| **SV1** servo park/idle/resume | L0–L1 (its own commit, `Work-Item: SV1`) | §5.9 | nothing | 12 |
| **FW1** firmware arm timeout | L0–L1 (`Work-Item: FW1`) | §5.10 | nothing | 4 |
| **PE1** perception process | L1 | `perception/main.py` + session integration | U5 modules | 5 |
| **LA2** FastSAC unattended | L1 | `fastsac_sim` plugin, `run_gates`, `evaluate_policy(gate)`, pipelines on the real trainer, `ScalarMonitor` extraction, rollout dataset registration; acceptance A1–A4 | SU3, SU4 S4.1–S4.3 | 16 |
| **LA3** offline + fine-tune (sim) | L2 | `tdmpc2_offline`, `sb3_sac_her` (sim), `tdmpc2_finetune` (sim); `--tier1` mode in the entry points; smoke tests | SR2, SU5 S5.1–S5.2 | 12 |
| **RT0** root-owned tools | L2 | `so101-approve`/`gate-sign`/`ack`, `so101-lab-install`, `init-approver`, record verification, golden-screen tests | nothing | 8 |
| **LA4** real arm, Tier 0 | L3 | real-arm tools behind the flag; session manager (pinned argv, pre-checks, cmd-owner lock, open/close, weld check, `sessions.jsonl`); hw-artifact checks; arm-time ledger; the §5.8 matrix on the fake servo; acceptance A5 (fake-backend tests) | **LA3** (→ SU5), RT0, SV1, FW1, PE1; first real use needs H2, H3, H5, H7, H11 | 22 |
| **LA5** Tier 1 unattended | L4 | signed `tier.json` + rules reader, cap ramp, quiet hours, min gap, per-day cap, soak verification at admission, per-episode-batch revalidation; acceptance A6 | LA4; H13 **and** the 120-min strict soak **and** HS8 (attended park/resume) **and** one clean attended 60-min session | 10 |
| Deferred | — | `compare_runs`, `start_sweep`, frames, log tail; LeRobot dataset kind; later plugins; MCP Tasks; SQLite index | evidence / a plugin that needs it | — |

≈ 135 agent-hours plus ≈ 2 GPU-h of smoke runs and ≈ 6 attended human-hours (HS8, FW1 test, the first 60-min session, approver setup). Each unit: ARCHITECTURE sync in the same commit, a review note in `plans/reviews/`.

## 10. Acceptance (definition of done, testable)

| # | Criterion | Shown by |
|---|---|---|
| **A1** Unattended sim end to end | `start_run(fastsac_sim, reach, pipeline={evaluate: {1000, dr 4, render 16, gate S4_exit}, export: true}, review_policy=pause)`, then Claude Code is closed. Later: `finished` within budget; `eval.json`, `gate.json.numeric`, renders, `review_checklist.md`, `policy_export.pt` present and checked; policy `sim_gate_numeric_passed`; `gate_checklist` item; ledger debited by progress time; no human action; full audit trail. | GPU: the S4.5 reach run through the lab. CI: `test_lab_supervisor.py::test_pipeline_fake`, `test_lab_contract.py::test_jobs_survive_server_exit`. |
| **A2** Review triggers unattended | attempt-class trigger → pause (process exited, GPU released, next job runs), `control_run(resume, note)` continues from `ckpt_last.pt`; third identical trigger stops; probe continues; nothing else changes. | `test_lab_supervisor.py::test_review_policies`. |
| **A3** Budgets, thermal, health, stall, disk | wall → checkpoint + `budget_exhausted` + debit; 4th attempt refused; fake 86 °C / memory 96 °C → pause all GPU jobs with the lease held, resume at 75/85 °C; six empty samples → pause + item; stall ladder event → stop → kill; low disk → graceful stop; the job's own deadline fires with the supervisor dead. | `test_lab_supervisor.py`, `test_lab_budgets.py`, `test_lab_job.py`. |
| **A4** Crash, suspend, reboot, session independence | `kill -9` job → `crashed`; `kill -9` supervisor → jobs continue, systemd restarts it, re-adoption by pid/cmdline/boot_id; simulated suspend → `host_slept`, approvals void; simulated reboot → runs `crashed`, real queue voided; closing Claude Code changes nothing. | `test_lab_supervisor.py`. |
| **A5** Tier 0 signed approval + session (fake backend) | `request_real_run` → `awaiting_approval`; lab processes cannot write records (`WriteGuard`, AST); `so101-approve` under a pty with a test key writes a verifying record; expired/consumed/hash-changed/wrong-key/new-session refused; the session manager launches `servo --backend fake` with the pinned argv, the fake button arms it, the job attaches under the lock, runs, parks, exits; `STOP_REQ` at cap; weld-refused ARM → item. **First real use (after LA4, attended):** HS1 sysid excitation and HS4 zero-shot through `request_real_run` + `so101-approve`, producing their planned artifacts. | `test_lab_records_tools.py`, `test_lab_authority.py`, `test_lab_session.py`; HS1/HS4 review notes. |
| **A6** Tier 1 unattended (fake backend, then real) | signed `tier.json` + rules: two runs admitted without approval inside one fake session, parked between them, idle rest via SV1, `RESUME_REQ` re-energise; thermal trip during a run → wait + continue; thermal while parked → no auto re-energise; NaN → safe state + item + blocked until a signed ack; latch → item; quiet hours / min gap / per-day cap refusals; cap → `STOP_REQ` → latch; clean-session verdict recorded; cap raise refused without a clean session. **Real (after LA5 prerequisites):** the first unattended 60-min session (HS5 fine-tuning) ends by cap or idle with the arm resting torque-off and a clean verdict; the ramp proceeds only after that. | `test_lab_real_fake.py`, `test_lab_session.py`; session logs and summaries in `plans/reviews/`. |
| **A7** Dashboard | loopback only; GET only; shows runs, metrics summary, triggers, trips/thermal/MCU events, needs-human items with the exact commands, pending approvals (display only), session panel, links to TensorBoard and videos; no write control. | `test_lab_dashboard.py`; screenshot in the LA1 review note. |
| **A8** Constraints | `safety/`, `servo/`, `firmware/` touched only by SV1/FW1 commits with trailers (guard green); `test_lab_deps.py` green; every lab commit touches ARCHITECTURE; no §10 row edited by the build (rows P1–P9 are added only by the user). | `scripts/check_protected_paths.py` in `ci`; review notes. |
| **A9** SV1 obligations | parked HOLD never trips STALE_CMD; idle → rest + torque off with ≤ 5 ticks of post-torque-off motion (fake/MuJoCo), `RESUME_REQ` → full re-energise → RUN; refused for fresh-process/human causes and below the 1 cm floor; heartbeat gaps never > 200 ms across two park/resume cycles; tracemalloc no-allocation test green; parity tests unchanged. **Hardware:** HS8 artifact. | the SV1 tests of §5.9; `runs/hw/park-resume-<ts>.json`. |
| **A10** FW1 | vectors pass on FakeMcu and the real MCU (H3 extension); PROTOCOL.md and host decoding updated; the servo's default arm wait stays below 120 s. | `test_firmware_vectors.py`; H3 artifact. |
| **A11** Token discipline | `tools/list` ≤ 12 kB; default responses ≤ 8k tokens; no raw logs/videos; untrusted-data envelopes on every AI-written or log field. | `test_lab_contract.py`. |

## 11. Remaining genuinely open user decisions

| # | Decision | Recommended default |
|---|---|---|
| U1 | Package name: `so101_rl.lab` vs `so101_rl.platform` | `lab` (short; avoids the stdlib name `platform`) |
| U2 | Probes do not count as attempts (P6) | yes; ≤ 10 min GPU each, ≤ 1 GPU-h per task pool |
| U3 | Sim review-trigger policy: AI as first reviewer with `pause` (P7) vs `stop` always | `pause` for attempt-class sim runs, max two pauses; real stays `stop` + signed ack |
| U4 | Clean-session definition and ramp parameters (P5): ≤ 1 auto-resumed thermal trip allowed, quiet hours 23:00–07:00, min gap 2 min, 240 arm-min/day, local-midnight boundary | as listed |
| U5 | Memory-junction pause/resume thresholds (P8): 95/85 °C | as listed (GDDR6X Tj max is 105 °C) |
| U6 | Proposals inbox at `plans/proposals/` (P9) vs `runs/proposals/` | `plans/proposals/` |

Implementation defaults no longer presented as user decisions (recorded in ARCHITECTURE §9 when built): JSON-Schema configs with `jsonschema`; hidden real-arm tools behind the flag; exclusive GPU lease; cooperative pause with SIGTERM fallback; poll + `get_run(wait)`; parsed §10 as a resource; `tier.json` holds the rules; static plugin table; `pixi run -e` job spawning; `ScalarMonitor` extraction; `home` as the park pose (decided by the user, item 8).

## 12. Sources consulted

Repository: `CLAUDE.md`; `so101-self-learning-rl-plan.md` (§1, §2A, §5, §8, §10, §11); `docs/ARCHITECTURE.md` v0.15; `plans/sim-track-implementation-plan.md`; `plans/phase0-implementation-plan.md`; `plans/camera-parity-plan.md`; `pyproject.toml`; `README.md`; `tests/{conftest,test_architecture_sync,test_pixi_tasks}.py`; `src/so101_rl/{config/schema.py,logging/checks.py,viz/monitor.py,viz/video.py,safety/state_machine.py,servo/shm.py,servo/main.py}`; `firmware/watchdog_mcu/PROTOCOL.md`; `docs/HARDWARE_CHECKLIST.md`. House MCP pattern: `~/tools/claude_code/mcp/servers/hardware-test-mcp/{README.md,server.py}`.

Second Brain (`~/Documents/Vaults/Local`): `05-Wiki/concepts/MCP-Server-Development.md`; `MCP-Tool-Design-and-Optimization.md`; `MCP-Server-Architecture.md`; `MCP-Server-Pattern-Choice.md`; `Hardware-Test-Harness-Pattern.md`; `Arm-Safety-Envelope.md`; `Agentic-Safety.md`; `Mechanical-Guarantees-vs-Prompt-Prose.md`; `Harness-Engineering.md`; `Agentic-Loops.md`; `Autonomous-ML-Training-Loop.md`; `Autoresearch-ML-Loop-Integration.md`; `05-Wiki/source-library/lib-autoresearch.md`; `04-Archive/2026-07-19-gpu-campaign-3day-lessons.md`; `04-Archive/2026-09-10-claude-driven-so101-recording-research.md`; `05-Wiki/research/2026-09-12-review-agents-need-read-only-tools.md`.
