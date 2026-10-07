# AI training connector (`so101_rl.lab`, SV1, FW1, OS1, PE1) — Implementation Plan

> **Version 1 (2026-10-07).** Design: `plans/ai-connector-design.md` v3 (6d8bf65; §9.3 phased build, §10 acceptance A1–A12, with A1–A3/A7/A8/A11 taken from v2 31b2fa8 where v3 says "as v2"). Decisions: `so101-self-learning-rl-plan.md` §10 rows dated 2026-10-07 (P1–P14, P3a, P5a; all binding) and §12 v4. Architecture: `docs/ARCHITECTURE.md` v0.17. Sim track: `plans/sim-track-implementation-plan.md` (SU3/SU4/SU5/SR2/SR3 **not built**; FS1–FS12 as built). Camera parity: `plans/camera-parity-plan.md` (SUC, **in progress in this checkout**: untracked `servo/probe.py`). Hardware: `plans/phase0-implementation-plan.md` H-series, `docs/HARDWARE_CHECKLIST.md`. House MCP patterns: `~/tools/claude_code/mcp/servers/{hardware-test-mcp,ros2-mcp,skills-mcp}`.
>
> **Scope discipline (anti-hypertrophy, design §0/§4):** 12 MCP tools in LA1 (`request_real_run` only in LA4); a static in-repo trainer table, no entry-point discovery; no plugin for any non-RL trainer; no `execute_script`; no email/cloud notifications; no new hardware beyond the H-track. Everything the plan adds outside `src/so101_rl/lab/` is one of the four decided work items (SV1, FW1, OS1, PE1) or a shared write listed in `execution_shape`.
>
> **Hardware rule for every task in this file:** no task of the user's account opens a serial device. Software tasks run on the fake backend (`servo --backend fake`), CPU MuJoCo or MuJoCo Warp, and `pixi run -e cpu test` stays green without the arm. Tasks marked **HW** are human-run, attended (Tier 0) until H13, and listed in the hardware units at the end.
>
> **This plan writes no code and commits nothing.** The master commits; executors stage per task. Every commit that changes a module, process, interface, safety-check order or pixi feature touches `docs/ARCHITECTURE.md` in the same commit (`tests/test_architecture_sync.py`).

## Overview

| Item | Value |
|---|---|
| Objective | An AI client (Claude Code over stdio MCP) can create, run, monitor, pause, evaluate, gate and export **sim** training runs fully unattended under the GPU budgets and thermal limits (jobs outlive the MCP session), and — after the signed tier promotion and the session rules — run **real-arm** jobs whose authority is enforced by another Linux account from a root-owned servo snapshot that verifies Ed25519-signed human records itself. The connector adds no safety layer and removes none. |
| Deliverables | `src/so101_rl/lab/**` (registry, provenance, audit, ledger, trainers API + `fake`, supervisor, job entry, session manager, MCP server, dashboard, CLI, `policy.yaml`, user units); `src/so101_rl/authority/__init__.py` (stdlib-only record verification, Tier 1 rules, nonce + arm-minute ledgers; installed verbatim as `so101_authority.py`); `tools/so101_records/{approve,gate_sign,ack}.py` + shims; `deploy/**` (udev, sysusers, tmpfiles, system unit, polkit, install/update/verify scripts); SV1 in `servo/`, `safety/`, `config/`; FW1 in `firmware/watchdog_mcu/` + `servo/heartbeat.py`; PE1 `perception/main.py`; LA2/LA3 plugins; `.mcp.json`; tests `tests/test_lab_*.py`, `tests/test_authority*.py`, `tests/test_records_tools.py`, `tests/test_perception_main.py`, SV1/FW1 tests; `docs/ARCHITECTURE.md` v0.18+; `docs/HARDWARE_CHECKLIST.md` (OS1, H3 extension, HS8, HS9); plan §12 v3 text. |
| Domain | Process supervision (systemd user units), MCP server (Python SDK 2.3), safety-critical servo state machine (SV1), MicroPython firmware (FW1), Linux account separation and signed records (OS1), perception process (PE1), RL training pipelines (LA2/LA3). |
| Technologies | python 3.12, pixi, `mcp 2.3.0` (conda-forge; **FastMCP is renamed `mcp.server.mcpserver.MCPServer` in 2.x**, verified 2026-10-07 in a throwaway `pixi exec` env), `jsonschema 4.26`, `nvidia-ml-py 13.x`, systemd 255 (`systemd --user` running, `Linger=yes` already), OpenSSH 9.6 (`ssh-keygen -Y sign/verify` with namespaces), numpy structured records (`servo/shm.py`), MicroPython RP2040 firmware `Core`. |
| Unit of this plan | Master passed no `Unit Id` / `Owns Files`; the connector is its own unit set, partitioned in `execution_shape` (exclusive file ownership within a wave; hand-offs annotated). |
| Preconditions (done) | Phase 0 waves 0–2 (U0–U7, U3); sim wave 1 (SU0, SU1, SU2, SR1, SUV) merged on `integration/sim-wave1` → `feature/camera-parity` (bf7346b, the current HEAD). Design v3 + §10 rows P1–P14 committed. |
| Preconditions (pending, gated units only) | SU3/SU4 (LA2), SU5/SR2 (LA3), SUC (PE1 real-camera path), H2/H3/H5/H7/H11 + HS8 (LA4-HW), H13 + HS9 (120-min strict soak with parked segments) + HS8 + one clean attended 60-min session (LA5-HW), OS1-HW install (every real session). |

### Environment and code facts checked on 2026-10-07

- **Host:** `systemd --user` is `running`, `loginctl show-user koen -p Linger` = `yes`, `systemd-run` 255; `ssh-keygen` OpenSSH 9.6p1 (`-Y sign -n <namespace>` supported); `nproc` 16; **`CONFIG_RT_GROUP_SCHED is not set`** on kernel 7.0.0-38 (the design's `cpu.rt_runtime_us` slice step is therefore N/A on this host; the servo's own `rt_active` stays the truth).
- **Accounts/devices (OS1 is entirely pending):** `id koen` = `koen adm dialout cdrom sudo dip plugdev users lpadmin` (**in `dialout`**, not in `video`); no `so101` group, no `so101-servo`/`so101-approver` users; `/etc/so101_rl`, `/opt/so101_rl`, `/var/lib/so101_rl` absent. One serial device today: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9024444-if00 -> ../../ttyACM0` (`root:dialout 0660`). `so101.yaml` has `port: /dev/ttyACM0`, `heartbeat_port: null`.
- **Packages:** `mcp`, `jsonschema`, `pynvml`/`nvidia-ml-py` are **not** in any pixi env; conda-forge has `mcp 2.3.0`, `jsonschema 4.26.0`, `nvidia-ml-py 13.615.71`. No `.mcp.json` and no `.claude/settings.json` in the repo.
- **mcp 2.3.0 API (probed):** `from mcp.server.mcpserver import MCPServer, Context`; `MCPServer(name, *, instructions=..., tools=..., ...)`; `MCPServer.run(transport="stdio")`, `run_stdio_async()`; `@mcp.tool(name=, description=, annotations=, structured_output: bool | None)`; `@mcp.resource(uri, *, name=, mime_type=, ...)` with `{param}` templates (`list_resource_templates`); `@mcp.prompt(name=)`; in-memory testing: `from mcp.client import Client; Client(server: MCPServer | StdioServerParameters | ...)`; `mcp.shared.memory.create_client_server_memory_streams()`; `mcp.server.stdio.stdio_server(stdin=None, stdout=None)` and `rebind_std_handle_to_fd` (for the stdout hijack). `mcp.server.fastmcp` does **not** exist in 2.x.
- **Servo/shm as built:** `CmdMode {IDLE 0, RUN 1, HOLD 2, STOP_REQ 3}`; `CMD_DTYPE` 64 B (`seq, t_mono_ns, state_seq, mode, episode, delta_ticks[6], crc32, _pad V4`); `STATE_DTYPE` 232 B (`read_ok` at 225, `crc32` at 228), stride 256, `SHM_LAYOUT_VERSION = 2`; `TICK_LOG_DTYPE` (no header; `RingWriter`); dedupe `seq == last_seq` → not new; `ServoCore.start(should_abort, arm_timeout_s, immediate_stop)`; `process.run(...)` exit codes 0–4; summary fields from `_core_summary`; `soak.analyse_tick_log` counts `hold_entries` = every HOLD entry; `SafetyStateMachine` rows as in its docstring (`_POLICY_RANK = {AUTO 0, FRESH_PROCESS 1, HUMAN 2}`); `TripCause` highest bit `BOX_PRESENT = 32768`; `SmState.COOLDOWN` unused; `firmware/main.py` 91 lines, budget 95 (`test_firmware_line_budget`); vector `disarmed_never_times_out` asserts DISARMED stays DISARMED after 105 s (FW1 **changes this semantic** at 120 s; the vector is renamed and shortened).
- **Servo snapshot imports:** `servo/`, `safety/`, `config/` import only `so101_rl.{servo,config,safety,logging}` and lazily `so101_rl.sim` (mujoco backend). The snapshot tree is therefore `src/so101_rl/{__init__.py,servo,safety,config,logging,authority}` (plus `pyproject.toml`, `pixi.lock`).
- **Perception:** `OBJECT_TABLE_DTYPE` already carries `fiducial_ok` (u1); `So101GoalEnv._update_die` treats a detection as visible only when `fiducial_ok` and the detection is at most one policy period + 2 frames old. `FakeCamera(rgb, depth_m, intr)`, `D435Camera`, `HsvDepthDieTracker.detect/centroid_base`, `FiducialCheck.check(rgb) -> FiducialReport`, `ObjectTableShm.create/write` exist; no perception entry point exists (PE1).
- **Tests/conventions:** `tests/conftest.py` markers `hardware, gpu, slow, regression` (a new `integration` marker is added by LP0); `tests/test_pixi_tasks.py` fails on tasks whose module is missing; `tests/test_architecture_sync.py` requires every package dir under `src/so101_rl/` in ARCHITECTURE §4 and a matching version row; the house MCP servers return errors as dicts (`{"ok": false, "error": ...}`), log to stderr only, gate motion tools by an env var read at import, use plain type hints, and have no SDK-client tests (this plan adds them).
- **Vault patterns consulted:** `05-Wiki/concepts/MCP-Server-Pattern-Choice.md` (stateless server, spawned work), `Hardware-Test-Harness-Pattern.md` (typed tools, bounds in code, no `execute_script`, read-only first).

## Specialist Guidance Summary

**mcp-server-architect** (LA1; full report in the session log):
- **Transport MCP server ↔ supervisor: files only.** Request files in `runs/queue/` (tmp → fsync → `os.replace`), the supervisor polls at 1 Hz and writes `runs/supervisor/heartbeat.json` every tick; the server reports `SUPERVISOR_DOWN` when that heartbeat is > 5 s old and never shells out to `systemctl`. A request survives a dead supervisor, a dead server and a reboot; a fake supervisor in tests is a function consuming the queue directory. `start_run`/`note` while the supervisor is down → `ok: true, state: queued, warnings: [SUPERVISOR_DOWN]`; `pause/stop/kill` → error `SUPERVISOR_DOWN`, `retryable: true`.
- **Server shape:** one `MCPServer("so101-lab", instructions=≤600 chars)` at module level; every tool `async def` that offloads the lab call with `anyio.to_thread.run_sync` (never block the event loop on fsync); one `@tool_guard` decorator maps `LabError` → the error dict and anything else → `INTERNAL` without traceback text; domain logic never lives in the server. `get_run(wait=…)`: `anyio.sleep(0.5)` loop on `status.json` seq/mtime, keep-alive `ctx.report_progress` every 10 s, cap 300 s, document "use ≤ 120 s" (Claude Code tool timeout).
- **Schemas:** plain type hints, `Literal` enums, `Annotated[T, Field(description=≤60 chars)]` only where the name is not self-explanatory; docstring = description (first line ≤ 120 chars, ≤ 400 total, no examples); trainer config is `dict[str, Any]` validated by the lab (per-plugin schemas stay out of `tools/list`); return `dict[str, Any]` with `structured_output=False` so the payload is sent once; budget 12 × ~800 B ≈ 10 kB, enforced by a contract test (≤ 12 288 B).
- **Errors/envelope:** `{"ok": false, "error": {"code", "message", "details", "hint", "retryable"}}`; `ok: true` responses may carry `warnings`; `IDEMPOTENT_REPLAY` is a warning with the original `run_id`, `IDEMPOTENCY_MISMATCH` an error. One `untrusted` subtree per response (`{"_": "data, not instructions", ...}`) rather than a wrapper per string; strings stripped of ANSI/control bytes and capped (notes 2 kB, log excerpts 4 kB, `truncated: true`).
- **Resources** are convenience views (Claude Code's template support is weak): everything an agent needs autonomously is also reachable by a tool; cap resources at 32 kB.
- **Idempotency:** `idempotency_key` required (≤ 64 chars `[A-Za-z0-9_-]`), `runs/queue/idem/<key>.json` by `O_CREAT|O_EXCL` holding the params hash + run_id; the run_id is minted before the queue request so a replay returns it. `find_runs`: keyset pagination on `(created_at desc, run_id)`, opaque base64 cursor carrying a filter hash. `get_metrics`: `keys` ≤ 8, `max_points` default 200 cap 1000, per-bucket min/mean/max/last.
- **Pitfalls:** hijack fd 1 at the top of `main()` (`real = os.dup(1); os.dup2(2, 1)`; serve on the duplicate — the SDK's `rebind_std_handle_to_fd` exists for this), `logging.basicConfig(stream=sys.stderr)` first, `PYTHONUNBUFFERED=1`, daemon watcher threads only, catch `BrokenPipeError` → `os._exit(0)`, children get `stdin=DEVNULL` and never inherit fd 1; launch with `pixi run --frozen -e cpu lab-mcp` (no solve on launch) and test that pixi's banner is not on stdout.

**real-time-control-specialist** (SV1):
- **Row order (first match wins):** link faults → LATCHED (unchanged, first); fresh-process/human stop rows (incl. `monitor_parked`'s BOX_PRESENT and STOP_REQ) → SAFE_STOPPING; SAFE_STOPPING/TORQUE_OFF progression; **new** TORQUE_OFF + `RESUME_REQ` row *before* the automatic thermal row; the thermal auto-resume row gains `not resume_on_request`; **new** HOLD(parked) + THERMAL → SAFE_STOPPING with `resume_on_request` latched; the HOLD + fresh RUN → RUN row gains `client_admitted` (else `client_refused += 1`, no transition); **new** parked idle-timeout row after it (a live client wins a tie); **new** PARK row after the stop rows and *before* the stale-age and consecutive-misses rows (a fresh PARK must not lose to a 5-miss trip); the stale-age and misses rows get an explicit `not parked` guard (order alone is not enough).
- **Allocation-free:** fault causes stay int bitmasks (`IDLE_TIMEOUT`, `_RESUME_MASK`, `_TERMINAL_MASK`); one preallocated `SmInputs`/`SmOutputs` mutated in place; the idle timer is a clock deadline set once on entry (`parked_deadline_ns = now + park_timeout_ns`; one comparison per tick), never a tick counter (overruns would stretch it). `mark_energized` sets parked, restarts the deadline, resets `consecutive_misses`; `_enter_run` clears parked.
- **Pitfalls:** never re-latch the parked target from present (a saturating tracking clamp relative to present ratchets the written target down with a sagging arm): clamp the *fixed* latched target each tick (`written = present + clip(latched − present, ±E)`), let BOX_PRESENT be the backstop, and pin it with `test_parked_target_does_not_ratchet_with_sag`; reset `consecutive_misses` entering/leaving parked; `hold_entries` must count only unparked HOLD entries so parking cannot mask flapping; add invariants (zero parked STALE_CMD trips, every energise followed by a parked entry).
- **Authority check in the servo:** the specialist recommends an asynchronous worker thread; the alternative kept here (decision D1 below) is **synchronous in the tick with the heartbeat sent immediately before and a 120 ms subprocess timeout** (worst-case gap 33 + 120 = 153 ms < 200 ms and < 5 missed acks), with a BOOT-time `ssh-keygen` warm-up against a self-test record so the page cache is hot. Record the duration, pin `max(diff(hb_ts)) ≤ 200 ms` during a 60 ms fake verification.
- **Exit-on-terminal in the runner loop, not the core:** the core exposes `terminal_reason` after `tick()`; the runner breaks, runs `shutdown`, returns a distinct exit code for a terminal trip vs a clean close; idle TORQUE_OFF (`faults == IDLE_TIMEOUT`) is not terminal. Under systemd: `Type=simple`, `Restart=no` (a restart would re-energise), `KillMode=mixed`, `SendSIGHUP=no`, `TimeoutStopSec = bound_s + 10` (one SIGTERM to the main pid, SIGKILL to the group after the timeout, the MCU opens the relay).

**House MCP survey** (Explore agent): FastMCP + `mcp.run()` at module level, errors as dicts, motion tools gated by an env var read at import and documented in the docstring (`WRITE/MOTION tool — gated by …`), bounds enforced in code, docstrings as descriptions, no resources/prompts, no SDK-client tests, servers launched through a wrapper that `exec`s the right Python (the system `python3` lacks `mcp`, which is why two house servers currently fail to connect — hence `pixi run --frozen -e cpu lab-mcp` here).

## What can be built now vs. gated

| Unit | Status | Depends on (software) | Gated by (hardware / other tracks) |
|---|---|---|---|
| **LP0** pixi `lab` feature, markers | **BUILD NOW** (sequential, first) | — | — |
| **LA0** lab core | **BUILD NOW** | LP0 | — |
| **LA1** MCP + dashboard + unattended fake run | **BUILD NOW** | LA0 | — |
| **SV1** servo park/idle/resume/exit (software) | **BUILD NOW** (own grill review) | LP0 (markers); codes against FL7 (authority interface) with a lazy import | HS8 (attended 60-min parked session + thermal hold) is HW |
| **FW1** firmware arm timeout (software) | **BUILD NOW** (own grill review) | — | H3 extension on the real MCU is HW |
| **OS1-SW** authority module, records tools, deploy files, verify script | **BUILD NOW** (grill review recommended, D3) | LP0 | OS1-HW install is HW/human (sudo) |
| **PE1-SW** `perception/main.py` (fake camera path) | **BUILD NOW** | — | SUC for `calib/camera.yaml` (hand-off noted); real D435 path needs H8 |
| **LA4-SW** session manager, `request_real_run`, A5/A6 matrix on `servo --backend fake` with the real authority module and a temp records store | buildable after wave 1 | LA1, SV1, FW1, OS1-SW, PE1-SW | — (fake backend) |
| **LA2** FastSAC unattended pipelines | GATED | LA0, SU3, SU4, SUV | GPU for acceptance (~2 GPU-h) |
| **LA3** SB3 + TD-MPC2 plugins, `--tier1` mode | GATED | LA2, SU5, SR2 | — |
| **LA5-SW** Tier 1 mirror: tier reader, rules, cap ramp, A6 | buildable after LA4-SW | LA4-SW | — |
| **OS1-HW** install (accounts, udev, snapshot, units, polkit, SHA256SUMS) | HW/human | OS1-SW, SV1 (snapshot contents) | sudo session, by hand, pinned commit |
| **FW1-HW** H3 extension | HW | FW1 | H3 |
| **HS8** parked-session + thermal hold | HW | SV1, OS1-HW | H2, H3, H5, H7, H11 |
| **LA4-HW** first real use via the connector (HS1, HS4) | HW | LA3, LA4-SW, OS1-HW, FW1-HW, HS8 | H2, H3, H5, H7, H11 |
| **HS9** 120-min strict soak with ≥ 2 parked segments | HW | SV1, OS1-HW | H10 |
| **LA5-HW** first unattended 60-min session (two-phase clean) | HW | LA5-SW, LA4-HW, HS9, HS8 | H13, one clean attended 60-min session |

## Frozen Interfaces (FL-series)

A unit may add behaviour but may not change a listed signature, schema field, layout version, enum value, file name or ordering without updating this section **and** `docs/ARCHITECTURE.md` §5/§6 in the same change. F1–F11 (Phase 0) and FS1–FS12 (sim track) stay in force. Where the design (`plans/ai-connector-design.md`) and this section differ, this section is the as-planned detail of the design and wins for the executor; the §10 rows win over both.

### FL1. Trainer plugin API — `src/so101_rl/lab/trainers/api.py`, `src/so101_rl/lab/trainers/__init__.py`

```python
Domain = Literal["sim", "real", "offline"]
Lease = Literal["gpu", "arm", "cpu"]
BudgetClass = Literal["attempt", "probe", "eval"]
RunKind = Literal["attempt", "probe", "eval", "export", "gates", "pipeline"]

@dataclass(frozen=True)
class Capabilities:
    domains: frozenset[Domain]
    obs_layouts: tuple[str, ...]            # ("fs3.v1",)
    action_space: str                       # "joint_delta_6"
    inputs: tuple[str, ...]                 # subset of ("policy_export", "checkpoint", "dataset")
    outputs: tuple[str, ...]                # subset of ("checkpoint", "policy_export", "eval", "dataset")
    gpu: Literal["required", "optional", "none"]
    vram_gb_estimate: float
    resumable: bool
    evaluates: bool
    exports: bool
    role: Literal["core_rl", "accelerator"]
    env_name: Literal["default", "cpu", "robot"]   # the pixi env the job segment runs in

@dataclass(frozen=True)
class JobPlan:
    domain: Domain
    leases: frozenset[Lease]
    budget_key: str                         # "s4.reach", "s4.die_to_zone", "s5.reach.finetune.sb3", "probe.reach", "fake.<task>"
    budget_class: BudgetClass
    gpu_h_estimate: float
    arm_min_estimate: float
    disk_gb_estimate: float
    requires: tuple[str, ...]               # artifact requirements checked by lab.authority (SIM column of design §5.1):
                                            #   "gate:L1:<task>", "gate:L2:<task>", "sim_params:verified", "hw:<artifact-id>"
    resumable_fields: tuple[str, ...]       # config keys control_run(resume, overrides=) may change
    unattended_ok: bool

@dataclass(frozen=True)
class RunLimits:                            # v3: ONLY from a verified signed record (Tier 0) or the signed tier rules (Tier 1)
    episodes: int | None
    arm_min_cap: int | None
    speed_scale_max: float | None
    source: Literal["none", "approval", "tier1"]

@dataclass(frozen=True)
class EvalReport:
    n: int; successes: int; wilson95: tuple[float, float]
    final_err_m: dict[str, float]           # {"p50": .., "p90": ..}
    trips: dict[str, int]; renders: tuple[str, ...]; extra: Mapping[str, float]

@dataclass(frozen=True)
class ArtifactRef: path: str; sha256: str; kind: str          # path relative to run_dir unless absolute under runs/

@dataclass(frozen=True)
class ArtifactManifest: items: tuple[ArtifactRef, ...]; ok: bool; problems: tuple[str, ...]

class ScalarMonitor(Protocol):              # implemented in viz/monitor.py (LA0 extracts it from TrainingMonitor)
    def log(self, step: int, scalars: Mapping[str, float]) -> None    # metrics.csv row + TensorBoard scalars
    def text(self, step: int, tag: str, text: str) -> None
    def csv_fields(self) -> dict[str, int]                            # {"viz_disabled": 0|1}
    def close(self) -> None

class ServoView(Protocol):                  # LA4; read-only view of the attached servo's state record + summary
    state: int; parked: bool; arm_ns: int; fault_bits: int; client_refused: int; park_refused: int; resume_refused: int
    def summary(self) -> Mapping[str, Any] | None

class RunContext(Protocol):
    run_id: str; segment: int; run_dir: Path; cfg: Mapping[str, Any]; task: TaskSpec
    device: str; vram_fraction: float; deadline_ns: int; monitor: ScalarMonitor
    def heartbeat(self, *, step: int | None = None, phase: str = "train") -> None   # progress-coupled, from inside the loops, never a thread
    def should_stop(self) -> bool
    def should_pause(self) -> bool
    def checkpoint(self, path: Path, *, step: int) -> None            # records ckpt in status.json; copies to ckpt_last.pt
    def event(self, kind: str, **payload: Any) -> None
    def review_trigger(self, name: str, **details: Any) -> None       # review.json + event; the supervisor applies P7
    def servo(self) -> ServoView | None
    def limits(self) -> RunLimits
    def record_usage(self) -> None                                    # usage.json atomic rewrite (high-water)

class TrainerPlugin(Protocol):
    name: str; version: str; config_version: int; capabilities: Capabilities
    def config_schema(self) -> dict                                   # JSON Schema 2020-12, additionalProperties false, const config_version
    def validate(self, cfg: Mapping[str, Any]) -> Mapping[str, Any]   # jsonschema → frozen mapping; raises ConfigInvalid(path, rule)
    def plan(self, cfg: Mapping[str, Any], task: TaskSpec, init_from: ArtifactRef | None) -> JobPlan
    def train(self, ctx: RunContext) -> None
    def resume(self, ctx: RunContext) -> None
    def evaluate(self, ctx: RunContext, eval_cfg: Mapping[str, Any]) -> EvalReport
    def export(self, ctx: RunContext) -> ArtifactRef
    def artifacts(self, run_dir: Path) -> ArtifactManifest           # a failed checker → run state `failed` regardless of rc

TRAINERS: Final[Mapping[str, type[TrainerPlugin]]]
# LA0: {"fake": FakeTrainer}
# LA2: + {"fastsac_sim": FastSacSim}
# LA3: + {"sb3_sac_her": Sb3SacHer, "tdmpc2_offline": TdMpc2Offline, "tdmpc2_finetune": TdMpc2Finetune}
# Static. tests/test_lab_trainers.py::test_every_listed_trainer_has_a_runnable_module fails on a listed plugin whose rl/ entry module is missing.
```

Not configurable by any plugin (the lab refuses the keys with `CONFIG_INVALID`): thermal thresholds, caps, halt rules, speed ladder, tier rules, `tensorboard` on/off, the metrics contract, DR ranges. Plugin config schemas are served as the resource `lab://schemas/trainer/{name}`.

**`fake` trainer config (schema 1):** `{config_version: 1, steps: int 1..100000, step_time_s: number 0..1, checkpoint_every: int ≥ 1, metrics: "ramp"|"noisy"|"flat", behaviour: null | "crash_at:<step>" | "stall_at:<step>" | "oom_at:<step>" | "review_trigger_at:<step>:<name>" | "servo_trip:<cause>" | "late_attach:<s>" | "park_refused" | "resume_refused", eval: {n: int, success_rate: number}, export: bool, domain: "sim"|"real"}`. `domain: "real"` is admitted only in LA4 tests (fake backend).

### FL2. Run layout, manifests, events, usage — `src/so101_rl/lab/registry.py`, `src/so101_rl/lab/schemas/*.schema.json`

```
runs/                                   # group so101, g+rX after OS1-HW (approver + servo read; koen writes)
  <run_id>/
    run.json                            # manifest, immutable after creation (schema run.v1)
    status.json                         # mutable, atomic replace (status.v1)
    events.jsonl                        # append-only, one JSON object per line (event.v1)
    usage.json                          # atomic replace, high-water fields (usage.v1)
    metrics.csv                         # source of truth (plugin via ScalarMonitor)
    tensorboard/                        # ScalarMonitor
    ckpt_<step>.pt, ckpt_last.pt
    review.json                         # review triggers (list)
    eval.json, renders/, review_checklist.md     # evaluate
    gate.json                           # numeric gate verdict (gate.v1)
    policy_export.pt                    # export
    artifacts.json                      # ArtifactManifest at exit (artifacts.v1)
    notes.md                            # write_note (untrusted)
    segments/<seg>.json                 # {seg, pid, unit, boot_id, started_at, ended_at, exit_code, reason}
    approval_request.json | attach.json | limits.json      # LA4
  queue/
    start/<created_at>-<run_id>.json    # start requests (atomic: tmp → fsync → replace)
    control/<t>-<run_id>-<action>.json  # control requests
    idem/<idempotency_key>.json         # O_CREAT|O_EXCL: {params_sha256, run_id, created_at}
  supervisor/
    heartbeat.json                      # {t, pid, boot_id, leases: {...}, queue_len, gpu_health}
    state.json                          # re-adoption snapshot (running units ↔ run ids)
    gpu_health.json                     # last NVML sample(s)
  items/<item_id>.json                  # needs-human queue (item.v1); closed items keep the file with closed_at
  audit.jsonl                           # every tool call / supervisor decision (audit.v1)
  ledger.json                           # derived cache; `lab status` recomputes from usage.json files
  HALT                                  # presence = compute halt (control_run(kill) never creates it; `lab halt` does)
  sessions.jsonl                        # LA4: two-phase session verdicts (session.v1)
  notebook.md                           # write_note without run_id
  .registry.lock                        # flock for run creation / idempotency
```

- **`run_id`** = `<trainer>-<task>-<YYYYMMDD>-<HHMMSS>-<4 hex>`; grammar `^[a-z_]+-[a-z0-9_]+-[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$` (design §5.3; shared with the records tools through `so101_rl.authority.RUN_ID_RE`). Created under `flock runs/.registry.lock`; the directory is created with `run.json` written last (an attacher that sees no `run.json` treats the run as not yet created).
- **`run.json` (run.v1):** `{schema: 1, run_id, created_at (RFC 3339 with offset), kind: RunKind, trainer, trainer_version, config_version, task, domain, config: {...}, config_sha256, init_from: {ref: run_id|path, sha256} | null, plan: JobPlan, provenance: {git_sha, git_dirty: bool, pixi_lock_sha256, env, so101_yaml_sha256, sim_params_sha256|null, calibration_sha256, host, boot_id, lab_version}, idempotency_key, parent_run: run_id|null, pipeline: {evaluate: {...}|null, export: bool} | null, note: string|null}`.
- **`status.json` (status.v1):** `{schema: 1, run_id, state, segment, pid, unit, boot_id, started_at, updated_at, heartbeat_at, step, phase, progress_frac|null, leases: [Lease], deadline_at, pause_reason|null, stop_reason|null, last_event_seq, client_id|null}`. `state ∈ {queued, starting, running, paused, finished, stopped, crashed, refused, expired, awaiting_approval}`; `stop_reason ∈ {budget_exhausted, review_stop, user_stop, stall, disk_low, thermal_gpu_timeout, halt, servo_trip:<cause>, servo_needs_human, servo_latched, servo_wait_timeout, client_refused, session_ended:cap, quiet_hours, arm_minutes, min_gap}`; `pause_reason ∈ {review, thermal_gpu, gpu_health_unknown, servo_wait, user}`.
- **`events.jsonl` (event.v1):** `{seq: int (per run, monotonic), t, kind, run_id, segment, data: {}}`; `kind ∈ {created, queued, refused, lease_granted, segment_started, heartbeat, heartbeat_stall, paused, resumed, checkpoint, review_trigger, eval_done, gate, export, finished, stopped, crashed, expired, item_opened, item_closed, note, host_slept, reboot_detected, session_request, session_open, session_closed, servo_state, attach, client_refused, verdict}`. Any `data.message` is untrusted (FL3 envelope on read).
- **`usage.json` (usage.v1):** `{schema: 1, run_id, task, trainer, kind, budget_key, budget_class, gpu_h_progress, gpu_h_wall, arm_min, segments: [{seg, started_at, ended_at, gpu_h_progress, arm_min}], high_water: {gpu_h_progress, attempts_counted: 1|0}, updated_at}`. **GPU-h = progress-holding time**: the sum over heartbeat intervals whose `step` advanced, each interval capped at `policy.stall.cap_min` (30). The ledger (`lab/ledger.py`) is derived from every `usage.json` under `runs/` plus `import-run` entries (`runs/ledger_imports.jsonl`); a recomputation that yields fewer attempts than `ledger.json`'s high-water for any `budget_key` writes `runs/HALT` with reason `ledger_shrink` and opens an item.
- **`gate.json` (gate.v1):** `{schema: 1, task, kind: "L1"|"L2"|"s4_exit_numeric"|"zero_shot", passed: bool, values: {...}, source_run: run_id, computed_at, human_signed: {record_sha256, seq} | null}` — numeric only; the human half is the signed `gate` record (FL6).
- **`items/<item_id>.json` (item.v1):** `{schema: 1, item_id: "it-<YYYYMMDD>-<6 hex>", kind, run_id|null, session_id|null, created_at, closed_at|null, requires: "ack"|"approval"|"physical"|"none", summary (untrusted), command: string|null (display only, e.g. "so101-ack it-...")}`; `kind ∈ {review_real, gpu_health_unknown, ledger_shrink, disk_critical, stall_killed, servo_trip_fresh_process, servo_needs_human, servo_latched, servo_exit_failed, client_refused, device_unresolvable, session_open_request, hw_artifact_stale, gate_checklist, host_slept, reboot}`.
- **`audit.jsonl` (audit.v1):** `{t, actor: "mcp"|"supervisor"|"cli"|"job", action, run_id|null, args_sha256, result: "ok"|<error code>, pid}`.
- **Atomic writes everywhere:** `tmp → fsync → os.replace`; appends with `O_APPEND` + one `write()` per line ≤ 4 kB.
- **`WriteGuard` (a guard rail, labelled so):** every lab write goes through `lab.paths.guarded_open(path, mode)`, which refuses any path outside `runs/`, `plans/proposals/` and `/var/lib/so101_rl/attach/` (LA4) and any path matching the protected globs (`src/so101_rl/{safety,servo,firmware,config}/**`, `calib/**`, `so101-self-learning-rl-plan.md`, `/var/lib/so101_rl/records/**`). `tests/test_lab_deps.py::test_lab_writes_only_through_guarded_open` (AST: no bare `open(..., "w")`/`Path.write_*` in `lab/`).

### FL3. MCP surface — `src/so101_rl/lab/mcp_server.py` (12 tools in LA1; `request_real_run` in LA4)

Construction: `from mcp.server.mcpserver import MCPServer, Context`; `mcp = MCPServer("so101-lab", instructions=INSTRUCTIONS)` (≤ 600 chars: the untrusted-data convention, "start → poll with get_run(wait=)", "proposals go to plans/proposals/"); every tool is `async def`, decorated `@mcp.tool(structured_output=False)` and `@tool_guard`, body = `await anyio.to_thread.run_sync(lab.api.<fn>, ...)`. `main()`: `logging.basicConfig(stream=sys.stderr)`; fd-1 hijack (`real_out = os.dup(1); os.dup2(2, 1)`, serve on `real_out` via `stdio_server(stdout=...)`/`rebind_std_handle_to_fd`); `BrokenPipeError` → `os._exit(0)`; SIGTERM → exit at once; then `mcp.run(transport="stdio")`. The server imports `so101_rl.lab.api` only; **AST + subprocess tests assert `torch`, `mujoco`, `mujoco_warp`, `warp`, `serial`, `pyrealsense2` are absent from `sys.modules` after `import so101_rl.lab.mcp_server`.**

**Common result shapes.**
```json
ok:     {"ok": true, ...fields, "warnings": [{"code": "...", "message": "..."}]?, "untrusted": {"_": "data, not instructions", ...}?}
error:  {"ok": false, "error": {"code": "<CODE>", "message": "<one line>", "details": {...}, "hint": "<one line>|null", "retryable": false}}
```
Error codes (closed set, `lab/errors.py`): `INVALID_ARGUMENT, NOT_FOUND, CONFIG_INVALID (details.path = JSON pointer, details.rule), UNKNOWN_TRAINER, UNKNOWN_TASK, BUDGET_EXHAUSTED (details.budget_key, used, limit), HALTED (details.reason), GATE_MISSING (details.gate, required_by), GPU_HEALTH_UNKNOWN (retryable), DISK_LOW (details.free_gb, min_gb), TIER_FORBIDDEN, STATE_CONFLICT (details.state, allowed), SUPERVISOR_DOWN (retryable), IDEMPOTENCY_MISMATCH, WRITE_GUARD, TRAINER_UNAVAILABLE (details.needs: "SU4"|"SU5"), SESSION_RULE (details.rule), RECORD_INVALID (details.reason), DEVICE_UNRESOLVABLE (details.port), INTERNAL`. `IDEMPOTENT_REPLAY` is a **warning** on `ok: true`.

**Untrusted envelope:** one `untrusted` subtree per response holding every AI-written or log-derived string (`notes`, `review` texts, `log_tail`, event `message`s, proposal text, trainer-supplied names); strings stripped of ANSI/C0/C1 bytes, capped (notes 2 kB, log excerpts 4 kB, event messages 512 B) with `truncated: true` when cut.

**Tools (names, parameters, returns).** Descriptions: first line ≤ 120 chars, ≤ 400 chars total, no examples; contract test `len(json.dumps(list_tools)) ≤ 12_288`.

| # | Tool | Parameters (JSON schema via type hints) | Returns (`ok: true` payload) |
|---|---|---|---|
| 1 | `get_status()` | — | `{halt: bool, supervisor: {alive, heartbeat_age_s, pid, boot_id}, leases: {gpu: run_id|null, arm: run_id|null, cpu: [run_id]}, queue: [{run_id, kind, waiting_s, ttl_s}], running: [{run_id, state, step, phase, heartbeat_age_s}], gpu: {temp_c, mem_temp_c|null, util_pct, vram_used_gb, health: "known"|"unknown"|"hot"}, disk: {free_gb, runs_gb}, ledger: {budget_key: {attempts_used, attempts_max, gpu_h_used, gpu_h_max, probe_gpu_h_used}}, needs_human: [{item_id, kind, run_id, requires, created_at}], arm: null | {unit_state, session_id, arm_ns, parked, cap_min, elapsed_min, snapshot_sha, last_session: {servo_clean, weld_ok}, tier_pointer: {tier, seq, sha256}|null, records_store_check_age_s}}` (`arm` is null until LA4) |
| 2 | `find_runs(task: str|None, trainer: str|None, state: str|None, kind: str|None, since: str|None, limit: int = 20 (≤ 50), cursor: str|None)` | | `{runs: [{run_id, state, kind, trainer, task, created_at, updated_at, step, stop_reason}], next_cursor: str|null}` (keyset on `(created_at desc, run_id)`; cursor = base64 of last key + filter hash; filter change with an old cursor → `INVALID_ARGUMENT`) |
| 3 | `get_run(run_id: str, wait: Literal["none","change","terminal"] = "none", timeout_s: int = 0 (≤ 300), since_seq: int = 0, include: list[Literal["events","usage","artifacts","review","eval","gate","notes"]] = ["events"])` | | `{run: run.json, status: status.json, events: [last 50 ≥ since_seq], usage?, artifacts?, review?, eval?, gate?, untrusted: {notes?, event_messages}}`; `wait="change"` returns when `status.last_event_seq > since_seq` or on timeout (`timed_out: true`) |
| 4 | `get_metrics(run_id: str, keys: list[str] (1..8), max_points: int = 200 (≤ 1000), step_range: tuple[int,int]|None)` | | `{columns: [...], n_raw, stride, buckets: [{step, min: {...}, mean: {...}, max: {...}, last: {...}}]}` from `metrics.csv` |
| 5 | `check_gates(task: str, policy_ref: str|None)` | | `{task, gates: {L1: gate.json|null, L2: ..., s4_exit_numeric: ..., s4_exit_signed: {record_sha256, seq}|null, zero_shot: ...}, real_admissible: bool, reasons: [str]}` — derived only from artifacts and records; executes nothing |
| 6 | `start_run(trainer: str, task: str, config: dict, kind: Literal["attempt","probe","eval","export","gates","pipeline"] = "attempt", idempotency_key: str, init_from: str|None, pipeline: dict|None, note: str|None)` | | `{run_id, state: "queued"|"refused", plan: JobPlan, refusal: error|null, warnings?}`; refusals come back as `ok: false` with the codes above; `kind="probe"` is limited to `policy.budgets.probe_max_min` |
| 7 | `control_run(run_id: str, action: Literal["pause","resume","stop","kill","note"], overrides: dict|None, reason: str|None)` | | `{run_id, state, accepted: bool, request_id}`; `overrides` only with `resume` and only for `plan.resumable_fields` (else `CONFIG_INVALID`); `kill` = SIGKILL via the supervisor; `stop` on a real job = the job sends `STOP_REQ` after `PARK` |
| 8 | `evaluate_policy(policy_ref: str, task: str, episodes: int (1..2000), dr_stage: int|None, gate: bool = false, render: int = 0 (≤ 16), idempotency_key: str)` | | `{run_id}` (an `eval` run on the trainer that produced the policy; `TRAINER_UNAVAILABLE` until LA2 for non-fake policies) |
| 9 | `export_policy(run_id: str, idempotency_key: str)` | | `{run_id}` (an `export` run) |
| 10 | `run_gates(task: str, idempotency_key: str)` | | `{run_id}` (a `gates` run: `sim-gates` under the cpu lease; `TRAINER_UNAVAILABLE` until LA2) |
| 11 | `write_note(text: str (≤ 4096), run_id: str|None, tags: list[str] (≤ 8))` | | `{note_id, path}`; appended to `runs/<run_id>/notes.md` or `runs/notebook.md` as `- <t> [<tags>] <text>` |
| 12 | `propose_decision(title: str (≤ 120), text: str (≤ 16384), supersedes: str|None)` | | `{path: "plans/proposals/<YYYY-MM-DD>-<slug>.md"}`; the ONLY lab write outside `runs/` (P9); never edits §10 |
| LA4 | `request_real_run(trainer, task, config, kind: Literal["zero_shot_eval","finetune","excitation"], episodes: int (1..100), speed_scale: Literal[0.5,0.75,1.0], arm_min_cap: int (1..120), policy_ref: str|None, ai_notes: str|None (≤ 2048), idempotency_key)` | registered only when `SO101_LAB_REAL_ARM=1` (a guard rail, labelled in the description) | `{run_id, state: "awaiting_approval"|"queued"|"refused", approval_request: path, command: "so101-approve <run_id>"}`; ≤ 1 open request per kind |

**Resources** (read-only views; each ≤ 32 kB with a truncation marker): `lab://registry/trainers`, `lab://registry/tasks`, `lab://schemas/{name}` (`run`, `status`, `event`, `usage`, `item`, `trainer/<plugin>`), `lab://runs` (index ≤ 200 rows), `lab://runs/{run_id}`, `lab://runs/{run_id}/events` (last 200), `lab://decisions` (the plan's §10 table, read-only extract), `lab://needs-human`, `lab://sessions` (LA4), `lab://policy`. **Prompts:** `plan_next_run(goal)`, `review_run(run_id)`, `triage_needs_human()`.

**Token discipline (A11):** `tools/list` ≤ 12 288 B; every default response on the fixture tree ≤ 32 kB (≈ 8k tokens); no raw logs, no video bytes, no file contents other than the capped excerpts.

**`.mcp.json`** (repo root, LA1):
```json
{"mcpServers": {"so101-lab": {"type": "stdio", "command": "pixi",
  "args": ["run", "--frozen", "-e", "cpu", "lab-mcp"],
  "env": {"PYTHONUNBUFFERED": "1"}}}}
```
`SO101_LAB_REAL_ARM` is **not** set here (LA4 documents adding it by hand as a guard rail).

### FL4. Supervisor, queue files, units — `src/so101_rl/lab/supervisor/`, `src/so101_rl/lab/units/`

- **Request files.** `queue/start/<created_at>-<run_id>.json` = `{run_id, requested_at, requested_by: "mcp"|"cli"|"pipeline", ttl_s}` (the manifest is already in `runs/<run_id>/run.json`, state `queued`); `queue/control/<t>-<run_id>-<action>.json` = `{request_id, run_id, action, overrides, reason, requested_at}`. The supervisor consumes a file by renaming it into `queue/done/` after acting (idempotent on restart).
- **Loop (1 Hz):** read `HALT`; sample GPU health (every 10 s); detect suspend (`time.clock_gettime(CLOCK_BOOTTIME) − CLOCK_MONOTONIC` jump > 5 s → event `host_slept`, approvals void, real queue voided) and reboot (`/proc/sys/kernel/random/boot_id` ≠ `status.json.boot_id` of a running run → `crashed`; `supervisor/state.json.boot_id` ≠ current → real queue voided, event `reboot_detected`); re-adopt running units (`systemctl --user show -p MainPID,ActiveState,ExecMainStartTimestamp so101-job-*`), reconcile with `status.json.pid`/`boot_id`; expire queued entries past their TTL (sim 24 h, real 4 h); apply control requests; stall ladder; disk check; admit queued runs FIFO with `arm` first while leases allow; write `heartbeat.json`.
- **Leases:** `gpu × 1`, `arm × 1`, `cpu × 2` (policy.yaml). A thermal/health pause **holds** the gpu lease (auto-resume at 75/85 °C); a review/AI pause **releases** leases (P7). Pausing = `control stop` with `pause_reason` → the job checkpoints and exits 0 with `state: paused`.
- **Launcher protocol** (`lab/supervisor/units.py`): `class Launcher(Protocol): def start(self, unit: str, argv: list[str], *, props: Mapping[str, str], cwd: Path, env: Mapping[str, str]) -> None; def show(self, unit: str) -> UnitState; def stop(self, unit: str) -> None; def kill(self, unit: str) -> None; def list_units(self, glob: str) -> list[UnitState]`. `SystemdUserLauncher` runs `systemd-run --user --unit <unit> --collect --property=KillMode=mixed --property=TimeoutStopSec=130 [--property=MemoryMax=<policy>] --working-directory=<repo> --setenv=PYTHONUNBUFFERED=1 --setenv=SO101_LAB_RUNS=<runs> -- pixi run --frozen -e <env_name> python -m so101_rl.lab.job --run runs/<run_id> --segment <seg>`; unit name `so101-job-<run_id>-<seg>.service` (perception: `so101-perception-<session_id>.service`, LA4). `FakeLauncher` (tests) runs the job entry in a subprocess/thread and records calls; `SubprocessLauncher` (no systemd user session: CI) is used only when `systemctl --user is-system-running` fails and is **never** used for real-arm runs.
- **Stall ladder:** `heartbeat_at` older than 30 min → event `heartbeat_stall` + warning; 45 min → `stop`; 50 min → `kill` → `crashed` + item `stall_killed`. The job's **own hard deadline** (`deadline_at` = start + budget wall + 5 min) fires inside the job even if the supervisor is dead (`RunContext.should_stop()` → checkpoint → exit `stopped: budget_exhausted`).
- **Thermal/health (P8):** NVML via `nvidia-ml-py` (`pynvml`), GPU core pause ≥ 85 °C / resume ≤ 75 °C, memory junction pause ≥ 95 / resume ≤ 85 when the field is observable; six consecutive failed samples → pause all GPU jobs (lease held) + item `gpu_health_unknown`; `health: unknown` blocks new GPU admissions (`GPU_HEALTH_UNKNOWN`).
- **Budgets (P6):** per `budget_key`: `max_attempts` (3), `gpu_h_max` (reach 6, die 30, from policy.yaml); probes: ≤ `probe_max_min` (10) wall, pooled ≤ `probe_gpu_h_per_task` (1), not attempts; evals are `eval` class (counted in GPU-h, not attempts). Admission refuses `BUDGET_EXHAUSTED` when `attempts_used ≥ max_attempts` or `gpu_h_used + estimate > gpu_h_max`.
- **Review policies (P7):** `review_trigger` on an attempt → `paused: review` (process exits after a checkpoint, leases released); `control_run(resume)` continues from `ckpt_last.pt`; the **third** trigger with the same `name` on the same run → `stopped: review_stop`; probes continue (event only); real runs (LA4) → `stopped` + item `review_real` (needs a signed ack). The method is never switched automatically.
- **Pipelines:** `kind="pipeline"` creates the parent attempt; on `finished`, the supervisor creates the children in order: `eval` (with `gate: true`) → numeric gate (`gate.json`) → `export` (only if the numeric gate passed); children carry `parent_run`; `check_gates` shows the chain. The S4 exit gate stays half human (the signed `gate` record, FL6).
- **Disk:** `free_gb < policy.disk.min_free_gb` (20) → no admissions (`DISK_LOW`); `< critical_free_gb` (5) → running jobs get `should_stop` at the next checkpoint (`stopped: disk_low`) + item `disk_critical`; per-run quota `plan.disk_gb_estimate × 2`.
- **Notifications:** `notify-send` when available + the dashboard; nothing else.
- **User units** (installed by `pixi run lab install-units` into `~/.config/systemd/user/`, templates in `lab/units/`): `so101-lab-supervisor.service` (`Type=simple`, `ExecStart=pixi run --frozen -e cpu lab-supervisor`, `WorkingDirectory=<repo>`, `Restart=on-failure`, `RestartSec=5`, `KillMode=process`, `Environment=PYTHONUNBUFFERED=1`, `WantedBy=default.target`), `so101-lab-dashboard.service` (same shape, `lab-dashboard`). `install-units` never touches system units.

### FL5. `src/so101_rl/lab/policy.yaml` (strict: exactly these keys; loaded by `lab/paths.py:load_policy`)

```yaml
schema: 1
leases: {gpu: 1, arm: 1, cpu: 2}
queue_ttl_h: {sim: 24, real: 4}
budgets:
  max_attempts: 3
  gpu_h_max: {"s4.reach": 6.0, "s4.die_to_zone": 30.0, "s5.reach.finetune.sb3": 8.0, "s5.reach.finetune.tdmpc2": 8.0, "s5.die_to_zone.finetune.sb3": 15.0, "fake.reach": 1.0, "fake.die_to_zone": 1.0}
  probe_max_min: 10
  probe_gpu_h_per_task: 1.0
stall: {warn_min: 30, stop_min: 45, kill_min: 50, cap_min: 30}
thermal: {gpu_pause_c: 85, gpu_resume_c: 75, mem_pause_c: 95, mem_resume_c: 85, sample_s: 10, empty_samples_pause: 6}
disk: {min_free_gb: 20, critical_free_gb: 5}
review: {max_pauses_same_trigger: 2}
suspend: {jump_s: 5}
session:            # LA4
  arm_timeout_s: 110
  exit_wait_s_extra: 10          # exit_wait_s = bound_s + 10
  device_prefix: "/dev/serial/by-id/"
  park_tolerance_ticks: 20
  resume_wait_min: 15
  perception_max_age_s: 1.0
records:            # LA4/LA5
  store: /var/lib/so101_rl/records
  allowed_signers: /etc/so101_rl/allowed_signers
  attach_dir: /var/lib/so101_rl/attach
  servo_state_dir: /var/lib/so101_rl/servo
  expiry_h: 2
  approved_skew_s: 60
```

### FL6. Signed human records (P2, P11, P14) — formats, store, grammars

**Common body** (canonical JSON: `json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"`; every string ASCII `0x20..0x7E`): `{type: "approval"|"gate"|"tier"|"ack", schema: 1, seq: int (monotonic per type), supersedes_sha256: hex|null (required for tier and gate), created_at: RFC 3339 with offset, signer: "so101-approver", ...type fields}`. File pair `<name>.json` + `<name>.sig`; signature = `ssh-keygen -Y sign -f <key> -n so101-<type> <file>` over the exact bytes written; verification = `ssh-keygen -Y verify -f /etc/so101_rl/allowed_signers -I so101-approver -n so101-<type> -s <file>.sig < <file>`; a record whose `type` does not match the namespace it verifies under is invalid.

**Store** `/var/lib/so101_rl/records/` (approver-owned, dirs `0755`, files `0644`, koen read-only): `approvals/<run_id>.<seq>.json(.sig)`, `gates/<eval_run_id>.<seq>.json(.sig)`, `tiers/tier.<seq>.json(.sig)`, `acks/<item_id>.<seq>.json(.sig)`, `current/tier` and `current/gate.<task>` (one line: `<sha256> <seq>`, approver-written), `revoked.jsonl` (each line a signed mini-record `{type: "revoke", schema: 1, seq, sha256, reason, created_at}` + `.sig` lines in `revoked.sigs`), `seq/<type>` (last seq used; approver-written). **Anti-rollback:** a verifier accepts a tier/gate record only if its sha256 is the current pointer, it is not revoked, and `seq ≥` the highest `seq` the verifier has ever accepted for that type (the servo persists `highest_seen.json` in its state dir; the lab persists `runs/supervisor/records_seen.json`, a mirror).

**Type fields.**
- `approval`: `{run_id, request_sha256 (over the bytes of approval_request.json as read), config_sha256, policy_sha256|null, hw_artifacts: [{id, path, sha256}], approved_at, expires_at (= approved_at + expiry_h ≤ 2 h), nonce: 32 hex (128-bit, os.urandom), session_id: int (the servo's arm_ns: the session must be open), limits: {episodes: int 1..100, arm_min_cap: int 1..120, speed_scale_max: 0.5|0.75|1.0}, ai_text_shown: bool}`. Phrase: `APPROVE <run_id> <episodes>x<speed_scale> <CHAL>`.
- `gate`: `{eval_run_id, task, eval_sha256, checklist_sha256, renders_manifest_sha256, success: number, wilson_lo: number}`. Phrase: `SIGN <eval_run_id> <success_pct> <CHAL>` where `success_pct` = `round(success*100)`.
- `tier`: `{tier: 0|1, by, date, milestone_sha, snapshot_sha, soak_reports: [{id: "attended_30min"|"goal_directed_120min"|"park_resume", path, sha256}], hw_artifacts: [{id, path, sha256}], rules: {tasks: [..], kinds: [..], trainers: [..], policies: [{policy_id, sha256}], session_cap_min: 60|120|240|480, clean_sessions_at_cap: int, max_episodes_per_run: int, min_gap_min: 2, quiet_hours: ["23:00","07:00"], max_arm_min_per_day: 240, day_boundary: "local-midnight", speed_scale_max, require_rt_active: true, require_camera_moved_gate: true, valid_until}}`. Phrase: `TIER <tier> <session_cap_min> <CHAL>`.
- `ack`: `{item_id, kind, run_id|null, session_id|null, servo_summary_sha256|null, note}`. Phrase: `ACK <item_id> <kind> <CHAL>`.

**Grammars (`so101_rl.authority.grammar`, applied to every field of every request and record; any byte outside `0x20..0x7E` or `\n` → reject, never sanitise; file size ≤ 64 kB):** `RUN_ID_RE = ^[a-z_]+-[a-z0-9_]+-[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$`; `ITEM_ID_RE = ^it-[0-9]{8}-[0-9a-f]{6}$`; `SHA256_RE = ^[0-9a-f]{64}$`; `NONCE_RE = ^[0-9a-f]{32}$`; `kind ∈ {zero_shot_eval, finetune, excitation}`; `task ∈ {reach, die_to_zone}`; `trainer ∈ TRAINERS`; `episodes` int 1..100; `speed_scale ∈ {0.5, 0.75, 1.0}` (decimal text exactly `0.5|0.75|1.0`); `arm_min_cap` int 1..120; paths: relative, under `runs/` or `calib/`, no `..`, chars `[A-Za-z0-9._/-]`; RFC 3339 timestamps with an explicit offset; `CHAL` = 5 lowercase letters from `os.urandom`.

**`approval_request.json`** (koen-written by the lab, LA4; request.v1): `{schema: 1, run_id, kind, task, trainer, episodes, speed_scale, arm_min_cap, config_sha256, policy: {policy_id, path, sha256}|null, hw_artifacts: [{id, path, sha256}], session_id: int, created_at, ai_notes: str|null}` — exactly these keys; the records tool renders machine fields first, then a highlighted diff against the last approved request of the same kind, then the escaped and boxed `ai_notes` only with `--show-ai-text`.

**Session request** (koen-written, servo-validated; `/var/lib/so101_rl/attach/sessions/<session_id_request>.json`, request-session.v1): `{schema: 1, request_id: "<YYYYMMDD>-<HHMMSS>-<4 hex>", tier_claim: 0|1, requested_cap_min: int 1..480, run_ids_expected: [run_id], tick_log: bool, perception: bool, created_at}`. The unit instance name `%i` is the `request_id`; the authoritative `session_id` is the servo's `arm_ns` (FL8).

**Attach claim** (koen-written by the job, servo-read; `/var/lib/so101_rl/attach/<request_id>/<client_id>.json`, attach.v1): `{schema: 1, run_id, client_id: int (u32, non-zero), claim: "approval"|"tier1", record_path: str|null, run_dir: str, created_at}`; copied to `runs/<run_id>/attach.json`.

**Servo session store** (`so101-servo`-written; `/var/lib/so101_rl/servo/`): `nonces.jsonl` (`{nonce, record_sha256, run_id, session_id, t}`), `highest_seen.json` (`{tier: seq, gate: seq}`), `arm_minutes.jsonl` (`{session_id, start_local, end_local, minutes_energised}`), `sessions/<request_id>/{state.json, summary.json, tick_log.bin, authority.jsonl, first_mcu_status.json}`; `state.json` = `{phase: "boot_wait"|"armed"|"parked"|"running"|"torque_off"|"exit", since, pid, arm_ns|null, deadline_at|null}` (written at every phase change; the lab's "the servo is waiting — press the button" item is raised only when `phase == boot_wait`). `authority.jsonl` = `{t, check: "session"|"attach"|"resume", client_id|null, run_id|null, verdict: "ok"|<code>, detail, duration_ms}`.

**`runs/sessions.jsonl`** (lab-written, session.v1): `{request_id, session_id (arm_ns), tier, cap_min, opened_at, closed_at, servo_exit_code, servo_clean: bool|null, servo_clean_reasons: [..], weld_ok: bool|null (filled at the NEXT session's first status), first_mcu_status_next: int|null, clean: bool|null, arm_min}`.

### FL7. Authority module — `src/so101_rl/authority/__init__.py` (stdlib only; snapshotted; installed verbatim as `/usr/local/lib/so101_rl/so101_authority.py`)

One file, no imports from `so101_rl` (AST-tested), no third-party imports; used by (a) the servo (`servo/main.py`, `servo/core.py` through a callable), (b) the records tools (`import so101_authority` from their install dir), (c) the lab mirror (`import so101_rl.authority`). Protected path (`Work-Item: OS1`).

```python
AUTHORITY_VERSION: Final[int] = 1
RECORD_TYPES = ("approval", "gate", "tier", "ack"); NAMESPACE = {t: f"so101-{t}" for t in RECORD_TYPES}
RUN_ID_RE, ITEM_ID_RE, SHA256_RE, NONCE_RE, PATH_RE: Final[re.Pattern]
class RecordError(ValueError): code: str          # "GRAMMAR", "SIGNATURE", "NAMESPACE", "TYPE", "SEQ", "NOT_CURRENT", "REVOKED", "EXPIRED", "SKEW", "HASH", "SESSION", "NONCE", "RULE", "TIMEOUT"
def canonical_bytes(body: Mapping) -> bytes
def sha256_bytes(b: bytes) -> str; def sha256_file(path: Path, *, max_bytes: int = 1 << 30) -> str
def read_exact(path: Path, *, max_bytes: int = 65536, expect_uid: int | None = None) -> bytes   # O_RDONLY|O_NOFOLLOW|O_CLOEXEC, fstat regular/owner/size, one read
def validate_request(raw: bytes) -> dict           # approval_request.json grammar → dict (raises RecordError("GRAMMAR"))
def validate_record(body: Mapping, type_: str) -> None
def challenge() -> str                              # 5 lowercase letters from os.urandom
def parse_phrase(line: str, *, kind: str, expect: Mapping[str, str]) -> bool   # exact match of APPROVE/SIGN/TIER/ACK phrase incl. CHAL
@dataclass(frozen=True)
class Verified: type: str; seq: int; sha256: str; body: Mapping; path: Path
class Verifier:
    def __init__(self, records_dir: Path, allowed_signers: Path, *, ssh_keygen: str = "/usr/bin/ssh-keygen",
                 now: Callable[[], float] = time.time, seen_path: Path | None = None, timeout_s: float = 0.12) -> None
    def verify_file(self, path: Path, type_: str) -> Verified          # signature + namespace + type + schema + grammar
    def verify_current(self, type_: str, key: str | None = None) -> Verified   # + current pointer + revocation + seq ≥ highest seen (persists seen)
    def check_approval(self, rec: Verified, *, run_json_sha: str, config_sha: str, policy_sha: str | None,
                       hw_shas: Mapping[str, str], session_id: int, now: float) -> RunLimitsDict   # expiry sanity, skew ≤ 60 s, hashes, session
    def check_tier(self, rec: Verified, *, snapshot_sha: str, soak_shas: Mapping[str, str], hw_shas: Mapping[str, str], now: float) -> Mapping
class NonceLedger:
    def __init__(self, path: Path) -> None
    def consume(self, nonce: str, *, record_sha256: str, run_id: str, session_id: int) -> bool   # O_APPEND + fsync; False if already present
class ArmMinuteLedger:
    def __init__(self, path: Path, *, tz: datetime.tzinfo) -> None
    def minutes_today(self, now: datetime.datetime) -> int            # sessions STARTING in the local calendar day
    def add(self, session_id: int, start: datetime.datetime, end: datetime.datetime, minutes: int) -> None
class Tier1Rules:
    @classmethod
    def from_record(cls, rec: Verified) -> Tier1Rules
    def admit_session(self, *, now: datetime.datetime, tier_claim: int, cap_min: int, minutes_today: int, clean_sessions_at_cap: int) -> None   # raises RecordError("RULE", detail)
    def admit_run(self, *, now: datetime.datetime, run_json: Mapping, policy_sha: str | None, last_run_end: datetime.datetime | None, minutes_today: int) -> RunLimitsDict
class ServoAuthority:                       # facade used by servo/main.py; every verdict appended to authority.jsonl
    def __init__(self, *, records_dir: Path, state_dir: Path, allowed_signers: Path, session_request: Path, snapshot_sha: str, clock: Callable[[], int], tz: datetime.tzinfo) -> None
    def warm_up(self) -> None                                          # one verify of a self-test record (BOOT; page cache)
    def verify_session(self) -> Mapping                                # pre-ARM: tier record (tier ≥ claim), cap, quiet hours, per-day minutes, devices; raises RecordError → servo exit 2
    def verify_attach(self, claim_path: Path, *, arm_ns: int, now_ns: int) -> AttachVerdict   # first RUN / RESUME_REQ of a new client_id; consumes the nonce on OK
    def close_session(self, *, arm_ns: int, start_ns: int, end_ns: int, minutes: int) -> None
@dataclass(frozen=True)
class AttachVerdict: ok: bool; code: str; limits: RunLimitsDict | None; duration_ns: int
RunLimitsDict = Mapping[str, int | float | None]   # {"episodes", "arm_min_cap", "speed_scale_max", "source"}
```

Tests use **real cryptography**: a throwaway Ed25519 key per test (`ssh-keygen -t ed25519 -N '' -f <tmp>/key`), a temp `allowed_signers`, a temp store; no fake crypto. `tests/test_authority.py` covers every `RecordError.code` with a positive control, `test_authority_is_stdlib_only` (AST), `test_installed_copy_is_byte_identical` (the install script copies this file; the test checks the script references `src/so101_rl/authority/__init__.py`).

### FL8. SV1 — servo park / idle / resume / exit (`servo/shm.py`, `safety/types.py`, `safety/state_machine.py`, `servo/core.py`, `servo/process.py`, `servo/main.py`, `servo/soak.py`, `config/{schema,loader}.py`, `config/so101.yaml`, `env/backend.py`)

1. **`servo/shm.py` (layout version 3).** `CmdMode.PARK = 4`, `CmdMode.RESUME_REQ = 5`. `CMD_DTYPE` = `seq u8, t_mono_ns i8, state_seq u8, mode u4, episode u4, delta_ticks f4[6], client_id u4, crc32 u4` (64 B; `_pad` replaced by `client_id` at offset 56). `STATE_DTYPE` v3 appends after `read_ok` (225): `parked u1 @226, client_refused u1 @227, park_refused u1 @228, resume_refused u1 @229, arm_ns i8 @232, crc32 u4 @240` → `STATE_SIZE == 248`, stride 256 and `TOTAL_SIZE` unchanged (asserts updated). `SHM_LAYOUT_VERSION = 3`; `attach` refuses 2. `last_cmd_seq` in the state record lets a new writer continue from `last + 1`; **dedupe is monotonic**: `seq <= last_seq` → not new (a repeated or older seq is ignored).
2. **`safety/types.py`.** `TripCause.IDLE_TIMEOUT = 65536`; `ResumePolicy.ON_REQUEST = "on_request"`; `RESUME_BY_CAUSE[IDLE_TIMEOUT] = ON_REQUEST`; `state_machine._POLICY_RANK = {AUTO: 0, ON_REQUEST: 1, FRESH_PROCESS: 2, HUMAN_PHYSICAL_RESET: 3}`.
3. **`config`.** New strict section `idle: {park_timeout_s: 600 (30..3600), park_tolerance_ticks: 20 (1..100)}` → `IdleConfig`, `So101Config.idle`. New function `config.loader.check_session_device_paths(port: str, heartbeat_port: str | None, *, prefix: str, owner_uid: int) -> None` (P13: both must start with `prefix`, resolve, and be owned by `owner_uid`; raises `ValueError` with the offending path; used by `servo/main.py` only when `--session` is given). The shipped `so101.yaml` `port`/`heartbeat_port` are **unchanged** (the pinned unit argv passes `--port /dev/serial/by-id/so101-arm --heartbeat-port /dev/serial/by-id/so101-mcu`).
4. **`safety/state_machine.py`.** New `SmInputs` fields: `park_requested: bool`, `resume_requested: bool`, `cmd_mode: int`, `park_in_tol: bool`, `client_admitted: bool`, `now_ns: int`. New `SmOutputs` fields: `parked: bool`, `latch_target: bool` (on the PARK-accept step), `park_refused_inc: bool`, `resume_refused_inc: bool`, `client_refused_inc: bool`. New state: `self._parked`, `self._parked_deadline_ns`, `self._resume_on_request`. Rows (first match wins), inserted into the existing order:
   - (existing) LATCHED absorbing; link faults → LATCHED.
   - (existing) TORQUE_OFF: **new first sub-row**: `resume_requested and client_admitted and block_ok and (fault_bits & ~_RESUME_MASK) == 0 and not _human_required and not _fresh_required and (not fault_bits & THERMAL or (may_resume_thermal and thermal budget ok))` → CHECK, `reenergize_requested` (`_RESUME_MASK = IDLE_TIMEOUT | THERMAL | STALE_CMD`); `resume_requested` otherwise → `resume_refused_inc`, no transition. Then the existing thermal-only auto row with the extra conjunct `not self._resume_on_request`.
   - (existing) SAFE_STOPPING progression.
   - (existing) RUN/HOLD/pre-run stop rows (fresh/human causes incl. BOX_PRESENT from `monitor_parked`, STOP_REQ) → SAFE_STOPPING; on entry from a parked HOLD set `_resume_on_request = True`.
   - (existing) CHECK + THERMAL → TORQUE_OFF.
   - RUN/HOLD + THERMAL → SAFE_STOPPING (existing); **if parked: `_resume_on_request = True`** (no auto re-energise while parked, P3a).
   - **new** HOLD(parked) + `cmd_fresh (RUN) and age ≤ stale and block_ok`: if `client_admitted` → `_enter_run` (resync, `parked = False`), else `client_refused_inc`, stay parked.
   - **new** HOLD(parked) + `now_ns ≥ _parked_deadline_ns` → SAFE_STOPPING, `fault_bits |= IDLE_TIMEOUT`, `_resume_on_request = True`.
   - **new** RUN / HOLD(unparked) + `park_requested`: `block_ok and park_in_tol` → HOLD, `_parked = True`, `_parked_deadline_ns = now_ns + park_timeout_ns`, `latch_target = True`; else `park_refused_inc`, no transition (deltas of that command ignored).
   - (existing) RUN/HOLD `consecutive_misses ≥ N` → SAFE_STOPPING (STALE_CMD): guarded `not self._parked`. RUN `cmd_age > stale` → HOLD: unchanged (RUN only). HOLD(unparked) + fresh → RUN: unchanged.
   - `mark_energized()`: when `session_mode` (constructor flag `land_parked: bool`, set by the core when `--session` **or** `cfg.idle` says so — default **True** for every build; the fake/mujoco backends land parked too, so parity tests observe the same machine) → state **HOLD**, `_parked = True`, deadline started, `fault_bits &= ~(_AUTO_CAUSES | IDLE_TIMEOUT)`, `_resync_pending = True` (delivered on the first RUN); the core's `_reset_command_counters` runs as today. `_enter_run()` clears `_parked` and `IDLE_TIMEOUT` with `_AUTO_CAUSES`. `resume_policy()` reports `ON_REQUEST` for `IDLE_TIMEOUT`.
5. **`servo/core.py`.** `_poll_cmd`: monotonic dedupe; reads `client_id`; sets `inp.park_requested = new and mode == PARK`, `inp.resume_requested = new and mode == RESUME_REQ`; a **new `client_id`** on a RUN or RESUME_REQ command triggers the authority callable **before** `sm.step` (see 6); `inp.park_in_tol = all(|present − home| ≤ park_tolerance_ticks)` (preallocated int32 scratch); `inp.now_ns = t0`. HOLD duty: on `latch_target` latch `_hold[...] = block.position`; parked ticks run `monitor_parked` = `pipeline.monitor_only(block, v)` **plus** the present-pose box check incl. the gripper floor points (`fk.tip_and_floor_z_from_ticks(present)` + `in_box_gripper(..., present tolerances)` → `BOX_PRESENT` trip) **plus** the clamped re-write `written = present + clip(hold − present, −E, E)` through `clamp.clamp_tracking` with the **fixed** `_hold` (never re-latched from present while parked), then `check_range` and `write_goal`. Counters `client_refused`, `park_refused`, `resume_refused` (u8 saturating) and `arm_ns` published in the state record and tick log; `terminal_reason: int` property = the cause mask when `state is TORQUE_OFF and fault_bits & _TERMINAL_MASK` or `state is LATCHED` (`_TERMINAL_MASK = USER_STOP | _FRESH_PROCESS_CAUSES | _HUMAN_CAUSES`; `IDLE_TIMEOUT`/`THERMAL` alone are not terminal). `arm_ns` = `clock.monotonic_ns()` at the tick ARM was accepted (0 before ARM / without an MCU). `first_mcu_status`/`last_mcu_status` captured from `hb.last_status` (byte) at the first `status()` and at exit.
6. **Authority hook (D1).** `ServoCore(..., authority: Callable[[int, int], AttachVerdict] | None = None)` (args `client_id, arm_ns`). On the first RUN or RESUME_REQ from a `client_id` not in the admitted set (a preallocated `np.int64[8]` ring of admitted ids + a parallel `denied_until_ns[8]`), while `state ∈ {HOLD(parked), TORQUE_OFF}`: the core sends this tick's heartbeat **first** (`_send_heartbeat()` right after a good read), then calls `authority(client_id, arm_ns)` with the helper's own `timeout_s = 0.12`; `ok` → admitted; refusal or timeout → `client_refused += 1`, `denied_until_ns = now + 1 s` (re-checks at most once per second while the client keeps writing). `inp.client_admitted` is the result. The verification tick is recorded with its duration in the tick log (`auth_ns`) and the summary (`auth_checks`, `auth_max_ms`); it is the one documented allocating tick besides re-energise. Without `--session` (`authority is None`) every client is admitted and the summary says `authority: none`.
7. **`servo/process.py` / `servo/main.py`.** New CLI: `--session <file>`, `--records <dir>`, `--state-dir <dir>`, `--allowed-signers <file>`, `--snapshot-sha <hex>`, `--exit-on-terminal`; `--arm-timeout-s` default stays `None`. With `--session`: the request is validated (FL6), `ServoAuthority.verify_session()` runs **before** the port is opened (failure → exit 2, `state.json.phase = "exit"`, no button asked for), `check_session_device_paths` runs for the `feetech` backend, `max_ticks = requested_cap_min × 60 × servo_hz`, `--tick-log`/`--summary` default to `<state-dir>/sessions/<request_id>/`, `state.json` is written at every phase change, `warm_up()` runs before BOOT. `ServoRunner.loop(max_ticks, *, exit_on_terminal)`: after each tick, `if exit_on_terminal and core.terminal_reason: break`. Exit codes: `0` clean (`--max-ticks`, `STOP_REQ`/USER_STOP, signal), `1` exception, `2` start-up/session refused, `3` RT, `4` locked, **`5` terminal trip** (fresh-process/human cause with `--exit-on-terminal`), **`6` LATCHED** (D2). Summary gains `pid, arm_ns, first_mcu_status, last_mcu_status, mcu_armed, exit_reason, client_refused, park_refused, resume_refused, parked_hold_entries, idle_rests, resume_cycles, auth_checks, auth_max_ms, session_request_id, snapshot_sha`.
8. **`servo/soak.py`.** `analyse_tick_log` gains `parked_hold_entries`, `idle_rests` (IDLE_TIMEOUT rising edges), `resume_cycles` (CHECK entries after an IDLE_TIMEOUT), `parked_stale_trips` (must be 0); `hold_entries` counts only **unparked** HOLD entries; the heartbeat-gap rule spans parked segments; `check_soak_report(strict=True)` additionally fails on `parked_stale_trips > 0` and on `reenergize_count > parked_hold_entries` (every energise followed by a parked entry). CLI `--park-every-min M --park-for-s S` makes the random-policy client drive to home, `PARK`, wait, `RESUME_REQ`, resume (for HS9).
9. **`env/backend.py`.** `ShmBackend(cfg, clock, *, client_id: int = 0)`; `_write_cmd(..., client_id)`; on attach the backend reads `last_cmd_seq` from the state record and continues from `last + 1`; new methods `send_park()`, `send_resume_req()`; the servo in session mode refuses `client_id == 0` (counted as `client_refused`).
10. **Tick log.** `TICK_LOG_DTYPE` appends `parked u1, client_refused u1, park_refused u1, resume_refused u1, auth_ns i8, arm_ns i8` at the record end (old logs become unreadable by the new dtype; the summary records `tick_log_itemsize`); `soak.py` and `sim/sysid/dataset.py` read fields by name.

### FL9. FW1 — firmware arm timeout (`firmware/watchdog_mcu/{main.py,PROTOCOL.md,test_vectors.json,README.md}`, `servo/heartbeat.py`, `tests/test_firmware_vectors.py`, `tests/test_heartbeat.py`)

- `main.py`: `ARM_TMO = 120000`; `NONE, TIMEOUT, UNDERVOLT, SEQ, ESTOP, TEST, RESET, WELD, ARMTMO = range(9)`; `self.dt` (DISARMED entry tick) set whenever the button hold completes (`self.state, self.rst, self.dt = DISARMED, None, now`); in `poll`, right after the button rows: `if self.state == DISARMED and self.d(now, self.dt) >= ARM_TMO: self.trip(ARMTMO)`; the `0x5A` row clears nothing new (ARM leaves DISARMED). Expected 94–95 lines; `test_firmware_line_budget` may rise to 100 with the reason recorded (D7).
- `PROTOCOL.md`: trip reason `8 ARM_TIMEOUT`; state row `DISARMED | ≥ 120 000 ms since entering DISARMED without an accepted ARM (wrap-safe ticks_diff) | LATCHED, reason 8`; a later 500 ms press re-enters DISARMED and restarts the window; example bytes `0x82` (LATCHED, reason 8) and `0x80` (DISARMED after the press, reason kept); "a press means: arm within 2 minutes".
- Vectors: rename `disarmed_never_times_out` → `disarmed_119999ms_still_disarmed` (steps: preamble; `dt 5000 → 60`; `dt 114999 → 60`); add `disarmed_120000ms_latches_reason_8` (`dt 120000 → 82`), `arm_at_119000ms_accepted` (`dt 119000, tx 5a → 05`), `press_after_arm_timeout_reenters_disarmed_and_restarts_window` (`… 82`, button 500 ms → `80`, `dt 119999 → 80`, `dt 1 → 82`), `wrap_safe_arm_timeout_30bit` (`start_ms` near 2**30, `wrap_bits: 30`). `test_vectors_cover_protocol_invariants`: the four new names required; the status-nibble bound becomes `≤ 8`.
- `servo/heartbeat.py`: `REASON_ARM_TIMEOUT = 8`, `ARM_TIMEOUT_MS = 120_000`, `REASON_NAMES = {0: "NONE", …, 8: "ARM_TIMEOUT"}`, `McuStatus.reason_name` property; `FakeMcu`: `_disarmed_since_ns` set in both DISARMED entries (`_poll` and `reset_button`), `_poll` latches `REASON_ARM_TIMEOUT` when `state == DISARMED and now − _disarmed_since_ns ≥ ARM_TIMEOUT_MS·_MS`; byte-identical to the firmware on every vector (`test_fakemcu_satisfies_every_vector`).
- `servo/main.py`: `--arm-timeout-s` default `None` (unchanged; `tests/test_servo_process.py::test_arm_timeout_cli_default_is_none`). The session unit passes `--arm-timeout-s 110` (A10).
- `docs/HARDWARE_CHECKLIST.md` H3 step 10 (FW1-HW): press without ARM → `0x80` for 120 s then `0x82`, relay stays open; press again → `0x80`; ARM within 2 min → `0x05`; artifact field `arm_timeout_s` in `runs/hw/deadman-<ts>.json`.

### FL10. PE1 — perception process (`src/so101_rl/perception/main.py`)

`python -m so101_rl.perception.main --camera {d435,fake} [--serial 819312071240] [--hand-eye calib/hand_eye.yaml] [--fiducial calib/fiducial.yaml] [--tracker src/so101_rl/perception/hsv_red_die.yaml] [--shm-name so101_objects] [--hz 30] [--fiducial-every-s 5] [--status PATH] [--max-frames N] [--fake-scene {die_visible,die_hidden,camera_moved}]`. Creates the object table (`ObjectTableShm.create`, owner), publishes the die **centroid** (`centroid_base`) with `visible`, `t_mono_ns`, and `fiducial_ok` (= the last `FiducialCheck.check` verdict, re-checked every `--fiducial-every-s`; **0 until the first check and whenever `--fiducial` is absent in `d435` mode**: die-task jobs then refuse to start); writes `status.json` (`{fps, frames, last_frame_t, fiducial_ok, fiducial_err_px, detections, camera: {kind, serial}}`) every second; exits `0` at `--max-frames`/SIGTERM, `2` camera not found, `1` exception; logs to stderr. `fake` renders the synthetic scene used by `tests/test_die_convention.py` (die visible / hidden / camera nudged 2 mm). pixi task `perception = "python -m so101_rl.perception.main"`. The transient unit `so101-perception-<request_id>.service` is started by the session manager (LA4) with `--camera d435`. Hand-off: when SUC lands `calib/camera.yaml`, `--camera d435` takes intrinsics/extrinsics from it (SUC's loader); until then `D435Camera` intrinsics from the device.

### FL11. Dashboard — `src/so101_rl/lab/dashboard.py` (GET only, loopback only)

Stdlib `http.server.ThreadingHTTPServer(("127.0.0.1", 6007), Handler)`; any other bind address is refused at start; every non-GET method → 405; routes: `/` (HTML, auto-refresh 5 s), `/api/status`, `/api/runs?limit=`, `/api/runs/<run_id>`, `/api/items`, `/api/session` (LA4), `/files/<run_id>/<rel>` (static, only `.mp4 .png .json .md .csv` under `runs/<run_id>/`, normalised path must stay inside, ≤ 50 MB), `/tb` → 302 to `http://127.0.0.1:6006`. Every string HTML-escaped; AI-written text rendered inside a visibly boxed "AI-written text (untrusted)" block. Session panel (LA4): unit state, snapshot sha, `arm_ns`, parked flag, cap/elapsed, arm minutes today, phase-1/phase-2 verdicts of the last session, `SHA256SUMS` check age, exact console commands for pending records (display only).

### FL12. Two-phase clean-session verdict (P5a) — `src/so101_rl/lab/session.py` (LA4), consumed by LA5

- **Phase 1 `servo_clean`** (at servo exit, from `summary.json`): `exit_code == 0`; `exit_reason ∈ {max_ticks, stop_req, idle_timeout_cap}`; `fault_names` contain no `_FRESH_PROCESS_CAUSES`/`_HUMAN_CAUSES` other than the closing `USER_STOP`; `thermal_trips_monitor ≤ 1` **including a parked thermal trip later resumed on request**; `tick_stats.p999_ms < 5` when `rt_active`; `hb_gaps_over_200ms == 0`; `client_refused == 0`; `parked_stale_trips == 0`.
- **Phase 2 `weld_ok`** (at the next session's first `status()`, from that servo's `first_mcu_status.json`): `reason ∈ {1 TIMEOUT, 6 MCU_RESET, 8 ARM_TIMEOUT}`, not `7 WELD`, and ARM accepted.
- `clean = servo_clean and weld_ok`; written to `runs/sessions.jsonl` (FL6) and mirrored by the servo in its own `sessions/` store; a cap raise (a new signed tier record with `seq + 1`) is accepted only when `clean_sessions_at_cap ≥ 1` at the current cap in **both** stores.

## Tasks

Legend: **SW** = software, runs now on the fake backend / CPU; **GPU** = needs the 3080 (`default` env); **HW** = hardware/human, Tier 0 attended, prerequisites named; **GRILL** = adversarial review step (3 attackers, 3 defenders, written to `plans/reviews/<date>-<unit>-review.md`, like `plans/reviews/2026-10-01-u3-review.md`; blocking + should-fix items fixed before the unit merges). Acceptance commands run from the repo root in the `cpu` env unless stated (`pixi run -e cpu <task>`); `Est.` = agent-hours (+ GPU-h / human-h). Every unit's last task is its ARCHITECTURE/README sync (a shared write, append-only, applied after the unit's other tasks).

---

### Unit LP0 — pixi `lab` feature, markers, pre-registered tasks (wave 0, sequential, **BUILD NOW**)

#### Task LP0.1 (SW): `lab` feature, `integration` marker, pixi tasks whose modules exist
- **Files:** `pyproject.toml`, `pixi.lock`, `README.md` (environment table + one line in Entry points), `tests/conftest.py`, `tests/test_env_sanity.py`
- **Specialist:** pixi-workspace-maintainer (worker)
- **Dependencies:** None
- **Steps:**
  1. `[tool.pixi.feature.lab.dependencies]`: `mcp = "2.3.*"`, `jsonschema = ">=4.26,<5"`, `nvidia-ml-py = ">=13,<14"` (conda-forge); `ffmpeg` is already pulled by the `train` feature (verify with `pixi list -e cpu ffmpeg`; if absent add `ffmpeg = "*"` to `train`). Environments `default = ["sim","perception","train","gpu","lab"]`, `cpu = ["sim","perception","train","cpu","lab"]`; `robot` unchanged.
  2. Markers: add `integration: needs a running systemd --user session` to `[tool.pytest.ini_options].markers` and `tests/conftest.py` (`pytest_configure`); `pytest_collection_modifyitems` skips `integration` when `systemctl --user is-system-running` is not `running`/`degraded` or `XDG_RUNTIME_DIR` is unset. `pixi run test` keeps `-m "not hardware and not gpu"` (integration tests run on this host, are skipped on CI without a user session).
  3. Tasks added **only** for modules that exist at the end of LP0 — i.e. none yet; instead extend the "planned tasks" comment with: `lab-mcp (lab.mcp_server), lab-supervisor (lab.supervisor.main), lab-dashboard (lab.dashboard), lab (lab.cli), test-lab (pytest tests/test_lab_*.py tests/test_authority*.py tests/test_records_tools.py), perception (perception.main)`. Each lands in its unit's sync task (`tests/test_pixi_tasks.py` guards this).
  4. `pixi install -e cpu && pixi install -e default && pixi install -e robot`; record solved `mcp`, `jsonschema`, `nvidia-ml-py` versions in the README table.
  5. `tests/test_env_sanity.py::test_lab_deps_importable` (`import mcp.server.mcpserver, jsonschema, pynvml` in the `cpu`/`default` envs; `pynvml` import must not need a driver).
- **Tests:** `test_env_sanity.py::test_lab_deps_importable`, `test_pixi_tasks.py` (unchanged, must stay green).
- **Acceptance:** `pixi install -e cpu && pixi install -e default && pixi install -e robot` → exit 0; `pixi run -e cpu python -c "from mcp.server.mcpserver import MCPServer; import jsonschema, pynvml; print('ok')"` → `ok`; `pixi run -e cpu test` → all pass (≥ the current count); `pixi run -e cpu pytest tests/test_env_sanity.py tests/test_pixi_tasks.py -q` → pass.
- **Est.:** 1 agent-h.

**Commit after LP0.1:** `LP0: lab pixi feature (mcp 2.3, jsonschema, nvidia-ml-py), integration marker`.

---

### Unit LA0 — lab core (wave 1, **BUILD NOW**; parallel with SV1, FW1, OS1-SW, PE1-SW)

Owns `src/so101_rl/lab/**` except `mcp_server.py`, `dashboard.py`, `session.py`, `tier1.py`, `trainers/{fastsac_sim,sb3_sac_her,tdmpc2_*}.py`; plus `src/so101_rl/viz/monitor.py` (hand-off from SUV, done: the `ScalarMonitor` extraction) and `tests/test_lab_{registry,ledger,trainers,fake_trainer,job,supervisor,budgets,deps,cli,monitor_scalar}.py`, `tests/test_monitor.py` (additions).

#### Task LA0.1 (SW): paths, policy, schemas, registry, WriteGuard, provenance, audit
- **Files:** `src/so101_rl/lab/__init__.py`, `lab/paths.py`, `lab/policy.yaml`, `lab/errors.py`, `lab/schemas/{run,status,event,usage,gate,item,audit,artifacts,request,request-session,attach,session}.v1.schema.json`, `lab/registry.py`, `lab/provenance.py`, `lab/audit.py`, `tests/test_lab_registry.py`, `tests/test_lab_deps.py`
- **Specialist:** python-best-practices; code-architecture-mentor (module boundaries)
- **Dependencies:** LP0
- **Steps:** FL2 + FL5. `paths.py`: `runs_root()` (env `SO101_LAB_RUNS`, default `<repo>/runs`), `load_policy()` (strict, `jsonschema` against `lab/schemas/policy.v1.schema.json`), `guarded_open`, `atomic_write_json`, `append_jsonl`. `errors.py`: `LabError(code, message, details, hint, retryable)` + the closed code set. `registry.py`: `new_run_id(trainer, task)`, `create_run(manifest) -> Path` (flock + `run.json` last), `read_run/read_status/write_status/append_event/next_event_seq`, `find_runs(filters, limit, cursor)`, `idempotency_claim(key, params_sha) -> Claim(replay|new|mismatch)`, `set_halt/clear_halt/is_halted`, `items` CRUD. `provenance.py`: git sha/dirty (`git` subprocess with `stdin=DEVNULL`; falls back to `unknown`), sha256s of `pixi.lock`, `so101.yaml`, `calib/sim_params.yaml`, the calibration json, `boot_id`, host. `audit.py`: `audit(actor, action, run_id, args) -> None`.
- **Tests:** `test_lab_registry.py::test_run_id_grammar_matches_authority_regex`, `::test_create_run_writes_run_json_last_and_is_immutable`, `::test_status_atomic_replace_survives_crash_between_tmp_and_replace` (simulate), `::test_events_seq_monotonic_and_append_only`, `::test_find_runs_keyset_cursor_and_filter_hash`, `::test_idempotency_replay_vs_mismatch`, `::test_halt_file_blocks_admission`, `::test_schemas_are_draft_2020_12_with_additional_properties_false` (every schema), `::test_policy_strict_unknown_key_refused`; `test_lab_deps.py::test_lab_never_imports_servo_main_core_feetech_or_safety_clamps` (AST: allowed `servo.shm`, `servo.heartbeat` (status decoding only), `safety.types`, `authority`), `::test_nothing_imports_lab` (AST over `src/so101_rl` minus `lab/`), `::test_lab_writes_only_through_guarded_open`, `::test_write_guard_refuses_protected_paths` (every protected glob + a path traversal).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_registry.py tests/test_lab_deps.py -q` → pass (≥ 14 tests).
- **Est.:** 4 agent-h.

#### Task LA0.2 (SW): trainer API, static table, `fake` trainer, `ScalarMonitor` extraction
- **Files:** `src/so101_rl/lab/trainers/__init__.py`, `lab/trainers/api.py`, `lab/trainers/fake.py`, `lab/schemas/trainer/fake.v1.schema.json`, `src/so101_rl/viz/monitor.py` (extract `ScalarMonitor`: metrics.csv writer + TensorBoard scalars + `add_text`; `TrainingMonitor.on_metrics/add_text` delegate to it; behaviour unchanged), `tests/test_lab_trainers.py`, `tests/test_lab_fake_trainer.py`, `tests/test_monitor.py` (+ `test_scalar_monitor_csv_and_tb`, `test_training_monitor_delegates_to_scalar_monitor`)
- **Specialist:** lerobot-specialist (trainer contract), testing-specialist
- **Dependencies:** LA0.1
- **Steps:** FL1 verbatim. `fake.py`: every behaviour string of FL1 (`crash_at`, `stall_at` stops heartbeating but keeps the process alive, `oom_at` raises `torch.cuda.OutOfMemoryError`-shaped `MemoryError` text without importing torch, `review_trigger_at`, `servo_trip:<cause>` / `late_attach` / `park_refused` / `resume_refused` reserved for LA4 and refused by `validate` until then with `CONFIG_INVALID`), `metrics` patterns, `checkpoint_every` writes `ckpt_<step>.pt` (a small JSON blob, no torch), `evaluate` writes `eval.json` with the configured success rate and a Wilson interval (reuse the formula; `rl.deploy_policy.wilson_interval` does not exist yet, so `lab/stats.py:wilson_interval` is the one implementation and SU5 may import it later), `export` writes `policy_export.pt` (JSON blob) + sha256, `artifacts` checks presence/sizes/sha. `ScalarMonitor` writes `metrics.csv` rows `env_step|update, wall_s, gpu_h_used, viz_disabled, <scalars>` with a fixed header from the first call.
- **Tests:** `test_lab_trainers.py::test_table_is_static_and_every_entry_runnable`, `::test_config_schema_strict_const_version`, `::test_validate_rejects_non_configurable_keys` (thermal thresholds etc.), `::test_plan_fields_complete`; `test_lab_fake_trainer.py::test_train_writes_metrics_ckpt_usage`, `::test_crash_at_raises_after_checkpoint`, `::test_stall_at_stops_heartbeats_process_alive`, `::test_review_trigger_written`, `::test_evaluate_and_export_artifacts_checked`, `::test_artifacts_failure_marks_failed` (a truncated `policy_export.pt`).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_trainers.py tests/test_lab_fake_trainer.py tests/test_monitor.py -q -m "not gpu"` → pass.
- **Est.:** 4 agent-h.

#### Task LA0.3 (SW): job entry (`lab/job.py`) and `RunContext`
- **Files:** `src/so101_rl/lab/job.py`, `lab/context.py`, `tests/test_lab_job.py`
- **Specialist:** python-best-practices
- **Dependencies:** LA0.2
- **Steps:** `python -m so101_rl.lab.job --run runs/<id> --segment <n> [--phase train|resume|evaluate|export|gates]`: loads `run.json`, the plugin, `policy.yaml`; writes `segments/<n>.json` with `os.getpid()`, `$INVOCATION_ID`/unit name (`systemctl --user` env) and `boot_id`; sets `torch.cuda.set_per_process_memory_fraction` only inside plugins that import torch (the job itself does not import torch); `RunContext.heartbeat` → `status.json` (`heartbeat_at`, `step`, `phase`) at most every 2 s and `usage.json` (progress-holding time) at most every 30 s; `should_stop` = control request `stop` present **or** `deadline_at` passed **or** `HALT` present; `should_pause` = control `pause` present; `checkpoint` records and copies `ckpt_last.pt`; exit codes: 0 finished/paused/stopped-cleanly, 1 crashed, 2 refused (bad manifest/plan), 3 artifacts failed; final `status.json.state` + `artifacts.json`; OOM is **not** caught (crashed, L12).
- **Tests:** `test_lab_job.py::test_train_segment_end_to_end_fake`, `::test_own_deadline_stops_without_supervisor` (deadline 2 s, no supervisor files), `::test_pause_request_checkpoints_and_exits_paused`, `::test_stop_request_sets_stop_reason`, `::test_crash_sets_crashed_and_segment_exit_code`, `::test_resume_from_ckpt_last_continues_step`, `::test_job_never_imports_torch_for_fake` (subprocess `sys.modules` check), `::test_usage_progress_time_capped_per_gap` (a 40-min heartbeat gap counts 30 min).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_job.py -q` → pass; `pixi run -e cpu python -m so101_rl.lab.job --run <tmp>/runs/fake-reach-...` on a hand-made manifest → exit 0, `metrics.csv` ≥ 3 rows, `usage.json`, `artifacts.json.ok == true`.
- **Est.:** 3 agent-h.

#### Task LA0.4 (SW): supervisor — queue, leases, launcher, stall ladder, health, suspend/reboot, TTLs, disk, review policies, pipelines, items, notifications
- **Files:** `src/so101_rl/lab/supervisor/__init__.py`, `lab/supervisor/main.py`, `lab/supervisor/queue.py`, `lab/supervisor/units.py`, `lab/supervisor/health.py`, `lab/supervisor/stall.py`, `lab/supervisor/clocks.py`, `lab/supervisor/items.py`, `lab/supervisor/notify.py`, `lab/supervisor/pipelines.py`, `lab/supervisor/review.py`, `lab/supervisor/disk.py`, `lab/authority.py` (the SIM admission mirror: config valid, `requires` satisfied, ledger, disk, not HALT, GPU health known), `lab/ledger.py`, `tests/test_lab_supervisor.py`, `tests/test_lab_budgets.py`, `tests/test_lab_ledger.py`
- **Specialist:** code-architecture-mentor; real-time-control-specialist is **not** needed (no RT)
- **Dependencies:** LA0.3
- **Steps:** FL4 verbatim; `main.py` = `python -m so101_rl.lab.supervisor.main [--runs DIR] [--once] [--launcher systemd|subprocess]` (`--once` runs one loop iteration: the deterministic test hook); `health.py` wraps `pynvml` with a `Sampler` protocol and a `FakeSampler` (tests inject 86 °C / 96 °C / empty); `clocks.py` exposes `suspend_jump_s()` and `boot_id()` with injectable readers; `ledger.py` derives from every `usage.json` + `ledger_imports.jsonl`, keeps the high-water in `ledger.json`, halts on shrink; `review.py` implements P7; `pipelines.py` creates the child runs; `items.py` opens/closes items and calls `notify.py` (`notify-send` if on PATH).
- **Tests (fake launcher unless marked):** `test_lab_supervisor.py::test_fifo_arm_first_and_leases`, `::test_queue_file_consumed_once_across_restart`, `::test_reads_queue_while_server_absent` (no MCP involved), `::test_stall_ladder_30_45_50` (fake time), `::test_kill_9_job_becomes_crashed`, `::test_readoption_by_unit_name_after_restart`, `::test_suspend_jump_marks_host_slept_and_voids_approvals`, `::test_reboot_boot_id_marks_crashed_and_voids_real_queue`, `::test_ttl_expiry_sim_24h_real_4h`, `::test_review_pause_resume_third_trigger_stops_probe_continues` (A2), `::test_pipeline_fake_train_eval_gate_export` (A1 CI half), `::test_thermal_pause_holds_gpu_lease_and_resumes` (A3), `::test_six_empty_nvml_samples_pause_all_and_open_item`, `::test_low_disk_blocks_admission_critical_stops_jobs`, `::test_heartbeat_json_written_every_tick`; `test_lab_budgets.py::test_fourth_attempt_refused`, `::test_probe_not_an_attempt_and_pooled_cap`, `::test_gpu_h_is_progress_time_with_stall_cap`, `::test_wall_budget_checkpoint_debit_and_budget_exhausted`; `test_lab_ledger.py::test_derived_from_usage_files`, `::test_import_run_cli_adds_entry`, `::test_shrink_halts_and_opens_item` (the gate shown to fail); **`@integration`** `::test_systemd_user_transient_unit_runs_fake_job` (real `systemd-run --user`, unit prefix `so101-test-job-`, temp runs dir, `--collect`), `::test_supervisor_restart_kills_no_job` (A4: start a 60 s fake job as a transient unit, `systemctl --user restart` the test supervisor unit, assert the job finished).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_supervisor.py tests/test_lab_budgets.py tests/test_lab_ledger.py -q` → pass (integration tests run on this host, skipped elsewhere); `systemctl --user list-units 'so101-test-*'` → empty afterwards (`--collect`).
- **Est.:** 9 agent-h.

#### Task LA0.5 (SW): CLI (`lab status|halt|unhalt|audit|import-run|install-units|session-request`) and user units
- **Files:** `src/so101_rl/lab/cli.py`, `lab/units/so101-lab-supervisor.service`, `lab/units/so101-lab-dashboard.service`, `tests/test_lab_cli.py`
- **Dependencies:** LA0.4
- **Steps:** `install-units` renders the templates with the repo path, writes `~/.config/systemd/user/`, runs `systemctl --user daemon-reload` and prints the `enable --now` commands (does not enable by itself: the user decides); `--dry-run` prints; `session-request` (debug, LA4 fills it) prints the request JSON it would write; `status` prints `get_status()` as a table; `import-run <dir> --budget-key K --gpu-h H --attempt` appends to `ledger_imports.jsonl`.
- **Tests:** `test_lab_cli.py::test_install_units_dry_run_renders_repo_path_and_kill_mode_process`, `::test_halt_unhalt_roundtrip_and_audit`, `::test_import_run_requires_flags`.
- **Acceptance:** `pixi run -e cpu lab install-units --dry-run` → prints both unit texts with `KillMode=process`; `pixi run -e cpu lab status` → table (empty runs dir OK).
- **Est.:** 2 agent-h.

#### Task LA0.6 (SW): LA0 sync (shared write)
- **Files:** `pyproject.toml` (tasks `lab-supervisor`, `lab`, `test-lab` — modules exist now), `docs/ARCHITECTURE.md` (§3: the supervisor service + transient job units; §4: `lab/` tree, dependency arrows `lab → rl, wm, env, sim, viz, logging, config, perception (launch only)`, `lab → servo.shm (attach)`, `lab → safety.types`, `lab → authority`, `nothing imports lab`; §5 FL1/FL2 pointers; §8 `lab` feature + tasks; §10 row), `README.md` (entry points)
- **Est.:** 0.5 agent-h.

**Commit after LA0.6:** `LA0: so101_rl.lab core — registry, ledger, trainers API + fake, supervisor on systemd --user transient units, job entry, CLI`.

---

### Unit LA1 — MCP server, dashboard, unattended fake run (wave 2, **BUILD NOW** after LA0)

#### Task LA1.1 (SW): `lab/api.py` facade + `lab/mcp_server.py` (12 tools, resources, prompts, errors, envelopes)
- **Files:** `src/so101_rl/lab/api.py`, `src/so101_rl/lab/mcp_server.py`, `src/so101_rl/lab/untrusted.py`, `.mcp.json`, `tests/test_lab_mcp.py`, `tests/test_lab_contract.py`
- **Specialist:** mcp-server-architect (consulted above)
- **Dependencies:** LA0
- **Steps:** FL3 verbatim. `api.py` is the synchronous Python API every front end uses (MCP, CLI, dashboard): one function per tool with the same parameters, raising `LabError`. `mcp_server.py`: thin async wrappers (`anyio.to_thread.run_sync`), `@tool_guard`, fd-1 hijack, stderr logging, `BrokenPipeError → os._exit(0)`, `instructions`. `untrusted.py`: `envelope(**fields)`, `clean(text, cap)`. Resources and the three prompts as FL3.
- **Tests:** `test_lab_mcp.py` (in-memory `mcp.client.Client(mcp)`): one test per tool on a fixture runs tree (`conftest` fixture `lab_tree` with 6 runs incl. finished/crashed/paused/queued), `::test_get_run_wait_change_returns_on_event` (a thread appends an event after 1 s), `::test_get_run_wait_times_out_flag`, `::test_start_run_refusals[each code]`, `::test_control_run_state_conflict`, `::test_write_note_and_propose_decision_paths_and_write_guard`, `::test_supervisor_down_semantics` (stale heartbeat: `start_run` queued + warning, `stop` → `SUPERVISOR_DOWN`), `::test_untrusted_envelope_on_every_ai_or_log_string`, `::test_resources_capped_32kb`, `::test_prompts_render`; `test_lab_contract.py::test_exactly_12_tools_with_exact_names`, `::test_tools_list_under_12288_bytes`, `::test_every_default_response_under_32kb`, `::test_no_heavy_modules_after_import` (subprocess), `::test_stdio_no_banner_and_clean_eof` (spawn `python -m so101_rl.lab.mcp_server`, send `initialize`, first stdout line is JSON-RPC id 1, close stdin → exit 0 within 5 s), `::test_launch_through_pixi_run_is_silent_on_stdout` (`pixi run --frozen -e cpu lab-mcp`), `::test_request_real_run_absent_without_env_flag`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_mcp.py tests/test_lab_contract.py -q` → pass (≥ 30 tests); `claude mcp list` (manual) shows `so101-lab` connected after `.mcp.json` lands.
- **Est.:** 7 agent-h.

#### Task LA1.2 (SW): dashboard
- **Files:** `src/so101_rl/lab/dashboard.py`, `src/so101_rl/lab/templates/index.html`, `tests/test_lab_dashboard.py`
- **Specialist:** webui-specialist (one small HTML page; no framework)
- **Dependencies:** LA1.1 (`lab/api.py`)
- **Steps:** FL11. Stdlib only; one HTML template with inline CSS; JSON endpoints; 5 s refresh; AI text boxed; links to TensorBoard (`pixi run dashboard`, port 6006) and `/files/<run>/videos/*.mp4`.
- **Tests:** `test_lab_dashboard.py::test_binds_loopback_only_refuses_other_hosts`, `::test_post_put_delete_405`, `::test_path_traversal_refused`, `::test_files_only_allowed_suffixes_and_inside_run`, `::test_index_renders_fixture_and_escapes_ai_text`, `::test_api_status_matches_api_module`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_dashboard.py -q` → pass; `pixi run -e cpu lab-dashboard --runs <fixture>` + `curl -s http://127.0.0.1:6007/api/status | python -m json.tool` → JSON; `curl -X POST http://127.0.0.1:6007/` → 405; screenshot saved for the LA1 review note (A7).
- **Est.:** 3 agent-h.

#### Task LA1.3 (SW, `@integration`): the unattended fake run with Claude Code closed (A4 + A1 CI half)
- **Files:** `tests/test_lab_unattended.py`, `scripts/lab_unattended_fake_run.sh`
- **Dependencies:** LA1.1, LA0.4
- **Steps:** The test starts a test supervisor as a transient user unit (`systemd-run --user --unit so101-test-supervisor-<pid> … lab-supervisor --runs <tmp>`), spawns the MCP server as a subprocess through `mcp.client.stdio.StdioServerParameters(command="pixi", args=["run","--frozen","-e","cpu","lab-mcp"], env={"SO101_LAB_RUNS": tmp})`, calls `start_run(fake, reach, {steps: 300, step_time_s: 0.2, …}, idempotency_key=…)`, then **closes the client and kills the server process** (`SIGKILL`), waits, and asserts via `lab.api.get_run` that the run reaches `finished` with every artifact, `events.jsonl` shows no effect of the client's death, `audit.jsonl` has the `start_run` entry, and the ledger is debited. A second case: `systemctl --user kill -s KILL so101-test-supervisor-…`, assert the job finished anyway; the unit restarts (`Restart=on-failure`) and re-adopts. The script is the manual equivalent for the user (prints the run id and `watch`-able commands).
- **Tests:** `test_lab_unattended.py::test_fake_run_outlives_mcp_client` (A4 "closing Claude Code changes nothing"), `::test_fake_run_outlives_supervisor_kill_9_and_is_readopted` (A4), `::test_pipeline_fake_end_to_end_unattended` (A1 CI half: `kind=pipeline` with evaluate + export).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_unattended.py -q` → pass on this host (skipped without a user session); `bash scripts/lab_unattended_fake_run.sh` → prints `finished` within 3 min with the artifact list.
- **Est.:** 3 agent-h.

#### Task LA1.4 (SW): LA1 sync (shared writes) — ARCHITECTURE v0.18, plan §12 v3, README, pyproject tasks
- **Files:** `docs/ARCHITECTURE.md` (header status "LA0/LA1 built"; §3 the account boundary (target), the three services, transient job/perception units, the servo system unit + snapshot (target); §4 `lab/mcp_server.py`, `lab/dashboard.py`, `WriteGuard` (guard rail); §5 FL3 pointer; §6 new item 7 "Connector authority" (who enforces what; guard rails labelled; duty fields documentation-only until enforced); §8 `.mcp.json`, tasks `lab-mcp`, `lab-dashboard`; §9/§10 row 0.18), `so101-self-learning-rl-plan.md` §12 (v3 text: two accounts, snapshot, servo-side verification, session lifecycle, SV1 landing-in-parked and exit-on-terminal, two-phase verdict, HS8/HS9 — **the master applies this edit**), `pyproject.toml` (`lab-mcp`, `lab-dashboard`), `README.md`
- **Est.:** 1 agent-h.

**Commit after LA1.4:** `LA1: 12-tool stdio MCP server, read-only dashboard, unattended fake runs under systemd --user; ARCHITECTURE v0.18`.

---

### Unit SV1 — servo park / idle / resume / exit (wave 1, **BUILD NOW**, `Work-Item: SV1`, **GRILL**)

Owns `src/so101_rl/servo/{shm,core,process,main,soak}.py`, `src/so101_rl/safety/{types,state_machine}.py`, `src/so101_rl/config/{schema,loader}.py`, `src/so101_rl/config/so101.yaml` (`idle:` section only), `src/so101_rl/env/backend.py` (`client_id` + `send_park/send_resume_req` only), `tests/test_{shm,state_machine,servo_core,servo_process,soak,config,env_backend}.py` (additions), `tests/test_sv1_*.py`. Does **not** touch `servo/heartbeat.py` (FW1), `servo/probe.py` (SUC), `safety/pipeline.py` (unchanged: `monitor_parked` is composed in the core from existing `pipeline.monitor_only` + `So101FK.in_box_gripper` + `clamp.clamp_tracking`).

#### Task SV1.1 (SW): shm layout v3, `CmdMode`, types, config `idle:`, device-path check
- **Files:** `src/so101_rl/servo/shm.py`, `src/so101_rl/safety/types.py`, `src/so101_rl/config/schema.py`, `src/so101_rl/config/loader.py`, `src/so101_rl/config/so101.yaml`, `src/so101_rl/env/backend.py`, `tests/test_shm.py`, `tests/test_types.py`, `tests/test_config.py`, `tests/test_config_single_source.py`, `tests/test_env_backend.py`
- **Specialist:** real-time-control-specialist (consulted)
- **Dependencies:** LP0
- **Steps:** FL8 items 1–3, 9. Keep every existing offset up to `read_ok`; new asserts (`STATE_SIZE == 248`, `crc32` at 240, stride 256). `LocalIpc` unchanged in shape. `ShmBackend(client_id=)`, seq continuation from `last_cmd_seq`, `send_park`, `send_resume_req`. `IdleConfig` strict; `check_session_device_paths`.
- **Tests:** `test_shm.py::test_layout_v3_sizes_offsets_and_attach_refuses_v2`, `::test_monotonic_dedupe_older_or_equal_seq_ignored` (writer-side helper `is_new(seq, last)`), `::test_client_id_roundtrip`; `test_types.py::test_idle_timeout_on_request_and_policy_rank`; `test_config.py::test_idle_section_strict_and_ranges`, `::test_check_session_device_paths_refuses_non_by_id_unresolvable_wrong_owner` (tmp symlinks + `os.stat` monkeypatch for the owner), `::test_check_session_device_paths_accepts_by_id_owned` (positive control); `test_env_backend.py::test_shm_backend_continues_seq_from_state_record_and_sets_client_id`, `::test_send_park_and_resume_req_modes`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_shm.py tests/test_types.py tests/test_config.py tests/test_config_single_source.py tests/test_env_backend.py -q` → pass; the whole suite still green (layout bump: `pixi run -e cpu test`).
- **Est.:** 3 agent-h.

#### Task SV1.2 (SW): state-machine rows (park, idle timeout, resume on request, landing parked, admitted clients)
- **File:** `src/so101_rl/safety/state_machine.py`
- **Files (tests):** `tests/test_state_machine.py` (additions), `tests/test_sv1_state_machine.py`
- **Specialist:** real-time-control-specialist (row order as consulted)
- **Dependencies:** SV1.1
- **Steps:** FL8 item 4 in exactly that order; update the docstring transition table; `land_parked=True` default; `park_timeout_ns` from `cfg.idle`; no per-tick allocation (ints/bools only).
- **Tests (`FakeClock`):** `test_sv1_state_machine.py::test_park_accepted_within_tolerance_latches_target`, `::test_park_refused_outside_tolerance_or_bad_block_counts` (twin: `::test_park_accepted_at_tolerance_boundary_20`), `::test_parked_hold_never_trips_stale_cmd_10000_ticks` (twin: `::test_unparked_hold_trips_stale_after_5_misses`), `::test_parked_fresh_run_admitted_enters_run_with_resync`, `::test_parked_fresh_run_unadmitted_counts_client_refused_stays_parked`, `::test_idle_timeout_600s_to_safe_stopping_then_torque_off_on_request` (twin: `::test_idle_timeout_not_reached_at_599s`), `::test_resume_req_to_check_once_then_reenergize_requested`, `::test_resume_req_refused[NAN|USER_STOP|LATCHED|fresh_required|human_required|unadmitted|thermal_dwell|thermal_budget|bad_block]` (positive control `::test_resume_req_accepted_after_idle_timeout`), `::test_thermal_while_parked_sets_resume_on_request_no_auto_reenergise_60s` (twin: `::test_thermal_unparked_auto_reenergises_after_dwell`), `::test_mark_energized_lands_parked_hold_not_run_and_30s_without_client_stays` (900 steps; twin: `::test_land_parked_false_lands_run` — the fixture flag exists only for this twin), `::test_idle_timeout_cleared_on_mark_energized_and_enter_run`, `::test_stop_req_while_parked_goes_safe_stopping`, `::test_box_present_while_parked_goes_safe_stopping_fresh_process`, `::test_a_live_client_wins_a_tie_with_the_idle_deadline`, `::test_parked_stale_cmd_bit_never_raised`; existing 37 tests unchanged except the three that assert `mark_energized → RUN` (now `→ HOLD(parked)`, then RUN on the first fresh command — each updated with a comment citing P3a).
- **Acceptance:** `pixi run -e cpu pytest tests/test_state_machine.py tests/test_sv1_state_machine.py -q` → pass (≥ 60 tests).
- **Est.:** 5 agent-h.

#### Task SV1.3 (SW): core — `monitor_parked`, latched target, counters, `arm_ns`, authority hook, `terminal_reason`, tick log
- **Files:** `src/so101_rl/servo/core.py`, `tests/test_servo_core.py` (additions), `tests/test_sv1_core.py`
- **Specialist:** real-time-control-specialist
- **Dependencies:** SV1.2
- **Steps:** FL8 items 5, 6, 10. The HOLD duty branches on `sm.parked`; the latched target is set only on `out.latch_target` (PARK accept) and on landing parked (target = present after the ramp); the parked re-write clamps the fixed `_hold` against present (never re-latched); `monitor_parked` adds the present-pose gripper-floor check; `_auth_check(client_id)` with heartbeat-first and the 120 ms timeout; admitted-id ring; `terminal_reason`; `first_mcu_status` at `_wait_for_arm`'s first `status()`, `last_mcu_status` at shutdown; tick-log fields; tracemalloc filters unchanged.
- **Tests (FakeBus + MujocoBus parametrised where motion matters):** `test_sv1_core.py::test_park_from_home_rest_reached_torque_off_at_most_5_ticks_motion_then_resume_req_full_reenergize_parked_then_fresh_run` (the design's core sequence), `::test_every_reenergise_lands_parked_and_no_client_30s_never_leaves` (A6 late attach), `::test_parked_target_does_not_ratchet_with_sag` (FakeBus present offset −30 ticks on `shoulder_lift`: written target stays within E of latched over 10 000 ticks; twin `::test_relatch_from_present_would_ratchet` implemented against a deliberately wrong helper), `::test_monitor_parked_trips_box_present_on_sagging_gripper_point` (present pose with the lowest gripper point 3 mm below the floor → BOX_PRESENT → SAFE_STOPPING; twin `::test_parked_sag_1mm_inside_tolerance_no_trip`), `::test_heartbeat_every_tick_parked_and_idle_torque_off`, `::test_new_client_id_runs_authority_once_heartbeat_first_duration_recorded` (fake authority callable sleeping 60 ms of FakeClock time: `max(diff(hb_ts)) ≤ 200 ms`, `auth_ns` logged; twin `::test_sync_authority_400ms_breaks_the_gap_bound` asserting the failure is detected by the soak rule), `::test_denied_client_backoff_1s_and_counted`, `::test_client_id_zero_in_session_mode_refused`, `::test_terminal_reason_set_for_fresh_human_latched_not_for_idle_thermal`, `::test_zero_net_allocation_over_1000_parked_ticks` (twin `::test_alloc_detector_catches_per_tick_dict`), `::test_first_and_last_mcu_status_and_arm_ns_in_summary_fields`, `::test_tick_log_new_fields_written`; `test_servo_core.py::test_zero_net_allocation_over_1000_ticks_after_warmup` unchanged and green.
- **Acceptance:** `pixi run -e cpu pytest tests/test_servo_core.py tests/test_sv1_core.py tests/test_reenergize_backends.py tests/test_action_units.py tests/test_batched_parity.py -q -m "not gpu"` → pass (parity Layer A unchanged: `sim/batched_servo.py` untouched).
- **Est.:** 7 agent-h.

#### Task SV1.4 (SW): process/CLI — `--session/--records/--state-dir/--allowed-signers/--snapshot-sha/--exit-on-terminal`, exit codes 5/6, state.json, summary fields
- **Files:** `src/so101_rl/servo/process.py`, `src/so101_rl/servo/main.py`, `tests/test_servo_process.py` (additions), `tests/test_sv1_process.py`
- **Dependencies:** SV1.3; codes against FL7 with `from so101_rl.authority import ServoAuthority` imported **lazily inside the `--session` branch** (OS1-SW lands it in the same wave; the tests below monkeypatch `sys.modules["so101_rl.authority"]` with a fake until then, and the integration wave re-runs them against the real module)
- **Steps:** FL8 item 7. `run(..., exit_on_terminal: bool = False, authority=None, state_dir=None, session=None)`; `ServoRunner.loop(max_ticks, *, exit_on_terminal)`; `EXIT_TERMINAL = 5`, `EXIT_LATCHED = 6`; `_core_summary` additions; `state.json` phase writes (`boot_wait` before `_wait_for_arm`, `armed`, `parked`, `running` on the first RUN, `torque_off`, `exit`); `--session` implies the request validation + `verify_session()` before `bus.connect()`, `max_ticks` from the cap, default log/summary paths, `warm_up()`.
- **Tests:** `test_sv1_process.py::test_exit_on_terminal_max_ticks_pid_gone_fakemcu_latched_exit_0` (subprocess, fake backend, `--summary`; FakeMcu is in-process in the fake backend, so the assertion is on the summary's `last_mcu_status`... **no**: the test runs `servo.main` in-process with `run(...)` and a shared `FakeMcu` so the test can advance it 1.1 s after exit and assert `latched` with reason 1; the subprocess twin asserts the pid is gone and `exit_code` in the summary), `::test_exit_on_terminal_stop_req_exit_0`, `::test_exit_on_terminal_injected_nan_exit_5` (twin: `::test_idle_timeout_torque_off_does_not_exit_pid_alive`), `::test_exit_on_terminal_latched_exit_6`, `::test_without_flag_trips_keep_ticking_as_today` (regression guard), `::test_single_sigterm_clean_stop_second_escalates` (existing item-7 tests stay), `::test_session_request_refused_before_port_open_exit_2_state_json_exit` (fake authority raising `RecordError`; asserts `bus.connect` was never called), `::test_session_mode_sets_max_ticks_from_cap_and_default_paths`, `::test_arm_timeout_cli_default_is_none`, `::test_state_json_phases_in_order`, `::test_summary_has_sv1_fields`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_servo_process.py tests/test_sv1_process.py -q` → pass; `pixi run -e cpu servo-fake --max-ticks 300 --exit-on-terminal --summary /tmp/s.json` → exit 0, summary `exit_reason: "max_ticks"`, `parked_hold_entries ≥ 1`.
- **Est.:** 5 agent-h.

#### Task SV1.5 (SW): soak extension (parked metrics, invariants, `--park-every-min`)
- **Files:** `src/so101_rl/servo/soak.py`, `tests/test_soak.py` (additions)
- **Dependencies:** SV1.4
- **Steps:** FL8 item 8; the random-policy client drives toward `home` with ≤ `max_delta_ticks` steps, sends `PARK` when within tolerance, waits `--park-for-s`, sends `RESUME_REQ` (admitted: no `--session`), resumes random motion; report fields `parked_hold_entries, idle_rests, resume_cycles, parked_stale_trips`.
- **Tests:** `test_soak.py::test_item_sv1_hold_entries_count_only_unparked`, `::test_sv1_parked_metrics_counted_from_tick_log`, `::test_sv1_gate_fails_on_parked_stale_trip` (shown to fail), `::test_sv1_gate_fails_when_an_energise_is_not_followed_by_parked_entry`, `::test_sv1_heartbeat_gap_rule_spans_parked_segments` (a parked segment with one 300 ms gap fails — the design's twin), `@slow ::test_sv1_fifteen_minute_fake_soak_two_park_resume_cycles_gaps_le_200ms` (fake clock accelerated where the harness allows; else 2 × 1-min cycles at real time with the bound asserted).
- **Acceptance:** `pixi run -e cpu pytest tests/test_soak.py -q -m "not slow"` → pass; `pixi run -e cpu soak-fake --minutes 2 --park-every-min 0.5 --park-for-s 10` → report with `resume_cycles ≥ 2`, gate advisory PASS on a non-RT host.
- **Est.:** 3 agent-h.

#### Task SV1.6 (GRILL): adversarial review of SV1
- **Files:** `plans/reviews/<date>-sv1-review.md`
- **Steps:** Spawn `code-review-orchestrator` on the SV1 diff with the attacker brief: "find any path where a parked arm moves without a fresh admitted RUN, any way the stale-command fuse can trip with no client, any ratchet of the parked target, any exit path that leaves heartbeats running after a terminal trip, any allocation in the parked tick, any change to the existing safe-state/thermal/heartbeat obligations" plus `safety/pipeline.py` untouched, parity tests green, `test_zero_net_allocation_*` green. Defenders answer; blocking + should-fix fixed; regression tests per fixed item in `tests/test_sv1_review.py`.
- **Acceptance:** review file with `Verdict: APPROVE` after fixes; `pixi run -e cpu ci` green; `scripts/check_protected_paths.py` (OS1-SW) reports every SV1 commit carries `Work-Item: SV1`.
- **Est.:** 4 agent-h.

#### Task SV1.7 (SW): SV1 sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§3 CLI flags, exit codes 5/6, state.json; §5 `CmdMode.PARK/RESUME_REQ`, `client_id`, layout v3, state record v3 fields, summary fields, tick-log fields, authority hook; §6 layer 5 rows IDLE_TIMEOUT/ON_REQUEST, parked rules, landing in parked HOLD, exit-on-terminal; §10 row), `README.md` (`servo-fake --exit-on-terminal`, `soak-fake --park-every-min`), `docs/HARDWARE_CHECKLIST.md` (append HS8 procedure: 60-min attended parked session with three park → idle → resume cycles **and** the thermal hold: per-joint temperature slope over a 10-min hold at `home` after a 20-min run, pass max < 50 °C; artifact `runs/hw/park-resume-<ts>.json`; shown-to-fail: a PARK sent 30 ticks away from home must be refused)
- **Est.:** 0.5 agent-h.

**Commit after SV1.7:** `SV1: servo park/idle/resume, landing in parked HOLD, exit-on-terminal, client_id + monotonic dedupe, soak parked metrics` with trailer `Work-Item: SV1`.

---

### Unit FW1 — firmware arm timeout (wave 1, **BUILD NOW**, `Work-Item: FW1`, **GRILL**)

Owns `firmware/watchdog_mcu/{main.py,PROTOCOL.md,test_vectors.json,README.md}`, `src/so101_rl/servo/heartbeat.py`, `tests/test_firmware_vectors.py`, `tests/test_heartbeat.py`.

#### Task FW1.1 (SW): firmware `Core`, PROTOCOL.md, vectors
- **Files:** `firmware/watchdog_mcu/main.py`, `firmware/watchdog_mcu/PROTOCOL.md`, `firmware/watchdog_mcu/test_vectors.json`, `firmware/watchdog_mcu/README.md`, `tests/test_firmware_vectors.py`
- **Specialist:** jetank-hardware-specialist is **not** right; use `hardware-interface-specialist` (protocol semantics) briefly
- **Dependencies:** None
- **Steps:** FL9 bullets 1–3. Keep `main()` untouched. If the line count exceeds 95, raise the budget test to 100 with a docstring reason (D7).
- **Tests:** every existing vector + the four new ones on `Core`; `test_vectors_cover_protocol_invariants` extended; `test_firmware_line_budget`; `test_firmware_is_lint_clean`; `test_main_with_stubbed_machine` unchanged (the stubbed run arms within 1.2 s).
- **Acceptance:** `pixi run -e cpu pytest tests/test_firmware_vectors.py -q -k "firmware_core or invariants or budget or lint or main"` → pass; `grep -c ARM_TIMEOUT firmware/watchdog_mcu/PROTOCOL.md` ≥ 2.
- **Est.:** 1.5 agent-h.

#### Task FW1.2 (SW): host decoding + FakeMcu
- **Files:** `src/so101_rl/servo/heartbeat.py`, `tests/test_heartbeat.py`, `tests/test_firmware_vectors.py` (`test_fakemcu_satisfies_every_vector` now covers the new vectors)
- **Dependencies:** FW1.1
- **Steps:** FL9 bullet 4; `SimulatedOperatorLink` unchanged (ARM follows the press within a tick).
- **Tests:** `test_heartbeat.py::test_fakemcu_disarmed_latches_reason_8_at_120s_and_press_restarts_window`, `::test_reason_name_mapping_includes_8`, `::test_fakemcu_arm_at_119s_accepted`; `test_fakemcu_satisfies_every_vector[*]` incl. the new ones.
- **Acceptance:** `pixi run -e cpu pytest tests/test_firmware_vectors.py tests/test_heartbeat.py tests/test_servo_process.py -q` → pass (the fake backend start-up still arms).
- **Est.:** 1 agent-h.

#### Task FW1.3 (GRILL): adversarial review of FW1
- **Files:** `plans/reviews/<date>-fw1-review.md`
- **Steps:** attacker brief: "can a press ever leave the relay closed or re-close it; can the 120 s window be extended by serial traffic; is `ticks_diff` wrap handled for the new timer; does any existing vector change meaning; is the reason nibble ≤ 15 and decoded everywhere (`McuStatus`, summary, dashboard)". Fix + regression vectors.
- **Acceptance:** review `APPROVE`; `pixi run -e cpu ci` green.
- **Est.:** 1 agent-h.

#### Task FW1.4 (SW): FW1 sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§6 layer 2: reason 8 row, the two-timer rule (servo ≤ 110 s from its start; MCU 120 s from DISARMED entry); §10 row), `docs/HARDWARE_CHECKLIST.md` (H3 step 10 + artifact field, FL9 last bullet)
- **Est.:** 0.5 agent-h.

**Commit after FW1.4:** `FW1: watchdog DISARMED → LATCHED after 120 s without ARM (reason 8 ARM_TIMEOUT); FakeMcu + vectors + PROTOCOL` with trailer `Work-Item: FW1`.

---

### Unit OS1-SW — authority module, records tools, deploy files, protected-path guard (wave 1, **BUILD NOW**, `Work-Item: OS1`; **GRILL recommended**, D3)

Owns `src/so101_rl/authority/__init__.py`, `tools/so101_records/{approve,gate_sign,ack}.py`, `deploy/**`, `scripts/check_protected_paths.py`, `docs/protected_work_items.yaml`, `tests/test_authority.py`, `tests/test_records_tools.py`, `tests/test_deploy_files.py`, `tests/test_protected_paths.py`. Everything here runs **without sudo**: tests use temp roots, throwaway Ed25519 keys and a temp `allowed_signers`.

#### Task OS1.1 (SW): `so101_rl.authority` — canonical records, grammars, verifier, nonce + arm-minute ledgers, Tier 1 rules, servo facade
- **File:** `src/so101_rl/authority/__init__.py`
- **Files (tests):** `tests/test_authority.py`
- **Specialist:** none in the harness for signed-record security (gap logged); consult `mcp-server-architect`'s security notes and the design §5.3 verbatim
- **Dependencies:** LP0 (markers only)
- **Steps:** FL7 verbatim; stdlib only (`json, hashlib, os, re, subprocess, time, datetime, zoneinfo, pathlib, dataclasses`); `ssh-keygen` by absolute path (`/usr/bin/ssh-keygen`, overridable for tests only via the constructor); every subprocess with `stdin=DEVNULL`, `capture_output=True`, `timeout`, `env={}`; `read_exact` with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC` + `fstat`; verdict logging via a caller-provided append function.
- **Tests (real keys):** `test_authority.py::test_canonical_bytes_sorted_ascii_newline`, `::test_run_id_item_id_sha_nonce_grammars`, `::test_validate_request_rejects_unicode_control_oversize_badpaths_and_accepts_valid`, `::test_sign_then_verify_roundtrip_each_type`, `::test_wrong_namespace_rejected`, `::test_type_namespace_mismatch_rejected`, `::test_wrong_key_rejected` (second throwaway key), `::test_tampered_byte_rejected`, `::test_current_pointer_required_for_tier_and_gate`, `::test_revoked_sha_rejected`, `::test_seq_rollback_rejected_highest_seen_persisted`, `::test_approval_expiry_sanity[expired|skew_future|too_long_window]`, `::test_approval_hash_mismatch[run_json|config|policy|hw_artifact]`, `::test_approval_session_binding`, `::test_nonce_consumed_once_then_replay_refused`, `::test_arm_minute_ledger_local_midnight_boundary`, `::test_tier1_rules[quiet_hours|min_gap|per_day_cap|allow_list_task|allow_list_policy_sha|cap_exceeds_record|tier_claim_above_record|valid_until]` (each with a positive control), `::test_servo_authority_verify_session_and_attach_consumes_nonce_and_logs`, `::test_verify_timeout_is_denied_not_hang` (a fake `ssh-keygen` that sleeps 1 s under `timeout_s=0.12`), `::test_module_is_stdlib_only_and_imports_nothing_from_so101_rl` (AST).
- **Acceptance:** `pixi run -e cpu pytest tests/test_authority.py -q` → pass (≥ 30 tests); `python3 -I -c "import sys; sys.path.insert(0,'src/so101_rl/authority'); import __init__"` is **not** the test — instead `tests/test_records_tools.py` (OS1.2) imports the installed copy by file path.
- **Est.:** 8 agent-h.

#### Task OS1.2 (SW): records tools `so101-approve`, `so101-gate-sign`, `so101-ack` + shims
- **Files:** `tools/so101_records/approve.py`, `tools/so101_records/gate_sign.py`, `tools/so101_records/ack.py`, `tools/so101_records/_common.py` (TTY/uid/env checks, rendering, diff, challenge prompt; imports `so101_authority`), `deploy/bin/so101-approve`, `deploy/bin/so101-gate-sign`, `deploy/bin/so101-ack` (shims: `exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=<approver home> TERM="$TERM" /usr/bin/python3 -I -E -s /usr/local/lib/so101_rl/<tool>.py "$@"`), `tests/test_records_tools.py`
- **Specialist:** none (security review in OS1.5)
- **Dependencies:** OS1.1
- **Steps:** design §5.3 verbatim: console/uid/env checks (`os.ttyname(0)` must match `^/dev/tty[1-9][0-9]*$`; refuse `/dev/pts/*`; refuse when `SUDO_UID|SUDO_USER|SUDO_COMMAND` set; refuse when `os.getuid() != pwd.getpwnam("so101-approver").pw_uid`); `read_exact` of the request; strict validation; render machine fields first, then a highlighted diff vs the last approved request of the same kind, `ai_notes` boxed only with `--show-ai-text`; `CHAL` shown once directly above the prompt; exact phrase; body; `ssh-keygen -Y sign`; `O_WRONLY|O_CREAT|O_EXCL` write of `.json` + `.sig`; update `seq/<type>` and (tier/gate) `current/*`; `so101-ack` requires the item file and records the servo summary sha; `so101-gate-sign --tier <tier.json>` signs the tier body after rendering its rules. **Test-only flags** `--root <dir>`, `--allow-pty`, `--uid <n>`, `--key <path>`, `--allowed-signers <path>`; `tests/test_deploy_files.py::test_shims_pass_no_test_flags` pins that the shims never pass them. The key path in production comes from `pwd.getpwuid(os.getuid()).pw_dir + "/.ssh/so101_approver_ed25519"`.
- **Tests:** `test_records_tools.py` runs the tools as subprocesses from a temp install dir (`python3 -I -E -s <tmp>/approve.py --root <tmp> --allow-pty --uid <os.getuid()> --key <tmpkey> --allowed-signers <tmp>/allowed_signers <run_id>`, stdin = the phrase): `::test_approve_writes_verifying_record_with_challenge` (then `Verifier.verify_file` OK), `::test_wrong_challenge_word_refused_no_file`, `::test_pasted_phrase_without_chal_refused`, `::test_pty_refused_without_allow_pty`, `::test_sudo_env_refused`, `::test_wrong_uid_refused`, `::test_request_with_unicode_or_control_bytes_refused_never_sanitised`, `::test_oversize_request_refused`, `::test_symlinked_request_refused` (O_NOFOLLOW), `::test_second_record_same_name_o_excl_refused`, `::test_seq_increments_and_supersedes_set_for_tier_and_gate`, `::test_current_pointer_updated_and_revocation_written`, `::test_ai_text_hidden_by_default_shown_with_flag`, `::test_diff_against_last_approved_rendered_above_prompt`, `::test_gate_sign_and_ack_phrases`, `::test_exact_bytes_signed` (the signed bytes equal the file bytes), `::test_env_i_scrub_in_shim_text` (static check of the shim files).
- **Acceptance:** `pixi run -e cpu pytest tests/test_records_tools.py -q` → pass (≥ 17 tests; where CI cannot change uid the uid test runs with the override and asserts the refusal path by monkeypatching `os.getuid`).
- **Est.:** 7 agent-h.

#### Task OS1.3 (SW): deploy files — sysusers, tmpfiles, udev, system unit, polkit, allowed_signers template, install/update/verify scripts (dry-run testable)
- **Files:** `deploy/sysusers.d/so101.conf` (`g so101 -`, `u so101-servo - "SO-101 servo" /var/lib/so101_rl/servo /usr/sbin/nologin`, `u so101-approver - "SO-101 approver" /home/so101-approver /bin/bash`, `m koen so101`, `m so101-servo so101`, `m so101-approver so101`), `deploy/tmpfiles.d/so101.conf` (`d /var/lib/so101_rl 0755 root root`, `d /var/lib/so101_rl/records 0755 so101-approver so101-approver`, `d /var/lib/so101_rl/records/{approvals,gates,tiers,acks,current,seq} 0755 so101-approver so101-approver`, `d /var/lib/so101_rl/servo 0750 so101-servo so101`, `d /var/lib/so101_rl/attach 2775 koen so101`, `d /var/lib/so101_rl/attach/sessions 2775 koen so101`), `deploy/udev/99-so101-serial.rules` (`SUBSYSTEM=="tty", ATTRS{serial}=="<ARM_SERIAL>", SYMLINK+="serial/by-id/so101-arm", OWNER="so101-servo", MODE="0600"`; likewise `so101-mcu`; the serials are filled by the installer from `udevadm info`), `deploy/systemd/so101-servo@.service` (FL8 argv pinned: `ExecStart=/opt/so101_rl/current/env/bin/python -m so101_rl.servo.main --backend feetech --port /dev/serial/by-id/so101-arm --heartbeat-port /dev/serial/by-id/so101-mcu --session /var/lib/so101_rl/attach/sessions/%i.json --records /var/lib/so101_rl/records --allowed-signers /etc/so101_rl/allowed_signers --state-dir /var/lib/so101_rl/servo --snapshot-sha ${SO101_SNAPSHOT_SHA} --require-rt --arm-timeout-s 110 --exit-on-terminal`, `EnvironmentFile=/etc/so101_rl/snapshot.env`, `WorkingDirectory=/opt/so101_rl/current`, `User=so101-servo`, `Group=so101`, `SupplementaryGroups=so101`, `Type=simple`, `Restart=no`, `LimitRTPRIO=50`, `LimitMEMLOCK=infinity`, `CPUAffinity=15` (the last CPU; `SO101_SERVO_CPU` matches), `KillMode=mixed`, `KillSignal=SIGTERM`, `SendSIGHUP=no`, `TimeoutStopSec=30` (= `bound_s` 20 + 10), `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ReadWritePaths=/var/lib/so101_rl/servo /dev/shm`, `DeviceAllow=/dev/serial/by-id/so101-arm rw` + `so101-mcu rw`), `deploy/polkit/50-so101-servo.rules` (JavaScript: `action.id == "org.freedesktop.systemd1.manage-units" && action.lookup("unit").match(/^so101-servo@[0-9A-Za-z-]+\.service$/) && ["start","stop","status"].includes(action.lookup("verb")) && subject.user == "koen" → YES`), `deploy/allowed_signers.template` (`so101-approver namespaces="so101-approval,so101-gate,so101-tier,so101-ack" ssh-ed25519 <PUBKEY>`), `deploy/install.sh` (refuses to run as non-root; `--dry-run --root <dir>` prints/executes every step against a fake root; steps: sysusers, tmpfiles, udev rules with serials, remove koen from `dialout` (`gpasswd -d`), copy `authority/__init__.py` → `/usr/local/lib/so101_rl/so101_authority.py` + the three tools + shims (0755 root), install the unit + polkit rule, write `/etc/so101_rl/allowed_signers` from the approver's pubkey (read from `/home/so101-approver/.ssh/so101_approver_ed25519.pub`, which the human generated by hand), build the snapshot (`deploy/build_snapshot.sh <sha>`), write `/etc/so101_rl/snapshot.env` (`SO101_SNAPSHOT_SHA=<sha>`), write `/etc/so101_rl/SHA256SUMS` over every root-owned file and the snapshot tree, `chgrp -R so101 runs && chmod -R g+rX runs`), `deploy/build_snapshot.sh` (`git archive <sha> pyproject.toml pixi.lock src/so101_rl/__init__.py src/so101_rl/{servo,safety,config,logging,authority} | tar -x -C /opt/so101_rl/<sha>`; `pixi install --locked -e robot --manifest-path /opt/so101_rl/<sha>/pyproject.toml` as root with `PIXI_CACHE_DIR=/var/cache/so101_rl/pixi`; `ln -s .pixi/envs/robot env`; `chown -R root:root`, `chmod -R a+rX,go-w`; D4), `deploy/update_snapshot.sh` (prints `git diff <old>..<new> -- src/so101_rl/{servo,safety,config,authority}` for review, requires `pixi run -e cpu test` green at the new sha, builds, refreshes SHA256SUMS, switches `current`, appends a line to the checklist results table template), `deploy/verify.sh` (runs as koen: `sha256sum -c /etc/so101_rl/SHA256SUMS`, `id koen` has no `dialout`, device owners, `systemctl cat so101-servo@.service` argv matches the repo unit, polkit rule present, `records/` not writable by koen; exit 0/1 — A12 scriptable), `deploy/README.md` (the install checklist text, the "never run AI-suggested commands in the signing session" rule, the suggested **user-owned** Claude Code permission rule text for `.claude/settings.json` (deny `Edit` on protected paths — documented only, the plan never edits settings), the `RT_GROUP_SCHED` note (not set on this kernel: skip the slice step)), `tests/test_deploy_files.py`
- **Specialist:** ansible-specialist is overkill; the install is a reviewed shell script by design (read in full at a pinned commit before `sudo`)
- **Dependencies:** OS1.1, OS1.2; SV1 for the unit's argv flags (the unit file is written against FL8 now)
- **Tests:** `test_deploy_files.py::test_unit_argv_matches_fl8_and_has_one_sigterm_semantics` (`KillMode=mixed`, `SendSIGHUP=no`, `TimeoutStopSec=30`, `Restart=no`, `--arm-timeout-s 110`, `--exit-on-terminal`), `::test_unit_uses_by_id_ports`, `::test_polkit_rule_matches_only_servo_template_and_three_verbs`, `::test_sysusers_tmpfiles_modes_as_designed`, `::test_install_dry_run_against_fake_root_creates_expected_tree_and_sha256sums` (fake `sha256sum`, `udevadm`, `gpasswd`, `pixi` shims on PATH), `::test_build_snapshot_tree_allowlist_only` (dry-run lists exactly the FL-allow-listed paths), `::test_verify_sh_fails_on_tampered_file_and_on_dialout_membership` (fake root; shown to fail), `::test_shims_env_i_absolute_paths_no_test_flags`, `::test_allowed_signers_namespaces_exact`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_deploy_files.py -q` → pass; `bash deploy/install.sh --dry-run --root /tmp/so101root` → exit 0 and prints every step; `bash deploy/verify.sh --root /tmp/so101root` → exit 0 on the dry-run tree.
- **Est.:** 5 agent-h.

#### Task OS1.4 (SW): protected-path guard
- **Files:** `docs/protected_work_items.yaml` (`protected: [src/so101_rl/safety/**, src/so101_rl/servo/**, src/so101_rl/authority/**, firmware/**, src/so101_rl/config/so101.yaml, deploy/**]`, `work_items: [SV1, FW1, OS1]`, `since: <base commit sha>`), `scripts/check_protected_paths.py` (`--range <rev>..<rev>`: every commit touching a protected path carries `Work-Item: <id>` with a listed id; exit 1 otherwise; `--staged` mode for a pre-commit check), `tests/test_protected_paths.py` (a temp git repo: a commit touching `safety/x.py` without the trailer fails, with it passes; `--staged` same), `pyproject.toml` `ci` depends on `check-protected` (= `python scripts/check_protected_paths.py --range origin/main..HEAD`, best-effort when `origin/main` is absent → `--range HEAD~20..HEAD`).
- **Acceptance:** `pixi run -e cpu pytest tests/test_protected_paths.py -q` → pass; `pixi run -e cpu python scripts/check_protected_paths.py --range b109e18..HEAD` → exit 0 (no protected paths touched since the base except by trailer-carrying commits).
- **Est.:** 1.5 agent-h.

#### Task OS1.5 (GRILL, recommended): adversarial review of OS1-SW
- **Files:** `plans/reviews/<date>-os1-review.md`
- **Steps:** attacker brief: "forge/replay/rollback a record; sign from a pty, via su/sudo, with an agent; sneak a byte through the grammar; make the verifier accept a record under the wrong namespace or a future-dated approval; escape the snapshot allow-list; widen the polkit rule; leak a test flag into a shim; a koen-writable path in the trust chain".
- **Acceptance:** review `APPROVE`; `pixi run -e cpu ci` green.
- **Est.:** 3 agent-h.

#### Task OS1.6 (SW): OS1-SW sync (shared write)
- **Files:** `docs/ARCHITECTURE.md` (§4 `authority/` + `tools/`, `deploy/`; arrows `servo.main → authority`, `lab → authority`, `authority → (stdlib only)`; §8 host requirements, OS1 artefacts, the unit, polkit; §10 row), `docs/HARDWARE_CHECKLIST.md` (new section **"OS1 — two-account install (human-run)"**: the ordered checklist = `deploy/README.md` steps with the read-in-full-at-a-pinned-commit rule, key generation by hand as `so101-approver` on a text console (`/usr/bin/ssh-keygen -t ed25519 -a 256 -C so101-approver`, diceware passphrase, never `ssh-add`), `sudo bash deploy/install.sh <sha>`, `deploy/verify.sh`, a results-table row; plus the Safety-warnings bullet "never run AI-suggested commands in the signing session"; plus "koen leaves `dialout`: `servo --backend feetech` from the user account stops working by design"), `README.md`
- **Est.:** 0.5 agent-h.

**Commit after OS1.6:** `OS1-SW: stdlib authority module (records, Tier 1 rules, nonce/arm-minute ledgers), console-only records tools, deploy files, protected-path guard` with trailer `Work-Item: OS1`.

---

### Unit OS1-HW — the two-account install (HW/human; after OS1-SW and SV1; before any real session)

| Step | Who | What | Pass |
|---|---|---|---|
| OS1-H1 | human, console | Read `deploy/install.sh`, `deploy/build_snapshot.sh`, `deploy/verify.sh`, the unit, the polkit rule and the shims **in full at the pinned commit** (`git show <sha>:deploy/…`) | notes in the checklist row |
| OS1-H2 | human, console as `so101-approver` (after `sudo bash deploy/install.sh --accounts-only`) | `ssh-keygen -t ed25519 -a 256 -C so101-approver` into `~/.ssh/so101_approver_ed25519` with a diceware passphrase | `~/.ssh` 0700, key 0600 |
| OS1-H3 | human, sudo (never in an AI-driven session) | `sudo bash deploy/install.sh <sha>` (sysusers, tmpfiles, udev with the two adapter serials, `gpasswd -d koen dialout`, tools + shims, unit, polkit, allowed_signers, snapshot build + `current`, `snapshot.env`, SHA256SUMS, `runs/` group perms); log out/in (group change) | `deploy/verify.sh` exit 0 |
| OS1-H4 | human | A12 checks by hand: `ls -l /dev/serial/by-id/so101-*` owned by `so101-servo 0600`; `id koen` without `dialout`; `so101-approve` from a `/dev/pts` terminal refuses; `sudo -u so101-approver so101-approve` refuses; `systemctl cat so101-servo@.service` shows the pinned argv; `systemctl start so101-servo@test` as koen is allowed by polkit and `systemctl start sshd` is not; `sha256sum -c /etc/so101_rl/SHA256SUMS` passes; `touch /var/lib/so101_rl/records/x` as koen fails | results-table row `OS1`, artifact `runs/hw/os1-<ts>.json {verify_exit, dialout_removed, devices_owned, pty_refused, sudo_refused, polkit_scoped, sums_ok}` |
| OS1-H5 | human | Shown-to-fail: edit one byte of `/usr/local/lib/so101_rl/so101_authority.py` in a copy and run `sha256sum -c` → must fail; restore | row note |

Est.: 4 human-h (+ snapshot build time; network or a warm pixi cache needed once).

---

### Unit PE1-SW — perception process (wave 1, **BUILD NOW**)

#### Task PE1.1 (SW): `perception/main.py`
- **File:** `src/so101_rl/perception/main.py`
- **Files (tests):** `tests/test_perception_main.py`
- **Specialist:** robotics-vision-navigator (fiducial cadence); hardware-interface-specialist (camera by serial)
- **Dependencies:** none (U5 built); hand-off from SUC for `calib/camera.yaml` when it lands (SUC owns `perception/{camera,frames,die_tracker}.py` concurrently — PE1 only *calls* them)
- **Steps:** FL10 verbatim; `fake` scene built from `tests/test_die_convention.py`'s renderer helpers (import from `so101_rl.sim` lazily; the `fake` path needs the `sim` feature, which `cpu`/`default` have); publish at `--hz`; `fiducial_ok` cadence; `status.json`; SIGTERM → close the shm (owner unlinks) and exit 0.
- **Tests:** `test_perception_main.py::test_fake_scene_publishes_centroid_visible_and_fiducial_ok`, `::test_die_hidden_publishes_invisible_held_position`, `::test_camera_moved_scene_sets_fiducial_ok_zero`, `::test_fiducial_zero_until_first_check_and_without_yaml_in_d435_mode` (d435 path with a monkeypatched camera factory), `::test_status_json_written_and_max_frames_exit_0`, `::test_env_consumes_table_from_main` (`So101GoalEnv` with `SimBackend` + the table from a `main` thread sees `die_visible`), `::test_missing_camera_exit_2`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_perception_main.py -q` → pass; `pixi run -e cpu perception --camera fake --max-frames 60 --status /tmp/p.json` → exit 0, `fps ≈ 30`, `fiducial_ok: 1`.
- **Est.:** 4 agent-h.

#### Task PE1.2 (SW): PE1 sync (shared write)
- **Files:** `pyproject.toml` (`perception` task), `docs/ARCHITECTURE.md` (§3 perception process as built: entry point, cadence, `fiducial_ok` semantics, the transient unit (LA4); §4 `perception/main.py`; §10 row), `README.md`
- **Est.:** 0.3 agent-h.

**Commit after PE1.2:** `PE1: perception process entry point (D435/fake), object table + fiducial verdict, status file`.

---

### Unit LA2 — FastSAC unattended pipelines (GATED on SU3, SU4, SUV; wave 2 alongside LA1)

#### Task LA2.1 (SW + GPU): `fastsac_sim` plugin, `gates` and `eval` runs, S4 pipelines
- **Files:** `src/so101_rl/lab/trainers/fastsac_sim.py`, `lab/schemas/trainer/fastsac_sim.v1.schema.json`, `src/so101_rl/rl/train_sim_sac.py` (hand-off from SU4: an additive library entry `run_training(cfg: TrainConfig, *, hooks: TrainHooks) -> None` where `TrainHooks` = `{on_eval(step, metrics), should_stop(), should_pause(), on_checkpoint(path, step)}`; the CLI keeps working), `tests/test_lab_fastsac.py`
- **Specialist:** lerobot-specialist; cuda-performance-engineer (VRAM fraction with the supervisor's single gpu lease)
- **Dependencies:** LA0, SU3 (`DomainRandomizer`), SU4 (`fast_sac`, `gates`, `train_sim_sac`, `eval_sim`, `policy_export`, `rollout_logger`), SUV (done)
- **Steps:** `Capabilities(domains={sim}, gpu="required", vram_gb_estimate=2.5, resumable=True, evaluates=True, exports=True, env_name="default")`; `plan()` → `budget_key f"s4.{task}"`, `leases={gpu}`, `requires=("gate:L1:<task>", "gate:L2:<task>")` plus `"sim_params:verified"` for `die_to_zone` (D-DIE-PARAMS); `train` = `run_training` with hooks wired to `RunContext` (heartbeat per eval, `review_trigger` from S4.3's `review.json` names, checkpoints); `resume` = `--resume ckpt_last.pt` + `overrides` limited to `resumable_fields=("budget_min", "dr_max_stage", "eval_every")`; `evaluate` = `eval_sim.evaluate(...)` → `eval.json`, `renders/`, `review_checklist.md`, `gate.json` (numeric); `export` = `policy_export.export_policy`; `run_gates` tool → a `gates` run calling `rl.gates.main` under the cpu lease → `runs/<id>/gate.json` copied to `runs/gates/<task>-<ts>.json` (where `check_gates` looks). `TRAINER_UNAVAILABLE` disappears for `fastsac_sim`.
- **Tests:** `test_lab_fastsac.py::test_plan_requires_gates_and_die_needs_verified_params`, `::test_config_schema_refuses_thermal_and_caps_keys`, `::test_resume_overrides_limited`, `@gpu @slow ::test_train_2min_budget_through_job_entry` (metrics ≥ 3 rows, ckpt, usage debited by progress time), `@gpu ::test_evaluate_gate_export_chain`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_fastsac.py -q -m "not gpu"` → pass; `pixi run pytest tests/test_lab_fastsac.py -q -m gpu` → pass (~0.3 GPU-h).
- **Est.:** 8 agent-h + 0.5 GPU-h.

#### Task LA2.2 (GPU RUN, `@integration`): A1–A3 on the real trainer — the unattended reach pipeline with Claude Code closed
- **Files:** `tests/test_lab_unattended_fastsac.py`, `plans/reviews/<date>-la2-unattended-run.md`
- **Dependencies:** LA2.1, LA1.3, `calib/sim_params.yaml` (v0 acceptable; the run is repeated after HS3)
- **Steps:** through the MCP client: `run_gates(reach)` → `start_run(fastsac_sim, reach, {budget_min: 30, nworld: 1024, …}, kind="pipeline", pipeline={evaluate: {episodes: 1000, dr_stage: 4, render: 16, gate: true}, export: true})`; close the client; wait; assert `finished`, `eval.json`, `gate.json`, renders, `review_checklist.md`, `policy_export.pt`, ledger debited by progress time, audit trail complete, no human action (A1). A2: inject a review trigger via config (`review_trigger_at`-equivalent: S4.3's UTD trigger forced by `grad_steps_per_env_step: 1` + a low threshold override allowed only in the `probe` kind) → pause → resume → third stop. A3: thermal fake sampler 86 °C → pause with the gpu lease held → 74 °C → resume; stall → kill; disk low → graceful stop; the job's own deadline with the supervisor killed.
- **Acceptance:** `pixi run pytest tests/test_lab_unattended_fastsac.py -q` → pass (≈ 1 GPU-h incl. the 30-min budget run); the review note lists the run id, the eval numbers and the screenshot of the dashboard.
- **Est.:** 3 agent-h + 1.5 GPU-h.

#### Task LA2.3 (SW): LA2 sync — `docs/ARCHITECTURE.md` (§3 trainer process via the lab; §4 plugin; §10 row), `README.md`. Est. 0.3 agent-h.

**Commit after LA2.3:** `LA2: fastsac_sim plugin, gates/eval/export runs and pipelines; unattended reach pipeline (A1–A3)`.

---

### Unit LA3 — SB3 + TD-MPC2 plugins, `--tier1` mode (GATED on SU5, SR2; wave 3)

#### Task LA3.1 (SW): three plugins
- **Files:** `src/so101_rl/lab/trainers/sb3_sac_her.py`, `lab/trainers/tdmpc2_offline.py`, `lab/trainers/tdmpc2_finetune.py`, `lab/schemas/trainer/{sb3_sac_her,tdmpc2_offline,tdmpc2_finetune}.v1.schema.json`, `tests/test_lab_plugins_s5.py`
- **Specialist:** lerobot-specialist
- **Dependencies:** LA2, SU5 (`sb3_bridge`, `train_sac_her`, `tdmpc2_offline`, `train_tdmpc2`, `deploy_policy`), SR2 (`her`, `tdmpc2_env`)
- **Steps:** `sb3_sac_her`: domains `{sim, real}`; sim via `SimBackend` or `ShmBackend` against a `servo --backend mujoco` child (the job spawns it with `--shm-name lab-<run_id>`; never serial); real via `ShmBackend(client_id=…)` (LA4); `budget_key f"s5.{task}.finetune.sb3"`; `tdmpc2_offline`: domain `offline`, `dataset {id | from_run}` auto-registers `runs/datasets/<id>.json`; `tdmpc2_finetune`: `{sim, real}`. Real-arm entry points gain `--tier1 <tier record>` replacing `--attended` for unattended runs and take limits from `ctx.limits()` (`rl/deploy_policy.py`, `rl/train_sac_her.py`, `rl/train_tdmpc2.py`, `sim/sysid/run_excitation.py` — hand-offs from SU5/SU2, additive flags only).
- **Tests:** `test_lab_plugins_s5.py::test_sb3_sim_300_steps_through_job` (`@slow`), `::test_sb3_shm_mode_spawns_mujoco_servo_child_never_serial` (monkeypatched `serial.Serial` raises), `::test_tdmpc2_offline_registers_dataset_from_run`, `::test_tdmpc2_finetune_init_from_offline_ckpt`, `::test_tier1_flag_refuses_without_record_and_takes_limits_from_ctx`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_plugins_s5.py -q -m "not slow"` → pass; `pixi run -e cpu pytest tests/test_lab_plugins_s5.py -q -m slow` → pass < 10 min.
- **Est.:** 9 agent-h.

#### Task LA3.2 (SW): LA3 sync — ARCHITECTURE §4/§10, README, plugin table. Est. 0.3 agent-h.

**Commit after LA3.2:** `LA3: sb3_sac_her, tdmpc2_offline, tdmpc2_finetune plugins; --tier1 mode on the real-arm entry points`.

---

### Unit LA4-SW — real-arm sessions, Tier 0 (software on the fake backend; wave 3 after LA1, SV1, FW1, OS1-SW, PE1-SW)

Owns `src/so101_rl/lab/session.py`, `lab/mcp_server.py` (**only** the `request_real_run` registration block — hand-off from LA1, sequential), `lab/trainers/fake.py` (**only** the real-domain behaviours `servo_trip:<cause>`, `late_attach`, `park_refused`, `resume_refused` — hand-off from LA0), `tests/test_lab_session.py`, `tests/test_lab_authority_mirror.py`, `tests/test_lab_real_fake.py`.

#### Task LA4.1 (SW): session manager — request file, mirror checks, device identities, servo unit start, attach claims, cmd-owner lock, two-phase verdict, arm-time ledger mirror
- **File:** `src/so101_rl/lab/session.py`
- **Files (tests):** `tests/test_lab_session.py`
- **Specialist:** real-time-control-specialist (attach/lock semantics)
- **Dependencies:** LA1, SV1 (FL8), FW1 (FakeMcu reason 8), OS1-SW (FL6/FL7), PE1-SW (FL10)
- **Steps:** FL6 + FL12 + design §5.6: `SessionManager(runs, policy, authority_mirror, launcher, servo_ctl)`; `servo_ctl` = `SystemctlServoControl` (`systemctl start|stop|show so101-servo@<request_id>`; subprocess, no sudo — polkit) with a `FakeServoControl` that runs `python -m so101_rl.servo.main --backend fake --session <file> --records <tmp> --allowed-signers <tmp> --state-dir <tmp> --exit-on-terminal --shm-name lab-test-<pid>` as a child (**never serial**); `open_session(request)` = mirror checks (hw artifacts fresh, tier record valid, devices resolvable + owned (skipped for the fake backend), RT limits readable) → write the request file → start the unit → poll `state.json`; on `phase == boot_wait` open item `session_open_request` + notify; `DEVICE_UNRESOLVABLE` → item, no start; `attach(run)` = allocate `client_id` (u32 random, non-zero), write the attach claim, take `flock` on `<attach_dir>/<request_id>/cmd.lock` (the cmd-owner lock; held by the job segment), hand the job `ShmBackend(client_id=…)`; `close_session()` = the job's last act is `PARK`; the manager sends nothing on the job's behalf (panel 8); then `STOP_REQ` via the job → wait `exit_wait_s = bound_s + 10` for the pid to vanish → else `systemctl stop` (one SIGTERM) → else item `servo_exit_failed`; phase-1 verdict from `summary.json`; phase-2 at the next session's first `status.json`; `runs/sessions.jsonl`; arm-minute mirror (`runs/arm_minutes.jsonl`); `host_slept`/reboot void approvals and the real queue.
- **Tests (fake servo child, real authority module, temp records store with a throwaway key):** `test_lab_session.py::test_request_file_schema_and_unit_instance_name`, `::test_device_identity_refusal_before_any_button` (fake `os.stat` owner / missing symlink → `DEVICE_UNRESOLVABLE`, no child started), `::test_two_timer_sequencing_press_before_servo_latches_reason_8_and_next_request_says_press_again` (FakeMcu), `::test_two_timer_sequencing_press_after_servo_waits_arms_within_110s`, `::test_no_press_within_110s_servo_exits_2_item_timed_out`, `::test_attach_claim_client_id_and_monotonic_seq_continuation`, `::test_cmd_owner_lock_second_job_blocked`, `::test_stop_req_exit_pid_gone_fakemcu_latched` (A5), `::test_systemctl_fallback_one_sigterm_simulated`, `::test_two_phase_verdict_servo_clean_then_weld_ok_next_session`, `::test_weld_reason_7_fails_phase_2`, `::test_cap_raise_refused_without_weld_ok`, `::test_host_slept_voids_approvals_and_real_queue`, `::test_late_attach_after_reenergise_stays_parked_30s`.
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_session.py -q` → pass (≥ 14).
- **Est.:** 9 agent-h.

#### Task LA4.2 (SW): `request_real_run`, approval requests, the lab's REAL mirror, fake real-domain behaviours, items with console commands
- **Files:** `src/so101_rl/lab/mcp_server.py` (the `request_real_run` block, registered only when `SO101_LAB_REAL_ARM=1`, description "**guard rail, not a boundary**"), `src/so101_rl/lab/authority.py` (REAL columns of design §5.1 as mirrors: Tier 0 → `awaiting_approval`, ≤ 1 open request per kind, hash recomputation at admission and at job start; Tier 1 → LA5), `src/so101_rl/lab/trainers/fake.py` (real-domain behaviours), `src/so101_rl/lab/dashboard.py` (session panel — hand-off from LA1), `tests/test_lab_authority_mirror.py`, `tests/test_lab_real_fake.py`
- **Dependencies:** LA4.1
- **Steps:** `request_real_run` → `approval_request.json` (FL6) + item `approval_pending` with `command: "so101-approve <run_id>"` (display only) → state `awaiting_approval`; the supervisor admits the run only when `Verifier.verify_file` + `check_approval` pass against the current store (mirror) and the servo is open (`arm_ns` known); the job writes `limits.json` from the record and takes `ctx.limits()` from it; a tripped real run needs a signed ack (item `requires: ack`) before any real admission (`SESSION_RULE` otherwise); the failure matrix (design §5.7) on the fake servo child with the fake trainer's real-domain behaviours.
- **Tests:** `test_lab_authority_mirror.py::test_tier0_request_awaiting_approval_and_one_open_per_kind`, `::test_lab_processes_cannot_write_records_store` (WriteGuard + the store path is not writable in the temp layout), `::test_admission_recomputes_hashes_and_refuses_changed_config` (`RECORD_INVALID`), `::test_expired_consumed_wrong_key_wrong_namespace_refused`, `::test_tripped_run_blocks_real_admissions_until_signed_ack`; `test_lab_real_fake.py::test_matrix_thermal_during_run_pauses_servo_wait_then_resumes`, `::test_matrix_thermal_while_parked_resume_req_after_dwell`, `::test_matrix_idle_timeout_then_resume_req_then_run`, `::test_matrix_nan_trip_exit_latched_item_blocked_until_ack`, `::test_matrix_stop_req_clean_close`, `::test_matrix_job_crash_stale_cmd_exit_item`, `::test_matrix_client_refused_counted_job_ends`, `::test_matrix_supervisor_restart_touches_nothing`, `::test_full_a5_sequence_fake` (request → approval by the records tool under the test flags → session → fake button → ARM → HOLD(parked) → attach → admitted → RUN → PARK → STOP_REQ → pid gone + LATCHED).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_authority_mirror.py tests/test_lab_real_fake.py tests/test_lab_contract.py -q` → pass (`test_exactly_12_tools` still 12 without the env flag; a second contract test asserts 13 with it); `SO101_LAB_REAL_ARM=1 pixi run -e cpu pytest tests/test_lab_contract.py -q -k thirteen` → pass.
- **Est.:** 8 agent-h.

#### Task LA4.3 (SW): LA4 sync — ARCHITECTURE (§3 session lifecycle as built, §5 FL6/FL12 pointers, §6 item 7 enforcement table, §8 `SO101_LAB_REAL_ARM` documented as a guard rail; §10 row), README, `docs/HARDWARE_CHECKLIST.md` (the session-opening procedure: `deploy/verify.sh` first, then the request, the button only when the dashboard says the servo is waiting). Est. 0.5 agent-h.

**Commit after LA4.3:** `LA4-SW: real-arm session manager, Tier 0 approvals via the connector, failure matrix on the fake servo (A5 fake)`.

---

### Unit LA5-SW — Tier 1 unattended (software; wave 4 after LA4-SW)

#### Task LA5.1 (SW): tier reader mirror, rules, cap ramp, soak verification at admission, per-batch revalidation, A6 on the fake backend
- **Files:** `src/so101_rl/lab/tier1.py`, `src/so101_rl/lab/authority.py` (Tier 1 column: mirror of `Tier1Rules`), `src/so101_rl/lab/session.py` (tier-1 attach claims), `tests/test_lab_tier1.py`
- **Specialist:** none needed beyond the design
- **Dependencies:** LA4-SW
- **Steps:** `verify_tier_for_admission()` = `Verifier.verify_current("tier")` + `check_tier` (snapshot sha = `/opt/so101_rl/current` target's sha, soak report shas incl. the 120-min report's parked metrics via `check_soak_report(strict=True)`, hw artifact shas, `valid_until`); rules mirror (`SESSION_RULE` with the rule name); cap ramp from `runs/sessions.jsonl` (clean at the current cap ≥ 1 in **both** stores — the servo store is read by group); per-batch revalidation before every real admission; queued real entries re-validated after `host_slept`/reboot.
- **Tests (fake servo + real authority module):** `test_lab_tier1.py::test_two_runs_one_session_no_approvals_parked_between`, `::test_late_attach_stays_parked`, `::test_idle_rest_resume_req_parked_run`, `::test_thermal_during_run_and_while_parked_per_matrix`, `::test_nan_exit_latched_item_blocked_until_ack`, `::test_refusals[quiet_hours|min_gap|per_day_cap|allow_list|cap|tier_claim]`, `::test_two_phase_verdict_recorded_and_cap_raise_refused_without_weld_ok`, `::test_admission_reverifies_soak_report_hash` (a changed report byte → `RECORD_INVALID`), `::test_servo_side_refusal_counted_lab_mirror_agrees` (the lab lets a request through on purpose via a test hook; the servo refuses; `client_refused` recorded — the mirror is an early warning, the servo holds).
- **Acceptance:** `pixi run -e cpu pytest tests/test_lab_tier1.py -q` → pass (A6 fake).
- **Est.:** 7 agent-h.

#### Task LA5.2 (SW): LA5 sync — ARCHITECTURE §6 autonomy tiers (Tier 1 rules as enforced by the servo, mirrored by the lab), §10 row; README; `docs/HARDWARE_CHECKLIST.md` (HS9 + LA5-HW procedure). Est. 0.5 agent-h.

**Commit after LA5.2:** `LA5-SW: Tier 1 mirror (signed tier record, rules, cap ramp, soak revalidation); A6 on the fake backend`.

---

### Hardware-gated units (all **HW**, attended until H13; each appends a results-table row and an artifact under `runs/hw/`)

| Id | Task | Needs (hardware) | Needs (software) | Procedure (abridged) | Artifact / pass | Est. (human-h) |
|---|---|---|---|---|---|---|
| **OS1-HW** | Two-account install | the two USB adapters present | OS1-SW, SV1 | the OS1-H1…H5 table above | `runs/hw/os1-<ts>.json`, `deploy/verify.sh` exit 0 | 4 |
| **FW1-HW** | H3 extension | H3 | FW1 | flash the new `main.py`; replay every vector incl. the four new ones; press without ARM → `0x80` for 120 s then `0x82`; press again → `0x80`; ARM within 2 min | `runs/hw/deadman-<ts>.json` gains `arm_timeout_s ≈ 120`, `vectors_passed` incl. the new names | 0.5 |
| **HS8** | Parked session + thermal hold | H2, H3, H5, H7, H11, OS1-HW | SV1 | operator-opened 60-min session on the real bus (`so101-servo@…` via the lab or by hand with the pinned argv), three park → idle → resume cycles driven by `soak --park-every-min 15 --park-for-s 660` (idle timeout at 600 s fires once per cycle); then a 20-min random run followed by a 10-min hold at `home` logging per-joint temperatures | `runs/hw/park-resume-<ts>.json {cycles: 3, parked_hold_entries, idle_rests, resume_cycles, hb_max_gap_ms ≤ 200, temp_slope_c_per_min per joint, max_temp_c_after_hold < 50}`; on failure `park_timeout_s` is lowered (a proposal, P3a) | 2 |
| **HS9** (new id, D9) | 120-min goal-directed strict soak with ≥ 2 parked segments | H10, HS8 | SV1 | `REQUIRE_RT=1 pixi run soak --backend feetech --minutes 120 --park-every-min 50 --park-for-s 180` attended | `runs/soak-<ts>/report.json` passes `check_soak_report(strict=True)` incl. the parked invariants; its sha256 goes into the tier record (`goal_directed_120min`) | 2.5 |
| **LA4-HW** | First real use via the connector | HS8, OS1-HW, FW1-HW | LA3, LA4-SW | HS1 (`request_real_run(kind=excitation)`) and HS4 (`zero_shot_eval`, 50 episodes) through the connector + `so101-approve` on the console; the planned HS1/HS4 artifacts | `runs/deploy-reach-<ts>/eval.json`, `runs/sysid/<ts>/`, `runs/sessions.jsonl` with two clean sessions; review note | 4 |
| **LA5-HW** | First unattended 60-min session | H13, HS9, HS8, one clean attended 60-min session (two-phase) | LA5-SW | `so101-gate-sign --tier` signs the tier record (cap 60); the AI runs HS5 fine-tuning batches; the human is away; the session ends by cap or idle with the arm resting torque-off; the next ARM gives `weld_ok` | `runs/sessions.jsonl` clean; review note; only then the ramp (120) may be signed | 2 (+ unattended arm time) |

## Acceptance mapping (design §10 A1–A12; every row has a scriptable check)

| # | Criterion (design) | Scriptable check | Unit |
|---|---|---|---|
| **A1** Unattended sim end to end | CI half: `pixi run -e cpu pytest tests/test_lab_unattended.py::test_pipeline_fake_end_to_end_unattended -q` (fake trainer; the MCP client process is killed after `start_run`; `finished`, `eval.json`, `gate.json`, `policy_export.pt`, ledger debited, audit complete). GPU half: `pixi run pytest tests/test_lab_unattended_fastsac.py::test_reach_pipeline_unattended -q` (FastSAC 30-min budget via the connector with Claude Code closed) | LA1.3, LA2.2 |
| **A2** Review triggers unattended | `pixi run -e cpu pytest tests/test_lab_supervisor.py::test_review_pause_resume_third_trigger_stops_probe_continues -q`; GPU twin in `test_lab_unattended_fastsac.py::test_review_trigger_pause_resume_stop` | LA0.4, LA2.2 |
| **A3** Budgets, thermal, health, stall, disk, own deadline | `pixi run -e cpu pytest tests/test_lab_supervisor.py tests/test_lab_budgets.py tests/test_lab_job.py -q -k "thermal or empty_nvml or stall or disk or budget or deadline"` | LA0.3, LA0.4 |
| **A4** Crash, suspend, reboot, session independence (real `systemd --user`) | `pixi run -e cpu pytest tests/test_lab_supervisor.py -q -m integration` + `tests/test_lab_unattended.py::test_fake_run_outlives_mcp_client` + `::test_fake_run_outlives_supervisor_kill_9_and_is_readopted`; suspend/reboot simulations in `test_lab_supervisor.py` | LA0.4, LA1.3 |
| **A5** Tier 0 signed approval + session (fake backend) | `pixi run -e cpu pytest tests/test_records_tools.py tests/test_lab_authority_mirror.py tests/test_lab_session.py tests/test_lab_real_fake.py::test_full_a5_sequence_fake -q`. **First real use:** LA4-HW row (HS1, HS4 through `request_real_run` + console approval) | OS1.2, LA4.1, LA4.2, LA4-HW |
| **A6** Tier 1 unattended (fake, then real) | `pixi run -e cpu pytest tests/test_lab_tier1.py tests/test_lab_session.py -q`. **Real:** LA5-HW row | LA5.1, LA5-HW |
| **A7** Dashboard | `pixi run -e cpu pytest tests/test_lab_dashboard.py -q`; screenshot in the LA1 review note | LA1.2 |
| **A8** Constraints | `pixi run -e cpu python scripts/check_protected_paths.py --range <base>..HEAD` → exit 0; `pixi run -e cpu pytest tests/test_lab_deps.py tests/test_protected_paths.py -q`; every lab commit touches ARCHITECTURE (`tests/test_architecture_sync.py`); no §10 row edited by the build (the master applies §12 only); `deploy/verify.sh` exit 0 after OS1-HW (OS1 files match `SHA256SUMS`); `test_lab_deps.py::test_lab_imports_no_snapshot_code` (the lab never imports `servo.main/core`) | OS1.4, LA0.1 |
| **A9** SV1 obligations | `pixi run -e cpu pytest tests/test_sv1_state_machine.py tests/test_sv1_core.py tests/test_sv1_process.py tests/test_soak.py tests/test_servo_core.py tests/test_batched_parity.py -q -m "not gpu"` (landing in parked HOLD, PARK tolerance, `monitor_parked` gripper-point trip, exit-on-terminal pid gone + LATCHED in the three cases, monotonic dedupe, soak parked metrics with the shown-to-fail twin, tracemalloc, parity unchanged). **HW:** HS8 artifact incl. the thermal hold | SV1.2–SV1.5, HS8 |
| **A10** FW1 | `pixi run -e cpu pytest tests/test_firmware_vectors.py tests/test_heartbeat.py tests/test_servo_process.py::test_arm_timeout_cli_default_is_none -q`; `grep -- "--arm-timeout-s 110" deploy/systemd/so101-servo@.service`; **HW:** FW1-HW vectors on the real MCU | FW1.1, FW1.2, OS1.3, FW1-HW |
| **A11** Token discipline | `pixi run -e cpu pytest tests/test_lab_contract.py -q` (`tools/list` ≤ 12 kB, responses ≤ 32 kB, no raw logs/video, envelopes) | LA1.1 |
| **A12** OS1 | `bash deploy/verify.sh` → exit 0 (devices owned by `so101-servo`, koen not in `dialout`, pinned argv, polkit scope, `sha256sum -c`); pty/sudo refusals in `tests/test_records_tools.py`; **HW:** OS1-H4 by hand | OS1.3, OS1-HW |

**"Full unattended sim run with Claude Code closed"** is therefore demonstrated twice: with the fake trainer in LA1 (`scripts/lab_unattended_fake_run.sh`, no GPU, ~3 min) and with FastSAC in LA2 (`test_lab_unattended_fastsac.py`, ~1 GPU-h).

## Execution Strategy

- **Wave 0 (sequential, ~1 h):** LP0 — pyproject `lab` feature, `integration` marker, lock. Everything else codes against the installed `mcp`/`jsonschema`/`pynvml`.
- **Wave 1 (parallel, 5 units, exclusive files):** LA0 (lab core), SV1 (servo; GRILL), FW1 (firmware; GRILL), OS1-SW (authority + records tools + deploy; GRILL recommended), PE1-SW. Hand-offs inside the wave are interface-only: SV1 imports `so101_rl.authority` lazily inside the `--session` branch and tests it with a fake module; OS1-SW's unit file is written against FL8; the integration of wave 1 re-runs `tests/test_sv1_process.py` and `tests/test_deploy_files.py` together.
- **Wave 2 (parallel):** LA1 (MCP + dashboard + unattended fake run) ‖ LA2 (gated on SU3/SU4 — starts only when those are merged; otherwise the wave is LA1 alone).
- **Wave 3 (parallel):** LA4-SW (needs LA1, SV1, FW1, OS1-SW, PE1-SW) ‖ LA3 (gated on SU5/SR2).
- **Wave 4:** LA5-SW (needs LA4-SW).
- **Hardware wave (interleaved, human):** OS1-HW as soon as wave 1 is merged; FW1-HW with H3; HS8 after H2/H3/H5/H7/H11 + OS1-HW; HS9 after H10 + HS8; LA4-HW after LA3 + LA4-SW + HS8; LA5-HW after H13 + HS9 + one clean attended session.
- **Always-parallel:** quality review + test generation per unit (executor standard step); the GRILL steps are extra and dedicated (SV1.6, FW1.3, OS1.5).
- **Branch base:** `feature/camera-parity` (bf7346b) carries everything this plan reads; the camera-parity unit (SUC) is being built in this checkout — SUC's files (`servo/probe.py`, `calib/camera.yaml`, `sim/sim_tracker.py`, `viz/video.py`, `perception/{camera,frames,die_tracker}.py`, `assets/so101/so101_rl_scene.xml`) are owned by nobody in this plan.

Rough software effort: **≈ 170 agent-hours** (LP0 1, LA0 22.5, LA1 14, SV1 27.5, FW1 4, OS1-SW 25, PE1 4.3, LA2 11.3, LA3 9.3, LA4-SW 17.5, LA5-SW 7.5, + ~25 h of integration/review/fix slack across the GRILL and wave merges), **≈ 3 GPU-hours** (LA2 acceptance 2, LA3 smoke 1), **≈ 16 attended human-hours** (OS1-HW 4 incl. key generation, FW1-HW 0.5, HS8 2, HS9 2.5, LA4-HW 4, LA5-HW 2, plus console approvals during LA4-HW). Critical path to the first unattended fake run: LP0 → LA0 → LA1 ≈ 37 agent-h. Critical path to the first real session via the connector: + SV1 ‖ OS1-SW (wave 1, 27.5 h wall) → LA4-SW 17.5 → OS1-HW/HS8 (human) → LA4-HW, gated on LA3 (gated on SU5/SR2).

## Git Commits

One commit per unit as listed under each unit (`LP0`, `LA0`, `LA1`, `SV1` (+ `Work-Item: SV1`), `FW1` (+ `Work-Item: FW1`), `OS1-SW` (+ `Work-Item: OS1`), `PE1`, `LA2`, `LA3`, `LA4-SW`, `LA5-SW`), plus one commit per GRILL review note, one per hardware artifact row (`docs/HARDWARE_CHECKLIST.md` results table + `runs/hw/*.json` are gitignored except the row), and the master's plan edits (`so101-self-learning-rl-plan.md` §12 v3 text; any confirmed decision from the list below as a §10 row). Every commit touching `src/so101_rl/{safety,servo,authority}/`, `firmware/`, `deploy/` or `config/so101.yaml` carries its `Work-Item:` trailer (`scripts/check_protected_paths.py` in `ci` from OS1.4 on). The master commits; executors stage per task.

## How to Run

Single-session run (default):
1. Open this plan in the executor and proceed wave by wave, unit by unit; run each task's acceptance command before moving on; `integration` tests run on this host (a `systemd --user` session exists); GPU acceptance (LA2) in the `default` env.

Parallel run (wave 1 and the gated waves carry `parallelizable-via-worktree: true`):
1. Spawn one worktree per parallel unit:
   ```bash
   /worktree create feature/connector-la0-core
   /worktree create feature/connector-sv1-servo-park
   /worktree create feature/connector-fw1-arm-timeout
   /worktree create feature/connector-os1-authority
   /worktree create feature/connector-pe1-perception
   ```
   wave 2: `feature/connector-la1-mcp`, `feature/connector-la2-fastsac`; wave 3: `feature/connector-la4-sessions`, `feature/connector-la3-plugins`; wave 4: `feature/connector-la5-tier1`.
2. Open a terminal per worktree, `cd` into the printed path, run `claude`, execute that unit's tasks (file sets are exclusive within a wave; `docs/ARCHITECTURE.md`/`README.md`/`pyproject.toml` edits only in the unit's final sync task, append-only).
3. Merge each wave into the integration branch (`integration/connector-wave<N>`) before starting the next; the GRILL reviews run on the unit branch before its merge; the hardware wave runs from `main` after OS1-HW.

This section is suggestive — the user owns the decision. See `commands/worktree.md` and `skills/worktree/SKILL.md`.

## execution_shape

```yaml
execution_shape:
  plan: plans/connector-implementation-plan.md
  phase: "AI training connector: LA0-LA5, SV1, FW1, OS1, PE1"
  units:
    - id: LP0
      name: lab-pixi-feature-and-markers
      wave: 0
      depends_on: []
      parallelizable_via_worktree: false
      tasks: [LP0.1]
      owns_files:
        - pyproject.toml                          # lab feature, integration marker, planned-task comment
        - pixi.lock
        - README.md                               # environment table row
        - tests/conftest.py                       # integration marker + skip rule
        - tests/test_env_sanity.py
    - id: LA0
      name: lab-core
      wave: 1
      depends_on: [LP0]
      parallelizable_via_worktree: true
      tasks: [LA0.1, LA0.2, LA0.3, LA0.4, LA0.5, LA0.6]
      owns_files:
        - src/so101_rl/lab/__init__.py
        - src/so101_rl/lab/paths.py
        - src/so101_rl/lab/policy.yaml
        - src/so101_rl/lab/errors.py
        - src/so101_rl/lab/schemas/**
        - src/so101_rl/lab/registry.py
        - src/so101_rl/lab/provenance.py
        - src/so101_rl/lab/audit.py
        - src/so101_rl/lab/ledger.py
        - src/so101_rl/lab/stats.py
        - src/so101_rl/lab/authority.py          # SIM mirror now; LA4/LA5 extend (sequential hand-off)
        - src/so101_rl/lab/context.py
        - src/so101_rl/lab/job.py
        - src/so101_rl/lab/cli.py
        - src/so101_rl/lab/units/**
        - src/so101_rl/lab/supervisor/**
        - src/so101_rl/lab/trainers/__init__.py
        - src/so101_rl/lab/trainers/api.py
        - src/so101_rl/lab/trainers/fake.py       # LA4 adds the real-domain behaviours (sequential hand-off)
        - src/so101_rl/viz/monitor.py             # hand-off from SUV (done): ScalarMonitor extraction
        - tests/test_lab_registry.py
        - tests/test_lab_deps.py
        - tests/test_lab_trainers.py
        - tests/test_lab_fake_trainer.py
        - tests/test_lab_job.py
        - tests/test_lab_supervisor.py
        - tests/test_lab_budgets.py
        - tests/test_lab_ledger.py
        - tests/test_lab_cli.py
        - tests/test_monitor.py
    - id: SV1
      name: servo-park-idle-resume-exit
      wave: 1
      depends_on: [LP0]
      parallelizable_via_worktree: true
      work_item: SV1
      grill_review: SV1.6
      tasks: [SV1.1, SV1.2, SV1.3, SV1.4, SV1.5, SV1.6, SV1.7]
      owns_files:
        - src/so101_rl/servo/shm.py
        - src/so101_rl/servo/core.py
        - src/so101_rl/servo/process.py
        - src/so101_rl/servo/main.py
        - src/so101_rl/servo/soak.py
        - src/so101_rl/safety/types.py
        - src/so101_rl/safety/state_machine.py
        - src/so101_rl/config/schema.py           # IdleConfig
        - src/so101_rl/config/loader.py           # idle section, check_session_device_paths
        - src/so101_rl/config/so101.yaml          # idle: section only
        - src/so101_rl/env/backend.py             # client_id, send_park, send_resume_req only
        - tests/test_shm.py
        - tests/test_types.py
        - tests/test_config.py
        - tests/test_config_single_source.py
        - tests/test_env_backend.py
        - tests/test_state_machine.py
        - tests/test_servo_core.py
        - tests/test_servo_process.py
        - tests/test_soak.py
        - tests/test_sv1_state_machine.py
        - tests/test_sv1_core.py
        - tests/test_sv1_process.py
        - tests/test_sv1_review.py
        - plans/reviews/*-sv1-review.md
    - id: FW1
      name: firmware-arm-timeout
      wave: 1
      depends_on: []
      parallelizable_via_worktree: true
      work_item: FW1
      grill_review: FW1.3
      tasks: [FW1.1, FW1.2, FW1.3, FW1.4]
      owns_files:
        - firmware/watchdog_mcu/main.py
        - firmware/watchdog_mcu/PROTOCOL.md
        - firmware/watchdog_mcu/test_vectors.json
        - firmware/watchdog_mcu/README.md
        - src/so101_rl/servo/heartbeat.py
        - tests/test_firmware_vectors.py
        - tests/test_heartbeat.py
        - plans/reviews/*-fw1-review.md
    - id: OS1-SW
      name: authority-records-tools-deploy
      wave: 1
      depends_on: [LP0]
      parallelizable_via_worktree: true
      work_item: OS1
      grill_review: OS1.5
      tasks: [OS1.1, OS1.2, OS1.3, OS1.4, OS1.5, OS1.6]
      owns_files:
        - src/so101_rl/authority/__init__.py
        - tools/so101_records/**
        - deploy/**
        - scripts/check_protected_paths.py
        - docs/protected_work_items.yaml
        - tests/test_authority.py
        - tests/test_records_tools.py
        - tests/test_deploy_files.py
        - tests/test_protected_paths.py
        - plans/reviews/*-os1-review.md
    - id: PE1-SW
      name: perception-process
      wave: 1
      depends_on: []
      parallelizable_via_worktree: true
      tasks: [PE1.1, PE1.2]
      owns_files:
        - src/so101_rl/perception/main.py         # SUC owns camera/frames/die_tracker concurrently; PE1 only calls them
        - tests/test_perception_main.py
    - id: LA1
      name: mcp-server-dashboard-unattended
      wave: 2
      depends_on: [LA0]
      parallelizable_via_worktree: true
      tasks: [LA1.1, LA1.2, LA1.3, LA1.4]
      owns_files:
        - src/so101_rl/lab/api.py
        - src/so101_rl/lab/mcp_server.py          # LA4 adds the request_real_run block (sequential hand-off)
        - src/so101_rl/lab/untrusted.py
        - src/so101_rl/lab/dashboard.py           # LA4 adds the session panel (sequential hand-off)
        - src/so101_rl/lab/templates/**
        - .mcp.json
        - scripts/lab_unattended_fake_run.sh
        - tests/test_lab_mcp.py
        - tests/test_lab_contract.py
        - tests/test_lab_dashboard.py
        - tests/test_lab_unattended.py
    - id: LA2
      name: fastsac-pipelines
      wave: 2
      depends_on: [LA0, SU3, SU4]                 # SU3/SU4 are sim-track units, not yet built
      gated: true
      parallelizable_via_worktree: true
      tasks: [LA2.1, LA2.2, LA2.3]
      owns_files:
        - src/so101_rl/lab/trainers/fastsac_sim.py
        - src/so101_rl/lab/schemas/trainer/fastsac_sim.v1.schema.json
        - src/so101_rl/rl/train_sim_sac.py        # hand-off from SU4: additive run_training(hooks)
        - tests/test_lab_fastsac.py
        - tests/test_lab_unattended_fastsac.py
        - plans/reviews/*-la2-unattended-run.md
    - id: LA4-SW
      name: real-arm-sessions-tier0-fake
      wave: 3
      depends_on: [LA1, SV1, FW1, OS1-SW, PE1-SW]
      parallelizable_via_worktree: true
      tasks: [LA4.1, LA4.2, LA4.3]
      owns_files:
        - src/so101_rl/lab/session.py
        - src/so101_rl/lab/mcp_server.py          # hand-off from LA1: request_real_run block only
        - src/so101_rl/lab/dashboard.py           # hand-off from LA1: session panel only
        - src/so101_rl/lab/authority.py           # hand-off from LA0: REAL Tier 0 mirror
        - src/so101_rl/lab/trainers/fake.py       # hand-off from LA0: real-domain behaviours
        - tests/test_lab_session.py
        - tests/test_lab_authority_mirror.py
        - tests/test_lab_real_fake.py
    - id: LA3
      name: s5-plugins-tier1-flag
      wave: 3
      depends_on: [LA2, SU5, SR2]                 # SU5/SR2 are sim-track units, not yet built
      gated: true
      parallelizable_via_worktree: true
      tasks: [LA3.1, LA3.2]
      owns_files:
        - src/so101_rl/lab/trainers/sb3_sac_her.py
        - src/so101_rl/lab/trainers/tdmpc2_offline.py
        - src/so101_rl/lab/trainers/tdmpc2_finetune.py
        - src/so101_rl/lab/schemas/trainer/sb3_sac_her.v1.schema.json
        - src/so101_rl/lab/schemas/trainer/tdmpc2_offline.v1.schema.json
        - src/so101_rl/lab/schemas/trainer/tdmpc2_finetune.v1.schema.json
        - src/so101_rl/rl/deploy_policy.py        # hand-off from SU5: --tier1 flag only
        - src/so101_rl/rl/train_sac_her.py        # hand-off from SU5: --tier1 flag only
        - src/so101_rl/rl/train_tdmpc2.py         # hand-off from SU5: --tier1 flag only
        - src/so101_rl/sim/sysid/run_excitation.py  # hand-off from SU2: --tier1 flag only
        - tests/test_lab_plugins_s5.py
    - id: LA5-SW
      name: tier1-mirror
      wave: 4
      depends_on: [LA4-SW]
      parallelizable_via_worktree: false
      tasks: [LA5.1, LA5.2]
      owns_files:
        - src/so101_rl/lab/tier1.py
        - src/so101_rl/lab/authority.py           # hand-off from LA4: Tier 1 mirror
        - src/so101_rl/lab/session.py             # hand-off from LA4: tier-1 attach claims
        - tests/test_lab_tier1.py
    - id: HW
      name: hardware-gated-connector-track
      wave: "H (interleaved; OS1-HW after wave 1; FW1-HW with H3; HS8 after H2/H3/H5/H7/H11 + OS1-HW; HS9 after H10 + HS8; LA4-HW after LA3 + LA4-SW + HS8; LA5-HW after H13 + HS9 + a clean attended session)"
      depends_on: [OS1-SW, SV1, FW1, LA4-SW, LA5-SW, LA3]
      parallelizable_via_worktree: false
      tasks: [OS1-HW, FW1-HW, HS8, HS9, LA4-HW, LA5-HW]
      owns_files:
        - runs/hw/**                              # gitignored
        - runs/sessions.jsonl                     # gitignored
        - plans/reviews/*-la4-first-real-use.md
        - plans/reviews/*-la5-first-unattended-session.md
  waves:
    - {wave: 0, units: [LP0]}
    - {wave: 1, units: [LA0, SV1, FW1, OS1-SW, PE1-SW]}
    - {wave: 2, units: [LA1, LA2]}
    - {wave: 3, units: [LA4-SW, LA3]}
    - {wave: 4, units: [LA5-SW]}
    - {wave: "H", units: [HW]}
  shared_writes:
    - path: docs/ARCHITECTURE.md
      rule: "Updated at the END of every unit (LA0.6, LA1.4 (v0.18), SV1.7, FW1.4, OS1.6, PE1.2, LA2.3, LA3.2, LA4.3, LA5.2). Append-only edits to §3/§4 lists, §5/§6 tables and the §10 changelog; bump the version line. Never edited inside a unit task other than its sync task. Section 6 item 7 'Connector authority' labels every guard rail as such."
    - path: pyproject.toml
      rule: "LP0 owns it in wave 0 (lab feature, markers, planned-task comment). Later units add ONLY the pixi task for a module they created, in their sync task (tests/test_pixi_tasks.py); the integrator applies dependency additions between waves."
    - path: pixi.lock
      rule: "Regenerated by LP0 only; any later lock change is an integrator action between waves."
    - path: README.md
      rule: "LP0 rewrites the environment table rows for the lab packages. Later units append one line under 'Entry points' per new pixi task in their sync task."
    - path: so101-self-learning-rl-plan.md
      rule: "Only the master edits it: §12 v3 text (after LA1), and §10 rows for decisions D1–D9 the user confirms. No unit edits it; the build never adds or alters a §10 row."
    - path: plans/connector-implementation-plan.md
      rule: "This file. Status notes appended by the integrator after each wave; frozen-interface changes edited here together with docs/ARCHITECTURE.md."
    - path: plans/ai-connector-design.md
      rule: "As-built notes only (appended by the integrator after wave 1 and after LA4-SW); the design text is not rewritten."
    - path: docs/HARDWARE_CHECKLIST.md
      rule: "Append-only: SV1.7 (HS8), FW1.4 (H3 step 10), OS1.6 (OS1 install section + safety bullet), LA4.3 (session-opening procedure), LA5.2 (HS9, LA5-HW). The HW track appends a results-table row per artifact."
    - path: docs/protected_work_items.yaml
      rule: "Created by OS1.4; later rows (new work-item ids) only by the integrator."
    - path: .claude/settings.json
      rule: "NOT written by any unit. deploy/README.md documents the suggested deny rules; the user applies them by hand (a guard rail against honest mistakes, P10)."
```

## Decisions made in this plan that the user may want to confirm

1. **D1 — Authority check runs synchronously inside the servo tick** (first RUN/RESUME_REQ of a new `client_id`, arm parked or torque off): heartbeat sent first, `ssh-keygen -Y verify` subprocess with a **120 ms timeout** (worst-case heartbeat gap ≈ 153 ms < 200 ms, < 5 missed acks), a BOOT-time warm-up, a 1 s denial back-off, duration recorded. The consulted specialist prefers an asynchronous SCHED_OTHER worker thread (no gap at all, but a thread + GIL in the gc-frozen RT process); switch to it only if HS8 measures gaps > 200 ms.
2. **D2 — Servo exit codes:** `5` = terminal fresh-process/human trip under `--exit-on-terminal`, `6` = LATCHED; `0` stays for `--max-ticks`/`STOP_REQ`/signals (`servo_clean` requires 0).
3. **D3 — `so101_rl.authority` is its own stdlib-only package** (snapshotted next to servo/safety/config; protected path `Work-Item: OS1`; installed verbatim as `/usr/local/lib/so101_rl/so101_authority.py` for the records tools; the lab imports it for mirrors). The design's RT0 is folded into OS1-SW as the user asked; a **dedicated GRILL review (OS1.5) is recommended** because it is trust-root code.
4. **D4 — Snapshot env built by a root-run `pixi install --locked -e robot`** inside `/opt/so101_rl/<sha>/` (allow-listed tree from `git archive`), `env -> .pixi/envs/robot`; reproducible from `pixi.lock`, no prefix rewriting (needs network or a warm pixi cache once per snapshot). Alternative: `conda-pack` of the dev env.
5. **D5 — LA4 and LA5 are split into -SW (fake servo backend + the real authority module on a temp store; buildable after wave 1) and -HW (gated exactly as the user listed: LA4-HW on LA3, SV1, FW1, PE1, OS1, H-track; LA5-HW on H13, HS9, HS8, a clean attended session).**
6. **D6 — Shared-memory layout v3:** `STATE_DTYPE` grows 232 → 248 B (stride 256 and segment size unchanged; `attach` refuses v2); tick-log fields appended (old tick logs are not readable by the new dtype; the summary records the record size).
7. **D7 — Firmware line budget** may rise 95 → 100 if the three ARM_TIMEOUT lines overflow it (expected 94–95); the reason goes in the test docstring as for the 80 → 95 bump.
8. **D8 — MCP server ↔ supervisor via files only** (`runs/queue/`, `runs/supervisor/heartbeat.json`; no socket/D-Bus), and the MCP server is launched with `pixi run --frozen -e cpu lab-mcp` (no separate `lab` env; the import guard is test-enforced).
9. **D9 — New hardware id HS9** for the 120-min goal-directed strict soak **with ≥ 2 parked segments** (P5's "≥ 120-min strict soak verified at promotion"), run with `soak --park-every-min/--park-for-s`; its report sha goes into the tier record. **The shipped `so101.yaml` ports stay unchanged**; the by-id device paths live only in the pinned unit argv (`--port /dev/serial/by-id/so101-arm --heartbeat-port /dev/serial/by-id/so101-mcu`), so `servo-fake` keeps working on any host.

## Gaps and notes for the master orchestrator

- **The working tree switched branches during planning:** the session started on `integration/sim-wave1` (b109e18); HEAD is now `feature/camera-parity` (bf7346b, a merge that contains b109e18) with the untracked `src/so101_rl/servo/probe.py` of the camera-parity unit (SUC). All inputs read are identical to their committed versions. This plan file is untracked and unaffected; the master should commit it on the branch it chooses.
- **No specialist exists for signed-record / account-separation security or for systemd process supervision** (logged in `system-improvements.md`); the MCP architect and the RT-control specialist were consulted instead. Their reports (verbatim) are in the session log.
- **`mcp 2.3.0` API names were verified in a throwaway `pixi exec` env**, not in the project envs (LP0 installs them). `Client(server)` in-memory testing, `structured_output=False`, resource templates and `rebind_std_handle_to_fd` exist; nothing in this plan relies on `mcp.server.fastmcp`.
- **LA2/LA3 depend on sim-track units that are not built** (SU3, SU4, SU5, SR2). LA2 additionally needs an additive library entry in SU4's `rl/train_sim_sac.py` (`run_training(cfg, hooks=)`); the sim-track plan's S4.3 should grow that hook when SU4 is built, or LA2 adds it after SU4 merges (listed as LA2's owned hand-off).
- **koen is in `dialout` today;** OS1-HW removes it, after which `servo --backend feetech` from the user account stops working by design (the design's intent). H-track procedures that start the servo by hand (HS1 etc.) will then run through the unit; `docs/HARDWARE_CHECKLIST.md` HS1–HS3 text should be amended by OS1.6 to say so.
- **`CONFIG_RT_GROUP_SCHED` is not set** on this kernel: the design's slice `cpu.rt_runtime_us` step is skipped; `LimitRTPRIO=50` in the unit suffices for `SCHED_FIFO` (verify in OS1-H4 with `rt_active: true` in a test session).
- The dashboard, supervisor and MCP server run in the `cpu` env (torch is present but never imported by them; AST + subprocess tests pin it). A minimal `lab` env was considered and rejected to keep the env count at three (ARCHITECTURE §8).
- `tests/test_lab_unattended.py` creates real transient user units named `so101-test-*`; they are `--collect`ed, and the test's teardown runs `systemctl --user stop 'so101-test-*'` defensively.
