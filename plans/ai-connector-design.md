# AI training connector (`so101_rl.lab`) — design

> **Version 3 (2026-10-07).** v1 (481cd2f) → plan critic → v2 (31b2fa8) → security and safety specialist panels → the user's §10 rows **P10 (authority enforcement with two Linux accounts)** and **P11 (approval hardening)** (d671348). v3 folds in P10/P11 and every safety-panel finding (1–11); where a finding is shaped by a user decision this is stated. **Design only; nothing in this file is implemented.** The user asked: *"Update this plan/package so that it can be used by an AI to train various models. Make it like a connector or MCP. For now focus on RL models; other models should be possible in the future."*
>
> **Binding user answers (plan §10, 2026-10-07; summarised, the rows win over this text):**
> - P1 connector: `so101_rl.lab` + thin stdio MCP (12 tools in LA1); detached jobs under a `systemd --user` supervisor; read-only local dashboard; only FastSAC, SB3 SAC+HER, TD-MPC2 implemented; done = unattended sim runs + unattended real runs after Tier 1.
> - P2/P10/P11 human records: approvals, S4-exit ticks, acks and the tier record are Ed25519-signed; **two dedicated Linux accounts**: `so101-approver` signs only from a text-console login with a scrubbed environment; `so101-servo` solely owns the arm and MCU serial devices and runs a root-owned, human-reviewed **snapshot** of `servo/` + `safety/` + config as a **system unit**, verifying the signed records itself before ARM, before `RESUME_REQ` and before the first RUN from a new client, and keeping the consumed-nonce ledger and the authoritative session id. The editable lab/supervisor/MCP can only fail to schedule; they cannot move the arm unsigned. Claude Code permission rules, env flags and `WriteGuard` are **guard rails against honest mistakes**, labelled so throughout. Challenge word, strict grammars, anti-rollback, namespaces, expiry sanity, ≤ 1 open request per kind; no hardware security key for now.
> - P3 SV1 (servo park / idle / resume), P4 FW1 (firmware arm timeout 120 s; `--arm-timeout-s 110`; at Tier 1 the connector may start the servo inside that window), P5 Tier 1 session rules (cap ramp 60 → 120 → 240 → 480 after clean sessions; power-cycle + weld check; quiet hours 23:00–07:00; 2-min gap; allow-list; 240 arm-min/day ending at local midnight; ≥ 120-min strict soak verified at promotion and every admission), P6 budgets, P7 review triggers, P8 thermal fail-closed, P9 proposals inbox.
> - Constraints: `safety/`, `servo/`, `firmware/` change only through SV1/FW1; the connector respects the dependency rules; ARCHITECTURE in the same commit as any module change.
>
> **Sources of truth it obeys:** plan §10 (all rows incl. 2026-10-06/07 "Re-energise start floor value" = tip z ≥ 0.010 m base frame, "Gripper floor clearance" = lowest wrist/gripper point ≥ table + 1 mm with a 2 mm BOX_PRESENT tolerance, box floor −0.015 m), ARCHITECTURE v0.17, the sim-track plan, the Phase 0 plan, `firmware/watchdog_mcu/PROTOCOL.md`, `safety/state_machine.py`, `servo/{shm,main,soak}.py`.
>
> **Hardware rule:** no process of the user's account opens the arm or MCU serial devices; only `so101-servo` can (udev). The connector contains no bus code.

## 0. The design in one paragraph

`so101_rl.lab` is the **Python API** (run registry under `runs/`, four RL trainer plugins, task registry, authority checks, derived budget ledger, audit, a `systemd --user` **supervisor** that schedules detached job **segments as transient user units**, a real-arm **session manager**, a read-only **dashboard**); `lab.mcp_server` is a thin stdio adapter with 12 tools. Real-arm authority is **not** in the lab: the servo runs as `so101-servo` from a root-owned snapshot as a system unit, owns the serial devices, and verifies the signed human records (made by `so101-approver` from a text console) at ARM, at `RESUME_REQ` and at every new client's first RUN; the lab's identical checks only decide what to *schedule*. Two decided changes outside the connector make unattended sessions possible and safe: SV1 (park at `home`, idle rest + torque off, resume-on-request, every re-energise landing in parked HOLD, exit on terminal trips) and FW1 (120 s arm timeout). Sessions are human-opened (button), bounded (`--max-ticks`, cap ramp), end in the safe state and in a servo exit so the relay opens, and are judged clean in two phases (servo-clean at exit, weld-ok at the next ARM). The connector adds no safety layer and removes none.

## 1. Layering

### 1.1 Layers, accounts, dependency direction

```
koen (user account)                                                  │ so101-approver (console login only)
┌──────────────────────────────┬──────────────────────────────┐      │  /home/so101-approver (0700): signing key
│ Claude Code (stdio client)   │ Browser → 127.0.0.1:6007     │      │  so101-approve / gate-sign / ack  (root-owned shims)
└──────────────┬───────────────┴──────────────┬───────────────┘      │  → /var/lib/so101_rl/records/ (approver-owned, koen r/o)
┌──────────────▼───────────────┐ ┌────────────▼───────────────┐      │
│ lab.mcp_server (12 tools)    │ │ lab.dashboard (GET only)   │      │ so101-servo (system unit, snapshot)
└──────────────┬───────────────┘ └────────────┬───────────────┘      │  /opt/so101_rl/<sha>/ (root-owned): servo/, safety/,
┌──────────────▼──────────────────────────────▼───────────────┐      │    config/, authority helper, frozen robot env
│ so101_rl.lab  (editable; fails to schedule, never moves)    │      │  /dev/serial/by-id/* (udev: owner so101-servo 0600)
│ registry · trainers · tasks · authority(mirror) · budgets   │      │  /var/lib/so101_rl/servo/ : nonces.jsonl, sessions/,
│ jobs (supervisor, transient units) · session manager · audit│      │    summaries, tick logs, arm-minute ledger
└──────────────┬──────────────────────────────────────────────┘      │  verifies records before ARM / RESUME_REQ / first RUN
┌──────────────▼──────────────────────────────────────────────┐      │
│ rl/ wm/ env/ sim/ viz/ logging/ config/ perception/          │      │ shared: /dev/shm/<servo segment> (group so101, 0660)
└──────────────┬──────────────────────────────────────────────┘      │         /var/lib/so101_rl/attach/<session>/ (group so101)
               │ shm (ShmBackend, object table); attach files        │         runs/ (group so101, readable by both accounts)
┌──────────────▼──────────────┐   ┌──────────────────────────┐       │
│ servo (snapshot, so101-servo)│──►│ safety (snapshot)        │ ◄─────┘
└─────────────────────────────┘   └──────────────────────────┘
```

Dependency arrows for ARCHITECTURE §4 (unchanged from v2 except the account boundary): `lab → rl, wm, env, sim, viz, logging, config, perception (launch only)`; `lab → servo.shm` (attach; command writes only by the attached job under the cmd-owner lock and with its `client_id`); `lab → safety.types` (constants); `lab.mcp_server`, `lab.dashboard → lab` only; nothing imports `lab`. The lab never imports `servo.main`, `servo.core`, `servo.feetech_bus`, `safety.pipeline/clamp/safe_state` or `pyserial` (AST tests). The lab starts the servo only through `systemctl start so101-servo@<session_id>` (polkit rule, §5.2), never by exec.

### 1.2 Why a Python API with a thin MCP on top; why the authority is outside the lab

The lab's checks are mirrors that make the AI's experience precise (refusals with reasons) and cheap (nothing is scheduled that the servo would refuse). The servo's checks are the ones that hold: they run as another user from root-owned code the user account cannot alter, on records the user account cannot forge. Pattern: the hybrid of `05-Wiki/concepts/MCP-Server-Pattern-Choice.md` (stateless server, spawned training); no `execute_script` (`Hardware-Test-Harness-Pattern.md`).

## 2. Trainer plugin interface (unchanged from v2 except where marked)

### 2.1 Contract (FL1–FL3, `so101_rl/lab/trainers/api.py`)

```python
Domain = Literal["sim", "real", "offline"]
@dataclass(frozen=True)
class Capabilities: domains: frozenset[Domain]; obs_layouts: tuple[str, ...]; action_space: str; inputs: tuple[str, ...]; outputs: tuple[str, ...]
    gpu: Literal["required","optional","none"]; vram_gb_estimate: float; resumable: bool; evaluates: bool; exports: bool
    role: Literal["core_rl","accelerator"]; env_name: Literal["default","cpu","robot"]
@dataclass(frozen=True)
class JobPlan: domain: Domain; leases: frozenset[Literal["gpu","arm","cpu"]]; budget_key: str; budget_class: Literal["attempt","probe","eval"]
    gpu_h_estimate: float; arm_min_estimate: float; disk_gb_estimate: float; requires: tuple[str, ...]; resumable_fields: tuple[str, ...]; unattended_ok: bool
class TrainerPlugin(Protocol):
    name: str; version: str; config_version: int; capabilities: Capabilities
    def config_schema(self) -> dict; def validate(self, cfg) -> dict; def plan(self, cfg, task, init_from) -> JobPlan
    def train(self, ctx) -> None; def resume(self, ctx) -> None; def evaluate(self, ctx, eval_cfg) -> EvalReport; def export(self, ctx) -> ArtifactRef
    def artifacts(self, run_dir) -> ArtifactManifest
class RunContext:  # segment, run_dir, cfg, task, device, vram_fraction, budget (own hard deadline), monitor (ScalarMonitor)
    def heartbeat(self, *, step=None, phase="train") -> None   # progress-coupled, from inside the loops, never a thread
    def should_stop(self) -> bool; def should_pause(self) -> bool; def checkpoint(self, path, *, step) -> None
    def event(self, kind, **payload) -> None; def review_trigger(self, name, **details) -> None
    def servo(self) -> ServoView | None   # real runs: state record (incl. parked, arm_ns = session id), summary; read-only
    def limits(self) -> RunLimits           # v3: episodes, arm_min_cap, speed_scale_max — taken ONLY from the verified signed record (Tier 0) or the signed tier rules (Tier 1), never from run.json
    def record_usage(self) -> None
```

Strict JSON-Schema configs (2020-12, `additionalProperties: false`, bounds, `const config_version`), validated with `jsonschema` into frozen dataclasses; schemas served as resources. Not configurable by any plugin: thermal thresholds, caps, halt rules, speed ladder, tier rules, `tensorboard`, the metrics contract, DR ranges. Static in-repo plugin table; no entry points.

### 2.2 Plugins (as v2)

`fastsac_sim` (S4.3/S4.2/SU3/viz; sim; `s4.<task>`), `sb3_sac_her` (S5.1/S5.2; sim via `SimBackend` or `ShmBackend` against a `servo --backend mujoco` child, real via `ShmBackend`; `s5.<task>.finetune.sb3`), `tdmpc2_offline` (D-TDMPC2-LOOP; offline; `dataset {id | from_run}` auto-registers), `tdmpc2_finetune` (sim/real), `fake` (tests; behaviours incl. `servo_trip:<cause>`, `late_attach`). The real-arm entry points gain `--tier1 <tier record>` (a `rl/` + `sim/sysid` change) replacing `--attended` for unattended runs; they take run limits from `ctx.limits()`.

### 2.3 Metrics and artifact contract (as v2)

`metrics.csv` (source of truth; `env_step|update, wall_s, gpu_h_used, viz_disabled` + plugin scalars), TensorBoard via `ScalarMonitor` (extracted from FS11's `TrainingMonitor`), `review.json`, `ckpt_*.pt`, `eval.json` + `renders/` + `review_checklist.md`, `gate.json` (numeric only), `usage.json`, `artifacts.json` (failed checker → `failed`, rc notwithstanding).

## 3. Run lifecycle and job runner

### 3.1 Process model

| Process | Account / unit | Lifetime | Owns | Must never |
|---|---|---|---|---|
| `lab-mcp` | koen; child of Claude Code (stdio) | the session | nothing durable | import torch/mujoco/serial; hold the GPU; write `calib/`, `config/`, the plan, any record |
| `lab-supervisor` | koen; `systemd --user` service `so101-lab-supervisor.service` (`Restart=on-failure`, linger enabled, **`KillMode=process`** so a restart never touches children; jobs are separate units anyway) | always | queue, leases (`gpu` ×1, `arm` ×1, `cpu` ×2), ledger cache, `gpu_health`, stall ladder, pipelines, needs-human queue, session manager, notifications | sign anything; write the shm command block while a job holds the cmd-owner lock; exec the servo |
| job segment | koen; **transient user unit** `so101-job-<run_id>-<seg>.service` via `systemd-run --user --unit … --collect -p KillMode=mixed -p TimeoutStopSec=130 [-p MemoryMax=…]` running `pixi run -e <env> python -m so101_rl.lab.job --run runs/<id>`; the job records `os.getpid()` and its unit name | one segment | its `run_dir`, its CUDA context, its own hard deadline, disk quota, its `client_id` | write outside `run_dir` (except registering through `lab`); outlive its deadline |
| perception | koen; transient user unit `so101-perception-<session_id>.service` (camera via group `video`) | the session | the D435, the object table, the fiducial verdict | — |
| `lab-dashboard` | koen; `systemd --user` service | always | nothing (GET only, 127.0.0.1:6007) | bind elsewhere; POST; write |
| **servo** | **`so101-servo`; system unit `so101-servo@<session_id>.service`** from the snapshot (§5.2) | one session, bounded by `--max-ticks` | the serial devices, every safety check, the heartbeat, record verification, the nonce ledger, the session id (`arm_ns`) | be started with `--no-mcu`; take limits from anything unsigned |
| `so101-approve` / `so101-gate-sign` / `so101-ack` | `so101-approver`; text console | one interaction | the signed record in the approver store | run from su/sudo or a pty; execute repo code |

Supervisor restart safety (panel 3): children are their own units, so a supervisor restart (`systemctl --user restart`) kills nothing; re-adoption is by unit name (`systemctl --user show -p MainPID,ActiveState so101-job-*`) plus `status.json` pid/`boot_id`. Acceptance A4 runs under a real `systemd --user` (`@pytest.mark.integration`).

### 3.2 Run identity and layout (as v2, plus)

`runs/<run_id>/attach.json` (the job's attach claim, copied to `/var/lib/so101_rl/attach/<session_id>/<client_id>.json`, §5.6), `runs/<run_id>/limits.json` (the limits the job actually used, copied from the verified record), `runs/sessions.jsonl` (two-phase verdicts). `runs/` is group `so101`, `g+rX`, so the approver and servo accounts can read requests and run manifests; nothing of theirs is writable by koen. Idempotency: `runs/idem/<sha256(key)>` (O_EXCL) + `flock runs/.registry.lock`.

### 3.3–3.8 Scheduling, pauses, ledger, thermal, review triggers, unattended operation

Unchanged from v2 in substance (P6–P8 are now logged): a pause is a checkpoint + process exit; thermal/health pauses hold the GPU lease and auto-resume; review/AI pauses release leases; FIFO with `arm` first; queue TTLs (24 h sim / 4 h real); reboot (`boot_id`) voids queued real entries; suspend (`CLOCK_BOOTTIME − CLOCK_MONOTONIC` jump > 5 s) → `host_slept`, approvals void, session revalidated; ledger derived from `usage.json` with a high-water mark (shrink → `HALT`); GPU-h = progress-holding time with a 30-min stall cap; stall ladder 30/45/50 min; the job's own hard deadline; running disk check and per-run quotas; NVML sampling (GPU 85/75 °C, memory 95/85 °C, six empty samples → pause all + item); pipelines train → eval → numeric gate → export; the S4 exit gate stays half human (signed `so101-gate-sign`); the needs-human queue (kinds as v2 plus `servo_exit_failed`, `client_refused`, `device_unresolvable`). The vault's auto-curriculum/recipe loops are not adopted (plan §10 wins).

## 4. MCP surface (as v2)

12 tools in LA1 — `get_status`, `find_runs`, `get_run` (with `wait`), `get_metrics`, `check_gates`, `start_run`, `control_run`, `evaluate_policy`, `export_policy`, `run_gates`, `write_note`, `propose_decision`; LA4 adds `request_real_run` behind `SO101_LAB_REAL_ARM=1` (**a guard rail, not a boundary**). Registries, events, decisions, needs-human, sessions as resources; three prompts; structured tool errors (+ `SESSION_RULE`, `RECORD_INVALID`, `DEVICE_UNRESOLVABLE`); untrusted-data envelopes on every AI-written or log field; start → poll (`get_run(wait=)` ≤ 300 s) + subscriptions; deferred tools unchanged. `get_status.arm` now reports the servo unit state, `arm_ns`, `parked`, cap/elapsed, the snapshot sha, the two-phase verdict of the last session, and the records store's current tier pointer.

## 5. Safety and authority model

Principle: what the AI may do is decided by code below the tool boundary **in another account**; what the arm may do is decided by ServoCore, the MCU and the e-stop. The connector adds no safety layer and removes none. Everything the user account can set (`.claude/settings.json` ask/deny rules, `SO101_LAB_REAL_ARM`, `WriteGuard`, the lab's mirrored checks) is a **guard rail against honest mistakes** and is labelled so wherever it appears.

### 5.1 Authority table (the enforcing party in **bold**)

| Action | SIM / offline | REAL, Tier 0 | REAL, Tier 1 |
|---|---|---|---|
| Start a sim/offline job | **lab**: config valid; L1+L2 PASS for S4; `sim_params` rule; ledger; disk; not `HALT`; GPU health known | n/a | n/a |
| Open a session (start the servo unit, ask for the button) | — | lab mirror: hw artifacts fresh, a valid signed tier record (tier ≥ 0), devices resolvable (§5.6), RT limits available; **servo (pre-ARM)**: valid, current, unrevoked signed tier record whose embedded sha256s of the hw artifacts and soak reports match the files; session request within the record (cap ≤ record cap); device identities by-id; else exit 2 before waiting for the button | as Tier 0 plus `tier == 1`, quiet hours, per-day arm minutes from the **servo's own ledger**, cap ramp rung |
| `request_real_run` | `TIER_FORBIDDEN` | lab mirror → `awaiting_approval` (≤ 1 open request per kind) | lab mirror: rules; → `queued` or `awaiting_approval` when outside the rules |
| First RUN / `RESUME_REQ` from a new client | — | **servo**: attach claim names an approval record; signature (namespace `so101-approval`), type, request hash recomputed over `run.json`/config/policy/hw artifacts, expiry sanity, `session_id == arm_ns`, nonce unconsumed (servo ledger) → consume; limits = the record's | **servo**: tier record valid; attach claim's `run.json` kind/task/trainer/policy sha256 in the allow-list; quiet hours; min gap; per-day minutes; cap; limits = the rules' |
| `control_run(stop)` on a real job | — | job sends `STOP_REQ`; the servo exits (§5.6) | same |
| Resume a tripped real run | — | a new approval **and** a signed ack of the item | a signed ack of the item, then rules |
| Write config/calib/plan/records/tier/§10 | never (lab `WriteGuard`, a guard rail; the records store is not writable by koen — a boundary) | never | never |

### 5.2 Accounts, stores, snapshot, units (work item **OS1**)

- **`so101-approver`**: home `0700`; key `~/.ssh/so101_approver_ed25519` generated **by hand as that user** with `/usr/bin/ssh-keygen -t ed25519 -a 256 -C so101-approver` and a diceware passphrase; never `ssh-add`. Signs only from a **Linux text-console login** (`/dev/tty[1-9]`): the shims refuse when `os.ttyname(0)` is a `/dev/pts/*`, when `SUDO_UID`/`SUDO_USER`/`SUDO_COMMAND` is set, or when `os.getuid()` is not the approver uid. Shims: `/usr/local/bin/so101-approve` = `exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=<approver home> TERM="$TERM" /usr/bin/python3 -I -E -s /usr/local/lib/so101_rl/approve.py "$@"`; `SSH_AUTH_SOCK`, `SSH_ASKPASS`, `DISPLAY`, `WAYLAND_DISPLAY` are unset by `env -i`; the key path comes from `pwd.getpwuid(os.getuid()).pw_dir`; `ssh-keygen` is called by absolute path. Records go to **`/var/lib/so101_rl/records/`** (approver-owned, `0755`/`0644`, koen read-only): `approvals/`, `gates/`, `tiers/`, `acks/`, `current/` (pointer files), `revoked.jsonl` (signed). Reading koen's `runs/<id>/approval_request.json` is by group `so101` read access.
- **`so101-servo`**: sole owner of the arm and MCU devices via udev (`SUBSYSTEM=="tty", ATTRS{serial}=="<arm adapter serial>", SYMLINK+="serial/by-id/so101-arm", OWNER="so101-servo", MODE="0600"`; likewise `so101-mcu`); **koen is removed from `dialout`**. Runs **a root-owned snapshot** `/opt/so101_rl/<git sha>/` = `src/so101_rl/{servo,safety,config}/` + the stdlib authority helper `so101_authority.py` + a frozen copy of the `robot` pixi environment, all root-owned, `0755`; `current` symlink. System unit **`so101-servo@.service`** (template, instance = session id): `User=so101-servo`, `Group=so101`, `LimitRTPRIO=50`, `LimitMEMLOCK=infinity`, `CPUAffinity=<servo cpu>`, `KillMode=mixed`, `KillSignal=SIGTERM`, `SendSIGHUP=no`, `TimeoutStopSec=<bound_s + 10>` (one SIGTERM, then SIGKILL — the runner treats a *second* signal as escalation to torque-off in place, so systemd must never send two), `ExecStart=/opt/so101_rl/current/env/bin/python -m so101_rl.servo.main --backend feetech --session /var/lib/so101_rl/attach/sessions/%i.json --records /var/lib/so101_rl/records --state-dir /var/lib/so101_rl/servo --require-rt --exit-on-terminal` (the **pinned argv lives in the root-owned unit**, not in lab code; the variable parts are in the session request file and are validated by the servo against the signed tier record). A **polkit rule** lets koen `start`/`stop`/`status` only `so101-servo@*.service`. `RT_GROUP_SCHED`: the unit's slice gets `cpu.rt_runtime_us` when the kernel has `CONFIG_RT_GROUP_SCHED`; the servo's existing `rt_active` reporting stays the truth. The servo writes its own pid, `arm_ns`, first/last MCU status and the summary into `/var/lib/so101_rl/servo/sessions/<id>/` (group-readable).
- **Install and update are human-run, reviewed procedures:** the install scripts are read in full at a pinned commit before `sudo`; reference sums of every root-owned file and of the snapshot tree are in root-owned **`/etc/so101_rl/SHA256SUMS`** (`sha256sum -c` is the first step of every session-opening procedure in `docs/HARDWARE_CHECKLIST.md`). Snapshot update = review `git diff <old sha>..<new sha> -- src/so101_rl/{servo,safety,config}`, `pixi run -e cpu test` green at the new sha, build to `/opt/so101_rl/<new sha>/`, refresh `SHA256SUMS`, switch `current`, record in the checklist. SV1 and FW1 checks therefore live in the snapshot (servo + helper) and in the firmware; the lab's copies are mirrors.
- `/etc/so101_rl/allowed_signers` (root-owned): one line `so101-approver namespaces="so101-approval,so101-gate,so101-tier,so101-ack" ssh-ed25519 <pubkey>`.
- Shared: group `so101` (koen, so101-approver, so101-servo); `/dev/shm/<servo segment>` created by the servo `0660` group `so101`; `/var/lib/so101_rl/attach/` (`2775` group `so101`, sticky).

### 5.3 Signed records (P2, P11)

**Common body** (canonical JSON, sorted keys, ASCII only): `type` ∈ {`approval`, `gate`, `tier`, `ack`}, `schema: 1`, `seq` (monotonic per type; tier and gate also carry `supersedes_sha256` of the previous record), `created_at` (RFC 3339 with offset; TZ from `/etc/localtime`), `signer: so101-approver`, type fields. Signature: `ssh-keygen -Y sign -f <key> -n so101-<type>` over the exact bytes written; verification `ssh-keygen -Y verify -f /etc/so101_rl/allowed_signers -I so101-approver -n so101-<type>`; a record whose namespace does not match its `type` is invalid. **Anti-rollback:** `current/<type>` pointer files (approver-written) name the current tier/gate sha; `revoked.jsonl` lists revoked shas (each line signed); a verifier accepts tier/gate records only if current, unrevoked and `seq` ≥ the highest seen (the servo persists the highest seen in its state dir).

**`so101-approve <run_id>`:** TTY/console/uid checks (§5.2) → open `runs/<run_id>/approval_request.json` **once** with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`, `fstat` (regular file, owner koen, size ≤ 64 kB), read the bytes → **strict validation**: every field by ASCII grammar and type (`run_id` `^[a-z_]+-[a-z0-9_]+-[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$`, `kind` enum, `task` enum, `episodes` int 1..100, `speed_scale` ∈ {0.5, 0.75, 1.0}, `arm_min_cap` int 1..120, sha256 hex, paths under `runs/` or `calib/` without `..`); **any** byte outside `0x20..0x7E` (plus `\n` between lines) → reject, never sanitise → render **machine fields first**, then a **highlighted diff** against the last approved request of the same kind (added/removed/changed values), **directly above the prompt**; AI text (`ai_notes`) is escaped and boxed under "AI-written text (untrusted)", `--no-ai-text` is the default and `--show-ai-text` opts in → **challenge word** `CHAL` (5 random lowercase letters from `os.urandom`, shown once, line directly above the prompt) → the human types exactly **`APPROVE <run_id> <episodes>x<speed_scale> <CHAL>`** → body `{type: approval, seq, run_id, request_sha256 (over the bytes read), config_sha256, policy_sha256, hw_artifacts: [{id, path, sha256}], approved_at, expires_at (= approved_at + policy.expiry_h, ≤ 2 h), nonce (128-bit), session_id (the servo's `arm_ns` read from the state record: the session must be open), limits: {episodes, arm_min_cap, speed_scale_max}, ai_text_shown: bool}` → written `O_WRONLY|O_CREAT|O_EXCL` to `records/approvals/<run_id>.<seq>.json` + `.sig`. **Expiry sanity at verification:** `expires_at − approved_at ≤ policy.expiry_h`; `approved_at ≤ now + 60 s`; `now < expires_at`. **≤ 1 open approval request per kind** (the lab refuses a second `request_real_run` of the same kind while one is `awaiting_approval`). Verification recomputes the sha256 of `run.json`, the config, the policy file and every listed hw artifact at lab admission, at job start and in the servo's attach check; a mismatch is `RECORD_INVALID`.

**`so101-gate-sign <eval_run_id>`** (`type: gate`, `seq`, `supersedes_sha256`, `eval_sha256`, `checklist_sha256`, `renders_manifest_sha256`, `success`, `wilson_lo`; phrase `SIGN <eval_run_id> <success_pct> <CHAL>`), **`so101-gate-sign --tier <tier.json>`** (`type: tier`, body = the full tier record incl. `rules`, `soak_reports: [{path, sha256}]`, `hw_artifacts: [{id, path, sha256}]`, `snapshot_sha`; phrase `TIER <tier> <session_cap_min> <CHAL>`), **`so101-ack <item_id>`** (`type: ack`, `item_id`, `kind`, `run_id`, `session_id`, `servo_summary_sha256`, `note`; phrase `ACK <item_id> <kind> <CHAL>`). The same console/uid/env rules; the same O_NOFOLLOW/fstat/exact-bytes rule; the same strict grammars.

**What this protects against:** forging or altering records (needs the passphrase and the console login), substituting the trust root or the signing/verifying code (root-owned, summed), a terminal hijack of koen's session (signing is impossible from a pty or via su/sudo), same-user verifier bypass (the verifier runs as `so101-servo` from the snapshot), replay (nonce ledger in the servo, session binding, expiry), rollback (seq, supersedes, current pointer, revocation), pasted/prepared phrases (challenge word), misread summaries (machine fields + diff above the prompt, retyped parameters, AI text off by default). **Residual risk, stated plainly (P11):** social engineering of a *correctly displayed* request — the human can still approve an unwise experiment that is exactly what it says; **sudo exposure** — koen has `sudo`, so a compromised koen session that obtains the sudo password can alter root-owned files (mitigations: never type the sudo password in an AI-driven session, verify `SHA256SUMS` before every session, the console-only signing rule); **bugs** in the reviewed scripts, the authority helper or the snapshot (mitigations: small stdlib-only code, golden-output tests, the review procedure); and anything physical. `docs/HARDWARE_CHECKLIST.md` gains the rule: **never run AI-suggested commands in the signing session.**

### 5.4 The Tier 1 record, its rules, the two-phase clean-session verdict (P5 + panel 5, 6)

The signed tier record (§5.3) carries `tier`, `by`, `date`, `milestone_sha`, `snapshot_sha`, `soak_reports` (`attended_30min`, `goal_directed_120min`, each with sha256), `hw_artifacts` (with sha256), and `rules` `{tasks, kinds, trainers, policies: [{policy_id, sha256}], session_cap_min (60 at promotion), clean_sessions_at_cap, max_episodes_per_run, min_gap_min: 2, quiet_hours: ["23:00","07:00"], max_arm_min_per_day: 240, day_boundary: "local-midnight", speed_scale_max, require_rt_active: true, require_camera_moved_gate: true, valid_until}`.

- **Verified** (lab mirror + servo pre-ARM): signature/current/unrevoked/seq; `check_soak_report(strict=True)` on the 120-min goal-directed report (now including the parked-segment metrics of §5.8-7) whose sha256 is in the body; `snapshot_sha == /opt/so101_rl/current`; hw artifact sha256s match; `valid_until` in the future.
- **Cap ramp 60 → 120 → 240 → 480:** a raise is a new signed tier record (`seq+1`, `supersedes_sha256`) and is accepted only if `clean_sessions_at_cap ≥ 1` at the current cap in `runs/sessions.jsonl` **and** in the servo's own `sessions/` store (the servo refuses a session request whose cap exceeds what its store justifies).
- **Two-phase clean verdict** (the weld verdict is only observable by the *next* servo): phase 1 **`servo_clean`** at exit from the summary: ended by cap (`--max-ticks`), by `STOP_REQ`, or by the idle timer + cap; no `_FRESH_PROCESS_CAUSES`/`_HUMAN_CAUSES` trip other than the closing `USER_STOP`; thermal trips ≤ 1 **including a parked thermal trip later resumed on request** (recommended, P5a); jitter within the soak gate; `exit_code == 0`. Phase 2 **`weld_ok`** at the next session's first `status()`: reason ∈ {1 TIMEOUT, 6 MCU_RESET, 8 ARM_TIMEOUT}, not 7 WELD, and ARM accepted. A session is **clean** only with both; a raise is therefore evaluable only after the following session's ARM.
- Quiet hours, min gap, allow-list, per-day minutes (servo-energised minutes incl. parked HOLD, summed over sessions starting in the local calendar day): enforced by the servo at attach (its own ledger) and mirrored by the lab at admission (`SESSION_RULE`).
- **Holding at home is thermally unverified** (panel 6; `home` is gravity-loaded; `duty_on_s`/`duty_off_s` are documentation-only until enforced, noted in ARCHITECTURE §6): **HS8** records per-joint temperature slope over a 10-min hold at `home` after a 20-min run; pass = max temperature < 50 °C at the end of the hold; otherwise `park_timeout_s` is lowered. Until HS8 data exists `park_timeout_s` stays 600 (as logged); the proposal after HS8 is a default aligned with `min_gap_min` (120–180 s).

### 5.5 `lab/policy.yaml` (as v2, plus) `session {arm_timeout_s: 110, exit_wait_s: bound_s + 10, device_prefix: "/dev/serial/by-id/", park_tolerance_ticks: 20}`, `records {store: /var/lib/so101_rl/records, allowed_signers: /etc/so101_rl/allowed_signers, expiry_h: 2, approved_skew_s: 60}`.

### 5.6 Session lifecycle (panel 1, 4, 5, 9, 10, 11)

1. **Request.** The session manager writes `/var/lib/so101_rl/attach/sessions/<session_id>.json` = `{tier_claim: 0|1, requested_cap_min, run_ids_expected, tick_log, perception: bool}` (koen-written, servo-validated) after its mirror checks: hw artifacts fresh; tier record valid; **device identities**: `so101.yaml` `port`/`heartbeat_port` must be `/dev/serial/by-id/...` paths (loader-enforced when the servo runs with `--session`), must resolve, and must be owned by `so101-servo`; if null/unresolvable → `DEVICE_UNRESOLVABLE`, item `device_unresolvable`, **no button is asked for**; RT limits readable from the unit (`rt_unavailable` otherwise).
2. **Start.** `systemctl start so101-servo@<session_id>` (polkit). The servo (snapshot) validates the request against the signed tier record (cap ≤ `session_cap_min`, tier claim ≤ record tier, quiet hours, per-day minutes), resolves the devices, runs `validate_geometry`, opens the port without a handshake, and **waits in BOOT for the button for at most `--arm-timeout-s 110`** (the value is in the request/unit, the CLI default stays `None`). Only now the dashboard and a desktop notification raise item `session_open_request`: "the servo is waiting — press the MCU button within N s".
3. **The two timers have different origins.** The servo's 110 s runs from servo start and bounds *its own wait*; FW1's 120 s runs from the MCU's **entry into DISARMED** (the button release) and bounds *a press nobody arms*. Normal case: servo waiting → press → MCU DISARMED → the servo sees the status and ARMs within one tick → ARMED (relay closed) → ping → first good read/write/ACKED heartbeat within 0.8 s → re-energise → **HOLD(parked) at home** (SV1, §5.8-2). Degenerate cases: no press within 110 s → servo exits 2, item closes as `timed_out`, no MCU change; a press while no servo waits → the MCU latches 120 s later with reason 8 → the next request tells the human to press again *after* the servo is up.
4. **Session id** = `arm_ns` (monotonic ns of ARM) in the state record and the summary; the servo is authoritative. Approvals and attach claims bind to it.
5. **Attach.** A job's first write to the command block carries its `client_id` (F3 command block gains a `client_id` u32 field: layout version bump, `attach` refuses the old version; **monotonic dedupe**: `seq <= last_seq` is no new command; every writer re-reads `seq` under the cmd-owner lock before its first write and continues from `last + 1`). The job has written `/var/lib/so101_rl/attach/<session_id>/<client_id>.json` = `{run_id, claim: "approval"|"tier1", record_path?}` beforehand. On the first RUN or `RESUME_REQ` from a new `client_id` the servo calls the authority helper; a refusal is **ignored + counted** (`client_refused` in the state record and summary; the arm stays parked / in TORQUE_OFF), and the job sees it in `ctx.servo()` and ends `stopped: client_refused`.
6. **Between runs:** the job's last act is `PARK` (SV1; accepted only within `park_tolerance_ticks` of home and `block_ok`). A job that detaches without parking causes the ordinary STALE_CMD trip → safe state → (with `--exit-on-terminal`) servo exit → session over; the manager does **not** park on a job's behalf (panel 8).
7. **Close** (panel 1 — the BLOCKER): `STOP_REQ` and trips do not by themselves exit today's servo (it ticks in TORQUE_OFF with heartbeats, the relay stays closed, no latch, no weld check, locks held). v3: **SV1 adds `--exit-on-terminal`**: after the safe-state sequence reaches TORQUE_OFF for `USER_STOP` or any `_FRESH_PROCESS_CAUSES`/`_HUMAN_CAUSES` cause, or when `--max-ticks` fires, the servo runs `ServoCore.shutdown` and **exits** (idle TORQUE_OFF from `IDLE_TIMEOUT` is not terminal and stays). Heartbeats then stop → MCU LATCHED (reason 1) → relay open. Fallback when the process does not exit within `exit_wait_s`: `systemctl stop so101-servo@<id>` = **exactly one SIGTERM** (the runner's normal shutdown through the safe-state sequence), then systemd's SIGKILL at `TimeoutStopSec` = `bound_s + 10`; a failed exit is item `servo_exit_failed`. Tests assert the servo pid is gone **and** FakeMcu is LATCHED after cap, after `STOP_REQ` and after a NaN trip.
8. **Weld check / power cycle:** the next session's first `status()` (§5.4 phase 2). SV1 adds to the summary: `pid`, `arm_ns`, `first_mcu_status`, `last_mcu_status`, `mcu_armed`, `exit_reason`, `client_refused`, `park_refused`, `resume_refused`.
9. **Perception** (die tasks): transient user unit started with the session; a die-task job refuses to start without an object table younger than 1 s and a fiducial OK. **PE1**: `perception/main.py` (entry point) is a perception work item.

### 5.7 Unattended real-arm operation: the failure matrix (with the servo-exit column)

| Event | Servo (snapshot; SV1 rows marked) | Servo exits? | Connector | Run state → item |
|---|---|---|---|---|
| Thermal trip during a run | SAFE_STOPPING → rest → TORQUE_OFF → dwell → CHECK → full re-energise → **HOLD(parked)** (SV1) | no | episode `truncated`; `paused: servo_wait` (arm lease held, arm-time clock stopped); on parked HOLD the job re-attaches and a fresh RUN promotes | `paused` → `running`; after the 4th trip in an hour: TORQUE_OFF, human → **exit** (terminal) → `stopped: servo_needs_human` → item |
| Thermal trip while parked (SV1) | → rest → TORQUE_OFF (THERMAL \| IDLE_TIMEOUT); **no automatic re-energise while parked**; `RESUME_REQ` honoured after the dwell, inside the hourly budget | no | next job's `RESUME_REQ`; waits ≤ `resume_wait_min` else `stopped: servo_wait_timeout` | counts toward the ≤ 1 clean-session allowance (P5a) |
| Idle timeout (SV1) | parked HOLD → SAFE_STOPPING → rest → TORQUE_OFF (IDLE_TIMEOUT); heartbeat + goal = present continue | no | event; next job `RESUME_REQ` → CHECK → full re-energise → HOLD(parked) → fresh RUN | — |
| Late attach after any re-energise (panel 2) | lands in **HOLD(parked)**, idle timer running; no STALE_CMD fuse until a client is in RUN | no | the job attaches whenever it is scheduled | — (test: no client for 30 s after re-energise never leaves parked HOLD) |
| NaN / stuck / OOD / RANGE / BOX_PRESENT (incl. the gripper-point floor at 2 mm) / N stale commands | SAFE_STOPPING → TORQUE_OFF; fresh process | **yes** (`--exit-on-terminal`) → LATCHED | `truncated`; checkpoint; exit | `stopped: servo_trip:<cause>` → `servo_trip_fresh_process`; real admissions blocked until a signed ack; next session needs the button |
| Protection bit / calibration mismatch / `STOP_REQ` | SAFE_STOPPING → TORQUE_OFF; human | **yes** | as above | `stopped: servo_needs_human` (or the clean close) |
| Heartbeat fail / MCU latched / deadman / e-stop | LATCHED | yes (LATCHED is absorbing; with `--exit-on-terminal` the runner exits after torque-off) | session `latched`; approvals/admissions void | `stopped: servo_latched` → item (physical reset) |
| Client refused at attach (bad/expired/consumed record, outside rules) | command ignored, counted | no | job ends | `stopped: client_refused` → item `client_refused` |
| Job halt rule / real review trigger / arm-time or per-day cap / quiet hour | — | no | finish episode, `PARK`, exit | `stopped: …` (+ `review_real`, `hw_artifact_stale` where applicable) |
| GPU hot or health unknown during a real run | — | no | pause at episode boundary, parked; after 20 min end the run | `paused` / `stopped: thermal_gpu_timeout` |
| Job crash while attached | holds last target; 5 misses → SAFE_STOPPING → TORQUE_OFF (STALE_CMD, fresh process) | **yes** | `crashed` | item `servo_trip_fresh_process` |
| Supervisor crash / restart | — | no (jobs and servo are their own units) | re-adoption by unit name; `--max-ticks` bounds the session regardless | event |
| Host suspend | heartbeats stop → LATCHED | runner exits on wake (LATCHED → exit) | `host_slept`, approvals void | items |
| Host reboot / power loss | LATCHED | process gone | `crashed` runs, real queue voided | items |
| Session cap (`--max-ticks` or manager `STOP_REQ` via the job) | shutdown safe-state → torque off | **yes** → LATCHED (1) | records phase 1; phase 2 at the next ARM | `session_ended:cap` |

Every row ends with the arm in a servo-defined safe state and, for every terminal row, a **servo exit so the relay opens**; items need a signed ack; no row restarts real motion without the servo's own automatic paths or a human act.

### 5.8 Work item SV1 — servo park / idle / resume / exit (user P3; panel 1, 2, 5, 7, 8, 9)

**Interface changes (snapshot + repo; ARCHITECTURE §5/§6 in the same commit).**
1. `servo/shm.py`: `CmdMode.PARK = 4`, `CmdMode.RESUME_REQ = 5`; command block gains `client_id` (u32) → command-block layout version bump (`attach` refuses the old); dedupe becomes **monotonic** (`seq <= last_seq` → no new command).
2. `safety/types.py`: `TripCause.IDLE_TIMEOUT`; `ResumePolicy.ON_REQUEST`; `RESUME_BY_CAUSE[IDLE_TIMEOUT] = ON_REQUEST`; `_POLICY_RANK = {AUTO: 0, ON_REQUEST: 1, FRESH_PROCESS: 2, HUMAN_PHYSICAL_RESET: 3}`.
3. `config`: strict `idle:` `{park_timeout_s: 600 (30..3600), park_tolerance_ticks: 20 (1..100)}`; `so101.yaml` `port`/`heartbeat_port` must be `/dev/serial/by-id/` paths when `--session` is given (loader check).
4. State record v3: `parked` u1, `client_refused` u1, `park_refused` u1, `resume_refused` u1, `arm_ns` i8 in the padding; tick log likewise. Summary gains the §5.6-8 fields.
5. `servo/main.py`: `--session <file>`, `--records <dir>`, `--state-dir <dir>`, `--exit-on-terminal`; `--arm-timeout-s` default stays `None`.

**State-machine edges (added; existing rows unchanged; first match wins).**

| from | condition | to | resume |
|---|---|---|---|
| RUN / HOLD | `PARK` received **and** every joint within `park_tolerance_ticks` of `home` **and** `block_ok` | HOLD, `parked = True`, target = present (latched) | — |
| RUN / HOLD | `PARK` otherwise | ignored; `park_refused += 1` | — |
| HOLD(parked) | stale-command and `consecutive_misses` rows **do not apply**; duty = `monitor_parked`: `monitor_only` **plus** the present-pose box check incl. the gripper points (tip floor −0.015 m, points ≥ table + 1 mm with the 2 mm tolerance → BOX_PRESENT trip) and the re-write of the latched target passed through `clamp_tracking` (|target − present| ≤ E) | HOLD(parked) | — |
| HOLD(parked) | fresh RUN command, age ≤ `stale_cmd_ms`, `block_ok` | RUN (resync), `parked = False` | existing row |
| HOLD(parked) | `parked_ns ≥ park_timeout` | SAFE_STOPPING (bounded) → TORQUE_OFF, fault IDLE_TIMEOUT | on request |
| HOLD(parked) | THERMAL | SAFE_STOPPING → TORQUE_OFF, THERMAL \| IDLE_TIMEOUT; `may_resume_thermal` does **not** re-energise while parked | on request |
| CHECK (any re-energise inside a `--session`) | `mark_energized` | **HOLD(parked)** at home: target = present after the ramp, idle timer started, heartbeat continuing (not RUN: the 500 ms STALE_CMD fuse must not run with no client) | — |
| TORQUE_OFF | faults ⊆ {IDLE_TIMEOUT, THERMAL, STALE_CMD-from-parked}; `not _human_required and not _fresh_required`; THERMAL (if set) has `may_resume_thermal` and the hourly budget; `RESUME_REQ` from a client the authority helper admitted; `block_ok` | CHECK (`reenergize_requested`) → full `reenergize` → HOLD(parked) | — |
| TORQUE_OFF | `RESUME_REQ` otherwise | ignored; `resume_refused += 1` | unchanged |
| TORQUE_OFF | `--exit-on-terminal` and the cause set contains USER_STOP or any `_FRESH_PROCESS_CAUSES`/`_HUMAN_CAUSES`, or `--max-ticks` reached | `ServoCore.shutdown` → process exit | — |

`mark_energized` and `_enter_run` clear `IDLE_TIMEOUT` together with `_AUTO_CAUSES`. Obligations kept (each tested): heartbeat every tick in every state incl. parked and idle; resync on the first good read entering RUN; the full `reenergize` with `should_abort`, strict ACKED heartbeat, `hand_over_on_abort`, the start rule (rest within tolerance **or** tip in the box with z ≥ `start_min_tip_z_m` = 0.010 m base frame) and `UnsafeStartPose` → TORQUE_OFF (human); `bound_s`; a stop signal or `STOP_REQ` anywhere goes through the existing paths; `tick()` allocation-free; the batched sim chain untouched (D-CHAIN).

**Soak (panel 7):** `analyse_tick_log` gains `parked_hold_entries`, `idle_rests`, `resume_cycles` from the `parked` flag and IDLE_TIMEOUT; `hold_entries` counts only unparked HOLD entries (policy late); the ≤ 200 ms heartbeat-gap rule spans parked segments; shown-to-fail twin: a parked segment with one 300 ms gap fails.

**Tests.** State machine: parked HOLD never trips STALE_CMD over 10 000 ticks; PARK refused outside tolerance / without `block_ok`; `monitor_parked` trips BOX_PRESENT on a sagging gripper point; timeout → IDLE_TIMEOUT; `RESUME_REQ` → CHECK once, refused for NAN/USER_STOP/LATCHED/`_fresh_required`/`_human_required`; thermal-while-parked no auto re-energise; **every re-energise lands in HOLD(parked), and no client for 30 s never leaves it**; `IDLE_TIMEOUT` cleared on `mark_energized`. Core on FakeBus/MujocoBus: PARK → rest reached, torque off, ≤ 5 ticks of motion after torque-off → `RESUME_REQ` → full re-energise → parked → fresh RUN; heartbeat gaps ≤ 200 ms over a 15-min fake session with two park/resume cycles; `--exit-on-terminal`: **pid gone + FakeMcu LATCHED** after `--max-ticks`, after `STOP_REQ`, after an injected NaN; monotonic dedupe; `client_id` attach verification via a fake helper. **HS8 (attended):** 60-min parked session with three park → idle → resume cycles **and** the thermal hold measurement of §5.4; artifact `runs/hw/park-resume-<ts>.json`.

### 5.9 Work item FW1 — firmware arm timeout (user P4; panel 4)

DISARMED → LATCHED after ≥ 120 000 ms **from entry into DISARMED** without an accepted ARM (reason 8 ARM_TIMEOUT; wrap-safe `ticks_diff`); a later press re-enters DISARMED and restarts the window. FakeMcu byte-identical; vectors (119 999 ms DISARMED; 120 000 ms LATCHED 8; ARM at 119 000 ms ARMED; press after timeout → DISARMED); PROTOCOL.md row + example byte; host decoding `ARM_TIMEOUT`; HARDWARE_CHECKLIST H3 extension. The servo's `--arm-timeout-s` CLI default stays `None`; the session unit passes 110 (A10 corrected).

### 5.10 Audit, kill switches, never-do list

`runs/audit.jsonl` (lab) and `/var/lib/so101_rl/servo/sessions/<id>/authority.jsonl` (servo: every verification verdict, nonce consumption, refusal). Compute kill: `control_run(kill)`, `runs/HALT`. Arm stop: the e-stop and the deadman; an MCP stop counts as zero safety value. The AI can never: edit safety/servo/firmware/snapshot/config; write calibration or `sim_params.yaml`; change caps, thresholds, halt rules, ladder, rules; create or alter any signed record or the tier record; start the servo with a different argv (the argv is root-owned); write the command block without `client_id` + lock; move the arm outside an admitted procedure; run arbitrary code through the connector; receive raw logs/video bytes; resume a tripped real run without a signed ack; lower the attempts count.

## 6. Experiment management for an AI user (as v2)

Summaries; stage control via config and `control_run(resume, overrides)`; gates as first-class actions (`run_gates`, `evaluate_policy(gate)`, signed `so101-gate-sign`, `request_real_run(zero_shot_eval)` → `wilson_lo ≥ 0.60` → `finetune`); notebook (`write_note`, untrusted data); proposals in `plans/proposals/` (P9).

## 7. Future model types (as v2): generic in shape, nothing built.

## 8. Dashboard, packaging, OS1, testing, observability

- **Dashboard** (as v2) plus a **session panel** showing the servo unit state, snapshot sha, `arm_ns`, parked flag, cap/elapsed, arm minutes today, phase-1/phase-2 verdicts of the last session, `SHA256SUMS` check age, and the exact console commands for pending records (display only).
- **Pixi:** feature `lab` (`mcp 2.3.*`, `jsonschema`, `nvidia-ml-py`), `ffmpeg` declared in `train`; envs `default`/`cpu` gain `lab`; `robot` unchanged (the snapshot is built from it). Tasks: `lab-mcp`, `lab-supervisor`, `lab-dashboard`, `lab` (`status`, `halt`, `unhalt`, `audit`, `import-run`, `install-units` (user units only), `session-request` (debug)), `test-lab`. No approval tasks.
- **OS1 (new work item, human-run, reviewed):** create group `so101` and users `so101-approver`, `so101-servo`; udev rules (by-id symlinks, owner, mode); remove koen from `dialout`; approver home + key (by hand); `/etc/so101_rl/{allowed_signers,SHA256SUMS}`; `/var/lib/so101_rl/{records,servo,attach}` ownership/modes; `/usr/local/lib/so101_rl/*.py` + shims; snapshot build `/opt/so101_rl/<sha>/` with the frozen `robot` env; system unit `so101-servo@.service`; polkit rule; `RT_GROUP_SCHED` slice setting; `runs/` group permissions; the **update procedure** (§5.2). Install scripts are read in full at a pinned commit; every step verified by `sha256sum -c /etc/so101_rl/SHA256SUMS`.
- **Protected-path guard** (as v2): `safety/`, `servo/`, `firmware/`, `so101.yaml` change only in commits with `Work-Item: SV1|FW1` listed in `docs/protected_work_items.yaml`.
- **Tests** (as v2, plus): `test_lab_session.py` — session request schema, device-identity refusal before the button, two-timer sequencing with FakeMcu (press before/after servo start, FW1 latch), attach claim + `client_id`, monotonic seq, cmd-owner lock, STOP_REQ → exit → LATCHED, `systemctl` fallback simulated, two-phase verdict, cap-raise refusal; `test_lab_records_tools.py` — the three tools from a temp install as a **different uid** where CI permits (else uid check mocked), console-tty refusal on a pty, `env -i` scrub, O_NOFOLLOW/fstat, strict grammar rejections (Unicode, control bytes, oversize), challenge word, exact-bytes signing, O_EXCL, seq/supersedes/current/revocation, expiry sanity, namespace mismatch; `test_lab_authority.py` — the §5.1 table incl. the servo-side helper's verdicts via the snapshot helper module; `test_lab_supervisor_systemd.py` (`@integration`) — A4 under a real `systemd --user` with transient units; SV1/FW1 tests per §5.8/§5.9.
- **Observability** (as v2) plus the servo's authority log and session store.

## 9. Deltas to propose and the phased build

### 9.1 Plan

§12 is updated to v3 (two accounts, snapshot, servo-side verification, session lifecycle, SV1 landing-in-parked and exit-on-terminal, two-phase verdict, HS8 thermal). **Proposed §10 amendments/rows (few):**

| # | Row | Proposed text | Reason |
|---|---|---|---|
| P3a | SV1 scope (amends P3) | Every re-energise inside a session lands in HOLD(parked) at home (no STALE_CMD fuse until a client is in RUN); PARK accepted only within `park_tolerance_ticks` of home with `block_ok`; parked HOLD runs the present-pose box check incl. the gripper points and clamps the re-write; `RESUME_REQ` needs an admitted client and no fresh/human fault; `--exit-on-terminal` makes USER_STOP, fresh-process/human trips and `--max-ticks` exit the process so the relay opens; monotonic command dedupe; `client_id` in the command block; summary fields for the two-phase verdict. `park_timeout_s` stays 600 until HS8 measures the hold thermals (pass: < 50 °C after a 10-min hold following a 20-min run), then a default near `min_gap_min` (120–180 s) is proposed. | Safety panel 1, 2, 5, 6, 8, 9 |
| P5a | Clean session (amends P5) | Two-phase verdict: `servo_clean` at exit + `weld_ok` at the next ARM (first MCU status reason ∈ {1, 6, 8}, not 7); a parked thermal trip later resumed on request counts toward the ≤ 1 allowance; a cap raise is a new signed tier record evaluable only after the next ARM. | Safety panel 5, 6 |
| P12 | Session lifecycle and units | Jobs and perception run as transient `systemd --user` units; the servo as the `so101-servo@<session>` system unit from the snapshot with the pinned argv in the unit; sessions close by `STOP_REQ` + `--exit-on-terminal`, with `systemctl stop` (one SIGTERM, then SIGKILL at `bound_s + 10`) as the fallback; the servo waits ≤ 110 s for the button and the dashboard asks for the press only once the servo is waiting. | Safety panel 1, 3, 4 |
| P13 | Device identities | `port`/`heartbeat_port` are `/dev/serial/by-id/` paths owned by `so101-servo`; unresolvable or wrongly owned devices refuse the session before any button is asked for. | Safety panel 11 |

### 9.2 ARCHITECTURE (v0.18 with the first lab unit)

§3: the account boundary, the three services, transient job/perception units, the servo system unit and snapshot, the records store; "When the policy is late … unless parked". §4: `lab/`, arrows, `WriteGuard` (guard rail), `perception/main.py`. §5: FL1–FL3, file schemas, `PARK`/`RESUME_REQ`/`client_id`, state record v3, summary fields, `ScalarMonitor`. §6: layer 2 (FW1 reason 8), layer 5 (IDLE_TIMEOUT, parked rules, landing in parked HOLD, exit-on-terminal), new item 7 "Connector authority" (who enforces what; guard rails labelled; duty fields documentation-only until enforced). §8: `lab` feature, deps, host requirements, OS1 artefacts, units, polkit, `.mcp.json`. §9/§10.

### 9.3 Phased build (re-estimated; agent-hours unless noted)

| Unit | Wave | Builds | Depends on | Est. |
|---|---|---|---|---|
| **LA0** core | L0 | registry, locks, idempotency, `WriteGuard`; provenance; audit; derived ledger + `import-run`; trainers API + table + `fake`; supervisor (transient units, leases, pipelines, NVML fail-closed, stall ladder, suspend/reboot, TTLs, disk, review policies, needs-human, notifications); job entry; `policy.yaml`; SIM authority; user units | — | 28 |
| **LA1** MCP + dashboard | L0 | 12 tools, resources, prompts, errors, envelopes; dashboard; `.mcp.json`; contract tests; ARCHITECTURE v0.18; plan §12 v3 | LA0 | 20 |
| **SV1** servo | L0–L1 (`Work-Item: SV1`) | §5.8 incl. soak extension, state record v3, `--session/--records/--state-dir/--exit-on-terminal`, helper hooks | — | 28 |
| **FW1** firmware | L0–L1 (`Work-Item: FW1`) | §5.9 | — | 4 |
| **RT0** records tools + authority helper | L1 | `so101-approve/gate-sign/ack`, `so101_authority.py` (verify, nonce ledger, rules, arm-minute ledger), record formats, golden tests | — | 20 |
| **OS1** accounts, udev, snapshot, units, polkit, SHA256SUMS, update procedure | L1 | §5.2, §8 | RT0, SV1 (snapshot contents) | 16 + 4 human-h |
| **PE1** perception process | L1 | `perception/main.py` + unit | U5 | 6 |
| **LA2** FastSAC unattended | L1 | plugin, gates, eval+gate, pipelines, `ScalarMonitor`; A1–A4 | SU3, SU4 | 16 |
| **LA3** offline + fine-tune (sim) | L2 | three plugins (sim), `--tier1` mode, smoke | SR2, SU5 | 12 |
| **LA4** real arm, Tier 0 | L3 | session manager (request files, `systemctl` via polkit, attach claims, cmd-owner lock, two-phase verdict, device checks), real-arm tool, arm-time ledger, matrix on `servo --backend fake` (fake authority helper); A5 (fake) | LA3, RT0, OS1, SV1, FW1, PE1; first real use: H2, H3, H5, H7, H11, HS8 | 26 |
| **LA5** Tier 1 unattended | L4 | signed tier reader, rules, cap ramp, soak verification at admission, per-batch revalidation; A6 | LA4; H13 + 120-min strict soak (with parked segments) + HS8 + one clean attended 60-min session (two-phase) | 12 |

≈ 190 agent-hours, ≈ 2 GPU-h, ≈ 12 attended human-hours (OS1 setup incl. key generation, HS8, FW1 test, first sessions).

## 10. Acceptance (changes from v2 marked)

- **A1–A3** as v2. **A4** (changed): under a real `systemd --user` (integration test): `systemctl --user restart so101-lab-supervisor` kills no job; `kill -9` job → `crashed`; re-adoption by unit name; suspend/reboot simulations; closing Claude Code changes nothing.
- **A5** (changed): Tier 0 on the fake backend: request → `awaiting_approval`; lab processes cannot write records (no path; store not writable); `so101-approve` from a temp install as the approver uid on a console-like tty writes a verifying record with the challenge word; expired/consumed/hash-changed/wrong-key/wrong-namespace/pty/sudo-env/Unicode-request cases refused; the session manager's request → fake servo waits → fake button → ARM → **HOLD(parked)** → job attaches with `client_id`, the fake authority helper admits → RUN → PARK → `STOP_REQ` → **servo pid gone, FakeMcu LATCHED**. First real use (after LA4, attended): HS1 and HS4 through `request_real_run` + console approval.
- **A6** (changed): Tier 1 on the fake backend: signed tier record; two runs in one session without approvals, parked between them; late attach after re-energise stays parked; idle rest → `RESUME_REQ` → parked → RUN; thermal during run / while parked per the matrix; NaN → exit → LATCHED → item → blocked until a signed ack; refusals for quiet hours / min gap / per-day cap / allow-list / cap; two-phase verdict recorded; cap raise refused without `weld_ok`. Real: the first unattended 60-min session (HS5) ends clean in both phases before the ramp proceeds.
- **A7, A8, A11** as v2 (A8 adds: OS1 files match `SHA256SUMS`; the lab imports no snapshot code).
- **A9** (changed): SV1 obligations incl. landing in parked HOLD, PARK tolerance, `monitor_parked` gripper-point trip, exit-on-terminal (pid gone + LATCHED in the three cases), monotonic dedupe, soak parked metrics with the shown-to-fail twin; HS8 artifact incl. the thermal hold data.
- **A10** (corrected): FW1 vectors pass on FakeMcu and the real MCU; the session unit passes `--arm-timeout-s 110`; the servo CLI default remains `None`.
- **A12** (new): OS1 — devices owned by `so101-servo`, koen not in `dialout` (`id koen`), the approver cannot sign from a pty or via sudo, the servo unit starts from the snapshot with the pinned argv, polkit lets koen start/stop only that unit, `sha256sum -c` passes.

## 11. Remaining genuinely open user decisions

| # | Decision | Recommended default |
|---|---|---|
| U7 | Scope of **servo-side** Tier 1 rule enforcement beyond record validity and the session cap: also allow-list (kind/task/trainer/policy sha), quiet hours, min gap and per-day arm minutes from the servo's own ledger (recommended), or records + cap only with the rest in the lab mirror | **full scope in the servo**: it is ~100 lines of stdlib logic in the authority helper and it is the only place the user account cannot bypass |

`park_timeout_s` after HS8 is a planned follow-up (P3a names the procedure), not a decision now. Everything else from v2's open list is now logged (P1–P11) or is an implementation default recorded in ARCHITECTURE §9.

## 12. Sources consulted

As v2, plus: plan §10 rows 2026-10-07 (P1–P11, "Gripper floor clearance", "Re-energise start floor value"), ARCHITECTURE v0.17 (§5 gripper floor as built, start rule), `safety/state_machine.py` (`_POLICY_RANK`, `_human_required`, `_fresh_required`, `mark_energized`, `_enter_run`), `servo/soak.py` (`analyse_tick_log`), `servo/shm.py` (`CmdMode`), `servo/main.py` (CLI), `firmware/watchdog_mcu/PROTOCOL.md` (states, reasons 1–7, ARM preconditions).
