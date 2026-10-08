# MVP plan: sim training driven by MCP, plus an attended zero-shot reach eval on the real arm

This plan was decided on 2026-10-08 (plan §10, row "MVP scope"). The user's words were: *"keep the package small, a minimal viable product."*

**Goal:** the smallest package in which an AI client (Claude Code) trains a **reach** policy in GPU sim through MCP, and a human then runs an **attended zero-shot reach evaluation** of that policy on the real arm.

**Existing code is frozen, not deleted.** Nothing that is built and tested gets removed. Nothing outside this MVP gets extended. A later cruft review (`simplify`) lists unused code to remove.

## In the MVP, in order

| # | Step | Source | Notes |
|---|---|---|---|
| M0 | Merge the camera unit (after its review fixes) and FW1 | `feature/camera-parity`, `feature/connector-fw1` | Both are already built. |
| M1 | **S3.1 domain randomisation, reach stages only** | sim plan Task S3.1 | Covers servo params (ratio group ±40 %, friction 0–0.21), servo delay, proprio noise and camera pose. The curriculum controller is reduced to a fixed stage list (S3.2 is cut down). |
| M2 | **S4.1 FastSAC + GPU replay** | sim plan Task S4.1 | Self-written minimal SAC with an SB3-shaped actor. |
| M3 | **S4.2 L1/L2 gates for reach only** | sim plan Task S4.2 (reach part) | Feasibility and predicate gates come before any reward tuning. |
| M4 | **S4.3 training loop, sim eval + exit gate, policy export** | sim plan Task S4.3 | Logs `metrics.csv` and TensorBoard, renders through `viz/video.py`, and records `camera_pose_method`. Rollout logging for TD-MPC2 is deferred. |
| M5 | **Minimal connector `so101_rl.lab`** | connector plan LA0 + LA1, cut down (see the next section) | |
| M6 | **S4.5 reach pre-training to the exit gate**, launched by the AI through MCP | sim plan Task S4.5 | Budget ≤ 6 GPU-h, ≤ 3 attempts. The human signs the exit-gate checklist. |
| M7 | **S5.1 `deploy_policy`, plus the zero-shot reach eval** | sim plan Task S5.1 (reach part) | The same script runs against `servo --backend mujoco` and against the real arm. On the real arm it is **attended at Tier 0**: a human at the e-stop starts the servo, with N = 50 episodes, a speed ladder of 0.5 → 1.0 and a Wilson interval. It needs the H-track hardware gates H2 (e-stop), H3 (watchdog MCU) and H5 (rest/home poses), plus the calibrated camera (camera-calibrate passing). It does **not** go through the connector. |

## Minimal connector (M5)

The design is the same as `plans/ai-connector-design.md`, cut down to the sim MVP. Each item below says what the MVP keeps and what it defers.

- **Run registry and job lifecycle:**
  - **Keep:**
    - one directory per run under `runs/` (`run.json`, `status.json`, `events.jsonl`, `usage.json`, `metrics.csv`);
    - a job runner that launches each job as a **transient `systemd --user` unit** (P12), so runs outlive Claude Code;
    - a progress-coupled heartbeat;
    - budget enforcement with a ledger derived from the run dirs (P6);
    - thermal fail-closed via NVML (P8);
    - stop and resume.
  - **Defer:** the long-lived supervisor daemon, queue and lease scheduling (one GPU job at a time is enough), and the suspend/reboot machinery beyond "mark the run interrupted".
- **Trainer interface:**
  - **Keep:** the trainer plugin protocol with two plugins, `fake` (for tests) and `fastsac_sim`.
  - **Defer:** SB3, TD-MPC2 and the real-domain plugins.
- **MCP server (stdio, `mcp` 2.3.0):** six tools.

  | Tool | Purpose |
  |---|---|
  | `list_tasks` | List the available tasks |
  | `start_run` | Start a sim training run |
  | `get_run` | Return status plus a downsampled metrics summary, optionally waiting |
  | `control_run` | Pause, resume or stop a run |
  | `evaluate_policy` | Run the sim eval and the numeric exit gate |
  | `list_runs` | List runs |

  - **Keep:** untrusted-data envelopes on all returned text, and structured errors.
  - **Visibility:** TensorBoard (`pixi run dashboard`) only, with **no web dashboard**. The web dashboard is deferred; the P1 "local webpage" will be revisited after the MVP.
- **Authority:** sim only. The MCP server exposes **no real-arm tools at all**. The exit-gate human sign-off is a simple local CLI (`pixi run lab gate-sign <run>`) for now. The signed-record and two-account machinery (P2, P10, P11, P14) is deferred until real-arm automation is built.
- **Acceptance:**
  - Claude Code starts a reach FastSAC run via MCP, closes, and the run finishes on its own.
  - A new Claude Code session reads the result, runs `evaluate_policy`, and sees the gate verdict.
  - Tests pass without the arm.

## Deferred, not cancelled (frozen; revisit after the MVP)

- **Sim track:**
  - die-to-zone (S4.6);
  - the full curriculum (S3.2);
  - SR2 (HER for TD-MPC2, the TD-MPC2 adapter, the state world model);
  - S5.2/S5.3 real fine-tuning and the gap report;
  - SR3 milestone scripts;
  - the pixel world model.
- **Connector:** LA0's supervisor daemon and queue, the LA1 dashboard and remaining tools, and LA2–LA5.
- **Safety work items** that exist only for unattended real-arm automation:
  - SV1 (park/idle/resume);
  - OS1 (two accounts, signed records, the root-owned snapshot);
  - PE1 (the perception process as a unit);
  - Tier 1.
- **Camera C.6** (H8 hand-eye refinement).

FW1 (done) stays, because it hardens the existing deadman for every real session.

## Hardware the user needs for M7

These hardware tasks come from the H-track (`docs/HARDWARE_CHECKLIST.md`):

1. **H2:** e-stop wired and tested.
2. **H3:** watchdog MCU flashed (with FW1) and tested.
3. **H5:** rest and home poses measured.
4. **Servo power and camera:**
   - The servo bus powered (the probe found it unpowered on 2026-10-07).
   - The D435 depth stream restored (a replug or `hardware_reset`).
   - Then `pixi run camera-calibrate --method arm-shape` and `pixi run camera-overlay` must pass.
5. **Optional:** remove or tape over the orange tip marker.
