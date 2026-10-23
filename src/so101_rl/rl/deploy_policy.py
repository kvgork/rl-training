"""Policy deployment and the zero-shot reach evaluation (MVP M7; sim plan S5.1, reach part).

``pixi run deploy-policy --policy runs/lab/<run>/policy.pt --protocol zero-shot [--attended]``
``pixi run deploy-policy --resume runs/deploy/<ts> [--attended]`` (continue a stopped run)

Runs an exported policy (``rl/policy_export.py``) at 10 Hz through :class:`~so101_rl.env.goal_env.So101GoalEnv` on a
:class:`~so101_rl.env.backend.ShmBackend` attached to an ALREADY-RUNNING servo process: on the real arm the human
starts ``servo --backend feetech`` (Tier 0, attended); in tests and for the sim twin it is ``servo --backend mujoco``.
This script never starts a servo and never opens a serial device; it only writes shm commands.

Everything that sim and real must agree on is the env's: the FS3 observation, ``tasks.action_to_delta_np``, the
reach goal rule (``BoxGoalSampler`` with the near-start redraw) and the success / termination predicate (at the
export's ``success_radius_mm`` when it records one, else REACH's 15 mm; plan §10 2026-10-14). The policy
input is ``observation ++ desired_goal`` (as ``eval_sim``); the action is ``policy(x) * speed_scale`` (the ladder
scales the delta magnitude; the env then clips and maps it).

* **Goal list:** drawn from ``--seed`` before RUN, independent of any measurement: per episode ``i`` a pair
  (first draw, redraw), two sequential ``BoxGoalSampler.sample`` calls on ``default_rng(seed)``, so pair ``i``
  depends only on the seed, the config and the calibration (the goal table). At the reset the env applies its
  near-start rule to the MEASURED start tip: the first draw, or the redraw if the first is within 2 success radii
  of the tip. A sim twin with the same seed, config and calibration therefore gets the same pairs; which member of
  a pair is used can differ between sim and real only when the first draw is near that radius. The pairs are in
  ``summary.json`` (``goals``), the goal used and ``goal_redrawn`` per episode in ``episodes.jsonl``.
* **Episode end:** at success (``info["is_success"]``; the env sends HOLD, no further RUN command until the next reset), at
  the 100-step limit, or when the servo left RUN for a non-success reason (a trip, whatever its ``fault_bits``) or
  stalled (``truncated``). Between episodes ``ShmBackend.reset`` (HOLD, then a bounded ramp to the start pose).
* **Real or sim:** decided from the servo that owns the shm segment: its single-instance lock file names the pid,
  ``/proc/<pid>/cmdline`` is parsed with the servo's OWN parser (``servo.main.build_parser``, no abbreviations), and
  fail-closed: anything but ``--backend mujoco`` / ``fake`` is the real arm and needs ``--attended`` (Tier 0: a human
  at the e-stop); a command line that does not parse cleanly (unknown or abbreviated options) is refused even with
  ``--attended``, and so is a real servo started with ``--no-mcu`` unless ``--accept-no-watchdog`` is also given (plan
  §10 2026-10-10: attended only, the human's hand on the servo PSU's switch is the e-stop). The servo's ``--config`` (default for and
  checked against ours), ``--calibration`` (else its config's ``calibration_path``, resolved against the servo's
  cwd; the FK must use it, a different ``--calibration`` is refused) and ``--tick-log`` (into the summary) are
  taken from the same command line.
* **Protocol** ``--protocol zero-shot`` (plan §10 D-ZEROSHOT, sim plan HS4): the speed ladder 0.5 -> 0.75 -> 1.0 with
  ``--rung-episodes`` (default 10, 10, 50: two short rungs to climb, then the N = 50 gate rung at full speed). The
  ladder climbs to the next rung only if the finished rung had no trip / stall and at least 80 % successes
  (``ceil(0.8 n)``; plan §10, 2026-10-08), else it stops (``halt_reason`` ``rung_below_threshold``); the full-speed
  rung has no advance check. It also stops at the first trip, stalled servo or failed reset. The fine-tune verdict
  (``finetune_eligible``) needs the zero-shot protocol, the whole ladder run and the last run not halted, and the
  Wilson 95 % lower bound of the full-speed rung >= 0.60 with n >= 50 there; fine-tuning itself is deferred in the
  MVP. ``--protocol single`` runs ``--episodes`` at one ``--speed-scale`` (never eligible; only informative).
* **Resume** ``--resume <run dir>`` (plan §10, 2026-10-08): after a stop (trip, stall, Ctrl-C), with a restarted
  servo, continue the same run: the finished episodes are kept, the seeded goal list continues at the next
  unfinished episode in its rung, new episodes are appended to the same ``episodes.jsonl`` and ``summary.json`` is
  rewritten over all of them (each resume recorded in ``resumes``). ``--policy``, ``--seed``, ``--protocol``, the
  plan and the servo's real / sim class are taken from the run; a differing value is refused.
* **Start order:** once the servo is in RUN it safe-stops after 0.5 s without a command (``max_consecutive_misses``),
  so this script must be attached BEFORE the servo reaches RUN: it waits up to ``--wait-servo-s`` for the servo
  process, classifies it read-only (``/proc``; every refusal happens here, nothing sent), then attaches AT ONCE and
  from the attach until RUN sends HOLD every 100 ms (never moves the arm): a keep-alive thread through the slow
  set-up after the attach (the goal table, ~7 s on a cold cache; the env; the first summary), then the main thread
  until RUN. Only a ``--resume`` whose goal list no longer matches is refused after the attach (HOLD only). Real arm:
  start the servo (it waits for the reset button), start this script, then press the button (any time after the
  attach). ``servo --backend mujoco`` presses it itself: start this script first or together with it.
* **Transitions** (ARCHITECTURE 0.43): every finished episode is also written as an F8 episode file
  ``episodes/ep_NNNNNN.h5`` (``logging.episode_logger``; its path is ``transitions`` in ``episodes.jsonl``), the
  real-arm data for fine-tuning. The steps are buffered during the episode and written after it under the HOLD
  keep-alive (no file I/O in the 10 Hz loop); an episode cut short by Ctrl-C or a halt inside it is not written.
* **Outputs** ``<out-root>/<ts>/``: ``episodes.jsonl`` (one line per episode, appended as it ends), ``summary.json``
  (written before RUN and at the end: per rung and overall success rate + Wilson 95 %, the verdict, the goal
  list, the servo's tick-log path, inference timing).
* **Ctrl-C / SIGTERM** (handled from the start of ``main``): before the attach exit 130 with nothing sent; after it,
  stop sending RUN commands (and the keep-alive), send HOLD, write what was collected, exit 130. Further signals are ignored while the
  final HOLD and the summary are written. The servo's own stop / safe-state logic handles the arm. The same holds
  after a normal end: the last command is HOLD, and 0.5 s later the servo runs its safe-state sequence to rest (a
  fresh servo process is needed for the next deployment).

* **Die-to-zone** (0.62, ``plans/die-to-zone-plan.md`` D5, sim plan HS6; plan §10 2026-10-16 "Die support in
  deploy-policy (D5)"): the task is the export meta's ``task_name`` (``--task`` only confirms it; reach is unchanged).
  For ``die_to_zone``:

  - **Goals:** the fixed zone goal (``ZoneSpec.goal_xyz``) for every episode, no sampler (the pairs in ``summary.json``
    are that goal twice; ``--seed`` is still recorded). The ladder defaults to 10 / 10 / 30 episodes (HS6) and the
    verdict's full-speed rung needs n >= 30.
  - **Perception:** the env reads the object table (``--object-table``, default the config's ``object_table_shm`` or
    ``so101_objects``). Refused before the attach when no perception process created it, it was never written, its
    record is older than 1 s, or its camera-moved fiducial is not OK; the same check before every episode stops the
    run (``halt_reason`` ``perception_stale``).
  - **Halt rule (HS6):** an episode whose tracker reported the die UNKNOWN (not visible) on more than 50 % of its
    steps stops the run (``tracker_unknown``: a human must look). Per episode ``episodes.jsonl`` records
    ``visible_steps`` / ``visible_fraction``, ``final_die_xyz`` (the last tracked position, ``null`` if never seen),
    ``final_die_visible``, success and trips; push vs place needs the tracker trace, which is in the F8 episode file.
  - **v0 refusal (plan §10 2026-10-16):** on a REAL servo a die export is refused unless it records
    ``sim_params_release`` >= 1 AND ``sim_params_verified`` true (a missing field = refused): the policy trained on the
    v0 prior under the S4.6 waiver is never deployed. Allowed on the sim twin (``servo --backend mujoco|fake``).
  - **Reset:** the arm goes home; the die start is the tray's job (H9). While ``die_start.measured_on_hardware`` is
    false (no tray), die episodes on the real arm are refused unless ``--manual-die-reset``: before every episode the
    arm is ramped home and held (HOLD keep-alive), and the run waits until the operator confirms the die is at its
    start, by Enter on a terminal or by creating ``<run dir>/die_ready`` (consumed; ``<run dir>/die_reset_waiting``
    holds the episode number while waiting). Attended only.

Exit codes: 0 done (whatever the success rate), 1 the ladder stopped (trip / stall / failed reset / rung below the
advance threshold / tracker UNKNOWN / stale perception), 2 refused (bad policy, no servo, real arm without
``--attended``, a v0 die export on the real arm, no perception writer, ...), 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "ADVANCE_MIN_SUCCESS_FRACTION",
    "DEFAULT_RUNG_EPISODES",
    "DIE_RUNG_EPISODES",
    "FINETUNE_MIN_WILSON_LO",
    "SPEED_LADDER",
    "DieResetGate",
    "GoalList",
    "ServoInfo",
    "advance_check",
    "check_die_on_servo",
    "check_object_table",
    "check_policy",
    "draw_goal_pairs",
    "main",
    "parse_servo_cmdline",
    "servo_info",
    "sim_params_problem",
    "summarize",
    "task_for_policy",
]

SPEED_LADDER: Final[tuple[float, ...]] = (0.5, 0.75, 1.0)
DEFAULT_RUNG_EPISODES: Final[tuple[int, ...]] = (10, 10, 50)
DIE_RUNG_EPISODES: Final[tuple[int, ...]] = (10, 10, 30)  # sim plan HS6: the ladder as HS4 with 30 die episodes
DIE_GATE_RUNG_MIN_EPISODES: Final[int] = 30
UNKNOWN_HALT_FRACTION: Final[float] = 0.5  # HS6: stop when the tracker is UNKNOWN on more than this share of steps
OBJECT_TABLE_MAX_AGE_S: Final[float] = 1.0  # a die run refuses an object table record older than this
DEFAULT_OBJECT_TABLE: Final[str] = "so101_objects"  # the perception process's default (connector plan PE1)
DIE_READY_FILE: Final[str] = "die_ready"
DIE_WAITING_FILE: Final[str] = "die_reset_waiting"
DEFAULTS: Final[dict[str, Any]] = {
    "protocol": "single", "speed_scale": 0.5, "episodes": 10,
    "rung_episodes": ",".join(map(str, DEFAULT_RUNG_EPISODES)), "seed": 7,
}  # fmt: skip
FINETUNE_MIN_WILSON_LO: Final[float] = 0.60  # plan §10 D-ZEROSHOT: real fine-tuning only if the lower bound >= 0.60
GATE_RUNG_MIN_EPISODES: Final[int] = 50  # plan §10 D-ZEROSHOT: 50 reach episodes (here: on the full-speed rung)
ADVANCE_MIN_SUCCESS_FRACTION: Final[tuple[int, int]] = (4, 5)  # plan §10 2026-10-08: climb at >= 80 %, no trips
SIM_SERVO_BACKENDS: Final[frozenset[str]] = frozenset({"mujoco", "fake"})
EXIT_OK, EXIT_HALTED, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 130
SERVO_MARKERS: Final[tuple[str, ...]] = ("so101_rl.servo.main", "so101-servo")
# summary.json keys that describe the run as a whole (kept across resumes); the rest describe the latest invocation
RUN_META_KEYS: Final[tuple[str, ...]] = (
    "policy", "policy_meta", "task", "success_radius_m", "protocol", "plan", "seed", "attended", "servo", "config",
    "calibration",
    "calibration_sha256", "calibration_changed_since_training", "tick_log", "started", "goals", "resumes",
    "object_table", "manual_die_reset", "unknown_halt_fraction",
)  # fmt: skip


class Refused(RuntimeError):
    """A precondition failed before any command was sent (exit 2)."""


def _log(msg: str) -> None:
    print(f"deploy-policy: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ servo discovery


@dataclasses.dataclass(frozen=True)
class ServoInfo:
    pid: int
    backend: str  # "feetech" | "mujoco" | "fake" | "unknown"
    calibration: Path | None
    tick_log: Path | None
    cmdline: list[str]
    config: Path | None = None  # the servo's --config (None: the packaged one)
    cwd: Path = Path()  # the servo's working directory (relative paths of its command line resolve against it)
    no_mcu: bool = False
    problem: str | None = None  # the command line did not parse cleanly (unknown / abbreviated options, ...)

    @property
    def is_real(self) -> bool:
        """Fail-closed: only a recognised sim backend on a cleanly parsed command line is not the real arm."""
        return self.problem is not None or self.backend not in SIM_SERVO_BACKENDS

    @property
    def mcu(self) -> bool:
        """A watchdog MCU link (real or, on the sim backends, the in-process FakeMcu) unless ``--no-mcu``."""
        return not self.no_mcu


def parse_servo_cmdline(pid: int, argv: Sequence[str], cwd: Path) -> ServoInfo:
    """Parse a servo's ``/proc/<pid>/cmdline`` with the servo's own parser (no abbreviations). Anything that does not
    parse cleanly (unknown or abbreviated options, a missing / unknown ``--backend``) sets ``problem``: the servo
    is then the real arm and is refused even with ``--attended``."""
    from so101_rl.servo.main import build_parser as servo_parser

    argv = list(argv)
    start = next((i + 1 for i, a in enumerate(argv) if a == SERVO_MARKERS[0] or a.endswith(SERVO_MARKERS[1])), None)
    if start is None:
        return ServoInfo(pid, "unknown", None, None, argv, cwd=cwd, problem="no servo entry point in the command line")
    ap = servo_parser()
    ap.allow_abbrev = False  # also fail-closed if the running servo's parser differs
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            ns, extra = ap.parse_known_args(argv[start:])
    except SystemExit:
        msg = err.getvalue().strip().splitlines()
        return ServoInfo(pid, "unknown", None, None, argv, cwd=cwd, problem=msg[-1] if msg else "argparse error")
    if extra:
        return ServoInfo(pid, "unknown", None, None, argv, cwd=cwd,
                         problem=f"unrecognised or abbreviated arguments {extra}")  # fmt: skip

    def _path(v: Path | None) -> Path | None:
        return None if v is None else (cwd / Path(v).expanduser()).resolve()

    return ServoInfo(pid, ns.backend, _path(ns.calibration), _path(ns.tick_log), argv, config=_path(ns.config),
                     cwd=cwd, no_mcu=bool(ns.no_mcu))  # fmt: skip


def servo_info(shm_name: str, wait_s: float = 0.0) -> ServoInfo:
    """The live servo process that owns ``shm_name`` (from its lock file and ``/proc``), waiting up to ``wait_s``
    for it to appear; :class:`Refused` if none. Reads files only: never takes the lock, never touches a device."""
    t_end = time.monotonic() + wait_s
    while True:
        try:
            return _servo_info(shm_name)
        except Refused:
            if time.monotonic() >= t_end:
                raise
            time.sleep(0.1)


def _servo_info(shm_name: str) -> ServoInfo:
    from so101_rl.servo.shm import lock_path

    path = Path(lock_path(shm_name))
    try:
        pid = int(path.read_text().split()[0])
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError, IndexError):
        raise Refused(f"no servo process owns shm {shm_name!r} (lock {path}): start the servo first") from None
    argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
    if SERVO_MARKERS[0] not in argv and not any(a.endswith(SERVO_MARKERS[1]) for a in argv):
        raise Refused(f"pid {pid} in {path} is not a servo process ({shlex.join(argv)[:200]}): start the servo first")
    try:
        cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        raise Refused(f"cannot read the working directory of servo pid {pid}") from None
    return parse_servo_cmdline(pid, argv, cwd)


def _yaml_key(path: Path, key: str) -> Any:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or key not in raw:
        raise ValueError(f"{path}: no {key!r}")
    return raw[key]


# ------------------------------------------------------------------------------------------------ policy


@dataclasses.dataclass(frozen=True)
class ResolvedServo:
    servo: ServoInfo
    config_path: Path
    calibration_path: Path
    calibration: Any
    calibration_sha256: str
    cfg: Any  # So101Config with the servo's calibration and shm name


def resolve_servo(shm_name: str, *, wait_s: float, attended: bool, accept_no_watchdog: bool,
                  config: Path | None = None, calibration: Path | None = None, log: Any = None) -> ResolvedServo:  # fmt: skip
    """Find and classify the servo that owns ``shm_name`` (read-only, nothing sent), apply the attach refusals and
    resolve its config and calibration. Raises :class:`Refused`. Shared by ``deploy-policy`` and ``train-reach-sac``.

    Refused: a command line that does not parse cleanly; the real arm without ``attended``; a real servo with
    ``--no-mcu`` without ``accept_no_watchdog``; a ``config`` / ``calibration`` that differs from the servo's."""
    from so101_rl.config import load_calibration, load_config

    log = log or _log
    log(f"looking for the servo on shm {shm_name!r} (up to {wait_s:.0f} s)")
    servo = servo_info(shm_name, wait_s=wait_s)
    if servo.problem is not None:
        raise Refused(f"the servo on shm {shm_name!r} (pid {servo.pid}) has a command line that does not parse "
                      f"cleanly ({servo.problem}): treated as the REAL arm and refused even with --attended")  # fmt: skip
    if servo.is_real and not attended:
        raise Refused(
            f"the servo on shm {shm_name!r} (pid {servo.pid}, --backend {servo.backend}) drives the REAL arm: "
            "Tier 0 needs a human at the e-stop; pass --attended"
        )
    if servo.is_real and servo.no_mcu and not accept_no_watchdog:
        raise Refused(f"the REAL-arm servo (pid {servo.pid}) runs with --no-mcu (no watchdog MCU): restart it "
                      "with the MCU (--heartbeat-port), or pass --accept-no-watchdog with your hand on the "
                      "servo power switch (plan §10 2026-10-10)")  # fmt: skip
    if servo.is_real and servo.no_mcu:
        log("NO WATCHDOG: nothing cuts power if this PC or the servo process hangs; keep your hand on the "
            "servo power switch for the whole run")  # fmt: skip
    from so101_rl.config.loader import DEFAULT_CONFIG_PATH

    config_path = Path(servo.config or DEFAULT_CONFIG_PATH)
    if config is not None and config.expanduser().resolve() != config_path.resolve():
        raise Refused(f"--config {config} differs from the servo's {config_path}")
    servo_cal = servo.calibration or (
        servo.cwd / Path(str(_yaml_key(config_path, "calibration_path"))).expanduser()
    ).resolve()
    if calibration is not None and calibration.expanduser().resolve() != servo_cal:
        raise Refused(f"--calibration {calibration} differs from the servo's {servo_cal}")
    cal = load_calibration(servo_cal)
    sha = hashlib.sha256(servo_cal.read_bytes()).hexdigest()
    cfg = dataclasses.replace(load_config(config_path, calibration=cal), shm_name=shm_name)
    return ResolvedServo(servo, config_path, servo_cal, cal, sha, cfg)


def check_policy(policy: Any, task: Any) -> None:
    """Refuse a policy for another task or input layout (``load_policy`` already checked the layout version)."""
    from so101_rl.env.obs import GOAL_DIM, OBS_LAYOUT_VERSION

    meta = policy.meta
    if meta.get("obs_layout_version") != OBS_LAYOUT_VERSION:
        raise Refused(f"policy obs_layout_version {meta.get('obs_layout_version')} != {OBS_LAYOUT_VERSION}")
    if meta.get("task_name") != task.name:
        raise Refused(f"policy task {meta.get('task_name')!r} != {task.name!r}")
    if policy.obs_dim != task.obs.dim + GOAL_DIM or policy.act_dim != 6:
        raise Refused(f"policy dims ({policy.obs_dim}, {policy.act_dim}) != ({task.obs.dim + GOAL_DIM}, 6) of task "
                      f"{task.name!r}")  # fmt: skip


def task_for_policy(policy: Any, requested: str | None = None) -> Any:
    """The env's :class:`~so101_rl.env.tasks.TaskSpec`: ``requested`` (``--task``) or the export's ``task_name``,
    checked against the export (:func:`check_policy`). Reach takes the export's ``success_radius_mm`` (none: REACH's
    15 mm; plan §10 2026-10-14); die_to_zone is the shipped ``DIE_TO_ZONE``."""
    from so101_rl.env.tasks import DIE_TO_ZONE, REACH

    tasks = {"reach": REACH, "die_to_zone": DIE_TO_ZONE}
    name = requested or policy.meta.get("task_name")
    if name not in tasks:
        raise Refused(f"policy task {name!r} is not one of {sorted(tasks)}")
    task = tasks[name]
    check_policy(policy, task)
    radius_mm = policy.meta.get("success_radius_mm")  # the radius the policy was trained at (plan §10 2026-10-14)
    if task.zone is None and radius_mm is not None:
        task = dataclasses.replace(REACH, success_radius_m=float(radius_mm) / 1000.0)
    return task


def sim_params_problem(meta: dict[str, Any]) -> str | None:
    """Why the export's ``sim_params`` are not a fitted, verified release (None = release >= 1 and verified). A
    missing field (exports before 0.62) counts as unverified."""
    rel, verified = meta.get("sim_params_release"), meta.get("sim_params_verified")
    if rel is None:
        return "the export records no sim_params release"
    if int(rel) < 1:
        return f"sim_params release {rel} (the MJCF prior, v0)"
    if verified is not True:
        return f"sim_params release {rel} is not recorded as verified (sim_params_verified {verified!r})"
    return None


NO_TRAY_HINT: Final[str] = "pass --manual-die-reset (attended: you place the die before every episode)"


def check_die_on_servo(meta: dict[str, Any], task: Any, servo: ServoInfo, *, manual_die_reset: bool,
                       log: Any = None, no_tray_hint: str = NO_TRAY_HINT) -> None:  # fmt: skip
    """The die task's real-arm refusals (no-op for reach): the v0 / unverified ``sim_params`` waiver (plan §10
    2026-10-16: never deployed on the real arm; the sim twin is allowed) and, while no tray exists
    (``die_start.measured_on_hardware`` false, H9), the missing ``manual_die_reset``."""
    if task.zone is None:
        return
    log = log or _log
    problem = sim_params_problem(meta)
    if servo.is_real:
        if problem is not None:
            raise Refused(f"die_to_zone on the REAL arm refused: {problem}. A die policy trained on sim_params v0 / "
                          "unverified (the S4.6 waiver, plan §10 2026-10-16) is NEVER deployed on the real arm; "
                          "retrain on a verified release >= 1 (HS1-HS3), or run it on the sim twin "
                          "(servo --backend mujoco)")  # fmt: skip
        if not task.die_start.measured_on_hardware and not manual_die_reset:
            raise Refused("die_to_zone on the REAL arm: no tray resets the die (H9 not measured, die_start is a "
                          f"placeholder); {no_tray_hint}")
    elif problem is not None:
        log(f"note: {problem}; allowed on the sim twin only (never on the real arm, plan §10 2026-10-16)")


def object_table_name(config_path: Path) -> str:
    """The config's ``object_table_shm`` when it has one, else :data:`DEFAULT_OBJECT_TABLE`."""
    try:
        return str(_yaml_key(config_path, "object_table_shm"))
    except ValueError:
        return DEFAULT_OBJECT_TABLE


def check_object_table(table: Any, name: str, *, max_age_s: float = OBJECT_TABLE_MAX_AGE_S,
                       tries: int = 20, period_s: float = 0.05) -> None:  # fmt: skip
    """:class:`Refused` unless the object table has a readable record no older than ``max_age_s`` (same monotonic
    clock as perception) with the camera-moved fiducial OK; up to ``tries`` reads ``period_s`` apart (a torn read or
    a writer between frames is retried)."""
    from so101_rl.perception.object_table import new_object_table

    rec = new_object_table()
    why = "never written (or every read torn)"
    for k in range(tries):
        if k:
            time.sleep(period_s)
        if not table.read(rec):
            continue
        age = (time.monotonic_ns() - int(rec["t_mono_ns"][0])) / 1e9
        if age > max_age_s:
            why = f"its last record is {age:.1f} s old (> {max_age_s:g} s): the perception process is not writing"
            continue
        if not bool(rec["fiducial_ok"][0]):
            why = "the camera-moved fiducial is not OK (camera moved, or not checked yet)"
            continue
        return
    raise Refused(f"object table {name!r}: {why}")


def open_object_table(name: str, **kw: Any) -> Any:
    """Attach (reader: never unlinks) and :func:`check_object_table`; :class:`Refused` when no perception process
    created the segment or its data is stale."""
    from so101_rl.perception.object_table import ObjectTableShm

    try:
        table = ObjectTableShm.attach(name)
    except (FileNotFoundError, RuntimeError) as exc:
        raise Refused(f"no perception writer: object table {name!r} cannot be attached ({exc}); start the "
                      "perception process first") from None  # fmt: skip
    try:
        check_object_table(table, name, **kw)
    except BaseException:
        table.close()
        raise
    return table


class DieResetGate:
    """``--manual-die-reset`` (attended, until the H9 tray exists): :meth:`wait` blocks until the operator confirms
    that the die is at its start, by Enter on a terminal stdin or by creating ``<run dir>/die_ready`` (consumed).
    While waiting, ``<run dir>/die_reset_waiting`` holds the episode number. A ``die_ready`` left from before the
    prompt does not count. The caller keeps the servo alive (HOLD keep-alive) meanwhile."""

    def __init__(self, run_dir: Path, log: Any = None, *, stdin: Any = None, poll_s: float = 0.1) -> None:
        self.ready = Path(run_dir) / DIE_READY_FILE
        self.waiting = Path(run_dir) / DIE_WAITING_FILE
        self.log = log or _log
        self.stdin = sys.stdin if stdin is None else stdin
        self.poll_s = poll_s

    def _tty(self) -> bool:
        try:
            return bool(self.stdin.isatty())
        except (AttributeError, ValueError):
            return False

    def wait(self, episode: int) -> float:
        """Seconds waited."""
        import select

        self.ready.unlink(missing_ok=True)
        self.waiting.write_text(f"{episode}\n")
        self.log(f"episode {episode}: the arm is at home. Place the die at its start, then press Enter (or create "
                 f"{self.ready})")  # fmt: skip
        t0 = time.monotonic()
        tty = self._tty()
        try:
            while True:
                if self.ready.exists():
                    self.ready.unlink(missing_ok=True)
                    break
                if tty:
                    r, _, _ = select.select([self.stdin], [], [], self.poll_s)
                    if r:
                        self.stdin.readline()
                        break
                else:
                    time.sleep(self.poll_s)
        finally:
            self.waiting.unlink(missing_ok=True)
        return time.monotonic() - t0


# ------------------------------------------------------------------------------------------------ goals


def draw_goal_pairs(sampler: Any, seed: int, n: int) -> np.ndarray:
    """float64[n, 2, 3]: per episode (first draw, redraw), two sequential ``sampler.sample`` calls on
    ``default_rng(seed)`` (pair ``i`` depends only on the seed and the sampler, not on ``n`` or any measurement)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2, 3))
    for i in range(n):
        out[i, 0] = sampler.sample(rng)
        out[i, 1] = sampler.sample(rng)
    return out


class GoalList:
    """``GoalSampler`` over pre-drawn pairs: after ``select(i)`` the first ``sample`` is pair ``i``'s first draw and
    any further one (the env's near-start redraw) its redraw."""

    def __init__(self, pairs: np.ndarray) -> None:
        self.pairs = np.asarray(pairs, np.float64)
        self._i: int | None = None
        self.draws = 0

    def select(self, i: int) -> None:
        self._i = i
        self.draws = 0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if self._i is None:
            raise RuntimeError("GoalList.sample before select()")
        g = self.pairs[self._i, min(self.draws, 1)].copy()
        self.draws += 1
        return g

    @property
    def redrawn(self) -> bool:
        return self.draws >= 2


# ------------------------------------------------------------------------------------------------ statistics


def _rate(successes: int, n: int) -> dict[str, Any]:
    from so101_rl.rl.eval_sim import wilson_interval

    lo, hi = wilson_interval(successes, n)
    return {"n": n, "successes": successes, "success_rate": successes / n if n else None, "wilson95": [lo, hi]}


def _is_trip(e: dict[str, Any]) -> bool:
    return bool(e.get("tripped", bool(e.get("trips")))) or bool(e.get("servo_stalled", False))


def advance_check(eps: Sequence[dict[str, Any]], n: int) -> dict[str, Any]:
    """The ladder advance rule (plan §10, 2026-10-08): no trip / stall in the rung and >= ceil(0.8 n) successes."""
    num, den = ADVANCE_MIN_SUCCESS_FRACTION
    need = -(-num * n // den)
    successes = sum(bool(e["success"]) for e in eps)
    trips = sum(_is_trip(e) for e in eps)
    return {"min_successes": need, "successes": successes, "trip_episodes": trips,
            "passed": trips == 0 and successes >= need and len(eps) >= n}  # fmt: skip


def summarize(
    episodes: Sequence[dict[str, Any]], speeds: Sequence[float], *, protocol: str, ladder_complete: bool,
    min_n: int = GATE_RUNG_MIN_EPISODES,
) -> dict[str, Any]:
    """Per-rung and overall success + Wilson 95 %, and the fine-tune verdict (on the full-speed rung, n >= ``min_n``:
    reach 50, die 30; only for the zero-shot protocol with ``ladder_complete``: every planned episode run and the last
    run not halted). Die episodes (``visible_fraction``) add the rung's mean visible fraction."""
    rungs = []
    for s in speeds:
        eps = [e for e in episodes if e["speed"] == s]
        row = _rate(sum(bool(e["success"]) for e in eps), len(eps))
        row["speed"] = s
        errs = np.asarray([e["final_error_m"] for e in eps if e["final_error_m"] is not None], float)
        row["final_error_m_p50"] = float(np.median(errs)) if errs.size else None
        row["mean_steps"] = float(np.mean([e["steps"] for e in eps])) if eps else None
        row["trip_episodes"] = sum(_is_trip(e) for e in eps)
        if any("visible_fraction" in e for e in eps):
            row["visible_fraction_mean"] = float(np.mean([e["visible_fraction"] for e in eps]))
        rungs.append(row)
    overall = _rate(sum(bool(e["success"]) for e in episodes), len(episodes))
    gate = next((r for r in rungs if r["speed"] == 1.0), None)
    lo = gate["wilson95"][0] if gate is not None and gate["n"] else None
    zero_shot = protocol == "zero-shot"
    return {
        "rungs": rungs,
        "overall": overall,
        "verdict": {
            "rung_speed": 1.0,
            "rung_n": gate["n"] if gate is not None else 0,
            "wilson_lo": lo,
            "min_wilson_lo": FINETUNE_MIN_WILSON_LO,
            "min_n": min_n,
            "protocol": protocol,
            "ladder_complete": bool(ladder_complete),
            "finetune_eligible": bool(
                zero_shot
                and ladder_complete
                and lo is not None
                and lo >= FINETUNE_MIN_WILSON_LO
                and gate["n"] >= min_n
            ),
        },
    }


def _table(summary: dict[str, Any]) -> str:
    lines = [f"{'speed':>6} {'n':>4} {'succ':>5} {'rate':>7} {'wilson95':>17} {'err p50 mm':>11} {'trips':>6}"]
    for r in [*summary["rungs"], {**summary["overall"], "speed": "all"}]:
        if not r["n"]:
            continue
        lo, hi = r["wilson95"]
        err = r.get("final_error_m_p50")
        lines.append(
            f"{r['speed']!s:>6} {r['n']:>4} {r['successes']:>5} {r['success_rate']:>7.3f} [{lo:.3f}, {hi:.3f}] "
            f"{'' if err is None else f'{err * 1e3:.1f}':>11} {r.get('trip_episodes', ''):>6}"
        )
    v = summary["verdict"]
    lines.append(
        f"fine-tune verdict ({v['protocol']}, ladder {'complete' if v['ladder_complete'] else 'NOT complete'}, "
        f"speed 1.0, n={v['rung_n']}): wilson_lo "
        f"{'n/a' if v['wilson_lo'] is None else format(v['wilson_lo'], '.3f')} >= {v['min_wilson_lo']}, "
        f"n >= {v['min_n']} -> "
        f"{'ELIGIBLE' if v['finetune_eligible'] else 'not eligible'}"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ episodes


@dataclasses.dataclass
class EpisodeTrace:
    """One episode's transitions, buffered in memory during the 10 Hz loop (no file I/O there) and written as an F8
    episode file after it (:func:`write_transitions`). ``final_obs`` is the observation after the last step."""

    steps: list[Any] = dataclasses.field(default_factory=list)
    final_obs: dict[str, np.ndarray] | None = None


def _run_episode(env: Any, policy: Any, speed: float, *, goals: GoalList, index: int,
                 infer_ms: list[float], trace: EpisodeTrace | None = None) -> dict[str, Any]:  # fmt: skip
    from so101_rl.logging.episode_logger import StepRecord
    from so101_rl.safety.types import SmState

    goals.select(index)
    obs, _ = env.reset()
    goal = obs["desired_goal"].astype(float)
    trips = 0
    tripped = False
    stalled = False
    clamped_steps = 0
    steps = 0
    visible_steps = 0
    success = False
    sm = int(SmState.RUN)
    die = env.task.zone is not None
    while True:
        x = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        t0 = time.perf_counter()
        a = policy(x)
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        cmd = np.asarray(a, np.float32) * np.float32(speed)
        prev = {k: np.array(v, np.float32) for k, v in obs.items()} if trace is not None else None
        obs, r, terminated, truncated, info = env.step(cmd)
        steps += 1
        if trace is not None:
            last = terminated or truncated
            trace.steps.append(StepRecord(
                t_mono_ns=time.monotonic_ns(), obs=prev, action_cmd=cmd,
                applied_delta_ticks=info["applied_delta_ticks"], target_ticks=info["target_ticks"],
                present_ticks=info["present_ticks"], reward=float(r), terminated=bool(terminated),
                truncated=bool(truncated), trips=int(info["trips"]),
                verifier=(1 if info["is_success"] else 2) if last else 0,
            ))  # fmt: skip
        clamped_steps += int(info["action_clamped"])
        visible_steps += int(bool(info["tracker_visible"])) if die else 0
        sm = int(info["sm_state"])
        if "trip" in info:  # the servo left RUN or reported fault bits (fault_bits may be 0)
            trips = int(info["trip"])
            tripped = True
        stalled = stalled or bool(info.get("servo_stalled", False))
        if terminated or truncated:
            success = bool(info["is_success"])  # terminated also covers a terminal trip (plan §10 2026-10-08)
            break
    tripped = tripped or (not success and sm != int(SmState.RUN))
    if trace is not None:
        trace.final_obs = {k: np.array(v, np.float32) for k, v in obs.items()}
    err = float(np.linalg.norm(obs["achieved_goal"].astype(float) - obs["desired_goal"].astype(float)))
    ep: dict[str, Any] = {
        "goal": [round(float(g), 5) for g in goal],
        "goal_redrawn": goals.redrawn,
        "success": success,
        "steps": steps,
        "final_error_m": err,
        "trips": trips,
        "tripped": tripped,
        "sm_state": sm,
        "servo_stalled": stalled,
        "clamped_steps": clamped_steps,
        "speed": speed,
    }
    if die:  # the die's last tracked position (held while invisible); never seen this episode: none, no error
        seen = getattr(env, "_die_seen_ns", None) is not None
        ep.update({
            "final_error_m": err if seen else None,
            "visible_steps": visible_steps,
            "unknown_steps": steps - visible_steps,
            "visible_fraction": visible_steps / steps if steps else 0.0,
            "final_die_xyz": [round(float(v), 5) for v in obs["achieved_goal"]] if seen else None,
            "final_die_visible": bool(env._visible()),
        })  # fmt: skip
    return ep


def episode_outcome(ep: dict[str, Any]) -> str:
    """The F8 ``outcome`` of a finished deploy episode."""
    if ep["success"]:
        return "success"
    if ep["tripped"]:
        return "trip"
    return "failure" if ep["servo_stalled"] else "truncated"


def write_transitions(logger: Any, trace: EpisodeTrace, ep: dict[str, Any], max_delta_ticks: Any) -> Path:
    """Write one finished episode as an F8 file (``<run>/episodes/ep_NNNNNN.h5``). Row t = (obs before step t, the
    command sent, the servo's record after it, reward, terminated, truncated; ``t_mono_ns`` = the host's
    CLOCK_MONOTONIC when the step returned, the servo tick log's clock); the observation after the last step is
    ``meta.final_obs``, so every row has its next observation (:func:`so101_rl.logging.transitions.load_transitions`).
    """
    if trace.final_obs is None or not trace.steps:
        raise ValueError("write_transitions: the episode has no steps")
    logger.start_episode({
        "source": "deploy-policy", "deploy_episode": ep["episode"], "speed": ep["speed"], "goal": ep["goal"],
        "max_delta_ticks": np.asarray(max_delta_ticks).tolist(),
        "final_obs": {k: v.tolist() for k, v in trace.final_obs.items()},
    })  # fmt: skip
    for rec in trace.steps:
        logger.log_step(rec)
    return logger.end_episode(episode_outcome(ep))


def wait_for_run(backend: Any, timeout_s: float, log: Any = None, period_s: float = 0.1) -> None:
    """Send HOLD every ``period_s`` (the servo safe-stops after 0.5 s without any command once it is in RUN) until
    the servo reports RUN or HOLD. Raises ``ServoNotRunning`` if it is tripped / stopping, ``ServoStateStale`` on
    timeout. HOLD never moves the arm."""
    from so101_rl.env.backend import RUNNABLE, ServoNotRunning, ServoStateStale
    from so101_rl.safety.types import SmState
    from so101_rl.servo.shm import new_state

    st = new_state()
    t_next = time.monotonic()
    t_end = t_next + timeout_s
    told = False
    while True:
        _hold(backend)
        t_next += period_s  # an absolute 100 ms grid: the read below does not stretch the HOLD period
        time.sleep(max(0.0, t_next - time.monotonic()))
        if backend.read_state(st):
            sm = int(st["sm_state"][0])
            if sm in RUNNABLE:
                return
            if sm in (int(SmState.SAFE_STOPPING), int(SmState.TORQUE_OFF), int(SmState.LATCHED)):
                raise ServoNotRunning(f"servo is in {SmState(sm).name} (fault_bits {int(st['fault_bits'][0]):#x}): "
                                      "restart the servo")
            if not told and log is not None:
                log(f"waiting for the servo to reach RUN (now {SmState(sm).name}; on the arm: press the reset button)")
                told = True
        if time.monotonic() >= t_end:
            raise ServoStateStale(f"servo did not reach RUN within {timeout_s:.0f} s")


@contextlib.contextmanager
def zero_delta_keepalive(backend: Any, period_s: float = 0.1) -> Iterator[None]:
    """A zero-delta RUN command every ``period_s`` from a helper thread while the arm must stay still IN RUN (an
    operator wait between episodes). Not HOLD: inside RUN a HOLD keeps the servo alive but its stuck monitor keeps
    seeing the last policy / reset delta with the arm still and trips STUCK after 3 s (the 2026-10-16 point_tip hold
    and the 2026-10-23 feasibility manual-reset wait). A zero delta moves nothing and resets the stuck count."""
    from so101_rl.config import N_JOINTS
    from so101_rl.servo.shm import new_state

    stop = threading.Event()
    zero = np.zeros(N_JOINTS, dtype=np.float32)

    def _run() -> None:
        st = new_state()
        warned = False
        t_next = time.monotonic()
        while not stop.is_set():
            try:
                seq = int(st["seq"][0]) if backend.read_state(st) else 0
                backend.send_delta(zero, seq, int(getattr(backend, "episode", 0)))
            except Exception:  # noqa: BLE001 - the servo's stale-command stop handles the arm if this fails
                if not warned:
                    print("deploy-policy: zero-delta keep-alive could not write the shm command", file=sys.stderr,
                          flush=True)  # fmt: skip
                    warned = True
            t_next += period_s
            stop.wait(max(0.0, t_next - time.monotonic()))

    t = threading.Thread(target=_run, name="zero-delta-keepalive", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


@contextlib.contextmanager
def hold_keepalive(backend: Any, period_s: float = 0.1, *, on_late: Callable[[float], None] | None = None,
                   late_s: float = 0.15) -> Iterator[None]:
    """A HOLD command every ``period_s`` from a helper thread while the caller does slow set-up after the attach
    (nothing else may send meanwhile: ``So101GoalEnv.__init__`` only calls ``backend.fk()``). HOLD never moves the
    arm; it only keeps the servo from safe-stopping. ``on_late(gap_s)`` is called (from the helper thread) for every
    gap between two HOLDs longer than ``late_s`` (the servo's stale-command limit): the thread itself was starved."""
    stop = threading.Event()

    def _run() -> None:
        warned = False
        t_next = time.monotonic()
        t_prev: float | None = None
        while not stop.is_set():
            if not _hold(backend) and not warned:
                print("deploy-policy: HOLD keep-alive could not write the shm command", file=sys.stderr, flush=True)
                warned = True
            now = time.monotonic()
            if on_late is not None and t_prev is not None and now - t_prev > late_s:
                on_late(now - t_prev)
            t_prev = now
            t_next += period_s
            stop.wait(max(0.0, t_next - time.monotonic()))

    t = threading.Thread(target=_run, name="hold-keepalive", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


def _hold(backend: Any) -> bool:
    """Best effort (the servo's own stale-command logic handles the arm if this fails); False on failure."""
    try:
        backend.send_hold(0, int(getattr(backend, "episode", 0)))
    except Exception:  # noqa: BLE001 - on the way out / in the keep-alive: report, never raise
        return False
    return True


# ------------------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deploy-policy", description=__doc__.splitlines()[0])
    ap.add_argument("--policy", type=Path, default=None, help="policy export (export-policy output); required "
                    "unless --resume")  # fmt: skip
    ap.add_argument("--resume", type=Path, default=None, help="continue the stopped run in this run dir")
    ap.add_argument("--task", default=None, choices=("reach", "die_to_zone"),
                    help="default: the export's task_name (a different value is refused)")
    ap.add_argument("--object-table", default=None, help="die_to_zone: the perception object table shm (default: "
                    f"the config's object_table_shm, else {DEFAULT_OBJECT_TABLE})")
    ap.add_argument("--manual-die-reset", action="store_true",
                    help="die_to_zone, attended, until the H9 tray exists: wait before every episode until the "
                    f"operator places the die (Enter, or <run dir>/{DIE_READY_FILE})")  # fmt: skip
    ap.add_argument("--config", type=Path, default=None, help="so101.yaml (default: the running servo's)")
    ap.add_argument("--calibration", type=Path, default=None, help="default: the running servo's")
    ap.add_argument("--shm-name", default=None, help="the running servo's shm segment (default: from config)")
    ap.add_argument("--attended", action="store_true", help="Tier 0: a human is at the e-stop (required on the arm)")
    ap.add_argument("--accept-no-watchdog", action="store_true", help="with --attended: allow a real servo started "
                    "with --no-mcu (no e-stop hardware; the human's hand on the servo power switch; plan §10 2026-10-10)")
    ap.add_argument("--protocol", choices=("single", "zero-shot"), default=None, help="default single")
    ap.add_argument("--speed-scale", type=float, default=None, choices=SPEED_LADDER, help="single protocol (0.5)")
    ap.add_argument("--episodes", type=int, default=None, help="single protocol (10)")
    ap.add_argument("--rung-episodes", default=None,
                    help=f"zero-shot protocol: episodes at speed 0.5,0.75,1.0 ({DEFAULTS['rung_episodes']}; "
                    f"die_to_zone {','.join(map(str, DIE_RUNG_EPISODES))})")  # fmt: skip
    ap.add_argument("--seed", type=int, default=None, help="goal list seed (7)")
    ap.add_argument("--out-root", type=Path, default=Path("runs/deploy"))
    ap.add_argument("--wait-servo-s", type=float, default=120.0, help="wait for the servo process / RUN state")
    return ap


def _plan(args: argparse.Namespace) -> list[tuple[float, int]]:
    if args.protocol == "single":
        if args.episodes < 1:
            raise Refused("--episodes must be >= 1")
        return [(args.speed_scale, args.episodes)]
    try:
        counts = [int(c) for c in str(args.rung_episodes).split(",")]
    except ValueError:
        raise Refused(f"--rung-episodes must be integers, got {args.rung_episodes!r}") from None
    if len(counts) != len(SPEED_LADDER) or min(counts) < 0:
        raise Refused(f"--rung-episodes needs {len(SPEED_LADDER)} counts >= 0, got {args.rung_episodes!r}")
    plan = [(s, n) for s, n in zip(SPEED_LADDER, counts, strict=True) if n > 0]
    if not plan:
        raise Refused("--rung-episodes: no episodes to run")
    return plan


def _resolve_run_args(args: argparse.Namespace, orig: dict[str, Any] | None,
                      task_name: str = "reach") -> list[tuple[float, int]]:  # fmt: skip
    """Fill the protocol arguments (defaults, or the original run's on ``--resume``) and return the plan; on a
    resume any explicitly given value that differs from the run is refused. ``task_name`` die_to_zone defaults the
    ladder to :data:`DIE_RUNG_EPISODES` (HS6)."""
    if orig is None:
        if args.policy is None:
            raise Refused("--policy is required (or --resume <run dir>)")
        if task_name == "die_to_zone" and args.rung_episodes is None:
            args.rung_episodes = ",".join(map(str, DIE_RUNG_EPISODES))
        for k, v in DEFAULTS.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
        return _plan(args)
    orig_plan = [(float(p["speed"]), int(p["episodes"])) for p in orig["plan"]]
    if args.policy is not None and str(args.policy.resolve()) != orig["policy"]:
        raise Refused(f"--policy {args.policy} differs from the run's {orig['policy']}")
    args.policy = Path(orig["policy"])
    for k in ("protocol", "seed"):
        if getattr(args, k) is not None and getattr(args, k) != orig[k]:
            raise Refused(f"--{k} {getattr(args, k)} differs from the run's {orig[k]}")
        setattr(args, k, orig[k])
    given = [k for k in ("speed_scale", "episodes", "rung_episodes") if getattr(args, k) is not None]
    by_speed = dict(orig_plan)
    if args.protocol == "single":
        (s0, n0), = orig_plan  # fmt: skip
        args.speed_scale = s0 if args.speed_scale is None else args.speed_scale
        args.episodes = n0 if args.episodes is None else args.episodes
    elif args.rung_episodes is None:
        args.rung_episodes = ",".join(str(by_speed.get(s, 0)) for s in SPEED_LADDER)
    plan = _plan(args)
    if plan != orig_plan:
        raise Refused(f"{', '.join('--' + k.replace('_', '-') for k in given)} give the plan {plan}, the run's is "
                      f"{orig_plan}")  # fmt: skip
    return plan


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        orig = json.loads((run_dir / "summary.json").read_text())
        lines = (run_dir / "episodes.jsonl").read_text().splitlines() if (run_dir / "episodes.jsonl").exists() else []
        episodes = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise Refused(f"--resume {run_dir}: cannot read the run ({type(exc).__name__}: {exc})") from None
    missing = [k for k in ("policy", "protocol", "plan", "seed", "servo", "goals") if k not in orig]
    if missing:
        raise Refused(f"--resume {run_dir}: summary.json lacks {missing} (not a resumable deploy-policy run)")
    return orig, episodes


def _check_resumed_episodes(episodes: list[dict[str, Any]], schedule: list[float], pairs: np.ndarray) -> None:
    if len(episodes) > len(schedule):
        raise Refused(f"--resume: {len(episodes)} episodes recorded, the plan has {len(schedule)}")
    for i, e in enumerate(episodes):
        if e.get("episode") != i or e.get("speed") != schedule[i]:
            raise Refused(f"--resume: episodes.jsonl line {i} is episode {e.get('episode')} at speed "
                          f"{e.get('speed')}, the plan expects episode {i} at {schedule[i]}")  # fmt: skip
        want = pairs[i, 1 if e.get("goal_redrawn") else 0]
        if not np.allclose(e.get("goal", []), want, atol=2e-5):
            raise Refused(f"--resume: episode {i} goal {e.get('goal')} is not goal {want.round(5).tolist()} of the "
                          "seeded list")  # fmt: skip


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stopping = [False]

    def _on_signal(_signum: int, _frame: Any) -> None:
        if stopping[0]:  # the final HOLD and the summary are being written: ignore further signals
            return
        stopping[0] = True
        raise KeyboardInterrupt

    old = {s: signal.signal(s, _on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _main(args, stopping)
    except KeyboardInterrupt:
        _log("interrupted before the attach: no command was sent")
        return EXIT_INTERRUPTED
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _main(args: argparse.Namespace, stopping: list[bool]) -> int:
    # torch and the policy load BEFORE the servo is looked up; the goal table (it needs the servo's config and
    # calibration) is built after the attach under the HOLD keep-alive (see "ATTACH NOW" below)
    import torch

    from so101_rl.env.backend import ResetFailed, ServoNotRunning, ServoStateStale, ShmBackend
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.logging.episode_logger import EpisodeLogger

    torch.set_num_threads(1)
    try:
        from so101_rl.config.loader import DEFAULT_CONFIG_PATH
        from so101_rl.env.goals import BoxGoalSampler
        from so101_rl.rl.eval_sim import CAL_CHANGED_MSG, calibration_changed
        from so101_rl.rl.policy_export import load_policy
        from so101_rl.safety.fk import KinematicChain, So101FK

        def _load(path: Path) -> Any:
            try:
                return load_policy(path)
            except (ValueError, KeyError, RuntimeError, OSError) as exc:
                raise Refused(f"cannot use policy {path}: {exc}") from None

        orig, prev_episodes = _load_run(args.resume) if args.resume is not None else (None, [])
        if orig is not None and args.task is not None and args.task != orig.get("task", "reach"):
            raise Refused(f"--task {args.task} differs from the run's {orig.get('task', 'reach')}")
        policy = _load(args.policy) if orig is None and args.policy is not None else None  # its task: the defaults
        task_name = args.task or (orig.get("task", "reach") if orig is not None else
                                  policy.meta.get("task_name") if policy is not None else "reach")  # fmt: skip
        plan = _resolve_run_args(args, orig, task_name=str(task_name))
        policy = policy or _load(args.policy)
        task = task_for_policy(policy, args.task or (orig.get("task", "reach") if orig is not None else None))
        die = task.zone is not None
        if not die and (args.manual_die_reset or args.object_table is not None):
            raise Refused("--manual-die-reset / --object-table are for die_to_zone only")
        shm_name = args.shm_name or str(_yaml_key(args.config or DEFAULT_CONFIG_PATH, "shm_name"))
        rs = resolve_servo(shm_name, wait_s=args.wait_servo_s, attended=args.attended,
                           accept_no_watchdog=args.accept_no_watchdog, config=args.config,
                           calibration=args.calibration)  # fmt: skip
        servo, config_path, servo_cal, calibration, cal_sha = (rs.servo, rs.config_path, rs.calibration_path,
                                                               rs.calibration, rs.calibration_sha256)  # fmt: skip
        if orig is not None and bool(orig["servo"].get("real")) != servo.is_real:
            raise Refused(f"--resume: the run was on a {'REAL' if orig['servo'].get('real') else 'sim'} servo, this "
                          f"one is {'REAL' if servo.is_real else 'sim'}")  # fmt: skip
        check_die_on_servo(policy.meta, task, servo, manual_die_reset=args.manual_die_reset)
        if orig is not None and orig.get("calibration_sha256") not in (None, cal_sha):
            raise Refused(f"--resume: the calibration {servo_cal} changed since the run started (sha256 differs); "
                          "start a new run")  # fmt: skip
        cal_changed = calibration_changed(policy.meta.get("calibration_sha256"), cal_sha)
        if cal_changed:
            _log(f"WARNING: {CAL_CHANGED_MSG} ({servo_cal})")
        elif cal_changed is None:
            _log("note: the policy records no calibration sha256 (trained before it was recorded); "
                 "make sure it was re-evaluated under the current calibration")  # fmt: skip
        cfg = rs.cfg
        fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
        schedule = [s for s, n in plan for _ in range(n)]
        table: Any = None
        table_name: str | None = None
        if die:  # read-only, before the attach (the last refusal here): no perception writer or stale data
            table_name = args.object_table or (orig or {}).get("object_table") or object_table_name(config_path)
            table = open_object_table(table_name)
    except Refused as exc:
        _log(f"refused: {exc}")
        return EXIT_REFUSED
    except (ValueError, OSError, KeyError, TypeError) as exc:  # a malformed config, calibration or run dir
        _log(f"refused: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    servo_doc = {"pid": servo.pid, "backend": servo.backend, "real": servo.is_real, "mcu": servo.mcu,
                 "no_watchdog_accepted": bool(servo.is_real and servo.no_mcu),
                 "shm_name": shm_name, "cmdline": servo.cmdline}  # fmt: skip
    if orig is None:
        out = args.out_root / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
        meta: dict[str, Any] = {
            "policy": str(args.policy.resolve()),
            "policy_meta": {k: v for k, v in policy.meta.items() if isinstance(v, (str, int, float, list, type(None)))},
            "task": task.name,
            "success_radius_m": task.success_radius_m,
            "protocol": args.protocol,
            "plan": [{"speed": s, "episodes": n} for s, n in plan],
            "seed": args.seed,
            "attended": bool(args.attended),
            "servo": servo_doc,
            "config": str(Path(config_path).resolve()),
            "calibration": str(servo_cal),
            "calibration_sha256": cal_sha,
            "calibration_changed_since_training": cal_changed,
            "tick_log": None if servo.tick_log is None else str(servo.tick_log),
            "started": started,
            "goals": None,  # drawn after the attach (below)
            "resumes": [],
        }  # fmt: skip
        if die:
            meta.update(object_table=table_name, manual_die_reset=bool(args.manual_die_reset),
                        unknown_halt_fraction=UNKNOWN_HALT_FRACTION)  # fmt: skip
    else:
        out = args.resume
        meta = {k: orig[k] for k in RUN_META_KEYS if k in orig}
        meta["resumes"] = [*orig.get("resumes", []), {
            "started": started, "servo_pid": servo.pid, "servo": servo_doc, "attended": bool(args.attended),
            "tick_log": None if servo.tick_log is None else str(servo.tick_log), "first_episode": len(prev_episodes),
        }]  # fmt: skip
        if die:
            meta["resumes"][-1]["manual_die_reset"] = bool(args.manual_die_reset)
    episodes: list[dict[str, Any]] = list(prev_episodes)
    infer_ms: list[float] = []
    halt: str | None = None
    halt_reason: str | None = None
    backend = env = None

    def _doc(code: int | None) -> dict[str, Any]:
        done = len(episodes) == len(schedule)
        doc = dict(meta)
        doc.update(summarize(episodes, [s for s, _ in plan], protocol=args.protocol,
                             ladder_complete=done and halt is None and code is not None,
                             min_n=DIE_GATE_RUNG_MIN_EPISODES if die else GATE_RUNG_MIN_EPISODES))  # fmt: skip
        inf = np.asarray(infer_ms, float)
        doc.update({
            "finished": None if code is None else datetime.now().astimezone().isoformat(timespec="seconds"),
            "halted": halt,
            "halt_reason": halt_reason,
            "exit_code": code,
            "inference_ms": {"p50": float(np.median(inf)), "max": float(inf.max())} if inf.size else None,
            "policy_overruns": int(backend.overruns) if backend is not None else 0,
            "episodes_file": str(out / "episodes.jsonl"),
        })
        return doc

    # ATTACH NOW, before anything slow. The servo was found and classified (read-only), so it may reach RUN at any
    # moment (``servo --backend mujoco`` presses its own button; on the arm the human does) and, once in RUN, it
    # safe-stops after 0.5 s without a command (STALE_CMD). Invariant: from the attach until the servo is in RUN a
    # HOLD goes out at least every 100 ms -- the keep-alive thread through the slow set-up (the goal table, ~7 s on
    # a cold cache; the env; the first summary), then ``wait_for_run`` on the main thread.
    keepalive = contextlib.ExitStack()
    try:
        backend = ShmBackend(cfg, calibration=calibration)
    except (FileNotFoundError, RuntimeError) as exc:
        _log(f"refused: cannot attach to the servo shm {shm_name!r}: {exc}")
        if table is not None:
            table.close()
        return EXIT_REFUSED
    keepalive.enter_context(hold_keepalive(backend))
    try:
        if die:  # the fixed zone goal, no sampler (the seed is still recorded)
            pairs = np.broadcast_to(task.zone.goal_xyz(), (len(schedule), 2, 3)).astype(np.float64)
        else:
            pairs = draw_goal_pairs(BoxGoalSampler(task.goal_bounds, fk=fk, limits=cfg.joints, seed=0), args.seed,
                                    len(schedule))  # fmt: skip
        if orig is not None:
            if not np.allclose(np.asarray(orig["goals"]["pairs"], float), pairs, atol=1e-9):
                raise Refused("--resume: the goal list drawn from the seed differs from the run's (config, "
                              "calibration or goal table changed)")  # fmt: skip
            _check_resumed_episodes(prev_episodes, schedule, pairs)
        else:
            rule = ("die_to_zone: the fixed zone goal (zone centre at the resting die height) for every episode; no "
                    "sampler, the seed is only recorded" if die else "per episode (first draw, redraw) from "
                    "default_rng(seed); the redraw is used if the first is within 2 success radii of the measured "
                    "start tip")  # fmt: skip
            meta["goals"] = {"seed": args.seed, "rule": rule, "pairs": pairs.tolist()}
        env = So101GoalEnv(backend, cfg, task, goal_sampler=GoalList(pairs), object_table=table)
        out.mkdir(parents=True, exist_ok=orig is not None)
        _write_json(out / "summary.json", _doc(None))  # before RUN: a resumable record even if killed
        if servo.tick_log is None:
            _log("the servo runs without --tick-log: no tick log for this deployment")
        _log(f"servo pid {servo.pid} --backend {servo.backend} ({'REAL ARM, attended' if servo.is_real else 'sim'}); "
             f"out {out}" + (f"; resuming at episode {len(prev_episodes)} of {len(schedule)}" if orig else ""))
    except (Refused, ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as exc:
        stopping[0] = True
        keepalive.close()
        _hold(backend)  # only HOLD was ever sent; the servo safe-stops 0.5 s after it reaches RUN
        with contextlib.suppress(Exception):
            (env or backend).close()
        if table is not None:
            table.close()
        if isinstance(exc, KeyboardInterrupt):
            _log("interrupted during the set-up after the attach: only HOLD was sent")
            return EXIT_INTERRUPTED
        _log(f"refused: {exc}" if isinstance(exc, Refused) else f"refused: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED
    rung_of = [k for k, (_s, n) in enumerate(plan) for _ in range(n)]
    rung_start = [sum(n for _s, n in plan[:k]) for k in range(len(plan))]
    code = EXIT_OK
    transitions = EpisodeLogger(out.parent, out.name, keep_full_frames="never")
    gate = DieResetGate(out) if die and args.manual_die_reset else None
    try:
        goals = env.goal_sampler
        with (out / "episodes.jsonl").open("a") as f:
            try:
                keepalive.close()  # wait_for_run's own HOLD loop takes over at once
                wait_for_run(backend, args.wait_servo_s, _log)
                again = servo_info(shm_name)  # the owner must still be the servo we classified (no swap meanwhile)
                if again.pid != servo.pid or again.is_real != servo.is_real:
                    halt_reason = "servo_changed"
                    raise ServoNotRunning(f"the servo on shm {shm_name!r} changed (pid {servo.pid} -> {again.pid})")
            except (ServoNotRunning, ServoStateStale, Refused) as exc:
                halt = f"{type(exc).__name__}: {exc}"
                halt_reason = halt_reason or "servo_not_running"
                todo: range = range(0)
            else:
                todo = range(len(episodes), len(schedule))
            for i in todo:
                k = rung_of[i]
                if i == rung_start[k] and k > 0 and args.protocol == "zero-shot":
                    prev_speed, prev_n = plan[k - 1]
                    chk = advance_check([e for e in episodes if e["speed"] == prev_speed], prev_n)
                    if not chk["passed"]:
                        halt = (f"rung {prev_speed}: {chk['successes']}/{prev_n} successes (need "
                                f"{chk['min_successes']}), {chk['trip_episodes']} trip / stall episodes: not "
                                f"climbing to {plan[k][0]}")  # fmt: skip
                        halt_reason = "rung_below_threshold"
                        break
                speed = schedule[i]
                trace = EpisodeTrace()
                try:
                    if gate is not None:  # arm home and held, then the operator places the die (attended)
                        backend.reset(np.asarray(cfg.safe_state.home_ticks))
                        with zero_delta_keepalive(backend):  # not HOLD: STUCK after 3 s in RUN
                            gate.wait(i)
                    if die:
                        check_object_table(table, str(table_name))
                    ep = _run_episode(env, policy, speed, goals=goals, index=i, infer_ms=infer_ms, trace=trace)
                except (ServoNotRunning, ResetFailed, ServoStateStale) as exc:
                    halt = f"{type(exc).__name__}: {exc}"
                    halt_reason = "reset_failed" if isinstance(exc, ResetFailed) else "servo_not_running"
                    break
                except Refused as exc:  # check_object_table: perception stopped or the camera moved
                    halt = f"before episode {i}: {exc}"
                    halt_reason = "perception_stale"
                    break
                ep["episode"] = i
                with hold_keepalive(backend):  # the env sent HOLD at the episode end; keep it fresh while writing
                    path = write_transitions(transitions, trace, ep, cfg.action.max_delta_ticks)
                ep["transitions"] = str(path.relative_to(out))
                episodes.append(ep)
                f.write(json.dumps(ep) + "\n")
                f.flush()
                e_mm = "   n/a" if ep["final_error_m"] is None else f"{ep['final_error_m'] * 1e3:6.1f}"
                print(f"ep {i:3d} speed {speed:4.2f} {'SUCCESS' if ep['success'] else 'fail   '} "
                      f"steps {ep['steps']:3d} err {e_mm} mm"
                      + (f" visible {ep['visible_fraction']:.2f}" if die else "")
                      + (f" TRIP {ep['trips']:#x}" if ep["tripped"] else "")
                      + (" STALLED" if ep["servo_stalled"] else ""), flush=True)  # fmt: skip
                if ep["tripped"] or ep["servo_stalled"]:
                    halt = (f"episode {i}: servo left RUN (sm_state {ep['sm_state']}, fault_bits {ep['trips']:#x})"
                            if ep["tripped"] else f"episode {i}: servo stalled")  # fmt: skip
                    halt_reason = "trip" if ep["tripped"] else "stall"
                    break
                if die and ep["unknown_steps"] > UNKNOWN_HALT_FRACTION * ep["steps"]:
                    halt = (f"episode {i}: the tracker reported the die UNKNOWN on {ep['unknown_steps']}/{ep['steps']} "
                            f"steps (> {UNKNOWN_HALT_FRACTION:.0%}): a human must check the die, the camera and the "
                            "perception process")  # fmt: skip
                    halt_reason = "tracker_unknown"
                    break
        if halt is not None:
            _log(f"ladder stopped: {halt}" + ("; restart the servo and rerun with --resume " + str(out)
                                               if halt_reason != "rung_below_threshold" else ""))  # fmt: skip
            code = EXIT_HALTED
    except KeyboardInterrupt:
        halt = "interrupted (Ctrl-C / SIGTERM)"
        halt_reason = "interrupted"
        code = EXIT_INTERRUPTED
        _log("interrupted: HOLD sent, no further commands; the servo safe-stops the arm to rest 0.5 s later")
    except Exception as exc:  # noqa: BLE001 - still HOLD and write the partial summary, then exit 1
        halt = f"error: {exc!r}"
        halt_reason = "error"
        code = EXIT_HALTED
        _log(f"stopped by an error: {exc!r}")
    finally:
        stopping[0] = True  # from here on SIGINT / SIGTERM are ignored: the HOLD and the summary must be written
        keepalive.close()
        if backend is not None:
            _hold(backend)  # last command is always HOLD (never a stale RUN)
        transitions.close()  # an episode interrupted mid-write stays an ep_*.h5.tmp, never an .h5
        with contextlib.suppress(Exception):
            if env is not None:
                env.close()
            elif backend is not None:
                backend.close()
        if table is not None:
            table.close()  # a reader: never unlinks the perception process's segment
    summary = _doc(code)
    _write_json(out / "summary.json", summary)
    print(_table(summary))
    print(f"run dir {out}  tick log {meta['tick_log']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
