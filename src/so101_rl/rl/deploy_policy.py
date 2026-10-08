"""Policy deployment and the zero-shot reach evaluation (MVP M7; sim plan S5.1, reach part).

``pixi run deploy-policy --policy runs/lab/<run>/policy.pt --protocol zero-shot [--attended]``
``pixi run deploy-policy --resume runs/deploy/<ts> [--attended]`` (continue a stopped run)

Runs an exported policy (``rl/policy_export.py``) at 10 Hz through :class:`~so101_rl.env.goal_env.So101GoalEnv` on a
:class:`~so101_rl.env.backend.ShmBackend` attached to an ALREADY-RUNNING servo process: on the real arm the human
starts ``servo --backend feetech`` (Tier 0, attended); in tests and for the sim twin it is ``servo --backend mujoco``.
This script never starts a servo and never opens a serial device; it only writes shm commands.

Everything that sim and real must agree on is the env's: the FS3 observation, ``tasks.action_to_delta_np``, the
reach goal rule (``BoxGoalSampler`` with the near-start redraw) and the success / termination predicate. The policy
input is ``observation ++ desired_goal`` (as ``eval_sim``); the action is ``policy(x) * speed_scale`` (the ladder
scales the delta magnitude; the env then clips and maps it).

* **Goal list:** drawn from ``--seed`` BEFORE the attach, independent of any measurement: per episode ``i`` a pair
  (first draw, redraw), two sequential ``BoxGoalSampler.sample`` calls on ``default_rng(seed)``, so pair ``i``
  depends only on the seed, the config and the calibration (the goal table). At the reset the env applies its
  near-start rule to the MEASURED start tip: the first draw, or the redraw if the first is within 2 success radii
  of the tip. A sim twin with the same seed, config and calibration therefore gets the same pairs; which member of
  a pair is used can differ between sim and real only when the first draw is near that radius. The pairs are in
  ``summary.json`` (``goals``), the goal used and ``goal_redrawn`` per episode in ``episodes.jsonl``.
* **Episode end:** at success (``terminated``; the env sends HOLD, no further RUN command until the next reset), at
  the 100-step limit, or when the servo left RUN for a non-success reason (a trip, whatever its ``fault_bits``) or
  stalled (``truncated``). Between episodes ``ShmBackend.reset`` (HOLD, then a bounded ramp to the start pose).
* **Real or sim:** decided from the servo that owns the shm segment: its single-instance lock file names the pid,
  ``/proc/<pid>/cmdline`` is parsed with the servo's OWN parser (``servo.main.build_parser``, no abbreviations), and
  fail-closed: anything but ``--backend mujoco`` / ``fake`` is the real arm and needs ``--attended`` (Tier 0: a human
  at the e-stop); a command line that does not parse cleanly (unknown or abbreviated options) is refused even with
  ``--attended``, and so is a real servo started with ``--no-mcu``. The servo's ``--config`` (default for and
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
  process, then sends HOLD every 100 ms (never moves the arm) until RUN. Real arm: start the servo (it waits for the
  reset button), start this script, then press the button. ``servo --backend mujoco`` presses it itself: start this
  script first.
* **Outputs** ``<out-root>/<ts>/``: ``episodes.jsonl`` (one line per episode, appended as it ends), ``summary.json``
  (written before the attach and at the end: per rung and overall success rate + Wilson 95 %, the verdict, the goal
  list, the servo's tick-log path, inference timing).
* **Ctrl-C / SIGTERM** (handled from the start of ``main``): before the attach exit 130 with nothing sent; after it,
  stop sending RUN commands, send HOLD, write what was collected, exit 130. Further signals are ignored while the
  final HOLD and the summary are written. The servo's own stop / safe-state logic handles the arm. The same holds
  after a normal end: the last command is HOLD, and 0.5 s later the servo runs its safe-state sequence to rest (a
  fresh servo process is needed for the next deployment).

Exit codes: 0 done (whatever the success rate), 1 the ladder stopped (trip / stall / failed reset / rung below the
advance threshold), 2 refused (bad policy, no servo, real arm without ``--attended``, ...), 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "ADVANCE_MIN_SUCCESS_FRACTION",
    "DEFAULT_RUNG_EPISODES",
    "FINETUNE_MIN_WILSON_LO",
    "SPEED_LADDER",
    "GoalList",
    "ServoInfo",
    "advance_check",
    "check_policy",
    "draw_goal_pairs",
    "main",
    "parse_servo_cmdline",
    "servo_info",
    "summarize",
]

SPEED_LADDER: Final[tuple[float, ...]] = (0.5, 0.75, 1.0)
DEFAULT_RUNG_EPISODES: Final[tuple[int, ...]] = (10, 10, 50)
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
    "policy", "policy_meta", "task", "protocol", "plan", "seed", "attended", "servo", "config", "calibration",
    "tick_log", "started", "goals", "resumes",
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


def check_policy(policy: Any, task: Any) -> None:
    """Refuse a policy for another task or input layout (``load_policy`` already checked the layout version)."""
    from so101_rl.env.obs import GOAL_DIM, OBS_LAYOUT_VERSION

    meta = policy.meta
    if meta.get("obs_layout_version") != OBS_LAYOUT_VERSION:
        raise Refused(f"policy obs_layout_version {meta.get('obs_layout_version')} != {OBS_LAYOUT_VERSION}")
    if meta.get("task_name") != task.name:
        raise Refused(f"policy task {meta.get('task_name')!r} != {task.name!r}")
    if policy.obs_dim != task.obs.dim + GOAL_DIM or policy.act_dim != 6:
        raise Refused(f"policy dims ({policy.obs_dim}, {policy.act_dim}) != ({task.obs.dim + GOAL_DIM}, 6)")


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
    episodes: Sequence[dict[str, Any]], speeds: Sequence[float], *, protocol: str, ladder_complete: bool
) -> dict[str, Any]:
    """Per-rung and overall success + Wilson 95 %, and the fine-tune verdict (on the full-speed rung; only for the
    zero-shot protocol with ``ladder_complete``: every planned episode run and the last run not halted)."""
    rungs = []
    for s in speeds:
        eps = [e for e in episodes if e["speed"] == s]
        row = _rate(sum(bool(e["success"]) for e in eps), len(eps))
        row["speed"] = s
        errs = np.asarray([e["final_error_m"] for e in eps], float)
        row["final_error_m_p50"] = float(np.median(errs)) if errs.size else None
        row["mean_steps"] = float(np.mean([e["steps"] for e in eps])) if eps else None
        row["trip_episodes"] = sum(_is_trip(e) for e in eps)
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
            "min_n": GATE_RUNG_MIN_EPISODES,
            "protocol": protocol,
            "ladder_complete": bool(ladder_complete),
            "finetune_eligible": bool(
                zero_shot
                and ladder_complete
                and lo is not None
                and lo >= FINETUNE_MIN_WILSON_LO
                and gate["n"] >= GATE_RUNG_MIN_EPISODES
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


def _run_episode(env: Any, policy: Any, speed: float, *, goals: GoalList, index: int,
                 infer_ms: list[float]) -> dict[str, Any]:  # fmt: skip
    from so101_rl.safety.types import SmState

    goals.select(index)
    obs, _ = env.reset()
    goal = obs["desired_goal"].astype(float)
    trips = 0
    tripped = False
    stalled = False
    clamped_steps = 0
    steps = 0
    success = False
    sm = int(SmState.RUN)
    while True:
        x = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        t0 = time.perf_counter()
        a = policy(x)
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        obs, _r, terminated, truncated, info = env.step(a * np.float32(speed))
        steps += 1
        clamped_steps += int(info["action_clamped"])
        sm = int(info["sm_state"])
        if "trip" in info:  # the servo left RUN or reported fault bits (fault_bits may be 0)
            trips = int(info["trip"])
            tripped = True
        stalled = stalled or bool(info.get("servo_stalled", False))
        if terminated or truncated:
            success = bool(terminated)
            break
    tripped = tripped or (not success and sm != int(SmState.RUN))
    err = float(np.linalg.norm(obs["achieved_goal"].astype(float) - obs["desired_goal"].astype(float)))
    return {
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


def wait_for_run(backend: Any, timeout_s: float, log: Any = None) -> None:
    """Send HOLD every policy period (the servo safe-stops after 0.5 s without any command once it is in RUN) until
    the servo reports RUN or HOLD. Raises ``ServoNotRunning`` if it is tripped / stopping, ``ServoStateStale`` on
    timeout. HOLD never moves the arm."""
    from so101_rl.env.backend import RUNNABLE, ServoNotRunning, ServoStateStale
    from so101_rl.safety.types import SmState
    from so101_rl.servo.shm import new_state

    st = new_state()
    t_end = time.monotonic() + timeout_s
    told = False
    while True:
        _hold(backend)
        time.sleep(0.1)
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
def hold_keepalive(backend: Any, period_s: float = 0.1) -> Iterator[None]:
    """A HOLD command every ``period_s`` from a helper thread while the caller does slow set-up after the attach
    (nothing else may send meanwhile: ``So101GoalEnv.__init__`` only calls ``backend.fk()``). HOLD never moves the
    arm; it only keeps the servo from safe-stopping."""
    stop = threading.Event()

    def _run() -> None:
        warned = False
        while not stop.is_set():
            if not _hold(backend) and not warned:
                print("deploy-policy: HOLD keep-alive could not write the shm command", file=sys.stderr, flush=True)
                warned = True
            stop.wait(period_s)

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
    ap.add_argument("--task", default="reach", choices=("reach",))
    ap.add_argument("--config", type=Path, default=None, help="so101.yaml (default: the running servo's)")
    ap.add_argument("--calibration", type=Path, default=None, help="default: the running servo's")
    ap.add_argument("--shm-name", default=None, help="the running servo's shm segment (default: from config)")
    ap.add_argument("--attended", action="store_true", help="Tier 0: a human is at the e-stop (required on the arm)")
    ap.add_argument("--protocol", choices=("single", "zero-shot"), default=None, help="default single")
    ap.add_argument("--speed-scale", type=float, default=None, choices=SPEED_LADDER, help="single protocol (0.5)")
    ap.add_argument("--episodes", type=int, default=None, help="single protocol (10)")
    ap.add_argument("--rung-episodes", default=None,
                    help=f"zero-shot protocol: episodes at speed 0.5,0.75,1.0 ({DEFAULTS['rung_episodes']})")  # fmt: skip
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


def _resolve_run_args(args: argparse.Namespace, orig: dict[str, Any] | None) -> list[tuple[float, int]]:
    """Fill the protocol arguments (defaults, or the original run's on ``--resume``) and return the plan; on a
    resume any explicitly given value that differs from the run is refused."""
    if orig is None:
        if args.policy is None:
            raise Refused("--policy is required (or --resume <run dir>)")
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
    # everything slow (torch, the policy, the goal table) happens BEFORE attaching: once the servo is in RUN it
    # safe-stops after 0.5 s without a command, so the HOLD keep-alive must start right after the attach
    import torch

    from so101_rl.env.backend import ResetFailed, ServoNotRunning, ServoStateStale, ShmBackend
    from so101_rl.env.goal_env import So101GoalEnv

    torch.set_num_threads(1)
    try:
        from so101_rl.config import load_calibration, load_config
        from so101_rl.config.loader import DEFAULT_CONFIG_PATH
        from so101_rl.env.goals import BoxGoalSampler
        from so101_rl.env.tasks import REACH
        from so101_rl.rl.policy_export import load_policy
        from so101_rl.safety.fk import KinematicChain, So101FK

        orig, prev_episodes = _load_run(args.resume) if args.resume is not None else (None, [])
        plan = _resolve_run_args(args, orig)
        try:
            policy = load_policy(args.policy)
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            raise Refused(f"cannot use policy {args.policy}: {exc}") from None
        check_policy(policy, REACH)
        shm_name = args.shm_name or str(_yaml_key(args.config or DEFAULT_CONFIG_PATH, "shm_name"))
        _log(f"looking for the servo on shm {shm_name!r} (up to {args.wait_servo_s:.0f} s)")
        servo = servo_info(shm_name, wait_s=args.wait_servo_s)
        if servo.problem is not None:
            raise Refused(f"the servo on shm {shm_name!r} (pid {servo.pid}) has a command line that does not parse "
                          f"cleanly ({servo.problem}): treated as the REAL arm and refused even with --attended")  # fmt: skip
        if servo.is_real and not args.attended:
            raise Refused(
                f"the servo on shm {shm_name!r} (pid {servo.pid}, --backend {servo.backend}) drives the REAL arm: "
                "Tier 0 needs a human at the e-stop; pass --attended"
            )
        if servo.is_real and servo.no_mcu:
            raise Refused(f"the REAL-arm servo (pid {servo.pid}) runs with --no-mcu (no watchdog MCU): restart it "
                          "with the MCU (--heartbeat-port)")  # fmt: skip
        if orig is not None and bool(orig["servo"].get("real")) != servo.is_real:
            raise Refused(f"--resume: the run was on a {'REAL' if orig['servo'].get('real') else 'sim'} servo, this "
                          f"one is {'REAL' if servo.is_real else 'sim'}")  # fmt: skip
        config_path = servo.config or DEFAULT_CONFIG_PATH
        if args.config is not None and args.config.expanduser().resolve() != Path(config_path).resolve():
            raise Refused(f"--config {args.config} differs from the servo's {config_path}")
        servo_cal = servo.calibration or (
            servo.cwd / Path(str(_yaml_key(config_path, "calibration_path"))).expanduser()
        ).resolve()
        if args.calibration is not None and args.calibration.expanduser().resolve() != servo_cal:
            raise Refused(f"--calibration {args.calibration} differs from the servo's {servo_cal}")
        calibration = load_calibration(servo_cal)
        cfg = dataclasses.replace(load_config(config_path, calibration=calibration), shm_name=shm_name)
        fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)
        schedule = [s for s, n in plan for _ in range(n)]
        pairs = draw_goal_pairs(BoxGoalSampler(REACH.goal_bounds, fk=fk, limits=cfg.joints, seed=0), args.seed,
                                len(schedule))  # fmt: skip
        if orig is not None:
            if not np.allclose(np.asarray(orig["goals"]["pairs"], float), pairs, atol=1e-9):
                raise Refused("--resume: the goal list drawn from the seed differs from the run's (config, "
                              "calibration or goal table changed)")  # fmt: skip
            _check_resumed_episodes(prev_episodes, schedule, pairs)
    except Refused as exc:
        _log(f"refused: {exc}")
        return EXIT_REFUSED
    except (ValueError, OSError, KeyError, TypeError) as exc:  # a malformed config, calibration or run dir
        _log(f"refused: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    servo_doc = {"pid": servo.pid, "backend": servo.backend, "real": servo.is_real, "mcu": servo.mcu,
                 "shm_name": shm_name, "cmdline": servo.cmdline}  # fmt: skip
    if orig is None:
        out = args.out_root / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
        out.mkdir(parents=True, exist_ok=False)
        meta: dict[str, Any] = {
            "policy": str(args.policy.resolve()),
            "policy_meta": {k: v for k, v in policy.meta.items() if isinstance(v, (str, int, float, list, type(None)))},
            "task": REACH.name,
            "protocol": args.protocol,
            "plan": [{"speed": s, "episodes": n} for s, n in plan],
            "seed": args.seed,
            "attended": bool(args.attended),
            "servo": servo_doc,
            "config": str(Path(config_path).resolve()),
            "calibration": str(servo_cal),
            "tick_log": None if servo.tick_log is None else str(servo.tick_log),
            "started": started,
            "goals": {"seed": args.seed, "rule": "per episode (first draw, redraw) from default_rng(seed); the redraw "
                      "is used if the first is within 2 success radii of the measured start tip",
                      "pairs": pairs.tolist()},
            "resumes": [],
        }  # fmt: skip
    else:
        out = args.resume
        meta = {k: orig[k] for k in RUN_META_KEYS if k in orig}
        meta["resumes"] = [*orig.get("resumes", []), {
            "started": started, "servo_pid": servo.pid, "servo": servo_doc, "attended": bool(args.attended),
            "tick_log": None if servo.tick_log is None else str(servo.tick_log), "first_episode": len(prev_episodes),
        }]  # fmt: skip
    episodes: list[dict[str, Any]] = list(prev_episodes)
    infer_ms: list[float] = []
    halt: str | None = None
    halt_reason: str | None = None
    backend = env = None

    def _doc(code: int | None) -> dict[str, Any]:
        done = len(episodes) == len(schedule)
        doc = dict(meta)
        doc.update(summarize(episodes, [s for s, _ in plan], protocol=args.protocol,
                             ladder_complete=done and halt is None and code is not None))  # fmt: skip
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

    _write_json(out / "summary.json", _doc(None))  # before the attach: a resumable record even if killed
    if servo.tick_log is None:
        _log("the servo runs without --tick-log: no tick log for this deployment")
    _log(f"servo pid {servo.pid} --backend {servo.backend} ({'REAL ARM, attended' if servo.is_real else 'sim'}); "
         f"out {out}" + (f"; resuming at episode {len(prev_episodes)} of {len(schedule)}" if orig else ""))
    rung_of = [k for k, (_s, n) in enumerate(plan) for _ in range(n)]
    rung_start = [sum(n for _s, n in plan[:k]) for k in range(len(plan))]
    code = EXIT_OK
    try:
        backend = ShmBackend(cfg, calibration=calibration)
        with hold_keepalive(backend):  # the env set-up after the attach: keep the servo commanded meanwhile
            env = So101GoalEnv(backend, cfg, REACH, goal_sampler=GoalList(pairs))
        goals = env.goal_sampler
        with (out / "episodes.jsonl").open("a") as f:
            try:
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
                try:
                    ep = _run_episode(env, policy, speed, goals=goals, index=i, infer_ms=infer_ms)
                except (ServoNotRunning, ResetFailed, ServoStateStale) as exc:
                    halt = f"{type(exc).__name__}: {exc}"
                    halt_reason = "reset_failed" if isinstance(exc, ResetFailed) else "servo_not_running"
                    break
                ep["episode"] = i
                episodes.append(ep)
                f.write(json.dumps(ep) + "\n")
                f.flush()
                print(f"ep {i:3d} speed {speed:4.2f} {'SUCCESS' if ep['success'] else 'fail   '} "
                      f"steps {ep['steps']:3d} err {ep['final_error_m'] * 1e3:6.1f} mm"
                      + (f" TRIP {ep['trips']:#x}" if ep["tripped"] else "")
                      + (" STALLED" if ep["servo_stalled"] else ""), flush=True)  # fmt: skip
                if ep["tripped"] or ep["servo_stalled"]:
                    halt = (f"episode {i}: servo left RUN (sm_state {ep['sm_state']}, fault_bits {ep['trips']:#x})"
                            if ep["tripped"] else f"episode {i}: servo stalled")  # fmt: skip
                    halt_reason = "trip" if ep["tripped"] else "stall"
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
        if backend is not None:
            _hold(backend)  # last command is always HOLD (never a stale RUN)
        with contextlib.suppress(Exception):
            if env is not None:
                env.close()
            elif backend is not None:
                backend.close()
    summary = _doc(code)
    _write_json(out / "summary.json", summary)
    print(_table(summary))
    print(f"run dir {out}  tick log {meta['tick_log']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
