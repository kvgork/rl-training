"""Policy deployment and the zero-shot reach evaluation (MVP M7; sim plan S5.1, reach part).

``pixi run deploy-policy --policy runs/lab/<run>/policy.pt --protocol zero-shot [--attended]``

Runs an exported policy (``rl/policy_export.py``) at 10 Hz through :class:`~so101_rl.env.goal_env.So101GoalEnv` on a
:class:`~so101_rl.env.backend.ShmBackend` attached to an ALREADY-RUNNING servo process: on the real arm the human
starts ``servo --backend feetech`` (Tier 0, attended); in tests and for the sim twin it is ``servo --backend mujoco``.
This script never starts a servo and never opens a serial device; it only writes shm commands.

Everything that sim and real must agree on is the env's: the FS3 observation, ``tasks.action_to_delta_np``, the
``BoxGoalSampler`` with the near-start redraw (seeded by ``--seed``, so a sim twin replays the same goal list), and
the success / termination predicate. The policy input is ``observation ++ desired_goal`` (as ``eval_sim``); the
action is ``policy(x) * speed_scale`` (the ladder scales the delta magnitude; the env then clips and maps it).

* **Episode end:** at success (``terminated``; the env sends HOLD, no further RUN command until the next reset), at
  the 100-step limit, or on a trip / stalled servo (``truncated``). Between episodes ``ShmBackend.reset`` (HOLD, then
  a bounded ramp to the start pose).
* **Real or sim:** decided from the servo that owns the shm segment (its single-instance lock file names the pid; its
  ``/proc/<pid>/cmdline`` gives ``--backend``), fail-closed: anything but ``mujoco`` / ``fake`` is the real arm and
  needs ``--attended`` (Tier 0: a human at the e-stop). The servo's ``--calibration`` and ``--tick-log`` are taken
  from the same command line (the FK must use the servo's calibration; the tick-log path goes into the summary).
* **Protocol** ``--protocol zero-shot`` (plan §10 D-ZEROSHOT, sim plan HS4): the speed ladder 0.5 -> 0.75 -> 1.0 with
  ``--rung-episodes`` (default 10, 10, 50: two short rungs to climb, then the N = 50 gate rung at full speed). The
  ladder stops at the first trip, stalled servo or failed reset. The fine-tune verdict (``finetune_eligible``) is the
  Wilson 95 % lower bound of the full-speed rung >= 0.60 with n >= 50 there; fine-tuning itself is deferred in the
  MVP. ``--protocol single`` runs ``--episodes`` at one ``--speed-scale`` (its verdict is only informative).
* **Start order:** once the servo is in RUN it safe-stops after 0.5 s without a command (``max_consecutive_misses``),
  so this script must be attached BEFORE the servo reaches RUN: it waits up to ``--wait-servo-s`` for the servo
  process, then sends HOLD every 100 ms (never moves the arm) until RUN. Real arm: start the servo (it waits for the
  reset button), start this script, then press the button. ``servo --backend mujoco`` presses it itself: start this
  script first.
* **Outputs** ``<out-root>/<ts>/``: ``episodes.jsonl`` (one line per episode, appended as it ends), ``summary.json``
  (per rung and overall: success rate + Wilson 95 %, the verdict, the servo's tick-log path, inference timing).
* **Ctrl-C / SIGTERM:** stop sending RUN commands, send HOLD, write what was collected, exit 130. The servo's own
  stop / safe-state logic handles the arm. The same holds after a normal end: the last command is HOLD, and 0.5 s
  later the servo runs its safe-state sequence to rest (a fresh servo process is needed for the next deployment).

Exit codes: 0 done (whatever the success rate), 1 the ladder stopped on a trip / stall / failed reset, 2 refused
(bad policy, no servo, real arm without ``--attended``), 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
    "DEFAULT_RUNG_EPISODES",
    "FINETUNE_MIN_WILSON_LO",
    "SPEED_LADDER",
    "ServoInfo",
    "check_policy",
    "main",
    "servo_info",
    "summarize",
]

SPEED_LADDER: Final[tuple[float, ...]] = (0.5, 0.75, 1.0)
DEFAULT_RUNG_EPISODES: Final[tuple[int, ...]] = (10, 10, 50)
FINETUNE_MIN_WILSON_LO: Final[float] = 0.60  # plan §10 D-ZEROSHOT: real fine-tuning only if the lower bound >= 0.60
GATE_RUNG_MIN_EPISODES: Final[int] = 50  # plan §10 D-ZEROSHOT: 50 reach episodes (here: on the full-speed rung)
SIM_SERVO_BACKENDS: Final[frozenset[str]] = frozenset({"mujoco", "fake"})
EXIT_OK, EXIT_HALTED, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 130


class Refused(RuntimeError):
    """A precondition failed before any command was sent (exit 2)."""


# ------------------------------------------------------------------------------------------------ servo discovery


@dataclasses.dataclass(frozen=True)
class ServoInfo:
    pid: int
    backend: str  # "feetech" | "mujoco" | "fake" | "unknown"
    calibration: Path | None
    tick_log: Path | None
    cmdline: list[str]

    @property
    def is_real(self) -> bool:
        """Fail-closed: only a recognised sim backend is not the real arm."""
        return self.backend not in SIM_SERVO_BACKENDS


def _flag(argv: Sequence[str], name: str) -> str | None:
    """The value of ``name`` in ``argv``; the LAST occurrence wins, as in argparse."""
    val = None
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
    return val


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
    if "so101_rl.servo.main" not in argv and not any(a.endswith("so101-servo") for a in argv):
        raise Refused(f"pid {pid} in {path} is not a servo process ({shlex.join(argv)[:200]}): start the servo first")
    try:
        cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        cwd = Path.cwd()
    backend = _flag(argv, "--backend") or "unknown"

    def _path(flag: str) -> Path | None:
        v = _flag(argv, flag)
        return None if v is None else (cwd / Path(v).expanduser()).resolve()

    return ServoInfo(pid, backend, _path("--calibration"), _path("--tick-log"), argv)


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


# ------------------------------------------------------------------------------------------------ statistics


def _rate(successes: int, n: int) -> dict[str, Any]:
    from so101_rl.rl.eval_sim import wilson_interval

    lo, hi = wilson_interval(successes, n)
    return {"n": n, "successes": successes, "success_rate": successes / n if n else None, "wilson95": [lo, hi]}


def summarize(episodes: Sequence[dict[str, Any]], speeds: Sequence[float]) -> dict[str, Any]:
    """Per-rung and overall success + Wilson 95 %, and the fine-tune verdict (on the full-speed rung)."""
    rungs = []
    for s in speeds:
        eps = [e for e in episodes if e["speed"] == s]
        row = _rate(sum(bool(e["success"]) for e in eps), len(eps))
        row["speed"] = s
        errs = np.asarray([e["final_error_m"] for e in eps], float)
        row["final_error_m_p50"] = float(np.median(errs)) if errs.size else None
        row["mean_steps"] = float(np.mean([e["steps"] for e in eps])) if eps else None
        row["trip_episodes"] = sum(bool(e["trips"]) for e in eps)
        rungs.append(row)
    overall = _rate(sum(bool(e["success"]) for e in episodes), len(episodes))
    gate = next((r for r in rungs if r["speed"] == 1.0), None)
    lo = gate["wilson95"][0] if gate is not None and gate["n"] else None
    return {
        "rungs": rungs,
        "overall": overall,
        "verdict": {
            "rung_speed": 1.0,
            "rung_n": gate["n"] if gate is not None else 0,
            "wilson_lo": lo,
            "min_wilson_lo": FINETUNE_MIN_WILSON_LO,
            "min_n": GATE_RUNG_MIN_EPISODES,
            "finetune_eligible": bool(
                lo is not None and lo >= FINETUNE_MIN_WILSON_LO and gate["n"] >= GATE_RUNG_MIN_EPISODES
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
        f"fine-tune verdict (speed 1.0, n={v['rung_n']}): wilson_lo "
        f"{'n/a' if v['wilson_lo'] is None else format(v['wilson_lo'], '.3f')} >= {v['min_wilson_lo']}, "
        f"n >= {v['min_n']} -> "
        f"{'ELIGIBLE' if v['finetune_eligible'] else 'not eligible'}"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ episodes


def _run_episode(env: Any, policy: Any, speed: float, *, seed: int | None, infer_ms: list[float]) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    goal = obs["desired_goal"].astype(float)
    trips = 0
    stalled = False
    clamped_steps = 0
    steps = 0
    success = False
    while True:
        x = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        t0 = time.perf_counter()
        a = policy(x)
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        obs, _r, terminated, truncated, info = env.step(a * np.float32(speed))
        steps += 1
        clamped_steps += int(info["action_clamped"])
        if "trip" in info:
            trips = int(info["trip"])
        stalled = stalled or bool(info.get("servo_stalled", False))
        if terminated or truncated:
            success = bool(terminated)
            break
    err = float(np.linalg.norm(obs["achieved_goal"].astype(float) - obs["desired_goal"].astype(float)))
    return {
        "goal": [round(float(g), 5) for g in goal],
        "success": success,
        "steps": steps,
        "final_error_m": err,
        "trips": trips,
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
    ap.add_argument("--policy", type=Path, required=True, help="policy export (export-policy output)")
    ap.add_argument("--task", default="reach", choices=("reach",))
    ap.add_argument("--config", type=Path, default=None, help="so101.yaml (default: the packaged one)")
    ap.add_argument("--calibration", type=Path, default=None, help="default: the running servo's --calibration")
    ap.add_argument("--shm-name", default=None, help="the running servo's shm segment (default: from config)")
    ap.add_argument("--attended", action="store_true", help="Tier 0: a human is at the e-stop (required on the arm)")
    ap.add_argument("--protocol", choices=("single", "zero-shot"), default="single")
    ap.add_argument("--speed-scale", type=float, default=0.5, choices=SPEED_LADDER, help="single protocol")
    ap.add_argument("--episodes", type=int, default=10, help="single protocol")
    ap.add_argument(
        "--rung-episodes",
        default=",".join(map(str, DEFAULT_RUNG_EPISODES)),
        help="zero-shot protocol: episodes at speed 0.5,0.75,1.0",
    )
    ap.add_argument("--seed", type=int, default=7, help="goal sampler seed (same seed = same goal list)")
    ap.add_argument("--out-root", type=Path, default=Path("runs/deploy"))
    ap.add_argument("--wait-servo-s", type=float, default=120.0, help="wait for the servo process / RUN state")
    return ap


def _plan(args: argparse.Namespace) -> list[tuple[float, int]]:
    if args.protocol == "single":
        return [(args.speed_scale, args.episodes)]
    try:
        counts = [int(c) for c in str(args.rung_episodes).split(",")]
    except ValueError:
        raise Refused(f"--rung-episodes must be integers, got {args.rung_episodes!r}") from None
    if len(counts) != len(SPEED_LADDER) or min(counts) < 0:
        raise Refused(f"--rung-episodes needs {len(SPEED_LADDER)} counts >= 0, got {args.rung_episodes!r}")
    return [(s, n) for s, n in zip(SPEED_LADDER, counts, strict=True) if n > 0]


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda m: print(f"deploy-policy: {m}", file=sys.stderr, flush=True)
    # everything slow (torch, the policy, the FK) happens BEFORE attaching: once the servo is in RUN it safe-stops
    # after 0.5 s without a command, so the HOLD keep-alive must start right after the attach
    import torch

    from so101_rl.env.backend import ResetFailed, ServoNotRunning, ServoStateStale, ShmBackend
    from so101_rl.env.goal_env import So101GoalEnv

    torch.set_num_threads(1)
    try:
        from so101_rl.config import load_calibration, load_config
        from so101_rl.env.tasks import REACH
        from so101_rl.rl.policy_export import load_policy

        plan = _plan(args)
        if args.episodes < 1 and args.protocol == "single":
            raise Refused("--episodes must be >= 1")
        try:
            policy = load_policy(args.policy)
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            raise Refused(f"cannot use policy {args.policy}: {exc}") from None
        check_policy(policy, REACH)
        base = load_config(args.config)
        shm_name = args.shm_name or base.shm_name
        servo = servo_info(shm_name, wait_s=args.wait_servo_s)
        if servo.is_real and not args.attended:
            raise Refused(
                f"the servo on shm {shm_name!r} (pid {servo.pid}, --backend {servo.backend}) drives the REAL arm: "
                "Tier 0 needs a human at the e-stop; pass --attended"
            )
        calib_path = args.calibration or servo.calibration or base.calibration_path
        if args.calibration is not None and servo.calibration is not None and args.calibration.resolve() != servo.calibration:
            raise Refused(f"--calibration {args.calibration} differs from the servo's {servo.calibration}")
        calibration = load_calibration(Path(calib_path).expanduser())
        cfg = dataclasses.replace(load_config(args.config, calibration=calibration), shm_name=shm_name)
    except Refused as exc:
        log(f"refused: {exc}")
        return EXIT_REFUSED
    except (ValueError, OSError, KeyError) as exc:  # a malformed --rung-episodes, config or calibration
        log(f"refused: {type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    out = args.out_root / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
    out.mkdir(parents=True, exist_ok=False)
    meta = {
        "policy": str(args.policy.resolve()),
        "policy_meta": {k: v for k, v in policy.meta.items() if isinstance(v, (str, int, float, list, type(None)))},
        "task": REACH.name,
        "protocol": args.protocol,
        "plan": [{"speed": s, "episodes": n} for s, n in plan],
        "seed": args.seed,
        "attended": bool(args.attended),
        "servo": {"pid": servo.pid, "backend": servo.backend, "real": servo.is_real, "shm_name": shm_name,
                  "cmdline": servo.cmdline},
        "calibration": str(Path(calib_path).expanduser().resolve()),
        "tick_log": None if servo.tick_log is None else str(servo.tick_log),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if servo.tick_log is None:
        log("the servo runs without --tick-log: no tick log for this deployment")
    log(f"servo pid {servo.pid} --backend {servo.backend} ({'REAL ARM, attended' if servo.is_real else 'sim'}); "
        f"out {out}")

    def _sigterm(_signum, _frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, _sigterm)
    backend = env = None
    episodes: list[dict[str, Any]] = []
    infer_ms: list[float] = []
    halt: str | None = None
    code = EXIT_OK
    try:
        backend = ShmBackend(cfg, calibration=calibration)
        with hold_keepalive(backend):  # building the goal table takes seconds: keep the servo commanded meanwhile
            env = So101GoalEnv(backend, cfg, REACH)
        with (out / "episodes.jsonl").open("a") as f:
            try:
                wait_for_run(backend, args.wait_servo_s, log)
                again = servo_info(shm_name)  # the owner must still be the servo we classified (no swap meanwhile)
                if again.pid != servo.pid or again.is_real != servo.is_real:
                    raise ServoNotRunning(f"the servo on shm {shm_name!r} changed (pid {servo.pid} -> {again.pid})")
            except (ServoNotRunning, ServoStateStale, Refused) as exc:
                halt = f"{type(exc).__name__}: {exc}"
                plan_left: list[tuple[float, int]] = []
            else:
                plan_left = plan
            for speed, n in plan_left:
                for _ in range(n):
                    try:
                        ep = _run_episode(env, policy, speed, seed=args.seed if not episodes else None,
                                          infer_ms=infer_ms)
                    except (ServoNotRunning, ResetFailed, ServoStateStale) as exc:
                        halt = f"{type(exc).__name__}: {exc}"
                        break
                    ep["episode"] = len(episodes)
                    episodes.append(ep)
                    f.write(json.dumps(ep) + "\n")
                    f.flush()
                    print(f"ep {ep['episode']:3d} speed {speed:4.2f} {'SUCCESS' if ep['success'] else 'fail   '} "
                          f"steps {ep['steps']:3d} err {ep['final_error_m'] * 1e3:6.1f} mm"
                          + (f" TRIP {ep['trips']:#x}" if ep["trips"] else "")
                          + (" STALLED" if ep["servo_stalled"] else ""), flush=True)
                    if ep["trips"] or ep["servo_stalled"]:
                        halt = f"episode {ep['episode']}: trip {ep['trips']:#x} stalled {ep['servo_stalled']}"
                        break
                if halt is not None:
                    break
        if halt is not None:
            log(f"ladder stopped: {halt}")
            code = EXIT_HALTED
    except KeyboardInterrupt:
        halt = "interrupted (Ctrl-C / SIGTERM)"
        code = EXIT_INTERRUPTED
        log("interrupted: HOLD sent, no further commands; the servo safe-stops the arm to rest 0.5 s later")
    except Exception as exc:  # noqa: BLE001 - still HOLD and write the partial summary, then exit 1
        halt = f"error: {exc!r}"
        code = EXIT_HALTED
        log(f"stopped by an error: {exc!r}")
    finally:
        if backend is not None:
            _hold(backend)  # last command is always HOLD (never a stale RUN)
        signal.signal(signal.SIGTERM, old_term)
        with contextlib.suppress(Exception):
            if env is not None:
                env.close()
            elif backend is not None:
                backend.close()
    summary = summarize(episodes, [s for s, _ in plan])
    inf = np.asarray(infer_ms, float)
    summary.update(meta)
    summary.update({
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "halted": halt,
        "exit_code": code,
        "inference_ms": {"p50": float(np.median(inf)), "max": float(inf.max())} if inf.size else None,
        "policy_overruns": int(backend.overruns) if backend is not None else 0,
        "episodes_file": str(out / "episodes.jsonl"),
    })
    _write_json(out / "summary.json", summary)
    print(_table(summary))
    print(f"run dir {out}  tick log {meta['tick_log']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
