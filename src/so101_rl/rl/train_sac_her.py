"""SB3 SAC + HER fine-tuning of the sim reach actor (sim plan S5.2 / HS5; pixi task ``train-reach-sac``).

``train-reach-sac --init-from <policy_export.pt> --backend {sim,shm} --total-steps N [--attended]``
``train-reach-sac --resume <run dir> [--total-steps N] [--attended] [--reset-actor]``

* **Model:** :class:`~so101_rl.rl.sb3_bridge.FinetuneSAC` on ``So101GoalEnv(REACH)``, the export's actor copied in
  (critics fresh), ``HerReplayBuffer`` (4 sampled goals, ``future``, the info dicts kept: the relabelled reward reads
  the trip bits), critic lr 1e-4, batch 256, ``gamma`` = the task's, ``tau`` 0.005, ``learning_starts`` 2000.
  Updates run **between episodes** (``train_freq=(1, "episode")``, one gradient step per collected step), never inside
  one: the 10 Hz loop never waits for an update. Every action is the actor's (no uniform-random warm-up).
* **Recipe** (plan §10 2026-10-13 "Anchored fine-tuning", after the 2026-10-13 collapse: the free SAC actor fell from
  ~10 mm to 20-35 cm final errors right after its warm-up): the actor is frozen for ``--warmup-frozen-actor`` (5000)
  gradient steps while the fresh critics learn, then learns at ``--actor-lr`` (1e-5) with the TD3+BC anchor to the sim
  actor (``--anchor-weight`` 0.4, 0 = plain SAC); the entropy coefficient is held at ``--ent-coef`` (0.01; ``auto`` =
  SB3's learned one). These recipe flags are stored in ``run.json``; on ``--resume`` a given flag replaces the stored
  value (logged in ``run.json["recipe_changes"]``), a missing stored one takes the default. ``--reset-actor`` (resume
  only) puts the export's actor back (critics, replay buffer and step count kept) and freezes it for
  ``--warmup-frozen-actor`` more gradient steps.
* **Baseline and rollback:** before learning, the export is evaluated deterministically on ``--eval-episodes`` fixed
  goals (drawn from ``--eval-seed`` as in ``deploy-policy``) = ``baseline.json``. At the first episode end after every
  ``--baseline-eval-every`` steps the current actor is evaluated on the same goals (``evals.jsonl``); the best actor so
  far is kept (``best_policy.pt``). The **actor** (the critics are kept) is rolled back to it when two consecutive
  evals fall below baseline − ``--rollback-margin``, at once when an eval episode trips (a tripped eval counts as
  success 0; the session then ends), and at once when the last ``--train-window`` (10) training episodes of the
  learning actor succeed less than baseline − margin. Each rollback is a line in ``rollbacks.jsonl``; the
  ``--max-rollbacks``-th (3) in a session ends it (``rollback_limit``).
* **Backends:** ``sim`` = the in-process CPU reference rig (``SimBackend``). ``shm`` = an already-running servo
  (``servo --backend feetech`` on the arm; the tests use ``mujoco`` / ``fake``), attached exactly like
  ``deploy-policy`` (``resolve_servo``: classified read-only from ``/proc``, the real arm needs ``--attended``;
  attach before RUN with the HOLD keep-alive, then ``wait_for_run``). **No watchdog MCU is required** (plan §10
  2026-10-12 "Real-arm fine-tuning without the watchdog MCU"): the session is stopped by the servo's own safety layer
  (THERMAL, PROTECTION_BIT, TRACKING, STUCK, box checks). While the updates run between episodes, the HOLD keep-alive
  runs (a servo in RUN safe-stops 0.5 s after the last command).
* **Session end** (:class:`HaltOnTrip`): a trip or a stall in an episode, a failed reset, the servo leaving RUN, a
  tripped eval episode, or ``--session-s`` (default the config's thermal ``duty_on_s``, 1200 s) of session time
  reached at an episode end. The halting step is kept (a terminal trip is a training sample); nothing is sent to the
  servo after it but HOLD. Then: model, replay buffer, best policy and export are saved, ``summary.json`` records
  ``halt_reason``, exit 1 (0 when ``--total-steps`` was reached). ``--resume <run dir>`` continues with a fresh servo
  (after ``duty_off_s`` for a ``session_cap`` halt): same export, seed and eval goals, ``--total-steps`` = the run's
  total unless given.
* **Outputs** ``<out-root>/<ts>/``: ``model.zip``, ``replay_buffer.pkl``, ``monitor.csv`` (sim), ``baseline.json``,
  ``evals.jsonl``, ``best_policy.pt``, ``policy_export.pt`` (the current actor, deployable), ``sessions.jsonl`` (one
  line per session), ``timing.jsonl`` (one line per collect/update cycle, below), ``rollbacks.jsonl`` and ``summary.json``. Exit 2 on a
  refusal (``--total-steps`` < 1, a bad export, no servo, the real arm without ``--attended``, ``--reset-actor``
  without ``--resume``).
* **Timing** (``timing.jsonl``, plan §10 2026-10-13 "Trainer stall diagnostics"): per cycle the episode's longest gap
  between two policy steps (``reset_step_s``: the episode's last step, which includes the auto-reset ramp), the eval
  time, the gap from the episode's last step to the first keep-alive HOLD, the
  update time, every keep-alive HOLD gap above the servo's stale-command limit (the helper thread was starved) and
  ``updates_to_first_step_s`` (the previous cycle's updates to this episode's first step). A gap above
  ``STALL_WARN_S`` inside an episode is also printed.
  ``--torch-threads`` (default 4) caps PyTorch's CPU threads so the updates leave cores for the keep-alive thread and
  the servo process.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import gymnasium
import numpy as np

EXIT_OK: Final[int] = 0
EXIT_HALTED: Final[int] = 1
EXIT_REFUSED: Final[int] = 2
EXIT_INTERRUPTED: Final[int] = 130
ARTIFACTS: Final[tuple[str, ...]] = ("model.zip", "replay_buffer.pkl", "baseline.json", "evals.jsonl", "best_policy.pt",
                                     "policy_export.pt", "sessions.jsonl", "summary.json")  # fmt: skip
STALL_WARN_S: Final[float] = 0.3  # a send gap this long is printed (servo: HOLD after 0.15 s, SAFE_STOPPING ~0.5 s)
RUN_KEYS: Final[tuple[str, ...]] = ("init_from", "seed", "eval_seed", "eval_episodes", "total_steps",
                                    "learning_starts", "baseline_eval_every", "backend")  # fmt: skip
# plan §10 2026-10-13 "Anchored fine-tuning"; stored in run.json, a flag given on --resume replaces the stored value
RECIPE_DEFAULTS: Final[dict[str, Any]] = {"warmup_frozen_actor": 5000, "actor_lr": 1e-5, "anchor_weight": 0.4,
                                          "ent_coef": 0.01, "rollback_margin": 0.15, "train_window": 10,
                                          "max_rollbacks": 3}  # fmt: skip


class Halt(Exception):
    """Raised inside the training loop's callbacks to end the session (``reason`` -> ``halt_reason``)."""


class HaltOnTrip(gymnasium.Wrapper):
    """Ends a session cleanly. After a step that tripped / stalled (``info["trip"]`` / ``info["servo_stalled"]``), a
    failed ``reset`` (the servo left RUN, the reset did not converge, a stale state) or :meth:`halt`, the env never
    touches the backend again: ``reset`` returns the last observation and ``step`` returns it with ``truncated`` and
    ``info["halted"]`` (the callback stops the loop on such a step, before SB3 stores it). The halting step itself is
    returned unchanged, so SB3 stores it."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.halt_reason: str | None = None
        self.halt_detail: str | None = None
        self._last_obs: Any = None

    @property
    def halted(self) -> bool:
        return self.halt_reason is not None

    def halt(self, reason: str, detail: str | None = None) -> None:
        if self.halt_reason is None:
            self.halt_reason, self.halt_detail = reason, detail

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        from so101_rl.env.backend import ResetFailed, ServoNotRunning, ServoStateStale

        if not self.halted:
            try:
                obs, info = self.env.reset(**kwargs)
            except (ServoNotRunning, ResetFailed, ServoStateStale) as exc:
                self.halt("reset_failed" if isinstance(exc, ResetFailed) else "servo_not_running",
                          f"{type(exc).__name__}: {exc}")  # fmt: skip
            else:
                self._last_obs = obs
                return obs, info
        return self._dummy_obs(), {"halted": True}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.halted:
            return self._dummy_obs(), 0.0, False, True, {"halted": True}
        obs, r, term, trunc, info = self.env.step(action)
        self._last_obs = obs
        if "trip" in info:
            self.halt("trip", f"servo left RUN or reported fault bits {int(info['trip']):#x} (sm_state "
                              f"{int(info['sm_state'])})")  # fmt: skip
        elif info.get("servo_stalled"):
            self.halt("stall", "the servo state stalled")
        return obs, r, term, trunc, info

    def _dummy_obs(self) -> Any:
        if self._last_obs is None:
            return {k: np.zeros(s.shape, s.dtype) for k, s in self.observation_space.spaces.items()}
        return copy.deepcopy(self._last_obs)


def evaluate(env: Any, policy: Any, pairs: np.ndarray) -> dict[str, Any]:
    """Deterministic episodes of ``policy`` on the fixed goal ``pairs`` (``deploy-policy``'s episode loop); the env's
    goal sampler is restored afterwards. Stops at the first tripped / stalled episode (``halted``). The caller must
    reset the env before training continues."""
    from so101_rl.rl.deploy_policy import GoalList, _run_episode

    saved = env.goal_sampler
    env.goal_sampler = goals = GoalList(pairs)
    eps: list[dict[str, Any]] = []
    try:
        for i in range(len(pairs)):
            eps.append(_run_episode(env, policy, 1.0, goals=goals, index=i, infer_ms=[]))
            if eps[-1]["tripped"] or eps[-1]["servo_stalled"]:
                break
    finally:
        env.goal_sampler = saved
    err = np.asarray([e["final_error_m"] for e in eps])
    return {"n": len(eps), "success_rate": float(np.mean([e["success"] for e in eps])),
            "trips": int(sum(e["tripped"] for e in eps)), "stalls": int(sum(e["servo_stalled"] for e in eps)),
            "halted": len(eps) < len(pairs), "final_error_m_p50": float(np.median(err))}  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train-reach-sac", description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--init-from", type=Path, help="policy_export.pt of the sim actor (a new run)")
    src.add_argument("--resume", type=Path, help="continue this run dir (its export, seed and eval goals)")
    ap.add_argument("--backend", choices=("sim", "shm"), default=None, help="new run: default sim")
    ap.add_argument("--total-steps", type=int, default=None, help="new run: required")
    rec = ap.add_argument_group("recipe (default: run.json on --resume, else RECIPE_DEFAULTS)")
    rec.add_argument("--warmup-frozen-actor", type=int, default=None, help="gradient steps with the actor frozen")
    rec.add_argument("--actor-lr", type=float, default=None)
    rec.add_argument("--anchor-weight", type=float, default=None, help="TD3+BC anchor to the sim actor; 0 = plain SAC")
    rec.add_argument("--ent-coef", type=_ent_coef, default=None, help="fixed entropy coefficient, or 'auto'")
    rec.add_argument("--rollback-margin", type=float, default=None)
    rec.add_argument("--train-window", type=int, default=None, help="training episodes in the rollback window")
    rec.add_argument("--max-rollbacks", type=int, default=None, help="per session; reaching it ends the session")
    ap.add_argument("--reset-actor", action="store_true", help="--resume only: the export's actor back, frozen again")
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--baseline-eval-every", type=int, default=1000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=4, help="PyTorch CPU threads (leave cores for the servo)")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sac-her"))
    shm = ap.add_argument_group("shm backend (an already-running servo)")
    shm.add_argument("--attended", action="store_true", help="required on the real arm: a human at the power switch")
    shm.add_argument("--shm-name", default=None, help="default: the config's shm_name")
    shm.add_argument("--wait-servo-s", type=float, default=120.0)
    shm.add_argument("--session-s", type=float, default=None, help="default: the config's thermal duty_on_s")
    shm.add_argument("--config", type=Path, default=None)
    shm.add_argument("--calibration", type=Path, default=None)
    return ap


def _ent_coef(v: str) -> float | str:
    return "auto" if v == "auto" else float(v)


def _log(msg: str) -> None:
    print(f"train-reach-sac: {msg}", file=sys.stderr, flush=True)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def _append_jsonl(path: Path, doc: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(doc) + "\n")


def _resolve_run(args: argparse.Namespace) -> tuple[dict[str, Any], Path, bool]:
    """``(run, out, resuming)``: the run's fixed settings (from the CLI, or from a resumed run's ``run.json``)."""
    if args.resume is None:
        if args.total_steps is None or args.total_steps < 1:
            raise ValueError("--total-steps must be >= 1")
        if args.reset_actor:
            raise ValueError("--reset-actor needs --resume")
        run = {k: getattr(args, k) for k in RUN_KEYS}
        run["init_from"] = str(args.init_from.resolve())
        run["backend"] = args.backend or "sim"
        run.update({k: d if getattr(args, k) is None else getattr(args, k) for k, d in RECIPE_DEFAULTS.items()})
        out = args.out_root / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
        return run, out, False
    out = args.resume
    if not (out / "run.json").is_file() or not (out / "model.zip").is_file():
        raise ValueError(f"--resume {out}: not a train-reach-sac run dir (run.json / model.zip missing)")
    run = json.loads((out / "run.json").read_text())
    if args.backend is not None and args.backend != run["backend"]:
        raise ValueError(f"--resume: the run used --backend {run['backend']}, not {args.backend}")
    if args.total_steps is not None:
        if args.total_steps < 1:
            raise ValueError("--total-steps must be >= 1")
        run["total_steps"] = args.total_steps
    changes = {}
    for k, d in RECIPE_DEFAULTS.items():
        given = getattr(args, k)
        new = given if given is not None else run.get(k, d)
        if k == "warmup_frozen_actor" and args.reset_actor and given is None:
            new = d  # a reset actor is frozen for the default warm-up, not the run's original one
        if run.get(k) != new:
            changes[k] = {"from": run.get(k), "to": new}
        run[k] = new
    if changes or args.reset_actor:
        run["_pending_change"] = {"changes": changes, "reset_actor": bool(args.reset_actor)}
    return run, out, True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from so101_rl.rl.policy_export import load_policy

    try:
        run, out, resuming = _resolve_run(args)
        init = load_policy(Path(run["init_from"]))
        if init.meta.get("task_name") != "reach":
            raise ValueError(f"the export's task is {init.meta.get('task_name')!r}, not 'reach'")
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        _log(f"refused: {exc}")
        return EXIT_REFUSED
    return _session(args, run, init, out, resuming)


def _session(args: argparse.Namespace, run: dict[str, Any], init: Any, out: Path, resuming: bool) -> int:
    import torch
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from so101_rl.config.loader import DEFAULT_CONFIG_PATH
    from so101_rl.env.backend import ResetFailed, ServoNotRunning, ServoStateStale
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.env.goals import BoxGoalSampler
    from so101_rl.env.tasks import REACH
    from so101_rl.rl import deploy_policy as dp
    from so101_rl.rl.sb3_bridge import (
        FinetuneSAC,
        GoalPolicyInput,
        actor_policy,
        load_actor_into_sb3,
        rollback_due,
    )

    torch.manual_seed(int(run["seed"]))
    torch.set_num_threads(max(1, int(args.torch_threads)))
    backend: Any = None
    model: Any = None
    baseline: dict[str, Any] | None = None
    evals: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    keepalive = contextlib.ExitStack()
    session: dict[str, Any] = {"started": datetime.now().astimezone().isoformat(timespec="seconds"),
                               "backend": run["backend"]}  # fmt: skip
    def _on_term(_sig: int, _frame: Any) -> None:  # SIGTERM like Ctrl-C: the finally still sends HOLD and saves
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, _on_term)
    try:
        # ---------------------------------------------------------------- backend (attach before RUN for shm)
        if run["backend"] == "shm":
            from so101_rl.env.backend import ShmBackend

            shm_name = args.shm_name or str(dp._yaml_key(args.config or DEFAULT_CONFIG_PATH, "shm_name"))
            rs = dp.resolve_servo(shm_name, wait_s=args.wait_servo_s, attended=args.attended,
                                  accept_no_watchdog=True, config=args.config, calibration=args.calibration,
                                  log=_log)  # fmt: skip  # no watchdog required: plan §10 2026-10-12
            prev = [json.loads(x) for x in (out / "sessions.jsonl").read_text().splitlines()] if resuming else []
            if prev and bool(prev[0].get("servo", {}).get("real")) != rs.servo.is_real:
                raise dp.Refused("--resume: the run's servo class (real / sim) differs from this servo's")
            cfg, calibration = rs.cfg, rs.calibration
            session["servo"] = {"pid": rs.servo.pid, "backend": rs.servo.backend, "real": rs.servo.is_real,
                                "mcu": rs.servo.mcu, "tick_log": None if rs.servo.tick_log is None
                                else str(rs.servo.tick_log), "calibration_sha256": rs.calibration_sha256}  # fmt: skip
            backend = ShmBackend(cfg, calibration=calibration)  # ATTACH NOW: HOLD every 100 ms until RUN
            keepalive.enter_context(dp.hold_keepalive(backend))
            session_s = float(args.session_s if args.session_s is not None else cfg.thermal.duty_on_s)
        else:
            from so101_rl.config import load_calibration, load_config
            from so101_rl.env.backend import SimBackend

            calibration = load_calibration(load_config().calibration_path)
            cfg = load_config(calibration=calibration)
            backend = SimBackend(cfg, calibration, seed=int(run["seed"]))
            session_s = float(args.session_s) if args.session_s is not None else float("inf")
        session["session_s"] = session_s
        out.mkdir(parents=True, exist_ok=resuming)
        genv = So101GoalEnv(backend, cfg, REACH)
        env = HaltOnTrip(Monitor(genv, str(out / "monitor.csv"), override_existing=not resuming)
                         if run["backend"] == "sim" else genv)  # fmt: skip
        pairs = dp.draw_goal_pairs(BoxGoalSampler(REACH.goal_bounds, fk=genv._fk, limits=cfg.joints, seed=0),
                                   int(run["eval_seed"]), int(run["eval_episodes"]))  # fmt: skip
        pending = run.pop("_pending_change", None)
        if not resuming:
            _write_json(out / "run.json", run)
        fixed_ent = None if run["ent_coef"] == "auto" else float(run["ent_coef"])
        kw = {"actor_learning_rate": float(run["actor_lr"]), "fixed_ent_coef": fixed_ent,
              "anchor_weight": float(run["anchor_weight"]), "device": args.device}  # fmt: skip
        if resuming:  # warmup_frozen_actor comes from model.zip (absolute, possibly moved by a --reset-actor)
            model = FinetuneSAC.load(out / "model.zip", env=env, **kw)
            model.load_replay_buffer(out / "replay_buffer.pkl")
            baseline = json.loads((out / "baseline.json").read_text())
            evals = [json.loads(x) for x in (out / "evals.jsonl").read_text().splitlines()]
            b = torch.load(out / "best_policy.pt", weights_only=False)
            best = {"success_rate": b["success_rate"], "state": b["state"], "step": b["step"]}
        else:
            model = FinetuneSAC(
                "MultiInputPolicy", env,
                policy_kwargs={"net_arch": list(init.meta["actor_hidden"]), "features_extractor_class": GoalPolicyInput},
                replay_buffer_class=HerReplayBuffer,
                # copy_info_dict: the relabelled reward reads the stored info (trip bits -> trip_penalty)
                replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
                ent_coef="auto_0.01", learning_rate=1e-4, learning_starts=int(run["learning_starts"]), batch_size=256,
                gamma=REACH.gamma, tau=0.005, train_freq=(1, "episode"), gradient_steps=-1, seed=int(run["seed"]),
                verbose=0, warmup_frozen_actor=int(run["warmup_frozen_actor"]), **kw,
            )  # fmt: skip
            load_actor_into_sb3(model, init)
        model.set_anchor(init)
        if pending is not None:
            if pending["reset_actor"]:
                model.reset_actor(init, int(run["warmup_frozen_actor"]))
                _log(f"actor reset to the export; frozen until gradient step {model.warmup_frozen_actor}")
            run.setdefault("recipe_changes", []).append({"at_step": int(model.num_timesteps), **pending,
                                                         "at": datetime.now().astimezone().isoformat(timespec="seconds")})
            _write_json(out / "run.json", run)
        if run["backend"] == "shm":
            keepalive.close()
            dp.wait_for_run(backend, args.wait_servo_s, _log)
            again = dp.servo_info(rs.cfg.shm_name)
            if again.pid != rs.servo.pid:
                raise ServoNotRunning(f"the servo changed (pid {rs.servo.pid} -> {again.pid})")
        t0 = time.monotonic()
        if baseline is None:
            baseline = {"policy": run["init_from"], "eval_seed": run["eval_seed"], "goals": pairs.tolist(),
                        **evaluate(genv, init, pairs)}  # fmt: skip
            _write_json(out / "baseline.json", baseline)
            (out / "evals.jsonl").touch()
            best = {"success_rate": baseline["success_rate"], "state": copy.deepcopy(model.policy.state_dict()),
                    "step": 0}  # fmt: skip
            _log(f"baseline: {baseline['success_rate']:.2f} success over {baseline['n']} episodes")
            if baseline["halted"]:
                env.halt("eval_trip", "a baseline eval episode tripped or stalled")
        session["start_step"] = int(model.num_timesteps)

        class Session(BaseCallback):
            """Evals / best / rollback at episode ends, the keep-alive during updates, and the session end."""

            def __init__(self) -> None:
                super().__init__()
                every, n = int(run["baseline_eval_every"]), int(model.num_timesteps)
                due = (n // every) * every  # an eval missed at a session boundary runs at the first episode end
                last = max((int(e["step"]) for e in evals), default=0)
                self.next_at = n if 0 < due and last < due else (n // every + 1) * every
                self.window: list[bool] = []  # training-episode successes of the learning actor
                self.n_rollbacks = 0
                self.cyc: dict[str, Any] = {}
                self.late: list[float] = []  # appended by the keep-alive thread
                self.t_step: float | None = None  # last policy step (on_step)
                self.t_hold: float | None = None  # keep-alive start
                self.t_upd_end: float | None = None  # end of the updates (on_rollout_start)

            def _on_step(self) -> bool:
                now = time.monotonic()
                info, done = self.locals["infos"][0], bool(self.locals["dones"][0])
                if self.t_step is not None:
                    gap = now - self.t_step
                    if self.t_upd_end is not None:  # the first step after the updates (includes its own 0.1 s wait)
                        self.cyc["updates_to_first_step_s"] = round(now - self.t_upd_end, 4)
                        self.t_upd_end = None
                    elif done:  # the episode's last step: the VecEnv's auto-reset (a commanded ramp) ran inside it
                        self.cyc["reset_step_s"] = round(gap, 4)
                    else:
                        self.cyc["max_step_gap_s"] = round(max(self.cyc.get("max_step_gap_s", 0.0), gap), 4)
                        if gap > STALL_WARN_S + 0.1:
                            _log(f"policy step gap {gap:.2f} s at step {self.num_timesteps} (servo HOLD after 0.15 s)")
                self.t_step = now
                if info.get("halted"):  # a halted env's step: not stored, the loop ends here
                    model.num_timesteps -= 1  # SB3 counted it before this callback; it never reached the servo
                    self._flush()
                    return False
                if done and not model.actor_frozen:
                    self.window.append(bool(info.get("is_success")))
                    w = int(run["train_window"])
                    if len(self.window) >= w:
                        rate = float(np.mean(self.window[-w:]))
                        if rate < baseline["success_rate"] - float(run["rollback_margin"]):
                            self._rollback("train_window", f"last {w} training episodes {rate:.2f}")
                return True

            def _rollback(self, reason: str, detail: str) -> None:
                """Actor <- the best one (critics kept); ``rollbacks.jsonl``; the session ends at ``max_rollbacks``."""
                model.rollback_actor(best["state"])
                self.window.clear()
                self.n_rollbacks += 1
                _append_jsonl(out / "rollbacks.jsonl", {"step": int(self.num_timesteps), "reason": reason,
                                                        "detail": detail, "to_step": best["step"]})  # fmt: skip
                _log(f"ROLLBACK ({reason}: {detail}) to the actor of step {best['step']}")
                if self.n_rollbacks >= int(run["max_rollbacks"]):
                    env.halt("rollback_limit", f"{self.n_rollbacks} rollbacks this session")

            def _on_rollout_end(self) -> None:
                if env.halted:
                    return
                if time.monotonic() - t0 >= session_s:
                    env.halt("session_cap", f"{session_s:.0f} s of session time (thermal duty_on_s)")
                    return
                if self.num_timesteps >= self.next_at:
                    self.next_at += int(run["baseline_eval_every"])
                    t = time.monotonic()
                    self._eval()
                    self.cyc["eval_s"] = round(time.monotonic() - t, 3)
                    self.t_step = None  # the eval episodes sent the commands meanwhile
                if run["backend"] == "shm" and not env.halted:
                    self.t_hold = time.monotonic()
                    if self.t_step is not None:
                        self.cyc["last_step_to_hold_s"] = round(self.t_hold - self.t_step, 4)
                    keepalive.enter_context(dp.hold_keepalive(backend, on_late=self.late.append))  # the updates run now

            def _on_rollout_start(self) -> None:
                keepalive.close()
                now = time.monotonic()
                if self.t_hold is not None:
                    self.cyc["updates_s"] = round(now - self.t_hold, 3)
                    self.t_hold = None
                    self.t_upd_end = now
                    if self.late:
                        _log(f"keep-alive starved: HOLD gaps {[round(g, 2) for g in self.late]} s during the updates")
                self._flush()

            def _flush(self) -> None:
                """Write the finished cycle (the previous episode, its eval and updates) to ``timing.jsonl``."""
                if self.late:
                    self.cyc["keepalive_late_s"] = [round(g, 3) for g in self.late]
                    self.late.clear()
                if self.cyc:
                    _append_jsonl(out / "timing.jsonl", {"step": int(self.num_timesteps), **self.cyc})
                self.cyc = {}

            def _eval(self) -> None:
                try:
                    r = {"step": self.num_timesteps, "actor_frozen": model.actor_frozen,
                         **evaluate(genv, actor_policy(model, init.meta), pairs)}  # fmt: skip
                except (ServoNotRunning, ServoStateStale, ResetFailed) as exc:
                    env.halt("eval_failed", f"{type(exc).__name__}: {exc}")
                    return
                if r["halted"]:
                    env.halt("eval_trip", f"eval at step {r['step']}: an episode tripped or stalled")
                elif r["success_rate"] > best["success_rate"]:
                    best.update(success_rate=r["success_rate"], state=copy.deepcopy(model.policy.state_dict()),
                                step=self.num_timesteps)  # fmt: skip
                r["rolled_back"] = r["halted"] or rollback_due(  # a tripped eval counts as success 0
                    [0.0 if e.get("halted") else e["success_rate"] for e in evals] + [r["success_rate"]],
                    baseline["success_rate"], margin=float(run["rollback_margin"]))  # fmt: skip
                if r["rolled_back"]:
                    r["rolled_back_to_step"] = best["step"]
                    self._rollback("eval_trip" if r["halted"] else "evals", f"eval at step {r['step']}")
                evals.append(r)
                _append_jsonl(out / "evals.jsonl", r)
                _log(f"eval @ {r['step']}: {r['success_rate']:.2f}" + (" ROLLBACK" if r["rolled_back"] else ""))
                model._last_obs = self.training_env.reset()  # the evals moved the arm: start a fresh episode

        cb = Session()
        remaining = int(run["total_steps"]) - int(model.num_timesteps)
        if remaining > 0 and not env.halted:
            model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=not resuming)
        reached = model.num_timesteps >= int(run["total_steps"]) and not env.halted
        session.update(halt_reason=None if reached else (env.halt_reason or "incomplete"), halt=env.halt_detail)
        session["rollbacks"] = cb.n_rollbacks
    except dp.Refused as exc:
        _log(f"refused: {exc}")
        session = {}
        return EXIT_REFUSED
    except KeyboardInterrupt:
        session.update(halt_reason="interrupted", halt="Ctrl-C / SIGTERM")
        _log("interrupted: HOLD sent; the servo safe-stops the arm to rest 0.5 s later")
    except (ServoNotRunning, ServoStateStale, ResetFailed) as exc:
        session.update(halt_reason="servo_not_running", halt=f"{type(exc).__name__}: {exc}")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        keepalive.close()
        if backend is not None and run["backend"] == "shm":
            dp._hold(backend)  # the last command is always HOLD
        if session and "start_step" in session:
            _save(out, model, init, run, best, session)
        with contextlib.suppress(Exception):
            if backend is not None:
                backend.close()
        signal.signal(signal.SIGTERM, old_term)
    return _finish(out, run, baseline, evals, best, session, model)


def _save(out: Path, model: Any, init: Any, run: dict[str, Any], best: dict[str, Any], session: dict[str, Any]) -> None:
    import torch

    from so101_rl.rl.sb3_bridge import export_sb3_actor

    model.save(out / "model.zip")
    model.save_replay_buffer(out / "replay_buffer.pkl")
    torch.save(best, out / "best_policy.pt")
    export_sb3_actor(model, init.meta, out / "policy_export.pt", finetuned_from=run["init_from"],
                     finetune_steps=int(model.num_timesteps), finetune_backend=run["backend"])  # fmt: skip
    session.update(finished=datetime.now().astimezone().isoformat(timespec="seconds"),
                   end_step=int(model.num_timesteps), n_updates=int(model._n_updates))  # fmt: skip
    _append_jsonl(out / "sessions.jsonl", session)


def _finish(out: Path, run: dict[str, Any], baseline: Any, evals: list[dict[str, Any]], best: Any,
            session: dict[str, Any], model: Any) -> int:  # fmt: skip
    if not session:
        return EXIT_REFUSED
    if "start_step" not in session:  # stopped before the session started (attach / RUN wait): nothing to save
        _log(f"stopped before training: {session.get('halt') or session.get('halt_reason')}")
        return EXIT_INTERRUPTED if session.get("halt_reason") == "interrupted" else EXIT_HALTED
    missing = [a for a in ARTIFACTS if a != "summary.json" and not (out / a).is_file()]
    _write_json(out / "summary.json", {
        **run, "steps_done": int(model.num_timesteps), "n_updates": int(model._n_updates),
        "baseline_success": baseline["success_rate"] if baseline else None, "evals": len(evals),
        "best_success": best["success_rate"] if best else None, "best_step": best["step"] if best else None,
        "halt_reason": session.get("halt_reason"), "halt": session.get("halt"), "missing_artifacts": missing,
    })  # fmt: skip
    if session.get("halt_reason"):
        _log(f"session ended: {session['halt_reason']}: {session.get('halt')}; resume with --resume {out}")
    print(f"run dir {out}")
    if missing:
        _log(f"missing artifacts {missing}")
        return EXIT_HALTED
    if session.get("halt_reason") == "interrupted":
        return EXIT_INTERRUPTED
    return EXIT_HALTED if session.get("halt_reason") else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
