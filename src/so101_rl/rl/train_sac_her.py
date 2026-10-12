"""SB3 SAC + HER fine-tuning of the sim reach actor (sim plan S5.2; pixi task ``train-reach-sac``).

``train-reach-sac --init-from <policy_export.pt> --backend sim --total-steps N [--warmup-frozen-actor 2000]
[--baseline-eval-every 1000 --eval-episodes 10]``

* **Model:** :class:`~so101_rl.rl.sb3_bridge.FinetuneSAC` on ``So101GoalEnv(REACH)``, the export's actor copied in
  (critics fresh), ``HerReplayBuffer`` (4 sampled goals, ``future``, the info dicts kept: the relabelled reward reads the trip bits), ``ent_coef="auto_0.01"``, lr 1e-4, batch 256,
  ``gamma`` = the task's, ``tau`` 0.005, ``learning_starts`` 2000. Updates run **between episodes**
  (``train_freq=(1, "episode")``, one gradient step per collected step), never inside one: on the arm the 10 Hz
  loop must not wait for an update. Every action is the actor's (no uniform-random warm-up); the actor and the
  entropy coefficient stay frozen for ``--warmup-frozen-actor`` gradient steps while the critics learn.
* **Baseline and rollback:** before learning, the export is evaluated deterministically on ``--eval-episodes``
  fixed goals (drawn from ``--eval-seed`` as in ``deploy-policy``) = ``baseline.json``. At the first episode end
  after every ``--baseline-eval-every`` steps the current actor is evaluated on the same goals (``evals.jsonl``);
  the best actor so far is kept, and when two consecutive evals fall below baseline − ``--rollback-margin`` the
  policy (actor and critics) is rolled back to the best one.
* **Backends:** ``sim`` = the in-process CPU reference rig (``SimBackend``: ServoCore + MujocoBus). ``shm`` (the real
  arm, HS5) is not built yet and is refused.
* **Outputs** ``<out-root>/<ts>/``: ``model.zip``, ``monitor.csv``, ``baseline.json``, ``evals.jsonl``,
  ``policy_export.pt`` (the final actor, deployable) and ``summary.json``. Exit 0 only when all are written; 2 on a
  refusal (``--total-steps`` < 1, a bad export, ``--backend shm``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_REFUSED: Final[int] = 2
ARTIFACTS: Final[tuple[str, ...]] = ("model.zip", "monitor.csv", "baseline.json", "evals.jsonl", "policy_export.pt",
                                     "summary.json")  # fmt: skip


def evaluate(env: Any, policy: Any, pairs: np.ndarray) -> dict[str, Any]:
    """Deterministic episodes of ``policy`` on the fixed goal ``pairs`` (``deploy-policy``'s episode loop); the env's
    goal sampler is restored afterwards. The caller must reset the env before training continues."""
    from so101_rl.rl.deploy_policy import GoalList, _run_episode

    saved = env.goal_sampler
    env.goal_sampler = goals = GoalList(pairs)
    try:
        eps = [_run_episode(env, policy, 1.0, goals=goals, index=i, infer_ms=[]) for i in range(len(pairs))]
    finally:
        env.goal_sampler = saved
    err = np.asarray([e["final_error_m"] for e in eps])
    return {"n": len(eps), "success_rate": float(np.mean([e["success"] for e in eps])),
            "trips": int(sum(e["tripped"] for e in eps)), "final_error_m_p50": float(np.median(err))}  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train-reach-sac", description=__doc__.splitlines()[0])
    ap.add_argument("--init-from", type=Path, required=True, help="policy_export.pt of the sim actor")
    ap.add_argument("--backend", choices=("sim", "shm"), default="sim")
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--warmup-frozen-actor", type=int, default=2000, help="gradient steps with the actor frozen")
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--baseline-eval-every", type=int, default=1000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--rollback-margin", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", type=Path, default=Path("runs/sac-her"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_steps < 1:
        print("train-reach-sac: refused: --total-steps must be >= 1", file=sys.stderr)
        return EXIT_REFUSED
    if args.backend == "shm":
        print("train-reach-sac: refused: --backend shm (real-arm fine-tuning, HS5) is not built yet", file=sys.stderr)
        return EXIT_REFUSED
    from so101_rl.rl.policy_export import load_policy

    try:
        init = load_policy(args.init_from)
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        print(f"train-reach-sac: refused: cannot use {args.init_from}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if init.meta.get("task_name") != "reach":
        print(f"train-reach-sac: refused: the export's task is {init.meta.get('task_name')!r}, not 'reach'",
              file=sys.stderr)  # fmt: skip
        return EXIT_REFUSED
    out = args.out_root / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
    out.mkdir(parents=True)
    return _train(args, init, out)


def _train(args: argparse.Namespace, init: Any, out: Path) -> int:
    import torch
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from so101_rl.env import make_goal_env
    from so101_rl.env.goals import BoxGoalSampler
    from so101_rl.env.tasks import REACH
    from so101_rl.rl.deploy_policy import draw_goal_pairs
    from so101_rl.rl.sb3_bridge import (
        FinetuneSAC,
        GoalPolicyInput,
        actor_policy,
        export_sb3_actor,
        load_actor_into_sb3,  # fmt: skip
        rollback_due,
    )

    torch.manual_seed(args.seed)
    env = make_goal_env("reach", backend="sim", sim_kwargs={"seed": args.seed})
    genv = env  # the So101GoalEnv the evals drive directly (Monitor only wraps it for SB3)
    pairs = draw_goal_pairs(BoxGoalSampler(REACH.goal_bounds, fk=genv._fk, limits=genv.cfg.joints, seed=0),
                            args.eval_seed, args.eval_episodes)  # fmt: skip
    model = FinetuneSAC(
        "MultiInputPolicy", Monitor(env, str(out / "monitor.csv")),
        policy_kwargs={"net_arch": list(init.meta["actor_hidden"]), "features_extractor_class": GoalPolicyInput},
        replay_buffer_class=HerReplayBuffer,
        # copy_info_dict: the relabelled reward reads the stored info (trip bits -> trip_penalty, plan §10 2026-10-08)
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
        ent_coef="auto_0.01", learning_rate=1e-4, learning_starts=args.learning_starts, batch_size=256,
        gamma=REACH.gamma, tau=0.005, train_freq=(1, "episode"), gradient_steps=-1,
        warmup_frozen_actor=args.warmup_frozen_actor, seed=args.seed, device=args.device, verbose=0,
    )  # fmt: skip
    load_actor_into_sb3(model, init)

    baseline = {"policy": str(args.init_from.resolve()), "eval_seed": args.eval_seed, "goals": pairs.tolist(),
                **evaluate(genv, init, pairs)}  # fmt: skip
    _write_json(out / "baseline.json", baseline)
    print(f"baseline: {baseline['success_rate']:.2f} success over {baseline['n']} episodes", flush=True)
    evals: list[dict[str, Any]] = []
    best = {"success_rate": baseline["success_rate"], "state": copy.deepcopy(model.policy.state_dict()), "step": 0}

    class BaselineEval(BaseCallback):
        """At the first episode end (rollout end) after every ``--baseline-eval-every`` steps: eval, best, rollback."""

        def __init__(self) -> None:
            super().__init__()
            self.next_at = args.baseline_eval_every

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps < self.next_at:
                return
            self.next_at += args.baseline_eval_every
            r = {"step": self.num_timesteps, "actor_frozen": model.actor_frozen,
                 **evaluate(genv, actor_policy(model, init.meta), pairs)}  # fmt: skip
            if r["success_rate"] > best["success_rate"]:
                best.update(success_rate=r["success_rate"], state=copy.deepcopy(model.policy.state_dict()),
                            step=self.num_timesteps)  # fmt: skip
            r["rolled_back"] = rollback_due([e["success_rate"] for e in evals] + [r["success_rate"]],
                                            baseline["success_rate"], margin=args.rollback_margin)  # fmt: skip
            if r["rolled_back"]:
                model.policy.load_state_dict(best["state"])
                r["rolled_back_to_step"] = best["step"]
            evals.append(r)
            with (out / "evals.jsonl").open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"eval @ {r['step']}: {r['success_rate']:.2f}" + (" ROLLBACK" if r["rolled_back"] else ""),
                  flush=True)  # fmt: skip
            model._last_obs = self.training_env.reset()  # the evals moved the arm: start a fresh episode

    (out / "evals.jsonl").touch()
    code = EXIT_OK
    try:
        model.learn(total_timesteps=args.total_steps, callback=BaselineEval())
    finally:
        model.save(out / "model.zip")
        export_sb3_actor(model, init.meta, out / "policy_export.pt", finetuned_from=str(args.init_from.resolve()),
                         finetune_steps=int(model.num_timesteps), finetune_backend=args.backend)  # fmt: skip
        env.close()
    missing = [a for a in ARTIFACTS if a != "summary.json" and not (out / a).is_file()]
    _write_json(out / "summary.json", {
        "init_from": str(args.init_from.resolve()), "backend": args.backend, "total_steps": int(model.num_timesteps),
        "n_updates": int(model._n_updates), "baseline_success": baseline["success_rate"], "evals": len(evals),
        "best_success": best["success_rate"], "best_step": best["step"], "missing_artifacts": missing,
    })  # fmt: skip
    if missing:
        print(f"train-reach-sac: missing artifacts {missing}", file=sys.stderr)
        code = EXIT_FAILED
    print(f"run dir {out}")
    return code


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
