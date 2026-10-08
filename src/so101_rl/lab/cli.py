"""Human CLI: ``pixi run lab status | stop <run_id> | gate-sign <run_id>``.

``gate-sign`` records the S4 exit-gate checklist sign-off for a training run in ``<run_dir>/gate.json``. MVP: a plain
local record (no cryptographic signature; signed records are deferred, plan §10 "MVP scope"). It is refused unless
the run's latest evaluation passed the numeric gate, and it never overwrites an existing record.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from so101_rl.lab import runs
from so101_rl.lab.api import Lab
from so101_rl.lab.errors import LabError

CHECKLIST = (
    "I watched the rendered eval rollouts (renders/) and the tip reaches and stays at the goal.",
    "No predicate gaming: no hovering at the 15 mm threshold, no oscillating in and out, no sliding through.",
    "The training curves in TensorBoard look sane (no divergence, no collapse late in training).",
    "The eval ran at the final DR stage with >= 1000 episodes, and the exported policy file exists.",
)


def gate_sign(lab: Lab, run_id: str, *, input_fn: Callable[[str], str] = input,
              print_fn: Callable[[str], None] = print) -> dict[str, Any]:
    run_dir = runs.run_dir_for(lab.root, run_id)
    run = runs.read_run(run_dir)
    if run.get("kind") == "eval":
        raise LabError("INVALID_ARGUMENT", "pass the training run, not the eval run")
    ev = lab._latest_eval(run_id)
    if not ev or ev.get("state") != "completed":
        raise LabError("GATE_MISSING", "no completed evaluation for this run", hint="run evaluate_policy first")
    eval_dir = lab.root / ev["run_id"]
    result = runs.read_json(eval_dir / "result.json") or {}
    if result.get("numeric_gate_passed") is not True:
        raise LabError("GATE_MISSING", "the latest evaluation did not pass the numeric gate",
                       details={"eval_run": ev["run_id"], "success_rate": result.get("success_rate"),
                                "wilson_lo": result.get("wilson_lo")})
    if (run_dir / "gate.json").exists():
        raise LabError("STATE_CONFLICT", "this run already has a gate record", details={"path": str(run_dir / "gate.json")})
    print_fn(f"Exit gate for {run_id} (eval {ev['run_id']}): success {result['success_rate']:.4f}, "
             f"Wilson lower bound {result['wilson_lo']:.4f}, episodes {result['episodes']}")
    print_fn(f"Renders and eval.json: {eval_dir}")
    ticks = []
    for i, item in enumerate(CHECKLIST, 1):
        ans = input_fn(f"[{i}/{len(CHECKLIST)}] {item} [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            raise LabError("GATE_MISSING", "checklist item not confirmed; nothing recorded", details={"item": i})
        ticks.append(item)
    if input_fn("Type the run id to sign: ").strip() != run_id:
        raise LabError("INVALID_ARGUMENT", "run id not confirmed; nothing recorded")
    record = {
        "schema": 1, "run_id": run_id, "signed": True, "signed_by": getpass.getuser(), "signed_at": runs.now_iso(),
        "eval_run": ev["run_id"], "numeric": {k: result.get(k) for k in
                                              ("success_rate", "wilson_lo", "wilson_hi", "episodes", "policy")},
        "checklist": ticks, "note": "MVP local record; not cryptographically signed",
    }
    fd = os.open(run_dir / "gate.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=1)
    runs.append_event(run_dir, "gate_signed", eval_run=ev["run_id"], by=record["signed_by"])
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lab", description="so101 lab: status, stop a run, sign the exit gate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="ledger and recent runs")
    s = sub.add_parser("stop", help="stop a run (checkpoint and exit)")
    s.add_argument("run_id")
    g = sub.add_parser("gate-sign", help="record the exit-gate checklist sign-off (interactive)")
    g.add_argument("run_id")
    a = p.parse_args(argv)
    lab = Lab()
    try:
        if a.cmd == "status":
            tasks = lab.list_tasks()
            for t in tasks["tasks"]:
                b = t["budget"]
                print(f"{t['name']}: attempts {b['attempts_used']}/{b['max_attempts']}, "
                      f"GPU-h {b['gpu_h_used']}/{b['gpu_h_max']}, probe GPU-h {b['probe_gpu_h_used']}/{b['probe_gpu_h_max']}")
            if tasks["halted"]:
                print(f"HALTED: see {lab.root / 'HALT'}")
            for r in lab.list_runs(limit=20)["runs"]:
                extra = f" gate={'PASS' if r['numeric_gate_passed'] else 'fail'}" if "numeric_gate_passed" in r else ""
                print(f"{r['run_id']:<52} {r['kind']:<7} {r['state']:<11} step={r['env_step']} gpu_h={r['gpu_h']}{extra}")
        elif a.cmd == "stop":
            out = lab.control_run(a.run_id, "stop", "cli", actor="human")
            print(f"{a.run_id}: stop requested (state {out['state']})")
        else:
            if not sys.stdin.isatty():
                print("gate-sign is interactive: run it in a terminal", file=sys.stderr)
                return 2
            rec = gate_sign(lab, a.run_id)
            print(f"recorded {lab.root / a.run_id / 'gate.json'} (by {rec['signed_by']})")
    except LabError as e:
        print(f"refused: {e.code}: {e.message}" + (f" ({e.hint})" if e.hint else ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
