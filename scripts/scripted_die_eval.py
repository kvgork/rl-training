"""Scripted die controller success per start stage on CPU MuJoCo (D2b of plans/die-to-zone-plan.md). No GPU.

Usage:
    pixi run -e cpu python scripts/scripted_die_eval.py [--episodes 50] [--nworld 10] [--stages 0,1,2,3]
        [--dr-stages 0,4] [--seed 0] [--out runs/scripted/die-eval-<stamp>.json]

Runs ``rl/scripted.py``'s controller in the batched die env (``eval_sim.build_env(..., device="cpu")``: the same env
code as training, physics on CPU MuJoCo through ``sim/cpu_sim.py``) with the task's own success predicate (tracker-
visible, in the zone, settled; the simulated tracker's occlusion / ring rule included), ``--episodes`` episodes per
(DR stage, start stage), and writes a json with the success rate, its Wilson interval and the failure modes (the
controller phase each failed episode ended in, trips, regrasps). Threads are capped (shared machine).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=50, help="episodes per (DR stage, start stage)")
    ap.add_argument("--nworld", type=int, default=10)
    ap.add_argument("--stages", default="0,1,2,3", help="die start stages (DIE_TO_ZONE.die_start; 3 = the well)")
    ap.add_argument("--dr-stages", default="0", help="domain-randomisation stages (sim/domain_rand.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", default=None, help="default: so101.yaml calibration_path")
    ap.add_argument("--sim-params", default="calib/sim_params.yaml")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    from so101_rl.env.tasks import DIE_TO_ZONE
    from so101_rl.rl import eval_sim
    from so101_rl.rl.scripted import ScriptedDieConfig, run_scripted_episodes

    ds = DIE_TO_ZONE.die_start
    out: dict = {"task": "die_to_zone", "device": "cpu (CPU MuJoCo, sim/cpu_sim.py)", "episodes_per_cell": args.episodes,
                 "seed": args.seed, "controller": ScriptedDieConfig().__dict__, "git_sha": eval_sim.git_sha(),
                 "results": []}  # fmt: skip
    t0 = time.time()
    for dr in (int(x) for x in args.dr_stages.split(",")):
        env = eval_sim.build_env("die_to_zone", args.nworld, stage=dr, seed=args.seed + 100 * dr, device="cpu",
                                 calibration=args.calibration, sim_params=args.sim_params)  # fmt: skip
        out.setdefault("env", {k: str(v) for k, v in env.metadata.items()})
        try:
            for k in (int(x) for x in args.stages.split(",")):
                env.set_start_stage(k)
                t1 = time.time()
                st = run_scripted_episodes(env, args.episodes)
                lo, hi = eval_sim.wilson_interval(st["successes"], st["episodes"])
                row = {"dr_stage": dr, "start_stage": k, "start_name": ds.stage_name(k), **st, "wilson_lo": lo,
                       "wilson_hi": hi, "wall_s": round(time.time() - t1, 1)}  # fmt: skip
                out["results"].append(row)
                print(f"DR {dr} S{k} {ds.stage_name(k):10s} success {st['successes']}/{st['episodes']} "
                      f"({st['success_rate']:.2f}, Wilson {lo:.2f}-{hi:.2f}) fails {st['fail_end_phase']} "
                      f"success phase {st['success_phase']} trips {st['trip_episodes']}", flush=True)  # fmt: skip
        finally:
            env.close()
    out["wall_s"] = round(time.time() - t0, 1)
    path = args.out or Path("runs/scripted") / f"die-eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
