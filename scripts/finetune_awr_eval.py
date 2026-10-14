"""Usage: pixi run python scripts/finetune_awr_eval.py  (after finetune_awr_offline.py; CPU only)

Re-score saved AWR actors in sim on a fresh, larger goal set (seed 13)."""
import dataclasses, json, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
from so101_rl.config import load_calibration, load_config
from so101_rl.env.backend import SimBackend
from so101_rl.env.goal_env import So101GoalEnv
from so101_rl.env.goals import BoxGoalSampler
from so101_rl.env.tasks import REACH
from so101_rl.rl import deploy_policy as dp
from so101_rl.rl.policy_export import load_policy
from so101_rl.rl.train_sac_her import evaluate
D = Path("runs/sac-her5/awr-offline")
init = load_policy(Path("runs/sac-sim/reach-5mm-20261014-065212/policy.pt"))
task = dataclasses.replace(REACH, success_radius_m=0.005)
cal = load_calibration(load_config().calibration_path); cfg = load_config(calibration=cal)
genv = So101GoalEnv(SimBackend(cfg, cal, seed=1), cfg, task)
pairs = dp.draw_goal_pairs(BoxGoalSampler(task.goal_bounds, fk=genv._fk, limits=cfg.joints, seed=0), 13, 300)
sim_sd = {k: v.clone() for k, v in init.actor.state_dict().items()}
out = {}
for name in ["sim_actor", "actor_betaNone_steps500.pt", "actor_beta0.5_steps500.pt", "actor_beta2.0_steps500.pt"]:
    if name == "sim_actor":
        init.actor.load_state_dict(sim_sd)
    else:
        sd = torch.load(D / name)
        init.actor.load_state_dict({k: v for k, v in sd.items() if not k.startswith("features_extractor.")})
    # score in chunks of 50 so one trip does not end the whole score
    eps = []
    for c in range(0, 300, 50):
        ev = evaluate(genv, init, pairs[c:c + 50]); genv.reset()
        eps.append(ev)
    n = sum(e["n"] for e in eps); s = sum(e["success_rate"] * e["n"] for e in eps)
    out[name] = {"n": n, "success": s / n, "trips": sum(e["trips"] for e in eps),
                 "p50_chunks": [round(e["final_error_m_p50"] * 1000, 2) for e in eps]}
    print(name, json.dumps(out[name]), flush=True)
(D / "awr_eval300.json").write_text(json.dumps(out, indent=1))
