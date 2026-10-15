"""Usage: pixi run python scripts/finetune_awr_offline.py OUT_DIR [N_EVAL] [RUN_DIR] [FROZEN_END]  (repo root; CPU only)

Defaults: RUN_DIR runs/sac-her5/20261014-174624-bbcf, FROZEN_END 7000 (round 1). The behaviour policy is the run's
``init_from`` export (the actor must have been frozen for the transitions used).

Option 2 prototype: advantage-weighted regression (AWR) of the 5 mm sim actor on the real HS5-5mm transitions.

No critic gradient: V(s, g) is regressed on Monte Carlo returns (HER-relabelled goals: every relabelled return ends
at its goal, so no truncation bias), and the actor maximises sum w * log pi(a | s, g) with w = exp(A / beta),
A = G - V. With uniform weights this is maximum likelihood of the behaviour policy (= the frozen sim actor), so the
fixed point is the sim actor itself; only the weighting moves it. Scored in sim on held-out goal pairs.

Data: transitions collected while the actor was frozen (step < FROZEN_END), i.e. by the stochastic sim actor.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
torch.manual_seed(0)
RUN = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runs/sac-her5/20261014-174624-bbcf")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FROZEN_END = int(sys.argv[4]) if len(sys.argv) > 4 else 7000
N_RELABEL = 8
N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 100

import dataclasses
from so101_rl.config import load_calibration, load_config
from so101_rl.env.backend import SimBackend
from so101_rl.env.goal_env import So101GoalEnv
from so101_rl.env.goals import BoxGoalSampler
from so101_rl.env.tasks import REACH, success_np
from so101_rl.rl import deploy_policy as dp
from so101_rl.rl.policy_export import load_policy
from so101_rl.rl.sb3_bridge import FinetuneSAC, actor_policy, load_actor_into_sb3
from so101_rl.rl.train_sac_her import evaluate

run = json.loads((RUN / "run.json").read_text())
init = load_policy(Path(run["init_from"]))
task = dataclasses.replace(REACH, success_radius_m=float(run["success_radius_mm"]) / 1000.0)
cal = load_calibration(load_config().calibration_path)
cfg = load_config(calibration=cal)
genv = So101GoalEnv(SimBackend(cfg, cal, seed=0), cfg, task)
model = FinetuneSAC.load(RUN / "model.zip", env=genv, device="cpu", actor_learning_rate=1e-5, fixed_ent_coef=0.01,
                         anchor_weight=0.4)
model.load_replay_buffer(RUN / "replay_buffer.pkl")
load_actor_into_sb3(model, init)  # model.zip's actor may have drifted; start from the run's export
model.set_anchor(init)
rb = model.replay_buffer
gamma = model.gamma

# ---------------------------------------------------------------- dataset: episodes fully inside the frozen phase
starts, lens = rb.ep_start[: rb.pos, 0], rb.ep_length[: rb.pos, 0]
eps = sorted({(int(s), int(L)) for s, L in zip(starts, lens) if s + L <= FROZEN_END})
O = rb.observations
rng = np.random.default_rng(0)
rows = []  # (index, goal, return)
n_orig_succ = 0
for s, L in eps:
    idx = np.arange(s, s + L)
    ag_next = rb.next_observations["achieved_goal"][idx, 0].astype(np.float64)
    infos = [rb.infos[i][0] for i in idx]
    g0 = O["desired_goal"][s, 0].astype(np.float64)
    n_orig_succ += bool(success_np(ag_next[-1], g0, task))

    def returns(goal, t0):
        """MC return from every t >= t0 for ``goal``; the episode ends at the first success (terminated)."""
        gg = np.broadcast_to(goal, ag_next[t0:].shape)
        r = genv.compute_reward(ag_next[t0:], gg, infos[t0:])
        ok = success_np(ag_next[t0:], gg, task)
        G = np.zeros(len(r)); g = 0.0
        for k in range(len(r) - 1, -1, -1):
            g = r[k] + gamma * (0.0 if ok[k] else g)
            G[k] = g
        return G, ok

    G, _ = returns(g0, 0)
    rows += [(idx[k], g0, G[k]) for k in range(L)]
    for _ in range(N_RELABEL):  # HER "future": goal = an achieved position later in the same episode
        t = int(rng.integers(0, L)); tp = int(rng.integers(t, L))
        G, _ = returns(ag_next[tp], t)
        rows.append((idx[t], ag_next[tp].copy(), G[0]))
I = np.array([r[0] for r in rows]); GOAL = np.array([r[1] for r in rows], np.float32)
RET = np.array([r[2] for r in rows], np.float32)
obs_all = torch.as_tensor(O["observation"][I, 0]); goal_all = torch.as_tensor(GOAL)
act_all = torch.as_tensor(rb.actions[I, 0]).clamp(-1 + 1e-5, 1 - 1e-5)
ret_all = torch.as_tensor(RET)
x_all = torch.cat([obs_all, goal_all], 1)
info = {"episodes": len(eps), "orig_goal_success": n_orig_succ, "samples": len(rows),
        "return_mean": float(RET.mean()), "return_std": float(RET.std())}
print(json.dumps(info), flush=True)

# ---------------------------------------------------------------- V(s, g) on MC returns, episode-split validation
ep_of = np.searchsorted(np.array([s for s, _ in eps]), I, side="right") - 1
val = (ep_of % 5) == 0
V = nn.Sequential(nn.Linear(x_all.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
opt = torch.optim.Adam(V.parameters(), lr=3e-4, weight_decay=1e-4)
tr_idx = np.flatnonzero(~val); va_idx = np.flatnonzero(val)
best = (1e9, None)
for it in range(6000):
    b = torch.as_tensor(rng.choice(tr_idx, 512))
    loss = ((V(x_all[b]).squeeze(1) - ret_all[b]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 499:
        with torch.no_grad():
            pv = V(x_all[va_idx]).squeeze(1)
            vl = float(((pv - ret_all[va_idx]) ** 2).mean())
            corr = float(np.corrcoef(pv.numpy(), RET[va_idx])[0, 1])
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in V.state_dict().items()}, corr)
V.load_state_dict(best[1])
with torch.no_grad():
    A = ret_all - V(x_all).squeeze(1)
info["V_val_mse"] = best[0]; info["V_val_corr"] = best[2]; info["ret_var"] = float(RET[va_idx].var())
info["A_std"] = float(A.std())
print(json.dumps(info), flush=True)

# ---------------------------------------------------------------- AWR actor updates, scored in sim
obs_dict = {"observation": obs_all, "desired_goal": goal_all,
            "achieved_goal": torch.as_tensor(O["achieved_goal"][I, 0])}
anchor_sd = {k: v.clone() for k, v in model.actor.state_dict().items()}
pairs = dp.draw_goal_pairs(BoxGoalSampler(task.goal_bounds, fk=genv._fk, limits=cfg.joints, seed=0), 11, N_EVAL)
with torch.no_grad():
    mu_sim = model.anchor_actor(obs_dict, deterministic=True)


def score(tag):
    with torch.no_grad():
        mu = model.actor(obs_dict, deterministic=True)
    ev = evaluate(genv, actor_policy(model, init.meta), pairs)
    genv.reset()
    r = {"tag": tag, "mu_shift_p50": float((mu - mu_sim).norm(dim=1).median()),
         "mu_shift_p90": float((mu - mu_sim).norm(dim=1).quantile(0.9)),
         **{k: ev[k] for k in ["n", "success_rate", "trips", "final_error_m_p50"]}}
    print(json.dumps(r), flush=True)
    return r


results = {"info": info, "runs": [score("sim_actor")]}
for beta in [None, 2.0, 1.0, 0.5]:  # None = uniform weights (control: must stay at the sim actor)
    model.actor.load_state_dict(anchor_sd)
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
    w_all = torch.ones_like(A) if beta is None else torch.exp((A / (beta * A.std())).clamp(max=np.log(20.0)))
    info_w = {"w_max": float(w_all.max()), "ess_frac": float(w_all.sum() ** 2 / (w_all ** 2).sum() / len(w_all))}
    done = 0
    for k in [500, 2000]:
        for _ in range(k - done):
            b = torch.as_tensor(rng.integers(0, len(A), 256))
            mean, log_std, _ = model.actor.get_action_dist_params({kk: v[b] for kk, v in obs_dict.items()})
            lp = model.actor.action_dist.proba_distribution(mean, log_std).log_prob(act_all[b])
            loss = -(w_all[b] * lp).sum() / w_all[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        done = k
        r = score(f"beta={beta} steps={k}"); r.update(info_w); r["beta"] = beta; r["steps"] = k
        results["runs"].append(r)
        torch.save({k2: v.clone() for k2, v in model.actor.state_dict().items()},
                   OUT / f"actor_beta{beta}_steps{k}.pt")
(OUT / "awr.json").write_text(json.dumps(results, indent=1))
print("wrote", OUT / "awr.json")
