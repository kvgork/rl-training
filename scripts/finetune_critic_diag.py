"""Usage: pixi run python scripts/finetune_critic_diag.py OUT_DIR  (from the repo root; CPU only)

Offline critic diagnosis of runs/sac-her5/20261014-174624-bbcf (CPU only, no arm).

A. Q(s,a) on the real stored transitions vs Monte Carlo discounted returns.
B. Q landscape around the sim actor's action: dQ/da size, Q gain along the gradient.
C. Replay the anchored actor update against the saved critic (critic frozen), then score the moved actors in sim.
"""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
RUN = Path("runs/sac-her5/20261014-174624-bbcf")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
res = {}

from so101_rl.config import load_calibration, load_config
from so101_rl.env.backend import SimBackend
from so101_rl.env.goal_env import So101GoalEnv
from so101_rl.env.goals import BoxGoalSampler
from so101_rl.env.tasks import REACH
from so101_rl.rl import deploy_policy as dp
from so101_rl.rl.policy_export import load_policy
from so101_rl.rl.sb3_bridge import FinetuneSAC, actor_policy
from so101_rl.rl.train_sac_her import evaluate

run = json.loads((RUN / "run.json").read_text())
init = load_policy(Path(run["init_from"]))
task = dataclasses.replace(REACH, success_radius_m=float(run["success_radius_mm"]) / 1000.0)
cal = load_calibration(load_config().calibration_path)
cfg = load_config(calibration=cal)
genv = So101GoalEnv(SimBackend(cfg, cal, seed=0), cfg, task)
model = FinetuneSAC.load(RUN / "model.zip", env=genv, device="cpu", actor_learning_rate=1e-5,
                         fixed_ent_coef=0.01, anchor_weight=0.4)
model.load_replay_buffer(RUN / "replay_buffer.pkl")
model.set_anchor(init)
rb = model.replay_buffer
N = rb.pos
gamma = model.gamma
print("transitions", N, "n_updates", model._n_updates, "gamma", gamma, flush=True)

# ---------------------------------------------------------------- A: Q vs MC return on real transitions
obs = {k: torch.as_tensor(v[:N, 0]) for k, v in rb.observations.items()}
act = torch.as_tensor(rb.actions[:N, 0])
rew = rb.rewards[:N, 0].astype(np.float64)
done = rb.dones[:N, 0].astype(bool)
tout = rb.timeouts[:N, 0].astype(bool)
starts = rb.ep_start[:N, 0]
lens = rb.ep_length[:N, 0]
term = done & ~tout
G = np.zeros(N)
ep_kind = np.empty(N, dtype=object)
ep_id = np.zeros(N, int)
eps = sorted(set(zip(starts.tolist(), lens.tolist())))
for k, (s, L) in enumerate(eps):
    idx = np.arange(s, s + L)
    g = 0.0
    for i in idx[::-1]:
        g = rew[i] + gamma * (0.0 if term[i] else g)
        G[i] = g
    succ_end = rew[idx[-1]] == 0.0
    kind = "success" if succ_end else ("trip" if term[idx[-1]] else "truncated")
    ep_kind[idx] = kind
    ep_id[idx] = k


def qmin(o, a):
    with torch.no_grad():
        q = torch.cat(model.critic(o, a), 1)
    return q.min(1).values.numpy(), q.numpy()


def batched(fn, n, bs=4096):
    outs = [fn(slice(i, min(i + bs, n))) for i in range(0, n, bs)]
    return [np.concatenate(z) for z in zip(*outs)]


def sl(o, s):
    return {k: v[s] for k, v in o.items()}


Q, Q2 = batched(lambda s: qmin(sl(obs, s), act[s]), N)
kinds = {}
for kind in ["success", "truncated", "trip"]:
    m = ep_kind == kind
    if m.sum() == 0:
        continue
    kinds[kind] = {"n_trans": int(m.sum()), "n_eps": int(len(set(ep_id[m]))), "mc_mean": float(G[m].mean()),
                   "q_mean": float(Q[m].mean()), "bias_mean": float((Q[m] - G[m]).mean()),
                   "corr": float(np.corrcoef(Q[m], G[m])[0, 1]) if m.sum() > 2 else None}
# segments by collection time: early (actor frozen, sim policy) vs late
seg = {}
for name, s in [("0-2000", slice(0, 2000)), ("2000-7000", slice(2000, 7000)), ("7000-end", slice(7000, N))]:
    m = np.zeros(N, bool); m[s] = True; m &= ep_kind == "success"
    if m.sum():
        seg[name] = {"n": int(m.sum()), "bias_mean": float((Q[m] - G[m]).mean()),
                     "corr": float(np.corrcoef(Q[m], G[m])[0, 1])}
res["A_q_vs_mc"] = {"by_episode_end": kinds, "success_eps_by_time": seg,
                    "critic_disagreement_mean_abs": float(np.abs(Q2[:, 0] - Q2[:, 1]).mean()),
                    "n_episodes": len(eps)}
print(json.dumps(res["A_q_vs_mc"], indent=1), flush=True)

# ---------------------------------------------------------------- B: Q landscape at the sim actor's action
anchor = model.anchor_actor
with torch.no_grad():
    mu_sim = anchor(obs, deterministic=True)
    _, logstd, _ = anchor.get_action_dist_params(obs)
std = logstd.exp()
a = mu_sim.clone().requires_grad_(True)
q = torch.cat(model.critic(obs, a), 1).min(1).values
q.sum().backward()
grad = a.grad.detach()
gn = grad.norm(dim=1)
dirn = grad / gn.clamp(min=1e-9)[:, None]
q0 = q.detach().numpy()
line = {}
for alpha in [0.02, 0.05, 0.1, 0.2, 0.4]:
    qa, _ = qmin(obs, (mu_sim + alpha * dirn).clamp(-1, 1))
    line[alpha] = {"q_gain_mean": float((qa - q0).mean()), "frac_better": float((qa > q0).mean())}
dist_data = (act - mu_sim).norm(dim=1)
res["B_landscape"] = {
    "q_at_mu_sim_mean": float(q0.mean()),
    "dq_da_norm": {"p50": float(gn.median()), "p90": float(gn.quantile(0.9)), "mean": float(gn.mean())},
    "policy_std_mean_per_joint": std.mean(0).tolist(),
    "stored_action_dist_from_mu_sim": {"p50": float(dist_data.median()), "p90": float(dist_data.quantile(0.9))},
    "q_along_gradient": line,
    "grad_direction_mean_per_joint": dirn.mean(0).tolist(),
}
print(json.dumps(res["B_landscape"], indent=1), flush=True)

# ---------------------------------------------------------------- C: replay the anchored actor update, score in sim
def actor_updates(m, k):
    """The actor half of FinetuneSAC.train (critic frozen), k gradient steps, HER batches of 256."""
    ent = torch.tensor(0.01)
    for _ in range(k):
        d = m.replay_buffer.sample(256, env=None)
        a_pi, lp = m.actor.action_log_prob(d.observations)
        qp = torch.cat(m.critic(d.observations, a_pi), 1).min(1, keepdim=True).values
        loss = (ent * lp.reshape(-1, 1) - qp).mean()
        with torch.no_grad():
            mu_s = m.anchor_actor(d.observations, deterministic=True)
        mse = F.mse_loss(m.actor(d.observations, deterministic=True), mu_s)
        loss = loss / qp.abs().mean().detach().clamp(min=1.0) + m.anchor_weight * mse
        m.actor.optimizer.zero_grad()
        loss.backward()
        m.actor.optimizer.step()


def actor_stats(m):
    with torch.no_grad():
        mu = m.actor(obs, deterministic=True)
        _, ls, _ = m.actor.get_action_dist_params(obs)
    qm, _ = qmin(obs, mu)
    return {"mu_shift_l2_p50": float((mu - mu_sim).norm(dim=1).median()),
            "mu_shift_l2_p90": float((mu - mu_sim).norm(dim=1).quantile(0.9)),
            "anchor_mse": float(F.mse_loss(mu, mu_sim)), "q_at_mu_mean": float(qm.mean()),
            "std_mean": float(ls.exp().mean())}


pairs = dp.draw_goal_pairs(BoxGoalSampler(task.goal_bounds, fk=genv._fk, limits=cfg.joints, seed=0), 7, 50)
model.critic.requires_grad_(False)
work = model  # deepcopy fails on the mujoco env; the original is not needed after B
for g in work.actor.optimizer.param_groups:
    g["lr"] = 1e-5
work.actor.set_training_mode(True)
res["C_replay"] = {}
done_k = 0
for k in [0, 600, 2000, 5000]:
    actor_updates(work, k - done_k)
    done_k = k
    st = actor_stats(work)
    ev = evaluate(genv, actor_policy(work, init.meta), pairs)
    genv.reset()
    st.update({kk: ev[kk] for kk in ["n", "success_rate", "trips", "final_error_m_p50"]})
    res["C_replay"][k] = st
    print(k, json.dumps(st), flush=True)

(OUT / "critic_diag.json").write_text(json.dumps(res, indent=1, default=float))
print("wrote", OUT / "critic_diag.json")
