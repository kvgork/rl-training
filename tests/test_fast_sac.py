"""FastSAC (MVP M2, FS10): SB3 actor layout, tanh log-prob, truncation bootstrapping, alpha bounds, toy learning."""

import math
import time

import numpy as np
import pytest
import torch

from so101_rl.rl.eval_sim import run_episodes
from so101_rl.rl.fast_sac import Actor, FastSAC, SacConfig, tanh_gaussian_log_prob
from so101_rl.rl.replay_gpu import Batch, GpuReplay
from so101_rl.rl.toy_batched_env import ToyBatchedReachEnv


def _small(**kw):
    base = {"actor_hidden": (64, 64), "critic_hidden": (128, 128), "batch_size": 256, "grad_steps_per_env_step": 4,
                "target_entropy": -1.0, "gamma": 0.95, "lr_actor": 1e-3, "lr_critic": 1e-3, "lr_alpha": 1e-3}  # fmt: skip
    base.update(kw)
    return SacConfig(**base)


def test_tanh_logprob_matches_torch_distributions():
    torch.manual_seed(0)
    mu, log_std = torch.randn(512, 3), torch.randn(512, 3).clamp(-2, 1)
    u = mu + log_std.exp() * torch.randn(512, 3)
    ref = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mu, log_std.exp()), [torch.distributions.TanhTransform(cache_size=1)]
    )
    a = torch.tanh(u)
    ok = a.abs() < 0.999  # TanhTransform's inverse is ill-conditioned at the bounds
    got = tanh_gaussian_log_prob(u, mu, log_std)
    want = ref.log_prob(a).sum(-1)
    rows = ok.all(dim=1)
    torch.testing.assert_close(got[rows], want[rows], rtol=1e-3, atol=1e-3)


def test_tanh_logprob_numerical_1d():
    """Change of variables against a numerical histogram of a = tanh(u)."""
    mu, ls = torch.tensor([[0.3]]), torch.tensor([[-0.5]])
    a_grid = torch.linspace(-0.95, 0.95, 39)
    u = torch.atanh(a_grid)[:, None]
    dens = tanh_gaussian_log_prob(u, mu.expand(39, 1), ls.expand(39, 1)).exp()
    samples = torch.tanh(mu + ls.exp() * torch.randn(2_000_000, 1, generator=torch.Generator().manual_seed(1)))
    hist = torch.histc(samples, bins=400, min=-1, max=1) / (2_000_000 * 0.005)
    idx = ((a_grid + 1) / 0.005).long().clamp(0, 399)
    np.testing.assert_allclose(dens.numpy(), hist[idx].numpy(), rtol=0.06, atol=0.02)


def test_actor_state_dict_keys_match_sb3():
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    env = gym.make("Pendulum-v1")
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (24,), np.float32)
    env.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

    class _Dummy(gym.Env):
        observation_space = env.observation_space
        action_space = env.action_space

        def reset(self, *, seed=None, options=None):
            return np.zeros(24, np.float32), {}

        def step(self, a):
            return np.zeros(24, np.float32), 0.0, False, False, {}

    model = sb3.SAC("MlpPolicy", _Dummy(), policy_kwargs={"net_arch": [256, 256, 256]}, device="cpu", buffer_size=10)
    ours = Actor(24, 6)
    sd_sb3 = model.actor.state_dict()
    sd = ours.state_dict()
    assert set(sd) == set(sd_sb3)
    assert all(sd[k].shape == sd_sb3[k].shape for k in sd)
    # copied weights give SB3's deterministic action
    model.actor.load_state_dict(sd)
    x = torch.randn(8, 24)
    torch.testing.assert_close(model.actor(x, deterministic=True), ours(x), rtol=1e-5, atol=1e-6)


def _batch(n, obs_dim=4, priv_dim=2, act_dim=2, *, term=0.0, trunc=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Batch(obs=torch.randn(n, obs_dim, generator=g), priv=torch.randn(n, priv_dim, generator=g),
                 act=torch.rand(n, act_dim, generator=g) * 2 - 1, rew=torch.full((n,), -1.0),
                 next_obs=torch.randn(n, obs_dim, generator=g), next_priv=torch.randn(n, priv_dim, generator=g),
                 terminated=torch.full((n,), term), truncated=torch.full((n,), trunc),
                 n_step_discount=torch.full((n,), 0.9))  # fmt: skip


def test_terminal_does_not_bootstrap_and_truncation_does():
    sac = FastSAC(4, 2, 2, _small(), device="cpu")
    term = _batch(16, term=1.0)
    assert torch.equal(sac.td_target(term), term.rew)  # a terminal transition never bootstraps
    torch.manual_seed(5)
    y_trunc = sac.td_target(_batch(16, trunc=1.0))
    torch.manual_seed(5)
    y_mid = sac.td_target(_batch(16, trunc=0.0))
    torch.testing.assert_close(y_trunc, y_mid)  # truncation bootstraps exactly like a mid-episode step
    assert not torch.allclose(y_trunc, term.rew)
    m = sac.update(_batch(64, trunc=1.0))
    assert all(torch.isfinite(v) for v in m.values())


def test_truncation_bootstraps_constant_reward():
    """Constant reward -1, never terminated, truncated at 10 steps: the learned Q approaches -1/(1-gamma) at every
    step of the episode (it would decay to the remaining-steps return if truncation zeroed the bootstrap)."""
    torch.manual_seed(0)
    env = ToyBatchedReachEnv(32, episode_steps=10, constant_reward=-1.0, seed=0)
    cfg = _small(gamma=0.8, critic_hidden=(64, 64), alpha_init=1e-4, log_alpha_bounds=(-9.2, -9.0), lr_critic=3e-3, tau=0.05)
    sac = FastSAC(6, 2, 2, cfg, device="cpu")
    rb = GpuReplay(20_000, 6, 2, 2, device="cpu", gamma=cfg.gamma)
    obs, priv = env.reset()[:2]
    steps = []
    for _ in range(400):
        x = torch.cat([obs, env.goal], 1)
        a = torch.rand(32, 2) * 2 - 1
        st = env.step(a)
        rb.add(x, priv, a, st.reward, torch.cat([st.final_obs, st.desired], 1), st.final_priv,
               st.terminated.float(), st.truncated.float())  # fmt: skip
        steps.append(env.episode_step.clone())
        obs, priv = st.obs, st.priv
        if rb.size >= 256:
            for _ in range(4):
                sac.update(rb.sample(256))
    b = rb.sample(2048)
    with torch.no_grad():
        q = sac.critic(torch.cat([b.obs, b.priv, b.act], -1)).mean(0)
    target = -1.0 / (1 - cfg.gamma)
    last = b.truncated == 1.0
    assert last.any()
    assert abs(float(q[last].mean()) - target) / abs(target) < 0.05
    assert abs(float(q[~last].mean()) - float(q[last].mean())) / abs(target) < 0.05


def test_alpha_bounds():
    sac = FastSAC(4, 2, 2, _small(target_entropy=50.0, lr_alpha=0.5), device="cpu")
    for _ in range(30):
        sac.update(_batch(64))
    assert float(sac.log_alpha) <= 0.0 + 1e-9
    sac = FastSAC(4, 2, 2, _small(target_entropy=-50.0, lr_alpha=0.5), device="cpu")
    for _ in range(30):
        sac.update(_batch(64))
    assert float(sac.log_alpha) >= -9.2 - 1e-6


def test_state_dict_roundtrip_and_dims_check():
    a = FastSAC(4, 2, 2, _small(), device="cpu")
    a.update(_batch(64))
    b = FastSAC(4, 2, 2, _small(), device="cpu")
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 4)
    torch.testing.assert_close(a.act(x, deterministic=True), b.act(x, deterministic=True))
    assert b.updates == 1
    with pytest.raises(ValueError):
        FastSAC(5, 2, 2, _small(), device="cpu").load_state_dict(a.state_dict())
    assert SacConfig.from_dict(a.cfg.to_dict()) == a.cfg


@pytest.mark.slow
@pytest.mark.timeout(240)
def test_learns_toy_reach_cpu():
    """64 toy worlds on CPU reach >= 90 % deterministic success well within 3 minutes."""
    torch.manual_seed(0)
    env, ev = ToyBatchedReachEnv(64, seed=0), ToyBatchedReachEnv(200, seed=1)
    sac = FastSAC(6, 2, 2, _small(), device="cpu")
    rb = GpuReplay(100_000, 6, 2, 2, device="cpu", gamma=0.95)
    obs, priv = env.reset()[:2]
    t0, best = time.monotonic(), 0.0
    for it in range(2000):
        x = torch.cat([obs, env.goal], 1)
        a = torch.rand(64, 2) * 2 - 1 if it < 20 else sac.act(x)
        st = env.step(a)
        rb.add(x, priv, a, st.reward, torch.cat([st.final_obs, st.desired], 1), st.final_priv,
               st.terminated.float(), st.truncated.float())  # fmt: skip
        obs, priv = st.obs, st.priv
        if rb.size >= 256:
            for _ in range(4):
                sac.update(rb.sample(256))
        if it % 100 == 99:
            best = max(best, run_episodes(ev, lambda z: sac.act(z, deterministic=True), 200)["success_rate"])
            if best >= 0.9 or time.monotonic() - t0 > 180:
                break
    assert best >= 0.9, best
    assert math.isfinite(float(sac.log_alpha))
