import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from so101_rl.rl.toy_env import ToyReachEnv


def test_check_env():
    check_env(ToyReachEnv(), skip_render_check=True)


def test_dict_observation_contract():
    env = ToyReachEnv()
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (4,) and obs["observation"].dtype == np.float32
    assert isinstance(env, gym.Env)


def _pd_policy(obs):
    pos, vel = obs["observation"][:2], obs["observation"][2:]
    return np.clip(6.0 * (obs["desired_goal"] - pos) - 3.0 * vel, -1, 1)


@pytest.mark.parametrize("seed", range(5))
def test_fixed_policy_reaches_goal(seed):
    env = ToyReachEnv()
    obs, _ = env.reset(seed=seed)
    for _ in range(env.episode_steps):
        obs, reward, terminated, truncated, info = env.step(_pd_policy(obs))
        if terminated or truncated:
            break
    assert terminated and info["is_success"]
    assert reward > 0  # success bonus outweighs the residual distance


def test_zero_policy_truncates_at_step_limit():
    env = ToyReachEnv(episode_steps=10)
    env.reset(seed=1)
    for i in range(10):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, np.float32))
    assert truncated and not terminated and i == 9


def test_compute_reward_vectorised():
    env = ToyReachEnv()
    a = np.zeros((7, 3, 2), np.float32)
    g = np.ones((7, 3, 2), np.float32)
    r = env.compute_reward(a, g, None)
    assert r.shape == (7, 3)
    assert np.allclose(r, -np.sqrt(2.0))
    near = env.compute_reward(np.zeros(2), np.zeros(2), {})
    assert near == pytest.approx(env.success_bonus)


def test_reset_is_deterministic_per_seed():
    a, _ = ToyReachEnv().reset(seed=3)
    b, _ = ToyReachEnv().reset(seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)
