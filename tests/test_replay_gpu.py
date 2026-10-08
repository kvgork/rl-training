"""GPU replay (MVP M2, FS10): ring semantics, pre-reset next_obs, terminated/truncated stored separately."""

import pytest
import torch

from so101_rl.rl.replay_gpu import GpuReplay


def _add(rb, k, n=4):
    v = torch.full((n, 3), float(k))
    rb.add(v, v[:, :2], v[:, :1], torch.full((n,), float(k)), v + 0.5, v[:, :2] + 0.5,
           torch.zeros(n), torch.zeros(n))  # fmt: skip


def test_ring_overwrite():
    rb = GpuReplay(10, 3, 2, 1, device="cpu")
    for k in range(4):
        _add(rb, k)
    assert rb.size == 10
    vals = set(rb._rew.tolist())
    assert vals == {1.0, 2.0, 3.0}  # step 0 overwritten, 2 slots of step 1 overwritten
    assert sorted(rb._rew.tolist()).count(3.0) == 4


def test_final_obs_used_for_reset_worlds():
    """next_obs passed for a resetting world is stored as given (the pre-reset final_obs), not the post-reset obs."""
    rb = GpuReplay(8, 2, 1, 1, device="cpu")
    obs = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    final = torch.tensor([[9.0, 9.0], [1.5, 1.5]])  # world 0 reset this step: its final_obs differs from the new obs
    rb.add(obs, obs[:, :1], torch.zeros(2, 1), torch.zeros(2), final, final[:, :1], torch.zeros(2), torch.tensor([1.0, 0.0]))
    b = rb.sample(64, generator=torch.Generator().manual_seed(0))
    w0 = b.obs[:, 0] == 0.0
    assert torch.all(b.next_obs[w0] == 9.0) and torch.all(b.truncated[w0] == 1.0) and torch.all(b.terminated == 0)
    assert torch.allclose(b.n_step_discount, torch.full((64,), 0.97))


def test_sample_shapes_and_errors():
    rb = GpuReplay(16, 3, 2, 1, device="cpu")
    with pytest.raises(RuntimeError):
        rb.sample(4)
    _add(rb, 1)
    b = rb.sample(5)
    assert b.obs.shape == (5, 3) and b.priv.shape == (5, 2) and b.act.shape == (5, 1) and b.rew.shape == (5,)
    assert rb.vram_bytes > 0
    with pytest.raises(ValueError):
        _add(rb, 1, n=17)
    with pytest.raises(NotImplementedError):
        GpuReplay(4, 1, 1, 1, device="cpu", n_step=3)
    with pytest.raises(ValueError):
        GpuReplay(0, 1, 1, 1, device="cpu")
