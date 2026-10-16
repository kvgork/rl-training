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


# ------------------------------------------------------------------------------------------------ demo mixing (D2b)
def test_demo_count():
    from so101_rl.rl.replay_gpu import demo_count

    assert demo_count(4096, 0.0) == 0
    assert demo_count(4096, 0.25) == 1024
    assert demo_count(10, 0.01) == 1  # a positive ratio always draws a demo row
    assert demo_count(4, 0.99) == 3  # the online replay always contributes
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            demo_count(8, bad)


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5])
def test_sample_mixed_honours_the_ratio(ratio):
    from so101_rl.rl.replay_gpu import demo_count, sample_mixed

    online, demo = GpuReplay(64, 3, 2, 1, device="cpu"), GpuReplay(64, 3, 2, 1, device="cpu")
    _add(online, 0, n=16)  # reward 0 marks online rows
    _add(demo, 1, n=16)  # reward 1 marks demo rows
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        b = sample_mixed(online, demo, 128, ratio, generator=g)
        k = demo_count(128, ratio)
        assert b.obs.shape == (128, 3) and b.n_step_discount.shape == (128,)
        assert int((b.rew == 1.0).sum()) == k and torch.all(b.rew[128 - k :] == 1.0)
        assert torch.all(b.obs[b.rew == 1.0] == 1.0) and torch.all(b.next_obs[b.rew == 0.0] == 0.5)


def test_sample_mixed_without_demos_is_the_plain_sample():
    """demo_ratio 0 (or no demo replay) draws exactly what replay.sample draws from the same generator state."""
    from so101_rl.rl.replay_gpu import sample_mixed

    online, demo = GpuReplay(64, 3, 2, 1, device="cpu"), GpuReplay(64, 3, 2, 1, device="cpu")
    for k in range(5):
        _add(online, k, n=8)
    _add(demo, 9, n=8)
    for d, ratio in ((demo, 0.0), (None, 0.0), (None, 0.25)):
        a = sample_mixed(online, d, 32, ratio, generator=torch.Generator().manual_seed(7))
        b = online.sample(32, generator=torch.Generator().manual_seed(7))
        assert all(torch.equal(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)
