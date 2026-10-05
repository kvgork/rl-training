"""FS3 policy observation layout (sim-first track S0.4)."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_rl.config import CENTER_TICK
from so101_rl.env.obs import (
    DIE_OBS,
    GOAL_DIM,
    OBS_LAYOUT_VERSION,
    PRIV_OBS,
    REACH_OBS,
    DieObs,
    DieObsT,
    build_policy_obs_np,
    build_policy_obs_torch,
    policy_input_dim,
)


def test_layout_version_constant():
    assert OBS_LAYOUT_VERSION == 1
    assert GOAL_DIM == 3


def test_dims():
    assert REACH_OBS.dim == 21 and DIE_OBS.dim == 26
    assert policy_input_dim(REACH_OBS) == 24 and policy_input_dim(DIE_OBS) == 29
    assert PRIV_OBS.dim == 61


@pytest.mark.parametrize("layout", [REACH_OBS, DIE_OBS, PRIV_OBS])
def test_slices_tile_the_vector_without_gaps(layout):
    covered = []
    for name in layout.__dataclass_fields__:
        sl = getattr(layout, name)
        if isinstance(sl, slice):
            covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(layout.dim))


def test_order_is_frozen():
    assert [REACH_OBS.q_norm, REACH_OBS.dq_norm, REACH_OBS.last_applied, REACH_OBS.tip_xyz] == [
        slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 21)
    ]
    assert (DIE_OBS.die_xyz, DIE_OBS.die_visible, DIE_OBS.die_age) == (slice(21, 24), slice(24, 25), slice(25, 26))
    assert REACH_OBS.die_xyz is None and not REACH_OBS.has_die and DIE_OBS.has_die


def _random_state(rng, n, cfg):
    present = rng.integers(0, 4096, size=(n, 6))
    vel = rng.integers(-3000, 3000, size=(n, 6))  # ticks/s, beyond the clip on purpose
    applied = rng.integers(-80, 80, size=(n, 6)).astype(np.float32)
    tip = rng.uniform(-0.3, 0.3, size=(n, 3))
    die_xyz = rng.uniform(-0.3, 0.3, size=(n, 3))
    vis = rng.random(n) < 0.5
    age = rng.integers(0, 100, size=n)
    return present, vel, applied, tip, die_xyz, vis, age


@pytest.mark.parametrize("layout", [REACH_OBS, DIE_OBS])
def test_np_and_torch_builders_identical(layout, cfg):
    rng = np.random.default_rng(0)
    n = 512
    present, vel, applied, tip, die_xyz, vis, age = _random_state(rng, n, cfg)
    die_np = DieObs(xyz=die_xyz, visible=vis, age_frames=age) if layout.has_die else None
    die_t = (
        DieObsT(xyz=torch.as_tensor(die_xyz), visible=torch.as_tensor(vis), age_frames=torch.as_tensor(age))
        if layout.has_die
        else None
    )
    a = build_policy_obs_np(
        layout, present_ticks=present, velocity=vel, applied_delta_ticks=applied, tip_xyz=tip, cfg=cfg, die=die_np,
        out=np.empty((n, layout.dim), np.float32),
    )
    b = build_policy_obs_torch(
        layout, present_ticks=torch.as_tensor(present, dtype=torch.int64), velocity=torch.as_tensor(vel),
        applied_delta_ticks=torch.as_tensor(applied), tip_xyz=torch.as_tensor(tip), cfg=cfg, die=die_t,
        out=torch.empty((n, layout.dim), dtype=torch.float32),
    )
    assert np.max(np.abs(a - b.numpy())) < 1e-6


def test_values_follow_the_definition(cfg):
    present = np.full(6, CENTER_TICK) + np.array([0, 2048, -1024, 10, 0, 0])
    per_tick = cfg.action.max_ticks_per_tick
    vel = np.array([0, 1, -1, 3, 0, 0]) * per_tick * cfg.rates.servo_hz  # 0, +1, -1, +3 (clipped to 2) caps per tick
    applied = np.array([1, -1, 0.5, 2, 0, 0]) * cfg.action.max_delta_ticks
    die = DieObs(xyz=np.array([0.2, -0.05, -0.012]), visible=False, age_frames=45)
    o = build_policy_obs_np(DIE_OBS, present_ticks=present, velocity=vel, applied_delta_ticks=applied,
                            tip_xyz=np.array([0.2, 0.0, 0.15]), cfg=cfg, die=die, out=np.empty(26, np.float32))
    np.testing.assert_allclose(o[0:6], [0, 1, -0.5, 10 / 2048, 0, 0], atol=1e-7)
    np.testing.assert_allclose(o[6:12], [0, 1, -1, 2, 0, 0], atol=1e-7)
    np.testing.assert_allclose(o[12:18], [1, -1, 0.5, 1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(o[18:21], [0.2, 0.0, 0.15], atol=1e-7)
    np.testing.assert_allclose(o[21:26], [0.2, -0.05, -0.012, 0.0, 1.0], atol=1e-7)  # held xyz, invisible, age capped


def test_die_block_required_exactly_for_die_layouts(cfg):
    kw = {"present_ticks": np.zeros(6), "velocity": np.zeros(6), "applied_delta_ticks": np.zeros(6),
          "tip_xyz": np.zeros(3), "cfg": cfg}
    with pytest.raises(ValueError, match="die"):
        build_policy_obs_np(DIE_OBS, die=None, out=np.empty(26, np.float32), **kw)
    with pytest.raises(ValueError, match="die"):
        build_policy_obs_np(REACH_OBS, die=DieObs(np.zeros(3), True, 0), out=np.empty(21, np.float32), **kw)
    with pytest.raises(ValueError, match="columns"):
        build_policy_obs_np(REACH_OBS, die=None, out=np.empty(26, np.float32), **kw)


def test_env_obs_and_tasks_do_not_import_safety():
    """D-DEPS: env/ never imports safety/ clamps (these modules import only config and each other)."""
    root = Path(__file__).parent.parent / "src" / "so101_rl" / "env"
    for name in ("obs.py", "tasks.py", "reward.py"):
        tree = ast.parse((root / name).read_text())
        mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any("safety" in m or m.startswith(("so101_rl.rl", "so101_rl.sim")) for m in mods), (name, mods)


def test_die_block_visible_age_and_never_seen(cfg):
    kw = {"present_ticks": np.full(6, CENTER_TICK), "velocity": np.zeros(6), "applied_delta_ticks": np.zeros(6),
          "tip_xyz": np.zeros(3), "cfg": cfg}
    seen = build_policy_obs_np(DIE_OBS, die=DieObs(np.array([0.1, 0.2, -0.012]), True, 0), out=np.empty(26), **kw)
    np.testing.assert_allclose(seen[21:26], [0.1, 0.2, -0.012, 1.0, 0.0])
    never = build_policy_obs_np(DIE_OBS, die=DieObs(np.zeros(3), False, 10**6), out=np.empty(26), **kw)
    np.testing.assert_allclose(never[21:26], [0, 0, 0, 0.0, 1.0])  # zeros until first seen, age saturates
    for age, expect in ((29, 29 / 30), (30, 1.0), (31, 1.0)):
        o = build_policy_obs_np(DIE_OBS, die=DieObs(np.zeros(3), False, age), out=np.empty(26), **kw)
        assert o[25] == pytest.approx(expect)
        t = build_policy_obs_torch(
            DIE_OBS, present_ticks=torch.full((1, 6), CENTER_TICK), velocity=torch.zeros(1, 6),
            applied_delta_ticks=torch.zeros(1, 6), tip_xyz=torch.zeros(1, 3), cfg=cfg,
            die=DieObsT(torch.zeros(1, 3), torch.tensor([True]), torch.tensor([age])), out=torch.empty(1, 26),
        )
        assert float(t[0, 25]) == pytest.approx(expect) and float(t[0, 24]) == 1.0


def test_torch_scale_cache_is_weak_and_bounded(calibration):
    """The per-(cfg, device) tensor cache must not keep configs alive (SU0 review NIT)."""
    import gc

    import torch

    from so101_rl.config import load_config
    from so101_rl.env import obs

    before = len(obs._TORCH_SCALES)
    cfgs = [load_config(calibration=calibration) for _ in range(3)]
    for c in cfgs:
        a = obs._torch_scales(c, torch.device("cpu"))
        assert obs._torch_scales(c, torch.device("cpu"))[0] is a[0]  # cached per (cfg, device)
    assert len(obs._TORCH_SCALES) == before + 3
    del cfgs, c
    gc.collect()
    assert len(obs._TORCH_SCALES) == before
