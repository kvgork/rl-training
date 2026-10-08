"""Reach domain randomisation (MVP M1; sim plan S3.1 reach subset): sim/domain_rand.py + the BatchedGoalEnv hooks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from so101_rl.sim.domain_rand import REACH_DR_STAGES, DomainRandomizer, DrStage, next_stage
from so101_rl.sim.sysid.params import read_sim_params_file

ROOT = Path(__file__).parent.parent
gpu = pytest.mark.gpu


@pytest.fixture(scope="module")
def spf():
    return read_sim_params_file(ROOT / "calib" / "sim_params.yaml", allow_unverified=True)


def _rnd(spf, n=10_000, stage=0, seed=0):
    return DomainRandomizer(spf.params, spf.dr, n, stage=stage, seed=seed)


def test_stage0_is_identity(spf):
    r = _rnd(spf, 64)
    p = r.sample_episode(torch.ones(64, dtype=torch.bool))
    for name in ("kp", "kv", "damping", "frictionloss", "armature", "forcerange"):
        np.testing.assert_allclose(getattr(p, name).numpy(), np.broadcast_to(getattr(spf.params, name), (64, 6)), rtol=1e-6)
    assert (p.delay_ticks == spf.params.delay_ticks).all()
    assert (r.proprio_noise == 0).all()
    np.testing.assert_allclose(r.dr_ratios().numpy(), 1.0, rtol=1e-6)


@pytest.mark.parametrize("stage", range(len(REACH_DR_STAGES)))
def test_ranges_within_spec(spf, stage):
    r = _rnd(spf, 10_000, stage=stage)
    p = r.sample_episode(torch.ones(10_000, dtype=torch.bool))
    s = REACH_DR_STAGES[stage]
    rel = s.servo_frac * spf.dr.servo_group_rel
    g = p.kp.numpy() / spf.params.kp
    assert g.min() >= 1 - rel - 1e-6 and g.max() <= 1 + rel + 1e-6
    # one ratio group per joint: kp, kv and armature move together (plan §10 "Servo parameter identification")
    np.testing.assert_allclose(p.kv.numpy() / spf.params.kv, g, rtol=1e-5)
    np.testing.assert_allclose(p.armature.numpy() / spf.params.armature, g, rtol=1e-5)
    np.testing.assert_allclose(p.damping.numpy(), np.broadcast_to(spf.params.damping, (10_000, 6)), rtol=1e-6)
    lo, hi = r.ranges()["frictionloss"]
    fl = p.frictionloss.numpy()
    assert fl.min() >= lo - 1e-7 and fl.max() <= hi + 1e-7
    d = p.delay_ticks.numpy() - spf.params.delay_ticks
    assert d.min() >= s.delay_extra_ticks[0] and d.max() <= s.delay_extra_ticks[1]
    assert (r.proprio_noise == s.proprio_noise_ticks).all()
    if rel > 0:  # the draws actually fill the range
        assert g.min() < 1 - 0.9 * rel and g.max() > 1 + 0.9 * rel


def test_final_stage_is_the_logged_full_range(spf):
    """Last stage: servo group +-40 % around the sim_params centre, frictionloss over 0..0.21 N m (plan §10)."""
    from so101_rl.rl.eval_sim import GATE_FINAL_STAGE

    assert REACH_DR_STAGES[-1].index == GATE_FINAL_STAGE
    r = _rnd(spf, 20_000, stage=GATE_FINAL_STAGE)
    rg = r.ranges()
    assert rg["servo_group"] == pytest.approx((0.6, 1.4))
    assert rg["frictionloss"] == pytest.approx((0.0, 0.21))
    p = r.sample_episode(torch.ones(20_000, dtype=torch.bool))
    assert p.frictionloss.min() < 0.005 and p.frictionloss.max() > 0.205
    g = p.kp.numpy() / spf.params.kp
    assert abs(g.mean() - 1.0) < 0.01  # centred on calib/sim_params.yaml


def test_per_episode_params_change_only_on_reset_mask(spf):
    r = _rnd(spf, 8, stage=4)
    p = r.sample_episode(torch.ones(8, dtype=torch.bool))
    before = {k: getattr(p, k).clone() for k in ("kp", "kv", "frictionloss", "armature", "delay_ticks")}
    mask = torch.tensor([True, False] * 4)
    r.sample_episode(mask)
    for k, old in before.items():
        new = getattr(r.params, k)
        assert torch.equal(new[~mask], old[~mask]), k
        if k != "delay_ticks":
            assert not torch.equal(new[mask], old[mask]), k


def test_next_stage_rule():
    assert next_stage(0, 0.69, 4) == 0
    assert next_stage(0, 0.70, 4) == 1
    assert next_stage(4, 1.0, 4) == 4
    assert next_stage(2, 0.9, 2) == 2
    assert next_stage(1, float("nan"), 4) == 1


def test_stage_validation(spf):
    with pytest.raises(ValueError):
        DrStage(0, "x", 1.5, 0.0, (0, 0), 0)
    with pytest.raises(ValueError):
        DrStage(0, "x", 0.5, 0.0, (2, 1), 0)
    with pytest.raises(ValueError):
        _rnd(spf, 4).set_stage(len(REACH_DR_STAGES))
    with pytest.raises(ValueError):
        DomainRandomizer(spf.params, spf.dr, 4, stages=(REACH_DR_STAGES[1],))


def test_same_seed_same_draws(spf):
    a = _rnd(spf, 32, stage=4, seed=3).sample_episode(torch.ones(32, dtype=torch.bool)).kp.clone()
    b = _rnd(spf, 32, stage=4, seed=3).sample_episode(torch.ones(32, dtype=torch.bool)).kp.clone()
    assert torch.equal(a, b)


# ------------------------------------------------------------------------------------------------ env hooks (GPU)
def _env(stage, n=16, noise=0):
    from so101_rl.config import load_calibration, load_config
    from so101_rl.env.batched_env import BatchedGoalEnv
    from so101_rl.env.tasks import REACH
    from so101_rl.sim.model import load_model

    cal = load_calibration(ROOT / "tests/data/so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    f = read_sim_params_file(ROOT / "calib" / "sim_params.yaml", allow_unverified=True)
    rnd = DomainRandomizer(f.params, f.dr, n, stage=stage, seed=1, device="cuda:0")
    env = BatchedGoalEnv(cfg, cal, REACH, n, sim_params=f.params, randomizer=rnd, model=load_model()[0], seed=0)
    env.start_noise_ticks = noise
    return env


@gpu
def test_env_applies_dr_per_world():
    """Same start, same actions: nominal worlds move identically, stage-4 worlds (servo group, friction, delay) do
    not; the privileged dr_params block reports the draws."""
    spreads = {}
    for stage in (0, 4):
        env = _env(stage)
        try:
            env.reset(phase_randomise=False)
            a = torch.zeros(env.nworld, 6, device=env.device)
            a[:, 1] = 1.0
            a[:, 2] = -1.0
            q = []
            for _ in range(4):
                env.step(a)
                q.append(env.ws.qpos[:, :5].clone())
            spreads[stage] = float((torch.stack(q).std(dim=1)).max())
            if stage == 4:
                from so101_rl.env.obs import PRIV_OBS

                st = env.step(a)
                ratios = st.priv[:, PRIV_OBS.dr_params][:, :6]  # kp / centre
                np.testing.assert_allclose(ratios.cpu().numpy(), env.randomizer.group_scale.cpu().numpy(), rtol=1e-5)
                assert (st.info["dr_stage"] == 4).all()
        finally:
            env.close()
    assert spreads[0] < 1e-5, spreads
    assert spreads[4] > 1e-3, spreads


@gpu
def test_proprio_noise_only_in_the_policy_obs():
    from so101_rl.config.schema import CENTER_TICK
    from so101_rl.env.obs import REACH_OBS

    env = _env(4, n=256)
    try:
        obs = env.reset(phase_randomise=False)[0]
        true = env.q_to_ticks(env.ws.qpos[:, env._q_adr])
        seen = torch.round(obs[:, REACH_OBS.q_norm].double() * 2048 + CENTER_TICK).to(torch.int64)
        diff = (seen - true).abs()
        assert int(diff.max()) <= REACH_DR_STAGES[4].proprio_noise_ticks
        assert int(diff.max()) > 0
        # achieved (reward / success) uses the un-noised reading
        tip_true = env.chain.fk_tip(true, torch.zeros(256, 3, dtype=torch.float64, device=env.device)).float()
        torch.testing.assert_close(env._achieved, tip_true)
    finally:
        env.close()
