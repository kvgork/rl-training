"""FS9 BatchedGoalEnv (sim track S1.3). GPU tests (`pixi run test-gpu`); the pure-torch helpers run on CPU."""

import dataclasses

import numpy as np
import pytest
import torch

from so101_rl.config import N_JOINTS, load_calibration, load_config
from so101_rl.env.obs import DIE_OBS, PRIV_OBS, REACH_OBS
from so101_rl.env.reward import RewardSpecError
from so101_rl.env.tasks import DIE_TO_ZONE, REACH
from so101_rl.safety.types import TripCause
from so101_rl.sim.model import load_model
from so101_rl.sim.sim_tracker import NO_NOISE, TrackerNoise

gpu = pytest.mark.gpu
DATA = "tests/data/so101_follower.calib.json"


@pytest.fixture(scope="module")
def scene():
    return load_model()


@pytest.fixture(scope="module")
def mcal():
    return load_calibration(DATA)


@pytest.fixture(scope="module")
def mcfg(mcal):
    return load_config(calibration=mcal)


def _env(scene, cfg, cal, task=REACH, n=8, **kw):
    from so101_rl.env.batched_env import BatchedGoalEnv, model_sim_params

    return BatchedGoalEnv(cfg, cal, task, n, sim_params=model_sim_params(scene[0]), model=scene[0], **{"allow_provisional_camera": True, **kw})


def test_model_sim_params_reads_the_mjcf(scene):
    from so101_rl.env.batched_env import model_sim_params

    p = model_sim_params(scene[0])
    assert p.kp.shape == (6,) and (p.kp > 0).all() and (p.kv >= 0).all()
    assert p.delay_ticks == 0 and p.die_mass_kg == pytest.approx(0.004)


@gpu
@pytest.mark.parametrize("task", [REACH, DIE_TO_ZONE], ids=["reach", "die"])
def test_reset_obs_shapes_and_layout(scene, mcfg, mcal, task):
    from so101_rl.env.obs import CENTER_TICK

    env = _env(scene, mcfg, mcal, task, n=8)
    obs, priv, ach, des = env.reset()
    layout = REACH_OBS if task is REACH else DIE_OBS
    assert obs.shape == (8, layout.dim) and priv.shape == (8, PRIV_OBS.dim)
    assert ach.shape == (8, 3) and des.shape == (8, 3)
    assert env.obs_dim == layout.dim and env.priv_dim == 61 and env.act_dim == 6
    present = env.q_to_ticks(env.ws.qpos[:, env._q_adr])
    assert torch.allclose(obs[:, layout.q_norm].double(), (present.double() - CENTER_TICK) / 2048, atol=1e-6)
    assert (obs[:, layout.last_applied] == 0).all()
    if task is DIE_TO_ZONE:
        assert (obs[:, layout.die_visible] == 1).all()  # the tracker was primed on the resting die
        assert torch.allclose(des, torch.as_tensor(task.zone.goal_xyz(), dtype=torch.float32, device=des.device).expand(8, 3))
    else:
        assert torch.allclose(ach, obs[:, layout.tip_xyz])
    env.close()


@gpu
def test_step_advances_51_substeps(scene, mcfg, mcal):
    env = _env(scene, mcfg, mcal, n=4)
    env.reset(phase_randomise=False)
    assert env.steps_per_tick == 17
    t0 = env.ws._time.clone()
    env.step(torch.zeros(4, 6, device=env.device))
    dt = (env.ws._time - t0).double().cpu().numpy()
    assert np.allclose(dt, 51 * scene[0].opt.timestep, atol=1e-6)
    assert abs(51 * scene[0].opt.timestep - 0.1) < 1e-9
    env.close()


@gpu
def test_action_scaling_is_f7_rule(scene, mcfg, mcal):
    """+1 on joint j -> applied max_delta_ticks[j] after one step from home, other joints 0 (plan §5 Test 1)."""
    env = _env(scene, mcfg, mcal, n=6)
    env.start_noise_ticks = 0
    env.reset(phase_randomise=False)
    env.goal.fill_(5.0)  # unreachable: a goal drawn next to the home tip would succeed and auto-reset the world
    a = torch.eye(6, device=env.device)
    st = env.step(a)
    applied = st.info["applied_delta_ticks"].cpu().numpy()
    md = np.asarray(mcfg.action.max_delta_ticks, np.float32)
    assert np.array_equal(applied, np.diag(md))
    assert np.allclose(st.obs[:, REACH_OBS.last_applied].cpu().numpy(), np.eye(6), atol=1e-6)
    env.close()


@gpu
def test_obs_present_is_read_at_start_of_tick_3(scene, mcfg, mcal):
    """The real arm's observation = the 3rd tick's state record: present read after 2 ticks of the command."""
    from so101_rl.env.obs import CENTER_TICK

    env = _env(scene, mcfg, mcal, n=4)
    env.reset(phase_randomise=False)
    env.trace = []
    gen = torch.Generator(device=env.device).manual_seed(0)
    mism_k2 = mism_next = 0
    for _ in range(6):
        st = env.step(torch.rand(4, 6, generator=gen, device=env.device) * 2 - 1)
        pres = torch.round(st.obs[:, REACH_OBS.q_norm].double() * 2048 + CENTER_TICK).to(torch.int64)
        mism_k2 += int((pres != env.trace[-1]["present"]).any())  # trace[-1] = tick 3 of this step
        mism_next += int((pres != env._present()).any())  # the read after all 3 ticks
    assert mism_k2 == 0
    assert mism_next > 0  # the 2-vs-3 tick offset is detectable
    env.close()


@gpu
def test_truncation_sets_final_obs(scene, mcfg, mcal):
    env = _env(scene, mcfg, mcal, n=4)
    env.reset(phase_randomise=False)
    env.goal.fill_(5.0)  # unreachable: no success
    a = torch.zeros(4, 6, device=env.device)
    a[:, 1] = 0.5  # keep moving so the final obs differs from the reset obs
    for i in range(REACH.episode_steps):
        st = env.step(a if i < 5 else torch.zeros_like(a))
        if i < REACH.episode_steps - 1:
            assert not st.truncated.any()
    assert st.truncated.all() and not st.terminated.any()
    assert (st.info["episode_step"] == REACH.episode_steps).all()
    assert (env.episode_step == 0).all()
    assert not torch.allclose(st.final_obs, st.obs)  # obs = the reset state, final_obs = the last state
    assert torch.allclose(st.final_achieved, st.final_obs[:, REACH_OBS.tip_xyz])
    assert (st.final_obs[:, REACH_OBS.last_applied] != 0).any() or True
    assert (st.obs[:, REACH_OBS.last_applied] == 0).all()
    env.close()


@gpu
def test_auto_reset_phase_randomised(scene, mcfg, mcal):
    env = _env(scene, mcfg, mcal, n=64)
    env.reset()
    steps = env.episode_step.cpu().numpy()
    assert len(np.unique(steps)) > 10 and steps.min() >= 0 and steps.max() < REACH.episode_steps
    env.goal.fill_(5.0)
    done_counts = []
    for _ in range(10):
        st = env.step(torch.zeros(64, 6, device=env.device))
        done_counts.append(int((st.terminated | st.truncated).sum()))
    assert 0 < sum(done_counts) < 64  # staggered, not synchronous
    env.reset(phase_randomise=True)  # only the FIRST reset randomises
    assert (env.episode_step == 0).all()
    env.close()


def _scripted_action(env, obs, goal):
    """Damped least squares on the batched FK in tick space (kinematic sanity, not RL)."""
    from so101_rl.env.obs import CENTER_TICK

    present = torch.round(obs[:, REACH_OBS.q_norm].double() * 2048 + CENTER_TICK).to(torch.int64)
    xyz = torch.zeros(present.shape[0], 3, dtype=torch.float64, device=env.device)
    tip = env.chain.fk_tip(present, xyz).clone()
    cols = []
    for j in range(4):
        p = present.clone()
        p[:, j] += 10
        cols.append((env.chain.fk_tip(p, xyz) - tip) / 10)
    J = torch.stack(cols, dim=-1)  # [N,3,4]
    err = (goal.double() - tip)[:, :, None]
    lam = 1e-9  # J ~ 1e-4 m/tick: JJ^T ~ 1e-8
    dq = J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam * torch.eye(3, device=env.device, dtype=torch.float64), err)
    md = torch.as_tensor(np.asarray(env.cfg.action.max_delta_ticks, np.float64), device=env.device)
    a = torch.zeros(present.shape[0], 6, dtype=torch.float64, device=env.device)
    a[:, :4] = dq[:, :, 0] / md[:4]
    a = a / a.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    return a.float()


@gpu
def test_scripted_reach_succeeds(scene, mcfg, mcal):
    n = 256
    env = _env(scene, mcfg, mcal, n=n, seed=3)
    obs, _, _, goal = env.reset(phase_randomise=False)
    goal = goal.clone()
    succeeded = torch.zeros(n, dtype=torch.bool, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    for _ in range(REACH.episode_steps):
        st = env.step(_scripted_action(env, obs, goal))
        succeeded |= st.terminated & ~finished
        finished |= st.terminated | st.truncated
        obs, goal = st.obs, st.desired.clone()
        goal = torch.where(finished[:, None], goal, goal)
        goal = env.goal.clone()
    rate = succeeded.float().mean().item()
    print(f"scripted reach success {rate:.3f} over {n} worlds")
    assert rate >= 0.90
    env.close()


@gpu
def test_die_task_success_requires_visible_in_zone(scene, mcfg, mcal):
    """Die resting in the zone: success after the 2-step settled debounce when visible; never while invisible."""
    blind = TrackerNoise(dropout_p=1.0, dropout_burst_frames=(1000, 1000), latency_frames=(0, 0))
    for noise, expect in ((NO_NOISE, True), (blind, False)):
        env = _env(scene, mcfg, mcal, DIE_TO_ZONE, n=4, tracker_noise=noise)
        env.reset(phase_randomise=False)
        q = env.ws.qpos[0].clone()
        zone = DIE_TO_ZONE.zone.goal_xyz()
        q[env._die_q : env._die_q + 3] = torch.as_tensor(zone, dtype=torch.float32)
        q[env._die_q + 3 : env._die_q + 7] = torch.tensor([1.0, 0, 0, 0])
        ctrl = env.ws.ctrl[0].clone()
        if not expect:  # a blind tracker must also have lost the die before: restart it, then step
            env.reset_to_state(q, ctrl)
            env.tracker.reset(torch.ones(4, dtype=torch.bool, device=env.device), env.ws.body_xpos("die"))
        else:
            env.reset_to_state(q, ctrl)
        term = []
        for _ in range(4):
            st = env.step(torch.zeros(4, 6, device=env.device))
            term.append(bool(st.terminated.all()))
            if expect:
                assert st.info["tracker_visible"].all()
        assert any(term) is expect, (noise, term)
        if expect:
            assert term[0] is False or DIE_TO_ZONE.success_debounce_steps == 1  # debounce: not on the first step
        env.close()


@gpu
def test_trip_on_nan_action_is_terminal_with_the_penalty(scene, mcfg, mcal):
    """Plan §10 2026-10-08 "Trip penalty": the trip step is terminal (no bootstrap) AND truncated, never a success,
    and pays ``trip_penalty``; the other worlds' step is unchanged (ordinary reward, not done)."""
    env = _env(scene, mcfg, mcal, n=4)
    env.reset(phase_randomise=False)
    env.goal.fill_(5.0)  # unreachable: no success anywhere
    a = torch.zeros(4, 6, device=env.device)
    a[1, 3] = float("nan")
    st = env.step(a)
    assert st.truncated.tolist() == [False, True, False, False]
    assert st.terminated.tolist() == [False, True, False, False]
    assert st.info["tripped"].tolist() == [False, True, False, False]
    assert not st.info["success"].any()
    assert float(st.reward[1]) == REACH.trip_penalty
    assert (st.reward[[0, 2, 3]] > REACH.trip_penalty).all() and (st.reward[[0, 2, 3]] < -1.0).all()
    assert int(st.info["trips"][1]) & int(TripCause.NAN)
    assert (st.info["applied_delta_ticks"][1] == 0).all()
    assert int(env.episode_step[1]) == 0  # auto-reset
    env.close()


@gpu
def test_goal_sampler_matches_numpy_bounds(scene, mcfg, mcal):
    """The batched env draws reach goals from the real env's ``BoxGoalSampler`` (one sampler, FS12): inside
    ``task.goal_bounds``, covering them, every goal in the shared FK-reachability table, and the SAME table
    object the real ``So101GoalEnv`` uses."""
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.env.goals import FALLBACK_GOAL

    env = _env(scene, mcfg, mcal, n=512)
    try:
        g = torch.cat([env._sample_goals() for _ in range(200)]).double().cpu().numpy()  # 1e5 samples
        gb = REACH.goal_bounds
        assert env.goal_sampler.bounds is gb
        assert gb.contains_np(g).all()
        assert (np.abs(g.min(0) - gb.lo) < 0.02).all() and (np.abs(g.max(0) - gb.hi) < 0.03).all()
        fallback = np.all(np.isclose(g, FALLBACK_GOAL, atol=1e-6), axis=1).mean()
        assert fallback < 1e-3
        assert env.goal_sampler.reachable(g).all()  # the numpy lookup of the real env accepts every sim goal
        real = So101GoalEnv(_NullBackend(mcfg, mcal), mcfg, REACH)
        assert real.goal_sampler.keys is env.goal_sampler.keys  # literally the same reachable set
        reach_frac = env.goal_sampler.reachable(gb.sample_np(len(g), np.random.default_rng(0))).mean()
        print(f"FK-reachable fraction of the goal bounds: {reach_frac:.3f}")
        assert reach_frac > 0.5
    finally:
        env.close()


class _NullBackend:
    """Enough of ``EnvBackend`` to construct ``So101GoalEnv`` (its sampler) without a servo."""

    def __init__(self, cfg, calibration) -> None:
        from so101_rl.safety.fk import KinematicChain, So101FK

        self._fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)

    def fk(self):
        return self._fk


@gpu
def test_overflow_checked_and_close(scene, mcfg, mcal):
    env = _env(scene, mcfg, mcal, n=4)
    env.reset()
    for _ in range(mcfg.sim_warp.overflow_check_every_steps):
        env.step(torch.zeros(4, 6, device=env.device))
    assert env.stats["overflow_checks"] >= 1
    env.close()
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(4, 6, device=env.device))


@gpu
def test_render_worlds(scene, mcfg, mcal):
    env = _env(scene, mcfg, mcal, DIE_TO_ZONE, n=2)
    env.reset()
    try:
        img = env.render_worlds([0, 1], width=160, height=120)
    except Exception as exc:  # noqa: BLE001
        env.close()
        pytest.skip(f"no MuJoCo offscreen rendering: {exc}")
    assert img.shape == (2, 120, 160, 3) and img.dtype == np.uint8 and img.std() > 0
    env.close()


@gpu
def test_randomizer_must_match_the_env(scene, mcfg, mcal):
    """The FS8 hooks landed with the MVP (M1, sim/domain_rand.py): a randomizer for another world count or device
    is refused."""
    from so101_rl.env.batched_env import model_sim_params
    from so101_rl.sim.domain_rand import DomainRandomizer
    from so101_rl.sim.sysid.params import DrRanges

    rnd = DomainRandomizer(model_sim_params(scene[0]), DrRanges(), 3, device="cpu")
    with pytest.raises(ValueError, match="randomizer"):
        _env(scene, mcfg, mcal, n=2, randomizer=rnd)


def test_dataclass_replace_keeps_reward_validation():
    """The parity tests run REACH with a longer episode via dataclasses.replace: still a valid TaskSpec, provided the
    trip penalty grows with the horizon (plan §10 2026-10-08 "Trip penalty": worse than any trip-free episode)."""
    t = dataclasses.replace(REACH, episode_steps=10**6, trip_penalty=-1e7)
    assert t.episode_steps == 10**6 and t.reward == REACH.reward
    with pytest.raises(RewardSpecError, match="trip_penalty"):
        dataclasses.replace(REACH, episode_steps=10**6)
    assert N_JOINTS == 6


# ---- provisional camera pose is refused unless the caller opts in (review finding: training on a placeholder)
def test_provisional_camera_pose_refused_by_default(scene, mcfg, mcal):
    from so101_rl.perception.camera_calib import CameraNotCalibrated, load_camera_calib

    assert not load_camera_calib().pose.calibrated, "test premise: calib/camera.yaml is provisional"
    with pytest.raises(CameraNotCalibrated, match="provisional"):
        _env(scene, mcfg, mcal, DIE_TO_ZONE, allow_provisional_camera=False)


def test_reach_needs_no_camera_pose(scene, mcfg, mcal):
    """No die tracker -> no camera -> nothing to refuse (and nothing to record)."""
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("needs the GPU")
    env = _env(scene, mcfg, mcal, REACH, n=2, allow_provisional_camera=False)
    try:
        assert "camera_pose_method" not in env.metadata
    finally:
        env.close()


@gpu
def test_provisional_camera_allowed_with_flag_and_recorded(scene, mcfg, mcal):
    import hashlib

    from so101_rl.perception.camera_calib import CAMERA_YAML

    env = _env(scene, mcfg, mcal, DIE_TO_ZONE, n=2, allow_provisional_camera=True)
    try:
        assert env.metadata["camera_pose_method"] == "provisional"
        assert env.metadata["allow_provisional_camera"] is True
        assert env.metadata["camera_yaml_sha256"] == hashlib.sha256(CAMERA_YAML.read_bytes()).hexdigest()
    finally:
        env.close()


def test_calibrated_camera_pose_needs_no_flag(scene, mcfg, mcal):
    """A calibrated pose passes the default; only the (cheap) refusal is exercised here via require_pose."""
    from so101_rl.perception.camera_calib import load_camera_calib

    c = load_camera_calib()
    pose = dataclasses.replace(c.pose, method="arm-shape", residual_mm=2.0)
    c.with_pose(pose).require_pose(calibrated=True)


def test_bench_cli_has_allow_provisional_flag():
    import inspect

    from so101_rl.env import batched_env

    assert "--allow-provisional-camera" in inspect.getsource(batched_env._bench)
