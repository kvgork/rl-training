"""``So101GoalEnv`` (FS12; sim plan R1.2; Phase 0 Task 8.2 superseded): Gym/SB3 contracts, the FS3 observation,
the FS4 reward/success via ``env/tasks.py``, trips, stale state and the die block from the object table. All on the
in-process ``SimBackend`` (CPU MuJoCo through the real ``ServoCore``)."""

from __future__ import annotations

import uuid
import warnings

import gymnasium
import numpy as np
import pytest
import torch

import so101_rl.env  # noqa: F401  (registers the gymnasium ids)
from so101_rl.env import make_goal_env
from so101_rl.env.backend import ServoStateStale, SimBackend
from so101_rl.env.goal_env import So101GoalEnv
from so101_rl.env.obs import DieObsT, build_policy_obs_torch
from so101_rl.env.tasks import DIE_TO_ZONE, REACH, reward_np
from so101_rl.perception.object_table import ObjectTableShm, new_object_table, set_object
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo.shm import new_state


@pytest.fixture
def reach_env(cfg, calibration):
    env = So101GoalEnv(SimBackend(cfg, calibration), cfg, REACH)
    yield env
    env.close()


@pytest.fixture
def table():
    t = ObjectTableShm.create(f"so101objtest_{uuid.uuid4().hex[:10]}")
    yield t
    t.close()


@pytest.fixture
def die_env(cfg, calibration, table):
    env = So101GoalEnv(SimBackend(cfg, calibration), cfg, DIE_TO_ZONE, object_table=table)
    yield env
    env.close()


def _publish_die(table, t_ns: int, xyz, *, visible=True, fiducial_ok=True) -> None:
    rec = new_object_table()
    rec["t_mono_ns"] = t_ns
    rec["fiducial_ok"] = int(fiducial_ok)
    set_object(rec, 0, "gripper_tip", np.zeros(3), 0.5, True, t_ns)
    set_object(rec, 1, "die", np.asarray(xyz), 0.9, visible, t_ns)
    table.write(rec)


def _now(env) -> int:
    return int(env.backend.clock.monotonic_ns())


def _next_record_t(env) -> int:
    """Time stamp of the state record the next step will read: the start of its 3rd servo tick."""
    return _now(env) + 2 * env.backend.servo_period_ns


# ---------------------------------------------------------------------------------------------------- contracts


@pytest.mark.parametrize("task", ["reach", "die_to_zone"])
def test_gymnasium_and_sb3_env_checkers_pass(cfg, calibration, task):
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    env = make_goal_env(task, cfg=cfg, calibration=calibration)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*infinity.*")  # Box(-inf, inf): the FS3 layout is unbounded
        check_env(env, skip_render_check=True)
        sb3_check_env(env, warn=True)
    env.close()


def test_registered_ids_build_the_two_tasks(cfg, calibration):
    for env_id, spec in (("So101Reach-v0", REACH), ("So101DieToZone-v0", DIE_TO_ZONE)):
        env = gymnasium.make(env_id, cfg=cfg, calibration=calibration)
        assert env.unwrapped.task is spec
        assert env.observation_space["observation"].shape == (spec.obs.dim,)
        env.close()
    with pytest.raises(ValueError, match="backend"):
        make_goal_env("reach", backend="serial", cfg=cfg, calibration=calibration)
    with pytest.raises(ValueError, match="task"):
        make_goal_env("pick", cfg=cfg, calibration=calibration)


def test_spaces_and_obs_dtypes(reach_env):
    obs, _info = reach_env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (21,) and all(v.dtype == np.float32 for v in obs.values())
    assert reach_env.observation_space.contains(obs)
    assert reach_env.action_space.shape == (6,) and reach_env.action_space.dtype == np.float32
    assert float(reach_env.action_space.low.min()) == -1.0 and float(reach_env.action_space.high.max()) == 1.0


def test_reset_seed_determines_the_goal(reach_env):
    a = reach_env.reset(seed=3)[0]["desired_goal"]
    b = reach_env.reset(seed=3)[0]["desired_goal"]
    c = reach_env.reset(seed=4)[0]["desired_goal"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert REACH.goal_bounds.contains_np(a)


def test_reset_redraws_a_goal_near_the_start_tip_like_the_batched_env(reach_env, monkeypatch):
    """Parity with ``BatchedGoalEnv``: a sampled reach goal within 2 success radii of the start tip is redrawn once."""
    from so101_rl.env.goals import GOAL_MIN_START_FACTOR

    reach_env.reset(seed=0)
    tip0 = reach_env.reset(seed=0)[0]["achieved_goal"].astype(np.float64)
    far = np.array([0.25, 0.05, 0.10])
    draws = iter([tip0 + [0.0, 0.0, 0.5 * GOAL_MIN_START_FACTOR * REACH.success_radius_m], far, far])
    monkeypatch.setattr(reach_env.goal_sampler, "sample", lambda rng: next(draws))
    obs = reach_env.reset(seed=0)[0]
    np.testing.assert_allclose(obs["desired_goal"], far.astype(np.float32))
    assert next(draws) is far  # exactly one redraw
    draws2 = iter([far, tip0])  # a far first draw is kept
    monkeypatch.setattr(reach_env.goal_sampler, "sample", lambda rng: next(draws2))
    np.testing.assert_allclose(reach_env.reset(seed=0)[0]["desired_goal"], far.astype(np.float32))


def test_step_before_reset_raises(reach_env):
    with pytest.raises(RuntimeError, match="reset"):
        reach_env.step(np.zeros(6, np.float32))


# ---------------------------------------------------------------------------------------------------- observation


def test_obs_equals_torch_builder_on_same_state(reach_env, cfg):
    reach_env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs, *_rest, _info = reach_env.step(rng.uniform(-1, 1, 6).astype(np.float32))
        st = new_state()
        assert reach_env.backend.read_state(st)
        rec = st[0]
        tip = reach_env.backend.fk().gripper_xyz_from_ticks(rec["present_ticks"], np.empty(3))
        t = build_policy_obs_torch(
            REACH.obs,
            present_ticks=torch.as_tensor(rec["present_ticks"][None].astype(np.int64)),
            velocity=torch.as_tensor(rec["velocity"][None].astype(np.int64)),
            applied_delta_ticks=torch.as_tensor(rec["applied_delta_ticks"][None]),
            tip_xyz=torch.as_tensor(tip[None]),
            cfg=cfg,
            die=None,
            out=torch.empty(1, 21),
        )
        np.testing.assert_allclose(obs["observation"], t[0].numpy(), atol=1e-6)
        np.testing.assert_allclose(obs["achieved_goal"], tip, atol=1e-6)


def test_die_obs_equals_torch_builder_on_same_state(die_env, cfg, table):
    die_env.reset(seed=0)
    _publish_die(table, _next_record_t(die_env), [0.21, -0.04, -0.012])
    obs, *_r, info = die_env.step(np.zeros(6, np.float32))
    st = new_state()
    die_env.backend.read_state(st)
    rec = st[0]
    tip = die_env.backend.fk().gripper_xyz_from_ticks(rec["present_ticks"], np.empty(3))
    t = build_policy_obs_torch(
        DIE_TO_ZONE.obs,
        present_ticks=torch.as_tensor(rec["present_ticks"][None].astype(np.int64)),
        velocity=torch.as_tensor(rec["velocity"][None].astype(np.int64)),
        applied_delta_ticks=torch.as_tensor(rec["applied_delta_ticks"][None]),
        tip_xyz=torch.as_tensor(tip[None]),
        cfg=cfg,
        die=DieObsT(torch.tensor([[0.21, -0.04, -0.012]]), torch.tensor([True]), torch.tensor([0])),
        out=torch.empty(1, 26),
    )
    np.testing.assert_allclose(obs["observation"], t[0].numpy(), atol=1e-6)
    assert info["tracker_visible"] is True


# ---------------------------------------------------------------------------------------------------- action / info


@pytest.mark.parametrize("j", [0, 3, 5])
def test_action_scaling_is_the_f7_rule(reach_env, cfg, j):
    reach_env.reset(seed=0)
    a = np.zeros(6, np.float32)
    a[j] = 1.0
    _obs, _r, _te, _tr, info = reach_env.step(a)
    expect = np.zeros(6)
    expect[j] = cfg.action.max_delta_ticks[j]
    np.testing.assert_array_equal(info["applied_delta_ticks"], expect)
    np.testing.assert_allclose(info["applied_action"], expect / cfg.action.max_delta_ticks)


def test_info_applied_delta_equals_the_state_record(reach_env):
    reach_env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(10):
        _obs, _r, _te, _tr, info = reach_env.step(rng.uniform(-1, 1, 6).astype(np.float32))
        st = new_state()
        reach_env.backend.read_state(st)
        np.testing.assert_array_equal(info["applied_delta_ticks"], st[0]["applied_delta_ticks"])  # what F8 logs
        np.testing.assert_array_equal(info["target_ticks"], st[0]["target_ticks"])
        np.testing.assert_array_equal(info["clamped_mask"], st[0]["clamped_mask"].astype(bool))
        assert info["state_seq"] == int(st[0]["seq"]) and info["cmd_seq"] == int(st[0]["last_cmd_seq"])
        assert info["action_clamped"] == bool(st[0]["clamped_mask"].any())
        for key in ("is_success", "tracker_visible", "read_ok", "trips", "tick", "applied_action", "sm_state"):
            assert key in info


def test_nan_action_is_a_terminal_trip_with_the_penalty_and_never_moves_the_goal(reach_env, cfg):
    reach_env.reset(seed=0)
    reach_env.step(np.zeros(6, np.float32))
    core, bus = reach_env.backend.core, reach_env.backend.bus
    goal = core.pipeline.goal.copy()
    target = core._target.copy()
    n = len(bus.written_goals)
    _obs, reward, terminated, truncated, info = reach_env.step(np.array([np.nan, 0, 0, 0, 0, 0], np.float32))
    # plan §10 2026-10-08 "Trip penalty": terminal (no bootstrap), never a success, fixed penalty
    assert truncated and terminated and info["tripped"] and not info["is_success"]
    assert reward == REACH.trip_penalty
    assert info["trip"] & TripCause.NAN and info["sm_state"] == SmState.SAFE_STOPPING
    assert np.array_equal(core.pipeline.goal, goal)  # the NaN delta never reached the goal
    assert not info["applied_delta_ticks"].any()
    first = bus.written_goals[n].astype(int)  # the first goal written after the NaN: the safe-stop ramp, not a jump
    assert np.abs(first - target.astype(int)).max() <= cfg.safe_state.ramp_ticks_per_tick
    with pytest.raises(RuntimeError, match="reset"):
        reach_env.step(np.zeros(6, np.float32))
    reach_env.reset(seed=0)  # a fresh servo core (the sim's "fresh process")
    _obs, _r, _te, truncated, info = reach_env.step(np.zeros(6, np.float32))
    assert not truncated and info["trips"] == 0


def test_step_limit_truncates(reach_env):
    reach_env.reset(seed=0, options={"goal": np.array([0.30, 0.15, 0.25])})  # far: no success with zero actions
    for k in range(REACH.episode_steps):
        _obs, _r, terminated, truncated, _info = reach_env.step(np.zeros(6, np.float32))
        assert not terminated
        assert truncated == (k == REACH.episode_steps - 1)


# ---------------------------------------------------------------------------------------------------- reward


def test_compute_reward_is_tasks_reward_np_vectorised_and_matches_step(reach_env):
    reach_env.reset(seed=2)
    rng = np.random.default_rng(2)
    ach, des, infos, rewards = [], [], [], []
    for _ in range(8):
        obs, r, _te, _tr, info = reach_env.step(rng.uniform(-1, 1, 6).astype(np.float32))
        assert r == float(reach_env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        assert r <= -REACH.reward.term("time").weight < 0
        ach.append(obs["achieved_goal"])
        des.append(obs["desired_goal"])
        infos.append(info)
        rewards.append(r)
    batch = reach_env.compute_reward(np.array(ach), np.array(des), np.array(infos, dtype=object))
    assert batch.shape == (8,) and batch.dtype == np.float32
    np.testing.assert_array_equal(batch, np.array(rewards, np.float32))
    np.testing.assert_array_equal(batch, reward_np(np.array(ach), np.array(des), REACH))
    relabelled = reach_env.compute_reward(np.array(ach), np.array(ach), infos)  # HER: goal = achieved
    np.testing.assert_allclose(relabelled, -REACH.reward.term("time").weight, rtol=1e-6)


def test_her_never_relabels_a_trip_as_a_success(reach_env):
    """Plan §10 2026-10-08 "Trip penalty": HER relabels the goal of a stored terminal-trip transition to its own
    achieved goal (where the success predicate holds); compute_reward still pays trip_penalty, never 0 / success."""
    reach_env.reset(seed=0)
    ok_obs, _r, _te, _tr, ok_info = reach_env.step(np.zeros(6, np.float32))
    trip_obs, trip_r, terminated, _tr, trip_info = reach_env.step(np.array([np.nan, 0, 0, 0, 0, 0], np.float32))
    assert terminated and trip_info["tripped"] and trip_r == REACH.trip_penalty
    ach = np.array([ok_obs["achieved_goal"], trip_obs["achieved_goal"]])
    infos = np.array([ok_info, trip_info], dtype=object)
    relabelled = reach_env.compute_reward(ach, ach, infos)  # HER "future"/"final": desired = achieved
    assert relabelled[0] == np.float32(-REACH.reward.term("time").weight)  # the ordinary step becomes a success
    assert relabelled[1] == np.float32(REACH.trip_penalty)  # the trip never does
    assert reach_env.compute_reward(trip_obs["achieved_goal"], trip_obs["achieved_goal"], trip_info) == REACH.trip_penalty
    for bits in (TripCause.RANGE, TripCause.BOX_PRESENT, TripCause.STUCK):  # every terminal trip cause, any goal
        info = {**ok_info, "trips": int(bits)}
        assert reach_env.compute_reward(ach[0], ach[0], info) == REACH.trip_penalty
    for bits in (TripCause.THERMAL, TripCause.STALE_CMD, TripCause.JOINT_LIMIT):  # not policy-caused: no penalty
        info = {**ok_info, "trips": int(bits)}
        assert reach_env.compute_reward(ach[0], ach[0], info) == np.float32(-REACH.reward.term("time").weight)
    with pytest.raises(KeyError):
        reach_env.compute_reward(ach[0], ach[0], {k: v for k, v in ok_info.items() if k != "trips"})  # never "no trip"


def test_die_compute_reward_uses_the_stored_visibility(die_env):
    die_env.reset(seed=0)
    z = DIE_TO_ZONE.zone.goal_xyz().astype(np.float32)
    seen = die_env.compute_reward(z, z, {"tracker_visible": True, "trips": 0})
    unseen = die_env.compute_reward(z, z, {"tracker_visible": False, "trips": 0})
    assert seen > unseen  # an invisible die in the zone is UNKNOWN, never success
    with pytest.raises(KeyError):
        die_env.compute_reward(z, z, {"trips": 0})  # never defaults to visible
    tripped = die_env.compute_reward(z, z, {"tracker_visible": True, "trips": int(TripCause.BOX_PRESENT)})
    assert tripped == DIE_TO_ZONE.trip_penalty  # a die in the zone on a trip step is no success either


# ---------------------------------------------------------------------------------------------------- success


def _jacobian_action(env, obs, goal) -> np.ndarray:
    """A proportional controller in tick space through a finite-difference Jacobian of the FK (kinematic sanity)."""
    st = new_state()
    env.backend.read_state(st)
    q = st[0]["present_ticks"].astype(np.int64)
    fk = env.backend.fk()
    tip = fk.gripper_xyz_from_ticks(q, np.empty(3))
    jac = np.zeros((3, 5))
    for i in range(5):
        dq = q.copy()
        dq[i] += 20
        jac[:, i] = (fk.gripper_xyz_from_ticks(dq, np.empty(3)) - tip) / 20.0
    dticks = np.linalg.lstsq(jac, goal - tip, rcond=None)[0]
    a = np.zeros(6)
    a[:5] = dticks / env.cfg.action.max_delta_ticks[:5]
    return np.clip(a, -1, 1).astype(np.float32)


def test_scripted_move_toward_goal_reaches_success_on_8_of_10_goals(reach_env):
    successes = 0
    for seed in range(10):
        obs, _ = reach_env.reset(seed=100 + seed)
        goal = obs["desired_goal"].astype(np.float64)
        for _ in range(REACH.episode_steps):
            obs, r, terminated, truncated, info = reach_env.step(_jacobian_action(reach_env, obs, goal))
            if terminated and not info["is_success"]:  # a policy-caused trip is terminal too (trip penalty)
                assert info["tripped"]
                break
            if terminated:
                assert np.linalg.norm(obs["achieved_goal"] - goal) <= 0.015 + 1e-6
                assert r == pytest.approx(-REACH.reward.term("time").weight)
                successes += 1
                break
            if truncated:
                break
    assert successes >= 8, successes


class _StaleReads:
    """Backend wrapper: the servo's block read fails (``read_ok = 0``) on every record (present = last good)."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def read_state(self, out):
        ok = self.inner.read_state(out)
        out["read_ok"] = 0
        return ok

    def accepted_state(self, out):
        return self.read_state(out)


def test_a_stale_reading_never_terminates_the_episode(cfg, calibration):
    env = So101GoalEnv(_StaleReads(SimBackend(cfg, calibration)), cfg, REACH)
    obs, _ = env.reset(seed=0)
    here = obs["achieved_goal"].astype(np.float64)
    env.reset(seed=0, options={"goal": here})  # the goal is where the tip already is: the predicate holds
    for _ in range(3):
        _obs, _r, terminated, _tr, info = env.step(np.zeros(6, np.float32))
        assert not info["read_ok"] and not terminated and not info["is_success"]
        assert info["tracker_visible"] is False
    good = So101GoalEnv(SimBackend(cfg, calibration), cfg, REACH)  # the same situation with good reads ends at once
    good.reset(seed=0, options={"goal": here})
    _obs, _r, terminated, _tr, info = good.step(np.zeros(6, np.float32))
    assert terminated and info["is_success"]


class _Stalls:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.stall = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def wait_policy_tick(self):
        if self.stall:
            raise ServoStateStale("test: servo stalled")
        self.inner.wait_policy_tick()


def test_a_stalled_servo_truncates_and_needs_a_reset(cfg, calibration):
    b = _Stalls(SimBackend(cfg, calibration))
    env = So101GoalEnv(b, cfg, REACH)
    env.reset(seed=0)
    b.stall = True
    _obs, _r, terminated, truncated, info = env.step(np.zeros(6, np.float32))
    assert truncated and not terminated and info["servo_stalled"] and not info["read_ok"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(6, np.float32))


# ---------------------------------------------------------------------------------------------------- die block


def test_die_never_seen_without_an_object_table(cfg, calibration):
    env = So101GoalEnv(SimBackend(cfg, calibration), cfg, DIE_TO_ZONE)
    obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(obs["observation"][21:26], [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(obs["achieved_goal"], np.zeros(3))
    np.testing.assert_allclose(obs["desired_goal"], DIE_TO_ZONE.zone.goal_xyz())
    _obs, _r, terminated, _tr, info = env.step(np.zeros(6, np.float32))
    assert info["tracker_visible"] is False and not terminated


def test_die_visible_held_and_aged_from_the_object_table(die_env, table):
    die_env.reset(seed=0)
    p = np.array([0.20, -0.05, -0.012])
    _publish_die(table, _next_record_t(die_env), p)
    obs, *_r, info = die_env.step(np.zeros(6, np.float32))
    np.testing.assert_allclose(obs["observation"][21:26], [*p, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs["achieved_goal"], p, atol=1e-6)
    _publish_die(table, _next_record_t(die_env), [0.3, 0.3, 0.3], visible=False)  # occluded
    obs, *_r, info = die_env.step(np.zeros(6, np.float32))
    assert info["tracker_visible"] is False
    np.testing.assert_allclose(obs["achieved_goal"], p, atol=1e-6)  # held last-known position
    assert obs["observation"][24] == 0.0 and obs["observation"][25] == pytest.approx(3 / 30)  # 3 frames old
    obs, *_r, info = die_env.step(np.zeros(6, np.float32))  # no new table record: the old one is stale
    assert info["tracker_visible"] is False and obs["observation"][25] == pytest.approx(6 / 30)


def test_die_stale_table_or_moved_camera_is_invisible(die_env, table):
    die_env.reset(seed=0)
    _publish_die(table, _now(die_env) - 1_000_000_000, [0.2, -0.05, -0.012])  # a 1 s old detection
    _obs, *_r, info = die_env.step(np.zeros(6, np.float32))
    assert info["tracker_visible"] is False
    _publish_die(table, _next_record_t(die_env), [0.2, -0.05, -0.012], fiducial_ok=False)
    _obs, *_r, info = die_env.step(np.zeros(6, np.float32))
    assert info["tracker_visible"] is False and info["fiducial_ok"] is False


def test_die_success_needs_two_settled_visible_steps_in_the_zone(die_env, table):
    die_env.reset(seed=0)
    z = DIE_TO_ZONE.zone.goal_xyz()
    _publish_die(table, _next_record_t(die_env), z + [0.005, 0, 0])
    *_x, terminated, _tr, info = die_env.step(np.zeros(6, np.float32))
    assert not terminated  # first in-zone step: debounce 1/2
    _publish_die(table, _next_record_t(die_env), z + [-0.005, 0, 0])  # moved 10 mm: not settled, restart
    *_x, terminated, _tr, info = die_env.step(np.zeros(6, np.float32))
    assert not terminated
    _publish_die(table, _next_record_t(die_env), z + [-0.004, 0, 0])  # moved 1 mm: settled, 2nd step
    _obs, reward, terminated, _tr, info = die_env.step(np.zeros(6, np.float32))
    assert terminated and info["is_success"]
    assert reward == pytest.approx(-DIE_TO_ZONE.reward.term("time").weight)


def test_die_invisible_in_the_zone_never_succeeds(die_env, table):
    die_env.reset(seed=0)
    z = DIE_TO_ZONE.zone.goal_xyz()
    _publish_die(table, _next_record_t(die_env), z)
    die_env.step(np.zeros(6, np.float32))
    for _ in range(3):
        _publish_die(table, _next_record_t(die_env), z, visible=False)
        _obs, r, terminated, _tr, info = die_env.step(np.zeros(6, np.float32))
        assert not terminated and not info["is_success"] and r < -1.0
