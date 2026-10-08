"""Modular non-positive reward (env/reward.py; plan §10 2026-10-06 "Reward structure", SU0 review BLOCKING).

The original ``-dist/scale + bonus * predicate`` paid up to +1 per step, so dipping in and out of the zone or
hiding the die every other step out-earned finishing. These tests pin the fix: every step reward is <= -w_time < 0 for any
configuration the validator accepts, unsuccessful steps are <= -(w_success + w_time), and every gaming policy
(including keeping the die moving inside the zone forever) earns strictly less discounted return than a finish.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_rl.config import RewardSpec, RewardTermSpec, load_tasks_config
from so101_rl.config.schema import REWARD_TERM_PARAMS
from so101_rl.env import reward as R
from so101_rl.env.reward import REWARD_TERMS, RewardSpecError, validate_reward_spec
from so101_rl.env.tasks import (
    DIE_TO_ZONE,
    REACH,
    debounce_np,
    reward_np,
    reward_terms_np,
    reward_terms_torch,
    reward_torch,
    success_np,
    task_from_config,
)


def _spec(**weights) -> RewardSpec:
    """A spec from name=weight; ``time`` defaults to the shipped 0.05 (pass ``time=None`` to omit it)."""
    weights = {"time": 0.05, **weights}
    terms = []
    for name, w in weights.items():
        if w is None:
            continue
        if name == "distance":
            terms.append(RewardTermSpec("distance", w, scale_m=0.3, zero_on_success=True))
        else:
            terms.append(RewardTermSpec(name, w))
    return RewardSpec(tuple(terms))


def _with_reward(task, rspec: RewardSpec):
    """``task`` with reward ``rspec`` and a trip penalty that satisfies ``validate_trip_penalty`` for it (heavier
    weights make the worst step worse, so the shipped -310 can be too mild)."""
    bound = task.episode_steps * R.worst_step_reward(rspec, max_dist_m=task.max_dist_m)
    return dataclasses.replace(task, reward=rspec, trip_penalty=min(task.trip_penalty, bound))


# ---------------------------------------------------------------- validator


def test_registry_matches_config_term_table():
    assert set(REWARD_TERMS) == set(REWARD_TERM_PARAMS) >= {"distance", "success", "clamp", "action_l2", "time"}
    assert all(t.sup <= 0.0 for t in REWARD_TERMS.values())


@pytest.mark.parametrize("spec", [REACH, DIE_TO_ZONE], ids=lambda s: s.name)
def test_shipped_rewards_validate(spec):
    b = validate_reward_spec(spec.reward)
    assert b.max_step == -0.05 and b.max_unsuccessful_step == -1.05


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (RewardSpec(()), "no terms"),
        (_spec(distance=1.0), "success"),  # no success term: an unsuccessful step could earn 0
        (_spec(distance=1.0, success=0.0), "success"),
        (_spec(distance=1.0, success=1.0, time=0.0), "time"),  # a tie with finishing: wiggle in the zone forever
        (_spec(distance=1.0, success=1.0, time=None), "time"),
        (_spec(distance=1.0, success=1.0, time=-0.05), "weight"),
        (_spec(distance=1.0, success=-1.0), "weight"),  # the old "+bonus" in disguise
        (_spec(distance=-0.5, success=1.0), "weight"),
        (_spec(success=1.0, clamp=float("nan")), "weight"),
        (_spec(success=1.0, clamp=math.inf), "weight"),
        (RewardSpec((RewardTermSpec("success", 1.0), RewardTermSpec("bonus", 1.0))), "unknown"),
        (RewardSpec((RewardTermSpec("success", 1.0), RewardTermSpec("success", 1.0))), "twice"),
        (RewardSpec((RewardTermSpec("success", 1.0), RewardTermSpec("distance", 1.0, scale_m=None))), "scale_m"),
        (RewardSpec((RewardTermSpec("success", 1.0), RewardTermSpec("distance", 1.0, scale_m=0.0))), "scale_m"),
    ],
)
def test_validator_rejects(spec, match):
    with pytest.raises(RewardSpecError, match=match):
        validate_reward_spec(spec)


def test_validator_rejects_a_term_that_can_be_positive(monkeypatch):
    bonus = R.RewardTerm("success", -1.0, 1.0, REWARD_TERMS["success"].np_fn, REWARD_TERMS["success"].torch_fn)
    monkeypatch.setattr(R, "REWARD_TERMS", {**REWARD_TERMS, "success": bonus})
    with pytest.raises(RewardSpecError, match="positive"):
        R.validate_reward_spec(_spec(success=1.0))


def test_task_spec_construction_runs_the_validator():
    with pytest.raises(RewardSpecError):
        dataclasses.replace(DIE_TO_ZONE, reward=_spec(distance=1.0, success=-1.0))
    with pytest.raises(RewardSpecError):
        dataclasses.replace(REACH, reward=_spec(distance=1.0))
    with pytest.raises(RewardSpecError, match="time"):
        dataclasses.replace(REACH, reward=_spec(distance=1.0, success=1.0, time=0.0))
    with pytest.raises(ValueError, match="settle"):
        dataclasses.replace(REACH, settle_m=0.003)  # settle needs >= 2 debounce steps


def test_module_constants_come_from_the_shipped_config(cfg):
    assert load_tasks_config() == cfg.tasks
    assert task_from_config("reach", cfg.tasks) == REACH
    assert task_from_config("die_to_zone", cfg.tasks) == DIE_TO_ZONE
    with pytest.raises(ValueError, match="unknown task"):
        task_from_config("stack", cfg.tasks)


# ---------------------------------------------------------------- terms and info


def test_terms_sum_to_the_reward_np_and_torch():
    goal = DIE_TO_ZONE.zone.goal_xyz()
    a = np.array([goal, goal + [0.05, 0, 0], goal + [0, 0, 0.03]])
    vis = np.array([True, True, True])
    terms = reward_terms_np(a, np.tile(goal, (3, 1)), DIE_TO_ZONE, visible=vis)
    assert set(terms) == {"distance", "success", "time"}
    np.testing.assert_allclose(sum(terms.values()), reward_np(a, np.tile(goal, (3, 1)), DIE_TO_ZONE, visible=vis),
                               atol=1e-6)
    np.testing.assert_allclose(terms["success"], [0.0, -1.0, -1.0])
    np.testing.assert_allclose(terms["time"], [-0.05] * 3)  # on the success step too
    tt = reward_terms_torch(torch.as_tensor(a), torch.as_tensor(goal).expand(3, 3), DIE_TO_ZONE,
                            visible=torch.as_tensor(vis))
    for k in terms:
        np.testing.assert_allclose(terms[k], tt[k].numpy(), atol=1e-12)


def test_info_terms_read_the_stored_transition_and_never_default():
    spec = _with_reward(REACH, _spec(distance=1.0, success=1.0, clamp=0.5, action_l2=0.2))
    g = np.array([0.2, 0.0, 0.1])
    a = np.tile(g, (2, 1))  # both successful: only the info terms contribute
    infos = [{"action_clamped": True, "applied_action": np.ones(6)},
             {"action_clamped": False, "applied_action": np.zeros(6)}]
    r = reward_np(a, a, spec, info=infos)  # HER: a sequence of the stored infos
    np.testing.assert_allclose(r, [-0.5 - 0.2 - 0.05, -0.05])
    rt = reward_torch(torch.as_tensor(a), torch.as_tensor(a), spec,
                      info={"action_clamped": torch.tensor([True, False]),
                            "applied_action": torch.stack([torch.ones(6), torch.zeros(6)])})
    np.testing.assert_allclose(rt.numpy(), r)
    with pytest.raises(KeyError):
        reward_np(a, a, spec, info=[{"applied_action": np.zeros(6)}] * 2)  # missing flag is not "not clamped"
    with pytest.raises(KeyError):
        reward_np(a, a, spec)
    # a weight-0 term is skipped and needs no info
    zero = _with_reward(REACH, _spec(distance=1.0, success=1.0, clamp=0.0))
    assert reward_np(g, g, zero) == pytest.approx(-0.05)  # only the time term


# ---------------------------------------------------------------- property: never positive, for any valid spec


@st.composite
def valid_specs(draw):
    terms = [RewardTermSpec("success", draw(st.floats(1e-3, 10.0))), RewardTermSpec("time", draw(st.floats(1e-3, 10.0)))]
    if draw(st.booleans()):
        terms.append(RewardTermSpec("distance", draw(st.floats(0.0, 10.0)), scale_m=draw(st.floats(1e-3, 5.0)),
                                    zero_on_success=draw(st.booleans())))
    for name in ("clamp", "action_l2"):
        if draw(st.booleans()):
            terms.append(RewardTermSpec(name, draw(st.floats(0.0, 10.0))))
    order = draw(st.permutations(range(len(terms))))
    return RewardSpec(tuple(terms[i] for i in order))


@settings(max_examples=150, deadline=None)
@given(rspec=valid_specs(), seed=st.integers(0, 2**32 - 1), die=st.booleans())
def test_property_no_step_reward_is_positive(rspec, seed, die):
    """For any spec the validator accepts and any states: reward <= -w_time < 0, and <= -(w_success + w_time) on
    unsuccessful steps (numpy and torch agree)."""
    b = validate_reward_spec(rspec)
    task = _with_reward(DIE_TO_ZONE if die else REACH, rspec)
    rng = np.random.default_rng(seed)
    n = 256
    desired = (np.tile(task.zone.goal_xyz(), (n, 1)) if die else task.goal_bounds.sample_np(n, rng))
    scale = rng.choice([0.0, 0.005, 0.03, 0.3])
    achieved = desired + rng.normal(0, 1, size=(n, 3)) * scale
    achieved[: n // 8] = desired[: n // 8]  # exact hits (HER relabelling)
    visible = rng.random(n) < 0.7 if die else None
    info = {"action_clamped": rng.random(n) < 0.3, "applied_action": rng.uniform(-1, 1, size=(n, 6))}
    r = reward_np(achieved, desired, task, visible=visible, info=info)
    ok = success_np(achieved, desired, task, visible=visible)
    w_time = rspec.term("time").weight
    assert b.max_step == -w_time < 0.0
    assert np.all(r <= -w_time + 1e-6)  # float32 reward
    assert np.all(r < 0.0)
    assert np.all(r[~ok] <= b.max_unsuccessful_step + 1e-6)
    rt = reward_torch(torch.as_tensor(achieved), torch.as_tensor(desired), task,
                      visible=None if visible is None else torch.as_tensor(visible),
                      info={k: torch.as_tensor(v) for k, v in info.items()})
    np.testing.assert_allclose(r, rt.numpy(), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- gaming policies vs finishing


def _episode_return(task, track, visible, gamma, horizon, info=None):
    """Discounted return of a die tracker trajectory under the real reward + settle-debounce termination."""
    goal = task.zone.goal_xyz()
    g, count, prev = 0.0, np.zeros((), int), track[0]
    for t in range(horizon):
        p, v = track[min(t, len(track) - 1)], visible[min(t, len(visible) - 1)]  # hold the last state
        vis = np.bool_(v)
        g += gamma**t * float(reward_np(p, goal, task, visible=vis, info=info))
        done, count = debounce_np(success_np(p, goal, task, visible=vis), count, task, achieved=p, prev_achieved=prev)
        prev = p
        if done:
            return g, True
    return g, False


def _policies(task):
    goal = task.zone.goal_xyz()
    e = task.zone.half_edge_m
    approach = [goal + [0.10 - 0.01 * k, 0, 0] for k in range(10)]  # 10 cm away, 1 cm per step
    inside, outside = goal + [e - 0.002, 0, 0], goal + [e + 0.002, 0, 0]
    tail = 2000
    return {
        "finish": (np.array([*approach, goal, goal]), [True] * 12),
        # dip in and out of the zone forever (each in-zone step paid +1 under the old reward)
        "alternate": (np.array(approach + [inside, outside] * tail), [True] * (10 + 2 * tail)),
        # sit in the zone but hide the die every other step: never 2 consecutive visible steps
        "hide": (np.array(approach + [goal] * 2 * tail), [True] * 10 + [True, False] * tail),
        # slide through the zone (and out the other side) at 5 mm per step, back and forth: never settled
        "slide": (np.array(approach + [goal + [x, 0, 0] for x in np.r_[np.arange(-0.05, 0.05, 0.005),
                                                                         np.arange(0.05, -0.05, -0.005)]] * 100),
                  [True] * (10 + 40 * 100)),
        # keep the die moving 5 mm per step INSIDE the zone forever: in the zone every step, never settled
        "wiggle": (np.array(approach + [goal + [0.0025 * (-1) ** k, 0, 0] for k in range(2 * tail)]),
                   [True] * (10 + 2 * tail)),
    }


@pytest.mark.parametrize("gamma", [0.98, 0.995, 0.999])
def test_finishing_beats_every_gaming_policy(gamma):
    horizon = 3000  # long horizon: gaming would pay off most here under a positive reward
    pol = _policies(DIE_TO_ZONE)
    returns = {}
    for name, (track, vis) in pol.items():
        g, done = _episode_return(DIE_TO_ZONE, track, vis, gamma, horizon)
        returns[name] = g
        assert done == (name == "finish"), name  # none of the gaming policies ever terminates
    # STRICTLY less for every gaming policy, "wiggle" (keep moving inside the zone forever) included: the `time`
    # term is -w_time on in-zone steps too, so there is no tie with finishing.
    for name in ("alternate", "hide", "slide", "wiggle"):
        assert returns["finish"] > returns[name], (name, returns)


@settings(max_examples=40, deadline=None)
@given(rspec=valid_specs())
def test_finishing_beats_gaming_for_any_valid_spec(rspec):
    task = _with_reward(DIE_TO_ZONE, rspec)
    info = {"action_clamped": False, "applied_action": np.zeros(6)}
    pol = _policies(task)
    fin, done = _episode_return(task, *pol["finish"], 0.98, 400, info=info)
    assert done
    for name in ("alternate", "hide", "slide", "wiggle"):
        g, d = _episode_return(task, *pol[name], 0.98, 400, info=info)
        assert not d and fin > g, (name, fin, g)


def test_old_positive_reward_is_gamed_broken_twin():
    """The broken-input twin: the pre-fix reward (-dist/0.3 + 1 * predicate) makes alternating beat finishing.
    It is not expressible as a valid RewardSpec (it needs a negative 'success' weight, which is rejected)."""
    goal = DIE_TO_ZONE.zone.goal_xyz()

    def old_return(track, vis, gamma=0.98, horizon=3000):
        g, count, prev = 0.0, np.zeros((), int), track[0]
        for t in range(horizon):
            p, v = track[t], np.bool_(vis[t])
            ok = success_np(p, goal, DIE_TO_ZONE, visible=v)
            g += gamma**t * (-np.linalg.norm(p - goal) / 0.3 + 1.0 * float(ok))
            done, count = debounce_np(ok, count, DIE_TO_ZONE, achieved=p, prev_achieved=prev)
            prev = p
            if done:
                break
        return g

    pol = _policies(DIE_TO_ZONE)
    assert old_return(*pol["alternate"]) > old_return(*pol["finish"])
    with pytest.raises(RewardSpecError):
        validate_reward_spec(_spec(distance=1.0, success=-1.0))
