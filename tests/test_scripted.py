"""Scripted die controller (D2b, ``rl/scripted.py``) on the CPU: IK, action range and joint limits, the state machine,
the numpy single-env path, one real CPU-MuJoCo die episode per start stage, and the demo collector's transition
format. No GPU: the batched env runs on ``sim/cpu_sim.py``."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_rl.config import load_calibration, load_config
from so101_rl.env.tasks import DIE_TO_ZONE
from so101_rl.rl.replay_gpu import GpuReplay
from so101_rl.rl.scripted import (
    GRIPPER,
    Phase,
    ScriptedDieController,
    collect_demos,
    die_yaw,
    dls_ik,
    env_state,
    run_scripted_episodes,
)

ROOT = Path(__file__).parent.parent
CAL = ROOT / "tests/data/so101_follower.calib.json"
HOME = [2047, 960, 2700, 2748, 2047, 2047]


@pytest.fixture(scope="module", autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def cfg_cal():
    cal = load_calibration(CAL)
    return load_config(calibration=cal), cal


@pytest.fixture
def ctl(cfg_cal):
    cfg, cal = cfg_cal
    return ScriptedDieController.from_calibration(cfg, cal, 1)


@pytest.fixture(scope="module")
def cpu_env():
    from so101_rl.rl import eval_sim

    env = eval_sim.build_env("die_to_zone", 2, stage=0, seed=3, device="cpu", calibration=str(CAL),
                             sim_params=str(ROOT / "calib/sim_params.yaml"))  # fmt: skip
    yield env
    env.close()


def _t(x, dt=torch.int64):
    return torch.as_tensor(np.asarray(x), dtype=dt).reshape(1, -1)


# ------------------------------------------------------------------------------------------------ IK
@pytest.mark.parametrize("xyz_above_table", [(0.139, -0.041, 0.0145), (0.20, -0.05, 0.0145), (0.138, -0.142, 0.017),
                                             (0.215, -0.10, 0.040), (0.20, 0.0, 0.040)])  # fmt: skip
def test_dls_ik_from_home_reaches_the_grasp_point_pointing_down(ctl, xyz_above_table):
    x, y, h = xyz_above_table
    sp = torch.tensor([[x, y, ctl.p.table_z_m + h]], dtype=torch.float64)
    q0 = ctl._ticks_to_rad(torch.tensor([HOME[:5]]))
    zero = torch.zeros(1, dtype=torch.float64)
    _, r_before = dls_ik(ctl._chain_T, q0, sp, g_local=ctl._g_local, yaw_target=zero, w_yaw=zero, q_lo=ctl._q_lo,
                         q_hi=ctl._q_hi, iters=0)  # fmt: skip
    q, r = dls_ik(ctl._chain_T, q0, sp, g_local=ctl._g_local, yaw_target=zero, w_yaw=zero, q_lo=ctl._q_lo,
                  q_hi=ctl._q_hi, iters=10)  # fmt: skip
    assert r[0, :3].norm() < 0.002 < r_before[0, :3].norm()  # position error falls from > 10 cm to < 2 mm
    assert ((q >= ctl._q_lo - 1e-9) & (q <= ctl._q_hi + 1e-9)).all()
    T = __import__("so101_rl.rl.scripted", fromlist=["gripper_fk"]).gripper_fk(ctl._chain_T, q)
    assert T[0, 2, 2] > math.cos(math.radians(8))  # the jaws point (nearly) straight down


def test_one_dls_iteration_reduces_the_error(ctl):
    sp = torch.tensor([[0.18, -0.06, ctl.p.table_z_m + 0.03]], dtype=torch.float64)
    q = ctl._ticks_to_rad(torch.tensor([[2200, 1700, 2500, 2800, 2047]]))
    zero = torch.zeros(1, dtype=torch.float64)
    errs = []
    for _ in range(4):
        q, r = dls_ik(ctl._chain_T, q, sp, g_local=ctl._g_local, yaw_target=zero, w_yaw=zero, q_lo=ctl._q_lo,
                      q_hi=ctl._q_hi, iters=1)  # fmt: skip
        errs.append(float(r[0].norm()))
    assert all(b < a for a, b in itertools.pairwise(errs)), errs


def test_die_yaw_is_the_most_horizontal_axis():
    for yaw in (0.0, 0.3, -1.2, 2.5):
        q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], dtype=torch.float64)
        got = float(die_yaw(q)[0])
        assert abs(((got - yaw + math.pi / 4) % (math.pi / 2)) - math.pi / 4) < 1e-9  # equal modulo 90 deg


# ------------------------------------------------------------------------------------------------ action contract
def test_actions_stay_in_range_and_commands_inside_the_joint_limits(cfg_cal):
    cfg, cal = cfg_cal
    n = 64
    c = ScriptedDieController.from_calibration(cfg, cal, n)
    g = torch.Generator().manual_seed(0)
    lo, hi = c.tick_lo, c.tick_hi
    goal = lo + (torch.rand(n, 6, generator=g) * (hi - lo)).long()
    die = torch.stack([0.12 + 0.12 * torch.rand(n, generator=g), -0.15 + 0.2 * torch.rand(n, generator=g),
                       torch.full((n,), c.p.table_z_m + 0.008)], 1).double()  # fmt: skip
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(n, 4)
    md = torch.as_tensor(np.asarray(cfg.action.max_delta_ticks, np.float64))
    for _ in range(12):  # an ideal servo: present = goal, the goal moves by the scaled action
        c.phase = torch.randint(0, len(Phase), (n,), generator=g)
        a = c.act(goal, goal, die, quat)
        assert a.dtype == torch.float32 and a.shape == (n, 6) and torch.isfinite(a).all()
        assert (a.abs() <= 1.0).all()
        delta = torch.round(a.double() * md).long()
        new = goal + delta
        # the arm command stays inside the joint limits (less the margin); the gripper inside cfg's limits
        assert ((new[:, :5] >= lo[:5]) & (new[:, :5] <= hi[:5])).all()
        assert ((new[:, 5] >= int(cfg.joints.min_ticks[5])) & (new[:, 5] <= int(cfg.joints.max_ticks[5]))).all()
        goal = new


def test_arm_delta_is_scaled_uniformly(ctl):
    """Far from the target every arm joint moves in proportion (the largest at max_delta_ticks): direction kept."""
    a = ctl.act(_t(HOME), _t(HOME), _t([0.20, -0.05, 0.0035], torch.float64), _t([1.0, 0, 0, 0], torch.float64))
    arm = a[0, :5].abs()
    assert float(arm.max()) == pytest.approx(1.0)
    assert int((arm > 0.05).sum()) >= 3


def test_act_np_equals_the_batched_call(cfg_cal):
    cfg, cal = cfg_cal
    a_np = ScriptedDieController.from_calibration(cfg, cal, 1).act_np(np.array(HOME), np.array(HOME),
                                                                       np.array([0.14, -0.04, 0.0035]), np.array([1.0, 0, 0, 0]))  # fmt: skip
    a_t = ScriptedDieController.from_calibration(cfg, cal, 1).act(_t(HOME), _t(HOME), _t([0.14, -0.04, 0.0035], torch.float64),
                                                                  _t([1.0, 0, 0, 0], torch.float64))  # fmt: skip
    assert a_np.dtype == np.float32 and a_np.shape == (6,)
    np.testing.assert_array_equal(a_np, a_t[0].numpy())
    with pytest.raises(ValueError):
        ScriptedDieController.from_calibration(cfg, cal, 2).act_np(np.array(HOME), np.array(HOME), np.zeros(3), np.array([1.0, 0, 0, 0]))


# ------------------------------------------------------------------------------------------------ state machine
def test_open_above_opens_the_gripper_first(ctl):
    a = ctl.act(_t(HOME), _t(HOME), _t([0.14, -0.04, 0.0035], torch.float64), _t([1.0, 0, 0, 0], torch.float64))
    assert int(ctl.phase[0]) == Phase.OPEN_ABOVE and float(a[0, GRIPPER]) == pytest.approx(1.0)


def test_empty_close_goes_back_to_open_above(ctl):
    ctl.phase[:] = int(Phase.CLOSE)
    ctl.phase_steps[:] = ctl.p.close_steps
    t = _t([2240, 1980, 2470, 2710, 2047, ctl.gripper_closed + 5])  # closed on nothing
    ctl.act(t, t, _t([0.20, -0.05, 0.0035], torch.float64), _t([1.0, 0, 0, 0], torch.float64))
    assert int(ctl.phase[0]) == Phase.OPEN_ABOVE


def test_a_lost_die_while_carrying_restarts_the_grasp(ctl):
    ctl.phase[:] = int(Phase.CARRY)
    t = _t([2240, 1900, 2380, 2860, 2047, ctl.gripper_closed + 100])
    far_die = _t([0.30, 0.10, 0.0035], torch.float64)
    ctl.act(t, t, far_die, _t([1.0, 0, 0, 0], torch.float64))
    assert int(ctl.phase[0]) == Phase.OPEN_ABOVE and int(ctl.regrasps[0]) == 1
    ctl.reset()
    assert int(ctl.phase[0]) == Phase.OPEN_ABOVE and int(ctl.regrasps[0]) == 0


def test_done_holds_still(ctl):
    ctl.phase[:] = int(Phase.DONE)
    r = list(ctl.p.retreat_ticks) + [2300]
    a = ctl.act(_t(r), _t(r), _t([0.14, -0.13, 0.0035], torch.float64), _t([1.0, 0, 0, 0], torch.float64))
    assert torch.equal(a, torch.zeros_like(a))  # zero deltas: no STUCK trip while the die settles


# ------------------------------------------------------------------------------------------------ CPU MuJoCo episodes
@pytest.mark.parametrize("stage", [0, 3])
def test_cpu_episode_succeeds(cpu_env, stage):
    """A real CPU MuJoCo die episode (task predicate: visible, in the zone, settled) succeeds at S0 (open table) and
    S3 (the well) for a fixed seed, passing through the whole grasp sequence without a trip."""
    from so101_rl.rl.scripted import ScriptedDiePolicy

    cpu_env.set_start_stage(stage)
    pol = ScriptedDiePolicy(cpu_env)
    cpu_env.reset(phase_randomise=False)
    pol.reset()
    seen: list[set[int]] = [set() for _ in range(cpu_env.nworld)]
    done = torch.zeros(cpu_env.nworld, dtype=torch.bool)
    success = torch.zeros_like(done)
    for _ in range(DIE_TO_ZONE.episode_steps):
        a = pol(cpu_env)
        for w in range(cpu_env.nworld):
            seen[w].add(int(pol.ctl.phase[w]))
        st = cpu_env.step(a)
        end = ~done & (st.terminated | st.truncated)
        assert not bool((end & st.info["tripped"]).any())
        success |= end & st.info["success"]
        done |= end
        if bool(done.all()):
            break
    assert bool(success.all()), success
    for s in seen:
        assert {Phase.OPEN_ABOVE, Phase.DESCEND, Phase.CLOSE, Phase.LIFT, Phase.CARRY, Phase.LOWER} <= s


def test_env_state_reads_the_privileged_state(cpu_env):
    cpu_env.reset(phase_randomise=False)
    goal, present, die, quat = env_state(cpu_env)
    assert goal.shape == present.shape == (cpu_env.nworld, 6) and die.shape == (cpu_env.nworld, 3)
    assert torch.equal(goal, present)  # right after a reset the chain is resynced to the present ticks
    assert torch.allclose(quat.norm(dim=1), torch.ones(cpu_env.nworld), atol=1e-5)


def test_run_scripted_episodes_reports(cpu_env):
    cpu_env.set_start_stage(0)
    st = run_scripted_episodes(cpu_env, 2)
    assert st["episodes"] == 2 and st["successes"] == 2 and st["trip_episodes"] == 0
    assert sum(st["success_phase"].values()) == 2 and st["fail_end_phase"] == {}


def test_collect_demos_writes_the_training_transition_format(cpu_env):
    """Kept episodes are complete and successful: the rows of a world's episode are consecutive transitions
    (next_obs[t] == obs[t+1]) whose last row is terminated by the success; the policy input is 29 wide."""
    cpu_env.set_start_stage(0)
    d_in = cpu_env.obs_dim + 3
    rb = GpuReplay(1000, d_in, cpu_env.priv_dim, 6, device="cpu", gamma=0.98)
    st = collect_demos(cpu_env, rb, 2)
    assert st["episodes"] >= 2 and st["kept"] == st["successes"] >= 2 and rb.size == st["transitions"] > 0
    assert d_in == 29
    obs, nxt, term, act = rb._obs[: rb.size], rb._next_obs[: rb.size], rb._term[: rb.size], rb._act[: rb.size]
    assert int(term.sum()) == st["kept"]  # one terminal (success) row per kept episode
    assert (act.abs() <= 1).all() and (rb._rew[: rb.size] <= 0).all()
    assert torch.allclose(obs[:, -3:], torch.tensor(DIE_TO_ZONE.zone.goal_xyz(), dtype=torch.float32).expand(rb.size, 3))
    # episodes are stored world by world in time order: inside an episode next_obs[t] == obs[t+1]
    ends = torch.nonzero(term).flatten().tolist()
    start = 0
    for e in ends:
        if e > start:
            assert torch.allclose(nxt[start:e], obs[start + 1 : e + 1])
        start = e + 1
