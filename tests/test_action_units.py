"""Action-unit tests 1-3 (plan Task 3.4): the SAME ``ServoCore.tick()`` drives FakeBus and MujocoBus.

Both backends boot the way the real arm does: limp at the rest pose, MCU latched until the button, ARM,
re-energise (ramp to home), RUN. Deviations from the plan text, both forced by as-built safety behaviour:

* The unit action is sent from **home**, not from rest: the measured rest pose is below the box floor, so RUN
  from rest would trip BOX_PRESENT on the first tick (ARCHITECTURE section 5).
* Goal sequences are compared **relative to the goal before the command**: on entering RUN the pipeline resyncs
  to the present pose, which differs between the two plants by a few ticks after the homing ramp.
* MujocoBus obstacle (Test 2): a programmatic hard stop on the joint (its MuJoCo joint range is narrowed to
  30 ticks above the present pose) instead of a wall geom; it blocks every joint the same way, gripper included.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, JointLimits
from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo.core import build_servo_core
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.heartbeat import FakeHeartbeatLink, FakeMcu
from so101_rl.servo.shm import CmdMode, LocalIpc, new_cmd, new_state
from so101_rl.servo.timing import FakeClock
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

NS = 1_000_000_000
BACKENDS = ("fake", "mujoco")


class SimClock:
    """MuJoCo-time clock: ``sleep_until`` steps the physics (the real 30 Hz loop's role)."""

    def __init__(self, bus: MujocoBus) -> None:
        self.bus = bus
        self.period_ns = round(bus.steps_per_tick * bus.model.opt.timestep * NS)
        self._offset = 1_000 * NS

    def monotonic_ns(self) -> int:
        return round(self.bus.data.time * NS) + self._offset

    def sleep_until(self, t_ns: int) -> None:
        n = -(-(t_ns - self.monotonic_ns()) // self.period_ns)
        if n > 0:
            self.bus.step_ticks(n)


class Rig:
    """One booted ServoCore on one backend, plus a policy client on LocalIpc."""

    def __init__(self, kind: str, cfg, calibration) -> None:
        self.kind, self.cfg = kind, cfg
        rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
        if kind == "fake":
            self.clock = FakeClock(start_ns=1_000 * NS)
            self.bus = FakeBus(self.clock, calibration=calibration, initial_ticks=rest)
        else:
            model, data = load_model()
            self.bus = MujocoBus(model, data, calibration, cfg)
            self.bus.connect()
            self.bus.reset_to_keyframe("rest")
            self.bus.set_torque(False)  # limp at rest, like the arm after power-up
            self.clock = SimClock(self.bus)
        self.mcu = FakeMcu(self.clock)
        self.ipc = LocalIpc()
        self.core = build_servo_core(
            cfg, calibration, self.bus, self.ipc, FakeHeartbeatLink(self.mcu), self.clock, log=lambda m: None
        )
        self.period = self.core.period_ns
        self.cmd = new_cmd()
        self.st = new_state()
        self.mcu.set_button(True)
        self.core.start(arm_timeout_s=5.0)
        self.mcu.set_button(False)
        assert self.core.state is SmState.RUN
        self.bus.written_goals.clear()

    def send(self, delta, mode=CmdMode.RUN) -> None:
        self.ipc.read_state(self.st)
        self.cmd["state_seq"] = self.st["seq"][0]
        self.cmd["t_mono_ns"] = self.clock.monotonic_ns()
        self.cmd["mode"] = int(mode)
        self.cmd["delta_ticks"][0] = np.asarray(delta, dtype=np.float32)
        self.ipc.write_cmd(self.cmd)

    def tick(self):
        t = self.clock.monotonic_ns()
        res = self.core.tick()
        self.clock.sleep_until(t + self.period)
        return res

    def state(self):
        self.ipc.read_state(self.st)
        return self.st[0]

    def settle(self, steps: int = 3) -> None:
        for _ in range(steps):
            self.send(np.zeros(6))
            for _ in range(3):
                self.tick()


def unit(j: int, value: float = 1.0) -> np.ndarray:
    a = np.zeros(6, dtype=np.float32)
    a[j] = value
    return a


@pytest.fixture(params=BACKENDS)
def kind(request):
    return request.param


# ---------------------------------------------------------------------------------------------------------------
# Test 1: unit action = exactly max_delta_ticks[j], in integer ticks, other joints untouched


def _unit_action_goals(kind, cfg, calibration, j):
    rig = Rig(kind, cfg, calibration)
    rig.settle()
    before = rig.bus.written_goals[-1].astype(np.int64)
    d = cfg.action.max_delta_ticks[j]
    rig.send(unit(j) * d)  # the env's scaling: delta_ticks = round(+1 * max_delta_ticks)
    for _ in range(3):
        assert rig.tick().wrote
    goals = np.array(rig.bus.written_goals[-3:], dtype=np.int64)
    return before, goals, rig


@pytest.mark.parametrize("j", range(6), ids=JOINT_NAMES)
def test1_unit_action_moves_exactly_max_delta_ticks(kind, cfg, calibration, j):
    before, goals, rig = _unit_action_goals(kind, cfg, calibration, j)
    assert goals.dtype.kind == "i"
    assert goals[-1][j] - before[j] == int(cfg.action.max_delta_ticks[j])
    others = [k for k in range(6) if k != j]
    assert (goals[:, others] == before[others]).all()
    assert float(rig.state()["applied_delta_ticks"][j]) == float(cfg.action.max_delta_ticks[j])


@pytest.mark.parametrize("j", range(6), ids=JOINT_NAMES)
def test1_both_backends_produce_identical_goal_sequences(cfg, calibration, j):
    seqs = {}
    for kind in BACKENDS:
        before, goals, _ = _unit_action_goals(kind, cfg, calibration, j)
        seqs[kind] = goals - before
    np.testing.assert_array_equal(seqs["fake"], seqs["mujoco"])


# ---------------------------------------------------------------------------------------------------------------
# Test 2: a blocked joint never winds the target up


def _block(rig: Rig, j: int, ticks_ahead: int) -> int:
    present = rig.core.block.position.astype(np.int32).copy()
    at = int(present[j]) + ticks_ahead
    if rig.kind == "fake":
        rig.bus.block(j, at)
    else:
        m = rig.bus.model
        jid = m.joint(JOINT_NAMES[j]).id
        q_now = rig.bus.ticks_to_q(present)[j]
        stop = present.copy()
        stop[j] = at
        q_at = rig.bus.ticks_to_q(stop)[j]
        lo, hi = m.jnt_range[jid]
        m.jnt_range[jid] = (lo, q_at) if q_at > q_now else (q_at, hi)
        m.jnt_limited[jid] = 1
    return at


def _unblock(rig: Rig, j: int, saved) -> None:
    if rig.kind == "fake":
        rig.bus.unblock(j)
    else:
        jid = rig.bus.model.joint(JOINT_NAMES[j]).id
        rig.bus.model.jnt_range[jid] = saved


def _check_tick(rig: Rig, j: int, E: int) -> bool:
    """One tick; asserts |written target - present| <= E and, in RUN, stored goal <= present + E. True if checked."""
    res = rig.tick()
    checked = res.wrote and res.state is not SmState.LATCHED
    if checked:
        tgt = rig.core._target  # what was written this tick
        assert abs(int(tgt[j]) - int(rig.core.block.position[j])) <= E
    if rig.core.state is SmState.RUN:
        assert int(rig.core.pipeline.goal[j]) <= int(rig.core.block.position[j]) + E  # no wind-up
    return checked


@pytest.mark.parametrize("j", range(6), ids=JOINT_NAMES)
def test2_blocked_joint_tracking_error_bounded_for_300_ticks(kind, cfg, calibration, j):
    rig = Rig(kind, cfg, calibration)
    rig.settle()
    E = int(cfg.action.tracking_error_ticks[j])
    at = _block(rig, j, 30)
    d = unit(j) * cfg.action.max_delta_ticks[j]
    checked = 0
    run_ticks = 0
    for k in range(300):
        if k % 3 == 0:
            rig.send(d)
        checked += _check_tick(rig, j, E)
        if rig.core.state is SmState.RUN:
            run_ticks += 1
            assert int(rig.core.block.position[j]) <= at + (0 if kind == "fake" else 8)  # MuJoCo: soft limit
    assert checked >= 290  # every tick wrote and was checked (RUN, then the stuck-action safe stop)
    assert run_ticks >= 60  # at least 2 s pushing against the stop in RUN


@pytest.mark.parametrize("j", range(6), ids=JOINT_NAMES)
def test2_removing_the_obstacle_moves_the_arm_at_most_E_further(kind, cfg, calibration, j):
    rig = Rig(kind, cfg, calibration)
    rig.settle()
    E = int(cfg.action.tracking_error_ticks[j])
    saved = None
    if kind == "mujoco":
        saved = rig.bus.model.jnt_range[rig.bus.model.joint(JOINT_NAMES[j]).id].copy()
    at = _block(rig, j, 30)
    d = unit(j) * cfg.action.max_delta_ticks[j]
    for k in range(45):  # 1.5 s: wound against the stop, before the 3 s stuck-action trip
        if k % 3 == 0:
            rig.send(d)
        _check_tick(rig, j, E)
    blocked_at = int(rig.core.block.position[j])
    assert blocked_at <= at + (0 if kind == "fake" else 8)  # the stop held (MuJoCo's limit is soft)
    _unblock(rig, j, saved)
    peak_target = int(rig.core._target[j])
    settled = blocked_at
    held_ticks = 0
    for _ in range(90):  # no new commands: RUN/HOLD on the held target, then the stale-command safe stop
        _check_tick(rig, j, E)
        if rig.core.state in (SmState.RUN, SmState.HOLD):  # before the safe stop ramps the arm to home/rest
            peak_target = max(peak_target, int(rig.core._target[j]))
            settled = int(rig.core.block.position[j])
            held_ticks += 1
    assert held_ticks >= 10  # the arm had time to move into the freed space
    # The COMMAND never asks for more than E beyond where the joint was blocked (anti-windup). The plant then
    # settles on it; MuJoCo's position servo overshoots by ~25 ticks on the way (plant dynamics, not a command:
    # the real STS3215 overshoot with P=16/D=32 is an H-track measurement).
    assert peak_target <= blocked_at + E
    assert blocked_at < settled <= blocked_at + E


# ---------------------------------------------------------------------------------------------------------------
# Test 3: NaN / an out-of-range goal never reach write_goal


def test3_nan_command_never_reaches_the_bus(kind, cfg, calibration):
    rig = Rig(kind, cfg, calibration)
    rig.settle()
    n = len(rig.bus.written_goals)
    last = rig.bus.written_goals[-1].copy()
    rig.send([np.nan, 0, 0, 0, 0, 0])
    rig.tick()
    st = rig.state()
    assert int(st["fault_bits"]) & TripCause.NAN
    assert rig.core.state is SmState.SAFE_STOPPING
    goals = np.array(rig.bus.written_goals[n:])
    assert goals.dtype.kind == "i" and ((goals >= 0) & (goals <= 4095)).all()
    assert np.abs(goals[0].astype(int) - last.astype(int)).max() <= cfg.safe_state.ramp_ticks_per_tick


def test3_goal_4096_never_reaches_the_bus_and_trips_range(kind, cfg, calibration):
    """A misconfigured joint limit above 4095 (defence in depth): check_range is the last gate."""
    g = 5  # the gripper: not in the FK, so the box never intervenes
    hi = cfg.joints.max_ticks.copy()
    hi[g] = 4200
    E = cfg.action.tracking_error_ticks.copy()
    E[g] = 5000  # let the target run away from a gripper that cannot follow (MuJoCo joint range)
    bad = dataclasses.replace(
        cfg,
        joints=JointLimits(min_ticks=cfg.joints.min_ticks, max_ticks=hi),
        action=dataclasses.replace(cfg.action, tracking_error_ticks=E),
    )
    rig = Rig(kind, bad, calibration)
    rig.settle()
    n = len(rig.bus.written_goals)
    tripped = False
    for k in range(400):
        if k % 3 == 0:
            rig.send(unit(g) * (60 if (k // 3) % 2 else 58))  # alternate: no stuck-action trip first
        rig.tick()
        if int(rig.state()["fault_bits"]) & TripCause.RANGE:
            tripped = True
            break
    assert tripped
    assert rig.core.state is SmState.SAFE_STOPPING
    goals = np.array(rig.bus.written_goals[n:])
    assert goals.max() <= 4095 and goals.min() >= 0
    assert int(rig.core.pipeline.goal[g]) >= 4096 or int(rig.core.pipeline.prev_target[g]) >= 4095
