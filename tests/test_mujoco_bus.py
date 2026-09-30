import inspect

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

PROTOCOL_MEMBERS = [n for n, v in vars(Bus).items() if callable(v) and not n.startswith("_")]


@pytest.fixture
def bus(calibration, cfg):
    model, data = load_model()
    b = MujocoBus(model, data, calibration, cfg)
    b.connect()
    return b


def _hold_ticks(bus) -> np.ndarray:
    out = BlockRead()
    bus.read_block(out)
    return out.position.copy()


def test_implements_bus_protocol(bus):
    # Bus is a plain (non runtime_checkable) Protocol in U0's frozen file: check structurally, incl. signatures.
    assert set(PROTOCOL_MEMBERS) >= {
        "connect", "disconnect", "ping_all", "read_block", "write_goal", "set_torque",
        "read_reg", "write_reg", "read_calibration_regs",
    }  # fmt: skip
    for name in PROTOCOL_MEMBERS:
        impl = getattr(bus, name)
        assert callable(impl), name
        want = list(inspect.signature(getattr(Bus, name)).parameters)[1:]
        have = list(inspect.signature(impl).parameters)
        assert have == want, (name, have, want)
    assert Bus in MujocoBus.__mro__  # explicit F10 `class MujocoBus(Bus)`


def test_ping_and_calibration_regs(bus, calibration):
    assert bus.ping_all() == dict.fromkeys(range(1, 7), True)
    assert bus.read_calibration_regs() == calibration
    bus.write_reg("P_Coefficient", 1, 16)
    assert bus.read_reg("P_Coefficient", 1) == 16
    assert bus.read_reg("Homing_Offset", 1) == calibration["shoulder_pan"].homing_offset
    with pytest.raises(KeyError):
        bus.read_reg("Never_Written", 1)


def test_read_block_shapes_and_static_fields(bus):
    out = BlockRead()
    assert bus.read_block(out) and out.ok
    for arr in (out.position, out.velocity, out.load, out.voltage_dv, out.temperature_c, out.status, out.moving):
        assert arr.shape == (6,) and arr.dtype == np.int32
    assert (out.position >= 0).all() and (out.position <= 4095).all()
    assert (out.temperature_c == 35).all() and (out.status == 0).all()


def test_read_fails_when_disconnected(bus):
    bus.disconnect()
    out = BlockRead()
    assert bus.read_block(out) is False and out.ok is False
    assert bus.write_goal(np.full(6, 2047, np.int32)) is False


def test_goal_to_present_converges_within_half_degree_in_30_ticks(bus, calibration):
    start = _hold_ticks(bus)
    goal = start.copy()
    goal[:5] += np.array([60, -80, 70, -50, 100], dtype=np.int32)  # ~5-9 deg, inside the calibrated ranges
    goal[5] += 150
    tol_ticks = 0.5 / 360.0 * 4096  # 0.5 deg
    out = BlockRead()
    for _ in range(30):
        assert bus.write_goal(goal)
        bus.step_ticks(1)
        assert bus.read_block(out)
    # 0.5 deg plus gravity sag < 1 tick, plus quantisation
    assert np.all(np.abs(out.position - goal) <= tol_ticks + 1), (out.position, goal)


def test_written_goals_intercept_records_raw_ints(bus):
    g1 = np.array([2000, 2100, 2200, 2300, 2400, 2500], dtype=np.int32)
    g2 = g1 + 1
    bus.write_goal(g1)
    bus.write_goal(g2)
    assert len(bus.written_goals) == 2
    np.testing.assert_array_equal(bus.written_goals[0], g1)
    np.testing.assert_array_equal(bus.written_goals[1], g2)
    assert all(g.dtype == np.int32 for g in bus.written_goals)
    g1[0] = 0  # the intercept holds a copy
    assert bus.written_goals[0][0] == 2000


@pytest.mark.parametrize(
    "bad",
    [
        np.full(5, 2047, np.int32),
        np.full(6, 4096, np.int32),
        np.full(6, -1, np.int32),
        np.full(6, 2047.0),
    ],
)
def test_invalid_goal_rejected_and_not_recorded(bus, bad):
    assert bus.write_goal(bad) is False
    assert bus.written_goals == []


def test_steps_per_tick_follows_servo_hz(bus, cfg):
    assert bus.steps_per_tick == round((1 / cfg.rates.servo_hz) / bus.model.opt.timestep)
    t0 = bus.data.time
    bus.step_ticks(3)
    assert bus.data.time - t0 == pytest.approx(3 * bus.steps_per_tick * bus.model.opt.timestep)


def test_action_delay_fifo_delays_by_n_ticks(calibration, cfg):
    model, data = load_model()
    bus = MujocoBus(model, data, calibration, cfg, action_delay_ticks=3)
    bus.connect()
    start = _hold_ticks(bus)
    goal = start.copy()
    goal[0] += 100
    ctrl0 = bus.ticks_to_q(start)
    for i in range(3):
        bus.write_goal(goal)
        np.testing.assert_allclose(bus.data.ctrl[bus.act_ids], ctrl0, atol=1e-9, err_msg=f"write {i}")
    bus.write_goal(goal)  # 4th write releases the first goal
    assert bus.data.ctrl[bus.act_ids][0] != pytest.approx(ctrl0[0])
    assert len(bus.written_goals) == 4  # the intercept sees what was written, not what was applied


def test_torque_off_lets_arm_fall_and_on_restores(bus):
    start = _hold_ticks(bus)
    assert bus.set_torque(False)
    assert not bus.torque_enabled.any()
    bus.step_ticks(30)
    sagged = _hold_ticks(bus)
    assert np.abs(sagged - start).max() > 50  # gravity wins without torque
    assert bus.read_reg("Torque_Enable", 2) == 0
    assert bus.set_torque(True)
    assert bus.torque_enabled.all()
    bus.write_goal(start)
    bus.step_ticks(60)
    np.testing.assert_allclose(_hold_ticks(bus)[:5], start[:5], atol=3)


def test_torque_off_by_motor_id_only_affects_that_motor(bus):
    bus.set_torque(False, ids=[2])
    assert bus.torque_enabled.tolist() == [True, False, True, True, True, True]


def test_load_reflects_gravity_and_saturates(bus):
    bus.step_ticks(10)
    out = BlockRead()
    bus.read_block(out)
    assert np.abs(out.load).max() > 0
    assert np.abs(out.load).max() <= 1000


def test_reset_to_keyframe_restores_gains_and_clears_state(bus):
    bus.write_goal(np.full(6, 2100, np.int32))
    bus.set_torque(False)
    bus.reset_to_keyframe("rest")
    assert bus.torque_enabled.all() and bus.written_goals == []
    assert bus.model.actuator_gainprm[bus.act_ids[0], 0] == pytest.approx(998.22)


def test_joint_order_matches_constants(bus):
    assert [bus.model.joint(i).name for i in range(6)] == list(JOINT_NAMES)
