"""``reenergize`` end-to-end on the two real Bus backends (FakeBus and MujocoBus), not just a logging double."""

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES
from so101_rl.safety.reenergize import reenergize
from so101_rl.safety.safe_state import SETTLE_TICKS
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.timing import FakeClock
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

NS = 1_000_000_000


class SimClock:
    """Clock backed by the MuJoCo time: ``sleep_until`` steps the physics, like the real 30 Hz loop would."""

    def __init__(self, bus: MujocoBus) -> None:
        self.bus = bus
        self.period_ns = round(bus.steps_per_tick * bus.model.opt.timestep * NS)

    def monotonic_ns(self) -> int:
        return round(self.bus.data.time * NS)

    def sleep_until(self, t_ns: int) -> None:
        n = -(-(t_ns - self.monotonic_ns()) // self.period_ns)  # ceil
        if n > 0:
            self.bus.step_ticks(n)


class Recorder:
    """Delegating proxy that logs the calls ``reenergize`` makes that define its order."""

    def __init__(self, bus: Bus) -> None:
        self._bus = bus
        self.log: list[tuple] = []

    def write_goal(self, goal):
        self.log.append(("write_goal", np.array(goal)))
        return self._bus.write_goal(goal)

    def read_reg(self, name, motor_id):
        v = self._bus.read_reg(name, motor_id)
        self.log.append(("read_reg", name, motor_id, v))
        return v

    def set_torque(self, enabled, ids=None):
        self.log.append(("set_torque", bool(enabled)))
        return self._bus.set_torque(enabled, ids)

    def __getattr__(self, name):
        return getattr(self._bus, name)


def _make_fake(cfg, calibration):
    clock = FakeClock()
    # a real arm that was parked at the rest pose (same start pose as the mujoco backend's `rest` keyframe)
    start = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int32)
    bus = FakeBus(clock, calibration=calibration, initial_ticks=start)
    bus.connect()
    return bus, clock


def _make_mujoco(cfg, calibration):
    model, data = load_model()
    bus = MujocoBus(model, data, calibration, cfg)
    bus.connect()
    bus.reset_to_keyframe("rest")
    bus.set_torque(False)  # a de-energised arm: re-energise starts from torque off
    return bus, SimClock(bus)


@pytest.fixture(params=["fake", "mujoco"])
def backend(request, cfg, calibration):
    make = _make_fake if request.param == "fake" else _make_mujoco
    bus, clock = make(cfg, calibration)
    return request.param, bus, clock


def test_backends_are_buses(backend):
    _, bus, _ = backend
    assert isinstance(bus, Bus)


def test_reenergize_order_success_and_final_pose(backend, cfg, calibration):
    _, bus, clock = backend
    rec = Recorder(bus)
    block = BlockRead()
    reenergize(rec, cfg, calibration, clock, block)  # raises on any failure

    kinds = [e[0] for e in rec.log]
    t_on = rec.log.index(("set_torque", True))
    assert kinds.count("set_torque") == 1
    # goal = present is written and read back from all six motors BEFORE torque goes on
    first_goal = kinds.index("write_goal")
    assert first_goal < t_on
    present_goal = rec.log[first_goal][1]
    readbacks = [e for e in rec.log[first_goal:t_on] if e[0] == "read_reg" and e[1] == "Goal_Position"]
    assert [e[2] for e in readbacks] == [calibration[n].id for n in JOINT_NAMES]
    assert [e[3] for e in readbacks] == present_goal.tolist()
    # nothing but the goal readback and goal writes happens between the first goal and torque on
    assert all(k in ("read_reg", "write_goal") for k in kinds[first_goal:t_on])

    assert bus.read_block(block) and block.ok
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    assert np.abs(block.position - home).max() <= SETTLE_TICKS
    assert np.asarray(bus.torque_enabled() if callable(bus.torque_enabled) else bus.torque_enabled).all()


def test_mujoco_serves_live_registers(cfg, calibration):
    bus, _ = _make_mujoco(cfg, calibration)
    out = BlockRead()
    assert bus.read_block(out)
    for j, n in enumerate(JOINT_NAMES):
        mid = calibration[n].id
        assert bus.read_reg("Present_Position", mid) == out.position[j]
        assert bus.read_reg("Goal_Position", mid) == out.position[j]  # nothing written: hold
        assert bus.read_reg("Torque_Enable", mid) == 0
    bus.write_reg("Lock", 1, 1)
    assert bus.read_reg("Lock", 1) == 1
    bus.write_reg("Torque_Enable", 1, 1)
    assert bus.read_reg("Torque_Enable", 1) == 1
    assert bus.torque_enabled[0] and not bus.torque_enabled[1]
