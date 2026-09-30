"""FeetechBus against a monkeypatched ``scservo_sdk`` (canned packets + per-motor register memory)."""

import time

import numpy as np
import pytest
import scservo_sdk as scs

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, RegisterConfig, load_config
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.feetech_bus import FeetechBus
from so101_rl.servo.timing import FakeClock

STS3215_MODEL_NUMBER = 777
ADDR = {
    "Firmware_Major_Version": 0,
    "Model_Number": 3,
    "Return_Delay_Time": 7,
    "Min_Position_Limit": 9,
    "Max_Position_Limit": 11,
    "P_Coefficient": 21,
    "Protection_Current": 28,
    "Homing_Offset": 31,
    "Operating_Mode": 33,
    "Overload_Torque": 36,
    "Over_Current_Protection_Time": 38,
    "Torque_Enable": 40,
    "Acceleration": 41,
    "Goal_Position": 42,
    "Lock": 55,
}


class World:
    """One object standing in for the serial line: register memory per motor + packet/port handler API."""

    def __init__(self, ids):
        self.mem = {i: bytearray(128) for i in ids}
        for m in self.mem.values():
            m[3:5] = STS3215_MODEL_NUMBER.to_bytes(2, "little")
            m[7] = 250  # factory Return_Delay_Time
        self.fail_sync_read = False
        self.silent = set()  # ids that never answer
        self.sync_writes = []  # (start_address, data_length, param bytes)
        self.reg_writes = []  # (id, addr, length, data bytes)
        self.closed = False
        self.sync_read_calls = 0
        self._pending = []
        self.is_open = False
        self.is_using = False

    # -- helpers for tests
    def set_reg(self, motor_id, addr, value, length):
        self.mem[motor_id][addr : addr + length] = int(value).to_bytes(length, "little")

    def get_reg(self, motor_id, addr, length):
        return int.from_bytes(self.mem[motor_id][addr : addr + length], "little")

    def set_block(self, motor_id, pos_raw, vel_raw, load_raw, volt, temp, status, moving):
        self.set_reg(motor_id, 56, pos_raw, 2)
        self.set_reg(motor_id, 58, vel_raw, 2)
        self.set_reg(motor_id, 60, load_raw, 2)
        self.set_reg(motor_id, 62, volt, 1)
        self.set_reg(motor_id, 63, temp, 1)
        self.set_reg(motor_id, 65, status, 1)
        self.set_reg(motor_id, 66, moving, 1)

    # -- port handler API
    def openPort(self):
        self.is_open = True
        self.closed = False
        return True

    def closePort(self):
        self.is_open = False
        self.closed = True

    def clearPort(self):
        pass

    def setBaudRate(self, baud):
        return True

    def setPacketTimeoutMillis(self, ms):
        pass

    # -- packet handler API
    def _alive(self, motor_id):
        return motor_id in self.mem and motor_id not in self.silent

    def ping(self, port, motor_id):
        if not self._alive(motor_id):
            return 0, scs.COMM_RX_TIMEOUT, 0
        return STS3215_MODEL_NUMBER, scs.COMM_SUCCESS, 0

    def _read_n(self, motor_id, addr, n):
        if not self._alive(motor_id):
            return 0, scs.COMM_RX_TIMEOUT, 0
        return self.get_reg(motor_id, addr, n), scs.COMM_SUCCESS, 0

    def read1ByteTxRx(self, port, motor_id, addr):
        return self._read_n(motor_id, addr, 1)

    def read2ByteTxRx(self, port, motor_id, addr):
        return self._read_n(motor_id, addr, 2)

    def writeTxRx(self, port, motor_id, addr, length, data):
        if not self._alive(motor_id):
            return scs.COMM_RX_TIMEOUT, 0
        self.mem[motor_id][addr : addr + length] = bytes(data)
        self.reg_writes.append((motor_id, addr, length, bytes(data)))
        return scs.COMM_SUCCESS, 0

    def syncReadTx(self, port, start, length, param, param_len):
        self.sync_read_calls += 1
        if self.fail_sync_read:
            return scs.COMM_TX_FAIL
        self._pending = [(i, start, length) for i in param[:param_len]]
        return scs.COMM_SUCCESS

    def readRx(self, port, motor_id, length):
        if not self._alive(motor_id):
            return [], scs.COMM_RX_TIMEOUT, 0
        start = next(s for i, s, _ in self._pending if i == motor_id)
        return list(self.mem[motor_id][start : start + length]), scs.COMM_SUCCESS, 0

    def syncWriteTxOnly(self, port, start, length, param, param_len):
        self.sync_writes.append((start, length, list(param[:param_len])))
        step = 1 + length
        for k in range(0, param_len, step):
            motor_id = param[k]
            self.mem[motor_id][start : start + length] = bytes(param[k + 1 : k + step])
        return scs.COMM_SUCCESS

    def getTxRxResult(self, comm):
        return f"comm={comm}"

    def getRxPacketError(self, error):
        return f"error={error}"


def _calib():
    return {
        n: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=-1647, range_min=500, range_max=3500)
        for i, n in enumerate(JOINT_NAMES)
    }


@pytest.fixture
def world(monkeypatch):
    w = World(range(1, 7))
    monkeypatch.setattr(scs, "PortHandler", lambda port: w)
    monkeypatch.setattr(scs, "PacketHandler", lambda version: w)
    return w


@pytest.fixture
def bus(world):
    cfg = load_config(calibration=_calib())
    b = FeetechBus(cfg, _calib(), clock=FakeClock(start_ns=123))
    b.connect()
    return b


def test_block_read_decodes_11_byte_packet(bus, world):
    world.set_block(1, pos_raw=2048, vel_raw=0x8000 | 300, load_raw=0x400 | 250, volt=121, temp=33, status=0x20, moving=1)
    for i in range(2, 7):
        world.set_block(i, 1000 * i % 4096, 10 * i, 50 * i, 120, 30 + i, 0, 0)
    out = BlockRead()
    assert bus.read_block(out) is True and out.ok is True
    assert out.position[0] == 2048
    assert out.velocity[0] == -300  # bit 15
    assert out.load[0] == -250  # bit 10 = negative load
    assert out.voltage_dv[0] == 121 and out.temperature_c[0] == 33
    assert out.status[0] == 0x20 and out.moving[0] == 1
    assert out.position[1] == 2000 and out.velocity[1] == 20 and out.load[1] == 100
    assert out.temperature_c[5] == 36
    assert out.t_mono_ns == 123
    assert out.position.dtype == np.int32


def test_single_sync_read_of_56_len_11_per_tick(bus, world):
    before = world.sync_read_calls
    bus.read_block(BlockRead())
    bus.read_block(BlockRead())
    assert world.sync_read_calls - before == 2


def test_comm_error_sets_ok_false_and_keeps_previous_values(bus, world):
    for i in range(1, 7):
        world.set_block(i, 1500 + i, 5, 7, 120, 40, 0, 0)
    out = BlockRead()
    assert bus.read_block(out)
    snapshot = {k: getattr(out, k).copy() for k in ("position", "velocity", "load", "voltage_dv", "temperature_c", "status", "moving")}
    world.fail_sync_read = True
    for i in range(1, 7):
        world.set_block(i, 3000, 9, 9, 99, 99, 9, 9)
    assert bus.read_block(out) is False
    assert out.ok is False
    for k, v in snapshot.items():
        assert np.array_equal(getattr(out, k), v), k


def test_missing_id_sets_ok_false_and_keeps_previous_values(bus, world):
    for i in range(1, 7):
        world.set_block(i, 1500 + i, 5, 7, 120, 40, 0, 0)
    out = BlockRead()
    assert bus.read_block(out)
    prev = out.position.copy()
    world.silent.add(4)
    for i in (1, 2, 3, 5, 6):
        world.set_block(i, 2222, 0, 0, 120, 40, 0, 0)
    assert bus.read_block(out) is False
    assert out.ok is False
    assert np.array_equal(out.position, prev)  # joints 1-3 were NOT half-updated


def test_ping_all_reports_silent_motor(bus, world):
    world.silent.add(2)
    assert bus.ping_all() == {1: True, 2: False, 3: True, 4: True, 5: True, 6: True}


def test_goal_position_is_raw_little_endian_two_bytes(bus, world):
    goal = np.array([0x0123, 4095, 0, 2048, 1, 300], dtype=np.int32)
    assert bus.write_goal(goal) is True
    start, length, param = world.sync_writes[-1]
    assert (start, length) == (42, 2)
    assert param[:3] == [1, 0x23, 0x01]
    assert param[3:6] == [2, 0xFF, 0x0F]
    assert len(param) == 6 * 3
    assert world.get_reg(1, ADDR["Goal_Position"], 2) == 0x0123


def test_write_goal_rejects_out_of_range(bus, world):
    with pytest.raises(ValueError):
        bus.write_goal(np.full(N_JOINTS, 4096, dtype=np.int32))
    with pytest.raises(ValueError):
        bus.write_goal(np.full(N_JOINTS, -1, dtype=np.int32))
    assert world.sync_writes == []


def test_homing_offset_sign_magnitude_decodes_minus_1647(bus, world):
    for i in range(1, 7):
        world.set_reg(i, ADDR["Homing_Offset"], 2048 + 1647, 2)  # sign bit 11 + magnitude 1647
        world.set_reg(i, ADDR["Min_Position_Limit"], 500 + i, 2)
        world.set_reg(i, ADDR["Max_Position_Limit"], 3500 - i, 2)
    assert bus.read_reg("Homing_Offset", 3) == -1647
    regs = bus.read_calibration_regs()
    assert [regs[n].homing_offset for n in JOINT_NAMES] == [-1647] * 6
    assert regs["elbow_flex"].range_min == 503 and regs["elbow_flex"].range_max == 3497
    assert regs["gripper"].id == 6 and regs["gripper"].drive_mode == 0


def test_read_write_reg_round_trip(bus, world):
    bus.write_reg("P_Coefficient", 2, 17)
    assert world.get_reg(2, ADDR["P_Coefficient"], 1) == 17
    assert bus.read_reg("P_Coefficient", 2) == 17


def test_disconnect_never_touches_torque(bus, world):
    for i in range(1, 7):
        world.set_reg(i, ADDR["Torque_Enable"], 1, 1)
    writes_before = len(world.reg_writes)
    bus.disconnect()
    assert world.closed is True
    assert len(world.reg_writes) == writes_before
    assert all(world.get_reg(i, ADDR["Torque_Enable"], 1) == 1 for i in range(1, 7))
    bus.disconnect()  # idempotent


def test_set_torque_subset_and_failure(bus, world):
    assert bus.set_torque(True, ids=[2, 3]) is True
    assert [world.get_reg(i, ADDR["Torque_Enable"], 1) for i in range(1, 7)] == [0, 1, 1, 0, 0, 0]
    assert bus.set_torque(False) is True
    assert all(world.get_reg(i, ADDR["Torque_Enable"], 1) == 0 for i in range(1, 7))
    world.silent.add(5)
    assert bus.set_torque(True) is False


def test_write_registers_writes_all_f1_registers_incl_protection(bus, world):
    reg = RegisterConfig()
    bus.write_registers(reg)
    for i in range(1, 7):
        assert world.get_reg(i, ADDR["Return_Delay_Time"], 1) == 0  # was 250
        assert world.get_reg(i, ADDR["Protection_Current"], 2) == reg.protection_current
        assert world.get_reg(i, ADDR["Overload_Torque"], 1) == reg.overload_torque
        assert world.get_reg(i, ADDR["Over_Current_Protection_Time"], 1) == reg.over_current_protection_time
        assert world.get_reg(i, ADDR["Operating_Mode"], 1) == reg.operating_mode
        assert world.get_reg(i, ADDR["P_Coefficient"], 1) == reg.p_coefficient
        assert world.get_reg(i, ADDR["Acceleration"], 1) == reg.acceleration
        assert world.get_reg(i, ADDR["Lock"], 1) == 0


def test_write_registers_refuses_while_torque_enabled(bus, world):
    world.set_reg(4, ADDR["Torque_Enable"], 1, 1)
    with pytest.raises(RuntimeError, match="torque"):
        bus.write_registers(RegisterConfig())
    assert world.get_reg(1, ADDR["Protection_Current"], 2) == 0


def test_connect_fails_when_a_motor_is_missing(world):
    world.silent.add(3)
    b = FeetechBus(load_config(calibration=_calib()), _calib())
    with pytest.raises(RuntimeError, match="Missing motor IDs"):
        b.connect()


def test_read_block_before_connect_raises(world):
    b = FeetechBus(load_config(calibration=_calib()), _calib())
    with pytest.raises(RuntimeError):
        b.read_block(BlockRead())


@pytest.mark.hardware
def test_block_read_latency():
    from so101_rl.config import load_calibration

    cfg = load_config()
    bus = FeetechBus(cfg, load_calibration(cfg.calibration_path))
    bus.connect()
    try:
        out = BlockRead()
        dts = []
        for _ in range(300):
            t0 = time.perf_counter_ns()
            ok = bus.read_block(out)
            dts.append(time.perf_counter_ns() - t0)
            assert ok
        assert np.percentile(dts, 99) / 1e6 < 6.0, f"p99 block read {np.percentile(dts, 99) / 1e6:.2f} ms"
        for motor_id in range(1, 7):
            assert bus.read_reg("Return_Delay_Time", motor_id) == 0
    finally:
        bus.disconnect()
