import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, RegisterConfig
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.fake_bus import FakeBus, FirstOrderPlant, ThermalModel
from so101_rl.servo.timing import FakeClock

MS = 1_000_000


def _calib():
    return {
        n: MotorCalibration(id=i + 1, drive_mode=i % 2, homing_offset=-1647 + i, range_min=100 + i, range_max=3900 - i)
        for i, n in enumerate(JOINT_NAMES)
    }


def _goal(v):
    return np.full(N_JOINTS, v, dtype=np.int32)


def _run(bus, clock, seconds, period_ms=33, out=None):
    out = out or BlockRead()
    for _ in range(int(seconds * 1000 / period_ms)):
        clock.advance(period_ms * MS)
        bus.read_block(out)
    return out


def test_satisfies_bus_protocol():
    bus: Bus = FakeBus(FakeClock())
    assert bus.ping_all() == dict.fromkeys(range(1, 7), True)


def test_plant_converges_to_goal():
    clock = FakeClock()
    bus = FakeBus(clock, initial_ticks=[2000] * 6)
    bus.set_torque(True)
    bus.write_goal(_goal(2500))
    out = _run(bus, clock, 2.0)
    assert out.ok
    assert np.all(np.abs(out.position - 2500) <= 1)
    assert np.all(out.moving == 0)


def test_plant_holds_without_torque():
    clock = FakeClock()
    bus = FakeBus(clock, initial_ticks=[2000] * 6)
    bus.write_goal(_goal(3000))
    out = _run(bus, clock, 1.0)
    assert np.all(out.position == 2000)


def test_plant_is_rate_limited():
    clock = FakeClock()
    bus = FakeBus(clock, plant=FirstOrderPlant(tau_s=0.01, max_speed_ticks_per_s=1000.0), initial_ticks=[1000] * 6)
    bus.set_torque(True)
    bus.write_goal(_goal(3000))
    clock.advance(100 * MS)
    out = BlockRead()
    bus.read_block(out)
    assert np.all(out.position <= 1000 + 100 + 1)
    assert np.all(out.velocity > 0)


def test_injected_read_failure_leaves_arrays_unchanged():
    clock = FakeClock()
    bus = FakeBus(clock, initial_ticks=[1234] * 6)
    out = BlockRead()
    clock.advance(33 * MS)
    assert bus.read_block(out) and out.ok
    before = {k: getattr(out, k).copy() for k in ("position", "velocity", "load", "voltage_dv", "temperature_c", "status", "moving")}
    t_before = out.t_mono_ns
    bus.faults.fail_next_read(2)
    for _ in range(2):
        clock.advance(33 * MS)
        assert bus.read_block(out) is False
        assert out.ok is False
        for k, v in before.items():
            assert np.array_equal(getattr(out, k), v), k
        assert out.t_mono_ns == t_before
    clock.advance(33 * MS)
    assert bus.read_block(out) and out.ok


def test_injected_write_failure_does_not_move_goal():
    clock = FakeClock()
    bus = FakeBus(clock)
    bus.faults.fail_next_write(1)
    assert bus.write_goal(_goal(1500)) is False
    assert bus.written_goals == []
    assert bus.write_goal(_goal(1500)) is True
    assert len(bus.written_goals) == 1


def test_written_goals_are_copies():
    bus = FakeBus(FakeClock())
    g = _goal(1000)
    bus.write_goal(g)
    g[:] = 7
    assert np.all(bus.written_goals[0] == 1000)


def test_write_goal_rejects_out_of_range_and_bad_shape():
    bus = FakeBus(FakeClock())
    with pytest.raises(ValueError):
        bus.write_goal(_goal(4096))
    with pytest.raises(ValueError):
        bus.write_goal(_goal(-1))
    with pytest.raises(ValueError):
        bus.write_goal(np.zeros(5, dtype=np.int32))


def test_dropped_motor_shows_in_ping_and_fails_read():
    clock = FakeClock()
    bus = FakeBus(clock)
    bus.faults.drop_motor(3)
    ping = bus.ping_all()
    assert ping[3] is False and all(v for k, v in ping.items() if k != 3)
    out = BlockRead()
    assert bus.read_block(out) is False and out.ok is False
    with pytest.raises(ConnectionError):
        bus.read_reg("Present_Position", 3)
    bus.faults.restore_motor(3)
    assert bus.ping_all()[3] is True
    assert bus.read_block(out) is True


def test_blocked_joint_never_passes_obstacle():
    clock = FakeClock()
    bus = FakeBus(clock, initial_ticks=[1000] * 6)
    bus.set_torque(True)
    bus.block("elbow_flex", 1300)
    bus.write_goal(_goal(2000))
    out = BlockRead()
    for _ in range(200):
        clock.advance(33 * MS)
        bus.read_block(out)
        assert out.position[2] <= 1300
    assert out.position[2] == 1300
    assert out.position[0] >= 1990  # the free joints got there
    assert out.load[2] > 0  # stalled against the obstacle: load reflects the tracking error
    bus.unblock("elbow_flex")
    _run(bus, clock, 2.0, out=out)
    assert out.position[2] >= 1990


def test_blocked_from_above_holds_lower_bound():
    clock = FakeClock()
    bus = FakeBus(clock, initial_ticks=[2000] * 6)
    bus.set_torque(True)
    bus.block(0, 1800)
    bus.write_goal(_goal(1000))
    out = _run(bus, clock, 3.0)
    assert out.position[0] == 1800
    assert out.position[1] <= 1005


def test_thermal_model_heats_under_load_and_cools():
    clock = FakeClock()
    bus = FakeBus(clock, thermal=ThermalModel(ambient_c=25.0, heat_c_per_s=0.5, tau_cool_s=60.0), initial_ticks=[1000] * 6)
    bus.set_torque(True)
    bus.block(1, 1100)
    bus.write_goal(_goal(3000))  # joint 1 pushes against the obstacle at full load
    out = _run(bus, clock, 120.0)
    assert out.temperature_c[1] > out.temperature_c[0] + 10
    hot = int(out.temperature_c[1])
    bus.unblock(1)
    bus.write_goal(_goal(1100))  # no sustained load, joints settle
    out = _run(bus, clock, 600.0)
    assert out.temperature_c[1] < hot - 10
    assert out.temperature_c[1] >= 25


def test_status_bits_and_voltage():
    clock = FakeClock()
    bus = FakeBus(clock)
    bus.faults.set_status_bits(2, 0x20)
    bus.set_voltage_dv(110)
    out = BlockRead()
    assert bus.read_block(out)
    assert out.status[1] == 0x20 and out.status.sum() == 0x20
    assert np.all(out.voltage_dv == 110)
    bus.faults.set_status_bits(2, 0)
    bus.read_block(out)
    assert out.status.sum() == 0


def test_latency_distribution_advances_clock():
    clock = FakeClock()
    bus = FakeBus(clock)
    bus.faults.latency_ms(2.0)
    out = BlockRead()
    t0 = clock.monotonic_ns()
    bus.read_block(out)
    assert clock.monotonic_ns() - t0 == 2 * MS
    assert out.t_mono_ns == clock.monotonic_ns()
    bus.faults.latency_ms(lambda rng: float(rng.uniform(1.0, 3.0)))
    for _ in range(20):
        t0 = clock.monotonic_ns()
        bus.read_block(out)
        assert 1 * MS <= clock.monotonic_ns() - t0 <= 3 * MS


def test_torque_events_recorded():
    clock = FakeClock()
    bus = FakeBus(clock)
    bus.set_torque(True)
    clock.advance(5 * MS)
    bus.set_torque(False, ids=[6, 5])
    assert [(e[1], e[2]) for e in bus.torque_events] == [(True, (1, 2, 3, 4, 5, 6)), (False, (6, 5))]
    assert bus.torque_events[1][0] == 5 * MS
    assert bus.torque_enabled().tolist() == [True, True, True, True, False, False]


def test_disconnect_never_touches_torque():
    bus = FakeBus(FakeClock())
    bus.connect()
    bus.set_torque(True)
    bus.disconnect()
    assert bus.torque_enabled().all()
    assert len(bus.torque_events) == 1


def test_read_calibration_regs_round_trips_construction_calibration():
    calib = _calib()
    bus = FakeBus(FakeClock(), calibration=calib)
    assert bus.read_calibration_regs() == calib


def test_write_registers_then_read_back():
    bus = FakeBus(FakeClock())
    assert bus.read_reg("Return_Delay_Time", 1) != 0  # factory value until configured
    bus.write_registers(RegisterConfig())
    for motor_id in range(1, 7):
        assert bus.read_reg("Return_Delay_Time", motor_id) == 0
        assert bus.read_reg("Protection_Current", motor_id) == RegisterConfig().protection_current
        assert bus.read_reg("Over_Current_Protection_Time", motor_id) == RegisterConfig().over_current_protection_time
    bus.write_reg("Overload_Torque", 4, 33)
    assert bus.read_reg("Overload_Torque", 4) == 33


def test_24h_soak_on_fake_clock_is_fast():
    clock = FakeClock()
    bus = FakeBus(clock)
    bus.set_torque(True)
    out = BlockRead()
    for _ in range(30 * 600):  # ten minutes of 30 Hz ticks
        clock.advance(33_333_333)
        bus.read_block(out)
    assert out.ok
