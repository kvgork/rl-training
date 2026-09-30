"""Deterministic in-memory bus (plan F2, Task 1.1): first-order plant, thermal model, fault injection.

``FakeBus`` implements :class:`so101_rl.servo.bus.Bus` on a :class:`~so101_rl.servo.timing.Clock`. The plant is
integrated lazily: every bus call first advances the physical state to ``clock.monotonic_ns()``, so a
``FakeClock`` makes a 24 h soak run in seconds and results do not depend on wall time.

Failure semantics mirror the real bus: an injected read failure returns ``False``, sets ``out.ok = False`` and
leaves every array untouched (never zeros); ``write_goal`` is an un-acked sync write, so a dropped motor does not
make it fail - only an injected write failure does.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, RegisterConfig
from so101_rl.servo.bus import BlockRead
from so101_rl.servo.timing import Clock

_MAX_SUBSTEP_S = 0.01
_TICK_MAX = 4095
_DEFAULT_TICK = 2047
_DEFAULT_VOLTAGE_DV = 120
_FACTORY_RETURN_DELAY = 250  # STS3215 factory value; configure() must bring it to 0

LatencyDist = float | Callable[[np.random.Generator], float]


class FirstOrderPlant:
    """``d present/dt = (goal - present) / tau``, rate limited to ``max_speed_ticks_per_s``.

    Deviation from the plan text (``tau_ticks_per_tick``): the constants are in seconds so the plant is
    independent of the servo tick rate.
    """

    def __init__(self, tau_s: float = 0.12, max_speed_ticks_per_s: float = 2500.0) -> None:
        if tau_s <= 0 or max_speed_ticks_per_s <= 0:
            raise ValueError("tau_s and max_speed_ticks_per_s must be positive")
        self.tau_s = tau_s
        self.max_speed = max_speed_ticks_per_s

    def step(self, present: np.ndarray, goal: np.ndarray, dt_s: float) -> np.ndarray:
        """New present positions (float ticks) after ``dt_s``; pure function."""
        delta = (goal - present) * (1.0 - math.exp(-dt_s / self.tau_s))
        limit = self.max_speed * dt_s
        return present + np.clip(delta, -limit, limit)


class ThermalModel:
    """``dT/dt = heat_c_per_s * (load / 1000)^2 - (T - ambient) / tau_cool_s`` per joint."""

    def __init__(self, ambient_c: float = 25.0, heat_c_per_s: float = 0.25, tau_cool_s: float = 300.0) -> None:
        if tau_cool_s <= 0:
            raise ValueError("tau_cool_s must be positive")
        self.ambient_c = ambient_c
        self.heat_c_per_s = heat_c_per_s
        self.tau_cool_s = tau_cool_s

    def step(self, temp_c: np.ndarray, load: np.ndarray, dt_s: float) -> np.ndarray:
        frac = load / 1000.0
        d_temp = self.heat_c_per_s * frac * frac - (temp_c - self.ambient_c) / self.tau_cool_s
        return temp_c + d_temp * dt_s


class FaultInjector:
    """Scripted bus faults. All counters are consumed by :class:`FakeBus`."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self._reads_to_fail = 0
        self._writes_to_fail = 0
        self._status_bits: dict[int, int] = {}
        self._dropped: set[int] = set()
        self._latency: LatencyDist = 0.0

    def fail_next_read(self, n: int = 1) -> None:
        self._reads_to_fail += n

    def fail_next_write(self, n: int = 1) -> None:
        self._writes_to_fail += n

    def set_status_bits(self, motor_id: int, bits: int) -> None:
        """Value reported in the ``Status`` (65) register of ``motor_id``; 0 clears it."""
        self._status_bits[motor_id] = int(bits) & 0xFF

    def latency_ms(self, dist: LatencyDist) -> None:
        """Per-operation bus latency: a constant in ms or ``f(rng) -> ms``."""
        self._latency = dist

    def drop_motor(self, motor_id: int) -> None:
        self._dropped.add(motor_id)

    def restore_motor(self, motor_id: int) -> None:
        self._dropped.discard(motor_id)

    def is_dropped(self, motor_id: int) -> bool:
        return motor_id in self._dropped

    def status_bits(self, motor_id: int) -> int:
        return self._status_bits.get(motor_id, 0)

    def consume_read_failure(self) -> bool:
        if self._reads_to_fail > 0:
            self._reads_to_fail -= 1
            return True
        return False

    def consume_write_failure(self) -> bool:
        if self._writes_to_fail > 0:
            self._writes_to_fail -= 1
            return True
        return False

    def sample_latency_ns(self) -> int:
        ms = self._latency(self.rng) if callable(self._latency) else float(self._latency)
        return max(0, round(ms * 1e6))


class FakeBus:
    """``Bus`` implementation on a simulated plant. Extra test hooks: ``written_goals``, ``torque_events``."""

    def __init__(
        self,
        clock: Clock,
        plant: FirstOrderPlant | None = None,
        faults: FaultInjector | None = None,
        *,
        calibration: dict[str, MotorCalibration] | None = None,
        thermal: ThermalModel | None = None,
        initial_ticks: Sequence[int] | None = None,
        load_gain: float = 5.0,
    ) -> None:
        self.clock = clock
        self.plant = plant if plant is not None else FirstOrderPlant()
        self.faults = faults if faults is not None else FaultInjector()
        self.thermal = thermal if thermal is not None else ThermalModel()
        self.load_gain = load_gain
        self.calibration = calibration or {
            n: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=_TICK_MAX)
            for i, n in enumerate(JOINT_NAMES)
        }
        self._ids = np.array([self.calibration[n].id for n in JOINT_NAMES], dtype=np.int64)
        start = np.full(N_JOINTS, _DEFAULT_TICK) if initial_ticks is None else np.asarray(initial_ticks)
        self._pos = start.astype(np.float64)
        self._goal = self._pos.copy()
        self._temp = np.full(N_JOINTS, self.thermal.ambient_c, dtype=np.float64)
        self._load = np.zeros(N_JOINTS, dtype=np.float64)
        self._vel = np.zeros(N_JOINTS, dtype=np.float64)
        self._torque = np.zeros(N_JOINTS, dtype=bool)
        self._voltage_dv = np.full(N_JOINTS, _DEFAULT_VOLTAGE_DV, dtype=np.int64)
        self._obstacles: dict[int, tuple[float, int]] = {}
        self._last_ns = clock.monotonic_ns()
        self._regs: dict[tuple[str, int], int] = {}
        for c in self.calibration.values():
            self._regs[("Homing_Offset", c.id)] = c.homing_offset
            self._regs[("Min_Position_Limit", c.id)] = c.range_min
            self._regs[("Max_Position_Limit", c.id)] = c.range_max
            self._regs[("ID", c.id)] = c.id
            self._regs[("Return_Delay_Time", c.id)] = _FACTORY_RETURN_DELAY
        self.connected = False
        self.written_goals: list[np.ndarray] = []
        self.written_goal_t_ns: list[int] = []
        self.torque_events: list[tuple[int, bool, tuple[int, ...]]] = []

    # -- plant ---------------------------------------------------------------------------------
    def _joint_index(self, joint: str | int) -> int:
        return JOINT_NAMES.index(joint) if isinstance(joint, str) else int(joint)

    def block(self, joint: str | int, at_ticks: float) -> None:
        """Place an obstacle at ``at_ticks``; ``present`` can never cross it from the side it is on now."""
        j = self._joint_index(joint)
        self._advance()
        self._obstacles[j] = (float(at_ticks), 1 if self._pos[j] <= at_ticks else -1)

    def unblock(self, joint: str | int) -> None:
        self._obstacles.pop(self._joint_index(joint), None)

    def set_voltage_dv(self, dv: int) -> None:
        self._voltage_dv[:] = dv

    def set_temperature(self, joint: str | int, temp_c: float) -> None:
        self._advance()
        self._temp[self._joint_index(joint)] = temp_c

    def _advance(self) -> None:
        now = self.clock.monotonic_ns()
        dt = (now - self._last_ns) / 1e9
        self._last_ns = max(self._last_ns, now)
        if dt <= 0:
            return
        n = max(1, math.ceil(dt / _MAX_SUBSTEP_S))
        h = dt / n
        start = self._pos.copy()
        for _ in range(n):
            moved = self.plant.step(self._pos, self._goal, h)
            self._pos = np.where(self._torque, moved, self._pos)
            for j, (at, side) in self._obstacles.items():
                self._pos[j] = min(self._pos[j], at) if side > 0 else max(self._pos[j], at)
            self._load = np.where(self._torque, np.clip((self._goal - self._pos) * self.load_gain, -1000, 1000), 0.0)
            self._temp = self.thermal.step(self._temp, self._load, h)
        self._vel = (self._pos - start) / dt

    def _wait_latency(self) -> None:
        lat = self.faults.sample_latency_ns()
        if lat:
            self.clock.sleep_until(self.clock.monotonic_ns() + lat)

    # -- Bus protocol --------------------------------------------------------------------------
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False  # never touches torque

    def ping_all(self) -> dict[int, bool]:
        return {int(i): not self.faults.is_dropped(int(i)) for i in self._ids}

    def read_block(self, out: BlockRead) -> bool:
        self._wait_latency()
        self._advance()
        if self.faults.consume_read_failure() or any(self.faults.is_dropped(int(i)) for i in self._ids):
            out.ok = False
            return False
        out.position[:] = np.clip(np.rint(self._pos), 0, _TICK_MAX).astype(np.int32)
        out.velocity[:] = np.rint(self._vel).astype(np.int32)
        out.load[:] = np.rint(self._load).astype(np.int32)
        out.voltage_dv[:] = self._voltage_dv
        out.temperature_c[:] = np.rint(self._temp).astype(np.int32)
        out.status[:] = [self.faults.status_bits(int(i)) for i in self._ids]
        out.moving[:] = (np.abs(self._vel) > 0.5).astype(np.int32)
        out.t_mono_ns = self.clock.monotonic_ns()
        out.ok = True
        return True

    def write_goal(self, goal_ticks: np.ndarray) -> bool:
        goal = np.asarray(goal_ticks)
        if goal.shape != (N_JOINTS,):
            raise ValueError(f"goal_ticks must have shape ({N_JOINTS},), got {goal.shape}")
        if goal.min() < 0 or goal.max() > _TICK_MAX:
            raise ValueError(f"goal_ticks outside 0..{_TICK_MAX}: {goal}")
        self._wait_latency()
        self._advance()
        if self.faults.consume_write_failure():
            return False
        for j, motor_id in enumerate(self._ids):
            if not self.faults.is_dropped(int(motor_id)):  # un-acked sync write: a dead motor is silent
                self._goal[j] = float(goal[j])
        self.written_goals.append(goal.astype(np.int32, copy=True))
        self.written_goal_t_ns.append(self.clock.monotonic_ns())
        return True

    def set_torque(self, enabled: bool, ids: Sequence[int] | None = None) -> bool:
        self._wait_latency()
        self._advance()
        if self.faults.consume_write_failure():
            return False
        target = tuple(int(i) for i in (self._ids if ids is None else ids))
        if any(self.faults.is_dropped(i) for i in target):
            return False  # acked write: a missing motor is an error here
        for j, motor_id in enumerate(self._ids):
            if int(motor_id) in target:
                self._torque[j] = enabled
        self.torque_events.append((self.clock.monotonic_ns(), bool(enabled), target))
        return True

    def torque_enabled(self) -> np.ndarray:
        return self._torque.copy()

    def read_reg(self, name: str, motor_id: int) -> int:
        self._advance()
        if self.faults.is_dropped(motor_id):
            raise ConnectionError(f"motor {motor_id} does not answer")
        j = int(np.flatnonzero(self._ids == motor_id)[0])
        live = {
            "Present_Position": int(np.clip(np.rint(self._pos[j]), 0, _TICK_MAX)),
            "Present_Temperature": int(np.rint(self._temp[j])),
            "Present_Voltage": int(self._voltage_dv[j]),
            "Torque_Enable": int(self._torque[j]),
            "Goal_Position": int(self._goal[j]),
        }
        if name in live:
            return live[name]
        return self._regs[(name, motor_id)]

    def write_reg(self, name: str, motor_id: int, value: int) -> None:
        self._advance()
        if self.faults.is_dropped(motor_id):
            raise ConnectionError(f"motor {motor_id} does not answer")
        if name == "Torque_Enable":
            self.set_torque(bool(value), [motor_id])
            return
        self._regs[(name, motor_id)] = int(value)

    def write_registers(self, registers: RegisterConfig) -> None:
        """Mirror of ``FeetechBus.write_registers``: every F1 register to every motor."""
        for f in dataclasses.fields(registers):
            name = "_".join(w.capitalize() for w in f.name.split("_"))
            for motor_id in self._ids:
                self._regs[(name, int(motor_id))] = int(getattr(registers, f.name))

    def read_calibration_regs(self) -> dict[str, MotorCalibration]:
        out: dict[str, MotorCalibration] = {}
        for name in JOINT_NAMES:
            c = self.calibration[name]
            out[name] = MotorCalibration(
                id=self._regs[("ID", c.id)],
                drive_mode=c.drive_mode,  # not stored on the servo; comes from config
                homing_offset=self._regs[("Homing_Offset", c.id)],
                range_min=self._regs[("Min_Position_Limit", c.id)],
                range_max=self._regs[("Max_Position_Limit", c.id)],
            )
        return out
