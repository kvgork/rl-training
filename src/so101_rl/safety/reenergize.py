"""The frozen re-energise order (plan F5 / ARCHITECTURE section 6.5, lerobot-specialist guidance).

Order, and why:

1. ``ping_all``: wait until every calibrated motor answers (a missing motor would get no goal and sag).
2. ``require_calibration``: read the motors' calibration registers back and compare with the JSON. A mismatch
   raises ``CalibrationMismatch`` and ``set_torque(True)`` is NEVER called (a wrong homing offset means every
   goal lands somewhere else).
3. write every ``RegisterConfig`` field to every motor (acceleration etc. must be set BEFORE torque).
4. ``read_block`` -> ``write_goal(present)`` -> read ``Goal_Position`` back from every motor and require it equals
   present. A failed read is never treated as ``present = 0``.
5. ``set_torque(True)``: only now. The arm does not jump because goal == present.
6. ramp home with the same per-tick step as the safe-state sequencer (and the tracking clamp), so the first
   policy command starts from a known pose.

Failure after step 5 leaves torque ON: the caller (``ServoCore.start``) must route the exception into its
``shutdown`` (bounded safe-state sequence).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, So101Config
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.timing import Clock

from . import clamp
from .calib_check import require_calibration
from .safe_state import SETTLE_TICKS, ramp_toward

_NS = 1_000_000_000
_PING_PERIOD_S = 0.05
_RAMP_SLACK_S = 1.0


def register_name(field_name: str) -> str:
    """``p_coefficient`` -> ``P_Coefficient`` (Feetech control-table name)."""
    return "_".join(part.capitalize() for part in field_name.split("_"))


def _ordered(calibration: dict[str, MotorCalibration]) -> list[tuple[str, MotorCalibration]]:
    return [(n, calibration[n]) for n in JOINT_NAMES if n in calibration]


def reenergize(
    bus: Bus,
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    clock: Clock,
    block: BlockRead,
    *,
    timeout_s: float = 5.0,
) -> None:
    """1. wait until all 6 ids answer (ping_all)  2. require_calibration  3. write_registers(cfg.registers)
    4. read_block -> write_goal(present) -> read back Goal_Position == present for all 6  5. set_torque(True)
    6. ramped home (via the SafeStateSequencer RAMP_HOME step).

    Raises ``TimeoutError`` (motors missing, block read failing, homing not reached), ``CalibrationMismatch``,
    or ``RuntimeError`` (goal read-back differs, bus write failed). The order is asserted by tests/test_reenergize.py.
    """
    motors = _ordered(calibration)
    if len(motors) != N_JOINTS:
        raise ValueError(f"calibration must contain all {N_JOINTS} joints, got {sorted(calibration)}")
    ids = [c.id for _, c in motors]
    deadline = clock.monotonic_ns() + int(timeout_s * _NS)

    def wait_until(what: str, ok) -> None:
        while not ok():
            now = clock.monotonic_ns()
            if now >= deadline:
                raise TimeoutError(f"re-energise: timed out waiting for {what}")
            clock.sleep_until(min(now + int(_PING_PERIOD_S * _NS), deadline))

    # 1. every motor must answer
    def all_answer() -> bool:
        answered = bus.ping_all()  # ONE ping per poll
        return all(answered.get(i, False) for i in ids)

    wait_until("all motor ids to answer ping", all_answer)

    # 2. calibration read-back, before anything is powered
    require_calibration(bus, calibration)

    # 3. registers (all of them, every time)
    for reg_field, value in asdict(cfg.registers).items():
        name = register_name(reg_field)
        for motor_id in ids:
            bus.write_reg(name, motor_id, int(value))

    # 4. goal = present, verified by read-back
    wait_until("a successful block read", lambda: bus.read_block(block) and block.ok)
    present = block.position.copy()
    if not clamp.check_range(present):
        raise RuntimeError(f"re-energise: present position outside 0..4095: {present.tolist()}")
    if not bus.write_goal(present):
        raise RuntimeError("re-energise: write_goal(present) failed")
    for (name, _), motor_id, want in zip(motors, ids, present, strict=True):
        got = bus.read_reg("Goal_Position", motor_id)
        if got != int(want):
            raise RuntimeError(f"re-energise: Goal_Position of {name} reads {got}, expected present {int(want)}")

    # 5. only now
    if not bus.set_torque(True):
        raise RuntimeError("re-energise: set_torque(True) failed")

    # 6. ramp home, same step and tracking clamp as the safe-state ramp
    _ramp_home(bus, cfg, clock, block, present, timeout_s)


def _ramp_home(
    bus: Bus, cfg: So101Config, clock: Clock, block: BlockRead, present: np.ndarray, timeout_s: float
) -> None:
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    step = int(cfg.safe_state.ramp_ticks_per_tick)
    period_ns = int(_NS / cfg.rates.servo_hz)
    target = present.astype(np.int32)
    nxt = np.empty(N_JOINTS, dtype=np.int32)
    lim = np.empty(N_JOINTS, dtype=np.int32)
    mask = np.empty(N_JOINTS, dtype=bool)
    worst = int(np.abs(home - present).max())
    budget_s = timeout_s + _RAMP_SLACK_S + (worst / max(step, 1)) / cfg.rates.servo_hz
    deadline = clock.monotonic_ns() + int(budget_s * _NS)
    next_tick = clock.monotonic_ns()
    while True:
        if clock.monotonic_ns() >= deadline:
            raise TimeoutError("re-energise: ramp home did not complete in time")
        got = bus.read_block(block) and block.ok
        if got and int(np.abs(block.position - home).max()) <= SETTLE_TICKS and bool((target == home).all()):
            return
        if block.ok:
            ramp_toward(target, home, step, nxt)
            clamp.clamp_joint_limits(nxt, cfg.joints, lim, mask)
            clamp.clamp_tracking(lim, block.position, cfg.action.tracking_error_ticks, target, mask)
        if not clamp.check_range(target):
            raise RuntimeError("re-energise: ramp target out of range")
        if not bus.write_goal(target):
            raise RuntimeError("re-energise: write_goal failed during ramp home")
        next_tick += period_ns
        clock.sleep_until(next_tick)
