"""The frozen re-energise order (plan F5 / ARCHITECTURE section 6.5, lerobot-specialist guidance).

Order, and why:

1. ``ping_all``: wait until every calibrated motor answers (a missing motor would get no goal and sag).
2. ``require_calibration``: read the motors' calibration registers back and compare with the JSON. A mismatch
   raises ``CalibrationMismatch`` and ``set_torque(True)`` is NEVER called (a wrong homing offset means every
   goal lands somewhere else).
3. Protection and control registers (review B3), all BEFORE torque:
   a. ``Lock = 0`` on every motor (EEPROM registers are ignored while locked);
   b. ``Torque_Enable`` read back as 0 on every motor (EEPROM writes on a live arm are refused);
   c. every ``RegisterConfig`` field written to every motor;
   d. every field read back from every motor; any difference raises before torque.
4. ``read_block`` -> start-pose check -> temperature / status check -> ``write_goal(present)`` -> read
   ``Goal_Position`` back from every motor and require it equals present. A failed read is never treated as
   ``present = 0``.
   * Start pose (review B1 d): the arm must be near the rest pose (every arm joint within
     ``safe_state.rest_tolerance_ticks``) or with its tip inside the box; anything else (the arm fell somewhere
     after a torque-off in place) needs a human.
   * Temperatures and status (review S5): every joint below ``thermal.trip_c`` and not 0/255, Status 0.
5. ``set_torque(True)``: only now. The arm does not jump because goal == present.
6. ramp home through the transit zone (``safety/transit.py``), with the same per-tick step and tracking clamp as
   the safe-state sequencer, so the first policy command starts from a known pose.

**Torque never stays on after a failure** (review S5): any exception after step 5 turns torque off on every
motor (``torque_off_retry``: up to 3 attempts) before it propagates; if even that fails, a note on the exception
says so. Temperatures and status bits are re-checked on every good read of the ramp. The caller still owns the
deadman and the state machine.

**Heartbeat** (review B2): ``tick_hook(ok)`` is called once per servo tick, ``ok`` = that tick's ``read_block``
succeeded AND its ``write_goal`` was accepted. The first such tick is step 4's ``write_goal(present)``. With a
hook, the time from entry to that first good tick, and every later gap between good ticks, must stay below
``heartbeat_gap_s`` (0.8 s, under the MCU's 1000 ms timeout), else ``TimeoutError``: an MCU that is about to
latch must not be raced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, So101Config
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.timing import Clock

from . import clamp
from .calib_check import require_calibration
from .fk import KinematicChain, So101FK
from .safe_state import SETTLE_TICKS
from .transit import TransitPlanner

_NS = 1_000_000_000
_PING_PERIOD_S = 0.05
_RAMP_SLACK_S = 1.0
_SENSOR_DEAD = (0, 255)
_N_ARM = 5
_TORQUE_OFF_ATTEMPTS = 3

TickHook = Callable[[bool], None]


class UnsafeStartPose(RuntimeError):
    """The arm is neither near rest nor inside the box: re-energising could swing it through anything."""


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
    tick_hook: TickHook | None = None,
    heartbeat_gap_s: float = 0.8,
    fk: So101FK | None = None,
) -> None:
    """1. wait until all 6 ids answer (ping_all)  2. require_calibration  3. Lock=0, torque-off check, registers,
    read-back  4. read_block -> start pose / temperature check -> write_goal(present) -> read back
    Goal_Position == present for all 6  5. set_torque(True)  6. ramp home through the transit zone.

    Raises ``TimeoutError`` (motors missing, block read failing, heartbeat gap, homing not reached),
    ``CalibrationMismatch``, ``UnsafeStartPose`` or ``RuntimeError`` (register / goal read-back differs, torque on
    before the register writes, hot or faulted motor, bus write failed). After step 5 every failure turns torque
    off before it propagates. The order is asserted by tests/test_reenergize.py.
    """
    motors = _ordered(calibration)
    if len(motors) != N_JOINTS:
        raise ValueError(f"calibration must contain all {N_JOINTS} joints, got {sorted(calibration)}")
    ids = [c.id for _, c in motors]
    t_entry = clock.monotonic_ns()
    deadline = t_entry + int(timeout_s * _NS)
    gap_ns = int(heartbeat_gap_s * _NS)
    # With a heartbeat hook the pre-torque phase must reach its first good tick before the MCU times out.
    pre_deadline = min(deadline, t_entry + gap_ns) if tick_hook is not None else deadline
    if fk is None:
        fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), calibration)

    def wait_until(what: str, ok, until_ns: int) -> None:
        while not ok():
            now = clock.monotonic_ns()
            if now >= until_ns:
                raise TimeoutError(f"re-energise: timed out waiting for {what}")
            clock.sleep_until(min(now + int(_PING_PERIOD_S * _NS), until_ns))

    # 1. every motor must answer
    def all_answer() -> bool:
        answered = bus.ping_all()  # ONE ping per poll
        return all(answered.get(i, False) for i in ids)

    wait_until("all motor ids to answer ping", all_answer, pre_deadline)

    # 2. calibration read-back, before anything is powered
    require_calibration(bus, calibration)

    # 3. registers (all of them, every time), with the guards of FeetechBus.write_registers and a read-back
    _write_registers_verified(bus, cfg, ids)

    # 4. goal = present, verified by read-back. Each attempt is one servo tick for the heartbeat hook.
    def first_good_tick() -> bool:
        if not (bus.read_block(block) and block.ok):
            if tick_hook is not None:
                tick_hook(False)
            return False
        if tick_hook is not None and clock.monotonic_ns() - t_entry > gap_ns:
            # wait_until evaluates ok() before its deadline: a slow register phase must not slip through
            raise TimeoutError("re-energise: no good tick within the heartbeat gap after entry (pre-torque too slow)")
        present = block.position
        if not clamp.check_range(present):
            raise RuntimeError(f"re-energise: present position outside 0..4095: {present.tolist()}")
        _check_start_pose(cfg, fk, present)
        _check_temperatures(cfg, block)
        if not bus.write_goal(present.copy()):
            raise RuntimeError("re-energise: write_goal(present) failed")
        if tick_hook is not None:
            tick_hook(True)
        return True

    wait_until("a successful block read", first_good_tick, pre_deadline)
    last_good_ns = clock.monotonic_ns()
    present = block.position.copy()
    for (name, _), motor_id, want in zip(motors, ids, present, strict=True):
        got = bus.read_reg("Goal_Position", motor_id)
        if got != int(want):
            raise RuntimeError(f"re-energise: Goal_Position of {name} reads {got}, expected present {int(want)}")

    # 5. only now
    try:
        if not bus.set_torque(True):
            raise RuntimeError("re-energise: set_torque(True) failed")
        # 6. ramp home through the transit zone, same step and tracking clamp as the safe-state ramp
        _ramp_home(bus, cfg, clock, block, present, timeout_s, fk, tick_hook, last_good_ns, gap_ns)
    except BaseException as exc:
        if not torque_off_retry(bus):
            exc.add_note("re-energise: torque-off after the failure ALSO failed; the deadman is the next layer")
        raise


def torque_off_retry(bus: Bus, ids: list[int] | None = None, attempts: int = _TORQUE_OFF_ATTEMPTS) -> bool:
    """``set_torque(False)`` up to ``attempts`` times; True once one is acknowledged. Never raises (a dead bus must
    not mask the error that brought us here)."""
    for _ in range(attempts):
        try:
            if bus.set_torque(False, ids):
                return True
        except Exception:  # noqa: BLE001, S112 - keep trying; the caller reports the failure
            continue
    return False


def _write_registers_verified(bus: Bus, cfg: So101Config, ids: list[int]) -> None:
    for motor_id in ids:
        bus.write_reg("Lock", motor_id, 0)
    for motor_id in ids:
        if bus.read_reg("Torque_Enable", motor_id) != 0:
            raise RuntimeError(f"re-energise: motor {motor_id} has torque on before the register writes")
    fields = [(register_name(f), int(v)) for f, v in asdict(cfg.registers).items()]
    for name, value in fields:
        for motor_id in ids:
            bus.write_reg(name, motor_id, value)
    for name, value in fields:
        for motor_id in ids:
            got = bus.read_reg(name, motor_id)
            if got != value:
                raise RuntimeError(f"re-energise: {name} of motor {motor_id} reads {got}, wrote {value}")


def _check_start_pose(cfg: So101Config, fk: So101FK, present: np.ndarray) -> None:
    rest = np.asarray(cfg.safe_state.rest_ticks, dtype=np.int64)
    near_rest = bool(
        (np.abs(present[:_N_ARM].astype(np.int64) - rest[:_N_ARM]) <= cfg.safe_state.rest_tolerance_ticks).all()
    )
    if near_rest:
        return
    xyz = fk.gripper_xyz_from_ticks(present, np.zeros(3))
    if fk.in_box(xyz, cfg.box):
        return
    raise UnsafeStartPose(
        f"re-energise refused: present {present.tolist()} (tip {np.round(xyz, 3).tolist()} m) is neither within "
        f"{cfg.safe_state.rest_tolerance_ticks} ticks of rest {rest.tolist()} nor inside the box"
    )


def _check_temperatures(cfg: So101Config, block: BlockRead) -> None:
    t = block.temperature_c
    if bool(np.isin(t, _SENSOR_DEAD).any()) or int(t.max()) >= cfg.thermal.trip_c:
        raise RuntimeError(f"re-energise refused: temperatures {t.tolist()} C (trip {cfg.thermal.trip_c} C)")
    if int(block.status.max()) != 0:
        raise RuntimeError(f"re-energise refused: firmware protection status {block.status.tolist()}")


def _ramp_home(
    bus: Bus,
    cfg: So101Config,
    clock: Clock,
    block: BlockRead,
    present: np.ndarray,
    timeout_s: float,
    fk: So101FK,
    tick_hook: TickHook | None,
    last_good_ns: int,
    gap_ns: int,
) -> None:
    home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)
    planner = TransitPlanner(fk, cfg.safe_state)
    step = planner.step_ticks
    period_ns = int(_NS / cfg.rates.servo_hz)
    target = present.astype(np.int32)
    nxt = np.empty(N_JOINTS, dtype=np.int32)
    lim = np.empty(N_JOINTS, dtype=np.int32)
    clamped = np.empty(N_JOINTS, dtype=np.int32)
    mask = np.empty(N_JOINTS, dtype=bool)
    worst = int(np.abs(home - present).max())
    budget_s = timeout_s + _RAMP_SLACK_S + (worst / max(step, 1)) / cfg.rates.servo_hz
    deadline = clock.monotonic_ns() + int(budget_s * _NS)
    next_tick = clock.monotonic_ns()
    while True:
        now = clock.monotonic_ns()
        if now >= deadline:
            raise TimeoutError("re-energise: ramp home did not complete in time")
        if tick_hook is not None and now - last_good_ns >= gap_ns:
            raise TimeoutError("re-energise: no successful read+write within the heartbeat gap")
        got = bool(bus.read_block(block)) and block.ok
        if got:
            _check_temperatures(cfg, block)  # torque is on now: a hot joint or a protection bit ends the ramp
        if got and int(np.abs(block.position - home).max()) <= SETTLE_TICKS and bool((target == home).all()):
            return
        if got:
            planner.next_target(target, home, nxt)
            clamp.clamp_joint_limits(nxt, cfg.joints, lim, mask)
            clamp.clamp_tracking(lim, block.position, cfg.action.tracking_error_ticks, clamped, mask)
            if planner.acceptable(target, clamped):
                target[...] = clamped
        if not clamp.check_range(target):
            raise RuntimeError("re-energise: ramp target out of range")
        if not bus.write_goal(target):
            raise RuntimeError("re-energise: write_goal failed during ramp home")
        if got:
            last_good_ns = clock.monotonic_ns()
        if tick_hook is not None:
            tick_hook(got)
        next_tick += period_ns
        clock.sleep_until(next_tick)
