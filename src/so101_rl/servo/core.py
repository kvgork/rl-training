"""Servo core: one 30 Hz tick of the safety-critical loop (plan F6; ARCHITECTURE sections 3, 5 and 6).

``ServoCore.tick()`` is the whole per-tick contract. The same code drives ``FakeBus``, ``MujocoBus`` and
``FeetechBus``; only the ``Bus`` and the ``Clock`` differ.

Tick order
----------
1. ``bus.read_block`` (one 11-byte sync read). A failed read never becomes ``present = 0``: nothing is written
   and no heartbeat is sent on that tick, and the state record says ``read_ok = 0``.
2. ``ipc.read_cmd`` with dedupe by ``seq``: a command is integrated at most once (the 10 Hz delta is spread over
   ``interp_ticks`` servo ticks by the pipeline, never re-added). Only a RUN-mode command is "fresh" for the state
   machine (an IDLE / HOLD command keeps the policy alive but never turns HOLD into RUN), and a RUN delta that
   arrives on a tick without ``use_policy`` stays pending until the next tick that applies (never dropped).
3. Outside RUN, ``pipeline.monitor_only`` is called first, so the thermal monitor sees every tick (thermal dwell,
   stale temperature). Then ``sm.step``.
4. RUN with ``use_policy``: ``resync`` (if requested) -> ``new_command`` (on a new seq) -> ``pipeline.apply``. If the
   apply reports a state-machine cause (anything but the TRACKING / JOINT_LIMIT / BOX clamps) the machine is
   stepped again on the same tick, so a RANGE or NaN target is never written.
5. The state's duty:

   ================  =============================================================================================
   RUN               ``check_range`` -> ``write_goal(target)``
   HOLD              re-write the last target (no extrapolation)
   SAFE_STOPPING     ``SafeStateSequencer.start`` (from present) / ``next`` -> write or torque off; ``done`` ->
                     ``sm.notify_safe_done()``
   TORQUE_OFF        torque off every tick (idempotent, acked: it is this tick's write for the heartbeat, so the
                     MCU stays armed during the thermal dwell). NEVER a goal write: on the STS3215 a Goal_Position
                     write switches torque back on (found on the arm 2026-10-10). Re-energise writes goal = present
                     itself before torque on, so a stale goal is never snapped to.
   CHECK             as TORQUE_OFF; on ``reenergize_requested`` the FULL ``reenergize`` runs right after this
                     tick's heartbeat (heartbeat immediately before, review obligation)
   LATCHED           torque off every tick; no goal writes and NO heartbeat: the MCU drops the relay (the latch
                     needs a human and the physical reset anyway)
   ================  =============================================================================================

6. Heartbeat: sent only when this tick's read succeeded AND its writes were accepted (``hb_sent``; for the
   un-acked ``write_goal`` "accepted" means transmitted). ``HeartbeatLink.send`` is tri-state (``HbSend``): the
   sequence number advances only when the frame was delivered (a lost echo still advances it). A send that is not
   ``ACKED`` is a miss; ``SmInputs.hb_misses`` (consecutive misses) and ``SmInputs.mcu`` (the MCU's last reported
   status, only after ARM) feed the next step, which latches after ``HB_MISS_BUDGET`` (5) misses or at once when
   the MCU reports latched (plan section 10, 2026-10-02). ``max_bus_failures`` consecutive good-read ticks whose
   writes were refused add ``BUS_FAIL`` to the next step's trips (the machine itself counts only failed reads), so
   a bus that reads but no longer writes latches even without an MCU link.
7. ``ipc.write_state`` every tick, also in fault states (``applied_delta_ticks`` / ``clamped_mask`` are zero on
   every tick where ``pipeline.apply`` did not run); then one tick-log record.

Start-up (the MCU boots LATCHED)
--------------------------------
``start()`` models start-up as (U3 review item 1) open the port WITHOUT a handshake -> BOOT, wait for the physical
button + ARM (no bus traffic: the servo side of the relay is unpowered) -> ARM closes the relay -> ping until every
motor answers -> one good read + goal write + ACKED heartbeat, all within ``BOOT_HEARTBEAT_WINDOW_S`` of ARM ->
``reenergize`` -> RUN. While waiting, ``McuStatus.latched`` is shown to the operator (repeated every
``ARM_REMINDER_S``) but NEVER fed into the state machine (LATCHED is absorbing; feeding the boot latch would force
a new process). After ARM the MCU's reported status is fed every tick (``SmInputs.mcu``).

Re-energise (boot and the automatic thermal resume) runs with ``tick_hook``: one heartbeat per good tick, and a
heartbeat that is not ACKED during re-energise aborts it (strict: ``reenergize`` then turns torque off). It also
gets ``should_abort`` (the runner's stop flag or a STOP_REQ), and a thermal resume never starts once a stop was
requested. A stop before ``set_torque(True)`` never energises. A stop during the ramp (plan section 10,
2026-10-02) leaves torque on (``hand_over_on_abort``) and the core hands over AT ONCE to the bounded
``SafeStateSequencer.run(..., tick_hook=...)`` back to rest, heartbeat continuing (the shutdown hook), then torque
off; the attempt then counts as a failed re-energise (start-up raises ``StartupError``). Afterwards ``mark_energized()`` and the stale-command counters are reset (the command block is drained,
so a command written before ARM is not "fresh" on the first RUN tick), so a stale count from the dwell cannot turn
the thermal resume into a STALE_CMD stop. A failure calls ``notify_reenergize_failed()``.

Shutdown
--------
``shutdown(reason)`` is idempotent. If the arm may be energised (RUN, HOLD, SAFE_STOPPING) it steps the machine
with ``stop_requested`` and runs ``SafeStateSequencer.run(bus, block, servo_hz=..., tick_hook=hb)`` (heartbeat
continues, bounded by ``bound_s``; lenient: a missed heartbeat does not stop the ramp, and a frame that was not
delivered does not advance the sequence, so it cannot latch SEQ_ERROR), then torque off on every motor, then
``disconnect``. Every process exit
(SIGINT, SIGTERM, exception, ``--max-ticks``) goes through it (``servo/process.py``).

Escalation: on every tick of a safe-state run (shutdown or the re-energise hand-over) the hook checks
``immediate_stop()`` (the runner: a SECOND stop signal) and the MCU's reported status (latched: relay open, e-stop).
Either raises :class:`ImmediateStop` out of the sequencer and torque goes off on that same tick.

State record and the log convention
-----------------------------------
The state record of tick *k* holds ``present_ticks`` read at the START of tick *k* (before its write) and
``target_ticks`` written in tick *k*; ``applied_delta_ticks`` is the post-clamp delta of the current command
(``Verdict.applied_delta_ticks``, the logged action). For one policy step *i* the episode log row (F8, used by
``sim.compare_real``) therefore pairs ``target_ticks`` / ``applied_delta_ticks`` from the LAST record of step *i*
with ``present_ticks`` from the FIRST record of step *i + 1*: that read is taken after the step-*i* target was
applied for one full policy step. :func:`log_row` does exactly that pairing.

Allocation
----------
Everything the tick touches is preallocated here or in the safety objects. ``tick()`` returns the SAME
:class:`TickResult` object every call (copy fields you want to keep). ``tests/test_servo_core.py`` checks with
``tracemalloc`` that 1000 ticks after warm-up leave no net allocation in the servo and safety code.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from so101_rl.config import JOINT_NAMES, N_JOINTS, MotorCalibration, So101Config
from so101_rl.safety import clamp
from so101_rl.safety.calib_check import CalibrationMismatch
from so101_rl.safety.config_check import validate_geometry
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.pipeline import SafetyPipeline
from so101_rl.safety.reenergize import ReenergizeAborted, reenergize, torque_off_retry
from so101_rl.safety.safe_state import SafeStateSequencer
from so101_rl.safety.state_machine import SafetyStateMachine, SmInputs, SmOutputs
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.thermal import ThermalMonitor
from so101_rl.safety.types import SmState, TripCause, Verdict
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.heartbeat import HbSend, HeartbeatLink, McuStatus
from so101_rl.servo.shm import CmdMode, ServoIpc, new_cmd, new_state
from so101_rl.servo.timing import Clock

__all__ = [
    "CLAMP_CAUSES",
    "TICK_LOG_DTYPE",
    "HeartbeatLost",
    "ImmediateStop",
    "ServoCore",
    "StartupError",
    "TickResult",
    "build_servo_core",
    "log_row",
]

_NS = 1_000_000_000
# Clamps are reported in Verdict.trips but are not state-machine causes (F5).
CLAMP_CAUSES = TripCause.TRACKING | TripCause.JOINT_LIMIT | TripCause.BOX
_SM_CAUSES = (2**32 - 1) & ~int(CLAMP_CAUSES)  # plain int mask
# state seq -> publish time ring, for the obs -> command latency. 64 records = 2.1 s at 30 Hz: a command computed
# from an older observation reports obs_to_cmd = 0 (unknown) instead of a wrong value.
_OBS_RING = 64
# After ARM: every motor must answer a ping and the first good read + goal write + ACKED heartbeat must come within
# this (the MCU's 1 s timeout runs from ARM; the STS3215 boot time behind the relay is a hardware-track measurement).
BOOT_HEARTBEAT_WINDOW_S = 0.8
ARM_REMINDER_S = 10.0  # while waiting for the button + ARM, repeat the operator message this often

# One record per servo tick (and per re-energise / shutdown tick), pushed into the RingWriter tick log.
TICK_LOG_DTYPE = np.dtype(
    [
        ("t_ns", "<i8"),  # servo clock at the start of the tick
        ("tick", "<u8"),  # ServoCore.tick() count (re-energise / shutdown ticks repeat the last value)
        ("jitter_ns", "<i8"),  # start - absolute deadline
        ("duration_ns", "<i8"),  # tick body
        ("phase", "u1"),  # 0 loop tick, 1 re-energise tick, 2 shutdown tick, 3 boot wait
        ("sm_state", "u1"),
        ("ok", "u1"),  # block read succeeded
        ("wrote", "u1"),  # every write of the tick was accepted
        ("hb_sent", "u1"),  # heartbeat attempted (read ok AND write ok)
        ("hb_ok", "u1"),  # its result (True without an MCU link)
        ("trips", "<u4"),
        ("fault_bits", "<u4"),
        ("present", "<i4", (N_JOINTS,)),
        ("target", "<i4", (N_JOINTS,)),
        ("temp_c", "u1", (N_JOINTS,)),
        ("cmd_seq", "<u8"),  # last command seq seen
        ("obs_to_cmd_ns", "<i8"),  # on a new command: apply time - publish time of the state it was computed from
        ("cmd_age_ns", "<i8"),  # policy write time -> servo receipt, on a new command (0 otherwise)
    ],
    align=True,
)
PHASE_LOOP, PHASE_REENERGIZE, PHASE_SHUTDOWN, PHASE_BOOT = 0, 1, 2, 3


class StartupError(RuntimeError):
    """Start-up could not reach RUN (MCU not armed, re-energise failed, aborted). Torque is off."""


class HeartbeatLost(RuntimeError):
    """A heartbeat was refused while the arm was being energised: the MCU is (about to be) latched."""


class ImmediateStop(RuntimeError):
    """Escalation during a safe-state run (second stop signal, MCU latched / e-stop): torque off now, no ramp."""


@dataclass
class TickResult:
    ok: bool = False
    state: SmState = SmState.BOOT
    trips: TripCause = TripCause.NONE
    wrote: bool = False
    hb_sent: bool = False
    overrun_ns: int = 0


def log_row(last_rec_of_step: np.ndarray, first_rec_of_next_step: np.ndarray) -> dict[str, np.ndarray]:
    """The F8 log convention from two STATE_DTYPE records (see the module docstring): the applied target and
    post-clamp delta of policy step *i* and the present position read after it was applied for one policy step."""
    a = last_rec_of_step.reshape(-1)[0]
    b = first_rec_of_next_step.reshape(-1)[0]
    return {
        "target_ticks": np.array(a["target_ticks"], dtype=np.int32),
        "applied_delta_ticks": np.array(a["applied_delta_ticks"], dtype=np.float32),
        "present_ticks": np.array(b["present_ticks"], dtype=np.int32),
    }


class ServoCore:
    """See the module docstring. Single-threaded: every method runs on the servo loop's thread.

    Build it with :func:`build_servo_core`: the constructor does NOT validate the safe-state geometry
    (``validate_geometry``) or wire the safety objects consistently; it exists for that function and for tests.
    """

    def __init__(
        self,
        cfg: So101Config,
        bus: Bus,
        ipc: ServoIpc,
        heartbeat: HeartbeatLink | None,
        pipeline: SafetyPipeline,
        sm: SafetyStateMachine,
        safe: SafeStateSequencer,
        clock: Clock,
        tick_log=None,
        *,
        calibration: dict[str, MotorCalibration],
        fk: So101FK,
        thermal: ThermalMonitor,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._bus = bus
        self._ipc = ipc
        self._hb = heartbeat
        self._pipe = pipeline
        self._sm = sm
        self._safe = safe
        self._clock = clock
        self._tick_log = tick_log
        self._calibration = calibration
        self._fk = fk
        self._thermal = thermal
        self._log_sink = log if log is not None else (lambda msg: print(f"servo: {msg}", file=sys.stderr, flush=True))
        self._ids = [calibration[n].id for n in JOINT_NAMES]

        self.period_ns = round(_NS / cfg.rates.servo_hz)
        self._policy_period_ns = round(_NS / cfg.rates.policy_hz)
        self._interp = max(1, int(cfg.action.interp_ticks))

        self.block = BlockRead()
        self._v = Verdict()
        self._in = SmInputs()
        self._out = SmOutputs()
        self._res = TickResult()
        self._target = np.zeros(N_JOINTS, dtype=np.int32)  # last target written (or to be written) this tick
        self._safe_target = np.zeros(N_JOINTS, dtype=np.int32)
        self._hold = np.zeros(N_JOINTS, dtype=np.int32)  # HOLD target, latched on entry
        self._last_state = SmState.BOOT
        self._write_failures = 0  # consecutive good-read ticks whose writes were refused
        self._max_bus_failures = int(cfg.rates.max_bus_failures)
        self._present_copy = np.zeros(N_JOINTS, dtype=np.int32)

        # command record and field views (no per-tick allocation of the views)
        self._cmd = new_cmd()
        self._c_seq = self._cmd["seq"]
        self._c_t = self._cmd["t_mono_ns"]
        self._c_state_seq = self._cmd["state_seq"]
        self._c_mode = self._cmd["mode"]
        self._c_delta = self._cmd["delta_ticks"][0]
        # state record and field views
        self._st = new_state()
        s = self._st
        self._s_t, self._s_tick, self._s_overruns = s["t_mono_ns"], s["tick"], s["overruns"]
        self._s_present, self._s_velocity, self._s_load = s["present_ticks"][0], s["velocity"][0], s["load"][0]
        self._s_voltage, self._s_temp = s["voltage_dv"][0], s["temp_c"][0]
        self._s_status, self._s_moving = s["status"][0], s["moving"][0]
        self._s_target, self._s_goal = s["target_ticks"][0], s["goal_ticks"][0]
        self._s_applied, self._s_clamped = s["applied_delta_ticks"][0], s["clamped_mask"][0]
        self._s_fault, self._s_sm, self._s_last_cmd, self._s_hb = s["fault_bits"], s["sm_state"], s["last_cmd_seq"], s["hb_ok"]
        self._s_read_ok = s["read_ok"]
        # tick-log record and field views
        self._lg = np.zeros(1, dtype=TICK_LOG_DTYPE)
        g = self._lg
        self._l_t, self._l_tick, self._l_jit, self._l_dur = g["t_ns"], g["tick"], g["jitter_ns"], g["duration_ns"]
        self._l_phase, self._l_sm, self._l_ok, self._l_wrote = g["phase"], g["sm_state"], g["ok"], g["wrote"]
        self._l_hbs, self._l_hbok, self._l_trips, self._l_fault = g["hb_sent"], g["hb_ok"], g["trips"], g["fault_bits"]
        self._l_present, self._l_target, self._l_temp = g["present"][0], g["target"][0], g["temp_c"][0]
        self._l_cmd, self._l_o2c, self._l_age = g["cmd_seq"], g["obs_to_cmd_ns"], g["cmd_age_ns"]
        # state seq -> publish time ring (obs -> command latency)
        self._obs_seq = np.zeros(_OBS_RING, dtype=np.int64)
        self._obs_t = np.zeros(_OBS_RING, dtype=np.int64)
        self._n_pub = 0

        # counters
        self.tick_count = 0
        self.overruns = 0
        self.calib_mismatches = 0
        self.reenergize_count = 0
        self.last_error: BaseException | None = None
        self._hb_seq = 0  # last sequence number DELIVERED to the MCU
        self._last_hb_ok = True  # the last heartbeat was ACKED (True without an MCU link)
        self._hb_misses = 0  # consecutive heartbeat sends that were not ACKED (normal running; HB_MISS_BUDGET)
        self._hb_hard_fail = False  # a strict-path miss (re-energise): latches on the next step
        self._abort: Callable[[], bool] = _never
        self._immediate: Callable[[], bool] = _never  # escalation: cut torque now (a second stop signal)
        self._armed = False  # MCU ARM succeeded (only then is the link used)
        self._last_cmd_seq = 0
        self._rx_ns = clock.monotonic_ns()
        self._delta_pending = False  # a RUN delta was received and not yet handed to pipeline.new_command
        self._cmd_is_run = False  # mode of the newest command
        self._applied_this_tick = False  # pipeline.apply ran on this tick (else applied delta is published as 0)
        self._interp_left = 1
        self._stop_requested = False
        self._safe_started = False
        self._reenergize_pending = False
        self._energized_once = False
        self._applied_since_energize = False
        self._shutdown_done = False
        self._o2c_ns = 0
        self._age_ns = 0
        self.last_hb_ns = 0

    # ------------------------------------------------------------------------------------------------------
    # properties for tests / the process runner
    @property
    def state(self) -> SmState:
        return self._sm.state

    @property
    def sm(self) -> SafetyStateMachine:
        return self._sm

    @property
    def pipeline(self) -> SafetyPipeline:
        return self._pipe

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def thermal(self) -> ThermalMonitor:
        return self._thermal

    @property
    def hb_misses(self) -> int:
        return self._hb_misses

    def _log(self, msg: str) -> None:
        try:
            self._log_sink(msg)
        except OSError:  # stderr gone (closed terminal): logging must never break the servo loop
            pass

    def consecutive_misses(self) -> int:
        return int((self._clock.monotonic_ns() - self._rx_ns) // self._policy_period_ns)

    # ------------------------------------------------------------------------------------------------------
    # start-up
    def start(
        self,
        *,
        should_abort: Callable[[], bool] | None = None,
        arm_timeout_s: float | None = None,
        immediate_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Open the port (no handshake) -> BOOT, wait for the physical button + ARM -> ping every motor -> good
        read + write + ACKED heartbeat -> ``reenergize`` -> RUN.

        ``should_abort`` is kept for the life of the core: it also aborts a running re-energise and blocks a
        thermal resume; a stop during the ramp hands over to the safe-state sequence (module docstring).
        ``immediate_stop`` (also kept) escalates any safe-state run to torque off now. Raises
        :class:`StartupError` (torque off) if the MCU never arms within ``arm_timeout_s``
        (None: wait forever, with a reminder every ``ARM_REMINDER_S``), if ``should_abort()`` becomes true, if
        the motors do not answer within ``BOOT_HEARTBEAT_WINDOW_S`` of ARM, or if re-energise fails."""
        abort = should_abort if should_abort is not None else _never
        self._abort = abort
        if immediate_stop is not None:
            self._immediate = immediate_stop
        self._bus.connect()  # opens the port only: before ARM the servo side is unpowered (review item 1)
        if self._hb is not None:
            self._wait_for_arm(abort, arm_timeout_s)
        # Right after ARM the servos are still booting behind the relay: ping tick by tick until every motor
        # answers, then the heartbeat immediately before reenergize (heartbeat_gap_s is measured from its entry).
        # Both within BOOT_HEARTBEAT_WINDOW_S, inside the MCU's 1 s timeout which the ARM itself restarted.
        t_arm = self._clock.monotonic_ns()
        nxt = t_arm
        answered = False
        while True:
            if not answered:
                answered = self._motors_answer()
            if answered and self._pre_reenergize_tick(PHASE_BOOT):
                break
            if not answered:
                self._publish(self._clock.monotonic_ns(), False, False, False, TripCause.NONE, PHASE_BOOT, 0, 0)
            if abort():
                raise StartupError("start-up aborted before re-energise")
            st = self._hb.last_status if (self._hb is not None and self._armed) else None
            if st is not None and st.latched:
                torque_off_retry(self._bus)
                raise StartupError(f"watchdog MCU latched again right after ARM (reason {st.reason})")
            if self._clock.monotonic_ns() - t_arm >= int(BOOT_HEARTBEAT_WINDOW_S * _NS):
                torque_off_retry(self._bus)
                what = "a good read + torque-off + heartbeat" if answered else "every motor to answer a ping"
                raise StartupError(f"no {what} within {BOOT_HEARTBEAT_WINDOW_S} s after ARM (servos not answering?)")
            nxt += self.period_ns
            self._clock.sleep_until(nxt)
        if abort():
            raise StartupError("start-up aborted before re-energise")
        if not self._run_reenergize():
            raise StartupError(f"start-up re-energise failed: {self.last_error!r}") from self.last_error

    def _motors_answer(self) -> bool:
        answered = self._bus.ping_all()
        return all(answered.get(i, False) for i in self._ids)

    def _wait_for_arm(self, abort: Callable[[], bool], timeout_s: float | None) -> None:
        hb = self._hb
        assert hb is not None
        t0 = self._clock.monotonic_ns()
        nxt = t0
        said = ""
        said_ns = t0
        while True:
            if abort():
                raise StartupError("start-up aborted while waiting for the watchdog MCU")
            st = hb.status()
            if not st.latched and not st.armed:  # DISARMED: the button was pressed
                st = hb.arm()
                if st.armed and not st.latched:
                    self._armed = True
                    self._log(f"watchdog MCU armed (status {st.to_byte():#04x})")
                    return
                msg = f"ARM refused (status {st.to_byte():#04x}: undervolt={st.undervolt}); retrying"
            elif st.latched:
                # boot latch (reason 6) or a trip: shown to the operator, never fed to the state machine
                msg = f"watchdog MCU latched (reason {st.reason}): press the physical reset button for 0.5 s"
            else:
                msg = "watchdog MCU already ARMED by someone else: waiting for it to time out and latch"
            now = self._clock.monotonic_ns()
            if msg != said or now - said_ns >= int(ARM_REMINDER_S * _NS):
                self._log(msg if msg != said else f"{msg} (still waiting after {(now - t0) / _NS:.0f} s)")
                said, said_ns = msg, now
            if timeout_s is not None and self._clock.monotonic_ns() - t0 > int(timeout_s * _NS):
                raise StartupError(f"watchdog MCU did not arm within {timeout_s} s ({msg})")
            self._boot_idle_tick(st)
            nxt += self.period_ns
            self._clock.sleep_until(nxt)

    def _boot_idle_tick(self, st: McuStatus) -> None:
        """No bus traffic before ARM (the servo side is unpowered): publish a record with ``read_ok = 0``."""
        self._publish(self._clock.monotonic_ns(), False, False, False, TripCause.NONE, PHASE_BOOT, 0, 0)

    def _pre_reenergize_tick(self, phase: int) -> bool:
        """Torque is off here and stays off: an acked torque-off (never a goal write: on the STS3215 that switches
        torque on, and re-energise refuses a motor with torque on), then a heartbeat if the read and the torque-off
        both succeeded. Returns True when that heartbeat was sent and accepted."""
        t0 = self._clock.monotonic_ns()
        ok = bool(self._bus.read_block(self.block)) and self.block.ok
        wrote = False
        hb = False
        if ok and clamp.check_range(self.block.position):
            self._target[...] = self.block.position
            wrote = torque_off_retry(self._bus, attempts=1)
            if wrote:
                hb = True
                self._last_hb_ok = self._send_heartbeat() == HbSend.ACKED  # strict: only an ACK counts
        self._publish(t0, ok, wrote, hb, TripCause.NONE, phase, 0, 0)
        return hb and self._last_hb_ok

    def _run_reenergize(self) -> bool:
        """Full ``reenergize`` with the heartbeat hook. True on success (``mark_energized`` + counter reset)."""
        self.reenergize_count += 1
        try:
            reenergize(
                self._bus,
                self.cfg,
                self._calibration,
                self._clock,
                self.block,
                tick_hook=self._reenergize_hook,
                fk=self._fk,
                should_abort=self._abort_requested,
                hand_over_on_abort=True,
            )
        except BaseException as exc:
            self.last_error = exc
            if isinstance(exc, ReenergizeAborted) and exc.torque_on:
                self._abort_handover()  # first: torque is on and the heartbeat must continue
            if isinstance(exc, CalibrationMismatch):
                self.calib_mismatches += 1
                self._sm.fault_bits |= TripCause.CALIB_MISMATCH
            if isinstance(exc, HeartbeatLost):
                self._last_hb_ok = False
                self._hb_hard_fail = True  # strict path: the next step latches (HEARTBEAT_FAIL)
            if isinstance(exc, ReenergizeAborted):
                self._log("re-energise aborted: stop requested")
            self._sm.notify_reenergize_failed()
            self._log(f"re-energise failed: {exc!r}")
            if not isinstance(exc, Exception):
                raise
            return False
        self._sm.mark_energized()
        self._energized_once = True
        self._applied_since_energize = False
        self._reset_command_counters()
        self._hb_misses = 0  # every re-energise tick was ACKED
        self._interp_left = 1
        return True

    def _abort_handover(self) -> None:
        """Stop during the re-energise ramp (plan section 10, 2026-10-02): torque is on. Bounded safe-state sequence
        from present back to rest with the heartbeat hook, then torque off (also after an escalation or a
        failure)."""
        self._log("re-energise: stop during the ramp, handing over to the safe-state sequence")
        try:
            self._safe.run(self._bus, self.block, servo_hz=self.cfg.rates.servo_hz, tick_hook=self._shutdown_hook)
        except Exception as exc:  # noqa: BLE001 - ImmediateStop or torque-off refused: torque off below either way
            self._log(f"re-energise hand-over: {exc!r}")
        finally:
            if not torque_off_retry(self._bus):
                self._log("re-energise hand-over: torque off refused 3 times; the deadman is the next layer")

    def _abort_requested(self) -> bool:
        return self._stop_requested or bool(self._abort())

    def _reenergize_hook(self, ok: bool) -> None:
        t0 = self._clock.monotonic_ns()
        if ok:
            self._last_hb_ok = self._send_heartbeat() == HbSend.ACKED  # strict: one miss aborts re-energise
            if not self._last_hb_ok:
                self._publish(t0, ok, True, True, TripCause.NONE, PHASE_REENERGIZE, 0, 0)
                raise HeartbeatLost("heartbeat refused during re-energise (MCU latched or link dead)")
        if ok:
            self._target[...] = self.block.position  # the log shows present here: the ramp target is reenergize's
        self._publish(t0, ok, ok, ok, TripCause.NONE, PHASE_REENERGIZE, 0, 0)

    def _reset_command_counters(self) -> None:
        """After ``mark_energized``: a stale count from before (thermal dwell, boot) must not stop the arm, and a
        command written before (while waiting for ARM, during the dwell) must not look fresh on the first RUN tick:
        the command block is drained (its seq becomes the last seen; a STOP_REQ in it still counts)."""
        if self._ipc.read_cmd(self._cmd):
            seq = int(self._c_seq[0])
            if seq != self._last_cmd_seq:
                self._last_cmd_seq = seq
                if int(self._c_mode[0]) == CmdMode.STOP_REQ:
                    self._stop_requested = True
        self._rx_ns = self._clock.monotonic_ns()
        self._delta_pending = False

    # ------------------------------------------------------------------------------------------------------
    # the tick
    def tick(self, deadline_ns: int | None = None) -> TickResult:
        if self._shutdown_done:
            raise RuntimeError("ServoCore.tick() after shutdown()")
        clock = self._clock
        t0 = clock.monotonic_ns()
        if deadline_ns is None:
            deadline_ns = t0
        block = self.block
        v = self._v
        sm = self._sm
        out = self._out
        inp = self._in

        ok = bool(self._bus.read_block(block)) and block.ok
        new_seq = self._poll_cmd(t0)
        fresh = new_seq and self._cmd_is_run  # IDLE / HOLD commands never resume RUN (review item 8c)
        self._applied_this_tick = False

        monitored = False
        trips = TripCause.NONE
        if sm.state is not SmState.RUN:
            trips = self._pipe.monitor_only(block, v).trips
            monitored = True
        age = t0 - self._rx_ns
        inp.block_ok = ok
        inp.cmd_fresh = fresh
        inp.cmd_age_ns = age
        inp.consecutive_misses = age // self._policy_period_ns
        if self._write_failures >= self._max_bus_failures:
            trips |= TripCause.BUS_FAIL  # consecutive refused writes: the same link fault as failed reads
        inp.trips = trips
        inp.hb_ok = not self._hb_hard_fail
        inp.hb_misses = self._hb_misses
        # the MCU's own status, only after ARM (the boot latch must never reach the machine: module docstring)
        inp.mcu = self._hb.last_status if (self._hb is not None and self._armed) else None
        inp.stop_requested = self._stop_requested
        inp.may_resume_thermal = self._thermal.may_resume()
        sm.step(inp, out)

        if out.state is SmState.RUN and out.use_policy:
            if out.resync_target:
                self._pipe.resync(block.position)
                self._interp_left = 1
            if self._delta_pending:  # also a delta that arrived on an earlier tick without use_policy (8a)
                self._delta_pending = False
                self._pipe.new_command(self._c_delta, v)
                self._interp_left = self._interp
            self._pipe.apply(block, self._interp_left, v)
            if self._interp_left > 1:
                self._interp_left -= 1
            trips = v.trips
            self._applied_since_energize = True
            self._applied_this_tick = True
            if int(trips) & _SM_CAUSES:
                inp.trips = trips
                inp.cmd_fresh = False
                sm.step(inp, out)  # e.g. RANGE / NAN / BOX_PRESENT / THERMAL: never write that target
        elif not monitored:
            trips = self._pipe.monitor_only(block, v).trips

        state = out.state
        wrote = False
        if state is SmState.RUN:
            if out.use_policy and clamp.check_range(v.target_ticks):
                wrote = bool(self._bus.write_goal(v.target_ticks))
                if wrote:
                    self._target[...] = v.target_ticks
        elif state is SmState.HOLD:
            if self._last_state is not SmState.HOLD:
                # latched once on entry: the last RUN target (never extrapolate); before any apply, the pose now
                self._hold[...] = self._pipe.prev_target if self._applied_since_energize else block.position
            if ok and clamp.check_range(self._hold):
                self._target[...] = self._hold
                wrote = bool(self._bus.write_goal(self._target))
        elif state is SmState.SAFE_STOPPING:
            wrote = self._safe_tick(ok)
        elif state is SmState.LATCHED:
            torque_off_retry(self._bus, attempts=1)  # every tick, idempotent; no goal write, no heartbeat
        else:  # TORQUE_OFF, CHECK, BOOT, IDLE_TORQUE_OFF: torque off every tick, never a goal write (STS3215:
            # a Goal_Position write switches torque on); the acked torque-off is the tick's write for the heartbeat
            torque_ok = torque_off_retry(self._bus, attempts=1)
            if ok:
                self._target[...] = block.position
                wrote = torque_ok
            if out.reenergize_requested:
                self._reenergize_pending = True

        if state is not SmState.RUN and state is not SmState.HOLD:
            self._delta_pending = False  # a stop or a fault drops an unapplied delta

        hb_sent = ok and wrote and state is not SmState.LATCHED
        if hb_sent:
            self._last_hb_ok = self._send_heartbeat() == HbSend.ACKED
            self._hb_misses = 0 if self._last_hb_ok else self._hb_misses + 1  # HB_MISS_BUDGET latches (next step)
        if ok and state is not SmState.LATCHED:
            self._write_failures = 0 if wrote else self._write_failures + 1
        self._last_state = state

        self.tick_count += 1
        o2c, age_rx = self._o2c_ns, self._age_ns
        self._o2c_ns = self._age_ns = 0
        self._publish(t0, ok, wrote, hb_sent, trips, PHASE_LOOP, o2c, age_rx, deadline_ns)

        res = self._res
        res.ok = ok
        res.trips = trips
        res.wrote = wrote
        res.hb_sent = hb_sent
        t_end = clock.monotonic_ns()
        over = t_end - (deadline_ns + self.period_ns)
        res.overrun_ns = max(0, over)
        if over > 0:
            self.overruns += 1

        # thermal resume: the FULL reenergize, right after this tick's heartbeat
        if self._reenergize_pending and sm.state is SmState.CHECK:
            # only right after an ACKED heartbeat, and never once a stop was requested (review item 7)
            if hb_sent and self._last_hb_ok and not self._abort_requested():
                self._reenergize_pending = False
                self._run_reenergize()
        elif sm.state is not SmState.CHECK:
            self._reenergize_pending = False
        res.state = sm.state
        return res

    def _poll_cmd(self, now: int) -> bool:
        if not self._ipc.read_cmd(self._cmd):
            return False
        seq = int(self._c_seq[0])
        if seq == self._last_cmd_seq:
            return False  # repeated seq = no new command (dedupe: a delta is integrated once)
        self._last_cmd_seq = seq
        self._rx_ns = now
        mode = int(self._c_mode[0])
        if mode == CmdMode.STOP_REQ:
            self._stop_requested = True
        self._cmd_is_run = mode == CmdMode.RUN
        if self._cmd_is_run:
            self._delta_pending = True  # a newer delta replaces an unapplied older one
        self._age_ns = now - int(self._c_t[0])
        # obs -> command: publish time of the state record the policy computed this command from
        sseq = int(self._c_state_seq[0])
        i = (sseq // 2) % _OBS_RING
        self._o2c_ns = now - int(self._obs_t[i]) if int(self._obs_seq[i]) == sseq and sseq else 0
        return True

    def _safe_tick(self, ok: bool) -> bool:
        safe = self._safe
        if not self._safe_started:
            if not ok:
                return False  # start from a fresh present only; max_bus_failures latches if reads keep failing
            safe.start(self.block.position)
            self._safe_started = True
        step = safe.next(self.block.position, self._safe_target, block_ok=ok)
        goal_ok = torque_ok = True
        attempted = False
        if step.write_target and ok:
            attempted = True
            goal_ok = clamp.check_range(self._safe_target) and bool(self._bus.write_goal(self._safe_target))
            if goal_ok:
                self._target[...] = self._safe_target
        if step.torque_off_ids:
            attempted = True
            torque_ok = torque_off_retry(self._bus, list(step.torque_off_ids))
            if not torque_ok:
                self._log(f"safe state: torque off of {step.torque_off_ids} refused 3 times; deadman is next")
        if step.done:
            self._sm.notify_safe_done()
            self._safe_started = False
        return attempted and goal_ok and torque_ok  # every write of the tick accepted (heartbeat condition)

    def _send_heartbeat(self) -> HbSend:
        """One heartbeat with the next sequence number. The number is consumed only if the frame was delivered
        (``NOT_DELIVERED`` reuses it next time: a gap would latch SEQ_ERROR; a lost echo still consumes it)."""
        self.last_hb_ns = self._clock.monotonic_ns()
        if self._hb is None or not self._armed:
            return HbSend.ACKED
        seq = (self._hb_seq + 1) & 0xFF
        res = self._hb.send(seq)
        if res != HbSend.NOT_DELIVERED:
            self._hb_seq = seq
        return res

    def request_stop(self) -> None:
        """Same as a ``CmdMode.STOP_REQ`` command: USER_STOP -> SAFE_STOPPING -> TORQUE_OFF."""
        self._stop_requested = True

    # ------------------------------------------------------------------------------------------------------
    def _publish(
        self,
        t0: int,
        ok: bool,
        wrote: bool,
        hb_sent: bool,
        trips: TripCause,
        phase: int,
        o2c: int,
        age_rx: int,
        deadline_ns: int | None = None,
    ) -> None:
        b = self.block
        self._s_t[0] = t0
        self._s_tick[0] = self.tick_count
        self._s_overruns[0] = self.overruns
        self._s_present[...] = b.position
        self._s_velocity[...] = b.velocity
        self._s_load[...] = b.load
        self._s_voltage[...] = b.voltage_dv
        self._s_temp[...] = b.temperature_c
        self._s_status[...] = b.status
        self._s_moving[...] = b.moving
        self._s_target[...] = self._target
        self._s_goal[...] = self._pipe.goal
        if self._applied_this_tick:
            self._s_applied[...] = self._v.applied_delta_ticks
            self._s_clamped[...] = self._v.clamped_mask
        else:  # HOLD, SAFE_STOPPING, TORQUE_OFF, ...: no delta was applied this tick (review item 10)
            self._s_applied[...] = 0.0
            self._s_clamped[...] = 0
        fault = int(self._sm.fault_bits) | (int(trips) & _SM_CAUSES)
        self._s_fault[0] = fault
        self._s_sm[0] = int(self._sm.state)
        self._s_last_cmd[0] = self._last_cmd_seq
        self._s_hb[0] = hb_sent and self._last_hb_ok
        self._s_read_ok[0] = ok
        self._ipc.write_state(self._st)
        seq = getattr(self._ipc, "last_state_seq", 0)
        if seq:
            i = (seq // 2) % _OBS_RING
            self._obs_seq[i] = seq
            self._obs_t[i] = t0
        if self._tick_log is not None:
            self._l_t[0] = t0
            self._l_tick[0] = self.tick_count
            self._l_jit[0] = t0 - deadline_ns if deadline_ns is not None else 0
            self._l_dur[0] = self._clock.monotonic_ns() - t0
            self._l_phase[0] = phase
            self._l_sm[0] = int(self._sm.state)
            self._l_ok[0] = ok
            self._l_wrote[0] = wrote
            self._l_hbs[0] = hb_sent
            self._l_hbok[0] = self._last_hb_ok
            self._l_trips[0] = int(trips)
            self._l_fault[0] = fault
            self._l_present[...] = b.position
            self._l_target[...] = self._target
            np.clip(b.temperature_c, 0, 255, out=self._present_copy)
            self._l_temp[...] = self._present_copy
            self._l_cmd[0] = self._last_cmd_seq
            self._l_o2c[0] = o2c
            self._l_age[0] = age_rx
            self._tick_log.push(self._lg[0])

    # ------------------------------------------------------------------------------------------------------
    def shutdown(self, reason: TripCause = TripCause.USER_STOP) -> None:
        """Bounded safe-state sequence (if energised) -> torque off -> disconnect. Idempotent; never raises for a
        bus that is already gone (the deadman is the next layer)."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        sm = self._sm
        sm.fault_bits |= reason
        try:
            if sm.state in (SmState.RUN, SmState.HOLD, SmState.SAFE_STOPPING):
                self._shutdown_sequence()
        except Exception as exc:  # noqa: BLE001 - whatever failed, torque off and disconnect still run below
            self.last_error = exc
            self._log(f"shutdown: {exc!r}")
        finally:
            try:
                if not torque_off_retry(self._bus):
                    self._log("shutdown: final torque off was refused; the deadman is the next layer")
            finally:
                try:
                    self._publish(self._clock.monotonic_ns(), False, False, False, TripCause.NONE, PHASE_SHUTDOWN, 0, 0)
                finally:
                    self._bus.disconnect()

    def _shutdown_sequence(self) -> None:
        """The arm may be energised: USER_STOP -> SAFE_STOPPING, then the blocking, bounded sequencer run."""
        sm = self._sm
        if sm.state is not SmState.SAFE_STOPPING:
            inp = self._in
            inp.block_ok = True
            inp.cmd_fresh = False
            inp.trips = TripCause.NONE
            inp.hb_ok = True
            inp.hb_misses = 0
            inp.mcu = None
            inp.stop_requested = True
            sm.step(inp, self._out)  # -> SAFE_STOPPING (USER_STOP)
        try:
            self._safe.run(self._bus, self.block, servo_hz=self.cfg.rates.servo_hz, tick_hook=self._shutdown_hook)
        except Exception as exc:  # noqa: BLE001 - torque-off refused 3x: report; the caller still tries torque off
            self.last_error = exc
            self._log(f"shutdown: safe-state sequence failed: {exc!r}")
        sm.notify_safe_done()

    def _cut_requested(self) -> bool:
        """Escalation during a safe-state run: a second stop signal, or the MCU reports latched (relay open)."""
        if self._immediate():
            return True
        st = self._hb.last_status if (self._hb is not None and self._armed) else None
        return st is not None and st.latched

    def _shutdown_hook(self, ok: bool) -> None:
        if self._cut_requested():
            self._log("safe state: escalation (second stop / MCU latched): torque off now")
            raise ImmediateStop("immediate stop requested during the safe-state sequence")
        t0 = self._clock.monotonic_ns()
        if ok:
            # lenient: keep going, the sequence is bounded anyway; an undelivered frame does not consume its seq
            self._last_hb_ok = self._send_heartbeat() == HbSend.ACKED
            self._hb_misses = 0 if self._last_hb_ok else self._hb_misses + 1
            self._target[...] = self.block.position
        self._publish(t0, ok, ok, ok, TripCause.NONE, PHASE_SHUTDOWN, 0, 0)


def _never() -> bool:
    return False


def build_servo_core(
    cfg: So101Config,
    calibration: dict[str, MotorCalibration],
    bus: Bus,
    ipc: ServoIpc,
    heartbeat: HeartbeatLink | None,
    clock: Clock,
    *,
    chain: KinematicChain | None = None,
    tick_log=None,
    log: Callable[[str], None] | None = None,
) -> ServoCore:
    """Validate the safe-state geometry (``safety.config_check.validate_geometry``, required before the safety
    stack is built) and wire the whole stack: FK with the calibration, thermal and stuck monitors, pipeline,
    state machine and a sequencer with ``tracking_error_ticks``, ``motor_ids`` and ``fk``."""
    chain = chain if chain is not None else KinematicChain.from_yaml(cfg.kinematics_path)
    validate_geometry(cfg, calibration, chain)  # raises ValueError("safe_state: ...")
    fk = So101FK(chain, calibration)
    thermal = ThermalMonitor(cfg.thermal, clock)
    stuck = StuckActionMonitor()
    pipeline = SafetyPipeline(cfg, fk, thermal, stuck)
    sm = SafetyStateMachine(cfg, clock)
    safe = SafeStateSequencer(
        cfg.safe_state,
        clock,
        tracking_error_ticks=cfg.action.tracking_error_ticks,
        motor_ids={n: calibration[n].id for n in JOINT_NAMES},
        fk=So101FK(chain, calibration),  # own scratch buffers: the sequencer and the pipeline never share an FK
    )
    return ServoCore(
        cfg,
        bus,
        ipc,
        heartbeat,
        pipeline,
        sm,
        safe,
        clock,
        tick_log,
        calibration=calibration,
        fk=fk,
        thermal=thermal,
        log=log,
    )
