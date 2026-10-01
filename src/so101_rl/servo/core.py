"""Servo core: one 30 Hz tick of the safety-critical loop (plan F6; ARCHITECTURE sections 3, 5 and 6).

``ServoCore.tick()`` is the whole per-tick contract. The same code drives ``FakeBus``, ``MujocoBus`` and
``FeetechBus``; only the ``Bus`` and the ``Clock`` differ.

Tick order
----------
1. ``bus.read_block`` (one 11-byte sync read). A failed read never becomes ``present = 0``: nothing is written
   and no heartbeat is sent on that tick.
2. ``ipc.read_cmd`` with dedupe by ``seq``: a command is integrated at most once (the 10 Hz delta is spread over
   ``interp_ticks`` servo ticks by the pipeline, never re-added).
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
   TORQUE_OFF        torque off (idempotent) and goal = present, so the MCU stays armed during the thermal dwell
                     and a re-energise never snaps to a stale goal
   CHECK             torque off and goal = present; on ``reenergize_requested`` the FULL ``reenergize`` runs right
                     after this tick's heartbeat (heartbeat immediately before, review obligation)
   LATCHED           torque off every tick; no goal writes and NO heartbeat: the MCU drops the relay (the latch
                     needs a human and the physical reset anyway)
   ================  =============================================================================================

6. Heartbeat: sent only when this tick's read succeeded AND its writes were accepted (``hb_sent``). Its result
   feeds ``SmInputs.hb_ok`` of the next step. ``max_bus_failures`` consecutive good-read ticks whose writes were
   refused add ``BUS_FAIL`` to the next step's trips (the machine itself counts only failed reads), so a bus that
   reads but no longer writes latches even without an MCU link.
7. ``ipc.write_state`` every tick, also in fault states; then one tick-log record.

Start-up (the MCU boots LATCHED)
--------------------------------
``start()`` models start-up as BOOT -> wait for the physical button + ARM -> heartbeat -> ``reenergize`` -> RUN.
While waiting, ``McuStatus.latched`` is shown to the operator but NEVER fed into the state machine (LATCHED is
absorbing; feeding the boot latch would force a new process). After ARM the MCU state reaches the machine only
through ``hb_ok`` (``HeartbeatLink.send`` returns False once the MCU is latched).

Re-energise (boot and the automatic thermal resume) runs with ``tick_hook``: one heartbeat per good tick, and a
heartbeat that fails during re-energise aborts it (``reenergize`` then turns torque off). Afterwards
``mark_energized()`` and the stale-command counters are reset, so a stale count from the dwell cannot turn the
thermal resume into a STALE_CMD stop. A failure calls ``notify_reenergize_failed()``.

Shutdown
--------
``shutdown(reason)`` is idempotent. If the arm may be energised (RUN, HOLD, SAFE_STOPPING) it steps the machine
with ``stop_requested`` and runs ``SafeStateSequencer.run(bus, block, servo_hz=..., tick_hook=hb)`` (heartbeat
continues, bounded by ``bound_s``), then torque off on every motor, then ``disconnect``. Every process exit
(SIGINT, SIGTERM, exception, ``--max-ticks``) goes through it (``servo/process.py``).

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
from so101_rl.safety.reenergize import reenergize, torque_off_retry
from so101_rl.safety.safe_state import SafeStateSequencer
from so101_rl.safety.state_machine import SafetyStateMachine, SmInputs, SmOutputs
from so101_rl.safety.stuck import StuckActionMonitor
from so101_rl.safety.thermal import ThermalMonitor
from so101_rl.safety.types import SmState, TripCause, Verdict
from so101_rl.servo.bus import BlockRead, Bus
from so101_rl.servo.heartbeat import HeartbeatLink, McuStatus
from so101_rl.servo.shm import CmdMode, ServoIpc, new_cmd, new_state
from so101_rl.servo.timing import Clock

__all__ = [
    "CLAMP_CAUSES",
    "TICK_LOG_DTYPE",
    "HeartbeatLost",
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
_OBS_RING = 64  # state seq -> publish time, for the obs -> command latency
BOOT_HEARTBEAT_WINDOW_S = 0.5  # after ARM: first good read + goal write (and its heartbeat) must come within this

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
    """See the module docstring. Single-threaded: every method runs on the servo loop's thread."""

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
        self._log = log if log is not None else (lambda msg: print(f"servo: {msg}", file=sys.stderr, flush=True))
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
        self._hb_seq = 0
        self._last_hb_ok = True
        self._armed = False  # MCU ARM succeeded (only then is the link used)
        self._last_cmd_seq = 0
        self._rx_ns = clock.monotonic_ns()
        self._has_delta = False
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

    def consecutive_misses(self) -> int:
        return int((self._clock.monotonic_ns() - self._rx_ns) // self._policy_period_ns)

    # ------------------------------------------------------------------------------------------------------
    # start-up
    def start(self, *, should_abort: Callable[[], bool] | None = None, arm_timeout_s: float | None = None) -> None:
        """BOOT -> wait for the physical button + ARM -> heartbeat -> ``reenergize`` -> RUN.

        Raises :class:`StartupError` (torque off) if the MCU never arms within ``arm_timeout_s`` (None: wait
        forever), if ``should_abort()`` becomes true, or if re-energise fails."""
        abort = should_abort if should_abort is not None else (lambda: False)
        self._bus.connect()
        if self._hb is not None:
            self._wait_for_arm(abort, arm_timeout_s)
        # The heartbeat immediately before reenergize (heartbeat_gap_s is measured from its entry). Right after ARM
        # the servos may still be booting, so retry tick by tick for at most BOOT_HEARTBEAT_WINDOW_S (well inside
        # the MCU's 1 s timeout, which the ARM itself restarted).
        t_arm = self._clock.monotonic_ns()
        nxt = t_arm
        while not self._pre_reenergize_tick(PHASE_BOOT):
            if abort():
                raise StartupError("start-up aborted before re-energise")
            if self._clock.monotonic_ns() - t_arm >= int(BOOT_HEARTBEAT_WINDOW_S * _NS):
                torque_off_retry(self._bus)
                raise StartupError(
                    f"no good read + goal write within {BOOT_HEARTBEAT_WINDOW_S} s after ARM (servos not answering?)"
                )
            nxt += self.period_ns
            self._clock.sleep_until(nxt)
        if abort():
            raise StartupError("start-up aborted before re-energise")
        if not self._run_reenergize():
            raise StartupError(f"start-up re-energise failed: {self.last_error!r}") from self.last_error

    def _wait_for_arm(self, abort: Callable[[], bool], timeout_s: float | None) -> None:
        hb = self._hb
        assert hb is not None
        t0 = self._clock.monotonic_ns()
        nxt = t0
        said = ""
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
            if msg != said:
                self._log(msg)
                said = msg
            if timeout_s is not None and self._clock.monotonic_ns() - t0 > int(timeout_s * _NS):
                raise StartupError(f"watchdog MCU did not arm within {timeout_s} s ({msg})")
            self._boot_idle_tick(st)
            nxt += self.period_ns
            self._clock.sleep_until(nxt)

    def _boot_idle_tick(self, st: McuStatus) -> None:
        t0 = self._clock.monotonic_ns()
        ok = bool(self._bus.read_block(self.block)) and self.block.ok
        self._publish(t0, ok, False, False, TripCause.NONE, PHASE_BOOT, 0, 0)

    def _pre_reenergize_tick(self, phase: int) -> bool:
        """Torque is off here: goal = present, then a heartbeat if the read and the write both succeeded. Returns
        True when that heartbeat was sent and accepted."""
        t0 = self._clock.monotonic_ns()
        ok = bool(self._bus.read_block(self.block)) and self.block.ok
        wrote = False
        hb = False
        if ok and clamp.check_range(self.block.position):
            self._target[...] = self.block.position
            wrote = bool(self._bus.write_goal(self._target))
            if wrote:
                hb = True
                self._last_hb_ok = self._send_heartbeat()
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
            )
        except BaseException as exc:
            self.last_error = exc
            if isinstance(exc, CalibrationMismatch):
                self.calib_mismatches += 1
                self._sm.fault_bits |= TripCause.CALIB_MISMATCH
            if isinstance(exc, HeartbeatLost):
                self._last_hb_ok = False  # the next step latches (HEARTBEAT_FAIL)
            self._sm.notify_reenergize_failed()
            self._log(f"re-energise failed: {exc!r}")
            if not isinstance(exc, Exception):
                raise
            return False
        self._sm.mark_energized()
        self._energized_once = True
        self._applied_since_energize = False
        self._reset_command_counters()
        self._safe_started = False
        self._interp_left = 1
        return True

    def _reenergize_hook(self, ok: bool) -> None:
        t0 = self._clock.monotonic_ns()
        if ok:
            self._last_hb_ok = self._send_heartbeat()
            if not self._last_hb_ok:
                self._publish(t0, ok, True, True, TripCause.NONE, PHASE_REENERGIZE, 0, 0)
                raise HeartbeatLost("heartbeat refused during re-energise (MCU latched or link dead)")
        if ok:
            self._target[...] = self.block.position  # the log shows present here: the ramp target is reenergize's
        self._publish(t0, ok, ok, ok, TripCause.NONE, PHASE_REENERGIZE, 0, 0)

    def _reset_command_counters(self) -> None:
        """After ``mark_energized``: a stale count from before (thermal dwell, boot) must not stop the arm."""
        self._rx_ns = self._clock.monotonic_ns()
        self._has_delta = False

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
        fresh = self._poll_cmd(t0)

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
        inp.hb_ok = self._last_hb_ok
        inp.mcu = None  # McuStatus reaches the machine only through hb_ok (boot latch: see module docstring)
        inp.stop_requested = self._stop_requested
        inp.may_resume_thermal = self._thermal.may_resume()
        sm.step(inp, out)

        if out.state is SmState.RUN and out.use_policy:
            if out.resync_target:
                self._pipe.resync(block.position)
                self._interp_left = 1
            if fresh and self._has_delta:
                self._pipe.new_command(self._c_delta, v)
                self._interp_left = self._interp
            self._pipe.apply(block, self._interp_left, v)
            if self._interp_left > 1:
                self._interp_left -= 1
            trips = v.trips
            self._applied_since_energize = True
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
        else:  # TORQUE_OFF, CHECK, BOOT, IDLE_TORQUE_OFF: torque off, goal = present
            torque_ok = torque_off_retry(self._bus, attempts=1) if out.torque_off_now else True
            if ok and clamp.check_range(block.position):
                self._target[...] = block.position
                wrote = bool(self._bus.write_goal(self._target)) and torque_ok
            if out.reenergize_requested:
                self._reenergize_pending = True

        hb_sent = ok and wrote and state is not SmState.LATCHED
        if hb_sent:
            self._last_hb_ok = self._send_heartbeat()
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
            if hb_sent and self._last_hb_ok:
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
        self._has_delta = mode == CmdMode.RUN
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

    def _send_heartbeat(self) -> bool:
        self._hb_seq = (self._hb_seq + 1) & 0xFF
        self.last_hb_ns = self._clock.monotonic_ns()
        if self._hb is None or not self._armed:
            return True
        return bool(self._hb.send(self._hb_seq))

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
        self._s_applied[...] = self._v.applied_delta_ticks
        self._s_clamped[...] = self._v.clamped_mask
        fault = int(self._sm.fault_bits) | (int(trips) & _SM_CAUSES)
        self._s_fault[0] = fault
        self._s_sm[0] = int(self._sm.state)
        self._s_last_cmd[0] = self._last_cmd_seq
        self._s_hb[0] = hb_sent and self._last_hb_ok
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
            inp.mcu = None
            inp.stop_requested = True
            sm.step(inp, self._out)  # -> SAFE_STOPPING (USER_STOP)
        try:
            self._safe.run(self._bus, self.block, servo_hz=self.cfg.rates.servo_hz, tick_hook=self._shutdown_hook)
        except Exception as exc:  # noqa: BLE001 - torque-off refused 3x: report; the caller still tries torque off
            self.last_error = exc
            self._log(f"shutdown: safe-state sequence failed: {exc!r}")
        sm.notify_safe_done()

    def _shutdown_hook(self, ok: bool) -> None:
        t0 = self._clock.monotonic_ns()
        if ok:
            self._last_hb_ok = self._send_heartbeat()  # lenient: keep going, the sequence is bounded anyway
            self._target[...] = self.block.position
        self._publish(t0, ok, ok, ok, TripCause.NONE, PHASE_SHUTDOWN, 0, 0)


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
