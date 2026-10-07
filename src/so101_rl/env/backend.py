"""Env backends (F7 ``EnvBackend``; FS12 of ``plans/sim-track-implementation-plan.md``; Phase 0 Task 8.1).

Both backends speak the servo's shared-memory protocol (``servo/shm.py``: a ``CMD_DTYPE`` record written by the
policy, a ``STATE_DTYPE`` record published by the servo every tick). The env never touches a bus and never
re-implements a safety check: it writes joint DELTAS and reads back what the servo applied.

* :class:`ShmBackend` -- the real-arm backend (S5): attaches to the servo process's segment (``servo`` /
  ``servo --backend fake|mujoco``). ``wait_policy_tick`` keeps an absolute 100 ms schedule, then polls until the
  servo publishes a record of a NEWER tick (bounded); a servo that stops ticking or publishes an old record
  raises :class:`ServoStateStale`. ``reset(q0)`` ramps to ``q0`` through the normal command path: a HOLD
  command, then RUN deltas of at most ``max_ticks_per_tick`` per joint (interpolated by the servo over 3 ticks:
  never a jump), until ``|target - q0| <= 2`` and ``|target - present| < 5`` on every joint with a good bus read
  for 2 consecutive policy ticks. It refuses (:class:`ServoNotRunning`) a servo outside RUN/HOLD: clearing a
  trip needs a fresh servo process or a human, never the policy.
* :class:`SimBackend` -- the parity reference rig (demoted, not the training env): the SAME ``ServoCore`` +
  ``SafetyPipeline`` in-process on a ``MujocoBus`` with ``LocalIpc``, a ``FakeHeartbeatLink`` and a simulated
  clock. Every ``reset`` boots a fresh core exactly like ``tests/test_action_units.py::Rig`` (limp arm at the
  start keyframe, button + ARM, re-energise, RUN), i.e. what a fresh servo process does; one policy step is
  3 x (``core.tick()``, ``bus.step_ticks(1)``). ``sim_params`` (FS7 ``SimParams`` or anything with
  ``apply_cpu(model)``) are applied to the CPU model BEFORE the bus is built, so the bus's saved actuator gains
  (restored on every torque-on) are the fitted ones.

Timing of the observation (both backends alike): a command is applied on the first servo tick after it arrives;
the record the policy reads at the end of the step is that of the 3rd tick, whose ``present_ticks`` was read
after 2 ticks of the new command's targets (the record publishes the reading taken at the START of its tick).
The F8 log pairing (``servo.core.log_row``) is separate and unchanged.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Self, runtime_checkable

import numpy as np

from so101_rl.config import N_JOINTS, MotorCalibration, So101Config, load_calibration
from so101_rl.safety.fk import KinematicChain, So101FK
from so101_rl.safety.types import SmState
from so101_rl.servo.shm import CmdMode, LocalIpc, ServoShm, new_cmd, new_state
from so101_rl.servo.timing import Clock, RealClock

if TYPE_CHECKING:
    import mujoco

    from so101_rl.servo.core import ServoCore
    from so101_rl.sim.mujoco_bus import MujocoBus

__all__ = [
    "EnvBackend",
    "ResetFailed",
    "ServoNotRunning",
    "ServoStateStale",
    "ShmBackend",
    "SimBackend",
    "SimTickClock",
]

_NS: Final[int] = 1_000_000_000
RESET_TARGET_TOL_TICKS: Final[int] = 2  # |target - q0| at the end of a reset ramp
RESET_TRACK_TOL_TICKS: Final[int] = 5  # |target - present| at the end of a reset ramp (plan: "< 5 ticks")
RESET_SETTLED_STEPS: Final[int] = 2  # consecutive policy ticks inside both tolerances
RUNNABLE: Final[frozenset[int]] = frozenset({int(SmState.RUN), int(SmState.HOLD)})


class ServoStateStale(RuntimeError):
    """No state record of a newer servo tick arrived within the poll bound, or the newest record is too old (the
    servo stalled or died). Not a safety trip: the servo's own deadline / heartbeat logic handles the arm."""


class ServoNotRunning(RuntimeError):
    """The servo is not in RUN or HOLD (a trip, a safe stop, torque off, latched): a fresh servo process or a human
    must resume it. The env never tries to clear it."""


class ResetFailed(RuntimeError):
    """The reset ramp did not converge in time, or a trip happened during it."""


@runtime_checkable
class EnvBackend(Protocol):
    """F7, unchanged."""

    def reset(self, q0_ticks: np.ndarray | None) -> None: ...

    def send_delta(self, delta_ticks: np.ndarray, state_seq: int, episode: int) -> int: ...

    def read_state(self, out: np.ndarray) -> bool: ...

    def wait_policy_tick(self) -> None: ...

    def fk(self) -> So101FK: ...


def _write_cmd(ipc: Any, rec: np.ndarray, now_ns: int, mode: CmdMode, delta: np.ndarray, state_seq: int,
               episode: int) -> None:

    rec["t_mono_ns"] = now_ns
    rec["state_seq"] = state_seq
    rec["mode"] = int(mode)
    rec["episode"] = episode
    rec["delta_ticks"][0] = np.asarray(delta, dtype=np.float32)  # NaN passes: the servo sanitizes and trips
    ipc.write_cmd(rec)


# ================================================================================================ ShmBackend


class ShmBackend:
    """Policy side of the servo shm segment (real arm, or ``servo --backend fake|mujoco``).

    ``cfg.shm_name`` names the segment (``shm_name=`` overrides). ``calibration`` defaults to
    ``load_calibration(cfg.calibration_path)``; it only feeds the FK (tip position from present ticks).
    """

    def __init__(
        self,
        cfg: So101Config,
        clock: Clock = RealClock(),  # noqa: B008  (F7 signature; RealClock is stateless)
        *,
        shm_name: str | None = None,
        calibration: dict[str, MotorCalibration] | None = None,
        attach_timeout_s: float = 10.0,
        poll_timeout_ticks: float = 2.0,
        max_state_age_ticks: float = 3.0,
        poll_interval_s: float = 0.001,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.name = shm_name if shm_name is not None else cfg.shm_name
        cal = calibration if calibration is not None else load_calibration(cfg.calibration_path)
        self._fk = So101FK(KinematicChain.from_yaml(cfg.kinematics_path), cal)
        self.policy_period_ns = round(_NS / cfg.rates.policy_hz)
        self.servo_period_ns = round(_NS / cfg.rates.servo_hz)
        self.poll_timeout_ns = round(poll_timeout_ticks * self.servo_period_ns)
        self.max_state_age_ns = round(max_state_age_ticks * self.servo_period_ns)
        self.poll_interval_ns = max(1, round(poll_interval_s * _NS))
        self.shm = self._attach(attach_timeout_s)
        self._cmd = new_cmd()
        self._st = new_state()
        self._accepted = new_state()  # the record wait_policy_tick accepted (the step's observation)
        self._next_boundary: int | None = None
        self._last_tick = -1  # servo tick of the last record wait_policy_tick accepted
        self.overruns = 0  # policy periods skipped because the caller was late
        self.tick_gaps: list[int] = []  # servo ticks between consecutive accepted records (diagnostic)
        self.last_cmd_seq = 0
        self.episode = 0

    def _attach(self, timeout_s: float) -> ServoShm:
        t_end = time.monotonic() + timeout_s
        while True:
            try:
                return ServoShm.attach(self.name)
            except (FileNotFoundError, RuntimeError):
                if time.monotonic() >= t_end:
                    raise
                time.sleep(0.02)

    def close(self) -> None:
        self.shm.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- F7 verbs ---------------------------------------------------------------------------------------
    def fk(self) -> So101FK:
        return self._fk

    def read_state(self, out: np.ndarray) -> bool:
        """Latest STATE record (seqlock read, CRC-checked, bounded retries). False = torn / never written."""
        return self.shm.read_state(out, retries=8)

    def accepted_state(self, out: np.ndarray) -> bool:
        """The record the last :meth:`wait_policy_tick` accepted (age- and newer-tick-checked): the step's
        observation, so the env never reads a later, unchecked record. False before the first wait."""
        if self._last_tick < 0:
            return False
        out[...] = self._accepted
        return True

    def send_delta(self, delta_ticks: np.ndarray, state_seq: int, episode: int) -> int:
        """RUN command; returns its seq (the value the servo echoes in ``last_cmd_seq``)."""
        return self._send(CmdMode.RUN, delta_ticks, state_seq, episode)

    def send_hold(self, state_seq: int = 0, episode: int = 0) -> int:
        return self._send(CmdMode.HOLD, np.zeros(N_JOINTS), state_seq, episode)

    def _send(self, mode: CmdMode, delta: np.ndarray, state_seq: int, episode: int) -> int:
        _write_cmd(self.shm, self._cmd, self.clock.monotonic_ns(), mode, delta, state_seq, episode)
        self.last_cmd_seq = self.shm.last_cmd_seq
        return self.last_cmd_seq

    def wait_policy_tick(self) -> None:
        """Sleep to the next point of an absolute 100 ms grid (a late caller skips missed points, counted in
        ``overruns``, instead of bursting), then poll until a record of a newer servo tick is published.

        Raises :class:`ServoStateStale` if none arrives within ``poll_timeout_ticks`` servo periods after the grid
        point, or if the newest record is older than ``max_state_age_ticks`` periods."""
        clock = self.clock
        now = clock.monotonic_ns()
        if self._next_boundary is None:
            self._next_boundary = now + self.policy_period_ns
        else:
            self._next_boundary += self.policy_period_ns
            if self._next_boundary <= now:  # the caller was late by a full period or more
                missed = (now - self._next_boundary) // self.policy_period_ns + 1
                self.overruns += int(missed)
                self._next_boundary += int(missed) * self.policy_period_ns
        clock.sleep_until(self._next_boundary)
        deadline = self._next_boundary + self.poll_timeout_ns
        st = self._st
        while True:
            if self.read_state(st):
                tick = int(st["tick"][0])
                if tick < self._last_tick:
                    raise ServoStateStale(
                        f"servo tick went back from {self._last_tick} to {tick}: the servo restarted; "
                        "build a new ShmBackend (the old segment is gone)"
                    )
                if tick > self._last_tick:
                    age = clock.monotonic_ns() - int(st["t_mono_ns"][0])
                    if age > self.max_state_age_ns:
                        raise ServoStateStale(f"servo state of tick {tick} is {age / 1e6:.0f} ms old")
                    if self._last_tick >= 0:
                        self.tick_gaps.append(tick - self._last_tick)
                    self._last_tick = tick
                    self._accepted[...] = st
                    return
            if clock.monotonic_ns() >= deadline:
                raise ServoStateStale(
                    f"no servo state newer than tick {self._last_tick} within "
                    f"{self.poll_timeout_ns / 1e6:.0f} ms of the policy tick (servo stalled or gone)"
                )
            clock.sleep_until(clock.monotonic_ns() + self.poll_interval_ns)  # a FakeClock advances, never hangs

    def reset(self, q0_ticks: np.ndarray | None) -> None:
        """HOLD, then a ramp of RUN deltas (<= ``max_ticks_per_tick`` per joint per command) toward ``q0_ticks``.

        ``None``: HOLD and resume RUN where the arm is (zero delta). Raises :class:`ServoNotRunning` if the servo
        is not in RUN/HOLD, :class:`ResetFailed` on a trip during the ramp or if it does not converge in time."""
        self.episode += 1
        st = self._st
        self.send_hold(0, self.episode)  # first thing: a fresh command (HOLD never moves the arm)
        self._next_boundary = None  # a reset restarts the policy grid
        self.wait_policy_tick()
        self._check_runnable(st, "before the reset")
        self.send_hold(int(st["seq"][0]), self.episode)
        self.wait_policy_tick()
        self._check_runnable(st, "after HOLD")
        step = self.cfg.action.max_ticks_per_tick.astype(np.int64)
        if q0_ticks is None:
            self.send_delta(np.zeros(N_JOINTS), int(st["seq"][0]), self.episode)
            self.wait_policy_tick()
            self._check_runnable(st, "after resuming RUN", trip_is_fatal=True)
            return
        q0 = np.asarray(q0_ticks, dtype=np.int64)
        if q0.shape != (N_JOINTS,):
            raise ValueError(f"q0_ticks must have shape ({N_JOINTS},)")
        lo, hi = self.cfg.joints.min_ticks.astype(np.int64), self.cfg.joints.max_ticks.astype(np.int64)
        if np.any(q0 < lo) or np.any(q0 > hi):
            raise ValueError(f"q0_ticks {q0.tolist()} outside the joint limits")
        goal0 = st["goal_ticks"][0].astype(np.int64)
        dist = int(np.max(np.abs(q0 - goal0) / np.maximum(step, 1)))
        max_steps = 2 * dist + 30  # 2x the ramp length plus settling (3 s)
        settled = 0
        for _ in range(max_steps):
            goal = st["goal_ticks"][0].astype(np.int64)
            delta = np.clip(q0 - goal, -step, step)
            self.send_delta(delta.astype(np.float32), int(st["seq"][0]), self.episode)
            self.wait_policy_tick()
            self._check_runnable(st, "during the reset ramp", trip_is_fatal=True)
            target = st["target_ticks"][0].astype(np.int64)
            present = st["present_ticks"][0].astype(np.int64)
            ok = (
                bool(st["read_ok"][0])
                and int(st["sm_state"][0]) == int(SmState.RUN)
                and np.all(np.abs(target - q0) <= RESET_TARGET_TOL_TICKS)
                and np.all(np.abs(target - present) < RESET_TRACK_TOL_TICKS)
            )
            settled = settled + 1 if ok else 0
            if settled >= RESET_SETTLED_STEPS:
                return
        raise ResetFailed(
            f"reset did not converge to {q0.tolist()} in {max_steps} policy steps (target "
            f"{st['target_ticks'][0].tolist()}, present {st['present_ticks'][0].tolist()})"
        )

    @staticmethod
    def _check_runnable(st: np.ndarray, when: str, *, trip_is_fatal: bool = False) -> None:
        sm = int(st["sm_state"][0])
        if sm not in RUNNABLE:
            raise ServoNotRunning(
                f"servo is in {SmState(sm).name} {when} (fault_bits {int(st['fault_bits'][0]):#x}): "
                "a fresh servo process or a human must resume it"
            )
        if trip_is_fatal and int(st["fault_bits"][0]):
            raise ResetFailed(f"servo reported fault_bits {int(st['fault_bits'][0]):#x} {when}")


# ================================================================================================ SimBackend


class SimTickClock:
    """Simulated clock for :class:`SimBackend`: time is ``start + ticks_stepped * period``; ``sleep_until`` steps
    the MuJoCo physics by the servo ticks that elapse (``tests/test_action_units.py::SimClock``, but monotonic
    across keyframe resets)."""

    def __init__(self, bus: MujocoBus, period_ns: int, start_ns: int = 1_000 * _NS) -> None:
        self.bus = bus
        self.period_ns = period_ns
        self._start = start_ns
        self.ticks = 0

    def monotonic_ns(self) -> int:
        return self._start + self.ticks * self.period_ns

    def sleep_until(self, t_ns: int) -> None:
        n = -(-(t_ns - self.monotonic_ns()) // self.period_ns)
        if n > 0:
            self.bus.step_ticks(n)
            self.ticks += n


class SimBackend:
    """In-process ServoCore + MujocoBus + LocalIpc + FakeHeartbeatLink + simulated clock (FS12)."""

    def __init__(
        self,
        cfg: So101Config,
        calibration: dict[str, MotorCalibration],
        *,
        action_delay_ticks: int = 0,
        sim_params: Any | None = None,
        seed: int = 0,
        start: Literal["home", "rest"] = "home",
        model: mujoco.MjModel | None = None,
    ) -> None:
        from so101_rl.sim.model import load_model
        from so101_rl.sim.mujoco_bus import MujocoBus

        if start not in ("home", "rest"):
            raise ValueError(f"start must be 'home' or 'rest', got {start!r}")
        self.cfg = cfg
        self.calibration = calibration
        self.start = start
        self.sim_params = sim_params
        if model is None:
            model, data = load_model()
        else:
            import mujoco

            data = mujoco.MjData(model)
        if sim_params is not None:
            sim_params.apply_cpu(model)  # before MujocoBus saves the actuator gains it restores on torque-on
        self.model, self.data = model, data
        self.bus: MujocoBus = MujocoBus(model, data, calibration, cfg, action_delay_ticks=action_delay_ticks, seed=seed)
        self.chain = KinematicChain.from_yaml(cfg.kinematics_path)
        self._fk = So101FK(self.chain, calibration)
        self.servo_period_ns = round(_NS / cfg.rates.servo_hz)
        self.ticks_per_policy_step = round(cfg.rates.servo_hz / cfg.rates.policy_hz)
        self.clock = SimTickClock(self.bus, self.servo_period_ns)
        self.ipc: LocalIpc | None = None
        self.core: ServoCore | None = None
        self.mcu = None
        self._cmd = new_cmd()
        self.boots = 0

    def fk(self) -> So101FK:
        return self._fk

    def reset(self, q0_ticks: np.ndarray | None) -> None:
        """Boot a fresh servo core with the arm limp at the start keyframe (``q0_ticks`` overrides its joint
        positions; it must be near rest or in the box above safe_state.start_min_tip_z_m, like any re-energise
        start pose). Ends in RUN after one published RUN record; the re-energise ramp has moved the arm to home."""
        import mujoco

        from so101_rl.servo.core import build_servo_core
        from so101_rl.servo.heartbeat import FakeHeartbeatLink, FakeMcu

        bus = self.bus
        bus.connect()  # set_torque is refused on a disconnected bus
        bus.reset_to_keyframe(self.start)
        if q0_ticks is not None:
            q0 = np.asarray(q0_ticks)
            if q0.shape != (N_JOINTS,):
                raise ValueError(f"q0_ticks must have shape ({N_JOINTS},)")
            if np.any(q0 < self.cfg.joints.min_ticks) or np.any(q0 > self.cfg.joints.max_ticks):
                raise ValueError(f"q0_ticks {q0.tolist()} outside the joint limits")
            q = bus.ticks_to_q(q0.astype(np.int64))
            self.data.qpos[bus.qpos_adr] = q
            self.data.qvel[bus.dof_adr] = 0.0
            self.data.ctrl[bus.act_ids] = q
            mujoco.mj_forward(self.model, self.data)
        bus.set_torque(False)  # limp at the start pose, like the arm after power-up
        self.ipc = LocalIpc()
        self.mcu = FakeMcu(self.clock, timeout_ms=self.cfg.rates.heartbeat_timeout_ms)
        self.core = build_servo_core(
            self.cfg, self.calibration, bus, self.ipc, FakeHeartbeatLink(self.mcu), self.clock,
            chain=self.chain, log=lambda _m: None,
        )
        self.mcu.set_button(True)
        self.core.start(arm_timeout_s=5.0)
        self.mcu.set_button(False)
        self.boots += 1
        if self.core.state is not SmState.RUN:
            raise ResetFailed(f"sim servo core did not reach RUN (state {self.core.state.name})")
        self._tick()  # one RUN record (resync) so read_state returns a RUN state of this boot

    def _tick(self) -> None:
        assert self.core is not None
        t = self.clock.monotonic_ns()
        self.core.tick()
        self.clock.sleep_until(t + self.servo_period_ns)  # = bus.step_ticks(1)

    def send_delta(self, delta_ticks: np.ndarray, state_seq: int, episode: int) -> int:
        return self._send(CmdMode.RUN, delta_ticks, state_seq, episode)

    def send_hold(self, state_seq: int = 0, episode: int = 0) -> int:
        return self._send(CmdMode.HOLD, np.zeros(N_JOINTS), state_seq, episode)

    def _send(self, mode: CmdMode, delta: np.ndarray, state_seq: int, episode: int) -> int:
        if self.ipc is None:
            raise RuntimeError("SimBackend: reset() first")
        _write_cmd(self.ipc, self._cmd, self.clock.monotonic_ns(), mode, delta, state_seq, episode)
        return self.ipc.last_cmd_seq

    def accepted_state(self, out: np.ndarray) -> bool:
        """In the sim the latest record IS the step's record (nothing ticks between the calls)."""
        return self.read_state(out)

    def read_state(self, out: np.ndarray) -> bool:
        if self.ipc is None:
            return False
        return self.ipc.read_state(out)

    def wait_policy_tick(self) -> None:
        """``servo_hz / policy_hz`` (= 3) x (``core.tick()``, ``bus.step_ticks(1)``)."""
        if self.core is None:
            raise RuntimeError("SimBackend: reset() first")
        for _ in range(self.ticks_per_policy_step):
            self._tick()

    def render(self, width: int = 320, height: int = 240, camera: str | int = -1) -> np.ndarray:
        """uint8[H, W, 3] of the current sim state (``mujoco.Renderer``; needs an OpenGL context)."""
        import mujoco

        with mujoco.Renderer(self.model, height, width) as r:
            r.update_scene(self.data, camera=camera)
            return r.render().copy()

    def close(self) -> None:
        self.core = None
        self.ipc = None
