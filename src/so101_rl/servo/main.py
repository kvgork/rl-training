"""``so101-servo`` CLI (plan F6): ``python -m so101_rl.servo.main --backend {feetech,fake,mujoco} ...``.

Backends:

* ``fake``: :class:`~so101_rl.servo.fake_bus.FakeBus` on the real clock, powered up at the home pose. Unless
  ``--heartbeat-port`` or ``--no-mcu`` is given it gets an in-process :class:`~so101_rl.servo.heartbeat.FakeMcu`
  that boots LATCHED like the firmware, plus a simulated operator who presses the reset button, so the full
  start-up path (boot latch -> button -> ARM -> re-energise) runs.
* ``mujoco``: :class:`~so101_rl.sim.mujoco_bus.MujocoBus` starting limp at the ``rest`` keyframe; the physics is
  stepped in real time by :class:`RealtimeSimClock`. Needs the ``sim`` pixi feature.
* ``feetech``: the real arm (:class:`~so101_rl.servo.feetech_bus.FeetechBus` on ``--port``). Requires the
  watchdog MCU on ``--heartbeat-port`` (or ``heartbeat_port`` in the config) unless ``--no-mcu`` is given
  explicitly.

``--max-ticks N`` stops after N loop ticks (start-up not counted) and exits through the normal shutdown.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

from so101_rl.config import load_calibration, load_config
from so101_rl.servo.heartbeat import FakeHeartbeatLink, FakeMcu, McuStatus
from so101_rl.servo.process import require_rt_from_env, run
from so101_rl.servo.timing import RealClock

_NS = 1_000_000_000


class SimulatedOperatorLink(FakeHeartbeatLink):
    """``FakeHeartbeatLink`` whose operator holds the physical reset button while the MCU is latched at start-up
    and lets go once it is armed (the fake backend has no human). It never presses the button after ARM: a trip
    during the run stays latched, as on the real arm."""

    def __init__(self, mcu: FakeMcu) -> None:
        super().__init__(mcu)
        self._armed_once = False

    def status(self) -> McuStatus:
        st = super().status()
        if st.latched and not self._armed_once:
            self.mcu.set_button(True)
        return st

    def arm(self) -> McuStatus:
        self.mcu.set_button(False)
        st = super().arm()
        self._armed_once = self._armed_once or st.armed
        return st


class RealtimeSimClock(RealClock):
    """Real time; ``sleep_until`` additionally steps the MuJoCo physics by the servo ticks that elapsed."""

    def __init__(self, bus) -> None:
        self._bus = bus
        self._period_ns = round(bus.steps_per_tick * bus.model.opt.timestep * _NS)
        self._t0 = super().monotonic_ns()
        self._stepped = 0

    def sleep_until(self, t_ns: int) -> None:
        super().sleep_until(t_ns)
        due = (self.monotonic_ns() - self._t0) // self._period_ns
        if due > self._stepped:
            self._bus.step_ticks(int(due - self._stepped))
            self._stepped = int(due)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="so101-servo", description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("feetech", "fake", "mujoco"), required=True)
    ap.add_argument("--config", type=Path, default=None, help="so101.yaml (default: the packaged one)")
    ap.add_argument("--calibration", type=Path, default=None, help="LeRobot calibration JSON (default: from config)")
    ap.add_argument("--port", default=None, help="servo bus serial port (feetech; default: from config)")
    ap.add_argument("--heartbeat-port", default=None, help="watchdog MCU serial port (default: from config)")
    ap.add_argument("--no-mcu", action="store_true", help="run without any watchdog link (never on an unattended arm)")
    ap.add_argument("--shm-name", default=None, help="servo shm segment (default: from config)")
    ap.add_argument("--require-rt", action="store_true", help="fail (exit 3) unless SCHED_FIFO + mlockall work")
    ap.add_argument("--max-ticks", type=int, default=None, help="stop after N loop ticks")
    ap.add_argument("--arm-timeout-s", type=float, default=None, help="give up waiting for the MCU button/ARM")
    ap.add_argument("--tick-log", type=Path, default=None, help="binary tick log (TICK_LOG_DTYPE records)")
    ap.add_argument("--summary", type=Path, default=None, help="write a JSON summary at exit")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_config(args.config) if args.calibration is None else None
    calib_path = args.calibration if args.calibration is not None else base.calibration_path
    calibration = load_calibration(Path(calib_path).expanduser())
    cfg = load_config(args.config, calibration=calibration)
    overrides = {}
    if args.port:
        overrides["port"] = args.port
    if args.shm_name:
        overrides["shm_name"] = args.shm_name
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    hb_port = args.heartbeat_port if args.heartbeat_port is not None else cfg.heartbeat_port
    clock = RealClock()

    if args.backend == "fake":
        from so101_rl.servo.fake_bus import FakeBus

        home = np.asarray(cfg.safe_state.home_ticks, dtype=np.int32)

        def bus_factory():
            return FakeBus(clock, calibration=calibration, initial_ticks=home, record_writes=False)

    elif args.backend == "mujoco":
        from so101_rl.sim.model import load_model
        from so101_rl.sim.mujoco_bus import MujocoBus

        model, data = load_model()
        sim_bus = MujocoBus(model, data, calibration, cfg)
        sim_bus.connect()
        sim_bus.reset_to_keyframe("rest")
        sim_bus.set_torque(False)  # a limp arm at rest, like the real one after power-up
        sim_bus.written_goals = _NoList()
        clock = RealtimeSimClock(sim_bus)

        def bus_factory():
            return sim_bus

    else:
        if hb_port is None and not args.no_mcu:
            print(
                "so101-servo: the feetech backend needs the watchdog MCU (--heartbeat-port / heartbeat_port); "
                "pass --no-mcu only for an attended bench test",
                file=sys.stderr,
            )
            return 2
        from so101_rl.servo.feetech_bus import FeetechBus

        def bus_factory():
            return FeetechBus(cfg, calibration, clock=clock)

    def hb_factory():
        if args.no_mcu:
            return None
        if hb_port is not None:
            from so101_rl.servo.heartbeat import SerialHeartbeatLink

            return SerialHeartbeatLink(hb_port, clock)
        if args.backend == "feetech":
            return None  # unreachable: refused above
        return SimulatedOperatorLink(FakeMcu(clock, timeout_ms=cfg.rates.heartbeat_timeout_ms))

    return run(
        cfg,
        bus_factory,
        hb_factory,
        require_rt=args.require_rt or require_rt_from_env(),
        max_ticks=args.max_ticks,
        clock=clock,
        calibration=calibration,
        tick_log_path=args.tick_log,
        summary_path=args.summary,
        arm_timeout_s=args.arm_timeout_s,
    )


class _NoList(list):
    """Drop-in for ``MujocoBus.written_goals`` that records nothing (a long headless run must not grow)."""

    def append(self, _item) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
