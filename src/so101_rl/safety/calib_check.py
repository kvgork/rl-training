"""Calibration read-back (plan F5, lerobot-specialist guidance).

Compares the calibration JSON against what the motors actually hold. The comparison is on DECODED values:
``Bus.read_calibration_regs`` returns ``Homing_Offset`` already sign-magnitude-decoded (raw ``2048 + 1647`` ->
``-1647``), so a naive compare of raw register bytes against the JSON would report a false mismatch; this module
never sees raw bytes. ``require_calibration`` is called before ANY torque enable (``reenergize`` step 2).

What this check can and cannot prove (review S11): ``id``, ``homing_offset`` and the position limits live in the
servo EEPROM and are compared. ``drive_mode`` is NOT stored on the STS3215; every backend's
``read_calibration_regs`` copies it from the calibration it was built with, so comparing it would always pass
and prove nothing. It is therefore not compared. A wrong ``drive_mode`` (a joint moving the wrong way) can only
be caught by a motion-sign test: command a small delta on each joint and check the sign of the Present_Position
change. That test belongs to the hardware track / the servo core (H-track, U3), not to this register check.

CLI (pixi task ``calib-check``, H7): ``python -m so101_rl.safety.calib_check [--port P] [--calibration JSON]
[--timeout S] [--out-dir runs/hw]``. READ-ONLY on the bus: the registers come from ``servo.probe.read_calibration``
(the read-only probe link: PING / READ / SYNC_READ only, no write method; it refuses while the servo process holds
the port lock). This module never touches the serial bus itself. Prints a per-joint table and writes
``<out-dir>/calib-check-<ts>.json`` on a match, ``<out-dir>/failed/calib-check-<ts>.json`` otherwise (so a failed
run never satisfies a gate that only looks for ``<out-dir>/calib-check-*.json``, e.g. ``sysid-excite --backend
real``). Exit 0 = every joint matches, 1 = mismatch, 2 = unable to read (calibration file unreadable, port missing,
port held by the servo, a motor silent, a read failed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from so101_rl.config import JOINT_NAMES, MotorCalibration
from so101_rl.servo.bus import Bus

_FIELDS = ("id", "homing_offset", "range_min", "range_max")  # drive_mode: not on the servo, see the module doc
_MISSING = -(2**31)  # register value reported when a motor did not appear in the read-back


@dataclass
class CalibReport:
    ok: bool
    mismatches: dict[str, tuple[str, int, int]] = field(default_factory=dict)  # motor -> (field, file, register)


class CalibrationMismatch(RuntimeError):
    def __init__(self, report: CalibReport) -> None:
        self.report = report
        detail = "; ".join(f"{m}.{f}: file={a} register={b}" for m, (f, a, b) in report.mismatches.items())
        super().__init__(f"calibration mismatch between JSON and motors: {detail}")


def check_calibration(bus: Bus, expected: dict[str, MotorCalibration]) -> CalibReport:
    """Every motor in ``expected`` must be present in the register read-back with identical id, homing_offset
    and position limits. One entry per mismatching motor (the first differing field). ``drive_mode`` is not
    checked (it is not a servo register; see the module docstring)."""
    actual = bus.read_calibration_regs()
    mismatches: dict[str, tuple[str, int, int]] = {}
    names = [n for n in JOINT_NAMES if n in expected] + [n for n in expected if n not in JOINT_NAMES]
    for name in names:
        want = expected[name]
        got = actual.get(name)
        if got is None:
            mismatches[name] = ("missing", want.id, _MISSING)
            continue
        for f in _FIELDS:
            a, b = int(getattr(want, f)), int(getattr(got, f))
            if a != b:
                mismatches[name] = (f, a, b)
                break
    return CalibReport(ok=not mismatches, mismatches=mismatches)


def require_calibration(bus: Bus, expected: dict[str, MotorCalibration]) -> None:
    """Raise ``CalibrationMismatch`` unless the motors match the JSON exactly."""
    report = check_calibration(bus, expected)
    if not report.ok:
        raise CalibrationMismatch(report)


# --------------------------------------------------------------------------------------------- CLI (H7)
EXIT_OK, EXIT_MISMATCH, EXIT_UNREADABLE = 0, 1, 2


class _Readback:
    """The registers already read by the probe, seen as the one ``Bus`` method ``check_calibration`` calls."""

    def __init__(self, regs: dict[str, MotorCalibration]) -> None:
        self._regs = regs

    def read_calibration_regs(self) -> dict[str, MotorCalibration]:
        return self._regs


def _table(expected: dict[str, MotorCalibration], actual: dict[str, MotorCalibration], report: CalibReport) -> str:
    cols = ("id", "homing_offset", "range_min", "range_max")
    head = f"{'joint':<14}" + "".join(f"{c + ' file/servo':>26}" for c in cols) + "  result"
    lines = [head, "-" * len(head)]
    for n, want in expected.items():
        got = actual.get(n)
        cells = "".join(
            f"{f'{getattr(want, c)}/' + ('-' if got is None else str(getattr(got, c))):>26}" for c in cols
        )
        bad = n in report.mismatches
        lines.append(f"{n:<14}{cells}  {'MISMATCH ' + report.mismatches[n][0] if bad else 'ok'}")
    return "\n".join(lines)


def _write(result: dict, out_dir: Path, ok: bool) -> Path:
    root = out_dir if ok else out_dir / "failed"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"calib-check-{dt.datetime.now(dt.UTC).strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return path


def main(argv: list[str] | None = None, *, link_factory=None) -> int:
    """See the module docstring. ``link_factory`` is the probe's link hook (tests pass a ``FakeBus`` adapter)."""
    from so101_rl.config import load_calibration, load_config
    from so101_rl.servo.probe import DEFAULT_TIMEOUT_S, ProbeRefused, read_calibration

    ap = argparse.ArgumentParser(prog="python -m so101_rl.safety.calib_check", description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=None, help="default: so101.yaml port")
    ap.add_argument("--calibration", default=None, help="calibration JSON (default: so101.yaml calibration_path)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="wall-clock bound for the read, s")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/hw"), help="artifact directory (default runs/hw)")
    args = ap.parse_args(argv)
    if not args.timeout > 0:
        ap.error("--timeout must be > 0")

    result: dict = {"ok": False, "port": None, "calibration_path": None, "joints": {}, "mismatches": {}}
    try:
        cfg = load_config(calibration=load_calibration(Path(args.calibration)) if args.calibration else None)
        cal_path = Path(args.calibration or cfg.calibration_path).expanduser()
        result["calibration_path"] = str(cal_path)
        expected = load_calibration(cal_path)
        port = args.port or cfg.port
        result["port"] = port
        actual = read_calibration(expected, port=port, link_factory=link_factory, timeout_s=args.timeout)
    except (ProbeRefused, OSError, ConnectionError, ValueError, KeyError) as e:
        result.update(exit_code=EXIT_UNREADABLE, error=f"{type(e).__name__}: {e}")
        print(f"calib-check: unable to read the calibration ({e})", file=sys.stderr)
        print(f"calib-check: wrote {_write(result, args.out_dir, ok=False)}", file=sys.stderr)
        return EXIT_UNREADABLE

    report = check_calibration(_Readback(actual), expected)
    code = EXIT_OK if report.ok else EXIT_MISMATCH
    result.update(
        ok=report.ok,
        exit_code=code,
        joints={n: {"file": asdict(expected[n]), "servo": asdict(actual[n])} for n in expected if n in actual},
        mismatches={m: {"field": f, "file": a, "servo": b} for m, (f, a, b) in report.mismatches.items()},
        note="drive_mode is not a servo register and is not compared",
    )
    print(f"calibration: {result['calibration_path']}\nport: {result['port']}")
    print(_table(expected, actual, report))
    print("calib-check: " + ("PASS (all joints match)" if report.ok else f"FAIL ({len(report.mismatches)} mismatch)"))
    print(f"calib-check: wrote {_write(result, args.out_dir, ok=report.ok)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
