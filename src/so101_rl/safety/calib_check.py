"""Calibration read-back (plan F5, lerobot-specialist guidance).

Compares the calibration JSON against what the motors actually hold. The comparison is on DECODED values:
``Bus.read_calibration_regs`` returns ``Homing_Offset`` already sign-magnitude-decoded (raw ``2048 + 1647`` ->
``-1647``), so a naive compare of raw register bytes against the JSON would report a false mismatch; this module
never sees raw bytes. ``require_calibration`` is called before ANY torque enable (``reenergize`` step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from so101_rl.config import JOINT_NAMES, MotorCalibration
from so101_rl.servo.bus import Bus

_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
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
    """Every motor in ``expected`` must be present in the register read-back with identical id, drive_mode,
    homing_offset and position limits. One entry per mismatching motor (the first differing field)."""
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
