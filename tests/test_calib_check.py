"""Calibration read-back (plan task 2.4). Uses a minimal local Bus double (not U1's FakeBus)."""

from dataclasses import replace

import pytest

from so101_rl.config import JOINT_NAMES, MotorCalibration
from so101_rl.safety.calib_check import CalibrationMismatch, check_calibration, require_calibration


def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    return -(raw & ((1 << sign_bit) - 1)) if raw & (1 << sign_bit) else raw


class RegBus:
    """Holds RAW sign-magnitude Homing_Offset registers and decodes them like the real bus (F2)."""

    def __init__(self, calibration: dict[str, MotorCalibration]):
        self.raw = {}
        for n, c in calibration.items():
            off = c.homing_offset
            raw = off if off >= 0 else (1 << 11) | -off  # bit 11 = sign
            self.raw[n] = {"id": c.id, "homing_offset": raw, "range_min": c.range_min, "range_max": c.range_max}
        self.drive_mode = {n: c.drive_mode for n, c in calibration.items()}
        self.missing: set[str] = set()

    def read_calibration_regs(self):
        out = {}
        for n, r in self.raw.items():
            if n in self.missing:
                continue
            out[n] = MotorCalibration(
                id=r["id"],
                drive_mode=self.drive_mode[n],
                homing_offset=decode_sign_magnitude(r["homing_offset"], 11),
                range_min=r["range_min"],
                range_max=r["range_max"],
            )
        return out


def test_matching_calibration_passes(calibration):
    bus = RegBus(calibration)
    report = check_calibration(bus, calibration)
    assert report.ok and report.mismatches == {}
    require_calibration(bus, calibration)  # does not raise


def test_sign_magnitude_register_matches_negative_file_value(calibration):
    """Register raw 2048 + 1647 decodes to -1647 and must MATCH the file's -1647."""
    bus = RegBus(calibration)
    assert calibration["shoulder_pan"].homing_offset == -1647
    assert bus.raw["shoulder_pan"]["homing_offset"] == 2048 + 1647
    assert check_calibration(bus, calibration).ok


def test_homing_offset_plus_one_is_a_mismatch(calibration):
    bus = RegBus(calibration)
    bus.raw["elbow_flex"]["homing_offset"] += 1  # corrupt by +1 (elbow offset is positive)
    report = check_calibration(bus, calibration)
    assert not report.ok
    assert report.mismatches == {
        "elbow_flex": ("homing_offset", calibration["elbow_flex"].homing_offset, calibration["elbow_flex"].homing_offset + 1)
    }
    with pytest.raises(CalibrationMismatch) as ei:
        require_calibration(bus, calibration)
    assert ei.value.report is report or ei.value.report.mismatches == report.mismatches
    assert "elbow_flex" in str(ei.value)


def test_wrong_sign_of_a_negative_offset_is_a_mismatch(calibration):
    bus = RegBus(calibration)
    bus.raw["shoulder_pan"]["homing_offset"] = 1647  # +1647 instead of -1647
    rep = check_calibration(bus, calibration)
    assert rep.mismatches["shoulder_pan"] == ("homing_offset", -1647, 1647)


@pytest.mark.parametrize("field", ["range_min", "range_max", "id"])
def test_other_fields_are_compared(calibration, field):
    bus = RegBus(calibration)
    bus.raw["wrist_flex"][field] += 5
    rep = check_calibration(bus, calibration)
    assert rep.mismatches["wrist_flex"][0] == field


def test_drive_mode_is_compared(calibration):
    bus = RegBus(calibration)
    bus.drive_mode["gripper"] ^= 1
    assert check_calibration(bus, calibration).mismatches["gripper"][0] == "drive_mode"


def test_missing_motor_is_a_mismatch(calibration):
    bus = RegBus(calibration)
    bus.missing.add("wrist_roll")
    rep = check_calibration(bus, calibration)
    assert rep.mismatches["wrist_roll"][0] == "missing" and not rep.ok


def test_all_motors_mismatching_are_reported(calibration):
    bus = RegBus(calibration)
    for n in JOINT_NAMES:
        bus.raw[n]["range_max"] -= 1
    assert set(check_calibration(bus, calibration).mismatches) == set(JOINT_NAMES)


def test_expected_subset_of_joints_only_checks_those(calibration):
    bus = RegBus(calibration)
    bus.raw["gripper"]["range_min"] += 10
    subset = {n: c for n, c in calibration.items() if n != "gripper"}
    assert check_calibration(bus, subset).ok


def test_replace_helper_sanity(calibration):
    # guards the fixture itself: a modified expectation must be seen
    bad = dict(calibration)
    bad["gripper"] = replace(bad["gripper"], range_max=bad["gripper"].range_max + 1)
    assert not check_calibration(RegBus(calibration), bad).ok
