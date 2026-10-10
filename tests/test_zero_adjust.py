"""`pixi run zero-adjust` (plan §10 2026-10-10): joint zeros set numerically from the 2-pose camera fit.

Everything runs on a ``FakeBus`` that models ``Present_Position = Actual_Position - Homing_Offset`` (the STS3215 /
lerobot semantics); the real port is never opened."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, load_calibration, rad_to_ticks, ticks_to_rad
from so101_rl.safety.calib_check import check_calibration
from so101_rl.servo import zero_adjust as za
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.probe import BusReadAdapter
from so101_rl.servo.shm import SerialPortLock
from so101_rl.servo.timing import FakeClock

CALIB = Path(__file__).parent / "data" / "so101_follower.calib.json"
REPORT_TVO = {
    "verdict": "joint_offsets",
    "separation_vif": 1.80,
    "offsets_deg": {"shoulder_lift": -4.9448, "elbow_flex": -0.1559, "wrist_flex": 0.9955, "wrist_roll": 3.354},
    "offsets_sigma_deg": {"shoulder_lift": 0.063, "elbow_flex": 0.052, "wrist_flex": 0.091, "wrist_roll": 0.28},
}
# a physical pose (rad) away from every limit
Q_TRUE = np.array([0.1, -0.4, 0.6, 0.3, 0.0, 0.0])


def _setup(tmp_path, *, offsets=None, drive_mode=None):
    """A copy of the calibration JSON, a port file and a fake arm whose READING lacks ``offsets`` (deg): the arm
    is physically at ``Q_TRUE`` but reads ``Q_TRUE - offset`` (``q_true = q_read + offset``)."""
    raw = json.loads(CALIB.read_text())
    if drive_mode:
        for j in drive_mode:
            raw[j]["drive_mode"] = 1
    cal_path = tmp_path / "so101_follower.json"
    cal_path.write_text(json.dumps(raw, indent=4))
    os.utime(cal_path, (1, 1))  # older than any report written by the test
    cal = load_calibration(cal_path)
    q_read = Q_TRUE.copy()
    for j, off in (offsets or {}).items():
        q_read[JOINT_NAMES.index(j)] -= math.radians(off)
    bus = FakeBus(FakeClock(), calibration=cal, initial_ticks=rad_to_ticks(q_read, cal))
    port = tmp_path / "ttyFAKE"
    port.write_text("")
    return cal_path, cal, bus, str(port)


def _report(tmp_path, tvo=REPORT_TVO):
    d = tmp_path / "camera-calib" / "20261010-084545"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"diagnostics": {"tilt_vs_offsets": tvo}}))
    return d


def _main(tmp_path, bus, cal_path, port, *args):
    argv = ["--calibration", str(cal_path), "--port", port, "--out-dir", str(tmp_path / "hw"), *args]
    return za.main(argv, bus_factory=lambda p, c: bus, link_factory=lambda p, ids: BusReadAdapter(bus, ids))


def _reading(bus, cal):
    return ticks_to_rad(np.array([bus.read_reg("Present_Position", cal[n].id) for n in JOINT_NAMES]), cal)


# --------------------------------------------------------------------------------------------- sign, end to end
@pytest.mark.parametrize("drive_mode", [None, ("shoulder_lift", "wrist_flex")])
def test_after_apply_the_arm_reads_its_true_angle(tmp_path, drive_mode):
    offsets = {"shoulder_lift": -4.94, "wrist_flex": 1.0}
    cal_path, cal, bus, port = _setup(tmp_path, offsets=offsets, drive_mode=drive_mode)
    before = _reading(bus, cal)
    assert abs(before[1] - Q_TRUE[1]) > math.radians(4.5)  # the miscalibrated reading
    args = [a for j, o in offsets.items() for a in ("--joint", f"{j}={o}")]
    assert _main(tmp_path, bus, cal_path, port, *args, "--apply", "--attended") == za.EXIT_OK
    new_cal = load_calibration(cal_path)
    after = _reading(bus, new_cal)
    one_tick = 2 * math.pi / 4096  # the fake arm's reading and the shift are each rounded to a tick
    for j in ("shoulder_lift", "wrist_flex"):
        i = JOINT_NAMES.index(j)
        assert abs(after[i] - Q_TRUE[i]) <= one_tick + 1e-9, (j, math.degrees(after[i] - Q_TRUE[i]))
    for j in ("shoulder_pan", "elbow_flex", "wrist_roll"):  # untouched
        assert after[JOINT_NAMES.index(j)] == before[JOINT_NAMES.index(j)]
    assert check_calibration(bus, new_cal).ok  # JSON matches the registers


def test_lerobot_recal_evidence_homing_minus_7_moves_reading_plus_7():
    """The user's evidence (recal -930 -> -937, reading +7): Present = Actual - Homing_Offset."""
    cal = load_calibration(CALIB)
    bus = FakeBus(FakeClock(), calibration=cal, initial_ticks=[2047] * 6)
    bus.write_reg("Homing_Offset", cal["shoulder_lift"].id, cal["shoulder_lift"].homing_offset - 7)
    assert bus.read_reg("Present_Position", cal["shoulder_lift"].id) == 2047 + 7


def test_shift_math_limits_follow_homing():
    cal = load_calibration(CALIB)
    shifts, skipped = za.shifts_from_offsets({"shoulder_lift": -4.9448, "elbow_flex": -0.156, "wrist_flex": 0.9955}, cal)
    assert skipped.keys() == {"elbow_flex"}
    s = {x.joint: x for x in shifts}
    sl = s["shoulder_lift"]
    assert sl.delta_ticks == -56
    assert sl.new.homing_offset == sl.old.homing_offset + 56
    assert (sl.new.range_min, sl.new.range_max) == (sl.old.range_min - 56, sl.old.range_max - 56)
    assert s["wrist_flex"].delta_ticks == 11
    assert za.delta_ticks(-4.9448, 1) == 56  # drive_mode 1 flips the reading direction


# --------------------------------------------------------------------------------------------- report input
def test_from_report_skips_small_and_wrist_roll(tmp_path):
    cal_path, cal, bus, port = _setup(tmp_path)
    rep = _report(tmp_path)
    offsets, _ = za.offsets_from_report(rep, cal_path)
    assert set(offsets) == {"shoulder_lift", "elbow_flex", "wrist_flex"}  # wrist_roll only via --joint
    assert _main(tmp_path, bus, cal_path, port, "--from-report", str(rep), "--apply", "--attended") == 0
    written = {(r, i) for r, i, _ in bus.reg_writes}
    ids = {cal["shoulder_lift"].id, cal["wrist_flex"].id}
    assert {i for _, i in written} == ids  # only the chosen joints
    assert {r for r, _ in written} == {"Lock", "Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"}
    assert bus.torque_events == [] and bus.written_goals == []
    (art,) = (tmp_path / "hw").glob("zero-adjust-*.json")
    rep_json = json.loads(art.read_text())
    assert rep_json["ok"] and rep_json["calib_check"]["ok"] and rep_json["from_report"] == str(rep)
    assert {s["joint"] for s in rep_json["shifts"]} == {"shoulder_lift", "wrist_flex"}
    assert Path(rep_json["backup"]).exists()
    # the same report again: the JSON is now newer than the report -> refused (no double apply)
    n = len(bus.reg_writes)
    assert _main(tmp_path, bus, cal_path, port, "--from-report", str(rep), "--apply", "--attended") == za.EXIT_REFUSED
    assert len(bus.reg_writes) == n


@pytest.mark.parametrize(
    "patch",
    [
        {"verdict": "ambiguous"},
        {"verdict": "both"},
        {"offsets_sigma_deg": {**REPORT_TVO["offsets_sigma_deg"], "wrist_flex": 0.8}},
        {"offsets_deg": {**REPORT_TVO["offsets_deg"], "shoulder_lift": -12.0}},
    ],
)
def test_report_refusals_write_nothing(tmp_path, patch):
    cal_path, _cal, bus, port = _setup(tmp_path)
    rep = _report(tmp_path, {**REPORT_TVO, **patch})
    assert _main(tmp_path, bus, cal_path, port, "--from-report", str(rep), "--apply", "--attended") == za.EXIT_REFUSED
    assert bus.reg_writes == []


@pytest.mark.parametrize(
    "joint_arg",
    ["gripper=1.0", "shoulder_lift=11", "wrist_roll=1.0", "shoulder_lift=-50", "elbow"],
)
def test_out_of_range_refused(tmp_path, joint_arg):
    """gripper never; |offset| > 10 deg; wrist_roll range is 0..4095 so any shift leaves 0..4095; a bad argument."""
    cal_path, _cal, bus, port = _setup(tmp_path)
    assert _main(tmp_path, bus, cal_path, port, "--joint", joint_arg, "--apply", "--attended") == za.EXIT_REFUSED
    assert bus.reg_writes == []


def test_homing_beyond_sign_magnitude_range_refused():
    cal = load_calibration(CALIB)
    cal = {**cal, "wrist_flex": cal["wrist_flex"].__class__(4, 0, -2040, 866, 3179)}
    with pytest.raises(za.ZeroAdjustRefused, match="Homing_Offset"):
        za.shifts_from_offsets({"wrist_flex": 1.0}, cal)  # -2040 - 11 = -2051


# --------------------------------------------------------------------------------------------- apply refusals
def test_apply_needs_attended(tmp_path):
    cal_path, _cal, bus, port = _setup(tmp_path)
    assert _main(tmp_path, bus, cal_path, port, "--joint", "shoulder_lift=-4", "--apply") == za.EXIT_REFUSED
    assert bus.reg_writes == []


def test_refuses_with_torque_on(tmp_path):
    cal_path, cal, bus, port = _setup(tmp_path)
    bus.set_torque(True, [cal["gripper"].id])  # ANY motor, not only the adjusted one
    before = cal_path.read_text()
    assert _main(tmp_path, bus, cal_path, port, "--joint", "shoulder_lift=-4", "--apply", "--attended") == za.EXIT_REFUSED
    assert bus.reg_writes == [] and cal_path.read_text() == before
    assert not list(tmp_path.glob("*.bak-*"))


def test_refuses_while_the_servo_holds_the_port(tmp_path):
    cal_path, _cal, bus, port = _setup(tmp_path)
    lock = SerialPortLock(port)
    try:
        assert _main(tmp_path, bus, cal_path, port, "--joint", "shoulder_lift=-4", "--apply", "--attended") == za.EXIT_REFUSED
    finally:
        lock.release()
    assert bus.reg_writes == []


def test_refuses_when_servo_registers_differ_from_json(tmp_path):
    cal_path, cal, bus, port = _setup(tmp_path)
    bus._regs[("Homing_Offset", cal["elbow_flex"].id)] += 3  # set directly: not a bus write
    assert _main(tmp_path, bus, cal_path, port, "--joint", "shoulder_lift=-4", "--apply", "--attended") == za.EXIT_REFUSED
    assert bus.reg_writes == []


def test_dry_run_writes_nothing_and_prints_the_table(tmp_path, capsys):
    cal_path, _cal, bus, port = _setup(tmp_path)
    rep = _report(tmp_path)
    before = cal_path.read_text()
    assert _main(tmp_path, bus, cal_path, port, "--from-report", str(rep)) == za.EXIT_OK
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "shoulder_lift" in out and "-56" in out and "skipped: elbow_flex" in out
    assert bus.reg_writes == [] and bus.torque_events == []
    assert cal_path.read_text() == before
    assert not (tmp_path / "hw").exists() and not list(tmp_path.glob("*.bak-*"))


# --------------------------------------------------------------------------------------------- failure + restore
def test_failure_after_a_write_prints_restore_and_restore_works(tmp_path, capsys):
    cal_path, cal, bus, port = _setup(tmp_path)
    original = cal_path.read_text()
    real_write = bus.write_reg
    calls = {"n": 0}

    def flaky(name, motor_id, value):
        if name == "Max_Position_Limit":
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("simulated bus error")
        real_write(name, motor_id, value)

    bus.write_reg = flaky
    code = _main(tmp_path, bus, cal_path, port, "--joint", "shoulder_lift=-4.94", "--apply", "--attended")
    assert code == za.EXIT_WRITTEN_FAILED
    err = capsys.readouterr().err
    assert "FAILED AFTER WRITING" in err and "Homing_Offset: -881 -> -825" in err and "--restore" in err
    assert "MAYBE (it raised): shoulder_lift (id 2) Max_Position_Limit: 3209 -> 3153" in err
    assert cal_path.read_text() == original  # no partial JSON
    (fail,) = (tmp_path / "hw" / "failed").glob("zero-adjust-*.json")
    backup = Path(json.loads(fail.read_text())["backup"])
    assert backup.read_text() == original
    assert not check_calibration(bus, load_calibration(cal_path)).ok  # the servo half-written

    bus.write_reg = real_write
    assert _main(tmp_path, bus, cal_path, port, "--restore", str(backup), "--apply", "--attended") == za.EXIT_OK
    assert check_calibration(bus, load_calibration(cal_path)).ok
    assert load_calibration(cal_path) == load_calibration(backup)
    assert bus.read_reg("Lock", cal["shoulder_lift"].id) == 0  # Lock back to what it was
