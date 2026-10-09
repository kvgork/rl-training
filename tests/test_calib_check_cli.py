"""`pixi run calib-check` (H7): a real CLI, read-only on the bus, exit 0 / 1 / 2, an artifact per run.

All runs go through ``servo.probe.read_calibration`` on a ``FakeBus`` (``BusReadAdapter``); the real port is never
opened here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so101_rl.config import JOINT_NAMES, load_calibration
from so101_rl.safety import calib_check
from so101_rl.servo.fake_bus import FakeBus
from so101_rl.servo.probe import BusReadAdapter, ProbeRefused, read_calibration
from so101_rl.servo.shm import SerialPortLock
from so101_rl.servo.timing import FakeClock

CALIB = Path(__file__).parent / "data" / "so101_follower.calib.json"
CAL = load_calibration(CALIB)
WRITE_NAMES = {"write", "set_torque", "enable_torque", "disable_torque", "action", "reboot", "factory", "regwrite", "syncwrite"}


class SpyBus:
    """Records every attribute of the bus that is touched (a write would show up here)."""

    def __init__(self, bus):
        self._bus = bus
        self.touched: list[str] = []

    def __getattr__(self, name):
        self.touched.append(name)
        return getattr(self._bus, name)


def _port(tmp_path):
    p = tmp_path / "ttyFAKE"
    p.write_text("")
    return str(p)


def _run(tmp_path, bus, *extra):
    out = tmp_path / "hw"
    argv = ["--port", _port(tmp_path), "--calibration", str(CALIB), "--out-dir", str(out), *extra]
    code = calib_check.main(argv, link_factory=lambda p, ids: BusReadAdapter(bus, ids))
    return code, out


def _assert_no_writes(bus: FakeBus, spy: SpyBus):
    assert bus.written_goals == [] and bus.torque_events == []
    assert not any(any(w in t.lower() for w in WRITE_NAMES) for t in spy.touched), spy.touched
    assert set(spy.touched) <= {"connect", "disconnect", "ping_all", "read_reg"}


def test_read_calibration_decodes_the_registers_without_writing(tmp_path):
    bus = FakeBus(FakeClock(), calibration=CAL)
    spy = SpyBus(bus)
    got = read_calibration(CAL, port=_port(tmp_path), link_factory=lambda p, ids: BusReadAdapter(spy, ids))
    assert got == CAL  # negative homing offsets survive the sign-magnitude round trip
    assert any(c.homing_offset < 0 for c in CAL.values())
    _assert_no_writes(bus, spy)


def test_cli_match_exits_0_prints_a_table_and_writes_the_artifact(tmp_path, capsys):
    bus = FakeBus(FakeClock(), calibration=CAL)
    spy = SpyBus(bus)
    code, out = _run(tmp_path, spy)
    assert code == 0
    text = capsys.readouterr().out
    assert "PASS" in text and all(n in text for n in JOINT_NAMES)
    arts = sorted(out.glob("calib-check-*.json"))
    assert len(arts) == 1 and not (out / "failed").exists()
    rep = json.loads(arts[0].read_text())
    assert rep["ok"] is True and rep["exit_code"] == 0 and rep["mismatches"] == {}
    assert rep["joints"]["shoulder_pan"]["servo"]["homing_offset"] == CAL["shoulder_pan"].homing_offset
    _assert_no_writes(bus, spy)


@pytest.mark.parametrize(("field", "delta"), [("homing_offset", 1), ("range_max", -3), ("range_min", 2)])
def test_cli_mismatch_exits_1_and_names_the_field(tmp_path, capsys, field, delta):
    """H7 shown-to-fail: a corrupted COPY of the JSON (the servos are untouched) is reported, never 'fixed'."""
    raw = json.loads(CALIB.read_text())
    raw["elbow_flex"][field] += delta
    bad = tmp_path / "corrupted.json"
    bad.write_text(json.dumps(raw))
    bus = FakeBus(FakeClock(), calibration=CAL)
    spy = SpyBus(bus)
    out = tmp_path / "hw"
    argv = ["--port", _port(tmp_path), "--calibration", str(bad), "--out-dir", str(out)]
    code = calib_check.main(argv, link_factory=lambda p, ids: BusReadAdapter(spy, ids))
    assert code == 1
    text = capsys.readouterr().out
    assert f"MISMATCH {field}" in text and "FAIL" in text
    assert not list(out.glob("calib-check-*.json"))  # a failed run never satisfies a presence-only gate
    (art,) = sorted((out / "failed").glob("calib-check-*.json"))
    rep = json.loads(art.read_text())
    assert rep["ok"] is False and rep["exit_code"] == 1
    assert rep["mismatches"] == {
        "elbow_flex": {"field": field, "file": raw["elbow_flex"][field], "servo": getattr(CAL["elbow_flex"], field)}
    }
    _assert_no_writes(bus, spy)


def test_cli_servo_register_differs_from_file_exits_1(tmp_path):
    """The other direction: the motors changed (e.g. a lerobot recalibration), the JSON did not."""
    bus = FakeBus(FakeClock(), calibration=CAL)
    bus._regs[("Homing_Offset", CAL["wrist_flex"].id)] += 7  # set directly: not a bus write
    code, out = _run(tmp_path, bus)
    assert code == 1
    (art,) = (out / "failed").glob("calib-check-*.json")
    assert json.loads(art.read_text())["mismatches"]["wrist_flex"]["field"] == "homing_offset"


def test_cli_refuses_with_exit_2_while_the_servo_holds_the_port(tmp_path, capsys):
    port = _port(tmp_path)
    lock = SerialPortLock(port)  # what servo.process.run holds for its bus port
    try:
        bus = FakeBus(FakeClock(), calibration=CAL)
        out = tmp_path / "hw"
        argv = ["--port", port, "--calibration", str(CALIB), "--out-dir", str(out)]
        assert calib_check.main(argv, link_factory=lambda p, ids: BusReadAdapter(bus, ids)) == 2
        assert bus.pings == 0  # refused before any bus traffic
        assert "servo process owns" in capsys.readouterr().err
        assert not list(out.glob("calib-check-*.json"))
        (art,) = (out / "failed").glob("calib-check-*.json")
        assert json.loads(art.read_text())["exit_code"] == 2
    finally:
        lock.release()


def test_cli_silent_motor_and_missing_port_exit_2(tmp_path):
    bus = FakeBus(FakeClock(), calibration=CAL)
    bus.faults.drop_motor(CAL["wrist_roll"].id)
    assert _run(tmp_path, bus)[0] == 2

    def no_device(p, ids):
        raise ProbeRefused(f"cannot open {p}: no such device")

    out = tmp_path / "hw2"
    argv = ["--port", _port(tmp_path), "--calibration", str(CALIB), "--out-dir", str(out)]
    assert calib_check.main(argv, link_factory=no_device) == 2
    assert calib_check.main(["--calibration", str(tmp_path / "nope.json"), "--out-dir", str(out)]) == 2


def test_calib_check_module_has_a_cli_entry_point():
    """The bug: `python -m so101_rl.safety.calib_check` was a silent exit-0 no-op."""
    src = Path(calib_check.__file__).read_text()
    assert 'if __name__ == "__main__":' in src
