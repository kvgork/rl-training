"""Soak harness and gate (plan Task 3.3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from so101_rl.safety.types import SmState, TripCause
from so101_rl.servo.core import PHASE_LOOP, TICK_LOG_DTYPE
from so101_rl.servo.soak import analyse_tick_log, check_soak_report, soak

ROOT = Path(__file__).parent.parent
CALIB = ROOT / "tests" / "data" / "so101_follower.calib.json"
FIELDS = (
    "rt_active",
    "kernel",
    "minutes",
    "tick_stats",
    "hb_max_gap_ms",
    "hb_gaps_over_200ms",
    "thermal_trips",
    "calib_mismatches",
    "max_temp_c",
    "rss_growth_mb",
    "gc_collections",
    "unexpected_safe_states",
)


@pytest.fixture(scope="module")
def fake_soak(tmp_path_factory):
    out = tmp_path_factory.mktemp("soak")
    path, rep = soak(backend="fake", minutes=5.0 / 60.0, out_root=out, calibration=CALIB)
    return path, rep


def test_five_second_fake_soak_writes_a_valid_report(fake_soak):
    path, _rep = fake_soak
    data = json.loads(path.read_text())
    assert set(FIELDS) <= set(data)
    assert data["servo_exit_code"] == 0
    assert 4.5 / 60 <= data["minutes"] <= 6.0 / 60
    assert data["tick_stats"]["samples"] >= 120  # ~150 ticks at 30 Hz
    assert data["policy_steps"] >= 40
    assert data["hb_gaps_over_200ms"] == 0 and data["hb_max_gap_ms"] < 200
    assert data["thermal_trips"] == 0 and data["calib_mismatches"] == 0
    assert data["unexpected_safe_states"] == 0
    assert set(data["max_temp_c"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}
    assert (path.parent / "ticks.bin").exists() and (path.parent / "servo_summary.json").exists()
    ok, _msgs = check_soak_report(path, strict=False)
    assert ok


def _edited(tmp_path, src: Path, **changes) -> Path:
    data = json.loads(src.read_text())
    for k, v in changes.items():
        if k == "p999_ms":
            data["tick_stats"]["p999_ms"] = v
        else:
            data[k] = v
    out = tmp_path / "report.json"
    out.write_text(json.dumps(data))
    return out


def _passing(tmp_path, src: Path) -> Path:
    return _edited(tmp_path, src, p999_ms=1.0, hb_gaps_over_200ms=0, thermal_trips=0, calib_mismatches=0)


def test_gate_passes_a_clean_report(tmp_path, fake_soak):
    ok, msgs = check_soak_report(_passing(tmp_path, fake_soak[0]), strict=True)
    assert ok and msgs == []


def test_gate_fails_on_p999_5_1_ms(tmp_path, fake_soak):
    p = _edited(tmp_path, _passing(tmp_path, fake_soak[0]), p999_ms=5.1)
    ok, msgs = check_soak_report(p, strict=True)
    assert not ok and any("p99.9" in m for m in msgs)


def test_gate_fails_on_one_heartbeat_gap_over_200_ms(tmp_path, fake_soak):
    p = _edited(tmp_path, _passing(tmp_path, fake_soak[0]), hb_gaps_over_200ms=1)
    ok, msgs = check_soak_report(p, strict=True)
    assert not ok and any("heartbeat" in m for m in msgs)


def test_gate_fails_on_one_thermal_trip(tmp_path, fake_soak):
    p = _edited(tmp_path, _passing(tmp_path, fake_soak[0]), thermal_trips=1)
    ok, msgs = check_soak_report(p, strict=True)
    assert not ok and any("thermal" in m for m in msgs)


def test_gate_fails_on_a_calibration_mismatch(tmp_path, fake_soak):
    p = _edited(tmp_path, _passing(tmp_path, fake_soak[0]), calib_mismatches=1)
    ok, _ = check_soak_report(p, strict=True)
    assert not ok


def test_non_strict_turns_failures_into_warnings(tmp_path, fake_soak):
    p = _edited(tmp_path, fake_soak[0], p999_ms=5.1, hb_gaps_over_200ms=1, thermal_trips=1)
    ok, msgs = check_soak_report(p, strict=False)
    assert ok and len(msgs) == 3 and all(m.startswith("WARN") for m in msgs)


def test_tick_log_analysis_counts_gaps_trips_and_stops():
    period = 33_333_333
    n = 60
    recs = np.zeros(n, dtype=TICK_LOG_DTYPE)
    recs["t_ns"] = np.arange(n) * period
    recs["phase"] = PHASE_LOOP
    recs["hb_sent"] = 1
    recs["hb_sent"][20:28] = 0  # 9 periods = 300 ms without a heartbeat
    recs["sm_state"] = int(SmState.RUN)
    recs["sm_state"][50:] = int(SmState.SAFE_STOPPING)
    recs["trips"][40:45] = int(TripCause.THERMAL)
    recs["temp_c"][:, 2] = 41
    recs["temp_c"][33, 2] = 58
    m = analyse_tick_log(recs, t_start_ns=0, t_end_ns=int(recs["t_ns"][-1]), period_ns=period)
    assert m["hb_gaps_over_200ms"] == 1 and 290 < m["hb_max_gap_ms"] < 310
    assert m["thermal_trips"] == 1
    assert m["unexpected_safe_states"] == 1
    assert m["max_temp_c"]["elbow_flex"] == 58
    assert m["tick_stats"]["samples"] == n


def test_soak_cli_writes_report(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "so101_rl.servo.soak", "--backend", "fake", "--minutes", "0.05", "--out", str(tmp_path),
         "--calibration", str(CALIB)],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert list(tmp_path.glob("soak-*/report.json"))
    assert "gate: PASS" in proc.stdout


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [("servo_exit_code", 1, "exited"), ("summary_found", False, "summary"), ("unexpected_safe_states", 1, "safe-state")],
)
def test_an_invalid_soak_fails_even_on_a_non_rt_host(tmp_path, fake_soak, field, value, needle):
    p = _edited(tmp_path, _passing(tmp_path, fake_soak[0]), **{field: value})
    for strict in (False, True):
        ok, msgs = check_soak_report(p, strict=strict)
        assert not ok and any(needle in m for m in msgs)
