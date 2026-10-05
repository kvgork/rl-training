"""scripts/bench_warp.py S0 gate status (CPU: the pure gate function; the GPU run is `pixi run bench-warp`)."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("bench_warp", Path(__file__).parent.parent / "scripts" / "bench_warp.py")
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _row(n=1024, sps=7e5, vram=800.0, overflow="none"):
    return {"nworld": n, "env_steps_per_s": sps, "process_vram_mb": vram, "overflow": overflow, "finite": True}


GOOD_PARITY = {"max_abs_dq_arm_rad": 4e-7}


def test_full_run_passes_and_fails_on_measurements():
    kw = {"steps": 2000, "njmax": None, "nconmax": None}
    assert bench.gate_status([_row()], GOOD_PARITY, **kw)[0] == "PASS"
    assert bench.gate_status([_row(sps=4e5)], GOOD_PARITY, **kw)[0] == "FAIL"
    assert bench.gate_status([_row(overflow="world 3: NEFC")], GOOD_PARITY, **kw)[0] == "FAIL"
    assert bench.gate_status([_row()], {"max_abs_dq_arm_rad": 2e-6}, **kw)[0] == "FAIL"


@pytest.mark.parametrize(
    ("rows", "parity", "kw", "reason"),
    [
        ([_row()], None, {"steps": 2000, "njmax": None, "nconmax": None}, "parity"),
        ([_row()], GOOD_PARITY, {"steps": 2000, "njmax": 512, "nconmax": None}, "overridden"),
        ([_row()], GOOD_PARITY, {"steps": 2000, "njmax": None, "nconmax": 64}, "overridden"),
        ([_row()], GOOD_PARITY, {"steps": 1999, "njmax": None, "nconmax": None}, "steps"),
        ([_row(n=256)], GOOD_PARITY, {"steps": 2000, "njmax": None, "nconmax": None}, "N=1024"),
    ],
)
def test_partial_runs_are_not_evaluated(rows, parity, kw, reason):
    """A run that cannot certify the shipped config never reads as PASS, even with passing numbers."""
    status, _, reasons = bench.gate_status(rows, parity, **kw)
    assert status == "NOT EVALUATED"
    assert any(reason in r for r in reasons), reasons
