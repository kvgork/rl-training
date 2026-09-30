import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

from so101_rl.safety.types import RESUME_BY_CAUSE, ResumePolicy, SmState, TripCause, Verdict


def test_trip_cause_values_are_distinct_bits():
    values = [c.value for c in TripCause if c is not TripCause.NONE]
    assert len(values) == 16
    assert all(v & (v - 1) == 0 for v in values)  # powers of two
    assert TripCause.NAN | TripCause.RANGE == 3
    assert TripCause.BOX_PRESENT == 32768 and TripCause.DEADMAN == 16384


def test_sm_state_order_frozen():
    assert [s.value for s in SmState] == list(range(9))
    assert SmState.RUN == 3 and SmState.LATCHED == 8


def test_resume_policy_table():
    assert RESUME_BY_CAUSE[TripCause.THERMAL] is ResumePolicy.AUTO
    assert RESUME_BY_CAUSE[TripCause.NAN] is ResumePolicy.FRESH_PROCESS
    assert RESUME_BY_CAUSE[TripCause.DEADMAN] is ResumePolicy.HUMAN_PHYSICAL_RESET
    assert ResumePolicy.FRESH_PROCESS.value == "fresh_process_and_checkpoint"
    # clamps are not state-machine trips
    for clamp in (TripCause.TRACKING, TripCause.JOINT_LIMIT, TripCause.BOX):
        assert clamp not in RESUME_BY_CAUSE


def test_verdict_dtypes():
    v = Verdict()
    assert v.target_ticks.dtype == np.int32 and v.goal_ticks.dtype == np.int32
    assert v.applied_delta_ticks.dtype == np.float32 and v.clamped_mask.dtype == bool
    assert v.target_ticks.shape == (6,) and v.trips is TripCause.NONE
    assert Verdict().target_ticks is not v.target_ticks  # no shared default arrays


SAFETY_DIR = Path(__file__).parent.parent / "src" / "so101_rl" / "safety"
FORBIDDEN = ("so101_rl.rl", "so101_rl.wm", "so101_rl.env")


def _safety_modules() -> list[str]:
    mods = []
    for path in sorted(SAFETY_DIR.rglob("*.py")):
        rel = path.relative_to(SAFETY_DIR).with_suffix("")
        parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
        mods.append(".".join(("so101_rl", "safety", *parts)))
    return mods


def test_safety_module_walk_finds_modules():
    assert "so101_rl.safety" in _safety_modules() and "so101_rl.safety.types" in _safety_modules()


def test_safety_does_not_import_rl_wm_env():
    """Import EVERY safety module (present and future) in a fresh interpreter and check sys.modules."""
    mods = _safety_modules()
    code = (
        "import importlib, sys\n"
        f"for m in {mods!r}: importlib.import_module(m)\n"
        f"bad = [m for m in sys.modules if m.startswith({FORBIDDEN!r})]\n"
        "assert not bad, bad\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr


def test_safety_sources_have_no_forbidden_import_statements():
    """Static check too: catches lazy imports inside functions that a plain import would not execute."""
    bad = []
    for path in sorted(SAFETY_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            for n in names:
                stripped = n.lstrip(".")
                if stripped.startswith(FORBIDDEN) or stripped.split(".")[0] in ("rl", "wm", "env"):
                    bad.append(f"{path.name}: {n}")
    assert not bad, bad
