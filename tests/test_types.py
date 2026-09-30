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


def test_safety_does_not_import_rl_wm_env():
    import subprocess
    import sys

    code = (
        "import sys, so101_rl.safety, so101_rl.safety.types; "
        "bad = [m for m in sys.modules if m.startswith(('so101_rl.rl', 'so101_rl.wm', 'so101_rl.env'))]; "
        "assert not bad, bad"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
