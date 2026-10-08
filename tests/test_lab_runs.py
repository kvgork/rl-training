"""lab.runs: run dirs, atomic files, idempotency, the closed state set, and the P6 ledger / budget checks."""

from __future__ import annotations

import json
import shutil

import pytest

from so101_rl.lab import runs
from so101_rl.lab.errors import LabError


def _create(root, key="k", kind="attempt", config=None, budget_key="fake.reach"):
    with runs.registry_lock(root):
        return runs.create_run(root, trainer="fake", task="reach", kind=kind, config=config or {"budget_min": 6.0},
                               budget_key=budget_key, idempotency_key=key)


def test_create_run_layout_and_provenance(tmp_path):
    d, replayed = _create(tmp_path)
    assert not replayed and runs.RUN_ID_RE.match(d.name)
    for f in ("run.json", "status.json", "usage.json", "events.jsonl"):
        assert (d / f).is_file(), f
    run = runs.read_run(d)
    assert run["task"] == "reach" and run["trainer"] == "fake" and run["kind"] == "attempt"
    prov = run["provenance"]
    assert set(prov) == {"git_sha", "git_dirty", "config_sha256", "sim_params_version", "camera_pose_method"}
    assert prov["config_sha256"] == runs.canonical_sha256({"budget_min": 6.0})
    assert runs.read_status(d)["state"] == "queued"
    assert runs.read_usage(d)["attempts"] == 1
    assert not list(d.glob(".*.tmp"))  # atomic writes leave no temp files


def test_idempotency_replay_and_mismatch(tmp_path):
    d1, _ = _create(tmp_path, key="same")
    d2, replayed = _create(tmp_path, key="same")
    assert replayed and d2 == d1
    with pytest.raises(LabError) as e:
        _create(tmp_path, key="same", config={"budget_min": 7.0})
    assert e.value.code == "IDEMPOTENCY_MISMATCH" and e.value.details["run_id"] == d1.name
    assert len(runs.list_run_dirs(tmp_path)) == 1


def test_idempotency_key_bounds(tmp_path):
    for bad in ("", "x" * 129):
        with pytest.raises(LabError) as e:
            _create(tmp_path, key=bad)
        assert e.value.code == "INVALID_ARGUMENT"


def test_dir_without_run_json_is_not_a_run(tmp_path):
    (tmp_path / "fake-reach-20260101-000000-abcd").mkdir()
    assert runs.list_run_dirs(tmp_path) == []
    with pytest.raises(LabError) as e:
        runs.run_dir_for(tmp_path, "fake-reach-20260101-000000-abcd")
    assert e.value.code == "NOT_FOUND"


@pytest.mark.parametrize("bad", ["../etc", "x", "fake-reach-2026-1", "FAKE-reach-20260101-000000-abcd", 3])
def test_run_id_grammar_refuses_paths(tmp_path, bad):
    with pytest.raises(LabError) as e:
        runs.run_dir_for(tmp_path, bad)
    assert e.value.code == "INVALID_ARGUMENT"


def test_status_state_set_is_closed(tmp_path):
    d, _ = _create(tmp_path)
    assert set(runs.STATES) == {"queued", "running", "paused", "stopped", "completed", "failed", "interrupted"}
    with pytest.raises(ValueError):
        runs.update_status(d, state="finished")
    s0 = runs.read_status(d)["seq"]
    assert runs.update_status(d, state="running")["seq"] == s0 + 1


def test_events_seq_monotonic(tmp_path):
    d, _ = _create(tmp_path)
    seqs = [runs.append_event(d, "x", i=i) for i in range(5)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5
    assert [e["seq"] for e in runs.read_events(d, last=100)] == list(range(1, 7))


def test_interrupted_only_when_active_and_dead(tmp_path):
    d, _ = _create(tmp_path)
    assert runs.mark_interrupted_if_dead(d, lambda st: False)["state"] == "queued"  # not active: untouched
    runs.update_status(d, state="running", active=True)
    assert runs.mark_interrupted_if_dead(d, lambda st: True)["state"] == "running"
    st = runs.mark_interrupted_if_dead(d, lambda st: False)
    assert st["state"] == "interrupted" and not st["active"]
    assert runs.read_events(d)[-1]["kind"] == "interrupted"


# ----------------------------------------------------------------------------------------------- ledger / P6
def test_ledger_counts_attempts_probes_evals(tmp_path):
    a, _ = _create(tmp_path, key="a")
    p, _ = _create(tmp_path, key="p", kind="probe")
    e, _ = _create(tmp_path, key="e", kind="eval")
    for d, h in ((a, 1.5), (p, 0.1), (e, 0.2)):
        u = runs.read_usage(d)
        u["gpu_h"] = h
        runs.atomic_write_json(d / "usage.json", u)
    led = runs.ledger(tmp_path)["fake.reach"]
    assert led == {"attempts_used": 1, "gpu_h_used": 1.5, "probe_gpu_h_used": 0.1, "eval_gpu_h": 0.2,
                   "gpu_h_reserved": 0.0, "probe_gpu_h_reserved": 0.0}  # budgets already used up: nothing reserved


def test_budget_attempts_cap_and_gpu_h():
    b = runs.BUDGETS["reach"]
    assert (b.max_attempts, b.gpu_h_max, b.probe_max_min, b.probe_gpu_h_max) == (3, 6.0, 10.0, 1.0)  # plan §10 P6
    runs.check_budget({"k": {"attempts_used": 2, "gpu_h_used": 0.0, "probe_gpu_h_used": 0.0}}, "k", b, "attempt", 1)
    with pytest.raises(LabError) as e:
        runs.check_budget({"k": {"attempts_used": 3, "gpu_h_used": 0.0, "probe_gpu_h_used": 0.0}}, "k", b, "attempt", 1)
    assert e.value.code == "BUDGET_EXHAUSTED"
    with pytest.raises(LabError) as e:
        runs.check_budget({"k": {"attempts_used": 1, "gpu_h_used": 5.5, "probe_gpu_h_used": 0.0}}, "k", b, "attempt", 1)
    assert e.value.code == "BUDGET_EXHAUSTED" and e.value.details["gpu_h_max"] == 6.0
    runs.check_budget({}, "k", b, "attempt", 6.0)  # exactly the budget is allowed


def test_probe_limits_are_not_attempts():
    b = runs.BUDGETS["reach"]
    full = {"k": {"attempts_used": 3, "gpu_h_used": 6.0, "probe_gpu_h_used": 0.0}}
    runs.check_budget(full, "k", b, "probe", 10 / 60)  # attempts exhausted: a probe is still fine
    with pytest.raises(LabError) as e:
        runs.check_budget(full, "k", b, "probe", 11 / 60)
    assert e.value.code == "CONFIG_INVALID"
    with pytest.raises(LabError) as e:
        runs.check_budget({"k": {"attempts_used": 0, "gpu_h_used": 0, "probe_gpu_h_used": 0.95}}, "k", b, "probe", 0.1)
    assert e.value.code == "BUDGET_EXHAUSTED"
    runs.check_budget(full, "k", b, "eval", 100.0)  # evals are never refused for budget


def test_ledger_halts_when_attempts_shrink(tmp_path):
    a, _ = _create(tmp_path, key="a")
    _create(tmp_path, key="b")
    assert runs.checked_ledger(tmp_path)["fake.reach"]["attempts_used"] == 2
    assert json.loads((tmp_path / "ledger.json").read_text())["attempts_high_water"] == {"fake.reach": 2}
    shutil.rmtree(a)  # a run dir disappears: the derived ledger must not silently give the attempt back
    with pytest.raises(LabError) as e:
        runs.checked_ledger(tmp_path)
    assert e.value.code == "HALTED" and e.value.details["reason"] == "ledger_shrink"
    with pytest.raises(LabError):
        runs.check_halt(tmp_path)


def test_launch_failed_attempt_is_not_counted(tmp_path):
    d, _ = _create(tmp_path)
    runs.update_status(d, state="failed", launch_failed=True)
    assert runs.ledger(tmp_path)["fake.reach"]["attempts_used"] == 0


def test_run_id_with_trailing_newline_is_refused(tmp_path):
    d, _ = _create(tmp_path)
    with pytest.raises(LabError) as e:
        runs.run_dir_for(tmp_path, d.name + "\n")
    assert e.value.code == "INVALID_ARGUMENT"


def test_torn_last_event_line_does_not_swallow_the_next(tmp_path):
    d, _ = _create(tmp_path)
    with (d / "events.jsonl").open("a") as f:
        f.write('{"seq": 99, "kind": "torn"')  # a crash mid-write
    runs.append_event(d, "after")
    assert runs.read_events(d, last=1)[0]["kind"] == "after"
