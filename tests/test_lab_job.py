"""lab.job (in process, fake trainer) and lab.thermal: heartbeat, progress-holding GPU-h, budget, pause, resume,
crash, and the P8 thermal fail-closed with a fake NVML sampler."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from so101_rl.lab import runs, thermal
from so101_rl.lab.job import Job
from so101_rl.lab.thermal import ThermalGuard


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeNvml:
    """Scripted samples; after the script runs out, the last one repeats."""

    def __init__(self, samples):
        self.samples = list(samples)
        self.calls = 0

    def sample(self):
        self.calls += 1
        return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]


def _run(root, config, key="k", kind="attempt"):
    with runs.registry_lock(root):
        d, _ = runs.create_run(root, trainer="fake", task="reach", kind=kind, config=config, budget_key="fake.reach",
                               idempotency_key=key)
    return d


def _cfg(**kw):
    from so101_rl.lab.trainers import get_trainer

    return get_trainer("fake").validate({"step_s": 0.0, **kw}, "reach")


def test_train_segment_completes(tmp_path):
    d = _run(tmp_path, _cfg(steps=120))
    assert Job(d, use_nvml=False).run_segment() == 0
    st = runs.read_status(d)
    assert st["state"] == "completed" and not st["active"] and st["env_step"] == 120
    assert st["last_ckpt"].endswith("ckpt_000000120.json")
    assert json.loads((d / "heartbeat").read_text())["env_step"] == 120
    rows = (d / "metrics.csv").read_text().splitlines()
    assert rows[0] == "env_step,wall_s,return,success_rate" and len(rows) == 13
    kinds = [e["kind"] for e in runs.read_events(d, last=100)]
    assert kinds[0] == "created" and "segment_started" in kinds and kinds[-1] == "completed"
    u = runs.read_usage(d)
    assert u["segments"][0]["final"] and u["gpu_h"] >= 0


def test_progress_time_counts_only_advancing_intervals_capped(tmp_path):
    d = _run(tmp_path, _cfg(steps=10))
    clk = Clock()
    j = Job(d, use_nvml=False, clock=clk)
    clk.t += 60
    j.heartbeat(1)  # advanced: 60 s
    clk.t += 600
    j.heartbeat(1)  # no progress: 0
    clk.t += 40 * 60
    j.heartbeat(2)  # a 40 min gap counts at most 30 min
    assert j.progress_s == pytest.approx(60 + 30 * 60)


def test_own_budget_stops_the_job(tmp_path):
    d = _run(tmp_path, _cfg(steps=10_000, budget_min=1.0))
    clk = Clock()
    j = Job(d, use_nvml=False, clock=clk)
    assert not j.should_stop()
    clk.t += 61
    j.heartbeat(5)
    assert j.should_stop() and j.stop_reason == "budget_exhausted"


def test_wall_deadline_stops_the_job(tmp_path):
    d = _run(tmp_path, _cfg(steps=10_000, budget_min=1.0))
    clk = Clock()
    j = Job(d, use_nvml=False, clock=clk)
    clk.t += 60 + 3600 + 1  # no progress at all, past budget + grace
    assert j.should_stop() and j.stop_reason == "deadline"


def test_pause_request_checkpoints_and_resume_continues(tmp_path):
    d = _run(tmp_path, _cfg(steps=300))
    runs.atomic_write_json(d / "control.json", {"action": "pause", "segment": 0})
    assert Job(d, use_nvml=False).run_segment() == 0
    st = runs.read_status(d)
    assert st["state"] == "paused" and st["pause_reason"] == "ai" and st["last_ckpt"]
    (d / "control.json").unlink()
    runs.update_status(d, segment=1, state="queued")
    assert Job(d, use_nvml=False).run_segment() == 0
    st = runs.read_status(d)
    assert st["state"] == "completed" and st["env_step"] == 300
    assert [s["seg"] for s in runs.read_usage(d)["segments"]] == [0, 1]


def test_stale_control_from_an_older_segment_is_ignored(tmp_path):
    d = _run(tmp_path, _cfg(steps=20))
    runs.update_status(d, segment=1, last_ckpt=None)
    runs.atomic_write_json(d / "control.json", {"action": "stop", "segment": 0})
    Job(d, use_nvml=False).run_segment()
    assert runs.read_status(d)["state"] == "completed"


def test_stop_request_sets_stop_reason(tmp_path):
    d = _run(tmp_path, _cfg(steps=300))
    runs.atomic_write_json(d / "control.json", {"action": "stop", "segment": 0})
    Job(d, use_nvml=False).run_segment()
    st = runs.read_status(d)
    assert st["state"] == "stopped" and st["stop_reason"] == "user_stop"


def test_crash_marks_failed_with_error(tmp_path):
    d = _run(tmp_path, _cfg(steps=100, crash_at=7))
    assert Job(d, use_nvml=False).run_segment() == 1
    st = runs.read_status(d)
    assert st["state"] == "failed" and not st["active"] and "fake crash at step 7" in st["error"]


def test_eval_job_writes_result_and_gate_verdict(tmp_path):
    src = _run(tmp_path, _cfg(steps=20, success_rate=0.99), key="t")
    Job(src, use_nvml=False).run_segment()
    ckpt = runs.read_status(src)["last_ckpt"]
    with runs.registry_lock(tmp_path):
        e, _ = runs.create_run(tmp_path, trainer="fake", task="reach", kind="eval", budget_key="fake.reach",
                               config={"episodes": 1000, "dr_stage": 4, "render": 0, "ckpt": ckpt},
                               idempotency_key="e", extra={"source_run": src.name})
    assert Job(e, use_nvml=False).run_segment() == 0
    res = json.loads((e / "result.json").read_text())
    assert res["numeric_gate_passed"] is True and res["episodes"] == 1000 and res["wilson_lo"] >= 0.93
    assert (e / "eval.json").exists() and res["policy"]


# ----------------------------------------------------------------------------------------------- thermal (P8)
def test_thermal_thresholds_match_plan():
    assert (thermal.PAUSE_C, thermal.RESUME_C, thermal.MEM_PAUSE_C, thermal.MEM_RESUME_C, thermal.EMPTY_LIMIT) == (
        85.0, 75.0, 95.0, 85.0, 6)


def test_thermal_hysteresis_core_and_memory():
    g = ThermalGuard(FakeNvml([(50.0, None)]))
    assert g.update(84.9, None) is None
    assert g.update(85.0, None) == "hot"
    assert g.update(80.0, None) == "hot"  # between 75 and 85: still held
    assert g.update(75.0, None) is None
    assert g.update(60.0, 95.0) == "hot"  # memory junction, when observable
    assert g.update(60.0, 86.0) == "hot"
    assert g.update(60.0, 85.0) is None


def test_thermal_fail_closed_on_six_empty_samples():
    g = ThermalGuard(FakeNvml([(None, None)]))
    for _ in range(5):
        assert g.update(None, None) is None
    assert g.update(None, None) == "unknown"
    assert g.update(80.0, None) == "unknown"  # a valid but not cool sample does not release
    assert g.update(70.0, None) is None


def test_nvml_sampler_failure_is_an_empty_sample(monkeypatch):
    import types

    fake = types.ModuleType("pynvml")

    def boom():
        raise RuntimeError("driver down")

    fake.nvmlInit = boom
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert thermal.NvmlSampler().sample() == (None, None)


def test_job_holds_while_hot_and_does_not_count_the_hold(tmp_path):
    d = _run(tmp_path, _cfg(steps=50))
    clk = Clock()
    seen = []
    nv = FakeNvml([(90.0, None), (80.0, None), (74.0, None)])

    def sleep(s):
        clk.t += s
        seen.append(runs.read_status(d)["state"])

    j = Job(d, thermal=ThermalGuard(nv, interval_s=5.0, clock=clk, sleep=sleep), clock=clk)
    clk.t += 10
    j.heartbeat(1)  # samples 90 °C -> hold; sleeps through 80 °C; releases at 74 °C
    assert seen and seen[0] == "paused"
    assert runs.read_status(d)["state"] == "running" and runs.read_status(d)["hold"] is None
    kinds = [e["kind"] for e in runs.read_events(d, last=100)]
    assert "thermal_hold" in kinds and "thermal_release" in kinds
    progress = j.progress_s
    clk.t += 1
    j.heartbeat(2)
    assert j.progress_s == pytest.approx(progress + 1)  # the 10 s hold is not progress-holding time


def test_job_thermal_unknown_holds_and_raises_needs_human_then_times_out(tmp_path):
    d = _run(tmp_path, _cfg(steps=50))
    clk = Clock()

    def sleep(s):
        clk.t += s

    g = ThermalGuard(FakeNvml([(None, None)]), interval_s=5.0, max_hold_s=60.0, clock=clk, sleep=sleep)
    j = Job(d, thermal=g, clock=clk)
    for i in range(1, 7):
        clk.t += 5
        j.heartbeat(i)
    assert j.stop_reason == "thermal_timeout" and j.should_stop()
    kinds = [e["kind"] for e in runs.read_events(d, last=100)]
    assert "needs_human" in kinds
    assert runs.read_status(d)["pause_reason"] == "thermal_unknown"


def test_full_thermal_job_stops_on_timeout(tmp_path):
    d = _run(tmp_path, _cfg(steps=50))
    g = ThermalGuard(FakeNvml([(99.0, None)]), interval_s=0.0, max_hold_s=0.05)
    assert Job(d, thermal=g).run_segment() == 0
    st = runs.read_status(d)
    assert st["state"] == "stopped" and st["stop_reason"] == "thermal_timeout" and st["last_ckpt"]


def test_pause_request_during_thermal_hold_is_honoured(tmp_path):
    d = _run(tmp_path, _cfg(steps=50))
    runs.atomic_write_json(d / "control.json", {"action": "pause", "segment": 0})
    g = ThermalGuard(FakeNvml([(99.0, None)]), interval_s=0.0, max_hold_s=60.0)
    j = Job(d, thermal=g)
    j.heartbeat(1)
    assert j.stop_reason == "ai_pause"


def test_fake_job_process_never_imports_torch(tmp_path):
    code = ("import sys, so101_rl.lab.job, so101_rl.lab.mcp_server, so101_rl.lab.cli; "
            "print(sorted(m for m in ('torch', 'mujoco', 'warp', 'serial', 'pyrealsense2') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]"
