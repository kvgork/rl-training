"""The ``fastsac_sim`` lab trainer on the frozen ``rl.train_sim_sac`` API, the trainer's thermal pause as a known
pause, and the launcher's import pinning (MVP task list 2026-10-08).

CPU: the job runs in process with the toy batched env behind ``eval_sim.build_env`` (as in test_train_sim_sac).
GPU (``gpu`` + ``integration``): a short real reach run through the lab API in a detached job process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from so101_rl.lab import runs
from so101_rl.lab.api import Lab
from so101_rl.lab.job import Job, check_import_root
from so101_rl.lab.launcher import Launcher, pid_alive
from so101_rl.lab.runs import REPO_ROOT
from so101_rl.lab.trainers import get_trainer

TINY = {"nworld": 32, "budget_min": 5.0, "dr_max_stage": 0, "eval_every_steps": 1_280, "eval_episodes": 16,
        "log_every_steps": 1_000, "max_env_steps": 32 * 80}  # fmt: skip


def _run(root: Path, config: dict, trainer: str = "fastsac_sim") -> Path:
    cfg = get_trainer(trainer).validate(config, "reach")
    with runs.registry_lock(root):
        d, _ = runs.create_run(root, trainer=trainer, task="reach", kind="probe", config=cfg, budget_key="reach",
                               idempotency_key="k")
    return d


@pytest.fixture
def toy(monkeypatch):
    from so101_rl.rl import eval_sim, train_sim_sac
    from so101_rl.rl.toy_batched_env import ToyBatchedReachEnv

    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None):
        return ToyBatchedReachEnv(nworld, seed=seed)

    monkeypatch.setattr(eval_sim, "build_env", build)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(train_sim_sac, "THERMAL_POLL_S", 0.0)
    real_load = train_sim_sac.load_sim_sac_config
    # the lab does not expose these (CPU speed only): small batches, no TensorBoard
    monkeypatch.setattr(train_sim_sac, "load_sim_sac_config", lambda raw: real_load(
        {**raw, "batch_size": 128, "grad_steps_per_env_step": 1, "warmup_steps": 5, "replay_capacity": 20_000,
         "tensorboard": False}))  # fmt: skip
    return train_sim_sac


def test_list_tasks_shows_fastsac_sim_available(tmp_path):
    out = Lab(tmp_path / "runs", Launcher("subprocess", python=[sys.executable])).list_tasks()
    t = {x["name"]: x for x in out["trainers"]}
    assert t["fastsac_sim"]["available"] is True and t["fastsac_sim"]["gpu"] is True
    assert "fastsac_sim" in out["tasks"][0]["trainers"] and "100 steps" in out["tasks"][0]["description"]
    assert {"log_every_steps", "max_env_steps"} <= set(t["fastsac_sim"]["config"])


def test_unknown_or_out_of_range_fastsac_config_is_refused():
    from so101_rl.lab.errors import LabError

    tr = get_trainer("fastsac_sim")
    for bad in ({"calibration": "x"}, {"nworld": 0}, {"eval_every_steps": 10}, {"tensorboard": False}):
        with pytest.raises(LabError):
            tr.validate(bad, "reach")
    assert tr.validate({}, "reach")["max_env_steps"] is None


def test_fastsac_job_completes_with_metrics_heartbeat_and_checkpoint(toy, tmp_path):
    d = _run(tmp_path, TINY)
    assert Job(d, use_nvml=False).run_segment() == 0
    st = runs.read_status(d)
    assert st["state"] == "completed" and not st["active"] and st["env_step"] == 32 * 80
    assert st["last_ckpt"] == str(d / "ckpt_last.pt") and Path(st["last_ckpt"]).exists()
    assert (d / "metrics.csv").read_text().count("\n") >= 3 and (d / "heartbeat").exists()
    ev = runs.read_events(d, last=100)
    assert any(e["kind"] == "checkpoint" for e in ev)
    end = ev[-1]
    assert end["kind"] == "completed" and end["data"]["trainer_status"] == "completed"
    assert "max_env_steps" in end["data"]["trainer_reason"]
    assert runs.read_usage(d)["gpu_h"] > 0


def test_get_run_shows_the_key_fastsac_metric_columns(toy, tmp_path):
    from so101_rl.lab.api import KEY_METRIC_COLS

    d = _run(tmp_path, TINY)
    assert Job(d, use_nvml=False).run_segment() == 0
    m = Lab(tmp_path, Launcher("subprocess", python=[sys.executable])).get_run(d.name)["untrusted"]["metrics"]
    assert m["columns"] == list(KEY_METRIC_COLS)
    assert all(len(r) == len(KEY_METRIC_COLS) for r in m["rows"])
    assert any(r[m["columns"].index("eval_success_rate")] is not None for r in m["rows"])


def test_fastsac_stop_request_maps_to_stopped_with_checkpoint(toy, tmp_path):
    d = _run(tmp_path, {**TINY, "max_env_steps": None})
    j = Job(d, use_nvml=False)
    n = {"k": 0}
    orig = j.should_stop

    def stop():
        n["k"] += 1
        if n["k"] == 20:
            j._signal_stop = True
        return orig()

    j.should_stop = stop
    assert j.run_segment() == 0
    st = runs.read_status(d)
    assert st["state"] == "stopped" and st["stop_reason"] == "user_stop" and Path(st["last_ckpt"]).exists()


def test_trainer_failure_reason_lands_in_status(toy, tmp_path, monkeypatch):
    toy_mod = toy
    monkeypatch.setattr(toy_mod, "read_gpu_temp_c", lambda: None)  # fail-closed thermal, then a pause timeout
    monkeypatch.setattr(toy_mod.time, "sleep", lambda s: None)
    real_load = toy_mod.load_sim_sac_config
    monkeypatch.setattr(toy_mod, "load_sim_sac_config", lambda raw: real_load({**raw, "thermal_max_pause_min": 1e-6}))
    d = _run(tmp_path, TINY)
    assert Job(d, use_nvml=False).run_segment() == 1
    st = runs.read_status(d)
    assert st["state"] == "failed" and "thermal pause longer than" in st["error"]
    kinds = [e["kind"] for e in runs.read_events(d, last=100)]
    assert "thermal_hold" in kinds  # the trainer's pause report reached the job


def test_trainer_thermal_pause_is_a_known_pause_not_a_stall(tmp_path):
    """``thermal_paused`` = 1 from on_metrics: status paused/thermal_trainer, no progress time for the pause, the
    wall deadline moves by the pause; ``thermal_paused`` = 0 releases it."""
    d = _run(tmp_path, {"steps": 10, "step_s": 0.0}, trainer="fake")
    clk = {"t": 1000.0}
    j = Job(d, use_nvml=False, clock=lambda: clk["t"])
    deadline0 = j.deadline
    clk["t"] += 10
    j.heartbeat(100)
    assert j.progress_s == pytest.approx(10)
    j.on_metrics(100, {"thermal_paused": 1.0, "thermal_needs_human": 0.0, "gpu_temp_c": 86.0})
    st = runs.read_status(d)
    assert st["state"] == "paused" and st["pause_reason"] == "thermal_trainer"
    clk["t"] += 600  # ten minutes paused, no heartbeats
    j.on_metrics(100, {"thermal_paused": 0.0, "gpu_temp_c": 74.0})
    st = runs.read_status(d)
    assert st["state"] == "running" and st["pause_reason"] is None
    assert j.deadline == pytest.approx(deadline0 + 600)
    clk["t"] += 5
    j.heartbeat(200)
    assert j.progress_s == pytest.approx(15)  # the 600 s pause is not progress-holding time
    kinds = [e["kind"] for e in runs.read_events(d, last=20)]
    assert kinds.count("thermal_hold") == 1 and kinds.count("thermal_release") == 1


# ------------------------------------------------------------------------------------------------ import pinning
def test_launcher_pins_this_checkouts_src():
    e = Launcher("subprocess", python=[sys.executable]).env(Path("/tmp/r"), "u")
    assert e["PYTHONPATH"] == str(REPO_ROOT / "src") and e["SO101_LAB_REPO"] == str(REPO_ROOT)


def test_check_import_root():
    assert check_import_root(str(REPO_ROOT)) is None
    assert "not from the launching checkout" in check_import_root("/nonexistent/elsewhere")
    assert "SO101_LAB_REPO is not set" in check_import_root(None)


def test_job_runs_this_checkout_even_when_pythonpath_points_elsewhere(tmp_path, monkeypatch):
    """A foreign ``so101_rl`` first on the caller's PYTHONPATH does not reach the job: the launcher pins
    ``<repo>/src``, and the job completes."""
    fake = tmp_path / "elsewhere" / "so101_rl"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise ImportError('foreign so101_rl imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(fake.parent))
    lab = Lab(tmp_path / "runs", Launcher("subprocess", python=[sys.executable]))
    try:
        r = lab.start_run("fake", "reach", {"steps": 20, "step_s": 0.0}, "pin")
        out = lab.get_run(r["run_id"], wait_s=60)
        assert out["status"]["state"] == "completed", (lab.root / r["run_id"] / "job.log").read_text()
    finally:
        for d in runs.list_run_dirs(lab.root):
            st = runs.read_status(d)
            if st.get("active") and pid_alive(st.get("pid"), st.get("pid_start")):
                os.kill(st["pid"], signal.SIGKILL)


def test_job_refuses_a_foreign_import_root_loudly(tmp_path):
    d = _run(tmp_path, {"steps": 5, "step_s": 0.0}, trainer="fake")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), SO101_LAB_REPO=str(tmp_path / "other-checkout"))
    r = subprocess.run([sys.executable, "-m", "so101_rl.lab.job", str(d)], env=env, capture_output=True, text=True,
                       timeout=60, check=False)  # fmt: skip
    assert r.returncode == 3 and "refusing to run" in r.stderr and "other-checkout" in r.stderr
    st = runs.read_status(d)
    assert st["state"] == "failed" and st["error"].startswith("import pinning")


# ------------------------------------------------------------------------------------------------ GPU, real env
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.timeout(600)
def test_fastsac_reach_probe_through_the_lab_api(tmp_path):
    lab = Lab(tmp_path / "runs", Launcher("subprocess", python=[sys.executable]))
    cfg = {"nworld": 64, "budget_min": 3.0, "eval_every_steps": 64 * 50, "eval_episodes": 32,
           "log_every_steps": 64 * 16, "max_env_steps": 64 * 100}  # fmt: skip
    try:
        r = lab.start_run("fastsac_sim", "reach", cfg, "gpu-probe", kind="probe")
        out = lab.get_run(r["run_id"], wait_s=280)
        if out["status"]["active"]:
            out = lab.get_run(r["run_id"], wait_s=280)
        log = (lab.root / r["run_id"] / "job.log").read_text()[-3000:]
        assert out["status"]["state"] == "completed", log
        assert out["status"]["env_step"] == 64 * 100 and out["status"]["has_checkpoint"]
        m = out["untrusted"]["metrics"]
        assert m["columns"][0] == "env_step" and m["n_rows"] >= 3
        assert "eval_success_rate" in m["columns"] and "critic_loss" in m["columns"]
    finally:
        for d in runs.list_run_dirs(lab.root):
            st = runs.read_status(d)
            if st.get("active") and pid_alive(st.get("pid"), st.get("pid_start")):
                os.kill(st["pid"], signal.SIGKILL)
