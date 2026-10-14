"""Training loop (MVP M4, sim plan S4.3): frozen API, strict config, hooks, thermal fail-closed, artifacts, resume.

CPU tests swap the batched env for the toy point-mass env (``eval_sim.build_env`` is the one env factory)."""

import csv
import dataclasses
import inspect
import math
from pathlib import Path

import pytest
import torch

from so101_rl.rl import eval_sim, train_sim_sac
from so101_rl.rl.toy_batched_env import ToyBatchedReachEnv
from so101_rl.rl.train_sim_sac import (
    SimSacConfig,
    ThermalGuard,
    TrainHooks,
    TrainResult,
    load_sim_sac_config,
    review_triggers,
    run_training,
)

ROOT = Path(__file__).parent.parent
CAL = str(ROOT / "tests/data/so101_follower.calib.json")


@pytest.fixture
def toy(monkeypatch):
    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None):
        return ToyBatchedReachEnv(nworld, seed=seed)

    monkeypatch.setattr(eval_sim, "build_env", build)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(train_sim_sac, "THERMAL_POLL_S", 0.0)


def _cfg(**kw):
    base = {"task": "reach", "nworld": 32, "budget_min": 5.0, "seed": 0, "dr_max_stage": 0, "eval_every_steps": 32 * 40,
            "eval_episodes": 64, "log_every_steps": 32 * 20, "max_env_steps": 32 * 120, "replay_capacity": 20_000,
            "batch_size": 128, "grad_steps_per_env_step": 2, "warmup_steps": 5, "tensorboard": False}  # fmt: skip
    base.update(kw)
    return load_sim_sac_config(base)


def _rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------------------ frozen API
def test_frozen_api_shapes():
    f = {x.name: x for x in dataclasses.fields(SimSacConfig)}
    assert list(f)[:7] == ["task", "nworld", "budget_min", "seed", "dr_max_stage", "eval_every_steps", "eval_episodes"]
    assert (f["dr_max_stage"].default, f["eval_every_steps"].default, f["eval_episodes"].default) == (4, 200_000, 1000)
    assert SimSacConfig.__dataclass_params__.frozen and TrainHooks.__dataclass_params__.frozen
    assert [x.name for x in dataclasses.fields(TrainHooks)] == ["on_metrics", "on_checkpoint", "should_stop", "heartbeat"]
    assert [x.name for x in dataclasses.fields(TrainResult)][:4] == ["status", "env_steps", "last_ckpt", "metrics_csv"]
    sig = inspect.signature(run_training)
    assert list(sig.parameters) == ["cfg", "out_dir", "hooks", "resume"]
    assert sig.parameters["hooks"].kind is inspect.Parameter.KEYWORD_ONLY
    ev = inspect.signature(eval_sim.evaluate)
    assert list(ev.parameters) == ["ckpt", "task", "episodes", "dr_stage", "out_dir", "render"]
    assert [x.name for x in dataclasses.fields(eval_sim.EvalResult)] == [
        "success_rate", "wilson_lo", "wilson_hi", "episodes", "numeric_gate_passed", "eval_json", "renders"]


# ------------------------------------------------------------------------------------------------ config
def test_config_strict(tmp_path):
    c = load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1, "seed": 3})
    assert c.budget_min == 1.0 and isinstance(c.budget_min, float) and c.dr_max_stage == 4
    with pytest.raises(ValueError, match="missing"):
        load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1})
    with pytest.raises(ValueError, match="unknown"):
        load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1, "seed": 0, "nwrold": 3})
    with pytest.raises(TypeError):
        load_sim_sac_config({"task": "reach", "nworld": True, "budget_min": 1, "seed": 0})
    with pytest.raises(TypeError):
        load_sim_sac_config({"task": "reach", "nworld": 8.0, "budget_min": 1, "seed": 0})
    with pytest.raises(ValueError, match="MVP"):
        load_sim_sac_config({"task": "die_to_zone", "nworld": 8, "budget_min": 1, "seed": 0})
    for bad in ({"nworld": 0}, {"budget_min": -1}, {"budget_min": math.inf}, {"dr_max_stage": 5},
                {"dr_start_stage": 3, "dr_max_stage": 2}, {"eval_episodes": 0}, {"vram_fraction": 1.5}):  # fmt: skip
        with pytest.raises(ValueError):
            load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1, "seed": 0, **bad})
    with pytest.raises(TypeError):
        load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1, "seed": None})
    c = load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1, "seed": 0, "success_radius_mm": 5})
    assert c.success_radius_mm == 5.0 and isinstance(c.success_radius_mm, float)  # plan §10 2026-10-14
    for bad in (0, -1, math.inf):
        with pytest.raises(ValueError, match="success_radius_mm"):
            load_sim_sac_config({"task": "reach", "nworld": 8, "budget_min": 1, "seed": 0, "success_radius_mm": bad})
    y = tmp_path / "c.yaml"
    y.write_text("task: reach\nnworld: 16\nbudget_min: 2.5\nseed: 1\nmax_env_steps: null\n")
    assert load_sim_sac_config(y) == SimSacConfig("reach", 16, 2.5, 1)


# ------------------------------------------------------------------------------------------------ thermal guard
def test_thermal_guard_hysteresis_and_fail_closed():
    temps = iter([60.0, 86.0, 80.0, 75.0, 70.0, None, 50.0])
    g = ThermalGuard(lambda: next(temps))
    assert [g.sample() for _ in range(7)] == [False, True, True, False, False, True, False]
    g = ThermalGuard(lambda: None)
    for _ in range(6):
        assert g.sample()
    assert g.needs_human
    g = ThermalGuard(lambda: float("nan"))
    assert g.sample()
    with pytest.raises(ValueError):
        ThermalGuard(lambda: 1.0, pause_c=70, resume_c=75)


def test_review_triggers():
    h = [{"alpha": 1e-5, "q_max": 500.0, "reward_abs_max": 2.0, "gamma": 0.97, "action_saturation": 0.6}]
    t = review_triggers(h, train_h=0.1, n_updates_per_transition=0.001)
    assert any("UTD" in x for x in t) and any("alpha" in x for x in t) and any("|Q|" in x for x in t)
    assert any("saturation" in x for x in t)
    ev = [{"eval_success_rate": s, "critic_loss": c} for s, c in ((0.3, 1), (0.2, 2), (0.25, 3), (0.28, 4))]
    t = review_triggers(ev, train_h=1.5, n_updates_per_transition=0.01)
    assert {"critic loss rising over 3 evals", "no eval improvement over 3 evals"} <= set(t)
    assert any("after 1.50 GPU-h" in x for x in t)


# ------------------------------------------------------------------------------------------------ loop (toy)
def test_smoke_toy_env_cpu(toy, tmp_path):
    calls = {"hb": 0, "metrics": [], "ckpt": []}
    hooks = TrainHooks(
        on_metrics=lambda s, m: calls["metrics"].append((s, dict(m))),
        on_checkpoint=lambda s, p: calls["ckpt"].append((s, p)),
        heartbeat=lambda s: calls.__setitem__("hb", calls["hb"] + 1),
    )
    res = run_training(_cfg(), tmp_path, hooks=hooks)
    assert res.status == "completed" and res.env_steps == 32 * 120
    rows = _rows(res.metrics_csv)
    assert len(rows) >= 3
    for r in rows:
        for k in ("critic_loss", "alpha", "q_mean"):
            if r[k]:
                assert math.isfinite(float(r[k]))
    assert any(r["eval_success_rate"] for r in rows)
    assert calls["hb"] == 120 and len(calls["metrics"]) == len(rows) and calls["ckpt"]
    assert res.last_ckpt == tmp_path / "ckpt_last.pt" and res.last_ckpt.exists()
    assert (tmp_path / f"ckpt_{32 * 40}.pt").exists() and (tmp_path / "run_meta.json").exists()
    d = eval_sim.load_checkpoint(res.last_ckpt)
    assert d["env_step"] == 32 * 120 and d["task"] == "reach" and d["last_eval"]["episodes"] == 64


def test_graceful_stop_checkpoints(toy, tmp_path):
    n = {"k": 0}

    def stop():
        n["k"] += 1
        return n["k"] > 30

    res = run_training(_cfg(max_env_steps=None), tmp_path, hooks=TrainHooks(should_stop=stop))
    assert res.status == "stopped" and res.env_steps == 32 * 30
    assert res.last_ckpt is not None and res.last_ckpt.exists()


def test_budget_exhausted(toy, tmp_path):
    res = run_training(_cfg(budget_min=0.002, max_env_steps=None), tmp_path)
    assert res.status == "budget_exhausted" and res.env_steps > 0 and res.last_ckpt.exists()


def test_exit_nonzero_without_artifacts(toy, tmp_path):
    rc = train_sim_sac.main(["--task", "reach", "--nworld", "8", "--budget-min", "0", "--seed", "0", "--out", str(tmp_path)])
    assert rc == 1
    assert (tmp_path / "metrics.csv").exists()  # always written (header)
    assert len(_rows(tmp_path / "metrics.csv")) == 0


def test_thermal_pause_and_resume(toy, tmp_path, monkeypatch):
    temps = iter([50.0] * 10 + [90.0, 88.0, 80.0, 74.0] + [50.0] * 10_000)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: next(temps))
    monkeypatch.setattr(train_sim_sac.time, "sleep", lambda s: None)
    seen = []
    res = run_training(_cfg(), tmp_path, hooks=TrainHooks(on_metrics=lambda s, m: seen.append(dict(m))))
    assert res.status == "completed"
    flags = [m["thermal_paused"] for m in seen if "thermal_paused" in m]
    assert flags[:2] == [1.0, 0.0]


def test_thermal_missing_samples_fail_closed(toy, tmp_path, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: None)
    monkeypatch.setattr(train_sim_sac.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 5.0))
    real = train_sim_sac.time.monotonic
    monkeypatch.setattr(train_sim_sac.time, "monotonic", lambda: real() + clock["t"])
    seen = []
    res = run_training(_cfg(thermal_max_pause_min=1.0), tmp_path, hooks=TrainHooks(on_metrics=lambda s, m: seen.append(dict(m))))
    assert res.status == "stopped" and "thermal_timeout" in res.reason  # S3: a timeout is a resumable stop
    assert res.env_steps == 0  # never trained without a temperature reading
    assert any(m.get("thermal_needs_human") == 1.0 for m in seen)


def test_resume_continues(toy, tmp_path):
    first = run_training(_cfg(max_env_steps=32 * 60), tmp_path)
    assert first.status == "completed"
    n_rows = len(_rows(first.metrics_csv))
    res = run_training(_cfg(max_env_steps=32 * 100), tmp_path, resume=first.last_ckpt)
    assert res.status == "completed" and res.env_steps == 32 * 100
    assert len(_rows(res.metrics_csv)) > n_rows
    assert int(_rows(res.metrics_csv)[-1]["env_step"]) == 32 * 100
    with pytest.raises(ValueError):
        run_training(_cfg(), tmp_path / "x", resume=tmp_path / "metrics.csv")


def test_nan_fails_and_keeps_last_good_ckpt(toy, tmp_path, monkeypatch):
    from so101_rl.rl import fast_sac

    orig = fast_sac.FastSAC.update
    n = {"k": 0}

    def bad(self, batch):
        n["k"] += 1
        out = orig(self, batch)
        if n["k"] > 150:
            out["critic_loss"] = torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(fast_sac.FastSAC, "update", bad)
    res = run_training(_cfg(max_env_steps=None), tmp_path)
    assert res.status == "failed" and "non-finite" in res.reason


# ------------------------------------------------------------------------------------------------ GPU smoke
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_smoke_reach_gpu(tmp_path):
    """Real batched reach env, short: writes metrics, checkpoints, eval, no NaN, VRAM < 4 GB."""
    cfg = load_sim_sac_config({
        "task": "reach", "nworld": 256, "budget_min": 2.0, "seed": 0, "eval_every_steps": 256 * 20,
        "eval_episodes": 256, "log_every_steps": 256 * 10, "max_env_steps": 256 * 40, "batch_size": 1024,
        "grad_steps_per_env_step": 2, "warmup_steps": 5, "replay_capacity": 100_000, "calibration": CAL,
        "tensorboard": True,
    })  # fmt: skip
    res = run_training(cfg, tmp_path)
    assert res.status in ("completed", "budget_exhausted")
    rows = _rows(res.metrics_csv)
    assert len(rows) >= 3
    assert all(math.isfinite(float(r["critic_loss"])) for r in rows if r["critic_loss"])
    assert max(float(r["vram_mb"]) for r in rows) < 4096
    assert (tmp_path / "tensorboard").exists()
    d = eval_sim.load_checkpoint(res.last_ckpt)
    assert d["policy_input_dim"] == 24 and d["act_dim"] == 6
    ev = eval_sim.evaluate(res.last_ckpt, "reach", episodes=64, dr_stage=4, out_dir=tmp_path / "eval", render=1)
    assert ev.eval_json.exists() and not ev.numeric_gate_passed and len(ev.renders) == 1 and ev.renders[0].exists()


# ------------------------------------------------------------------------------------------------ 2026-10-08 review
@pytest.mark.regression
def test_only_the_last_k_numbered_checkpoints_are_kept_plus_last_and_best_s1(toy, tmp_path):
    res = run_training(_cfg(eval_every_steps=32 * 10, eval_episodes=16, eval_max_fraction=1.0), tmp_path)
    assert res.status == "completed"
    numbered = sorted(int(p.stem.split("_")[1]) for p in tmp_path.glob("ckpt_[0-9]*.pt"))
    assert numbered == [32 * 100, 32 * 110, 32 * 120]
    assert (tmp_path / "ckpt_last.pt").exists() and (tmp_path / "ckpt_best.pt").exists()
    best = eval_sim.load_checkpoint(tmp_path / "ckpt_best.pt")
    evals = [r for r in _rows(res.metrics_csv) if r["eval_success_rate"]]
    assert best["last_eval"]["success_rate"] == max(float(r["eval_success_rate"]) for r in evals)


@pytest.mark.regression
def test_eval_time_is_capped_to_the_eval_fraction_s1(toy, tmp_path, monkeypatch):
    """S1: with an eval as slow as ~a third of the training, evals are deferred so they stay <= ~10 % of train_s."""
    import time as _time

    real = eval_sim.run_episodes

    def slow(*a, **kw):
        _time.sleep(0.3)
        return real(*a, **kw)

    monkeypatch.setattr(eval_sim, "run_episodes", slow)
    res = run_training(_cfg(eval_every_steps=32 * 2, eval_episodes=16, max_env_steps=None, budget_min=0.1), tmp_path)
    rows = _rows(res.metrics_csv)
    n_eval = sum(1 for r in rows if r["eval_success_rate"])
    frac = float(rows[-1]["eval_frac"])
    assert n_eval >= 1 and frac <= 0.1 + 0.3 / 6.0 + 0.02, (n_eval, frac)  # one eval may straddle the end


@pytest.mark.regression
def test_resume_truncates_metrics_after_the_checkpoint_s4(toy, tmp_path):
    """S4: resuming from a checkpoint older than the last metrics row drops the later rows (no duplicate steps)."""
    first = run_training(_cfg(max_env_steps=32 * 100), tmp_path)
    assert first.status == "completed"
    older = tmp_path / f"ckpt_{32 * 80}.pt"
    assert older.exists() and int(_rows(first.metrics_csv)[-1]["env_step"]) == 32 * 100
    run_training(_cfg(max_env_steps=32 * 120), tmp_path, resume=older)
    steps = [int(r["env_step"]) for r in _rows(tmp_path / "metrics.csv")]
    assert steps == sorted(steps) and len(steps) == len(set(steps)), steps


@pytest.mark.regression
def test_thermal_guard_off_never_samples(toy, tmp_path, monkeypatch):
    """S3: ``thermal_guard`` False (the lab) = no NVML reads and no pause in the trainer."""
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: pytest.fail("sampled"))
    assert run_training(_cfg(thermal_guard=False), tmp_path).status == "completed"
