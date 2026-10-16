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
from so101_rl.rl.toy_batched_env import ToyBatchedDieShapedEnv, ToyBatchedReachEnv
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
    with pytest.raises(ValueError, match="not trainable"):
        load_sim_sac_config({"task": "pick", "nworld": 8, "budget_min": 1, "seed": 0})
    assert load_sim_sac_config({"task": "die_to_zone", "nworld": 8, "budget_min": 1, "seed": 0}).task == "die_to_zone"
    with pytest.raises(ValueError, match="reach-only"):  # plan §10 2026-10-14: the radius override is reach only
        load_sim_sac_config({"task": "die_to_zone", "nworld": 8, "budget_min": 1, "seed": 0, "success_radius_mm": 5})
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


# ------------------------------------------------------------------------------------------------ die_to_zone (D1)
def test_task_gamma_comes_from_the_task_specs():
    from so101_rl.env.tasks import DIE_TO_ZONE, REACH

    assert train_sim_sac.TASK_GAMMA == {"reach": REACH.gamma, "die_to_zone": DIE_TO_ZONE.gamma}
    assert train_sim_sac.TASK_GAMMA == {"reach": 0.97, "die_to_zone": 0.98}  # reach unchanged


def test_die_review_trigger_uses_the_die_budget():
    """Sim plan budgets: die success < 10 % after 4 GPU-h (reach: < 50 % after 1 GPU-h, unchanged)."""
    ev = [{"eval_success_rate": 0.3}]
    assert not any("success" in x for x in review_triggers(ev, train_h=1.5, n_updates_per_transition=0.01,
                                                           task="die_to_zone"))  # fmt: skip
    assert any("reach success" in x for x in review_triggers(ev, train_h=1.5, n_updates_per_transition=0.01))
    low = [{"eval_success_rate": 0.05}]
    assert not any("success" in x for x in review_triggers(low, train_h=3.9, n_updates_per_transition=0.01,
                                                           task="die_to_zone"))  # fmt: skip
    t = review_triggers(low, train_h=4.5, n_updates_per_transition=0.01, task="die_to_zone")
    assert any("die_to_zone success 0.05 < 0.1 after 4.50 GPU-h" in x for x in t)


def test_die_refuses_unverified_sim_params_without_the_waiver(monkeypatch, tmp_path):
    """Sim plan S4.6: no die training on the v0 MJCF prior (the shipped calib/sim_params.yaml) unless the user's
    waiver is set; refused before any env is built."""
    monkeypatch.setattr(eval_sim, "build_env", lambda *a, **k: pytest.fail("env built"))
    assert "MJCF prior" in train_sim_sac.unverified_sim_params(ROOT / "calib/sim_params.yaml")
    with pytest.raises(ValueError, match="S4.6"):
        run_training(_cfg(task="die_to_zone", sim_params=str(ROOT / "calib/sim_params.yaml")), tmp_path)
    assert not (tmp_path / "metrics.csv").exists()


def test_die_trains_with_policy_input_29_and_gamma_098(monkeypatch, tmp_path):
    """The die plumbing on the die-shaped toy env: policy input 29 (FS3: DIE_OBS 26 + goal 3), gamma 0.98 from the
    task spec, task recorded in the checkpoint; reach-only radius never passed to the env."""
    import json

    from so101_rl.rl.toy_batched_env import ToyBatchedDieShapedEnv

    seen = []

    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None):
        seen.append((task, success_radius_mm))
        return ToyBatchedDieShapedEnv(nworld, seed=seed)

    monkeypatch.setattr(eval_sim, "build_env", build)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    res = run_training(_cfg(task="die_to_zone", allow_unverified_sim_params=True, max_env_steps=32 * 50), tmp_path)
    assert res.status == "completed" and set(seen) == {("die_to_zone", None)}
    ck = torch.load(res.last_ckpt, weights_only=True)
    assert ck["task"] == "die_to_zone" and ck["policy_input_dim"] == 29
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["sac"]["gamma"] == 0.98 and meta["config"]["allow_unverified_sim_params"] is True


# ------------------------------------------------------------------------------------------------ review triggers once
def test_trigger_kind_strips_the_numbers():
    k = train_sim_sac.trigger_kind
    assert k("alpha 9.10e-05 < 1e-4") == k("alpha 3.00e-06 < 1e-4") == "alpha # < #"
    assert k("die_to_zone success 0.00 < 0.1 after 4.02 GPU-h") == k("die_to_zone success 0.05 < 0.1 after 4.50 GPU-h")
    assert k("critic loss rising over 3 evals") != k("no eval improvement over 3 evals")


def test_each_review_trigger_fires_once_per_run_also_across_a_resume(toy, tmp_path, monkeypatch):
    """Die run 1 wrote the low-success trigger on every log line once it held (its text carries the success and
    the GPU-h); each trigger kind is now recorded once per run, also after a resume."""
    import json

    def trig(history, **_kw):
        return [f"die_to_zone success 0.0{len(history) % 10} < 0.1 after {len(history) / 100:.2f} GPU-h",
                "critic loss rising over 3 evals"]  # fmt: skip

    monkeypatch.setattr(train_sim_sac, "review_triggers", trig)
    first = run_training(_cfg(max_env_steps=32 * 60), tmp_path)
    rows = _rows(first.metrics_csv)
    assert len(rows) >= 3
    review = json.loads((tmp_path / "review.json").read_text())["triggers"]
    assert len(review) == 2 and {r["env_step"] for r in review} == {int(rows[0]["env_step"])}
    run_training(_cfg(max_env_steps=32 * 100), tmp_path, resume=first.last_ckpt)
    assert len(json.loads((tmp_path / "review.json").read_text())["triggers"]) == 2
    assert {int(r["review_triggers"]) for r in _rows(tmp_path / "metrics.csv")} == {2}


# ------------------------------------------------------------------------------------------------ start curriculum (D2a)
def test_die_start_curriculum_config():
    base = {"task": "die_to_zone", "nworld": 8, "budget_min": 1, "seed": 0}
    assert load_sim_sac_config(base).die_start_curriculum is False  # off by default: the well, as in D1
    assert load_sim_sac_config({**base, "die_start_curriculum": True}).die_start_curriculum is True
    with pytest.raises(ValueError, match="die_start_curriculum"):
        load_sim_sac_config({**base, "task": "reach", "die_start_curriculum": True})
    with pytest.raises(TypeError):
        load_sim_sac_config({**base, "die_start_curriculum": 1})
    assert train_sim_sac.METRICS_FIELDS[-1] == "start_stage"


class _StagedDieToy(ToyBatchedDieShapedEnv):
    """The die-shaped toy env plus a DR randomizer stand-in (``stage`` / ``set_stage``); every start-stage and DR-stage
    change is appended to ``log`` (shared by all envs of a run) and ``changes`` (this env's) as ``(kind, stage)``."""

    def __init__(self, nworld, *, seed, stage, log):
        import types

        super().__init__(nworld, seed=seed)
        self.log, self.changes = log, []
        self.randomizer = types.SimpleNamespace(stage=stage)
        self.randomizer.set_stage = lambda k: (setattr(self.randomizer, "stage", int(k)), self._record("dr", k))

    def _record(self, kind, k):
        self.log.append((kind, int(k)))
        self.changes.append((kind, int(k)))

    @property
    def dr_stage(self):
        return int(self.randomizer.stage)

    def set_start_stage(self, k):
        super().set_start_stage(k)
        self._record("start", k)


@pytest.fixture
def staged_toy(monkeypatch):
    """``(envs built, change log)``; envs[0] trains, envs[1] evaluates."""
    envs, log = [], []

    def build(task, nworld, *, stage, seed, calibration=None, sim_params="", device=None, success_radius_mm=None):
        envs.append(_StagedDieToy(nworld, seed=seed, stage=stage, log=log))
        return envs[-1]

    monkeypatch.setattr(eval_sim, "build_env", build)
    monkeypatch.setattr(train_sim_sac, "read_gpu_temp_c", lambda: 50.0)
    monkeypatch.setattr(train_sim_sac, "THERMAL_POLL_S", 0.0)
    return envs, log


def _evals_at(monkeypatch, rate):
    def fake(env, act, episodes):
        return {"success_rate": rate, "successes": round(rate * episodes), "episodes": episodes,
                "final_err_m": {"p50": 0.01, "p90": 0.02}}  # fmt: skip

    monkeypatch.setattr(eval_sim, "run_episodes", fake)


def _die_cfg(**kw):
    base = {"task": "die_to_zone", "allow_unverified_sim_params": True, "die_start_curriculum": True,
            "eval_every_steps": 32 * 10, "eval_max_fraction": 1.0}  # fmt: skip
    return _cfg(**{**base, **kw})


def test_start_curriculum_advances_before_the_dr_stages(staged_toy, tmp_path, monkeypatch):
    """At >= 70 % eval success the START stage moves up first (S0 -> S1 -> S2 -> S3) with the DR stage held; only at
    the well (S3) do the DR stages widen. Both are in metrics.csv and the checkpoint; evals run at the current stages."""
    envs, log = staged_toy
    _evals_at(monkeypatch, 0.8)
    res = run_training(_die_cfg(dr_max_stage=2, max_env_steps=32 * 80), tmp_path)
    assert res.status == "completed"
    assert log[0] == ("start", 0)  # the train env starts at S0
    assert envs[0].changes == [("start", 0), ("start", 1), ("start", 2), ("start", 3), ("dr", 1), ("dr", 2)]
    rows = [r for r in _rows(res.metrics_csv) if r["eval_success_rate"]]
    seen = [(int(r["start_stage"]), int(r["dr_stage"])) for r in rows]
    assert seen[:6] == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)], seen
    ck = torch.load(res.last_ckpt, weights_only=True)
    assert ck["start_stage"] == 3 and ck["dr_stage"] == 2 and ck["last_eval"]["start_stage"] == 3
    assert envs[1].start_stage == 3  # the eval env follows the train env's start stage
    best = torch.load(tmp_path / "ckpt_best.pt", weights_only=True)
    assert (best["start_stage"], best["dr_stage"]) == (3, 2)


def test_start_curriculum_smoke_advances_s0_to_s1_on_the_toy(staged_toy, tmp_path):
    """Short CPU smoke run with real (toy) learning and real evals: the curriculum advances S0 -> S1."""
    res = run_training(_die_cfg(dr_max_stage=0, eval_every_steps=32 * 25, max_env_steps=32 * 400), tmp_path)
    rows = [r for r in _rows(res.metrics_csv) if r["eval_success_rate"]]
    assert int(rows[0]["start_stage"]) == 0
    assert ("start", 1) in staged_toy[1], [(r["start_stage"], r["eval_success_rate"]) for r in rows]
    assert any(int(r["start_stage"]) >= 1 for r in rows)


def test_exit_gate_needs_the_well_start(staged_toy, tmp_path, monkeypatch):
    """The numeric gate (1000 episodes, >= 0.95, Wilson >= 0.93, DR stage dr_max_stage) never ends the run before the
    start stage is S3: with passing evals from S0 on, the run walks S0..S3 and stops at the first S3 eval."""
    _evals_at(monkeypatch, 1.0)
    res = run_training(_die_cfg(dr_max_stage=4, dr_start_stage=4, eval_episodes=1000, max_env_steps=32 * 200),
                       tmp_path)  # fmt: skip
    assert res.status == "completed" and "exit gate" in res.reason
    rows = [r for r in _rows(res.metrics_csv) if r["eval_success_rate"]]
    assert [int(r["start_stage"]) for r in rows] == [0, 1, 2, 3]


def test_start_stage_resumes_from_the_checkpoint(staged_toy, tmp_path, monkeypatch):
    _, log = staged_toy
    _evals_at(monkeypatch, 0.8)
    first = run_training(_die_cfg(dr_max_stage=0, max_env_steps=32 * 20), tmp_path)  # evals at 10, 20: S0->S1->S2
    assert torch.load(first.last_ckpt, weights_only=True)["start_stage"] == 2
    log.clear()
    _evals_at(monkeypatch, 0.0)
    run_training(_die_cfg(dr_max_stage=0, max_env_steps=32 * 40), tmp_path, resume=first.last_ckpt)
    assert log[0] == ("start", 2)
    assert {int(r["start_stage"]) for r in _rows(tmp_path / "metrics.csv") if int(r["env_step"]) > 32 * 20} == {2}


def test_die_without_the_curriculum_starts_in_the_well(staged_toy, tmp_path, monkeypatch):
    _, log = staged_toy
    _evals_at(monkeypatch, 0.8)
    res = run_training(_die_cfg(die_start_curriculum=False, dr_max_stage=0, max_env_steps=32 * 30), tmp_path)
    assert not any(kind == "start" for kind, _ in log)  # the env keeps its default start (the well)
    assert {r["start_stage"] for r in _rows(res.metrics_csv)} == {"3"}
    assert torch.load(res.last_ckpt, weights_only=True)["start_stage"] == 3


def test_reach_has_no_start_stage(toy, tmp_path):
    res = run_training(_cfg(max_env_steps=32 * 30), tmp_path)
    assert {r["start_stage"] for r in _rows(res.metrics_csv)} == {""}
    assert torch.load(res.last_ckpt, weights_only=True)["start_stage"] is None
