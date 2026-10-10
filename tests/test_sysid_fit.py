"""Sysid replay + CMA-ES fit (FS7, sim-first track S2.3) on synthetic ground truth.

The "real" data comes from the in-process ServoCore + MujocoBus rig (``run_excitation.run_synthetic``) with known
parameters, so every claim is checked against the truth.

Parametrisation (plan §10 2026-10-06): the default fit is ONE scale per joint on kp, kv and armature jointly (the
servo ratio group) plus the integer delay; frictionloss stays at the prior and is reported unidentified. The held-out
gate is ``MAX_RMS_PCT`` = 1 % (was 5 %).

Physics findings pinned here (measured 2026-10-06, reported to the plan):

* The MJCF STS3215 servo is a stiff position servo (kp 998) with a hard force limit (3.35 N m): its linear band is
  about ±2 ticks. Through the clamp chain (45 ticks per 10 Hz command) kp, kv and armature act almost only through
  their ratios (the servo behaves nearly the same when all three are scaled together) and frictionloss is invisible
  under ±1-tick noise. The fit therefore cannot promise every parameter within ±10 %; it promises that every
  parameter is either within tolerance or reported unidentified with a spread that covers the truth (S3 then
  randomises it widely), that the delay is exact, and that the fitted model explains held-out data as well as the
  ground truth does.
* A 3-tick (100 ms) delay can move the open-loop held-out RMS by at most about 45 ticks (one command's worth): about
  1.5 % of range, so the old 5 % gate was blind to it; the 1 % gate fails it (the "delay fixed at 0" twin).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES
from so101_rl.sim.compare_real import MAX_RMS_PCT
from so101_rl.sim.model import load_model
from so101_rl.sim.sysid.dataset import ExcitationDataset
from so101_rl.sim.sysid.excitation import ALL, AMPLITUDE_TICKS, ExcitationPlan, Segment, fit40_plan
from so101_rl.sim.sysid.fit import fit_sim_params, objective, objective_rms_pct
from so101_rl.sim.sysid.params import SERVO_GROUP_FIELDS, ServoGroup, SimParams
from so101_rl.sim.sysid.replay import TickMap, pooled_rms_pct, replay_segment_cpu
from so101_rl.sim.sysid.run_excitation import run_synthetic

HELDOUT_KINDS = {"chirp", "random"}
DATA = Path(__file__).parent / "data"
# truth: the servo ratio group scaled per joint (kp, kv, armature together), 3-tick delay; frictionloss x2 (not fitted:
# it must not matter for the held-out error, and it is reported unidentified)
S_TRUTH = np.array([0.6, 0.7, 0.8, 1.25, 1.4, 0.75])
# the fitted group scale must lie within 15 % of the truth (measured 2026-10-06: worst 9.5 %, shoulder_pan), well
# inside the ±40 % DR band that is randomised around it
GROUP_TOL = math.log(1.15)


@pytest.fixture(scope="module")
def prior():
    return SimParams.from_mjcf(load_model()[0])


@pytest.fixture(scope="module")
def truth(prior):
    return replace(ServoGroup.identity(prior).apply(prior, S_TRUTH).scaled(frictionloss=2.0), delay_ticks=3)


@pytest.fixture(scope="module")
def truth_run(tmp_path_factory, truth):
    """fit40 plan (~40 s of excitation) through ServoCore + MujocoBus with the truth, ±1 tick noise on present."""
    from so101_rl.config import load_calibration, load_config

    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    out = tmp_path_factory.mktemp("sysid") / "truth"
    _, errors = run_synthetic(fit40_plan(cfg), cfg, cal, out, truth=truth, present_noise_ticks=1, seed=1)
    assert errors == []
    ds = ExcitationDataset.from_run(out)
    train, held = ds.split(heldout_kinds=HELDOUT_KINDS, seed=0)
    return cal, cfg, ds, train, held


@pytest.fixture(scope="module")
def sines_run(tmp_path_factory, prior):
    """Step-free multisine data, noise-free (for the identifiability tests)."""
    from so101_rl.config import load_calibration, load_config

    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    c = {"centre_ticks": (0, 200, 0, 0, 0, 400)}
    amp = tuple(0.45 * a for a in AMPLITUDE_TICKS)
    plan = ExcitationPlan((Segment("multisine", ALL, 10.0, amp, c), Segment("multisine", ALL, 6.0, amp, c)), 10.0, 3,
                          name="sines", hold_s=0.5)
    out = tmp_path_factory.mktemp("sysid") / "sines"
    _, errors = run_synthetic(plan, cfg, cal, out, truth=prior.scaled(kp=0.8), present_noise_ticks=0)
    assert errors == []
    train, held = ExcitationDataset.from_run(out).split(heldout_kinds=set(), heldout_fraction_by_time=0.25, seed=0)
    return cal, cfg, train, held


def _open_loop_heldout(params, held, cal, cfg):
    model, data = load_model()
    sims = [replay_segment_cpu(s, model, data, cal, cfg, params, shooting_window_ticks=len(s)) for s in held.segments]
    return pooled_rms_pct(sims, held.segments, cal)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_synthetic_recovery_cpu(truth_run, prior, truth):
    """Default fit (ratio group + delay): the delay exactly, the group scale per joint within 15 % of the truth (or
    reported unidentified with a spread that covers it), held-out under the 1 % gate and at the
    truth's noise floor; frictionloss untouched at the prior and flagged unidentified."""
    cal, cfg, _, train, held = truth_run
    res = fit_sim_params(train, held, prior=prior, backend="cpu", popsize=8, generations=25, seed=0,
                         calibration=cal, cfg=cfg)
    p = res.params
    s = res.group_scale
    assert s is not None
    print("\ngroup scale fitted / truth:", np.round(s, 3).tolist(), "/", S_TRUTH.tolist())
    print("spread:", {k: round(v, 2) for k, v in res.spread.items() if "." in k})
    print("held-out %:", {k: round(v, 3) for k, v in res.heldout_rms_pct.items()}, "evals", res.evals,
          f"{res.wall_s:.1f} s", "delay scores", {k: round(v, 3) for k, v in res.delay_scores.items()})

    assert p.delay_ticks == 3  # exact
    assert max(res.heldout_rms_pct.values()) <= MAX_RMS_PCT == 1.0
    # kp, kv, armature moved as ONE group: the prior's ratios are kept exactly
    for f in SERVO_GROUP_FIELDS:
        np.testing.assert_allclose(getattr(p, f) / getattr(prior, f), s, rtol=1e-12)
    # the group scale is recovered within 15 %, or flagged with a covering spread
    err = np.abs(np.log(s / S_TRUTH))
    silent = [f"{n}: x{s[j] / S_TRUTH[j]:.2f} of truth, spread {res.spread[f'servo_group.{n}']:.2f}"
              for j, n in enumerate(JOINT_NAMES) if err[j] > GROUP_TOL and err[j] > res.spread[f"servo_group.{n}"]]
    assert not silent, silent
    # the fit explains held-out data as well as the ground truth (noise floor) and far better than the prior
    floor = _open_loop_heldout(truth, held, cal, cfg)
    prior_err = _open_loop_heldout(prior, held, cal, cfg)
    fit_err = np.array([res.heldout_rms_pct[n] for n in JOINT_NAMES])
    assert (fit_err <= 1.25 * floor + 0.005).all(), (fit_err, floor)
    assert prior_err.max() > 10 * fit_err.max()
    # frictionloss, damping, forcerange are held at the prior; frictionloss is reported unidentified
    np.testing.assert_array_equal(p.frictionloss, prior.frictionloss)
    np.testing.assert_array_equal(p.damping, prior.damping)
    np.testing.assert_array_equal(p.forcerange, prior.forcerange)
    assert "frictionloss" in res.unidentified


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_fit_with_delay_fixed_zero_fails_gate(truth_run, prior, truth):
    """The twin: same data, delay held at 0 -> the held-out error is an order of magnitude above the correct fit's
    and FAILS the 1 % gate (MAX_RMS_PCT). The old 5 % gate was blind to a 3-tick delay through the clamp chain."""
    cal, cfg, _, train, held = truth_run
    res = fit_sim_params(train, held, prior=prior, fixed=("damping", "forcerange", "frictionloss", "delay_ticks"),
                         backend="cpu", popsize=8, generations=10, seed=0, calibration=cal, cfg=cfg)
    assert res.params.delay_ticks == 0
    worst = max(res.heldout_rms_pct.values())
    good = _open_loop_heldout(replace(res.params, delay_ticks=3), held, cal, cfg)
    print("\ndelay-0 held-out %:", {k: round(v, 3) for k, v in res.heldout_rms_pct.items()})
    assert worst > MAX_RMS_PCT == 1.0  # fails the gate
    assert worst > 5 * _open_loop_heldout(truth, held, cal, cfg).max()  # vs the noise floor of the truth
    assert good.max() < worst
    assert worst < 5.0  # pinned finding: the old 5 % gate did not see a 100 ms delay at 45 ticks/command


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_unidentifiable_pair_flagged(sines_run, prior):
    """kp and damping together on step-free sinusoid data: damping is declared unidentified (spread > 0.5); kp alone
    on the same data is identified (control)."""
    cal, cfg, train, held = sines_run
    pair = fit_sim_params(train, held, prior=prior, fit=("kp", "damping"), fixed=("delay_ticks",), backend="cpu",
                          popsize=8, generations=15, seed=0, calibration=cal, cfg=cfg)
    assert pair.spread["damping"] > 0.5
    assert any(k.startswith("damping.") for k in pair.unidentified)
    alone = fit_sim_params(train, held, prior=prior, fit=("kp",), fixed=("delay_ticks",), backend="cpu",
                           popsize=8, generations=15, seed=0, calibration=cal, cfg=cfg)
    assert alone.spread["kp"] < 0.5 and alone.unidentified == ["frictionloss"]  # only the fixed, unidentifiable one
    np.testing.assert_allclose(alone.params.kp[:5] / prior.kp[:5], 0.8, rtol=0.10)
    # the gripper's calibrated range shrank to 2010..3265 (2026-10-09 recalibration): less excitation, looser fit
    np.testing.assert_allclose(alone.params.kp[5] / prior.kp[5], 0.8, rtol=0.15)


def test_multiple_shooting_reinit(truth_run, truth):
    """Every window restarts from the logged state: the first prediction of each window equals a 1-tick-window
    prediction from the same row, and the open-loop replay carries its own state instead."""
    cal, cfg, ds, _, _ = truth_run
    seg = ds.segments[0]
    model, data = load_model()
    wrong = replace(truth, delay_ticks=0)
    sims = {w: replay_segment_cpu(seg, model, data, cal, cfg, wrong, shooting_window_ticks=w, quantize=False)
            for w in (1, 10, len(seg))}
    starts = np.arange(10, len(seg) - 1, 10)
    np.testing.assert_allclose(sims[10][starts + 1], sims[1][starts + 1], atol=1e-9)  # re-init = the w=1 state
    assert not np.allclose(sims[10][starts + 1], sims[len(seg)][starts + 1])  # open loop carries its own state
    assert sims[10][0].tolist() == seg.present_ticks[0].tolist()
    # (Until 2026-10-06 this also asserted that the windowed replay stays closer to the log than the open-loop one.
    # That held only for the old kp 0.7 / kv 1.4 truth: with the ratio-group truth the force-limited servo does not
    # drift, a wrong delay is a pure time shift, and both replays are within 2 % of each other (mean 10.69 vs 10.52
    # ticks on segment 0). The mechanics above are what the fit relies on.)


def test_replay_reproduces_noise_free_truth(tmp_path, truth):
    """With the truth parameters and no noise, the 30 Hz replay reproduces the logged ticks almost exactly."""
    from so101_rl.config import load_calibration, load_config

    cal = load_calibration(DATA / "so101_follower.calib.json")
    cfg = load_config(calibration=cal)
    plan = ExcitationPlan((Segment("multisine", ALL, 4.0, tuple(0.3 * a for a in AMPLITUDE_TICKS),
                                   {"centre_ticks": (0, 120, 0, 0, 0, 200)}),), 10.0, 0, hold_s=0.5)
    run_synthetic(plan, cfg, cal, tmp_path / "r", truth=truth)
    ds = ExcitationDataset.from_run(tmp_path / "r")
    model, data = load_model()
    sims = [replay_segment_cpu(s, model, data, cal, cfg, truth) for s in ds.segments]
    err = pooled_rms_pct(sims, ds.segments, cal)
    assert err.max() < 0.07, err  # < ~1.5 ticks rms (elbow ~1.4 at its 2201-tick range: re-init velocity from rounded ticks)
    sims0 = [replay_segment_cpu(s, model, data, cal, cfg, replace(truth, delay_ticks=0)) for s in ds.segments]
    assert pooled_rms_pct(sims0, ds.segments, cal).max() > 10 * err.max()


def test_objective_deadzone():
    span = np.full(6, 100.0)
    real = np.zeros((3, 6))
    sim = np.zeros((3, 6))
    sim[1:, 0] = 0.4  # rounds to the logged tick: free
    sim[1:, 1] = 1.5  # 1 tick beyond the dead zone
    e = objective_rms_pct(sim, real, span, 0.5)
    assert e[0] == 0.0 and e[1] == pytest.approx(1.0)
    assert objective(np.array([[1, 1, 1, 1, 1, 3.0]])) == pytest.approx(8 / 6 + 0.6)


def test_fit_argument_validation(truth_run, prior):
    cal, cfg, _, train, held = truth_run
    with pytest.raises(ValueError, match="deadband"):
        fit_sim_params(train, held, prior=prior, fit=("deadband_ticks",), backend="cpu", calibration=cal, cfg=cfg)
    with pytest.raises(ValueError, match="unknown"):
        fit_sim_params(train, held, prior=prior, fit=("nope",), backend="cpu", calibration=cal, cfg=cfg)
    with pytest.raises(ValueError, match="backend"):
        fit_sim_params(train, held, prior=prior, fit=("delay_ticks",), backend="tpu", calibration=cal, cfg=cfg)
    with pytest.raises(ValueError, match="held-out"):
        fit_sim_params(train, ExcitationDataset([]), prior=prior, backend="cpu", calibration=cal, cfg=cfg)
    with pytest.raises(ValueError, match="jointly"):  # the group and one of its members at once
        fit_sim_params(train, held, prior=prior, fit=("servo_group", "kp"), backend="cpu", calibration=cal, cfg=cfg)


def test_default_parametrisation_is_the_ratio_group():
    """Plan §10 2026-10-06: the default fit is the servo ratio group + delay; frictionloss is fixed (unidentified)."""
    from so101_rl.sim.sysid.fit import DEFAULT_FIT, DEFAULT_FIXED, SERVO_GROUP, build_parser

    assert DEFAULT_FIT == (SERVO_GROUP, "delay_ticks")
    assert "frictionloss" in DEFAULT_FIXED and "frictionloss" not in DEFAULT_FIT
    args = build_parser().parse_args(["--run", "r", "--out", "o"])
    assert args.fit == DEFAULT_FIT and args.fixed == DEFAULT_FIXED and args.max_pct is None


def test_delay_only_grid_search_is_exact(truth_run, prior):
    """No continuous parameter: the fit is a pure delay grid search over 0..6 and still finds 3."""
    cal, cfg, _, train, held = truth_run
    short_train = train.restrict(max_ticks=150)
    res = fit_sim_params(short_train, held.restrict(max_ticks=150), prior=prior, fit=("delay_ticks",), backend="cpu",
                         workers=4, calibration=cal, cfg=cfg)
    assert res.params.delay_ticks == 3
    assert min(res.delay_scores, key=res.delay_scores.get) == 3 and len(res.delay_scores) == 7


@pytest.mark.gpu
def test_warp_and_cpu_objectives_agree(truth_run, prior, truth):
    """The same candidate set gives the same objective on MuJoCo Warp (float32) and CPU MuJoCo (float64)."""
    from so101_rl.sim.sysid.fit import _WarpEvaluator

    cal, cfg, _, train, _ = truth_run
    segs = train.restrict(max_ticks=240).segments
    cands = [truth, prior, replace(truth, delay_ticks=1), truth.scaled(kp=1.3, armature=0.8)]
    model, data = load_model()
    tmap = TickMap(cal, model)
    span = tmap.span
    cpu = np.stack([
        np.stack([objective_rms_pct(replay_segment_cpu(s, model, data, cal, cfg, p, quantize=False, tmap=tmap), s.present_ticks,
                                    span, 0.5) for s in segs]) for p in cands
    ])
    ev = _WarpEvaluator({"train": segs, "heldout": segs}, cal, cfg, capacity=len(cands), window=60, device="cuda:0")
    try:
        warp = ev.evaluate(cands, "train")
    finally:
        ev.close()
    j_cpu, j_warp = objective(cpu.mean(axis=1)), objective(warp.mean(axis=1))
    print("\nobjective cpu", np.round(j_cpu, 5), "warp", np.round(j_warp, 5))
    assert np.abs(j_cpu - j_warp).max() < 1e-3


@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_warp_fit_finds_delay_and_passes_heldout(truth_run, prior):
    cal, cfg, _, train, held = truth_run
    res = fit_sim_params(train, held, prior=prior, backend="warp", popsize=32, generations=8, seed=0,
                         calibration=cal, cfg=cfg)
    assert res.backend == "warp" and res.params.delay_ticks == 3
    assert max(res.heldout_rms_pct.values()) < 1.0


def test_shrinking_grid_includes_the_prior_and_reports_flat_valleys():
    """The 1-D block search evaluates the prior (x = 0) first, and a flat objective is reported as a wide spread
    (not the shrinking step), so a scalar 1-D block can still be declared unidentified."""
    from so101_rl.sim.sysid.fit import UNIDENTIFIED_SPREAD, _ShrinkingGrid1D

    for pop in (2, 7, 8):
        assert any(x[0] == 0.0 for x in _ShrinkingGrid1D(pop, 1.6).ask())
    sharp, flat = _ShrinkingGrid1D(8, 1.6), _ShrinkingGrid1D(8, 1.6)
    for _ in range(25):
        xs = sharp.ask()
        sharp.tell(xs, [float((x[0] - 0.3) ** 2) for x in xs])
        xs = flat.ask()
        flat.tell(xs, [1.0 + 1e-6 * float(x[0] ** 2) for x in xs])
    assert abs(sharp.result.xfavorite[0] - 0.3) < 1e-3 and sharp.result.stds[0] < 0.1
    assert flat.result.stds[0] > UNIDENTIFIED_SPREAD


# ---- CLI gate: --max-pct may only tighten MAX_RMS_PCT ------------------------------------------------------------
def _fake_cli(monkeypatch, tmp_path, heldout_pct: float):
    """Patch the fit CLI's data / fit / writer so ``main`` runs in milliseconds; returns the list of written paths."""
    from types import SimpleNamespace

    from so101_rl.sim.sysid import fit as fitmod

    held = SimpleNamespace(segments=[object()])
    ds = SimpleNamespace(split=lambda **kw: (SimpleNamespace(segments=[object()]), held))
    monkeypatch.setattr(fitmod.ExcitationDataset, "from_run", staticmethod(lambda run: ds))
    res = SimpleNamespace(
        train_rms_pct=dict.fromkeys(JOINT_NAMES, heldout_pct), heldout_rms_pct=dict.fromkeys(JOINT_NAMES, heldout_pct),
        params=SimpleNamespace(delay_ticks=0), delay_scores={}, evals=1, wall_s=0.0, group_scale=None, unidentified=[],
    )
    monkeypatch.setattr(fitmod, "fit_sim_params", lambda *a, **k: res)
    monkeypatch.setattr(fitmod, "meta_for", lambda *a, **k: None)
    written: list[Path] = []
    monkeypatch.setattr(fitmod, "write_result", lambda r, path, *, meta: written.append(Path(path)))
    argv = ["--run", str(tmp_path), "--out", str(tmp_path / "sim_params.yaml"), "--backend", "cpu",
            "--calibration", str(DATA / "so101_follower.calib.json")]
    return fitmod, argv, written


@pytest.mark.parametrize("pct", [MAX_RMS_PCT * 1.0001, 5.0, 0.0, -1.0, float("nan")])
def test_cli_refuses_a_gate_looser_than_max_rms_pct(monkeypatch, tmp_path, capsys, pct):
    """A --max-pct above the 1 % gate (or <= 0 / NaN) is refused before anything is fitted or written: no PASS."""
    fitmod, argv, written = _fake_cli(monkeypatch, tmp_path, heldout_pct=0.5)
    monkeypatch.setattr(fitmod, "fit_sim_params", lambda *a, **k: pytest.fail("fitted despite a refused gate"))
    assert fitmod.main([*argv, "--max-pct", str(pct)]) == 2
    out = capsys.readouterr()
    assert "PASS" not in out.out + out.err and "refused" in out.err
    assert written == []


@pytest.mark.parametrize("pct", [None, MAX_RMS_PCT, 0.5])
def test_cli_gate_at_or_below_max_rms_pct_behaves_as_before(monkeypatch, tmp_path, capsys, pct):
    """At or below MAX_RMS_PCT: a held-out error within the gate PASSes and writes --out; above it writes only the
    .unverified file and exits 1."""
    gate = MAX_RMS_PCT if pct is None else pct
    extra = [] if pct is None else ["--max-pct", str(pct)]
    fitmod, argv, written = _fake_cli(monkeypatch, tmp_path, heldout_pct=0.9 * gate)
    assert fitmod.main([*argv, *extra]) == 0
    assert written == [tmp_path / "sim_params.yaml"] and "PASS" in capsys.readouterr().out
    fitmod, argv, written = _fake_cli(monkeypatch, tmp_path, heldout_pct=1.1 * gate)
    assert fitmod.main([*argv, *extra]) == 1
    out = capsys.readouterr()
    assert written == [tmp_path / "sim_params.unverified.yaml"] and "PASS" not in out.out + out.err
