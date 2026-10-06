"""FS7 excitation plans, the (hardware-gated) runner and the excitation dataset (sim-first track S2.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, load_calibration, load_config
from so101_rl.safety.fk import KinematicChain
from so101_rl.sim.sysid.dataset import MIN_SEGMENT_TICKS, ExcitationDataset
from so101_rl.sim.sysid.excitation import (
    MULTISINE_BAND_HZ,
    ExcitationPlan,
    Segment,
    check_plan_in_box,
    default_plan,
    generate_deltas,
    named_plan,
    plan_offsets,
    scaled_plan,
    short_plan,
)
from so101_rl.sim.sysid.run_excitation import main as excite_main
from so101_rl.sim.sysid.run_excitation import real_mode_refusals, run_synthetic

DATA = Path(__file__).parent / "data"
CALIB = DATA / "so101_follower.calib.json"


@pytest.fixture(scope="module")
def setup():
    cal = load_calibration(CALIB)
    cfg = load_config(calibration=cal)
    return cal, cfg, KinematicChain.from_yaml(cfg.kinematics_path), np.asarray(cfg.safe_state.home_ticks)


@pytest.mark.parametrize("name", ["default", "short", "fit40"])
@pytest.mark.parametrize("die", [False, True])
def test_plan_in_box_kinematic_preview(setup, name, die):
    cal, cfg, chain, home = setup
    plan = named_plan(name, cfg, with_gripper_die=die)
    chk = check_plan_in_box(plan, cfg, cal, chain, home, margin_m=0.01)
    assert chk.ok, chk.reasons
    assert chk.min_box_margin_m >= 0.01 and chk.min_table_clearance_m > 0.03
    assert chk.n_ticks == len(generate_deltas(plan, cfg)) * cfg.action.interp_ticks


def test_default_plan_is_about_ten_minutes(setup):
    _, cfg, _, _ = setup
    plan = default_plan(cfg, with_gripper_die=False)
    minutes = len(generate_deltas(plan, cfg)) / cfg.rates.policy_hz / 60
    assert 8.0 <= minutes <= 12.0
    kinds = {s.kind for s in plan.segments}
    assert kinds == {"multisine", "chirp", "step", "ramp", "random"}
    assert "gripper" in {s.kind for s in default_plan(cfg, with_gripper_die=True).segments}
    assert {j for s in plan.segments for j in s.joints} == set(range(6))  # every joint is excited
    assert plan.segments[-1].joints == tuple(range(6))  # the combined all-joint segment comes last


def test_plan_out_of_box_rejected(setup):
    cal, cfg, chain, home = setup
    chk = check_plan_in_box(scaled_plan(default_plan(cfg, with_gripper_die=False), 10.0), cfg, cal, chain, home)
    assert not chk.ok and chk.first_violation is not None
    assert "leaves the box" in chk.reasons[0]
    # a plan that only pushes the tip down through the box floor
    down = ExcitationPlan((Segment("ramp", (1,), 1.0, 900.0, {"rate_ticks": 30.0, "centre_ticks": (0,)}),), 10.0, 0)
    assert not check_plan_in_box(down, cfg, cal, chain, home).ok


def test_multisine_band_below_nyquist(setup):
    _, cfg, _, _ = setup
    hz = cfg.rates.policy_hz
    assert MULTISINE_BAND_HZ[1] < hz / 2  # 4 Hz < 5 Hz Nyquist of the 10 Hz command
    plan = ExcitationPlan((Segment("multisine", (0,), 60.0, 100.0, {"centre_ticks": (0,)}),), hz, 0, hold_s=0.1)
    p, spans = plan_offsets(plan, cfg)
    x = p[spans[0].start + 1 : spans[0].stop + 1, 0]
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / hz)
    inband = spec[(f >= 0.15) & (f <= 4.2)].sum()
    assert inband / spec[f > 0.05].sum() > 0.97
    # combined segments give every joint its own (still in-band) frequencies
    combo = ExcitationPlan((Segment("multisine", (0, 3), 60.0, (100.0, 100.0), {"centre_ticks": (0, 0)}),), hz, 0)
    q, _ = plan_offsets(combo, cfg)
    assert np.corrcoef(q[:, 0], q[:, 3])[0, 1] < 0.5


def test_deltas_are_integral_and_respect_the_command_clamp(setup):
    _, cfg, _, _ = setup
    max_d = np.asarray(cfg.action.max_delta_ticks)
    for name in ("default", "short", "fit40"):
        plan = named_plan(name, cfg, with_gripper_die=True)
        d = generate_deltas(plan, cfg)
        assert d.dtype == np.float32 and d.shape[1] == 6
        np.testing.assert_array_equal(d, np.rint(d))
        assert (np.abs(d) <= max_d[None, :]).all(), name
        p, spans = plan_offsets(plan, cfg)
        for s in spans:  # every segment returns to the start pose
            assert np.abs(np.rint(p[s.stop])).max() == 0, (name, s.label)


def test_plan_is_deterministic_per_seed(setup):
    _, cfg, _, _ = setup
    a = generate_deltas(short_plan(cfg, seed=1), cfg)
    np.testing.assert_array_equal(a, generate_deltas(short_plan(cfg, seed=1), cfg))
    assert not np.array_equal(a, generate_deltas(short_plan(cfg, seed=2), cfg))
    with pytest.raises(ValueError, match="unknown plan"):
        named_plan("nope", cfg)
    with pytest.raises(ValueError):
        Segment("multisine", (6,), 1.0, 1.0)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory, setup):
    from so101_rl.sim.model import load_model
    from so101_rl.sim.sysid.params import SimParams

    cal, cfg, _, _ = setup
    out = tmp_path_factory.mktemp("excite") / "syn"
    truth = SimParams.from_mjcf(load_model()[0])
    doc, errors = run_synthetic(named_plan("fit40", cfg), cfg, cal, out, truth=truth)
    return out, doc, errors


def test_synthetic_run_artifacts(synthetic_run, setup):
    from so101_rl.logging.checks import check_episode

    _, cfg, _, _ = setup
    out, _, errors = synthetic_run
    assert errors == []
    plan = json.loads((out / "plan.json").read_text())
    assert plan["backend"] == "synthetic" and plan["joint_names"] == list(JOINT_NAMES)
    for row in plan["segments"]:
        assert row["completed"] and row["run_ticks"] > 0 and 0.0 <= row["clamp_active_frac"] <= 1.0
        assert row["t_stop_ns"] > row["t_start_ns"]
        assert set(row["clamp_active_by_stage"]) == {"tracking", "joint_limit", "box", "velocity"}
    eps = sorted((out / "episodes").glob("ep_*.h5"))
    assert len(eps) == len(plan["segments"])
    assert all(check_episode(e, cfg).ok for e in eps)


def test_split_by_time_and_kind(synthetic_run):
    out, _, _ = synthetic_run
    ds = ExcitationDataset.from_run(out)
    assert ds.kinds == ["chirp", "multisine", "ramp", "random", "step"]
    train, held = ds.split(heldout_kinds={"chirp", "random"}, heldout_fraction_by_time=0.2, seed=3)
    assert train.n_ticks + held.n_ticks <= ds.n_ticks
    assert ds.n_ticks - (train.n_ticks + held.n_ticks) < 3 * MIN_SEGMENT_TICKS * len(ds.segments)  # only scraps lost
    assert {s.kind for s in train.segments} == {"multisine", "ramp", "step"}
    held_kinds = [s.kind for s in held.segments]
    assert held_kinds.count("chirp") == 1 and held_kinds.count("random") == 1
    for kind in ("multisine", "ramp", "step"):  # a ~20 % time window of every other segment is held out
        full = sum(len(s) for s in ds.segments if s.kind == kind)
        part = sum(len(s) for s in held.segments if s.kind == kind)
        assert 0.15 * full <= part <= 0.25 * full, kind
    a, b = ds.split(heldout_kinds={"chirp"}, seed=3), ds.split(heldout_kinds={"chirp"}, seed=3)
    assert [len(s) for s in a[1].segments] == [len(s) for s in b[1].segments]  # seeded
    # history targets carry the FIFO context of every slice
    assert all(s.history_targets is not None and len(s.history_targets) >= 1 for s in train.segments)


def test_tick_segment_semantics(synthetic_run):
    out, _, _ = synthetic_run
    seg = ExcitationDataset.from_run(out).segments[1]
    sl = seg.slice(20, 80)
    np.testing.assert_array_equal(sl.target_ticks, seg.target_ticks[20:80])
    np.testing.assert_array_equal(sl.history_targets, seg.target_ticks[14:20])
    with pytest.raises(IndexError):
        seg.slice(5, 5)


def test_dataset_rejects_broken_runs(tmp_path, synthetic_run):
    out, _, _ = synthetic_run
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "plan.json").write_text((out / "plan.json").read_text())
    with pytest.raises(FileNotFoundError, match="tick log"):
        ExcitationDataset.from_run(bad)
    (bad / "tick_log.bin").write_bytes(b"\0" * 7)
    with pytest.raises(ValueError, match="multiple"):
        ExcitationDataset.from_run(bad)


def test_real_refused_without_attended_and_artifacts(tmp_path, monkeypatch):
    """--backend real: exit 2 without --attended or without the H2/H3/H5/H7 artifacts; no serial port and no shm
    segment is ever touched on the way."""
    import serial

    from so101_rl.servo import shm as shm_mod

    touched: list[str] = []

    def boom(*a, **k):
        touched.append("serial")
        raise AssertionError("serial.Serial constructed")

    def no_attach(*a, **k):
        touched.append("shm")
        raise AssertionError("shm attached")

    monkeypatch.setattr(serial, "Serial", boom)
    monkeypatch.setattr(shm_mod.ServoShm, "attach", classmethod(lambda cls, *a, **k: no_attach()))
    hw = tmp_path / "hw"
    hw.mkdir()
    out = tmp_path / "run"
    argv = ["--backend", "real", "--plan", "short", "--out", str(out), "--hw-dir", str(hw), "--calibration", str(CALIB)]
    assert excite_main(argv) == 2  # neither
    assert excite_main([*argv, "--attended"]) == 2  # no artifacts
    for a in ("estop", "deadman", "rest-pose", "calib-check"):
        (hw / f"{a}-20261006.json").write_text("{}")
    assert excite_main(argv) == 2  # artifacts but not attended
    assert real_mode_refusals(attended=True, hw_dir=hw) == []
    (hw / "deadman-20261006.json").unlink()
    assert any("deadman" in r for r in real_mode_refusals(attended=True, hw_dir=hw))
    assert touched == [] and not out.exists()


def test_cli_refuses_unknown_plan_and_nonempty_out(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "x").write_text("x")
    assert excite_main(["--backend", "synthetic", "--plan", "nope", "--out", str(tmp_path / "n"),
                        "--calibration", str(CALIB)]) == 2
    assert excite_main(["--backend", "synthetic", "--plan", "short", "--out", str(out), "--calibration", str(CALIB)]) == 2


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_dry_run_sim_writes_dataset(tmp_path):
    """The real dry run: a ``servo --backend mujoco`` child in real time, driven over its shm (~20 s plan)."""
    out = tmp_path / "dry"
    assert excite_main(["--backend", "sim", "--plan", "short", "--out", str(out), "--calibration", str(CALIB)]) == 0
    for name in ("tick_log.bin", "plan.json", "servo_summary.json"):
        assert (out / name).is_file()
    assert len(list((out / "episodes").glob("ep_*.h5"))) == 5
    plan = json.loads((out / "plan.json").read_text())
    assert plan["backend"] == "sim" and plan["servo_exit_code"] == 0 and plan["artifact_errors"] == []
    for row in plan["segments"]:
        assert 0.0 <= row["clamp_active_frac"] <= 0.6
    ds = ExcitationDataset.from_run(out)
    assert len(ds.segments) >= 3
