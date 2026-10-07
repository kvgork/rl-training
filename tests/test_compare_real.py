"""Sysid check: replay a logged episode (F8 layout) through the sim. Synthetic logs are produced from the sim itself."""

from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES
from so101_rl.servo.bus import BlockRead
from so101_rl.sim.compare_real import MAX_RMS_PCT, format_table, main, rms_joint_error_pct
from so101_rl.sim.model import load_model
from so101_rl.sim.mujoco_bus import MujocoBus

N_STEPS = 40


def _trajectory(n: int) -> np.ndarray:
    t = np.arange(n)
    rest = np.full(6, 2047)
    amp = np.array([350, 450, 450, 350, 500, 0])
    tg = rest + amp * np.sin(2 * np.pi * t[:, None] / 20.0)
    tg[:, 5] = 2047 + 250 * (1 - np.cos(2 * np.pi * t / 20.0))
    tg[0] = rest
    return np.rint(tg).astype(np.int32)


def _write_f8(path: Path, target: np.ndarray, present: np.ndarray) -> Path:
    """Only the datasets compare_real reads, with F8 dtypes/shapes (U6's EpisodeLogger owns the full layout)."""
    n = len(target)
    with h5py.File(path, "w") as f:
        f.create_dataset("t_mono_ns", data=np.arange(n, dtype="i8") * 100_000_000)
        f.create_dataset("target_ticks", data=target.astype("i4"))
        f.create_dataset("present_ticks", data=present.astype("i4"))
    return path


def _simulate(calibration, cfg, target: np.ndarray, *, delay_ticks: int = 0) -> np.ndarray:
    model, data = load_model()
    bus = MujocoBus(model, data, calibration, cfg, action_delay_ticks=delay_ticks)
    bus.connect()
    bus.reset_to_keyframe("home")  # energised, off the table
    # start where compare_real starts a replay: at the logged pose (the trajectory's first target), at rest
    data.qvel[:] = 0.0
    data.qpos[bus.qpos_adr] = bus.ticks_to_q(target[0])
    data.ctrl[bus.act_ids] = data.qpos[bus.qpos_adr]
    mujoco.mj_forward(model, data)
    per_step = round(cfg.rates.servo_hz / cfg.rates.policy_hz)
    out, present = BlockRead(), []
    for goal in target:
        bus.write_goal(goal)
        bus.step_ticks(per_step)
        bus.read_block(out)
        present.append(out.position.copy())
    return np.array(present)


@pytest.fixture
def logs(tmp_path, calibration, cfg):
    target = _trajectory(N_STEPS)
    present = _simulate(calibration, cfg, target)
    good = _write_f8(tmp_path / "good.h5", target, present)
    return target, present, good


def test_sim_generated_log_gives_near_zero_error(logs, calibration):
    _, _, good = logs
    model, data = load_model()
    errors = rms_joint_error_pct(good, model, data, calibration)
    assert list(errors) == list(JOINT_NAMES)
    assert max(errors.values()) < 0.5, errors


def test_time_shifted_log_exceeds_gate(logs, tmp_path, calibration):
    target, present, _ = logs
    shifted = np.concatenate([np.repeat(present[:1], 4, axis=0), present[:-4]])  # sim reading lags by 4 steps
    bad = _write_f8(tmp_path / "bad.h5", target, shifted)
    model, data = load_model()
    errors = rms_joint_error_pct(bad, model, data, calibration)
    assert max(errors.values()) > MAX_RMS_PCT, errors


def test_extra_latency_is_detected(tmp_path, calibration, cfg):
    """A 'real' arm with 4 ticks (133 ms) more latency than the model exceeds the gate on the fast joints."""
    target = _trajectory(N_STEPS)
    real = _simulate(calibration, cfg, target, delay_ticks=6)
    path = _write_f8(tmp_path / "laggy.h5", target, real)
    model, data = load_model()
    errors = rms_joint_error_pct(path, model, data, calibration)
    assert max(errors.values()) > 1.0, errors


def test_cli_exit_codes_and_table(logs, tmp_path, calibration, capsys):
    target, present, good = logs
    calib_json = Path(__file__).parent / "data" / "so101_follower.calib.json"
    assert main([str(good), "--calibration", str(calib_json)]) == 0
    table = capsys.readouterr().out
    assert all(n in table for n in JOINT_NAMES) and "FAIL" not in table

    shifted = np.concatenate([np.repeat(present[:1], 4, axis=0), present[:-4]])
    bad = _write_f8(tmp_path / "bad.h5", target, shifted)
    assert main([str(bad), "--calibration", str(calib_json)]) == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out and "exceed" in captured.err


def test_malformed_logs_rejected(tmp_path, calibration):
    model, data = load_model()
    wrong_shape = _write_f8(tmp_path / "w.h5", np.full((5, 6), 2047), np.full((4, 6), 2047))
    with pytest.raises(ValueError, match="not"):
        rms_joint_error_pct(wrong_shape, model, data, calibration)
    too_short = _write_f8(tmp_path / "s.h5", np.full((1, 6), 2047), np.full((1, 6), 2047))
    with pytest.raises(ValueError, match="at least 2"):
        rms_joint_error_pct(too_short, model, data, calibration)
    out_of_range = _write_f8(tmp_path / "o.h5", np.full((5, 6), 5000), np.full((5, 6), 2047))
    with pytest.raises(ValueError, match="rejected"):
        rms_joint_error_pct(out_of_range, model, data, calibration)


def test_format_table_marks_failures():
    t = format_table(dict.fromkeys(JOINT_NAMES, 1.0) | {"elbow_flex": 9.0})
    assert t.count("FAIL") == 1 and "elbow_flex" in t


def test_replay_uses_model_not_mutating_keyframes(logs, calibration):
    _, _, good = logs
    model, data = load_model()
    before = model.key("rest").qpos.copy()
    rms_joint_error_pct(good, model, data, calibration)
    np.testing.assert_array_equal(model.key("rest").qpos, before)
    assert isinstance(data, mujoco.MjData)


# ---------------------------------------------------------------------------------------------------------------
# S2.1: --sim-params and --tick-log (FS7)


def _simulate_with(calibration, cfg, target, params):
    """Like ``_simulate``, but the 'real' arm has ``params`` (physics + command delay)."""
    model, data = load_model()
    params.apply_cpu(model)  # before the bus (it caches the gains)
    bus = MujocoBus(model, data, calibration, cfg, action_delay_ticks=params.delay_ticks)
    bus.connect()
    bus.reset_to_keyframe("home")
    data.qvel[:] = 0.0
    data.qpos[bus.qpos_adr] = bus.ticks_to_q(target[0])
    data.ctrl[bus.act_ids] = data.qpos[bus.qpos_adr]
    mujoco.mj_forward(model, data)
    per_step = round(cfg.rates.servo_hz / cfg.rates.policy_hz)
    out, present = BlockRead(), []
    for goal in target:
        bus.write_goal(goal)
        bus.step_ticks(per_step)
        bus.read_block(out)
        present.append(out.position.copy())
    return np.array(present)


def test_sim_params_change_the_error(tmp_path, calibration, cfg):
    """A log from an arm with kp x 0.5 and a 3-tick delay fails the gate under the prior and is ~0 % with matching
    --sim-params: the gate discriminates. kp alone moves the error measurably but far below the old 5 % gate (the MJCF
    servo is force-limited, its linear band is about ±2 ticks); on this trajectory it is just above the 1 % gate
    (plan §10 2026-10-06; measured 1.16 % on shoulder_pan, 2026-10-06 integration)."""
    from dataclasses import replace

    from so101_rl.sim.sysid.params import SimParams

    prior = SimParams.from_mjcf(load_model()[0])
    target = _trajectory(N_STEPS)
    truth = replace(prior.scaled(kp=0.5), delay_ticks=3)
    log = _write_f8(tmp_path / "kp_delay.h5", target, _simulate_with(calibration, cfg, target, truth))
    m, d = load_model()
    under_prior = rms_joint_error_pct(log, m, d, calibration, cfg=cfg)
    m, d = load_model()
    matched = rms_joint_error_pct(log, m, d, calibration, cfg=cfg, sim_params=truth)
    assert max(under_prior.values()) > MAX_RMS_PCT, under_prior
    assert max(matched.values()) < 0.5, matched

    kp_only = _write_f8(tmp_path / "kp.h5", target, _simulate_with(calibration, cfg, target, prior.scaled(kp=0.5)))
    m, d = load_model()
    e = rms_joint_error_pct(kp_only, m, d, calibration, cfg=cfg)
    assert MAX_RMS_PCT == 1.0
    assert MAX_RMS_PCT < max(e.values()) < 5.0, e  # the 1 % gate sees a halved kp here; the old 5 % gate did not


def test_sim_params_cli(tmp_path, calibration, cfg, capsys):
    from dataclasses import replace

    from so101_rl.sim.sysid.params import SimParams, prior_meta

    prior = SimParams.from_mjcf(load_model()[0])
    truth = replace(prior, delay_ticks=3)
    target = _trajectory(N_STEPS)
    log = _write_f8(tmp_path / "d3.h5", target, _simulate_with(calibration, cfg, target, truth))
    yml = tmp_path / "sp.yaml"
    truth.to_yaml(yml, meta=prior_meta(date="2026-10-06", git_sha="x"))
    calib_json = Path(__file__).parent / "data" / "so101_follower.calib.json"
    assert main([str(log), "--calibration", str(calib_json)]) == 1
    capsys.readouterr()
    assert main([str(log), "--calibration", str(calib_json), "--sim-params", str(yml)]) == 0
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out and "delay 3 ticks" in out  # an unverified file is accepted by the check itself


def test_help_lists_sim_params(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "--sim-params" in text and "--tick-log" in text


def test_tick_log_replay(tmp_path, calibration, cfg):
    """30 Hz replay of a ServoCore tick log: ~0 with the arm's own params, > 5x larger with the wrong delay."""
    from dataclasses import replace

    from so101_rl.sim.compare_real import rms_tick_log_error_pct
    from so101_rl.sim.sysid.excitation import named_plan
    from so101_rl.sim.sysid.params import SimParams, prior_meta
    from so101_rl.sim.sysid.run_excitation import run_synthetic

    prior = SimParams.from_mjcf(load_model()[0])
    truth = replace(prior.scaled(kv=1.3), delay_ticks=2)
    run_synthetic(named_plan("short", cfg), cfg, calibration, tmp_path / "run", truth=truth)
    tl = tmp_path / "run" / "tick_log.bin"
    m, d = load_model()
    good = rms_tick_log_error_pct(tl, m, d, calibration, cfg=cfg, sim_params=truth)
    m, d = load_model()
    bad = rms_tick_log_error_pct(tl, m, d, calibration, cfg=cfg)  # prior, no delay
    assert max(good.values()) < 0.1, good
    assert max(bad.values()) > 5 * max(max(good.values()), 0.02), bad

    yml = tmp_path / "sp.yaml"
    truth.to_yaml(yml, meta=prior_meta(date="2026-10-06", git_sha="x"))
    calib_json = Path(__file__).parent / "data" / "so101_follower.calib.json"
    assert main(["--tick-log", str(tl), "--sim-params", str(yml), "--calibration", str(calib_json)]) == 0
    with pytest.raises(SystemExit):
        main([str(tl), "--tick-log", str(tl)])  # exactly one source
    with pytest.raises(SystemExit):
        main([])
