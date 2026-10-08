"""deploy-policy (MVP M7, sim plan S5.1 reach part): refusals, Wilson / verdict math, and end-to-end runs against a
``servo --backend mujoco|fake`` child process on a private shm name (never a serial device)."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_rl.env.backend import ShmBackend
from so101_rl.rl import deploy_policy as dp
from so101_rl.rl.eval_sim import wilson_interval
from so101_rl.rl.fast_sac import Actor
from so101_rl.rl.policy_export import EXPORT_VERSION, LoadedPolicy
from so101_rl.servo.shm import CmdMode, lock_path

ROOT = Path(__file__).parent.parent
CALIB = ROOT / "tests" / "data" / "so101_follower.calib.json"
M6_POLICY = ROOT / "runs" / "lab" / "fastsac_sim-reach-20261008-100813-93bc" / "policy.pt"


def _export(path: Path, **overrides) -> Path:
    """A tiny random reach export in the policy_export format (24 -> 6)."""
    actor = Actor(24, 6, (16, 16), (-5.0, 2.0))
    doc = {
        "format": "so101_rl.policy_export", "export_version": EXPORT_VERSION, "obs_layout_version": 1,
        "task_name": "reach", "obs_dim": 24, "act_dim": 6, "actor_hidden": [16, 16], "log_std_bounds": [-5.0, 2.0],
        "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
    }  # fmt: skip
    doc.update(overrides)
    torch.save(doc, path)
    return path


def _args(policy: Path, shm: str, out: Path, *extra: str) -> list[str]:
    return ["--policy", str(policy), "--shm-name", shm, "--out-root", str(out), "--wait-servo-s", "0", *extra]


# ------------------------------------------------------------------------------------------------ statistics


def test_wilson_known_values():
    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4) and hi == 1.0
    lo, hi = wilson_interval(45, 50)
    assert lo == pytest.approx(0.7864, abs=1e-3) and hi == pytest.approx(0.9565, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0713, abs=1e-3)


def _eps(speed: float, n: int, k: int) -> list[dict]:
    return [{"speed": speed, "success": i < k, "final_error_m": 0.01, "steps": 10, "trips": 0} for i in range(n)]


def test_summary_per_rung_and_verdict():
    eps = _eps(0.5, 10, 10) + _eps(0.75, 10, 9) + _eps(1.0, 50, 40)
    s = dp.summarize(eps, dp.SPEED_LADDER)
    assert [r["n"] for r in s["rungs"]] == [10, 10, 50]
    assert s["rungs"][1]["success_rate"] == 0.9 and s["overall"]["n"] == 70 and s["overall"]["successes"] == 59
    lo = wilson_interval(40, 50)[0]
    assert s["verdict"]["wilson_lo"] == pytest.approx(lo) and lo >= 0.6 and s["verdict"]["finetune_eligible"]
    # 30/50: lower bound 0.46 < 0.60
    assert not dp.summarize(_eps(1.0, 50, 30), dp.SPEED_LADDER)["verdict"]["finetune_eligible"]
    # a perfect but too short gate rung is not a verdict (n >= 50 on the full-speed rung)
    assert not dp.summarize(_eps(1.0, 10, 10), [1.0])["verdict"]["finetune_eligible"]
    # a ladder stopped before the full-speed rung has no verdict
    v = dp.summarize(_eps(0.5, 10, 10), dp.SPEED_LADDER)["verdict"]
    assert v["wilson_lo"] is None and v["rung_n"] == 0 and not v["finetune_eligible"]


# ------------------------------------------------------------------------------------------------ refusals


@pytest.fixture
def no_hardware(monkeypatch):
    """Any serial port or shm attach during a refusal test is a failure."""
    import serial

    def boom(*_a, **_k):
        raise AssertionError("a serial device was opened")

    monkeypatch.setattr(serial, "Serial", boom)
    monkeypatch.setattr(ShmBackend, "__init__", lambda *_a, **_k: boom())


@pytest.mark.parametrize(
    ("override", "match"),
    [({"obs_layout_version": 99}, "obs_layout_version"), ({"task_name": "die_to_zone"}, "task"),
     ({"export_version": 99}, "export_version"), ({"format": "x"}, "not a policy export")],
)  # fmt: skip
def test_refuses_bad_policy(tmp_path, capsys, no_hardware, override, match):
    p = _export(tmp_path / "p.pt", **override)
    assert dp.main(_args(p, "unused", tmp_path / "out")) == dp.EXIT_REFUSED
    assert match in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_check_policy_refuses_wrong_dims(tmp_path):
    from so101_rl.env.tasks import REACH

    actor = Actor(27, 6, (8,), (-5.0, 2.0))
    meta = {"obs_layout_version": 1, "task_name": "reach", "obs_dim": 27, "act_dim": 6}
    with pytest.raises(dp.Refused, match="dims"):
        dp.check_policy(LoadedPolicy(actor, meta), REACH)


def test_refuses_without_a_servo(tmp_path, capsys, no_hardware):
    p = _export(tmp_path / "p.pt")
    assert dp.main(_args(p, f"so101none_{uuid.uuid4().hex[:8]}", tmp_path / "out")) == dp.EXIT_REFUSED
    assert "start the servo first" in capsys.readouterr().err


@pytest.fixture
def pretend_servo():
    """A process whose command line LOOKS like a servo (``--backend <kind>``), registered as the owner of a private shm
    name's lock file. Nothing is opened: it only sleeps."""
    procs = []

    def make(backend: str | None) -> str:
        name = f"so101deploytest_{uuid.uuid4().hex[:10]}"
        argv = [sys.executable, "-c", "import time; time.sleep(60)", "-m", "so101_rl.servo.main"]
        if backend is not None:
            argv += ["--backend", backend]
        p = subprocess.Popen(argv)
        procs.append((p, name))
        Path(lock_path(name)).write_text(f"{p.pid}\n")
        return name

    yield make
    for p, name in procs:
        p.kill()
        p.wait()
        Path(lock_path(name)).unlink(missing_ok=True)


@pytest.mark.parametrize("backend", ["feetech", None, "something"])
def test_refuses_real_arm_without_attended(tmp_path, capsys, no_hardware, pretend_servo, backend):
    name = pretend_servo(backend)
    info = dp.servo_info(name)
    assert info.is_real  # fail-closed: only mujoco / fake are sim
    assert dp.main(_args(_export(tmp_path / "p.pt"), name, tmp_path / "out")) == dp.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "REAL arm" in err and "--attended" in err
    assert not (tmp_path / "out").exists()


def test_last_backend_flag_wins_like_argparse(pretend_servo):
    name = pretend_servo("mujoco")
    argv = ["python", "-m", "so101_rl.servo.main", "--backend", "mujoco", "--backend=feetech"]
    assert dp._flag(argv, "--backend") == "feetech"
    assert dp._flag(argv[:-1], "--backend") == "mujoco" and dp._flag(argv[:3], "--backend") is None
    assert not dp.servo_info(name).is_real


def test_malformed_rung_episodes_is_a_refusal(tmp_path, capsys, no_hardware):
    p = _export(tmp_path / "p.pt")
    assert dp.main(_args(p, "unused", tmp_path / "out", "--protocol", "zero-shot", "--rung-episodes", "a,b,c")) == 2
    assert "rung-episodes" in capsys.readouterr().err


def test_sim_servo_is_detected_with_its_calibration_and_tick_log(pretend_servo):
    name = pretend_servo("mujoco")
    info = dp.servo_info(name)
    assert info.backend == "mujoco" and not info.is_real and info.calibration is None and info.tick_log is None


def test_rung_episodes_parsing():
    args = dp.build_parser().parse_args(["--policy", "x", "--protocol", "zero-shot"])
    assert dp._plan(args) == [(0.5, 10), (0.75, 10), (1.0, 50)]
    args = dp.build_parser().parse_args(["--policy", "x", "--protocol", "zero-shot", "--rung-episodes", "0,2,3"])
    assert dp._plan(args) == [(0.75, 2), (1.0, 3)]
    with pytest.raises(dp.Refused):
        dp._plan(dp.build_parser().parse_args(["--policy", "x", "--protocol", "zero-shot", "--rung-episodes", "1,2"]))
    with pytest.raises(SystemExit):  # the ladder speeds only
        dp.build_parser().parse_args(["--policy", "x", "--speed-scale", "0.9"])


# ------------------------------------------------------------------------------------------------ end to end


class ServoChild:
    """``servo --backend mujoco|fake`` child on a private shm name (FakeMcu in-process, never a serial port)."""

    def __init__(self, tmp_path: Path, backend: str, *extra: str) -> None:
        self.name = f"so101deploy_{uuid.uuid4().hex[:10]}"
        self.tick_log = tmp_path / "ticks.bin"
        env = {k: v for k, v in os.environ.items() if k != "SO101_HW"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "so101_rl.servo.main", "--backend", backend, "--calibration", str(CALIB),
             "--shm-name", self.name, "--tick-log", str(self.tick_log), "--summary", str(tmp_path / "servo.json"),
             *extra],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )  # fmt: skip

    def stop(self) -> int:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                return self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return self.proc.wait()


@pytest.fixture
def command_log(monkeypatch):
    """Every shm command the deployment sends, with the reset boundaries: ``R0`` reset start, ``R1`` reset done."""
    log: list[str] = []
    send, reset = ShmBackend._send, ShmBackend.reset

    def _send(self, mode, delta, state_seq, episode):
        log.append("RUN" if mode == CmdMode.RUN else "HOLD" if mode == CmdMode.HOLD else str(int(mode)))
        return send(self, mode, delta, state_seq, episode)

    def _reset(self, q0):
        log.append("R0")
        reset(self, q0)
        log.append("R1")

    monkeypatch.setattr(ShmBackend, "_send", _send)
    monkeypatch.setattr(ShmBackend, "reset", _reset)
    return log


def _episode_segments(log: list[str]) -> list[list[str]]:
    """The commands after each completed reset up to the next reset (or the end)."""
    segs, cur = [], None
    for c in log:
        if c == "R0":
            if cur is not None:
                segs.append(cur)
            cur = None
        elif c == "R1":
            cur = []
        elif cur is not None:
            cur.append(c)
    if cur is not None:
        segs.append(cur)
    return segs


@pytest.mark.slow
@pytest.mark.timeout(400)
def test_mujoco_end_to_end_with_the_m6_policy(tmp_path, command_log, capsys):
    if not M6_POLICY.is_file():
        pytest.skip(f"the M6 policy export is not on this host ({M6_POLICY})")
    servo = ServoChild(tmp_path, "mujoco")
    try:
        code = dp.main(["--policy", str(M6_POLICY), "--shm-name", servo.name, "--out-root", str(tmp_path / "deploy"),
                        "--episodes", "10", "--speed-scale", "1.0", "--wait-servo-s", "120"])  # fmt: skip
    finally:
        assert servo.stop() == 0
    out = capsys.readouterr().out
    print(out)
    assert code == dp.EXIT_OK
    (run,) = (tmp_path / "deploy").iterdir()
    eps = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    assert len(eps) == 10
    for e in eps:
        assert set(e) >= {"goal", "success", "steps", "final_error_m", "trips", "speed", "episode"}
        assert len(e["goal"]) == 3 and e["speed"] == 1.0 and e["trips"] == 0 and 1 <= e["steps"] <= 100
        if e["success"]:
            assert e["final_error_m"] <= 0.015 + 1e-6
    successes = sum(e["success"] for e in eps)
    assert successes >= 8, out  # the batched sim eval: 99.2 %
    assert summary["overall"]["n"] == 10 and summary["overall"]["successes"] == successes
    assert summary["rungs"][0]["wilson95"] == list(wilson_interval(successes, 10))
    assert summary["tick_log"] == str(servo.tick_log) and summary["servo"]["backend"] == "mujoco"
    assert summary["servo"]["real"] is False and summary["halted"] is None and summary["exit_code"] == 0
    assert summary["inference_ms"]["p50"] < 20.0
    # each episode: exactly `steps` RUN commands, then HOLD only until the next reset (never driven past its end)
    segs = _episode_segments(command_log)
    assert len(segs) == 10
    for e, seg in zip(eps, segs, strict=True):
        assert seg[: e["steps"]] == ["RUN"] * e["steps"], (e, seg)
        assert seg[e["steps"] :] and set(seg[e["steps"] :]) == {"HOLD"}, (e, seg)
    assert command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_ctrl_c_sends_hold_and_keeps_what_was_collected(tmp_path, command_log, monkeypatch):
    """A KeyboardInterrupt in the middle of episode 2 (raised from the policy call, as Ctrl-C would be) -> HOLD right
    after the last RUN, the two finished episodes written, exit 130. A still policy and 5-step episodes keep it short."""
    calls = {"n": 0}

    def interrupting(self, obs):
        if command_log.count("R1") == 3:  # episode 2 is running
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
        return np.zeros(6, np.float32)

    def short_episode(env, policy, speed, *, seed, infer_ms):
        env.task = dataclasses.replace(env.task, episode_steps=5)
        return run_episode(env, policy, speed, seed=seed, infer_ms=infer_ms)

    run_episode = dp._run_episode
    monkeypatch.setattr(LoadedPolicy, "__call__", interrupting)
    monkeypatch.setattr(dp, "_run_episode", short_episode)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = dp.main(["--policy", str(_export(tmp_path / "p.pt")), "--shm-name", servo.name,
                        "--out-root", str(tmp_path / "deploy"), "--episodes", "5", "--wait-servo-s", "60"])  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_INTERRUPTED
    (run,) = (tmp_path / "deploy").iterdir()
    eps = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert [e["steps"] for e in eps] == [5, 5] and not any(e["success"] for e in eps)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["exit_code"] == 130 and "interrupted" in summary["halted"] and summary["overall"]["n"] == 2
    assert command_log[-3:] == ["RUN", "RUN", "HOLD"]  # two steps of episode 2, then HOLD and nothing else


def test_servo_info_reads_calibration_and_tick_log_from_the_servo_cmdline(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "so101_rl.servo.main",
                          "--backend", "mujoco", "--calibration", "cal.json", "--tick-log=logs/t.bin"], cwd=tmp_path)  # fmt: skip
    name = f"so101deploytest_{uuid.uuid4().hex[:10]}"
    try:
        Path(lock_path(name)).write_text(f"{p.pid}\n")
        info = dp.servo_info(name)
        assert info.calibration == (tmp_path / "cal.json").resolve()
        assert info.tick_log == (tmp_path / "logs" / "t.bin").resolve()
        assert info.pid == p.pid and info.backend == "mujoco"
    finally:
        p.kill()
        p.wait()
        Path(lock_path(name)).unlink(missing_ok=True)
