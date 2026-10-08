"""deploy-policy (MVP M7, sim plan S5.1 reach part): refusals, Wilson / verdict math, and end-to-end runs against a
``servo --backend mujoco|fake`` child process on a private shm name (never a serial device)."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
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


ZS = {"protocol": "zero-shot", "ladder_complete": True}


def test_summary_per_rung_and_verdict():
    eps = _eps(0.5, 10, 10) + _eps(0.75, 10, 9) + _eps(1.0, 50, 40)
    s = dp.summarize(eps, dp.SPEED_LADDER, protocol="zero-shot", ladder_complete=True)
    assert [r["n"] for r in s["rungs"]] == [10, 10, 50]
    assert s["rungs"][1]["success_rate"] == 0.9 and s["overall"]["n"] == 70 and s["overall"]["successes"] == 59
    lo = wilson_interval(40, 50)[0]
    assert s["verdict"]["wilson_lo"] == pytest.approx(lo) and lo >= 0.6 and s["verdict"]["finetune_eligible"]
    # 30/50: lower bound 0.46 < 0.60
    assert not dp.summarize(_eps(1.0, 50, 30), dp.SPEED_LADDER, **ZS)["verdict"]["finetune_eligible"]
    # a perfect but too short gate rung is not a verdict (n >= 50 on the full-speed rung)
    assert not dp.summarize(_eps(1.0, 10, 10), [1.0], **ZS)["verdict"]["finetune_eligible"]
    # a ladder stopped before the full-speed rung has no verdict
    v = dp.summarize(_eps(0.5, 10, 10), dp.SPEED_LADDER, **ZS)["verdict"]
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

    def make(backend: str | None, *extra: str) -> str:
        name = f"so101deploytest_{uuid.uuid4().hex[:10]}"
        argv = [sys.executable, "-c", "import time; time.sleep(60)", "-m", "so101_rl.servo.main"]
        if backend is not None:
            argv += ["--backend", backend]
        argv += list(extra)
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
    assert dp.parse_servo_cmdline(1, argv, Path("/")).backend == "feetech"
    assert dp.parse_servo_cmdline(1, argv[:-1], Path("/")).backend == "mujoco"
    assert dp.parse_servo_cmdline(1, argv[:3], Path("/")).is_real  # no --backend: does not parse -> real
    assert not dp.servo_info(name).is_real


@pytest.mark.parametrize("tail", [["--backend", "mujoco", "--backe", "feetech"], ["--back=feetech"],
                                  ["--backend", "fake", "--calib", "x.json"], ["--backend", "fake", "--bogus"]])  # fmt: skip
def test_abbreviated_or_unknown_servo_options_count_as_real(tail):
    info = dp.parse_servo_cmdline(1, ["python", "-m", "so101_rl.servo.main", *tail], Path("/"))
    assert info.is_real and info.problem is not None


def test_servo_parser_rejects_abbreviations():
    from so101_rl.servo.main import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--back", "fake"])
    assert build_parser().parse_args(["--backend", "fake"]).backend == "fake"


def test_abbreviated_backend_is_refused_even_attended(tmp_path, capsys, no_hardware, pretend_servo):
    name = pretend_servo("mujoco", "--backe", "feetech")
    code = dp.main(_args(_export(tmp_path / "p.pt"), name, tmp_path / "out", "--attended"))
    assert code == dp.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "REAL arm" in err and "does not parse" in err
    assert not (tmp_path / "out").exists()


def test_real_servo_with_no_mcu_is_refused(tmp_path, capsys, no_hardware, pretend_servo):
    name = pretend_servo("feetech", "--no-mcu", "--calibration", str(CALIB))
    assert dp.servo_info(name).no_mcu
    assert dp.main(_args(_export(tmp_path / "p.pt"), name, tmp_path / "out", "--attended")) == dp.EXIT_REFUSED
    assert "--no-mcu" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_attended_real_servo_gets_past_the_refusals(tmp_path, capsys, monkeypatch, pretend_servo):
    """A real-looking servo with --attended: no refusal; the run fails only at the shm attach (stubbed), which is
    the first thing after the read-only checks (exit 2, nothing written, nothing sent)."""

    def attach(*_a, **_k):
        raise RuntimeError("attach reached (test stub)")

    monkeypatch.setattr(ShmBackend, "__init__", attach)
    name = pretend_servo("feetech", "--calibration", str(CALIB), "--heartbeat-port", "/dev/null-mcu")
    code = dp.main(_args(_export(tmp_path / "p.pt"), name, tmp_path / "out", "--attended"))
    assert code == dp.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "attach reached" in err and "REAL" not in err.split("refused:")[-1]
    assert not (tmp_path / "out").exists()


def test_config_must_match_the_servo(tmp_path, capsys, no_hardware, pretend_servo):
    from so101_rl.config.loader import DEFAULT_CONFIG_PATH

    other = tmp_path / "other.yaml"
    other.write_text(DEFAULT_CONFIG_PATH.read_text())
    name = pretend_servo("mujoco", "--calibration", str(CALIB), "--config", str(DEFAULT_CONFIG_PATH))
    assert dp.servo_info(name).config == DEFAULT_CONFIG_PATH.resolve()
    assert dp.main(_args(_export(tmp_path / "p.pt"), name, tmp_path / "out", "--config", str(other))) == 2
    assert "--config" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_calibration_must_match_the_servo_config_when_the_servo_has_none(tmp_path, capsys, no_hardware,
                                                                         pretend_servo):  # fmt: skip
    name = pretend_servo("mujoco")  # its calibration: so101.yaml's calibration_path, not the test copy
    assert dp.main(_args(_export(tmp_path / "p.pt"), name, tmp_path / "out", "--calibration", str(CALIB))) == 2
    assert "--calibration" in capsys.readouterr().err


def test_malformed_rung_episodes_is_a_refusal(tmp_path, capsys, no_hardware):
    p = _export(tmp_path / "p.pt")
    assert dp.main(_args(p, "unused", tmp_path / "out", "--protocol", "zero-shot", "--rung-episodes", "a,b,c")) == 2
    assert "rung-episodes" in capsys.readouterr().err


def test_sim_servo_is_detected_with_its_calibration_and_tick_log(pretend_servo):
    name = pretend_servo("mujoco")
    info = dp.servo_info(name)
    assert info.backend == "mujoco" and not info.is_real and info.calibration is None and info.tick_log is None


def test_rung_episodes_parsing():
    def plan(*a):
        return dp._resolve_run_args(dp.build_parser().parse_args(["--policy", "x", *a]), None)

    assert plan("--protocol", "zero-shot") == [(0.5, 10), (0.75, 10), (1.0, 50)]
    assert plan("--protocol", "zero-shot", "--rung-episodes", "0,2,3") == [(0.75, 2), (1.0, 3)]
    assert plan() == [(0.5, 10)]
    with pytest.raises(dp.Refused):
        plan("--protocol", "zero-shot", "--rung-episodes", "1,2")
    with pytest.raises(dp.Refused):
        plan("--protocol", "zero-shot", "--rung-episodes", "0,0,0")
    with pytest.raises(dp.Refused):
        plan("--episodes", "0")
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


class _CmdLog(list):
    def __init__(self) -> None:
        super().__init__()
        self.run_deltas: list[np.ndarray] = []
        self.in_reset = False


def _short_episodes(monkeypatch, steps: int = 5) -> None:
    """Episodes of ``steps`` policy steps (a still or constant policy never succeeds: keeps the fake runs short)."""
    run_episode = dp._run_episode

    def short_episode(env, policy, speed, **kw):
        env.task = dataclasses.replace(env.task, episode_steps=steps)
        return run_episode(env, policy, speed, **kw)

    monkeypatch.setattr(dp, "_run_episode", short_episode)


@pytest.fixture
def command_log(monkeypatch):
    """Every shm command the deployment sends, with the reset boundaries: ``R0`` reset start, ``R1`` reset done."""
    log = _CmdLog()
    send, reset = ShmBackend._send, ShmBackend.reset

    def _send(self, mode, delta, state_seq, episode):
        log.append("RUN" if mode == CmdMode.RUN else "HOLD" if mode == CmdMode.HOLD else str(int(mode)))
        if mode == CmdMode.RUN and not log.in_reset:  # the policy's commands (not the reset ramp's)
            log.run_deltas.append(np.array(delta, np.float64))
        return send(self, mode, delta, state_seq, episode)

    def _reset(self, q0):
        log.append("R0")
        log.in_reset = True
        reset(self, q0)
        log.in_reset = False
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

    monkeypatch.setattr(LoadedPolicy, "__call__", interrupting)
    _short_episodes(monkeypatch)
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


# ------------------------------------------------------------------------------------------------ review fixes


def test_single_protocol_and_incomplete_ladder_are_never_eligible():
    eps = _eps(1.0, 50, 50)
    assert dp.summarize(eps, [1.0], protocol="zero-shot", ladder_complete=True)["verdict"]["finetune_eligible"]
    assert not dp.summarize(eps, [1.0], protocol="single", ladder_complete=True)["verdict"]["finetune_eligible"]
    assert not dp.summarize(eps, [1.0], protocol="zero-shot", ladder_complete=False)["verdict"]["finetune_eligible"]


@pytest.mark.parametrize(("n", "k", "trip", "passed"), [(10, 8, False, True), (10, 7, False, False),
                                                        (10, 10, True, False), (7, 6, False, True),
                                                        (7, 5, False, False), (1, 1, False, True)])  # fmt: skip
def test_advance_rule(n, k, trip, passed):
    eps = _eps(0.5, n, k)
    if trip:
        eps[-1]["tripped"] = True  # a trip with fault_bits 0 still counts
    chk = dp.advance_check(eps, n)
    assert chk["passed"] is passed and chk["min_successes"] == -(-4 * n // 5)
    assert not dp.advance_check(eps[:-1], n)["passed"]  # a rung not run to the end never passes


class _StubBackend:
    def __init__(self, sm: int) -> None:
        self.sm, self.holds = sm, 0

    def send_hold(self, *_a) -> int:
        self.holds += 1
        return 0

    def read_state(self, st) -> bool:
        st["sm_state"][0] = self.sm
        st["fault_bits"][0] = 0x4 if self.sm == 8 else 0
        return True


@pytest.mark.parametrize("state", ["SAFE_STOPPING", "TORQUE_OFF", "LATCHED"])
def test_wait_for_run_refuses_a_stopped_servo(state):
    from so101_rl.env.backend import ServoNotRunning
    from so101_rl.safety.types import SmState

    b = _StubBackend(int(SmState[state]))
    with pytest.raises(ServoNotRunning, match=state):
        dp.wait_for_run(b, 5.0)
    assert b.holds == 1  # HOLD only, never anything that moves the arm


def test_wait_for_run_times_out_and_returns_on_run():
    from so101_rl.env.backend import ServoStateStale
    from so101_rl.safety.types import SmState

    msgs: list[str] = []
    with pytest.raises(ServoStateStale):
        dp.wait_for_run(_StubBackend(int(SmState.IDLE_TORQUE_OFF)), 0.25, msgs.append)
    assert len(msgs) == 1 and "IDLE_TORQUE_OFF" in msgs[0]
    dp.wait_for_run(_StubBackend(int(SmState.RUN)), 1.0)


def test_goal_list_is_drawn_from_the_seed_only():
    class Seq:
        def sample(self, rng):
            return rng.uniform(size=3)

    a = dp.draw_goal_pairs(Seq(), 7, 5)
    assert np.array_equal(a, dp.draw_goal_pairs(Seq(), 7, 9)[:5])  # prefix-stable: pair i depends on (seed, i)
    assert not np.array_equal(a, dp.draw_goal_pairs(Seq(), 8, 5))
    gl = dp.GoalList(a)
    gl.select(3)
    assert np.array_equal(gl.sample(None), a[3, 0]) and not gl.redrawn
    assert np.array_equal(gl.sample(None), a[3, 1]) and gl.redrawn


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_while_waiting_for_the_servo_exits_130_without_traceback(tmp_path, sig):
    p = _export(tmp_path / "p.pt")
    proc = subprocess.Popen(
        [sys.executable, "-m", "so101_rl.rl.deploy_policy", "--policy", str(p), "--shm-name",
         f"so101none_{uuid.uuid4().hex[:8]}", "--out-root", str(tmp_path / "out"), "--wait-servo-s", "60"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    err = []
    for line in proc.stderr:
        err.append(line)
        if "looking for the servo" in line:
            break
    proc.send_signal(sig)
    _out, rest = proc.communicate(timeout=30)
    err = "".join(err) + rest
    assert proc.returncode == dp.EXIT_INTERRUPTED, err
    assert "Traceback" not in err and "interrupted before the attach" in err
    assert not (tmp_path / "out").exists()


def _run_dir(root: Path) -> tuple[Path, list[dict], dict]:
    (run,) = root.iterdir()
    eps = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    return run, eps, json.loads((run / "summary.json").read_text())


def _deploy(policy: Path, servo: ServoChild, out: Path, *extra: str) -> int:
    return dp.main(["--policy", str(policy), "--shm-name", servo.name, "--out-root", str(out), "--wait-servo-s", "60",
                    *extra])  # fmt: skip


@pytest.fixture
def const_policy(monkeypatch):
    monkeypatch.setattr(LoadedPolicy, "__call__", lambda self, obs: np.full(6, 0.4, np.float32))


def _inject(monkeypatch, kind: str, at_episode: int, at_step: int = 2) -> None:
    """The servo 'leaves RUN' (``trip``: sm_state HOLD, fault_bits 0) or stalls at global episode ``at_episode``."""
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.safety.types import SmState

    step = So101GoalEnv.step

    def patched(self, action):
        obs, r, term, trunc, info = step(self, action)
        if self.goal_sampler._i == at_episode and self._steps == at_step:
            if kind == "trip":
                info["trip"], info["trips"], info["sm_state"] = 0, 0, int(SmState.HOLD)
            else:
                info["servo_stalled"] = True
            self._needs_reset = True
            return obs, r, False, True, info
        return obs, r, term, trunc, info

    monkeypatch.setattr(So101GoalEnv, "step", patched)


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_zero_shot_ladder_order_and_speed_scaling(tmp_path, command_log, const_policy, monkeypatch):
    """--rung-episodes 1,1,1 against the fake servo with a constant action: rungs in order 0.5 / 0.75 / 1.0 and every
    RUN delta = action_to_delta(0.4 * speed). The advance threshold is lowered to 0 (the constant policy never
    succeeds); the rule itself is covered by test_advance_rule and test_ladder_stops_below_the_advance_threshold."""
    from so101_rl.config import load_calibration, load_config
    from so101_rl.env.tasks import action_to_delta_np

    _short_episodes(monkeypatch)
    monkeypatch.setattr(dp, "ADVANCE_MIN_SUCCESS_FRACTION", (0, 1))
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = _deploy(_export(tmp_path / "p.pt"), servo, tmp_path / "deploy", "--protocol", "zero-shot",
                       "--rung-episodes", "1,1,1")  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_OK
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert [e["speed"] for e in eps] == [0.5, 0.75, 1.0] and [e["episode"] for e in eps] == [0, 1, 2]
    assert summary["halted"] is None and summary["verdict"]["ladder_complete"] is True
    assert summary["servo"]["mcu"] is True and len(summary["goals"]["pairs"]) == 3
    for e in eps:
        pair = summary["goals"]["pairs"][e["episode"]]
        assert np.allclose(e["goal"], pair[1 if e["goal_redrawn"] else 0], atol=2e-5)
    max_delta = load_config(None, calibration=load_calibration(CALIB)).action.max_delta_ticks
    segs = _episode_segments(command_log)
    assert len(segs) == 3 and len(command_log.run_deltas) == 15
    for k, speed in enumerate((0.5, 0.75, 1.0)):
        assert segs[k][:5] == ["RUN"] * 5
        want = action_to_delta_np(np.full(6, 0.4, np.float32) * np.float32(speed), max_delta)
        for d in command_log.run_deltas[5 * k : 5 * k + 5]:
            np.testing.assert_array_equal(d, want)
    mags = [np.abs(command_log.run_deltas[5 * k]).sum() for k in range(3)]
    assert mags[0] / mags[2] == pytest.approx(0.5, abs=0.03) and mags[1] / mags[2] == pytest.approx(0.75, abs=0.03)
    assert command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_ladder_stops_below_the_advance_threshold(tmp_path, command_log, monkeypatch):
    """A still policy never succeeds: 0/1 at speed 0.5 < ceil(0.8) -> no climb, exit 1, rung_below_threshold."""
    monkeypatch.setattr(LoadedPolicy, "__call__", lambda self, obs: np.zeros(6, np.float32))
    _short_episodes(monkeypatch)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = _deploy(_export(tmp_path / "p.pt"), servo, tmp_path / "deploy", "--protocol", "zero-shot",
                       "--rung-episodes", "1,1,1")  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_HALTED
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert [e["speed"] for e in eps] == [0.5]
    assert summary["halt_reason"] == "rung_below_threshold" and not summary["verdict"]["finetune_eligible"]
    assert command_log.count("R0") == 1 and command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize("kind", ["trip", "stall"])
def test_trip_or_stall_stops_the_ladder(tmp_path, command_log, const_policy, monkeypatch, kind):
    """The servo leaves RUN with fault_bits 0 (or stalls) in episode 1: the ladder stops there (exit 1), the last
    command is HOLD, no RUN after the halting step, no later episode or rung."""
    _short_episodes(monkeypatch)
    _inject(monkeypatch, kind, at_episode=1, at_step=2)
    monkeypatch.setattr(dp, "ADVANCE_MIN_SUCCESS_FRACTION", (0, 1))
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = _deploy(_export(tmp_path / "p.pt"), servo, tmp_path / "deploy", "--protocol", "zero-shot",
                       "--rung-episodes", "2,2,2")  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_HALTED
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert [e["episode"] for e in eps] == [0, 1] and [e["steps"] for e in eps] == [5, 2]
    assert eps[1]["tripped" if kind == "trip" else "servo_stalled"] and eps[1]["trips"] == 0
    assert summary["halt_reason"] == kind and summary["rungs"][0]["trip_episodes"] == 1
    assert summary["rungs"][1]["n"] == 0 and not summary["verdict"]["finetune_eligible"]
    segs = _episode_segments(command_log)
    assert len(segs) == 2 and segs[1][:2] == ["RUN", "RUN"] and "RUN" not in segs[1][2:]
    assert sum(seg.count("RUN") for seg in segs) == 7 and command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_sigterm_mid_run_exits_130_and_ignores_a_second_signal(tmp_path, command_log, monkeypatch):
    """SIGTERM (from the policy call, in episode 1) -> HOLD, summary, exit 130; a second SIGTERM sent while the final
    HOLD is written is ignored (no traceback, the summary is still written)."""
    state = {"calls": 0, "second": False}

    def policy(self, obs):
        if command_log.count("R1") == 2:
            state["calls"] += 1
            if state["calls"] == 2:
                os.kill(os.getpid(), signal.SIGTERM)
        return np.zeros(6, np.float32)

    hold = dp._hold

    def hold_and_signal(backend):
        if state["calls"] >= 2 and not state["second"]:
            state["second"] = True
            os.kill(os.getpid(), signal.SIGTERM)
        return hold(backend)

    monkeypatch.setattr(LoadedPolicy, "__call__", policy)
    monkeypatch.setattr(dp, "_hold", hold_and_signal)
    _short_episodes(monkeypatch)
    before = signal.getsignal(signal.SIGTERM)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = _deploy(_export(tmp_path / "p.pt"), servo, tmp_path / "deploy", "--episodes", "3")
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_INTERRUPTED and state["second"]
    assert signal.getsignal(signal.SIGTERM) is before  # restored
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert len(eps) == 1 and summary["exit_code"] == 130 and summary["halt_reason"] == "interrupted"
    assert command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_servo_pid_change_after_run_is_refused(tmp_path, command_log, monkeypatch):
    real = dp.servo_info
    calls = {"n": 0}

    def swapped(name, wait_s=0.0):
        info = real(name, wait_s)
        calls["n"] += 1
        return info if calls["n"] == 1 else dataclasses.replace(info, pid=info.pid + 1)

    monkeypatch.setattr(dp, "servo_info", swapped)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = _deploy(_export(tmp_path / "p.pt"), servo, tmp_path / "deploy", "--episodes", "2")
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_HALTED and calls["n"] == 2
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert eps == [] and summary["halt_reason"] == "servo_changed"
    assert "RUN" not in command_log and "R0" not in command_log and command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(400)
def test_resume_after_a_trip_continues_the_goal_list(tmp_path, command_log, const_policy, monkeypatch):
    """Trip in the 2nd episode of the full-speed rung -> exit 1; a fresh servo and --resume continue at that rung's
    next episode with the seeded goals, append to the same run dir and write one combined summary."""
    _short_episodes(monkeypatch)
    monkeypatch.setattr(dp, "ADVANCE_MIN_SUCCESS_FRACTION", (0, 1))
    policy = _export(tmp_path / "p.pt")
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    with monkeypatch.context() as m:
        _inject(m, "trip", at_episode=3)
        try:
            code = _deploy(policy, servo, tmp_path / "deploy", "--protocol", "zero-shot", "--rung-episodes", "1,1,3",
                           "--seed", "11")  # fmt: skip
        finally:
            assert servo.stop() == 0
    assert code == dp.EXIT_HALTED
    run, eps1, s1 = _run_dir(tmp_path / "deploy")
    assert [e["episode"] for e in eps1] == [0, 1, 2, 3] and s1["halt_reason"] == "trip"
    servo2 = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = dp.main(["--resume", str(run), "--shm-name", servo2.name, "--wait-servo-s", "60"])
    finally:
        assert servo2.stop() == 0
    assert code == dp.EXIT_OK
    _run, eps, s2 = _run_dir(tmp_path / "deploy")  # still one run dir
    assert [e["episode"] for e in eps] == [0, 1, 2, 3, 4] and [e["speed"] for e in eps] == [0.5, 0.75, 1.0, 1.0, 1.0]
    assert eps[:4] == eps1
    assert s2["goals"] == s1["goals"] and s2["started"] == s1["started"] and s2["seed"] == 11
    for e in eps:
        assert np.allclose(e["goal"], s2["goals"]["pairs"][e["episode"]][1 if e["goal_redrawn"] else 0], atol=2e-5)
    (res,) = s2["resumes"]
    assert res["first_episode"] == 4 and res["servo_pid"] == servo2.proc.pid
    assert res["tick_log"] == str(servo2.tick_log)
    assert s2["overall"]["n"] == 5 and s2["rungs"][2]["n"] == 3 and s2["rungs"][2]["trip_episodes"] == 1
    assert s2["halted"] is None and s2["verdict"]["ladder_complete"] is True
    # a finished run resumed again: nothing to run, exit 0, nothing sent but HOLD
    servo3 = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    n_cmds = len(command_log)
    try:
        assert dp.main(["--resume", str(run), "--shm-name", servo3.name, "--wait-servo-s", "60"]) == dp.EXIT_OK
    finally:
        assert servo3.stop() == 0
    assert set(command_log[n_cmds:]) == {"HOLD"}


def _orig_summary(policy: Path) -> dict:
    return {"policy": str(policy.resolve()), "protocol": "zero-shot", "seed": 7,
            "plan": [{"speed": 0.5, "episodes": 2}, {"speed": 0.75, "episodes": 2}, {"speed": 1.0, "episodes": 3}],
            "servo": {"real": False}, "goals": {"pairs": []}}  # fmt: skip


@pytest.mark.parametrize(
    ("extra", "match"),
    [(["--policy", "other.pt"], "--policy"), (["--seed", "8"], "--seed"), (["--protocol", "single"], "--protocol"),
     (["--rung-episodes", "2,2,4"], "plan"), (["--rung-episodes", "2,2,3"], None), ([], None)],
)  # fmt: skip
def test_resume_refuses_a_different_protocol(tmp_path, extra, match):
    policy = _export(tmp_path / "p.pt")
    args = dp.build_parser().parse_args(["--resume", str(tmp_path), *extra])
    if match is None:
        assert dp._resolve_run_args(args, _orig_summary(policy)) == [(0.5, 2), (0.75, 2), (1.0, 3)]
        assert args.policy == policy.resolve() and args.seed == 7 and args.protocol == "zero-shot"
    else:
        with pytest.raises(dp.Refused, match=match):
            dp._resolve_run_args(args, _orig_summary(policy))


def test_resume_refuses_a_different_servo_class(tmp_path, capsys, no_hardware, pretend_servo):
    policy = _export(tmp_path / "p.pt")
    run = tmp_path / "run"
    run.mkdir()
    doc = _orig_summary(policy)
    doc["servo"]["real"] = True  # the run was on the arm; this servo is a sim one
    (run / "summary.json").write_text(json.dumps(doc))
    name = pretend_servo("mujoco", "--calibration", str(CALIB))
    assert dp.main(["--resume", str(run), "--shm-name", name, "--wait-servo-s", "0"]) == dp.EXIT_REFUSED
    assert "--resume" in capsys.readouterr().err
    assert not (run / "episodes.jsonl").exists()


def test_resume_refuses_a_non_run_dir(tmp_path, capsys, no_hardware):
    assert dp.main(["--resume", str(tmp_path), "--wait-servo-s", "0"]) == dp.EXIT_REFUSED
    assert "--resume" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ attach -> RUN keep-alive


def _record_cmds(monkeypatch) -> list[tuple[float, str]]:
    """(monotonic time, "ATTACH" | "HOLD" | "RUN") for every attach and shm command, with a cold goal-table cache:
    the ~7 s table build (what used to run between finding the servo and attaching) happens in the test."""
    from so101_rl.env import goals

    monkeypatch.setattr(goals, "_TABLE_CACHE", {})
    log: list[tuple[float, str]] = []
    init, send = ShmBackend.__init__, ShmBackend._send

    def _init(self, *a, **k):
        init(self, *a, **k)
        log.append((time.monotonic(), "ATTACH"))

    def _send(self, mode, delta, state_seq, episode):
        log.append((time.monotonic(), "RUN" if mode == CmdMode.RUN else "HOLD"))
        return send(self, mode, delta, state_seq, episode)

    monkeypatch.setattr(ShmBackend, "__init__", _init)
    monkeypatch.setattr(ShmBackend, "_send", _send)
    return log


def _assert_hold_from_attach_to_run(log: list[tuple[float, str]], max_gap_s: float = 0.15) -> None:
    """From the attach until the first RUN command only HOLD, never more than ~100 ms apart (servo limit 0.5 s)."""
    kinds = [k for _t, k in log]
    a, r = kinds.index("ATTACH"), kinds.index("RUN")
    assert set(kinds[a + 1 : r]) == {"HOLD"}, kinds[a : r + 1]
    times = [t for t, _k in log[a:r]]
    gaps = np.diff(times)
    assert times[-1] - times[0] > 1.0  # the slow set-up really ran after the attach (cold goal table)
    assert gaps.max() <= max_gap_s, f"HOLD gap {gaps.max() * 1e3:.0f} ms after the attach"


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["mujoco", "fake"])
def test_servo_default_start_reaches_episode_0_without_stale_cmd(tmp_path, monkeypatch, backend):
    """Regression (64679b6): the goal table was built between finding the servo and attaching, so a servo with its
    default (home) start that reaches RUN on its own (``--backend mujoco``: ~5 s; the fake: at once) safe-stopped with
    STALE_CMD (0x100) before episode 0. The servo is started first, as in the bug report."""
    _short_episodes(monkeypatch)
    policy = _export(tmp_path / "p.pt")
    log = _record_cmds(monkeypatch)
    servo = ServoChild(tmp_path, backend)
    try:
        code = _deploy(policy, servo, tmp_path / "deploy", "--episodes", "1")
    finally:
        assert servo.stop() == 0
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert code == dp.EXIT_OK, summary["halted"]
    assert summary["halted"] is None and len(eps) == 1 and eps[0]["trips"] == 0 and not eps[0]["tripped"]
    _assert_hold_from_attach_to_run(log)


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.timeout(300)
def test_resume_against_a_mujoco_default_start_has_no_stale_cmd(tmp_path, monkeypatch):
    """The --resume path attaches the same way: a stopped run (trip in episode 0, fake servo) resumed against a fresh
    ``servo --backend mujoco`` with its default start reaches the next episode without STALE_CMD."""
    _short_episodes(monkeypatch)
    policy = _export(tmp_path / "p.pt")
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    with monkeypatch.context() as m:
        _inject(m, "trip", at_episode=0)
        try:
            assert _deploy(policy, servo, tmp_path / "deploy", "--episodes", "2") == dp.EXIT_HALTED
        finally:
            assert servo.stop() == 0
    run, eps1, _s = _run_dir(tmp_path / "deploy")
    assert len(eps1) == 1
    log = _record_cmds(monkeypatch)
    servo2 = ServoChild(tmp_path, "mujoco")
    try:
        code = dp.main(["--resume", str(run), "--shm-name", servo2.name, "--wait-servo-s", "60"])
    finally:
        assert servo2.stop() == 0
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert code == dp.EXIT_OK, summary["halted"]
    assert len(eps) == 2 and eps[1]["trips"] == 0 and not eps[1]["tripped"] and summary["halted"] is None
    _assert_hold_from_attach_to_run(log)


# ------------------------------------------------------------------------------------------------ reset: lift then return

WALL_GOAL = (0.25358, -0.15443, 0.03161)  # seed-7 goal 33: the M6 policy ends with the tip on the y = -0.22 wall


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.timeout(400)
def test_reset_after_a_wall_press_does_not_trip_stuck(tmp_path, monkeypatch):
    """Regression (plan §10 2026-10-08 "Deploy reset: lift then return"): with the old joint ramp, the reset after
    an M6 episode on the wall goal pushed the tip into the box wall, the servo's clamp held it and STUCK (0x80)
    tripped in the reset. The episodes may fail; every reset must complete."""
    if not M6_POLICY.is_file():
        pytest.skip(f"the M6 policy export is not on this host ({M6_POLICY})")
    monkeypatch.setattr(dp, "draw_goal_pairs", lambda _s, _seed, n: np.tile(np.array(WALL_GOAL), (n, 2, 1)))
    servo = ServoChild(tmp_path, "mujoco")
    try:
        code = _deploy(M6_POLICY, servo, tmp_path / "deploy", "--episodes", "3", "--speed-scale", "1.0")
    finally:
        assert servo.stop() == 0
    _run, eps, summary = _run_dir(tmp_path / "deploy")
    assert code == dp.EXIT_OK, summary["halted"]
    assert summary["halted"] is None and len(eps) == 3
    assert all(e["trips"] == 0 and not e["tripped"] for e in eps)
    assert np.allclose(eps[0]["goal"], WALL_GOAL, atol=1e-4)
