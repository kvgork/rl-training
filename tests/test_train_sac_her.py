"""``train-reach-sac --backend shm`` (HS5 software): fine-tuning through a running servo's shm segment, against a
``servo --backend mujoco|fake`` child (never a serial device): the attach and refusals shared with ``deploy-policy``,
the HOLD keep-alive while the updates run between episodes, the session end (trip, session cap) and ``--resume``."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch
from test_deploy_policy import (  # noqa: F401
    M6_POLICY,
    ServoChild,
    _export,
    command_log,
    no_hardware,
    pretend_servo,
)

from so101_rl.rl import train_sac_her as t

FAST = ["--learning-starts", "20", "--warmup-frozen-actor", "1000", "--baseline-eval-every", "10000",
        "--eval-episodes", "1", "--wait-servo-s", "60"]  # fmt: skip


def _short(monkeypatch, steps: int = 10) -> None:
    """Episodes of ``steps`` policy steps (training and eval), so the 10 Hz runs stay short."""
    from so101_rl.env.goal_env import So101GoalEnv

    init = So101GoalEnv.__init__

    def short_init(self, *a, **k):
        init(self, *a, **k)
        self.task = __import__("dataclasses").replace(self.task, episode_steps=steps)

    monkeypatch.setattr(So101GoalEnv, "__init__", short_init)


def _run(tmp_path):
    (run,) = (tmp_path / "out").iterdir()
    sessions = [json.loads(x) for x in (run / "sessions.jsonl").read_text().splitlines()]
    return run, json.loads((run / "summary.json").read_text()), sessions


def test_real_arm_needs_attended_and_no_watchdog_is_accepted(tmp_path, capsys, no_hardware, pretend_servo):  # noqa: F811
    p = _export(tmp_path / "p.pt")
    name = pretend_servo("feetech", "--no-mcu")
    args = ["--init-from", str(p), "--backend", "shm", "--total-steps", "10", "--shm-name", name,
            "--wait-servo-s", "0", "--out-root", str(tmp_path / "out")]  # fmt: skip
    assert t.main(args) == t.EXIT_REFUSED
    assert "pass --attended" in capsys.readouterr().err
    with pytest.raises(AssertionError, match="serial device"):  # past every refusal: it attaches (no_hardware trips)
        t.main([*args, "--attended"])
    err = capsys.readouterr().err
    assert "NO WATCHDOG" in err and "refused" not in err


def test_keepalive_reports_its_own_starved_gaps():
    """A HOLD gap above the stale-command limit (the helper thread starved, here: one slow send) reaches ``on_late``;
    the regular 0.1 s period does not."""
    from so101_rl.rl.deploy_policy import hold_keepalive

    class SlowOnce:
        episode, n = 0, 0

        def send_hold(self, *_a):
            self.n += 1
            if self.n == 3:
                time.sleep(0.4)

    late: list[float] = []
    b = SlowOnce()
    with hold_keepalive(b, on_late=late.append):
        while b.n < 6:
            time.sleep(0.02)
    assert len(late) == 1 and 0.35 < late[0] < 1.0


def test_resume_refuses_a_non_run_dir_and_a_different_backend(tmp_path, capsys):
    assert t.main(["--resume", str(tmp_path)]) == t.EXIT_REFUSED
    (tmp_path / "run.json").write_text(json.dumps({"backend": "sim"}))
    (tmp_path / "model.zip").write_text("")
    assert t.main(["--resume", str(tmp_path), "--backend", "shm"]) == t.EXIT_REFUSED
    assert "--backend sim" in capsys.readouterr().err


def test_success_radius_is_fixed_per_run(tmp_path, capsys):
    """``--success-radius-mm`` (plan §10 2026-10-14): a new run stores it (default REACH's 15 mm), a resume refuses it
    and a run.json from before the flag reads as 15 mm."""
    from so101_rl.env.tasks import REACH

    assert t.RADIUS_MM_BEFORE == pytest.approx(REACH.success_radius_m * 1000.0)
    new = ["--init-from", str(tmp_path / "p.pt"), "--total-steps", "5", "--out-root", str(tmp_path / "out")]
    run, _out, _ = t._resolve_run(t.build_parser().parse_args(new))
    assert run["success_radius_mm"] == 15.0
    run, _out, _ = t._resolve_run(t.build_parser().parse_args([*new, "--success-radius-mm", "5"]))
    assert run["success_radius_mm"] == 5.0
    assert t.main([*new, "--success-radius-mm", "0"]) == t.EXIT_REFUSED
    old = {k: v for k, v in run.items() if k != "success_radius_mm"}
    (tmp_path / "run.json").write_text(json.dumps(old))
    (tmp_path / "model.zip").write_text("")
    run, _out, _ = t._resolve_run(t.build_parser().parse_args(["--resume", str(tmp_path)]))
    assert run["success_radius_mm"] == 15.0
    assert t.main(["--resume", str(tmp_path), "--success-radius-mm", "5"]) == t.EXIT_REFUSED
    assert "fixed for a run" in capsys.readouterr().err


def test_sim_run_uses_the_success_radius(tmp_path, monkeypatch):
    """The env, the HER reward and the baseline eval all see the run's radius: at 0.1 mm nothing succeeds."""
    from so101_rl.env.goal_env import So101GoalEnv

    _short(monkeypatch)
    seen: list[float] = []
    init = So101GoalEnv.__init__

    def spy(self, *a, **k):
        init(self, *a, **k)
        seen.append(self.task.success_radius_m)

    monkeypatch.setattr(So101GoalEnv, "__init__", spy)
    code = t.main(["--init-from", str(_export(tmp_path / "p.pt")), "--total-steps", "20", "--success-radius-mm", "0.1",
                   "--out-root", str(tmp_path / "out"), *FAST[:-2]])  # fmt: skip
    assert code == t.EXIT_OK and seen == [pytest.approx(1e-4)]
    run, summary, _ = _run(tmp_path)
    assert json.loads((run / "run.json").read_text())["success_radius_mm"] == 0.1
    assert json.loads((run / "baseline.json").read_text())["success_rate"] == 0.0


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_shm_session_keeps_the_servo_alive_through_updates_then_caps_and_resumes(tmp_path, command_log, monkeypatch):  # noqa: F811
    """Against the mujoco servo: a 1 s session cap ends the first session at its first episode end (session_cap, exit
    1, everything saved); a fresh servo and --resume continue to --total-steps (exit 0). The updates between
    episodes never let the servo go stale (it is still in RUN for the next episode)."""
    if not M6_POLICY.is_file():
        pytest.skip(f"the M6 policy export is not on this host ({M6_POLICY})")
    _short(monkeypatch)
    servo = ServoChild(tmp_path, "mujoco")
    try:
        code = t.main(["--init-from", str(M6_POLICY), "--backend", "shm", "--total-steps", "40", "--shm-name",
                       servo.name, "--session-s", "1", "--out-root", str(tmp_path / "out"), *FAST])  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == t.EXIT_HALTED
    run, summary, sessions = _run(tmp_path)
    assert summary["halt_reason"] == "session_cap" and all((run / a).is_file() for a in t.ARTIFACTS)
    assert sessions[0]["servo"]["backend"] == "mujoco" and sessions[0]["end_step"] == 10
    assert command_log[-1] == "HOLD"
    from so101_rl.rl.sb3_bridge import FinetuneSAC

    train, slow = FinetuneSAC.train, {"n": 0}

    def slow_train(self, *a, **k):  # each between-episode update takes 1 s > the servo's 0.5 s stale-command limit
        slow["n"] += 1
        time.sleep(1.0)
        return train(self, *a, **k)

    monkeypatch.setattr(FinetuneSAC, "train", slow_train)
    servo2 = ServoChild(tmp_path, "mujoco")
    try:
        code = t.main(["--resume", str(run), "--shm-name", servo2.name, "--wait-servo-s", "60"])
    finally:
        assert servo2.stop() == 0
    assert code == t.EXIT_OK
    _run_dir, summary, sessions = _run(tmp_path)
    assert len(sessions) == 2 and sessions[1]["start_step"] == 10 and summary["steps_done"] >= 40
    assert summary["halt_reason"] is None and summary["n_updates"] > 0
    assert json.loads((run / "baseline.json").read_text())["n"] == 1  # the baseline is not re-run on resume
    timing = [json.loads(x) for x in (run / "timing.jsonl").read_text().splitlines()]
    slow_cycles = [c for c in timing if c.get("updates_s", 0.0) >= 1.0]
    assert slow_cycles and all("last_step_to_hold_s" in c for c in slow_cycles)
    assert not any("keepalive_late_s" in c for c in timing)  # the keep-alive kept its 0.1 s period throughout
    assert any("updates_to_first_step_s" in c for c in timing) and any("reset_step_s" in c for c in timing)
    # slow updates ran between episodes and the next episodes still ran: the keep-alive held the servo in RUN/HOLD
    # (a stale-command safe stop would have failed the next reset: halt_reason servo_not_running)
    assert slow["n"] >= 2
    assert command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_trip_ends_the_session_and_keeps_the_trip_step(tmp_path, command_log, monkeypatch):  # noqa: F811
    """The servo 'leaves RUN' in training episode 1: the session ends there (halt_reason trip, exit 1), the trip step
    is in the saved replay buffer, nothing but HOLD is sent after it."""
    _short(monkeypatch)
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.safety.types import SmState

    step = So101GoalEnv.step
    calls = {"n": 0}

    def tripping(self, action):
        obs, r, term, trunc, info = step(self, action)
        calls["n"] += 1
        if calls["n"] == 10 + 3:  # baseline eval episode (10 steps), then step 3 of the first training episode
            info["trip"], info["trips"], info["sm_state"] = 0, 0, int(SmState.HOLD)
            self._needs_reset = True
            return obs, r, False, True, info
        return obs, r, term, trunc, info

    monkeypatch.setattr(So101GoalEnv, "step", tripping)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = t.main(["--init-from", str(_export(tmp_path / "p.pt")), "--backend", "shm", "--total-steps", "100",
                       "--shm-name", servo.name, "--out-root", str(tmp_path / "out"), *FAST])  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == t.EXIT_HALTED
    run, summary, _sessions = _run(tmp_path)
    assert summary["halt_reason"] == "trip" and summary["steps_done"] == 3
    from stable_baselines3.common.save_util import load_from_pkl

    buf = load_from_pkl(run / "replay_buffer.pkl")
    assert buf.pos == 3 and bool(buf.timeouts[2, 0] or buf.dones[2, 0])  # the trip step is the stored last one
    assert command_log[-1] == "HOLD"
    i_last_run = max(i for i, c in enumerate(command_log) if c == "RUN")
    assert set(command_log[i_last_run + 1 :]) <= {"HOLD"}
    assert torch.load(run / "best_policy.pt", weights_only=False)["step"] == 0
    assert np.isfinite(summary["baseline_success"])
