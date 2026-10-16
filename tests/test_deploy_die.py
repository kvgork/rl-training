# ruff: noqa: F811  (pytest fixtures imported from test_deploy_policy)
"""die_to_zone in the real-arm entry points (``plans/die-to-zone-plan.md`` D5; plan §10 2026-10-16 "Die support in
deploy-policy (D5)"): the v0 / unverified sim_params refusal on a real servo, the perception (object table) checks,
the HS6 ladder defaults and tracker-UNKNOWN halt, the manual die reset, and ``train-reach-sac --task die_to_zone``.
Servos are ``servo --backend mujoco|fake`` children or pretend processes (never a serial device); the object table
is written by a fake perception thread."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from test_deploy_policy import (  # noqa: F401
    CALIB,
    ServoChild,
    _export,
    _short_episodes,
    command_log,
    no_hardware,
    pretend_servo,
)

from so101_rl.env.backend import ShmBackend
from so101_rl.env.tasks import DIE_TO_ZONE
from so101_rl.perception.object_table import ObjectTableShm, new_object_table, set_object
from so101_rl.rl import deploy_policy as dp
from so101_rl.rl import train_sac_her as t
from so101_rl.rl.fast_sac import Actor
from so101_rl.rl.policy_export import EXPORT_VERSION

ZONE_GOAL = DIE_TO_ZONE.zone.goal_xyz()
VERIFIED = {"sim_params_release": 1, "sim_params_verified": True}


def _die_export(path: Path, **overrides) -> Path:
    """A tiny random die_to_zone export (29 -> 6); by default recorded on sim_params v0 (the waiver's case)."""
    actor = Actor(29, 6, (16, 16), (-5.0, 2.0))
    doc = {
        "format": "so101_rl.policy_export", "export_version": EXPORT_VERSION, "obs_layout_version": 1,
        "task_name": "die_to_zone", "obs_dim": 29, "act_dim": 6, "actor_hidden": [16, 16],
        "log_std_bounds": [-5.0, 2.0], "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
        "sim_params_release": 0, "sim_params_verified": False,
    }  # fmt: skip
    doc.update(overrides)
    for k in [k for k, v in doc.items() if v is _MISSING]:
        del doc[k]
    torch.save(doc, path)
    return path


_MISSING = object()


class FakePerception:
    """A perception-process stand-in: owns an object table and publishes the die at 30 Hz (``xyz``, ``visible``,
    ``fiducial_ok`` may be changed while it runs; ``stop()`` ends the writes, so the record goes stale)."""

    def __init__(self, *, xyz=ZONE_GOAL, visible: bool = True, fiducial_ok: bool = True, run: bool = True) -> None:
        self.name = f"so101objtest_{uuid.uuid4().hex[:10]}"
        self.table = ObjectTableShm.create(self.name)
        self.xyz, self.visible, self.fiducial_ok = np.asarray(xyz, float), visible, fiducial_ok
        self._stop = threading.Event()
        self.write()
        self._t = threading.Thread(target=self._run, daemon=True)
        if run:
            self._t.start()

    def write(self, t_ns: int | None = None) -> None:
        rec = new_object_table()
        set_object(rec, 0, "die", self.xyz, 0.9, self.visible, time.monotonic_ns() if t_ns is None else t_ns)
        rec["t_mono_ns"][0] = time.monotonic_ns() if t_ns is None else t_ns
        rec["fiducial_ok"][0] = 1 if self.fiducial_ok else 0
        self.table.write(rec)

    def _run(self) -> None:
        while not self._stop.wait(1 / 30):
            self.write()

    def stop(self) -> None:
        self._stop.set()
        if self._t.is_alive():
            self._t.join()

    def close(self) -> None:
        self.stop()
        self.table.close()


@pytest.fixture
def perception():
    made: list[FakePerception] = []

    def make(**kw) -> FakePerception:
        made.append(FakePerception(**kw))
        return made[-1]

    yield make
    for p in made:
        p.close()


def _args(policy: Path, shm: str, out: Path, *extra: str) -> list[str]:
    return ["--policy", str(policy), "--shm-name", shm, "--out-root", str(out), "--wait-servo-s", "0", *extra]


def _attach_stub(monkeypatch) -> None:
    def attach(*_a, **_k):
        raise RuntimeError("attach reached (test stub)")

    monkeypatch.setattr(ShmBackend, "__init__", attach)


# ------------------------------------------------------------------------------------------------ units


def test_sim_params_problem():
    assert dp.sim_params_problem(VERIFIED) is None
    assert "v0" in dp.sim_params_problem({"sim_params_release": 0, "sim_params_verified": True})
    assert "no sim_params release" in dp.sim_params_problem({})
    assert "not recorded as verified" in dp.sim_params_problem({"sim_params_release": 1})
    assert "not recorded as verified" in dp.sim_params_problem({"sim_params_release": 2, "sim_params_verified": False})


def test_export_records_sim_params_verified_and_sha(tmp_path):
    """``export_policy`` carries the checkpoint's sim_params release / verified / sha256 (None when absent)."""
    from so101_rl.env.obs import OBS_LAYOUT_VERSION
    from so101_rl.rl.policy_export import export_policy, load_policy

    actor = Actor(29, 6, (8,), (-5.0, 2.0))
    ck = {"format": "so101_rl.sim_sac_ckpt", "obs_layout_version": OBS_LAYOUT_VERSION, "task": "die_to_zone",
          "sac": {"dims": [29, 0, 6], "cfg": {"actor_hidden": [8], "log_std_bounds": [-5.0, 2.0]},
                  "actor": actor.state_dict()},
          "sim_params_release": 0, "sim_params_verified": False, "sim_params_sha256": "ab" * 32}  # fmt: skip
    torch.save(ck, tmp_path / "ck.pt")
    meta = load_policy(export_policy(tmp_path / "ck.pt", tmp_path / "p.pt")).meta
    assert (meta["sim_params_release"], meta["sim_params_verified"], meta["sim_params_sha256"]) == (0, False, "ab" * 32)
    del ck["sim_params_verified"], ck["sim_params_sha256"]
    torch.save(ck, tmp_path / "ck.pt")
    meta = load_policy(export_policy(tmp_path / "ck.pt", tmp_path / "p.pt")).meta
    assert meta["sim_params_verified"] is None and dp.sim_params_problem(meta) is not None


def test_task_comes_from_the_export(tmp_path):
    from so101_rl.rl.policy_export import load_policy

    die = load_policy(_die_export(tmp_path / "d.pt"))
    assert dp.task_for_policy(die) is DIE_TO_ZONE
    with pytest.raises(dp.Refused, match="task"):
        dp.task_for_policy(die, "reach")
    reach = load_policy(_export(tmp_path / "r.pt", success_radius_mm=5.0))
    assert dp.task_for_policy(reach).success_radius_m == pytest.approx(0.005)
    with pytest.raises(dp.Refused, match="task"):
        dp.task_for_policy(reach, "die_to_zone")


def test_die_ladder_defaults_and_verdict():
    def plan(task, *a):
        return dp._resolve_run_args(dp.build_parser().parse_args(["--policy", "x", *a]), None, task_name=task)

    assert plan("die_to_zone", "--protocol", "zero-shot") == [(0.5, 10), (0.75, 10), (1.0, 30)]
    assert plan("die_to_zone", "--protocol", "zero-shot", "--rung-episodes", "1,1,2") == [(0.5, 1), (0.75, 1), (1.0, 2)]
    assert plan("reach", "--protocol", "zero-shot") == [(0.5, 10), (0.75, 10), (1.0, 50)]
    eps = [{"speed": 1.0, "success": True, "final_error_m": None if i == 0 else 0.01, "steps": 10, "trips": 0,
            "visible_fraction": 0.8} for i in range(30)]  # fmt: skip
    s = dp.summarize(eps, [1.0], protocol="zero-shot", ladder_complete=True, min_n=dp.DIE_GATE_RUNG_MIN_EPISODES)
    assert s["verdict"]["finetune_eligible"] and s["verdict"]["min_n"] == 30
    assert s["rungs"][0]["final_error_m_p50"] == pytest.approx(0.01) and s["rungs"][0]["visible_fraction_mean"] == pytest.approx(0.8)
    assert not dp.summarize(eps, [1.0], protocol="zero-shot", ladder_complete=True)["verdict"]["finetune_eligible"]


def test_check_object_table(perception):
    p = perception(run=False)
    reader = ObjectTableShm.attach(p.name)
    try:
        dp.check_object_table(reader, p.name, tries=1)
        p.write(t_ns=time.monotonic_ns() - 3_000_000_000)
        with pytest.raises(dp.Refused, match="old"):
            dp.check_object_table(reader, p.name, tries=2, period_s=0.0)
        p.fiducial_ok = False
        p.write()
        with pytest.raises(dp.Refused, match="fiducial"):
            dp.check_object_table(reader, p.name, tries=2, period_s=0.0)
    finally:
        reader.close()
    with pytest.raises(dp.Refused, match="no perception writer"):
        dp.open_object_table(f"so101objnone_{uuid.uuid4().hex[:8]}")


def test_die_reset_gate_waits_for_the_file(tmp_path):
    msgs: list[str] = []
    gate = dp.DieResetGate(tmp_path, msgs.append, poll_s=0.01)
    gate.ready.write_text("")  # left over from before the prompt: does not count
    done = threading.Event()

    def operator():
        while not gate.waiting.exists():
            time.sleep(0.01)
        assert gate.waiting.read_text().strip() == "3"
        time.sleep(0.1)
        assert not done.is_set()  # still waiting: the stale file was removed at the prompt
        gate.ready.write_text("")

    th = threading.Thread(target=operator)
    th.start()
    waited = gate.wait(3)
    done.set()
    th.join()
    assert waited >= 0.1 and not gate.ready.exists() and not gate.waiting.exists()
    assert "Place the die" in msgs[0]


# ------------------------------------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    "meta",
    [{}, {"sim_params_release": _MISSING, "sim_params_verified": _MISSING},
     {"sim_params_release": 1, "sim_params_verified": _MISSING}, {"sim_params_release": 1}],
)  # fmt: skip
def test_v0_die_export_is_refused_on_a_real_servo(tmp_path, capsys, no_hardware, pretend_servo, perception, meta):
    """plan §10 2026-10-16: a die policy on sim_params v0 / unverified (or not recorded) is NEVER deployed on the real
    arm, whatever else is given; nothing is attached or written."""
    name = pretend_servo("feetech", "--calibration", str(CALIB), "--heartbeat-port", "/dev/null-mcu")
    p = perception()
    code = dp.main(_args(_die_export(tmp_path / "p.pt", **meta), name, tmp_path / "out", "--attended",
                         "--manual-die-reset", "--object-table", p.name))  # fmt: skip
    assert code == dp.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "NEVER deployed" in err and "REAL arm" in err
    assert not (tmp_path / "out").exists()


def test_the_attempt2_die_policy_is_refused_on_a_real_servo(tmp_path, capsys, no_hardware, pretend_servo, perception):
    """The real S4.6 attempt-2 export (v0 prior, waiver: never deployed; HARDWARE_CHECKLIST HS6 shown-to-fail)."""
    policy = Path(__file__).parent.parent / "runs" / "sac-sim" / "die-a2-20261016-185514" / "policy.pt"
    if not policy.is_file():
        pytest.skip(f"the attempt-2 die export is not on this host ({policy})")
    name = pretend_servo("feetech", "--calibration", str(CALIB), "--heartbeat-port", "/dev/null-mcu")
    p = perception()
    code = dp.main(_args(policy, name, tmp_path / "out", "--attended", "--manual-die-reset", "--object-table", p.name))
    assert code == dp.EXIT_REFUSED and "release 0" in capsys.readouterr().err


def test_verified_die_export_on_a_real_servo_needs_the_tray_or_manual_reset_and_perception(
    tmp_path, capsys, monkeypatch, pretend_servo, perception
):
    _attach_stub(monkeypatch)
    name = pretend_servo("feetech", "--calibration", str(CALIB), "--heartbeat-port", "/dev/null-mcu")
    policy = _die_export(tmp_path / "p.pt", **VERIFIED)
    out = tmp_path / "out"
    assert dp.main(_args(policy, name, out, "--attended")) == dp.EXIT_REFUSED
    assert "no tray" in capsys.readouterr().err
    missing = f"so101objnone_{uuid.uuid4().hex[:8]}"
    assert dp.main(_args(policy, name, out, "--attended", "--manual-die-reset", "--object-table", missing)) == 2
    assert "no perception writer" in capsys.readouterr().err
    p = perception(fiducial_ok=False)
    assert dp.main(_args(policy, name, out, "--attended", "--manual-die-reset", "--object-table", p.name)) == 2
    assert "fiducial" in capsys.readouterr().err
    p.fiducial_ok = True
    assert dp.main(_args(policy, name, out, "--attended", "--manual-die-reset", "--object-table", p.name)) == 2
    err = capsys.readouterr().err
    assert "attach reached" in err and "NEVER" not in err
    assert not out.exists()


def test_v0_die_export_is_allowed_on_the_sim_twin(tmp_path, capsys, monkeypatch, pretend_servo, perception):
    _attach_stub(monkeypatch)
    name = pretend_servo("mujoco", "--calibration", str(CALIB))
    p = perception()
    assert dp.main(_args(_die_export(tmp_path / "p.pt"), name, tmp_path / "out", "--object-table", p.name)) == 2
    err = capsys.readouterr().err
    assert "attach reached" in err and "sim twin only" in err


def test_die_on_the_sim_twin_still_needs_perception(tmp_path, capsys, no_hardware, pretend_servo):
    name = pretend_servo("mujoco", "--calibration", str(CALIB))
    missing = f"so101objnone_{uuid.uuid4().hex[:8]}"
    assert dp.main(_args(_die_export(tmp_path / "p.pt"), name, tmp_path / "out", "--object-table", missing)) == 2
    assert "start the perception process" in capsys.readouterr().err


def test_reach_refuses_die_only_flags(tmp_path, capsys, no_hardware):
    assert dp.main(_args(_export(tmp_path / "p.pt"), "unused", tmp_path / "out", "--manual-die-reset")) == 2
    assert "die_to_zone only" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ end to end


def _run_dir(tmp_path: Path) -> tuple[Path, list[dict], dict]:
    (run,) = (tmp_path / "deploy").iterdir()
    eps = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    return run, eps, json.loads((run / "summary.json").read_text())


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_die_deploy_against_the_mujoco_servo(tmp_path, capsys, command_log, perception):
    """A v0 die export on the sim twin: the fixed zone goal, the env reads the fake perception's die (visible, resting
    in the zone): every episode succeeds after the debounce, with the die position and visibility recorded."""
    p = perception()
    servo = ServoChild(tmp_path, "mujoco")
    try:
        code = dp.main(["--policy", str(_die_export(tmp_path / "p.pt")), "--shm-name", servo.name, "--object-table",
                        p.name, "--out-root", str(tmp_path / "deploy"), "--protocol", "zero-shot",
                        "--rung-episodes", "1,1,1", "--wait-servo-s", "120"])  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_OK, capsys.readouterr().err
    run, eps, summary = _run_dir(tmp_path)
    assert len(eps) == 3 and all(e["success"] for e in eps)
    for e in eps:
        assert e["goal"] == pytest.approx(ZONE_GOAL.tolist(), abs=1e-5) and not e["goal_redrawn"]
        assert e["visible_fraction"] == 1.0 and e["unknown_steps"] == 0 and e["final_die_visible"]
        assert e["final_die_xyz"] == pytest.approx(ZONE_GOAL.tolist(), abs=1e-5) and e["trips"] == 0
        assert e["steps"] == DIE_TO_ZONE.success_debounce_steps
        assert (run / e["transitions"]).is_file()
    assert summary["task"] == "die_to_zone" and summary["object_table"] == p.name
    assert summary["manual_die_reset"] is False and summary["unknown_halt_fraction"] == 0.5
    assert np.allclose(summary["goals"]["pairs"], np.broadcast_to(ZONE_GOAL, (3, 2, 3)))
    assert summary["verdict"]["min_n"] == 30 and summary["halted"] is None
    assert summary["servo"]["real"] is False and command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_tracker_unknown_on_most_steps_halts_the_run(tmp_path, command_log, monkeypatch, perception):
    """HS6: the tracker UNKNOWN on > 50 % of an episode's steps stops the run after that episode (exit 1)."""
    _short_episodes(monkeypatch, steps=5)
    p = perception(visible=False)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = dp.main(["--policy", str(_die_export(tmp_path / "p.pt")), "--shm-name", servo.name, "--object-table",
                        p.name, "--out-root", str(tmp_path / "deploy"), "--episodes", "3", "--wait-servo-s", "60"])  # fmt: skip
    finally:
        assert servo.stop() == 0
    assert code == dp.EXIT_HALTED
    _run, eps, summary = _run_dir(tmp_path)
    assert len(eps) == 1
    e = eps[0]
    assert not e["success"] and e["steps"] == 5 and e["unknown_steps"] == 5 and e["visible_fraction"] == 0.0
    assert e["final_die_xyz"] is None and e["final_error_m"] is None
    assert summary["halt_reason"] == "tracker_unknown" and "UNKNOWN on 5/5" in summary["halted"]
    assert command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_manual_die_reset_waits_for_the_operator_then_perception_stale_halts(tmp_path, command_log, monkeypatch,
                                                                             perception):
    """``--manual-die-reset``: before every episode the arm goes home (an extra reset) and the run waits for
    ``<run dir>/die_ready``; the HOLD keep-alive runs meanwhile (the servo stays in RUN). When perception stops after
    episode 1, the check before episode 2 stops the run (``perception_stale``)."""
    _short_episodes(monkeypatch, steps=5)
    p = perception()
    run_episode = dp._run_episode
    done = {"n": 0}

    def counting(env, policy, speed, **kw):
        ep = run_episode(env, policy, speed, **kw)
        done["n"] += 1
        if done["n"] == 2:
            p.stop()  # the perception process dies: its record goes stale
        return ep

    monkeypatch.setattr(dp, "_run_episode", counting)
    confirmations: list[int] = []
    stop = threading.Event()

    def operator():
        while not stop.is_set():
            for w in (tmp_path / "deploy").glob(f"*/{dp.DIE_WAITING_FILE}"):
                try:
                    k = int(w.read_text())
                except (OSError, ValueError):
                    continue
                if k not in confirmations:
                    time.sleep(0.6)  # longer than the servo's stale-command limit: the keep-alive must cover it
                    confirmations.append(k)
                    (w.parent / dp.DIE_READY_FILE).write_text("")
            time.sleep(0.02)

    th = threading.Thread(target=operator, daemon=True)
    th.start()
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = dp.main(["--policy", str(_die_export(tmp_path / "p.pt")), "--shm-name", servo.name, "--object-table",
                        p.name, "--out-root", str(tmp_path / "deploy"), "--episodes", "3", "--manual-die-reset",
                        "--wait-servo-s", "60"])  # fmt: skip
    finally:
        stop.set()
        th.join()
        assert servo.stop() == 0
    assert code == dp.EXIT_HALTED
    run, eps, summary = _run_dir(tmp_path)
    assert confirmations == [0, 1, 2] and len(eps) == 2 and all(e["success"] for e in eps)
    assert summary["halt_reason"] == "perception_stale" and "before episode 2" in summary["halted"]
    assert summary["manual_die_reset"] is True
    assert command_log.count("R1") == 5  # per confirmed episode: the home reset, then the env's reset; + ep 2's home
    assert not (run / dp.DIE_WAITING_FILE).exists()


# ------------------------------------------------------------------------------------------------ train-reach-sac


FAST = ["--learning-starts", "20", "--warmup-frozen-actor", "1000", "--baseline-eval-every", "10000",
        "--eval-episodes", "1"]  # fmt: skip


def _short_env(monkeypatch, steps: int = 10) -> None:
    from so101_rl.env.goal_env import So101GoalEnv

    init = So101GoalEnv.__init__

    def short_init(self, *a, **k):
        init(self, *a, **k)
        self.task = __import__("dataclasses").replace(self.task, episode_steps=steps)

    monkeypatch.setattr(So101GoalEnv, "__init__", short_init)


@pytest.mark.parametrize("meta", [{}, {"sim_params_release": _MISSING}])
def test_train_reach_sac_refuses_a_v0_die_export_on_a_real_servo(tmp_path, capsys, no_hardware, pretend_servo,
                                                                 meta):  # fmt: skip
    name = pretend_servo("feetech", "--no-mcu", "--calibration", str(CALIB))
    args = ["--init-from", str(_die_export(tmp_path / "p.pt", **meta)), "--backend", "shm", "--total-steps", "10",
            "--shm-name", name, "--wait-servo-s", "0", "--out-root", str(tmp_path / "out"), "--attended"]  # fmt: skip
    assert t.main(args) == t.EXIT_REFUSED
    assert "NEVER deployed" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_train_reach_sac_refuses_die_on_a_real_servo_without_a_tray(tmp_path, capsys, no_hardware, pretend_servo):
    name = pretend_servo("feetech", "--no-mcu", "--calibration", str(CALIB))
    args = ["--init-from", str(_die_export(tmp_path / "p.pt", **VERIFIED)), "--backend", "shm", "--total-steps",
            "10", "--shm-name", name, "--wait-servo-s", "0", "--out-root", str(tmp_path / "out"), "--attended"]  # fmt: skip
    assert t.main(args) == t.EXIT_REFUSED
    assert "wait for the tray" in capsys.readouterr().err


def test_train_reach_sac_task_must_match_the_export(tmp_path, capsys):
    die = _die_export(tmp_path / "p.pt")
    base = ["--total-steps", "5", "--out-root", str(tmp_path / "out")]
    assert t.main(["--init-from", str(die), "--task", "reach", *base]) == t.EXIT_REFUSED
    assert "not 'reach'" in capsys.readouterr().err
    assert t.main(["--init-from", str(die), "--success-radius-mm", "5", *base]) == t.EXIT_REFUSED
    assert "reach only" in capsys.readouterr().err
    assert t.main(["--init-from", str(_export(tmp_path / "r.pt")), "--task", "die_to_zone", *base]) == t.EXIT_REFUSED


def test_train_reach_sac_die_on_the_sim_backend(tmp_path, monkeypatch):
    """Structural: the die task on the CPU sim backend (no object table: the die is never visible), obs width 29,
    the zone goal for the eval, a die export written."""
    from so101_rl.env.goal_env import So101GoalEnv
    from so101_rl.rl.policy_export import load_policy

    _short_env(monkeypatch)
    seen = []
    init = So101GoalEnv.__init__

    def spy(self, *a, **k):
        init(self, *a, **k)
        seen.append((self.task.name, self.observation_space["observation"].shape[0], self.object_table))

    monkeypatch.setattr(So101GoalEnv, "__init__", spy)
    code = t.main(["--init-from", str(_die_export(tmp_path / "p.pt")), "--total-steps", "20", "--out-root",
                   str(tmp_path / "out"), *FAST])  # fmt: skip
    assert code == t.EXIT_OK and seen == [("die_to_zone", 26, None)]
    (run,) = (tmp_path / "out").iterdir()
    assert json.loads((run / "run.json").read_text())["task"] == "die_to_zone"
    baseline = json.loads((run / "baseline.json").read_text())
    assert baseline["success_rate"] == 0.0 and np.allclose(baseline["goals"], np.broadcast_to(ZONE_GOAL, (1, 2, 3)))
    meta = load_policy(run / "policy_export.pt").meta
    assert meta["task_name"] == "die_to_zone" and meta["obs_dim"] == 29 and meta["sim_params_release"] == 0


def test_train_reach_sac_die_with_an_object_table_halts_on_tracker_unknown(tmp_path, monkeypatch, perception):
    """With an object table the HS6 rule applies: the sim backend's simulated clock never matches the fake writer's,
    so the die is UNKNOWN on every step and the baseline eval ends the session (``tracker_unknown``)."""
    _short_env(monkeypatch)
    p = perception()
    code = t.main(["--init-from", str(_die_export(tmp_path / "p.pt")), "--total-steps", "20", "--out-root",
                   str(tmp_path / "out"), "--object-table", p.name, *FAST])  # fmt: skip
    assert code == t.EXIT_HALTED
    (run,) = (tmp_path / "out").iterdir()
    assert json.loads((run / "summary.json").read_text())["halt_reason"] == "tracker_unknown"
    assert json.loads((run / "baseline.json").read_text())["tracker_unknown"] is True
