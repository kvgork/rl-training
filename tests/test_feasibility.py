# ruff: noqa: F811  (pytest fixtures imported from test_deploy_policy / test_deploy_die)
"""The Phase 4 feasibility-gate runner (``rl/feasibility.py``; plan §10 2026-10-21 "Phase 4 feasibility-gate
runner"): trial positions, the 30 % gate verdict, the perception reads, the refusals, the halt rules against
``servo --backend fake`` with a fake perception writer, and the full loop against the MuJoCo sim twin
(``scripts/feasibility_sim_servo.py``: ``servo --backend mujoco`` + a perception writer fed by the SIM die pose +
die placement), with an operator thread answering the manual-reset gate. Never a serial device."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from test_deploy_die import FakePerception, perception  # noqa: F401
from test_deploy_policy import CALIB, ServoChild, command_log, no_hardware, pretend_servo  # noqa: F401

from so101_rl.env.tasks import DIE_MAX_FOOTPRINT_HALF_M, DIE_TO_ZONE, REACH
from so101_rl.perception.object_table import ObjectTableShm, new_object_table
from so101_rl.rl import feasibility as fz
from so101_rl.rl.scripted import JAW_MIDPOINT_GRIPPER, die_yaw

ROOT = Path(__file__).parent.parent
TWIN = ROOT / "scripts" / "feasibility_sim_servo.py"
REACH_POLICY = Path(os.environ.get("SO101_REACH_POLICY", ROOT / "runs/sac-sim/reach-3mm-20261015-201549/policy.pt"))


def _twin_module():
    spec = importlib.util.spec_from_file_location("feasibility_sim_servo", TWIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ units


def test_trial_positions_cycle_the_stages_inside_the_shrunk_rectangles():
    ds = DIE_TO_ZONE.die_start
    pos = fz.trial_positions(DIE_TO_ZONE, [0, 1, 2], 30, seed=7)
    assert [p["stage"] for p in pos[:4]] == ["S0", "S1", "S2", "S0"]
    assert pos == fz.trial_positions(DIE_TO_ZONE, [0, 1, 2], 30, seed=7)  # the seed alone fixes the list
    assert pos != fz.trial_positions(DIE_TO_ZONE, [0, 1, 2], 30, seed=8)
    m = DIE_MAX_FOOTPRINT_HALF_M - 0.0006  # rounding to 1 mm
    for p in pos:
        k = fz.STAGE_NAMES.index(p["stage"])
        x, y = p["xy"]
        assert any(x0 + m <= x <= x1 - m and y0 + m <= y <= y1 - m for x0, x1, y0, y1 in ds.table_stages[k].rects_xy_m)
    well = fz.trial_positions(DIE_TO_ZONE, [ds.well_stage], 2, seed=0)
    assert well[0]["stage"] == "S3" and well[0]["xy"] == list(ds.well_center_xy_m)
    assert fz._stage_index("table_far", DIE_TO_ZONE) == 1 and fz._stage_index("s2", DIE_TO_ZONE) == 2
    with pytest.raises(ValueError, match="unknown start stage"):
        fz._stage_index("S7", DIE_TO_ZONE)


def test_reach_points_lie_in_the_reach_goal_bounds():
    place = fz.place_point(fz.controller_params())
    z0 = -0.0045
    pts = fz.reach_points(DIE_TO_ZONE, place, tray=True, table_z=z0)
    names = [n for n, _ in pts]
    assert names[0] == "zone" and names[-1] == "S0" and "well" in names and {"S2[0]", "S2[1]", "S1"} <= set(names)
    for _n, p in pts:
        assert p[2] == pytest.approx(z0 + fz.REACH_HEIGHT_M)
        assert REACH.goal_bounds.contains_np(p[None])[0]
    assert "well" not in [n for n, _ in fz.reach_points(DIE_TO_ZONE, place, tray=False, table_z=z0)]


@pytest.mark.parametrize(("succ", "n", "planned", "halted", "status"), [
    (6, 20, 20, False, "PASS"), (5, 20, 20, False, "FAIL"), (20, 20, 20, True, "INCOMPLETE"),
    (10, 12, 20, False, "INCOMPLETE"), (0, 0, 20, False, "INCOMPLETE"),
])  # fmt: skip
def test_gate_verdict_is_30_percent_of_a_complete_run(succ, n, planned, halted, status):
    trials = [{"success": i < succ} for i in range(n)]
    v = fz.gate_verdict(trials, planned, halted=halted)
    assert v["status"] == status and v["pass_rate"] == 0.30 and v["planned"] == planned
    if n:
        from so101_rl.rl.eval_sim import wilson_interval

        assert v["wilson95"] == list(wilson_interval(succ, n)) and v["success_rate"] == pytest.approx(succ / n)


def test_summaries():
    def tr(stage, success, grasp, **kw):
        return {"stage": stage, "success": success, "grasp_success": grasp, "pushed": success and not grasp,
                "knocked": False, "tripped": False, "servo_stalled": False, "visible_fraction": 1.0,
                "zone_error_m": 0.02, "steps": 50, **kw}  # fmt: skip

    s = fz.summarize_scripted([tr("S0", True, True), tr("S1", True, False), tr("S1", False, False, knocked=True)], 3,
                              halted=False)  # fmt: skip
    assert s["status"] == "PASS" and s["per_stage"]["S1"]["successes"] == 1 and s["per_stage"]["S1"]["n"] == 2
    assert (s["grasp_successes"], s["pushed"], s["knocked"]) == (1, 1, 1)
    g = fz.summarize_grasp_heights([
        {"grasp_h_m": 0.018, "grasp_success": True, "die_rise_while_held_m": 0.03, "tripped": False, "trips": 0,
         "clamped_steps": 0, "end_phase": "DONE"},
        {"grasp_h_m": 0.0145, "grasp_success": True, "die_rise_while_held_m": 0.03, "tripped": False, "trips": 0,
         "clamped_steps": 0, "end_phase": "DONE"},
        {"grasp_h_m": 0.012, "grasp_success": False, "die_rise_while_held_m": 0.0, "tripped": True, "trips": 128,
         "clamped_steps": 3, "end_phase": "DESCEND"},
    ])  # fmt: skip
    assert g["lowest_lifted_grasp_h_m"] == 0.0145
    assert "lowest lifted: 14.5 mm" in fz._table({"grasp_height": g, "halted": None})


def test_trial_halt_rules():
    base = {"trial": 3, "tripped": False, "servo_stalled": False, "steps": 10, "visible_steps": 10, "sm_state": 3,
            "trips": 0}  # fmt: skip
    assert fz.trial_halt(base) is None
    assert fz.trial_halt({**base, "tripped": True, "trips": 128}).reason == "trip"
    assert fz.trial_halt({**base, "servo_stalled": True}).reason == "stall"
    assert fz.trial_halt({**base, "halt_reason": "perception_stale", "halt": "x"}).reason == "perception_stale"
    assert fz.trial_halt({**base, "visible_steps": 5}) is None  # exactly half: not more than 50 %
    h = fz.trial_halt({**base, "visible_steps": 4})
    assert h.reason == "tracker_unknown" and "6/10" in str(h)


def test_unknown_yaw_handling():
    """The jaw-midpoint grasp point, the bearing yaw turned 45 deg per attempt, and a quaternion the controller reads
    back as that yaw."""
    p = fz.controller_params()
    assert p.grasp_point_gripper == JAW_MIDPOINT_GRIPPER
    assert fz.yaw_guess((0.1, 0.1), 0) == pytest.approx(math.pi / 4)
    assert fz.yaw_guess((0.1, 0.1), 2) == pytest.approx(math.pi / 4 + math.pi / 2)
    assert fz.yaw_guess((0.1, 0.1), 1, "fixed", 10.0) == pytest.approx(math.radians(10) + math.pi / 4)
    for y in (0.0, 0.3, -1.2):
        q = torch.as_tensor(fz.yaw_quat(y))[None]
        assert float(die_yaw(q)[0]) == pytest.approx(y, abs=1e-9)
    g = fz.controller_params(grasp_h=0.022, place_xy=(0.14, -0.06))
    assert g.place_h == 0.022 and g.zone_xy_m == (0.14, -0.06) and g.place_offset_xy_m == (0.0, 0.0)
    assert fz.controller_params(grasp_h=0.012).place_h == 0.017


def test_read_and_sample_die(perception):
    p = perception(xyz=(0.14, -0.06, 0.0035), run=False)
    reader = ObjectTableShm.attach(p.name)
    rec = new_object_table()
    max_age = fz.die_max_age_ns(10.0)
    try:
        status, xyz, age = fz.read_die(reader, rec, time.monotonic_ns(), max_age)
        assert status == "visible" and xyz == pytest.approx([0.14, -0.06, 0.0035], abs=1e-6) and age < 0.5
        p.write(t_ns=time.monotonic_ns() - 500_000_000)  # an old detection: not fresh
        assert fz.read_die(reader, rec, time.monotonic_ns(), max_age)[0] == "invisible"
        p.visible = False
        p.write()
        assert fz.read_die(reader, rec, time.monotonic_ns(), max_age)[0] == "invisible"
        p.visible, p.fiducial_ok = True, False
        p.write()
        assert fz.read_die(reader, rec, time.monotonic_ns(), max_age)[0] == "fiducial"
        p.fiducial_ok = True
        th = threading.Thread(target=lambda: [p.write() or time.sleep(1 / 30) for _ in range(12)])
        th.start()
        s = fz.sample_die(reader, seconds=0.3)
        th.join()
        assert s.ok() and s.xyz == pytest.approx([0.14, -0.06, 0.0035], abs=1e-6) and s.spread_m < 1e-6
        p.visible = False
        p.write()
        s = fz.sample_die(reader, seconds=0.1)
        assert not s.ok() and s.visible == 0 and s.xyz is None
    finally:
        reader.close()


class _ScriptedTable:
    """An object-table stand-in replaying die positions (one per read), always fresh and visible."""

    def __init__(self, xyzs) -> None:
        self.xyzs, self.i = list(xyzs), 0

    def read(self, out) -> bool:
        from so101_rl.perception.object_table import set_object

        out[...] = new_object_table()
        now = time.monotonic_ns()
        set_object(out, 0, "die", self.xyzs[min(self.i, len(self.xyzs) - 1)], 0.9, True, now)
        out["t_mono_ns"][0], out["fiducial_ok"][0] = now, 1
        self.i += 1
        return True


def test_sample_die_drops_a_stray_read():
    """A read from before the die was put down (150 mm away) does not spoil the average or halt the trial; a die
    that keeps moving does."""
    good = (0.14, -0.06, 0.0035)
    s = fz.sample_die(_ScriptedTable([(0.15, -0.13, 0.0035)] + [good] * 40), seconds=0.2, period_s=0.01)
    assert s.ok() and s.inliers == s.visible - 1 and s.xyz == pytest.approx(good, abs=1e-6)
    torn = _ScriptedTable([good] * 40)
    read = torn.read
    calls = [0]

    def flaky(out):  # every third read torn: never counted as a read, never stale
        calls[0] += 1
        return calls[0] % 3 != 0 and read(out)

    torn.read = flaky
    s = fz.sample_die(torn, seconds=0.2, period_s=0.01)
    assert s.ok() and s.stale_reads == 0 and s.visible == s.reads
    moving = [(0.10 + 0.01 * i, -0.06, 0.0035) for i in range(40)]
    s = fz.sample_die(_ScriptedTable(moving), seconds=0.2, period_s=0.01)
    assert not s.ok() and s.inliers < s.visible / 2


# ------------------------------------------------------------------------------------------------ refusals


def test_refuses_a_grasp_height_into_the_table(capsys, no_hardware):
    assert fz.main(["--protocol", "grasp-height", "--grasp-heights", "0.02,0.009", "--wait-servo-s", "0"]) == 2
    assert "must be >= 0.011" in capsys.readouterr().err


def test_refuses_the_real_arm_without_attended(tmp_path, capsys, no_hardware, pretend_servo, perception):
    name = pretend_servo("feetech", "--calibration", str(CALIB), "--heartbeat-port", "/dev/null-mcu")
    p = perception()
    code = fz.main(["--protocol", "scripted", "--shm-name", name, "--object-table", p.name, "--out-root",
                    str(tmp_path / "out"), "--wait-servo-s", "0"])  # fmt: skip
    assert code == 2 and "--attended" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_refuses_without_a_perception_writer(tmp_path, capsys, monkeypatch, no_hardware, pretend_servo):
    monkeypatch.setattr(fz, "PERCEPTION_WAIT_S", 0.0)
    name = pretend_servo("mujoco", "--calibration", str(CALIB))
    code = fz.main(["--protocol", "scripted", "--shm-name", name, "--object-table", f"so101objnone_{uuid.uuid4().hex[:8]}",
                    "--out-root", str(tmp_path / "out"), "--wait-servo-s", "0"])  # fmt: skip
    assert code == 2 and "no perception writer" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ halt rules (fake servo)


def _operator(run_root: Path, stop: threading.Event, seen: list[int], *, after=None) -> threading.Thread:
    """Answers every ``die_reset_waiting`` with ``die_ready`` (after 0.6 s: longer than the servo's stale-command
    limit, so the HOLD keep-alive must cover the wait); ``after(k)`` runs once the trial ``k`` is released."""

    def run() -> None:
        while not stop.is_set():
            for w in run_root.glob(f"*/{'die_reset_waiting'}"):
                try:
                    k = int(w.read_text())
                except (OSError, ValueError):
                    continue
                if k in seen:
                    continue
                time.sleep(0.6)
                seen.append(k)
                (w.parent / "die_ready").write_text("")
                if after is not None:
                    after(k)
            time.sleep(0.02)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _fake_run(tmp_path, p, *extra, after=None):
    stop, seen = threading.Event(), []
    run_root = tmp_path / "feas"
    th = _operator(run_root, stop, seen, after=after)
    servo = ServoChild(tmp_path, "fake", "--fake-start", "rest")
    try:
        code = fz.main(["--protocol", "scripted", "--shm-name", servo.name, "--object-table", p.name, "--out-root",
                        str(run_root), "--wait-servo-s", "60", "--window-s", "0.3", *extra])  # fmt: skip
    finally:
        stop.set()
        th.join()
        assert servo.stop() == 0
    (run,) = run_root.iterdir()
    lines = (run / "trials.jsonl").read_text().splitlines() if (run / "trials.jsonl").exists() else []
    return code, run, [json.loads(x) for x in lines], json.loads((run / "summary.json").read_text()), seen


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_die_not_seen_before_the_grasp_halts(tmp_path, command_log, perception):
    p = perception(xyz=(0.138, -0.06, 0.0035), visible=False)
    code, run, trials, summary, seen = _fake_run(tmp_path, p, "--trials", "3")
    assert code == fz.EXIT_HALTED and seen == [0] and trials == []
    assert summary["halt_reason"] == "die_not_seen" and summary["scripted"]["status"] == "INCOMPLETE"
    assert command_log[-1] == "HOLD" and not (run / "die_reset_waiting").exists()


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_tracker_unknown_during_a_trial_halts(tmp_path, command_log, perception):
    """Visible during the pre-grasp window, then never again: the trial ends at --max-steps with the die UNKNOWN on
    > 50 % of its steps, and the run stops after it (the fake servo has no die: nothing is ever grasped)."""
    p = perception(xyz=(0.138, -0.06, 0.0035))

    def hide(_k):
        time.sleep(0.6)  # after the 0.3 s window
        p.visible = False

    code, _run, trials, summary, seen = _fake_run(tmp_path, p, "--trials", "3", "--max-steps", "12", after=hide)
    assert code == fz.EXIT_HALTED and seen == [0] and len(trials) == 1
    t = trials[0]
    assert t["steps"] == 12 and t["visible_fraction"] < 0.5 and not t["success"] and not t["tripped"]
    assert t["window"]["visible"] == t["window"]["reads"] and t["die_start_xyz"] == pytest.approx([0.138, -0.06, 0.0035], abs=1e-4)
    assert summary["halt_reason"] == "tracker_unknown" and command_log[-1] == "HOLD"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_perception_stopping_inside_a_trial_halts(tmp_path, command_log, perception):
    """The perception process dies during trial 0: once its record is > 1 s old the trial stops at once
    (``perception_stale``), long before --max-steps; the last command is HOLD."""
    p = perception(xyz=(0.138, -0.06, 0.0035))

    def kill(_k):
        time.sleep(0.6)
        p.stop()

    code, _run, trials, summary, _seen = _fake_run(tmp_path, p, "--trials", "2", "--max-steps", "100", after=kill)
    assert code == fz.EXIT_HALTED and len(trials) == 1
    assert trials[0]["halt_reason"] == "perception_stale" and trials[0]["steps"] < 40
    assert summary["halt_reason"] == "perception_stale" and command_log[-1] == "HOLD"


# ------------------------------------------------------------------------------------------------ full loop (sim twin)


class TwinChild:
    """``scripts/feasibility_sim_servo.py``: ``servo --backend mujoco`` + fake perception from the sim die pose."""

    def __init__(self, tmp_path: Path, *twin_args: str) -> None:
        tag = uuid.uuid4().hex[:10]
        self.name, self.table = f"so101feas_{tag}", f"so101feasobj_{tag}"
        self.control = tmp_path / "ctl"
        env = {k: v for k, v in os.environ.items() if k != "SO101_HW"}
        self.proc = subprocess.Popen(
            [sys.executable, str(TWIN), "--object-table", self.table, "--control", str(self.control), *twin_args,
             "so101_rl.servo.main", "--backend", "mujoco", "--calibration", str(CALIB), "--shm-name", self.name],
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


def _twin_run(tmp_path, *args, seed: int = 0, operator_delay_s: float = 0.0):
    twin = TwinChild(tmp_path)
    run_root = tmp_path / "feas"
    stop, placed = threading.Event(), []
    th = threading.Thread(target=_twin_module().auto_operator, args=(run_root, twin.control, stop),
                          kwargs={"seed": seed, "log": placed, "delay_s": operator_delay_s}, daemon=True)  # fmt: skip
    th.start()
    try:
        code = fz.main(["--shm-name", twin.name, "--object-table", twin.table, "--out-root", str(run_root),
                        "--wait-servo-s", "120", *args])  # fmt: skip
    finally:
        stop.set()
        th.join()
        assert twin.stop() == 0, twin.proc.stderr.read().decode()[-2000:]
    (run,) = run_root.iterdir()
    trials = [json.loads(x) for x in (run / "trials.jsonl").read_text().splitlines()]
    return code, run, trials, json.loads((run / "summary.json").read_text()), placed


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_full_loop_on_the_mujoco_twin_s0_and_a_grasp_height(tmp_path, command_log):
    """The plan's measurement in sim: the scripted controller fed by the object table (perception's window average,
    no yaw: the die is placed at a random yaw) succeeds from S0 for a fixed seed on CPU MuJoCo through the servo shm,
    and the grasp-height protocol lifts a resting die at 12 mm. Every trial waited for the operator (thread)."""
    code, run, trials, summary, placed = _twin_run(tmp_path, "--protocol", "grasp-height,scripted", "--trials", "2",
                                                   "--starts", "S0", "--grasp-heights", "0.012", "--seed", "7")  # fmt: skip
    assert code == fz.EXIT_OK, summary.get("halted")
    assert [k for k, _t, _p in placed] == [0, 1, 2] and all(p is not None for _k, _t, p in placed)
    g, *scripted = trials
    assert g["protocol"] == "grasp-height" and g["grasp_success"] and g["die_rise_while_held_m"] > 0.02
    assert not g["tripped"] and g["end_phase"] == "DONE"
    assert summary["grasp_height"]["lowest_lifted_grasp_h_m"] == 0.012
    for t in scripted:
        assert t["stage"] == "S0" and t["success"] and t["grasp_success"] and not t["pushed"] and not t["tripped"]
        assert t["visible_fraction"] == 1.0 and t["window"]["visible"] == t["window"]["reads"]
        assert t["placement_error_m"] < 0.002 and t["zone_error_m"] < 0.03
    s = summary["scripted"]
    assert s["status"] == "PASS" and s["successes"] == 2 and summary["servo"]["real"] is False
    assert command_log[-1] == "HOLD" and not (run / "die_target.json").exists()


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_reach_protocol_on_the_mujoco_twin(tmp_path, command_log):
    if not REACH_POLICY.is_file():
        pytest.skip(f"the 3 mm reach policy is not on this host ({REACH_POLICY})")
    code, _run, trials, summary, placed = _twin_run(tmp_path, "--protocol", "reach", "--reach-policy", str(REACH_POLICY))
    assert code == fz.EXIT_OK and placed == []  # no die trial, no operator step
    assert [t["name"] for t in trials] == ["zone", "place", "S2[0]", "S2[1]", "S1", "S0"]
    for t in trials:
        assert t["success"] and t["reach_error_m"] < 0.006 and not t["tripped"]
        assert t["ik"]["grasp"]["pos_residual_mm"] < 1.0 and t["ik"]["grasp"]["axis_tilt_deg"] < 5.0
    assert summary["reach"]["successes"] == 6 and "scripted" not in summary
    assert np.isfinite(summary["reach"]["reach_error_m_max"])



def test_the_operator_wait_holds_with_zero_delta_run_commands_not_hold():
    """2026-10-23 on the real arm: the manual-reset wait held the arm with HOLD inside RUN; the stuck monitor saw the
    reset's last delta with the arm still and tripped STUCK 3.6 s later. (The mujoco twin cannot reproduce it: its
    servo never reports every joint still.) The wait must send zero-delta RUN commands only."""
    import inspect

    from so101_rl.rl import deploy_policy as dp

    class Backend:
        episode = 3

        def __init__(self):
            self.sent = []

        def read_state(self, out):
            out["seq"][0] = 42
            return True

        def send_delta(self, delta, seq, episode):
            self.sent.append(("RUN", np.asarray(delta).copy(), seq, episode))

        def send_hold(self, *a):
            self.sent.append(("HOLD",))

    b = Backend()
    with dp.zero_delta_keepalive(b, period_s=0.02):
        time.sleep(0.15)
    assert len(b.sent) >= 4 and all(c[0] == "RUN" for c in b.sent)
    assert all(not c[1].any() and c[2] == 42 and c[3] == 3 for c in b.sent)
    src = inspect.getsource(fz) + inspect.getsource(dp)
    assert "zero_delta_keepalive(backend):  # not HOLD" in inspect.getsource(fz)
    assert "with hold_keepalive(backend):\n                            gate.wait" not in src
