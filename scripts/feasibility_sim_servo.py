"""Sim twin for ``feasibility-gate`` dry runs and tests: ``servo --backend mujoco`` plus a fake perception writer fed
by the SIM die pose, plus "hand placement" of the die.

    pixi run -e cpu python scripts/feasibility_sim_servo.py --object-table so101_objects --control /tmp/feas \\
        [--auto-operator runs/feasibility] [--bias-xy 0.01 0] [--hide-lifted] [--seed 0] \\
        so101_rl.servo.main --backend mujoco --shm-name so101

Everything after ``so101_rl.servo.main`` is the servo's own command line, unchanged: ``feasibility-gate`` (like
deploy-policy) classifies the servo from ``/proc/<pid>/cmdline`` with the servo's parser, which starts after that
marker, so this twin is seen as ``--backend mujoco`` (sim). The servo itself is ``so101_rl.servo.main.main`` untouched;
two hooks run in the servo thread between physics steps (never during one):

* **fake perception:** about every 33 ms the object table (``--object-table``, created here, single writer) gets the
  die's TRUE centroid from the MuJoCo state (``--bias-xy`` adds a fixed camera-like error), visible (``--hide-lifted``:
  not visible while the die is > 8 mm off the table, as if the gripper hid it), fiducial OK, timestamped now.
* **placement:** a ``<control>/place.json`` ``{"x", "y", "yaw"}`` puts the die there at rest (the operator's hand) and
  writes ``<control>/placed.json``.

``--auto-operator RUN_ROOT`` adds the operator: for every ``RUN_ROOT/*/die_reset_waiting`` it reads that run's
``die_target.json``, places the die at the target xy with a random yaw (``--seed``), then creates ``die_ready``. The
tests run :func:`auto_operator` in their own thread instead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

MARKER = "so101_rl.servo.main"
PERIOD_NS = 33_000_000
LIFTED_M = 0.008


def place_request(control: Path, x: float, y: float, yaw: float, timeout_s: float = 10.0) -> dict | None:
    """Ask the twin to put the die at ``(x, y)`` / ``yaw``; returns its ``placed.json`` (None on timeout)."""
    placed = control / "placed.json"
    placed.unlink(missing_ok=True)
    tmp = control / "place.json.tmp"
    tmp.write_text(json.dumps({"x": x, "y": y, "yaw": yaw}))
    tmp.replace(control / "place.json")
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if placed.exists():
            try:
                return json.loads(placed.read_text())
            except ValueError:
                pass
        time.sleep(0.02)
    return None


def auto_operator(run_root: Path, control: Path, stop: threading.Event, *, seed: int = 0,
                  log: list | None = None) -> None:  # fmt: skip
    """The operator for every run under ``run_root``: on ``die_reset_waiting`` place the die at ``die_target.json``'s
    xy with a random yaw, then create ``die_ready``. ``log`` collects ``(trial, target, placed)``."""
    rng = np.random.default_rng(seed)
    done: set[tuple[str, int]] = set()
    while not stop.is_set():
        for w in Path(run_root).glob("*/die_reset_waiting"):
            try:
                k = int(w.read_text())
                tgt = json.loads((w.parent / "die_target.json").read_text())
            except (OSError, ValueError):
                continue
            if (w.parent.name, k) in done or tgt.get("trial") != k:
                continue
            placed = place_request(control, float(tgt["xy"][0]), float(tgt["xy"][1]), float(rng.uniform(0, math.pi / 2)))
            done.add((w.parent.name, k))
            if log is not None:
                log.append((k, tgt, placed))
            stop.wait(0.2)  # the hand leaves: perception publishes the placed die before the confirmation
            (w.parent / "die_ready").write_text("")
        stop.wait(0.05)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if MARKER not in argv:
        print(f"feasibility_sim_servo: the servo's command line must follow {MARKER}", file=sys.stderr)
        return 2
    i = argv.index(MARKER)
    ap = argparse.ArgumentParser(prog="feasibility_sim_servo")
    ap.add_argument("--object-table", required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--auto-operator", type=Path, default=None)
    ap.add_argument("--bias-xy", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--hide-lifted", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv[:i])
    servo_argv = argv[i + 1 :]
    if "--backend" not in servo_argv or servo_argv[servo_argv.index("--backend") + 1] != "mujoco":
        print("feasibility_sim_servo: only --backend mujoco", file=sys.stderr)
        return 2

    import mujoco

    import so101_rl.servo.main as sm
    import so101_rl.sim.model as simmodel
    from so101_rl.env.tasks import DIE_HALF_EDGE_M
    from so101_rl.perception.object_table import ObjectTableShm, new_object_table, set_object

    args.control.mkdir(parents=True, exist_ok=True)
    table = ObjectTableShm.create(args.object_table)
    held: dict = {}
    orig_load = simmodel.load_model

    def load_model(*a, **k):
        model, data = orig_load(*a, **k)
        held["model"], held["data"] = model, data
        j = model.joint(simmodel.DIE_JOINT)
        held["q"], held["v"] = int(j.qposadr[0]), int(j.dofadr[0])
        return model, data

    simmodel.load_model = load_model
    rec = new_object_table()
    bias = np.asarray(args.bias_xy, float)
    z_rest = simmodel.TABLE_Z_M + DIE_HALF_EDGE_M
    last = [0]
    req = args.control / "place.json"

    def hook() -> None:
        m, d = held["model"], held["data"]
        q, v = held["q"], held["v"]
        if req.exists():
            try:
                p = json.loads(req.read_text())
            except ValueError:
                p = None
            if p is not None:
                req.unlink(missing_ok=True)
                d.qpos[q : q + 3] = (p["x"], p["y"], z_rest)
                yaw = float(p.get("yaw", 0.0))
                d.qpos[q + 3 : q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
                d.qvel[v : v + 6] = 0.0
                mujoco.mj_forward(m, d)
                tmp = args.control / "placed.json.tmp"
                tmp.write_text(json.dumps({"xyz": d.qpos[q : q + 3].tolist(), "yaw": yaw}))
                tmp.replace(args.control / "placed.json")
        now = time.monotonic_ns()
        if now - last[0] >= PERIOD_NS:
            last[0] = now
            xyz = d.qpos[q : q + 3].copy()
            visible = not (args.hide_lifted and xyz[2] > z_rest + LIFTED_M)
            xyz[:2] += bias
            rec["n"][0] = 0
            set_object(rec, 0, "die", xyz, 0.9 if visible else 0.0, visible, now)
            rec["t_mono_ns"][0] = now
            rec["fiducial_ok"][0] = 1
            table.write(rec)

    class Clock(sm.RealtimeSimClock):
        def sleep_until(self, t_ns: int) -> None:
            super().sleep_until(t_ns)
            hook()

    sm.RealtimeSimClock = Clock
    stop = threading.Event()
    if args.auto_operator is not None:
        threading.Thread(target=auto_operator, args=(args.auto_operator, args.control, stop),
                         kwargs={"seed": args.seed}, daemon=True).start()  # fmt: skip
    try:
        return sm.main(servo_argv)
    finally:
        stop.set()
        table.close()


if __name__ == "__main__":
    raise SystemExit(main())
