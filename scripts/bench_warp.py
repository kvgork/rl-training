"""MuJoCo Warp throughput / VRAM / overflow / parity benchmark and the S0 gate (sim-first track S0.3).

    pixi run bench-warp [--nworld 256,1024,2048] [--steps 2000] [--njmax N] [--nconmax N] [--out PATH]

Per N (each in a fresh child process, so VRAM is that N's own footprint): random ctrl inside ``ctrlrange`` resampled
every 50 mj_steps (as in the spike), one captured single-step graph per mj_step, the running max of ``nefc`` /
``nacon`` kept on the device every step (no host sync in the timed loop), and ``check_overflow()`` at the end
(MJWarp's per-world overflow flags are sticky, so every step is covered). Plus one CPU-vs-Warp parity run (sinusoid
around ``rest``, 300 steps). Writes ``runs/bench/warp-<ts>.json``; exit code per the gate status below.

S0 gate (plan §5 S0, sim-track plan S0.3), evaluated at N = 1024: env_steps_per_s >= 500000, process VRAM <= 1500 MB,
no overflow at the configured njmax / nconmax, arm parity <= 1e-6 rad over 300 steps.

Exit codes: 0 gate PASS, 1 gate FAIL, 2 gate NOT EVALUATED — the run cannot certify the gate because N=1024 was not
run, ``--no-parity`` skipped the parity check, ``--njmax`` / ``--nconmax`` overrode the configured sizes (the gate is
about the shipped config), or ``--steps`` < 2000. The measurements are still printed and written.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GATE_N = 1024
GATE_STEPS_PER_S = 500_000
GATE_VRAM_MB = 1500
GATE_PARITY_RAD = 1e-6
GATE_MIN_STEPS = 2000


def _process_vram_mb(pid: int) -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            return float(parts[1])
    return None


def _sim_cfg(args):
    from so101_rl.config import load_config

    calib = ROOT / "tests" / "data" / "so101_follower.calib.json"
    from so101_rl.config import load_calibration

    w = load_config(calibration=load_calibration(calib)).sim_warp
    return dataclasses.replace(
        w, njmax=args.njmax if args.njmax is not None else w.njmax,
        nconmax_per_world=args.nconmax if args.nconmax is not None else w.nconmax_per_world,
    )


def child_throughput(args) -> dict:
    import torch

    from so101_rl.sim.model import load_model
    from so101_rl.sim.warp_model import ContactBufferOverflow, WarpSim

    cfg = _sim_cfg(args)
    model, _ = load_model()
    n = args.child_n
    ws = WarpSim(model, cfg, n)
    k = model.key("rest")
    ones = torch.ones(n, dtype=torch.bool, device=ws.device)
    ws.reset_worlds(ones, torch.as_tensor(k.qpos, device=ws.device), torch.zeros(model.nv, device=ws.device),
                    torch.as_tensor(k.ctrl, device=ws.device))
    lo = torch.as_tensor(model.actuator_ctrlrange[:, 0], dtype=torch.float32, device=ws.device)
    hi = torch.as_tensor(model.actuator_ctrlrange[:, 1], dtype=torch.float32, device=ws.device)
    gen = torch.Generator(device=ws.device).manual_seed(0)
    max_nefc = torch.zeros((), dtype=torch.int32, device=ws.device)
    max_nacon = torch.zeros((), dtype=torch.int32, device=ws.device)

    def run(steps: int, track: bool) -> None:
        for s in range(steps):
            if s % 50 == 0:
                ws.ctrl.copy_(lo + (hi - lo) * torch.rand(ws.ctrl.shape, device=ws.device, generator=gen))
            ws.step(1)
            if track:
                torch.maximum(max_nefc, ws.nefc.max(), out=max_nefc)
                torch.maximum(max_nacon, ws.nacon[0], out=max_nacon)

    run(100, True)  # warm-up: graph capture, allocator
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run(args.steps, True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    overflow = "none"
    try:
        ws.check_overflow()
    except ContactBufferOverflow as e:
        overflow = str(e)
    finite = bool(torch.isfinite(ws.qpos).all())
    res = {
        "nworld": n,
        "steps": args.steps,
        "batch_steps_per_s": args.steps / dt,
        "env_steps_per_s": args.steps * n / dt,
        "wall_s": dt,
        "max_nefc": int(max_nefc),
        "njmax": ws.sizes.njmax,
        "max_nacon": int(max_nacon),
        "naconmax": ws.sizes.naconmax,
        "overflow": overflow,
        "solver_limit_worlds": ws.solver_limit_worlds,
        "finite": finite,
        "warp_vram_mb": ws.vram_bytes / 2**20,
        "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "process_vram_mb": _process_vram_mb(os.getpid()),
    }
    try:
        ws.close()
    except ContactBufferOverflow:
        pass
    return res


def child_parity(args) -> dict:
    import mujoco
    import torch

    from so101_rl.sim.model import load_model
    from so101_rl.sim.warp_model import WarpSim

    model, _ = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    mujoco.mj_forward(model, data)
    ws = WarpSim(model, _sim_cfg(args), 4)
    k = model.key("rest")
    ws.reset_worlds(torch.ones(4, dtype=torch.bool, device=ws.device), torch.as_tensor(k.qpos, device=ws.device),
                    torch.zeros(model.nv, device=ws.device), torch.as_tensor(k.ctrl, device=ws.device))
    rest = np.array(k.ctrl)
    kk = np.arange(1, 7) / 3
    arm = die = 0.0
    for t in range(300):
        c = rest + 0.3 * np.sin(2 * np.pi * 0.5 * t * model.opt.timestep * kk)
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        ws.ctrl.copy_(torch.as_tensor(c, dtype=torch.float32, device=ws.device).expand_as(ws.ctrl))
        ws.step(1)
        e = np.abs(ws.qpos.double().cpu().numpy() - data.qpos)
        arm, die = max(arm, float(e[:, :6].max())), max(die, float(e[:, 6:].max()))
    ws.close()
    return {"steps": 300, "worlds": 4, "max_abs_dq_arm_rad": arm, "max_abs_dq_die": die, "timestep_s": model.opt.timestep}


def _child(args, mode: str, n: int | None = None) -> dict:
    cmd = [sys.executable, __file__, "--child", mode, "--steps", str(args.steps)]
    if n is not None:
        cmd += ["--child-n", str(n)]
    if args.njmax is not None:
        cmd += ["--njmax", str(args.njmax)]
    if args.nconmax is not None:
        cmd += ["--nconmax", str(args.nconmax)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"bench child {mode} {n} failed ({out.returncode}):\n{out.stderr[-3000:]}")
    return json.loads(lines[-1])


def gate_status(
    rows: list[dict], parity: dict | None, *, steps: int, njmax: int | None, nconmax: int | None
) -> tuple[str, dict[str, bool], list[str]]:
    """``(status, per-check results, not-evaluated reasons)``; status is "PASS", "FAIL" or "NOT EVALUATED".

    The gate certifies the SHIPPED configuration at full length: any reason below makes the run NOT EVALUATED
    even if every measured number would pass (a partial run must not read as a pass)."""
    reasons: list[str] = []
    g = next((r for r in rows if r["nworld"] == GATE_N), None)
    if g is None:
        reasons.append(f"N={GATE_N} not in --nworld")
    if parity is None:
        reasons.append("parity skipped (--no-parity)")
    if njmax is not None or nconmax is not None:
        reasons.append(f"buffer sizes overridden (--njmax {njmax}, --nconmax {nconmax}); the gate uses so101.yaml")
    if steps < GATE_MIN_STEPS:
        reasons.append(f"--steps {steps} < {GATE_MIN_STEPS}")
    results: dict[str, bool] = {}
    if g is not None:
        results["env_steps_per_s"] = g["env_steps_per_s"] >= GATE_STEPS_PER_S
        results["vram_mb"] = g["process_vram_mb"] is not None and g["process_vram_mb"] <= GATE_VRAM_MB
        results["overflow"] = all(r["overflow"] == "none" and r["finite"] for r in rows)
    if parity is not None:
        results["parity"] = parity["max_abs_dq_arm_rad"] <= GATE_PARITY_RAD
    if reasons:
        return "NOT EVALUATED", results, reasons
    return ("PASS" if all(results.values()) else "FAIL"), results, reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nworld", default="256,1024,2048", help="comma-separated world counts")
    ap.add_argument("--steps", type=int, default=2000, help="timed mj_steps per world count")
    ap.add_argument("--njmax", type=int, default=None, help="override sim_warp.njmax (default: so101.yaml)")
    ap.add_argument("--nconmax", type=int, default=None, help="override sim_warp.nconmax_per_world")
    ap.add_argument("--out", type=Path, default=None, help="json path (default runs/bench/warp-<ts>.json)")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--child", choices=("throughput", "parity"), help=argparse.SUPPRESS)
    ap.add_argument("--child-n", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child == "throughput":
        print(json.dumps(child_throughput(args)))
        return 0
    if args.child == "parity":
        print(json.dumps(child_parity(args)))
        return 0

    nworlds = [int(x) for x in args.nworld.split(",") if x.strip()]
    rows = []
    print(f"{'N':>6} {'env_steps_per_s':>16} {'batch/s':>9} {'vram_mb':>8} {'warp_mb':>8} {'max_nefc':>9} "
          f"{'max_nacon':>10} overflow")
    for n in nworlds:
        r = _child(args, "throughput", n)
        rows.append(r)
        vram = r["process_vram_mb"]
        print(f"{n:>6} {r['env_steps_per_s']:>16,.0f} {r['batch_steps_per_s']:>9.1f} "
              f"{vram if vram is not None else float('nan'):>8.0f} {r['warp_vram_mb']:>8.0f} "
              f"{r['max_nefc']:>4}/{r['njmax']:<4} {r['max_nacon']:>5}/{r['naconmax']:<6} {r['overflow']}")
    parity = None if args.no_parity else _child(args, "parity")
    if parity:
        print(f"parity (CPU f64 vs Warp f32, 300 steps, sinusoid around rest): arm {parity['max_abs_dq_arm_rad']:.2e} rad, "
              f"die {parity['max_abs_dq_die']:.2e}")

    g = next((r for r in rows if r["nworld"] == GATE_N), None)
    status, gate, reasons = gate_status(rows, parity, steps=args.steps, njmax=args.njmax, nconmax=args.nconmax)
    overflow_all = "none" if all(r["overflow"] == "none" for r in rows) else "OVERFLOW"
    print(f"env_steps_per_s at N={GATE_N}: {g['env_steps_per_s']:.0f}" if g else f"N={GATE_N} not run")
    if g:
        print(f"vram_mb at N={GATE_N}: {g['process_vram_mb']}")
    print(f"overflow: {overflow_all}")
    if status == "NOT EVALUATED":
        print("S0 gate: NOT EVALUATED (" + "; ".join(reasons) + ")")
    else:
        print("S0 gate:", status, {k: ("ok" if v else "FAIL") for k, v in gate.items()})

    out = args.out or ROOT / "runs" / "bench" / f"warp-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import mujoco
    import mujoco_warp
    import torch
    import warp

    report = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {"mujoco": mujoco.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", "3.14"),
                     "warp": warp.__version__, "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "ctrl": "uniform in ctrlrange, resampled every 50 mj_steps; one captured graph per mj_step",
        "rows": rows,
        "parity": parity,
        "gate": {"n": GATE_N, "env_steps_per_s_min": GATE_STEPS_PER_S, "vram_mb_max": GATE_VRAM_MB,
                 "parity_rad_max": GATE_PARITY_RAD, "min_steps": GATE_MIN_STEPS, "status": status,
                 "not_evaluated_reasons": reasons, "results": gate, "passed": status == "PASS"},
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return {"PASS": 0, "FAIL": 1, "NOT EVALUATED": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
