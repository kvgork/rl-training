"""A sysid run directory from real-arm servo tick logs that were not excitation runs (pixi task ``sysid-from-logs``).

``python -m so101_rl.sim.sysid.from_deploy --out runs/sysid/<name> --train LOG [LOG ...] [--heldout LOG ...]``

``sysid-fit --run`` reads a run directory (``plan.json`` + ``tick_log.bin``, :class:`~.dataset.ExcitationDataset`).
An excitation run writes one; a ``deploy-policy`` session leaves only the servo's ``--tick-log``. This writes the same
layout from such logs: ``tick_log.bin`` is the logs concatenated in the given order (each a different servo process;
the tick counter restarts per log, which ends every contiguous run at a log boundary, and ``t_ns`` is
``CLOCK_MONOTONIC`` of the same boot), and ``plan.json`` has one segment per log covering its whole time range, of
kind ``deploy`` (``--train``) or ``deploy_heldout`` (``--heldout``). Fit with ``--heldout-kinds deploy_heldout`` so a
whole session is held out, not only the 20 % time windows. Only RUN ticks with a good read and an accepted write
become segments (the dataset's rule): policy episodes and the reset ramps between them.

Caveat recorded in ``plan.json``: policy-driven reaching is not designed excitation (fewer frequencies, mostly
small deltas near goals), so a fit from it identifies less than the HS3 excitation run would.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

TRAIN_KIND = "deploy"
HELDOUT_KIND = "deploy_heldout"


def build_run_dir(out: Path, train: list[Path], heldout: list[Path] | None = None) -> dict:
    """Write ``out/plan.json`` and ``out/tick_log.bin``; returns the plan."""
    from so101_rl.servo.core import TICK_LOG_DTYPE

    heldout = list(heldout or [])
    if not train:
        raise ValueError("need at least one --train tick log")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never mix into an existing run
    segments, chunks = [], []
    for i, (path, kind) in enumerate([(p, TRAIN_KIND) for p in train] + [(p, HELDOUT_KIND) for p in heldout]):
        raw = Path(path).read_bytes()
        if not raw or len(raw) % TICK_LOG_DTYPE.itemsize:
            raise ValueError(f"{path}: empty or not a TICK_LOG_DTYPE log ({TICK_LOG_DTYPE.itemsize} B records)")
        recs = np.frombuffer(raw, dtype=TICK_LOG_DTYPE)
        if np.any(np.diff(recs["t_ns"].astype(np.int64)) < 0):
            raise ValueError(f"{path}: t_ns goes backwards")
        if segments and int(recs["t_ns"][0]) <= segments[-1]["t_stop_ns"]:
            raise ValueError(f"{path}: overlaps or precedes the previous log (give the logs in time order)")
        segments.append({
            "index": i,
            "kind": kind,
            "label": Path(path).name,
            "t_start_ns": int(recs["t_ns"][0]),
            "t_stop_ns": int(recs["t_ns"][-1]) + 1,
            "source": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(recs),
        })
        chunks.append(raw)
    plan = {
        "source": "deploy tick logs (so101_rl.sim.sysid.from_deploy)",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "policy-driven reaching, not designed excitation: identifies less than an HS3 excitation run",
        "segments": segments,
    }
    (out / "tick_log.bin").write_bytes(b"".join(chunks))
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="new run directory (must not exist)")
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="servo tick logs to fit on, in time order")
    ap.add_argument("--heldout", type=Path, nargs="*", default=[], help="tick logs held out whole (after --train)")
    args = ap.parse_args(argv)
    from so101_rl.sim.sysid.dataset import ExcitationDataset

    build_run_dir(args.out, args.train, args.heldout)
    ds = ExcitationDataset.from_run(args.out)
    by_kind = {k: sum(len(s) for s in ds.segments if s.kind == k) for k in ds.kinds}
    print(f"{args.out}: {len(ds.segments)} RUN segments, ticks per kind {by_kind}")
    print(f"next: pixi run sysid-fit --run {args.out} --heldout-kinds {HELDOUT_KIND} --out <sim_params.yaml>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
