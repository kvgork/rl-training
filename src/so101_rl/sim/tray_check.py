"""Sim twin of hardware check H9: does the start-well tray return a dropped die to the well? (sim track S1.3, D-TRAY)

``pixi run tray-check --drops 100`` (``python -m so101_rl.sim.tray_check``): on the CPU MuJoCo scene, drop the die
``--drops`` times from random points inside the tray rim (random orientation, 30-50 mm above the table), let it
settle for 2 s, and count the drops that end within ``2 sigma`` of the well centre (``tasks.DIE_TO_ZONE.die_start``:
centre and ``sigma_xy_m``). PASS when the return rate is >= 0.95. Writes ``runs/tray/<ts>.json``; exit 0 = PASS,
1 = FAIL.

The arm is held at the ``home`` keyframe (energised, far above the tray). ``--no-tray`` disables the tray's
collisions: the gate shown to fail (a die dropped on the bare table does not come back).

Decision D-TRAY: resets of the batched env place the die at rest in the well from the ``die_start`` distribution
(no per-reset drop-and-settle); this check is what justifies that shortcut, and H9 replaces the placeholder
``die_start`` with the real tray's measured distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from so101_rl.env.tasks import DIE_TO_ZONE
from so101_rl.sim.model import DIE_EDGE_M, TABLE_Z_M, load_model

__all__ = ["PASS_RATE", "TrayReport", "main", "run_tray_check", "tray_geom_ids"]

PASS_RATE = 0.95
SETTLE_S = 2.0
DROP_Z_M = (0.030, 0.050)  # die centre above the table at release
TRAY_PREFIX = "tray_"


@dataclass
class TrayReport:
    drops: int
    returned: int
    return_rate: float
    radius_m: float  # 2 sigma
    well_center_xy_m: tuple[float, float]
    with_tray: bool
    seed: int
    max_final_speed_m_s: float
    final_dist_m: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.return_rate >= PASS_RATE


def tray_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array([g for g in range(model.ngeom) if (model.geom(g).name or "").startswith(TRAY_PREFIX)], np.int32)


def _rim_half_extent(model: mujoco.MjModel, center: np.ndarray) -> float:
    """Inner half-width of the rim around ``center`` (from the rim geoms), minus the die half edge."""
    ids = [g for g in tray_geom_ids(model) if model.geom(g).name.startswith(TRAY_PREFIX + "rim")]
    if not ids:
        raise ValueError("scene has no tray rim geoms (tray_rim_*)")
    inner = min(
        float(np.abs(model.geom_pos[g][:2] - center).max() - model.geom_size[g][:2].min()) for g in ids
    )
    return inner - DIE_EDGE_M / 2


def run_tray_check(drops: int = 100, *, seed: int = 0, with_tray: bool = True) -> TrayReport:
    if drops < 1:
        raise ValueError("drops must be >= 1")
    spec = DIE_TO_ZONE.die_start
    assert spec is not None
    center = np.asarray(spec.well_center_xy_m, np.float64)
    radius = 2.0 * spec.sigma_xy_m
    model, data = load_model()
    tray = tray_geom_ids(model)
    if len(tray) == 0:
        raise ValueError("scene has no tray geoms")
    half = _rim_half_extent(model, center)
    if not with_tray:
        model.geom_contype[tray] = 0
        model.geom_conaffinity[tray] = 0
    die_q = model.jnt_qposadr[model.joint("die_free").id]
    die_v = model.jnt_dofadr[model.joint("die_free").id]
    home = model.key("home").id
    nsteps = round(SETTLE_S / model.opt.timestep)
    rng = np.random.default_rng(seed)
    dists: list[float] = []
    vmax = 0.0
    for _ in range(drops):
        mujoco.mj_resetDataKeyframe(model, data, home)
        xy = center + rng.uniform(-half, half, size=2)
        z = TABLE_Z_M + rng.uniform(*DROP_Z_M)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        data.qpos[die_q : die_q + 3] = (xy[0], xy[1], z)
        data.qpos[die_q + 3 : die_q + 7] = quat
        data.qvel[die_v : die_v + 6] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(nsteps):
            mujoco.mj_step(model, data)
        dists.append(float(np.linalg.norm(data.qpos[die_q : die_q + 2] - center)))
        vmax = max(vmax, float(np.linalg.norm(data.qvel[die_v : die_v + 3])))
    returned = int(sum(d <= radius for d in dists))
    return TrayReport(
        drops=drops,
        returned=returned,
        return_rate=returned / drops,
        radius_m=radius,
        well_center_xy_m=(float(center[0]), float(center[1])),
        with_tray=with_tray,
        seed=seed,
        max_final_speed_m_s=vmax,
        final_dist_m=dists,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tray-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--drops", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/tray"))
    ap.add_argument("--no-tray", action="store_true", help="disable the tray collisions (shown-to-fail twin)")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    rep = run_tray_check(args.drops, seed=args.seed, with_tray=not args.no_tray)
    verdict = "PASS" if rep.passed else "FAIL"
    doc = {**asdict(rep), "passed": rep.passed, "pass_rate": PASS_RATE, "wall_s": round(time.monotonic() - t0, 2)}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(
        f"tray-check: return_rate={rep.return_rate:.3f} ({rep.returned}/{rep.drops} within {rep.radius_m * 1000:.0f} mm"
        f" of the well) tray={'on' if rep.with_tray else 'OFF'} -> {verdict}; wrote {path}"
    )
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
