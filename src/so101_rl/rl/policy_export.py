"""Policy export for deployment (sim plan S4.3 step 4 / FS10, MVP M4). FROZEN API for the connector.

``pixi run export-policy --ckpt runs/sac-sim/<run>/ckpt_last.pt --out runs/sac-sim/<run>/policy_export.pt``

The export is a plain dict saved with ``torch.save`` (tensors, numbers, strings, lists only: loadable with
``weights_only=True``, no pickled classes): the SB3-shaped actor's ``state_dict`` plus what deployment must check
(``export_version``, ``obs_layout_version``, task, input/action dims, hidden sizes, sim-params release, DR stage, git
sha, the checkpoint's last sim eval).

:func:`load_policy` returns a callable ``obs -> action``: ``obs`` float ``[..., D]`` is the FS3 policy input, i.e.
the task's observation (``env/obs.py``, reach: 21) with ``desired_goal`` (3) appended (``obs.policy_input_dim``);
the action is deterministic ``tanh(mu)`` in [-1, 1], float32 ``[..., 6]``. It refuses a different export or obs
layout version, and inputs of the wrong width.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

__all__ = ["EXPORT_VERSION", "LoadedPolicy", "export_policy", "load_policy"]

EXPORT_VERSION: Final[int] = 1
_FORMAT: Final[str] = "so101_rl.policy_export"


def export_policy(ckpt: Path, out: Path) -> Path:
    """Write the deployable actor of ``ckpt`` to ``out`` (atomic); returns ``out``."""
    from so101_rl.env.obs import OBS_LAYOUT_VERSION
    from so101_rl.rl.eval_sim import load_checkpoint

    d = load_checkpoint(Path(ckpt))
    if int(d.get("obs_layout_version", -1)) != OBS_LAYOUT_VERSION:
        raise ValueError(f"{ckpt}: obs_layout_version {d.get('obs_layout_version')} != {OBS_LAYOUT_VERSION}")
    sac = d["sac"]
    obs_dim, _priv, act_dim = (int(x) for x in sac["dims"])
    last = d.get("last_eval") or {}
    doc: dict[str, Any] = {
        "format": _FORMAT,
        "export_version": EXPORT_VERSION,
        "obs_layout_version": OBS_LAYOUT_VERSION,
        "task_name": str(d["task"]),
        "success_radius_mm": (d.get("config") or {}).get("success_radius_mm"),  # None = the task's (reach 15 mm)
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "actor_hidden": [int(h) for h in sac["cfg"]["actor_hidden"]],
        "log_std_bounds": [float(x) for x in sac["cfg"]["log_std_bounds"]],
        "actor": {k: v.detach().cpu().clone() for k, v in sac["actor"].items()},
        "sim_params_release": d.get("sim_params_release"),
        "calibration_sha256": d.get("calibration_sha256"),
        "dr_stage": int(d.get("dr_stage", 0)),
        "env_step": int(d.get("env_step", 0)),
        "git_sha": str(d.get("git_sha", "unknown")),
        "sim_success_rate": float(last.get("success_rate", float("nan"))),
        "sim_eval_dr_stage": int(last.get("dr_stage", -1)),
        "camera_pose_method": str(d.get("camera_pose_method", "none")),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(doc, tmp)
    tmp.replace(out)
    return out


class LoadedPolicy:
    """Callable ``obs -> action`` (numpy in, numpy out; CPU). ``meta`` holds the export's metadata."""

    def __init__(self, actor: torch.nn.Module, meta: dict[str, Any]) -> None:
        self.actor, self.meta = actor, meta
        self.obs_dim, self.act_dim = int(meta["obs_dim"]), int(meta["act_dim"])

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.shape[-1:] != (self.obs_dim,):
            raise ValueError(f"policy input must be [..., {self.obs_dim}] (FS3 obs ++ desired_goal), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("policy input contains non-finite values")
        a = self.actor(torch.from_numpy(x)).numpy().astype(np.float32)
        return np.clip(a, -1.0, 1.0)


def load_policy(path: Path) -> LoadedPolicy:
    """Load an export (refuses other formats / versions / obs layouts). See the module docstring."""
    from so101_rl.env.obs import OBS_LAYOUT_VERSION
    from so101_rl.rl.fast_sac import Actor

    d = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(d, dict) or d.get("format") != _FORMAT:
        raise ValueError(f"{path}: not a policy export")
    if d.get("export_version") != EXPORT_VERSION:
        raise ValueError(f"{path}: export_version {d.get('export_version')} != {EXPORT_VERSION}")
    if d.get("obs_layout_version") != OBS_LAYOUT_VERSION:
        raise ValueError(f"{path}: obs_layout_version {d.get('obs_layout_version')} != {OBS_LAYOUT_VERSION}")
    actor = Actor(int(d["obs_dim"]), int(d["act_dim"]), tuple(d["actor_hidden"]), tuple(d["log_std_bounds"]))
    actor.load_state_dict(d["actor"])
    actor.eval()
    meta = {k: v for k, v in d.items() if k != "actor"}
    return LoadedPolicy(actor, meta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export-policy", description="Export the deployable actor of a sim checkpoint")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default <ckpt dir>/policy_export.pt")
    a = ap.parse_args(argv)
    out = export_policy(a.ckpt, a.out or a.ckpt.parent / "policy_export.pt")
    load_policy(out)  # round-trip check before reporting success
    print(f"exported {out}")
    return 0 if out.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
