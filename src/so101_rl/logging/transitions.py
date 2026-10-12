"""(obs, action, reward, next_obs, terminated, truncated) arrays from F8 episode files written by ``deploy-policy``.

Row t of an episode file is the observation before step t, the command sent and the servo's record after it; the
next observation of row t is row t+1's, and of the last row ``meta.final_obs`` (``rl.deploy_policy.write_transitions``).
The action is the APPLIED one (lesson L4: train on what the arm did): ``applied_delta_ticks / max_delta_ticks``
clipped to [-1, 1], the same ``applied_action`` the env's reward terms see; the raw command is ``action_cmd``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

OBS_KEYS = ("observation", "achieved_goal", "desired_goal")


def load_episode(path: Path) -> dict[str, np.ndarray]:
    """One episode's transitions; refuses a file without ``meta.final_obs`` (not written by ``deploy-policy``)."""
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        if "final_obs" not in meta or "max_delta_ticks" not in meta:
            raise ValueError(f"{path}: no meta.final_obs / max_delta_ticks (not a deploy-policy episode)")
        obs = {k: f[f"obs/{k}"][()] for k in OBS_KEYS}
        applied = f["applied_delta_ticks"][()]
        out = {
            "action": np.clip(applied / np.asarray(meta["max_delta_ticks"], np.float32), -1.0, 1.0).astype(np.float32),
            "action_cmd": f["action_cmd"][()],
            "reward": f["reward"][()],
            "terminated": f["terminated"][()].astype(bool),
            "truncated": f["truncated"][()].astype(bool),
            "trips": f["trips"][()],
        }
    for k in OBS_KEYS:
        final = np.asarray(meta["final_obs"][k], np.float32)[None]
        out[k] = obs[k]
        out[f"next_{k}"] = np.concatenate([obs[k][1:], final])
    return out


def load_transitions(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """All episodes' transitions concatenated, plus ``episode`` (the index into ``paths`` of each row)."""
    eps = [load_episode(Path(p)) for p in paths]
    if not eps:
        raise ValueError("load_transitions: no episode files")
    out = {k: np.concatenate([e[k] for e in eps]) for k in eps[0]}
    out["episode"] = np.concatenate([np.full(len(e["reward"]), i) for i, e in enumerate(eps)])
    return out


__all__ = ["OBS_KEYS", "load_episode", "load_transitions"]
