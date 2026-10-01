"""Episode logger (F8): one HDF5 file per episode, written as ``.h5.tmp`` and renamed on close.

The logged action is the APPLIED post-clamp delta (``applied_delta_ticks``, from the env ``info``),
stored next to the raw policy output ``action_cmd`` (lesson L4: train on what the arm did).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from so101_rl.logging.schema import (
    FINAL_SUFFIX,
    FRAME_FULL_SHAPE,
    FRAME_SMALL_SHAPE,
    OBS_DIM,
    OUTCOMES,
    REQUIRED_DATASETS,
    TMP_SUFFIX,
    ObsDict,
    episode_name,
)

_EP_RE = re.compile(r"^ep_(\d+)\.h5(?:\.tmp)?$")
_git_sha_cache: str | None = None


@dataclass
class StepRecord:
    t_mono_ns: int
    obs: ObsDict
    action_cmd: np.ndarray
    applied_delta_ticks: np.ndarray
    target_ticks: np.ndarray
    present_ticks: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    trips: int
    verifier: int  # 0 UNKNOWN, 1 SUCCESS, 2 FAILURE
    frame_small: np.ndarray | None = None  # uint8 [96,96,3]; world-model training resolution
    frame_full: np.ndarray | None = None  # uint8 [480,640,3]; kept only per the keep_full_frames policy


def _git_sha() -> str:
    global _git_sha_cache
    if _git_sha_cache is None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                cwd=Path(__file__).parent,
            )
            _git_sha_cache = out.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            _git_sha_cache = "unknown"
    return _git_sha_cache


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _shaped(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    return arr


class EpisodeLogger:
    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        keep_full_frames: Literal["never", "failures", "always"] = "failures",
        retention_days: int = 14,
        config_dump: dict | None = None,
    ) -> None:
        if keep_full_frames not in ("never", "failures", "always"):
            raise ValueError(f"keep_full_frames: unknown policy {keep_full_frames!r}")
        if not run_id or "/" in run_id or run_id in (".", ".."):
            raise ValueError(f"run_id: not a plain directory name: {run_id!r}")
        self.root = Path(root)
        self.run_id = run_id
        self.keep_full_frames = keep_full_frames
        self.retention_days = retention_days
        self.config_dump = config_dump or {}
        self.episodes_dir = self.root / run_id / "episodes"
        self._h5: h5py.File | None = None
        self._tmp: Path | None = None
        self._final: Path | None = None
        self._episode = -1
        self._n = 0
        self._has_small = False
        self._full: list[np.ndarray] = []

    @property
    def is_open(self) -> bool:
        return self._h5 is not None

    def _next_index(self) -> int:
        best = -1
        for p in self.episodes_dir.iterdir():
            m = _EP_RE.match(p.name)
            if m:
                best = max(best, int(m.group(1)))
        return best + 1

    def start_episode(self, meta: dict) -> int:
        if self._h5 is not None:
            raise RuntimeError("start_episode: previous episode not ended")
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self._episode = self._next_index()
        self._final = self.episodes_dir / (episode_name(self._episode) + FINAL_SUFFIX)
        self._tmp = self.episodes_dir / (episode_name(self._episode) + TMP_SUFFIX)
        f = h5py.File(self._tmp, "w")
        for name, (dtype, shape) in REQUIRED_DATASETS.items():
            f.create_dataset(
                name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, chunks=(64, *shape)
            )
        f.attrs["run_id"] = self.run_id
        f.attrs["episode"] = self._episode
        f.attrs["config_dump"] = json.dumps(_jsonable(self.config_dump), sort_keys=True)
        f.attrs["git_sha"] = _git_sha()
        f.attrs["created"] = dt.datetime.now(dt.UTC).isoformat()
        f.attrs["meta"] = json.dumps(_jsonable(meta), sort_keys=True)
        self._h5 = f
        self._n = 0
        self._has_small = False
        self._full = []
        return self._episode

    def _append(self, name: str, value: Any) -> None:
        assert self._h5 is not None
        ds = self._h5[name]
        ds.resize(self._n + 1, axis=0)
        ds[self._n] = value

    def log_step(self, rec: StepRecord) -> None:
        f = self._h5
        if f is None:
            raise RuntimeError("log_step: no episode started")
        if set(rec.obs) != {"observation", "achieved_goal", "desired_goal"}:
            raise ValueError(f"obs: unexpected keys {sorted(rec.obs)}")
        row: dict[str, Any] = {
            "t_mono_ns": int(rec.t_mono_ns),
            "obs/observation": _shaped("obs.observation", rec.obs["observation"], (OBS_DIM,)),
            "obs/achieved_goal": _shaped("obs.achieved_goal", rec.obs["achieved_goal"], (3,)),
            "obs/desired_goal": _shaped("obs.desired_goal", rec.obs["desired_goal"], (3,)),
            "action_cmd": _shaped("action_cmd", rec.action_cmd, (6,)),
            "applied_delta_ticks": _shaped("applied_delta_ticks", rec.applied_delta_ticks, (6,)),
            "target_ticks": _shaped("target_ticks", rec.target_ticks, (6,)),
            "present_ticks": _shaped("present_ticks", rec.present_ticks, (6,)),
            "reward": float(rec.reward),
            "terminated": bool(rec.terminated),
            "truncated": bool(rec.truncated),
            "trips": int(rec.trips),
            "verifier": int(rec.verifier),
        }
        small = None
        if rec.frame_small is not None:
            small = _shaped("frame_small", rec.frame_small, FRAME_SMALL_SHAPE)
            if small.dtype != np.uint8:
                raise ValueError(f"frame_small: expected uint8, got {small.dtype}")
        if self._n == 0 and small is not None:
            f.create_dataset(
                "frames_small",
                shape=(0, *FRAME_SMALL_SHAPE),
                maxshape=(None, *FRAME_SMALL_SHAPE),
                dtype="u1",
                chunks=(1, *FRAME_SMALL_SHAPE),
                compression="gzip",
                compression_opts=1,
            )
            self._has_small = True
        if (small is not None) != self._has_small:
            raise ValueError("frame_small: must be present on every step or on none")
        full = None
        if rec.frame_full is not None and self.keep_full_frames != "never":
            full = _shaped("frame_full", rec.frame_full, FRAME_FULL_SHAPE)
            if full.dtype != np.uint8:
                raise ValueError(f"frame_full: expected uint8, got {full.dtype}")

        for name, value in row.items():
            self._append(name, value)
        if small is not None:
            self._append("frames_small", small)
        if full is not None:
            # the outcome (so the keep decision) is only known at end_episode
            self._full.append(full.copy())
        self._n += 1

    def end_episode(self, outcome: Literal["success", "failure", "truncated", "trip"]) -> Path:
        f = self._h5
        if f is None or self._tmp is None or self._final is None:
            raise RuntimeError("end_episode: no episode started")
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome: unknown {outcome!r}")
        keep_full = self.keep_full_frames == "always" or (
            self.keep_full_frames == "failures" and outcome != "success"
        )
        if keep_full and self._full and len(self._full) != self._n:
            raise ValueError(
                f"frame_full: {len(self._full)} frames for {self._n} steps; must be every step or none"
            )
        f.attrs["outcome"] = outcome
        if keep_full and self._full:
            f.create_dataset(
                "frames_full",
                data=np.stack(self._full),
                chunks=(1, *FRAME_FULL_SHAPE),
                compression="gzip",
                compression_opts=1,
            )
        f.flush()
        f.close()
        self._h5 = None
        with open(self._tmp, "rb") as raw:
            os.fsync(raw.fileno())
        os.replace(self._tmp, self._final)
        _fsync_dir(self.episodes_dir)
        final = self._final
        self._tmp = self._final = None
        self._full = []
        return final

    def close(self) -> None:
        """Release the handle of an unfinished episode; its ``.h5.tmp`` stays behind (never a ``.h5``)."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


__all__ = ["EpisodeLogger", "StepRecord"]
