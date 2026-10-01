"""Artifact checks, retention and disk budget for episode files (L12: check the artifact, not the exit code).

Every gate in :func:`check_episode` is exercised against a deliberately broken file in
``tests/test_artifact_checks.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from so101_rl.logging.schema import (
    FINAL_SUFFIX,
    FLOAT_DATASETS,
    MAX_TICK,
    OPTIONAL_DATASETS,
    OUTCOMES,
    REQUIRED_ATTRS,
    REQUIRED_DATASETS,
    TMP_SUFFIX,
)

if TYPE_CHECKING:
    from so101_rl.config.schema import So101Config

_GB = 1e9
_DAY_S = 86400.0


@dataclass
class ArtifactReport:
    ok: bool
    path: Path
    n_steps: int
    errors: list[str] = field(default_factory=list)


def _is_episode_file(p: Path) -> bool:
    return p.name.startswith("ep_") and (p.name.endswith(FINAL_SUFFIX) or p.name.endswith(TMP_SUFFIX))


def check_episode(path: Path, cfg: So101Config | None = None) -> ArtifactReport:
    """Validate one episode file. ``cfg`` enables the ``|applied_delta| <= max_delta`` gate."""
    path = Path(path)
    errors: list[str] = []
    n_steps = 0

    def done() -> ArtifactReport:
        return ArtifactReport(ok=not errors, path=path, n_steps=n_steps, errors=errors)

    if path.name.endswith(TMP_SUFFIX):
        errors.append("temporary file: episode was never finished (end_episode not called)")
        return done()
    if not path.name.endswith(FINAL_SUFFIX):
        errors.append(f"not an episode file: {path.name}")
        return done()
    if not path.is_file():
        errors.append("file does not exist")
        return done()

    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        errors.append(f"unreadable or truncated file: {exc}")
        return done()

    with f:
        for a in REQUIRED_ATTRS:
            if a not in f.attrs:
                errors.append(f"missing attr: {a}")
        if "outcome" in f.attrs and str(f.attrs["outcome"]) not in OUTCOMES:
            errors.append(f"bad outcome attr: {f.attrs['outcome']!r}")

        data: dict[str, np.ndarray] = {}
        lengths: dict[str, int] = {}
        for name, (dtype, shape) in {**REQUIRED_DATASETS, **OPTIONAL_DATASETS}.items():
            optional = name in OPTIONAL_DATASETS
            if name not in f:
                if not optional:
                    errors.append(f"missing dataset: {name}")
                continue
            ds = f[name]
            if not isinstance(ds, h5py.Dataset):
                errors.append(f"{name}: not a dataset")
                continue
            if ds.dtype != np.dtype(dtype):
                errors.append(f"{name}: dtype {ds.dtype}, expected {np.dtype(dtype)}")
            if ds.shape[1:] != shape:
                errors.append(f"{name}: per-step shape {ds.shape[1:]}, expected {shape}")
            lengths[name] = ds.shape[0]
            if name in REQUIRED_DATASETS:
                try:
                    data[name] = ds[()]
                except (OSError, RuntimeError) as exc:
                    errors.append(f"{name}: unreadable data (truncated file?): {exc}")
            else:
                # Frames are big: only prove they are readable, by touching the last chunk.
                try:
                    if ds.shape[0]:
                        _ = ds[ds.shape[0] - 1]
                except (OSError, RuntimeError) as exc:
                    errors.append(f"{name}: unreadable data (truncated file?): {exc}")

        req_lengths = {k: v for k, v in lengths.items() if k in REQUIRED_DATASETS}
        if req_lengths:
            n_steps = max(set(req_lengths.values()), key=list(req_lengths.values()).count)
            odd = {k: v for k, v in req_lengths.items() if v != n_steps}
            if odd:
                errors.append(f"mismatched lengths (expected {n_steps}): {odd}")
        for name in OPTIONAL_DATASETS:
            if name in lengths and lengths[name] != n_steps:
                errors.append(f"frame count mismatch: {name} has {lengths[name]} frames for {n_steps} steps")
        if n_steps == 0 and req_lengths:
            errors.append("episode has zero steps")

        for name in FLOAT_DATASETS:
            if name in data and not np.all(np.isfinite(data[name])):
                errors.append(f"{name}: contains NaN or inf")

        if "target_ticks" in data:
            t = data["target_ticks"]
            if t.size and (t.min() < 0 or t.max() > MAX_TICK):
                errors.append(f"target_ticks out of [0, {MAX_TICK}]: min {t.min()}, max {t.max()}")
        if cfg is not None and "applied_delta_ticks" in data:
            d = np.abs(data["applied_delta_ticks"])
            limit = np.asarray(cfg.action.max_delta_ticks)
            if d.size and d.shape[1:] == limit.shape and np.any(d > limit):
                errors.append("applied_delta_ticks exceeds max_delta_ticks")

    return done()


def check_run(root: Path) -> list[ArtifactReport]:
    """Check every episode file (finished or ``.tmp``) under ``root``; works on a runs root or one run dir."""
    files = sorted(p for p in Path(root).rglob("ep_*") if p.is_file() and _is_episode_file(p))
    return [check_episode(p) for p in files]


def prune(root: Path, retention_days: int, *, dry_run: bool = True) -> list[Path]:
    """Episode files under ``root`` whose mtime is older than ``retention_days``; deleted only if not ``dry_run``."""
    cutoff = time.time() - retention_days * _DAY_S
    stale = sorted(
        p for p in Path(root).rglob("ep_*") if p.is_file() and _is_episode_file(p) and p.stat().st_mtime < cutoff
    )
    if not dry_run:
        for p in stale:
            p.unlink()
    return stale


def disk_budget_gb(root: Path) -> float:
    """Total size of everything under ``root`` in GB (1e9 bytes)."""
    total = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
    return total / _GB
