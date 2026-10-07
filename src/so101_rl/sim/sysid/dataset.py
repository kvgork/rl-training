"""Excitation datasets: the 30 Hz tick-log segments of one excitation run (FS7, sim-first track S2.2/S2.3).

A run directory (written by ``run_excitation``) holds::

    plan.json         # segment table: kind, label, joints, t_start_ns / t_stop_ns (servo clock), clamp-active fraction
    tick_log.bin      # servo.core.TICK_LOG_DTYPE records (the servo's --tick-log), one per servo tick
    episodes/*.h5     # F8 episodes, one per segment (EpisodeLogger)

Tick-log convention (``servo/core.py``): ``present[k]`` is read at the START of tick ``k`` and ``target[k]`` is written
during tick ``k``, so ``present[k + 1]`` is the reading after ``target[k]`` was applied for one tick. A
:class:`TickSegment` is a maximal run of consecutive loop ticks in RUN with a good read and an accepted write inside
one plan segment's time window. ``history_targets`` are the targets of the ticks right before row 0 (up to
``MAX_DELAY_TICKS``), so a replay with a command delay can pre-fill its FIFO from the real log.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from so101_rl.config import N_JOINTS
from so101_rl.sim.sysid.params import MAX_DELAY_TICKS

MIN_SEGMENT_TICKS = 10  # shorter RUN runs carry no dynamics worth fitting


@dataclass(frozen=True, eq=False)
class TickSegment:
    target_ticks: np.ndarray  # int32[T, 6] written in tick k
    present_ticks: np.ndarray  # int32[T, 6] read at the start of tick k
    kind: str
    t0_ns: int  # servo clock of row 0
    history_targets: np.ndarray | None = None  # int32[H, 6] targets of the H ticks before row 0 (H <= MAX_DELAY_TICKS)
    label: str = ""
    segment_index: int = -1  # plan segment this came from

    def __post_init__(self) -> None:
        t = np.asarray(self.target_ticks, dtype=np.int32)
        p = np.asarray(self.present_ticks, dtype=np.int32)
        if t.ndim != 2 or t.shape[1] != N_JOINTS or t.shape != p.shape:
            raise ValueError(f"target_ticks {t.shape} / present_ticks {p.shape} must both be [T, {N_JOINTS}]")
        if len(t) < 2:
            raise ValueError("a segment needs at least 2 ticks")
        object.__setattr__(self, "target_ticks", t)
        object.__setattr__(self, "present_ticks", p)
        if self.history_targets is not None:
            h = np.asarray(self.history_targets, dtype=np.int32).reshape(-1, N_JOINTS)
            object.__setattr__(self, "history_targets", h[-MAX_DELAY_TICKS:] if len(h) else None)

    def __len__(self) -> int:
        return len(self.target_ticks)

    def slice(self, start: int, stop: int) -> TickSegment:
        """Rows ``[start, stop)``; the history is taken from the rows before ``start`` (or the parent's history)."""
        if not 0 <= start < stop <= len(self):
            raise IndexError(f"bad slice {start}:{stop} of {len(self)} rows")
        before = self.target_ticks[max(0, start - MAX_DELAY_TICKS) : start]
        if start < MAX_DELAY_TICKS and self.history_targets is not None:
            before = np.concatenate([self.history_targets, before])
        return replace(
            self,
            target_ticks=self.target_ticks[start:stop],
            present_ticks=self.present_ticks[start:stop],
            t0_ns=self.t0_ns + start,  # ordering key only (tick index offset), not a physical time
            history_targets=before if len(before) else None,
        )


@dataclass
class ExcitationDataset:
    """The tick segments of one run (or a split of them)."""

    segments: list[TickSegment]
    run_dir: Path | None = None
    plan: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run(cls, run_dir: Path, *, min_ticks: int = MIN_SEGMENT_TICKS) -> ExcitationDataset:
        from so101_rl.safety.types import SmState
        from so101_rl.servo.core import PHASE_LOOP, TICK_LOG_DTYPE

        run_dir = Path(run_dir)
        plan = json.loads((run_dir / "plan.json").read_text())
        log_path = run_dir / "tick_log.bin"
        if not log_path.is_file() or log_path.stat().st_size == 0:
            raise FileNotFoundError(f"{log_path}: missing or empty tick log")
        if log_path.stat().st_size % TICK_LOG_DTYPE.itemsize:
            raise ValueError(f"{log_path}: size is not a multiple of TICK_LOG_DTYPE ({TICK_LOG_DTYPE.itemsize} B)")
        recs = np.fromfile(log_path, dtype=TICK_LOG_DTYPE)
        good = (
            (recs["phase"] == PHASE_LOOP)
            & (recs["sm_state"] == int(SmState.RUN))
            & (recs["ok"] == 1)
            & (recs["wrote"] == 1)
        )
        segs: list[TickSegment] = []
        for s in plan.get("segments", []):
            if "t_start_ns" not in s or "t_stop_ns" not in s:
                continue
            in_win = good & (recs["t_ns"] >= int(s["t_start_ns"])) & (recs["t_ns"] < int(s["t_stop_ns"]))
            idx = np.flatnonzero(in_win)
            for run in _contiguous(idx, recs["tick"]):
                if len(run) < min_ticks:
                    continue
                i0 = int(run[0])
                h0 = i0
                while h0 > 0 and i0 - h0 < MAX_DELAY_TICKS and recs["phase"][h0 - 1] == PHASE_LOOP and (
                    int(recs["tick"][h0]) - int(recs["tick"][h0 - 1]) == 1
                ):
                    h0 -= 1
                segs.append(
                    TickSegment(
                        target_ticks=recs["target"][run],
                        present_ticks=recs["present"][run],
                        kind=str(s["kind"]),
                        t0_ns=int(recs["t_ns"][i0]),
                        history_targets=recs["target"][h0:i0] if h0 < i0 else None,
                        label=str(s.get("label", "")),
                        segment_index=int(s.get("index", -1)),
                    )
                )
        if not segs:
            raise ValueError(f"{run_dir}: no RUN tick-log records inside any plan segment window")
        return cls(segs, run_dir, plan)

    @property
    def n_ticks(self) -> int:
        return sum(len(s) for s in self.segments)

    @property
    def kinds(self) -> list[str]:
        return sorted({s.kind for s in self.segments})

    def split(
        self, *, heldout_kinds: set[str] | Iterable[str], heldout_fraction_by_time: float = 0.2, seed: int
    ) -> tuple[ExcitationDataset, ExcitationDataset]:
        """Held-out = every segment of ``heldout_kinds`` (by excitation kind) PLUS, from every other segment, one
        contiguous window of ``heldout_fraction_by_time`` of its ticks at a seeded random position (by time).
        Pieces shorter than ``MIN_SEGMENT_TICKS`` stay in train. Returns ``(train, heldout)``."""
        if not 0.0 <= heldout_fraction_by_time < 1.0:
            raise ValueError("heldout_fraction_by_time must be in [0, 1)")
        kinds = set(heldout_kinds)
        rng = np.random.default_rng(seed)
        train: list[TickSegment] = []
        held: list[TickSegment] = []
        for seg in self.segments:
            if seg.kind in kinds:
                held.append(seg)
                continue
            n = len(seg)
            w = round(heldout_fraction_by_time * n)
            if w < MIN_SEGMENT_TICKS or n - w < MIN_SEGMENT_TICKS:
                train.append(seg)
                continue
            a = int(rng.integers(0, n - w + 1))
            held.append(seg.slice(a, a + w))
            for lo, hi in ((0, a), (a + w, n)):
                if hi - lo >= MIN_SEGMENT_TICKS:
                    train.append(seg.slice(lo, hi))
        return ExcitationDataset(train, self.run_dir, self.plan), ExcitationDataset(held, self.run_dir, self.plan)

    def restrict(self, *, kinds: Iterable[str] | None = None, max_ticks: int | None = None) -> ExcitationDataset:
        """A subset (tests and quick fits): only ``kinds``; each segment truncated to ``max_ticks``."""
        out = []
        for s in self.segments:
            if kinds is not None and s.kind not in set(kinds):
                continue
            out.append(s.slice(0, min(len(s), max_ticks)) if max_ticks else s)
        return ExcitationDataset(out, self.run_dir, self.plan)


def tick_log_segments(path: Path, *, kind: str = "tick_log", min_ticks: int = 2) -> list[TickSegment]:
    """Every maximal run of consecutive RUN loop ticks (good read, accepted write) of a whole tick log."""
    from so101_rl.safety.types import SmState
    from so101_rl.servo.core import PHASE_LOOP, TICK_LOG_DTYPE

    path = Path(path)
    if path.stat().st_size % TICK_LOG_DTYPE.itemsize:
        raise ValueError(f"{path}: size is not a multiple of TICK_LOG_DTYPE ({TICK_LOG_DTYPE.itemsize} B)")
    recs = np.fromfile(path, dtype=TICK_LOG_DTYPE)
    good = (recs["phase"] == PHASE_LOOP) & (recs["sm_state"] == int(SmState.RUN)) & (recs["ok"] == 1) & (recs["wrote"] == 1)
    out = []
    for run in _contiguous(np.flatnonzero(good), recs["tick"]):
        if len(run) >= max(2, min_ticks):
            i0 = int(run[0])
            h0 = i0
            while h0 > 0 and i0 - h0 < MAX_DELAY_TICKS and recs["phase"][h0 - 1] == PHASE_LOOP and (
                int(recs["tick"][h0]) - int(recs["tick"][h0 - 1]) == 1
            ):
                h0 -= 1
            hist = recs["target"][h0:i0]
            out.append(TickSegment(recs["target"][run], recs["present"][run], kind, int(recs["t_ns"][i0]),
                                   history_targets=hist if len(hist) else None))
    return out


def _contiguous(idx: np.ndarray, tick: np.ndarray) -> list[np.ndarray]:
    """Split record indices into runs of consecutive records AND consecutive tick counters."""
    if len(idx) == 0:
        return []
    t = tick[idx].astype(np.int64)
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(t) != 1)) + 1
    return np.split(idx, breaks)


__all__ = ["MIN_SEGMENT_TICKS", "ExcitationDataset", "TickSegment", "tick_log_segments"]
