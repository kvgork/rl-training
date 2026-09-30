"""Object table shared memory (plan F9): perception writes, the policy/env reads. Seqlock'd, CRC32-checked.

Uses :class:`so101_rl.servo.shm.SeqlockBlock` (U0). The segment holds exactly one record; unlike the servo
segment it has no header: the record layout is pinned by ``OBJECT_TABLE_DTYPE`` and attach validates the size.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Self

import numpy as np

from so101_rl.servo.shm import SeqlockBlock, _attach_untracked

MAX_OBJECTS = 8

OBJECT_DTYPE = np.dtype(
    [("label", "S16"), ("xyz_base", "<f4", (3,)), ("confidence", "<f4"), ("visible", "<u1"), ("t_mono_ns", "<i8")],
    align=True,
)
OBJECT_TABLE_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_mono_ns", "<i8"),
        ("n", "<u4"),
        ("objects", OBJECT_DTYPE, (MAX_OBJECTS,)),
        ("fiducial_ok", "<u1"),
        ("crc32", "<u4"),
    ],
    align=True,
)
OBJECT_TABLE_SIZE = OBJECT_TABLE_DTYPE.itemsize


def new_object_table() -> np.ndarray:
    """Zeroed record, shape ``(1,)``."""
    return np.zeros(1, dtype=OBJECT_TABLE_DTYPE)


def set_object(
    rec: np.ndarray, i: int, label: str, xyz_base: np.ndarray, confidence: float, visible: bool, t_mono_ns: int
) -> None:
    """Fill slot ``i`` and grow ``n`` so that it covers it."""
    if not 0 <= i < MAX_OBJECTS:
        raise IndexError(f"object slot {i} outside 0..{MAX_OBJECTS - 1}")
    raw = label.encode()
    if len(raw) > 16:
        raise ValueError(f"label {label!r} longer than 16 bytes")
    o = rec["objects"][0, i]
    o["label"] = raw
    o["xyz_base"] = np.asarray(xyz_base, dtype=np.float32)
    o["confidence"] = confidence
    o["visible"] = 1 if visible else 0
    o["t_mono_ns"] = t_mono_ns
    rec["n"][0] = max(int(rec["n"][0]), i + 1)


class ObjectTableShm:
    """Single-writer (perception) / many-reader (policy) object table."""

    def __init__(self, shm: shared_memory.SharedMemory, *, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        self._block = SeqlockBlock(shm.buf[:OBJECT_TABLE_SIZE], OBJECT_TABLE_DTYPE)
        self._closed = False

    @classmethod
    def create(cls, name: str) -> ObjectTableShm:
        """Perception process only; unlinks on close. A stale segment of the same name is replaced."""
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=OBJECT_TABLE_SIZE)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
            shm = shared_memory.SharedMemory(name=name, create=True, size=OBJECT_TABLE_SIZE)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> ObjectTableShm:
        """Reader side; never unlinks, never registers with the resource tracker."""
        shm = _attach_untracked(name)
        if shm.size < OBJECT_TABLE_SIZE:
            shm.close()
            raise RuntimeError(f"object table {name!r}: segment is {shm.size} B, expected >= {OBJECT_TABLE_SIZE}")
        return cls(shm, owner=False)

    def write(self, rec: np.ndarray) -> None:
        if not self._owner:
            raise PermissionError("only the creating (perception) process may write the object table")
        self._block.write(rec)

    def read(self, out: np.ndarray) -> bool:
        """False = torn or never written; ``out`` untouched."""
        return self._block.read(out)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._block.release()
        self._block = None  # type: ignore[assignment]
        try:
            self._shm.close()
        except BufferError:
            pass
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
