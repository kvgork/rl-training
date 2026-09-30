"""Servo <-> policy shared memory (plan F3): one segment, two seqlock blocks, CRC32 on the payload.

Layout: ``[header 64 B][cmd block 64 B][state block 256 B]``. Command block: written by the policy, read by
the servo. State block: written by the servo, read by the policy. Each block is single-writer.

The first field of every record (``seq``) is the seqlock word: odd while the writer is inside, even and
strictly increasing (by 2 per write) when stable, 0 = never written. A repeated ``seq`` therefore means "no new
record". The CRC32 covers every byte between ``seq`` and the ``crc32`` field, so torn-read detection does not
depend on store ordering.

NOTE: the plan (F3) asserts ``STATE_DTYPE.itemsize == 192``; the listed fields sum to 232 bytes with natural
alignment. The fields are kept exactly as specified and the block stride is 256 (64-byte aligned).
"""

from __future__ import annotations

import multiprocessing.resource_tracker as _rt
import zlib
from enum import IntEnum
from multiprocessing import shared_memory
from typing import Final, Protocol, Self

import numpy as np

SHM_MAGIC: Final[int] = 0x534F31303152_4C31  # "SO101RL1"
SHM_LAYOUT_VERSION: Final[int] = 1


class CmdMode(IntEnum):
    IDLE = 0
    RUN = 1
    HOLD = 2
    STOP_REQ = 3


HEADER_DTYPE = np.dtype(
    [("magic", "<u8"), ("version", "<u4"), ("cmd_size", "<u4"), ("state_size", "<u4"), ("_pad", "V44")]
)  # 64 B

CMD_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_mono_ns", "<i8"),
        ("state_seq", "<u8"),
        ("mode", "<u4"),
        ("episode", "<u4"),
        ("delta_ticks", "<f4", (6,)),
        ("crc32", "<u4"),
        ("_pad", "V4"),
    ],
    align=True,
)

STATE_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_mono_ns", "<i8"),
        ("tick", "<u8"),
        ("overruns", "<u8"),
        ("present_ticks", "<i4", (6,)),
        ("velocity", "<i4", (6,)),
        ("load", "<i4", (6,)),
        ("voltage_dv", "<u1", (6,)),
        ("temp_c", "<u1", (6,)),
        ("status", "<u1", (6,)),
        ("moving", "<u1", (6,)),
        ("target_ticks", "<i4", (6,)),
        ("goal_ticks", "<i4", (6,)),
        ("applied_delta_ticks", "<f4", (6,)),  # the post-clamp delta actually applied this command (L4)
        ("clamped_mask", "<u1", (6,)),
        ("fault_bits", "<u4"),
        ("sm_state", "<u4"),
        ("last_cmd_seq", "<u8"),
        ("hb_ok", "<u1"),
        ("crc32", "<u4"),
    ],
    align=True,
)

HEADER_SIZE: Final[int] = 64
CMD_SIZE: Final[int] = CMD_DTYPE.itemsize
STATE_SIZE: Final[int] = STATE_DTYPE.itemsize
CMD_STRIDE: Final[int] = -(-CMD_SIZE // 64) * 64
STATE_STRIDE: Final[int] = -(-STATE_SIZE // 64) * 64
CMD_OFFSET: Final[int] = HEADER_SIZE
STATE_OFFSET: Final[int] = CMD_OFFSET + CMD_STRIDE
TOTAL_SIZE: Final[int] = STATE_OFFSET + STATE_STRIDE

assert HEADER_DTYPE.itemsize == 64
assert CMD_SIZE == 64
assert STATE_SIZE == 232


def new_cmd() -> np.ndarray:
    """Zeroed command record, shape ``(1,)``."""
    return np.zeros(1, dtype=CMD_DTYPE)


def new_state() -> np.ndarray:
    """Zeroed state record, shape ``(1,)``."""
    return np.zeros(1, dtype=STATE_DTYPE)


class SeqlockBlock:
    """Single writer, many readers, no locks. Records are numpy structured arrays of shape ``(1,)``."""

    def __init__(self, buf: memoryview, dtype: np.dtype) -> None:
        size = dtype.itemsize
        if len(buf) < size:
            raise ValueError(f"buffer of {len(buf)} B is smaller than one {size} B record")
        if dtype.names[0] != "seq" or dtype.names[-1] not in ("crc32", "_pad"):
            raise ValueError("record must start with 'seq' and contain 'crc32'")
        self.dtype = dtype
        self._size = size
        self._crc_off = dtype.fields["crc32"][1]
        self._raw = np.frombuffer(buf, dtype=np.uint8, count=size)
        self._seq_word = np.frombuffer(buf, dtype="<u8", count=1)
        self._crc_word = np.frombuffer(buf, dtype="<u4", count=1, offset=self._crc_off)
        self._payload = memoryview(self._raw)[8 : self._crc_off]
        self._snap = np.zeros(size, dtype=np.uint8)
        self._snap_payload = memoryview(self._snap)[8 : self._crc_off]
        self._seq = int(self._seq_word[0])  # writer-local copy

    @property
    def last_seq(self) -> int:
        """Seq published by the last :meth:`write` (writer side)."""
        return self._seq

    def write(self, rec: np.ndarray) -> None:
        """seq |= 1 -> payload -> crc -> seq = (seq | 1) + 1. The ``seq`` and ``crc32`` of ``rec`` are ignored."""
        src = rec.reshape(1).view(np.uint8)
        odd = self._seq | 1
        self._seq_word[0] = odd
        self._raw[8 : self._crc_off] = src[8 : self._crc_off]
        self._crc_word[0] = zlib.crc32(self._payload)
        self._seq = odd + 1
        self._seq_word[0] = self._seq

    def read(self, out: np.ndarray, retries: int = 3) -> bool:
        """False = torn / never written / bad CRC; ``out`` is untouched on False."""
        for _ in range(retries):
            s1 = int(self._seq_word[0])
            if s1 == 0 or s1 & 1:
                continue
            self._snap[:] = self._raw
            if int(self._seq_word[0]) != s1:
                continue
            if zlib.crc32(self._snap_payload) != int(self._snap[self._crc_off : self._crc_off + 4].view("<u4")[0]):
                continue
            out.reshape(1).view(np.uint8)[:] = self._snap
            return True
        return False

    def release(self) -> None:
        """Drop every view into the underlying buffer so the shared memory can be closed."""
        for name in ("_raw", "_seq_word", "_crc_word", "_payload"):
            obj = getattr(self, name, None)
            if isinstance(obj, memoryview):
                obj.release()
            setattr(self, name, None)


class ServoIpc(Protocol):
    """What ServoCore needs; :class:`ServoShm` and :class:`LocalIpc` implement it."""

    def read_cmd(self, out: np.ndarray) -> bool: ...

    def write_state(self, rec: np.ndarray) -> None: ...


class _Blocks:
    """Shared implementation of the four ipc verbs over two SeqlockBlocks."""

    _cmd: SeqlockBlock
    _state: SeqlockBlock

    def write_state(self, rec: np.ndarray) -> None:
        self._state.write(rec)

    def read_state(self, out: np.ndarray, retries: int = 3) -> bool:
        return self._state.read(out, retries)

    def write_cmd(self, rec: np.ndarray) -> None:
        self._cmd.write(rec)

    def read_cmd(self, out: np.ndarray, retries: int = 3) -> bool:
        return self._cmd.read(out, retries)

    @property
    def last_cmd_seq(self) -> int:
        return self._cmd.last_seq

    @property
    def last_state_seq(self) -> int:
        return self._state.last_seq


def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    """Attach without registering with the resource tracker (py3.12 has no ``track=False``).

    Registering and later unregistering would make the tracker unlink the servo-owned segment when a
    policy process exits, or raise KeyError noise when the tracker is shared with the creator.
    """
    original = _rt.register
    _rt.register = lambda *_args, **_kwargs: None  # type: ignore[assignment]
    try:
        return shared_memory.SharedMemory(name=name)
    finally:
        _rt.register = original


class ServoShm(_Blocks):
    def __init__(self, shm: shared_memory.SharedMemory, *, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        buf = shm.buf
        self._cmd = SeqlockBlock(buf[CMD_OFFSET : CMD_OFFSET + CMD_SIZE], CMD_DTYPE)
        self._state = SeqlockBlock(buf[STATE_OFFSET : STATE_OFFSET + STATE_SIZE], STATE_DTYPE)
        self._closed = False

    @classmethod
    def create(cls, name: str) -> ServoShm:
        """Servo process only; unlinks on close. A stale segment of the same name is replaced."""
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=TOTAL_SIZE)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
            shm = shared_memory.SharedMemory(name=name, create=True, size=TOTAL_SIZE)
        header = np.frombuffer(shm.buf, dtype=HEADER_DTYPE, count=1)
        header["magic"] = SHM_MAGIC
        header["version"] = SHM_LAYOUT_VERSION
        header["cmd_size"] = CMD_SIZE
        header["state_size"] = STATE_SIZE
        del header
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> ServoShm:
        """Policy side. Never unlinks and never registers with the resource tracker."""
        shm = _attach_untracked(name)
        try:
            if shm.size < TOTAL_SIZE:
                raise RuntimeError(f"shm {name!r}: segment is {shm.size} B, expected >= {TOTAL_SIZE}")
            header = np.frombuffer(shm.buf, dtype=HEADER_DTYPE, count=1).copy()
            if int(header["magic"][0]) != SHM_MAGIC:
                raise RuntimeError(f"shm {name!r}: bad magic {int(header['magic'][0]):#x}")
            if int(header["version"][0]) != SHM_LAYOUT_VERSION:
                raise RuntimeError(f"shm {name!r}: layout version {int(header['version'][0])} != {SHM_LAYOUT_VERSION}")
            if int(header["cmd_size"][0]) != CMD_SIZE or int(header["state_size"][0]) != STATE_SIZE:
                raise RuntimeError(f"shm {name!r}: record sizes differ from this build")
        except RuntimeError:
            shm.close()
            raise
        return cls(shm, owner=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cmd.release()
        self._state.release()
        self._cmd = self._state = None  # type: ignore[assignment]
        try:
            self._shm.close()
        except BufferError:  # a caller still holds a view; unlink anyway, the OS frees it on last close
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


class LocalIpc(_Blocks):
    """In-process ipc (simulation, tests): the same seqlock blocks over a private bytearray."""

    def __init__(self) -> None:
        self._mem = bytearray(TOTAL_SIZE)
        view = memoryview(self._mem)
        self._cmd = SeqlockBlock(view[CMD_OFFSET : CMD_OFFSET + CMD_SIZE], CMD_DTYPE)
        self._state = SeqlockBlock(view[STATE_OFFSET : STATE_OFFSET + STATE_SIZE], STATE_DTYPE)
