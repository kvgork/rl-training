import multiprocessing as mp
import multiprocessing.resource_tracker as rt
import os
import threading
import time
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from so101_rl.servo import shm as shm_mod
from so101_rl.servo.shm import (
    CMD_DTYPE,
    HEADER_DTYPE,
    SHM_LAYOUT_VERSION,
    SHM_MAGIC,
    STATE_DTYPE,
    CmdMode,
    LocalIpc,
    SeqlockBlock,
    ServoShm,
    new_cmd,
    new_state,
)


@pytest.fixture
def name():
    return f"so101test_{uuid.uuid4().hex[:10]}"


def test_dtype_itemsizes():
    assert HEADER_DTYPE.itemsize == 64
    assert CMD_DTYPE.itemsize == 64
    # plan F3 says 192, but the listed fields need 232 with natural alignment (see shm.py docstring)
    assert STATE_DTYPE.itemsize == 232
    assert shm_mod.STATE_STRIDE == 256 and shm_mod.STATE_OFFSET % 64 == 0


def test_create_attach_roundtrip(name):
    with ServoShm.create(name) as servo:
        pol = ServoShm.attach(name)
        cmd = new_cmd()
        cmd["mode"] = CmdMode.RUN
        cmd["delta_ticks"] = np.arange(6, dtype=np.float32)
        pol.write_cmd(cmd)
        got = new_cmd()
        assert servo.read_cmd(got)
        assert np.array_equal(got["delta_ticks"], cmd["delta_ticks"]) and got["mode"][0] == CmdMode.RUN

        st = new_state()
        st["present_ticks"] = [1, 2, 3, 4, 5, 6]
        st["hb_ok"] = 1
        servo.write_state(st)
        out = new_state()
        assert pol.read_state(out)
        assert out["present_ticks"][0].tolist() == [1, 2, 3, 4, 5, 6] and out["hb_ok"][0] == 1
        pol.close()


def test_header_mismatch_raises(name):
    with ServoShm.create(name) as servo:
        header = np.frombuffer(servo._shm.buf, dtype=HEADER_DTYPE, count=1)
        header["version"] = SHM_LAYOUT_VERSION + 1
        with pytest.raises(RuntimeError, match="version"):
            ServoShm.attach(name)
        header["version"] = SHM_LAYOUT_VERSION
        header["magic"] = SHM_MAGIC ^ 1
        with pytest.raises(RuntimeError, match="magic"):
            ServoShm.attach(name)
        header["magic"] = SHM_MAGIC
        header["state_size"] = 1
        with pytest.raises(RuntimeError, match="sizes"):
            ServoShm.attach(name)
        del header


def test_attach_missing_segment_raises():
    with pytest.raises(FileNotFoundError):
        ServoShm.attach("so101test_does_not_exist")


def test_owner_unlinks_attacher_does_not(name):
    servo = ServoShm.create(name)
    pol = ServoShm.attach(name)
    pol.close()
    ServoShm.attach(name).close()  # still there after the policy side closed
    servo.close()
    with pytest.raises(FileNotFoundError):
        ServoShm.attach(name)


def test_stale_segment_is_replaced(name):
    first = ServoShm.create(name)
    first._owner = False  # simulate a crashed servo that never unlinked
    first.close()
    with ServoShm.create(name) as second:
        assert second.read_cmd(new_cmd()) is False  # fresh, never written


def test_never_written_and_repeated_seq():
    ipc = LocalIpc()
    out = new_cmd()
    assert ipc.read_cmd(out) is False
    rec = new_cmd()
    ipc.write_cmd(rec)
    assert ipc.read_cmd(out) and int(out["seq"][0]) == 2
    first = int(out["seq"][0])
    assert ipc.read_cmd(out) and int(out["seq"][0]) == first  # same seq = "no new command"
    ipc.write_cmd(rec)
    assert ipc.read_cmd(out) and int(out["seq"][0]) == first + 2
    assert ipc.last_cmd_seq == first + 2


def test_out_untouched_on_false():
    ipc = LocalIpc()
    out = new_cmd()
    out["episode"] = 77
    assert ipc.read_cmd(out) is False
    assert out["episode"][0] == 77


def test_odd_seq_means_writer_inside():
    ipc = LocalIpc()
    ipc.write_cmd(new_cmd())
    ipc._cmd._seq_word[0] = 5  # writer stuck mid-write
    assert ipc.read_cmd(new_cmd(), retries=2) is False


def test_crc_detects_corruption():
    ipc = LocalIpc()
    rec = new_cmd()
    rec["delta_ticks"] = 3.0
    ipc.write_cmd(rec)
    assert ipc.read_cmd(new_cmd())
    ipc._cmd._raw[40] ^= 0xFF  # flip payload bits behind the lock's back
    assert ipc.read_cmd(new_cmd()) is False


def test_short_buffer_rejected():
    with pytest.raises(ValueError):
        SeqlockBlock(memoryview(bytearray(10)), CMD_DTYPE)


def _stress_writer(name, seconds, ready):
    pol = ServoShm.attach(name)
    rec = new_cmd()
    n = 0
    ready.set()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        n += 1
        rec["state_seq"] = n
        rec["t_mono_ns"] = n
        rec["episode"] = n & 0xFFFFFFFF
        rec["delta_ticks"] = float(n % 1_000_000)
        pol.write_cmd(rec)
    pol.close()


def test_torn_read_stress(name):
    """A writer *process* fills every payload field with one counter; the reader never accepts a mix."""
    with ServoShm.create(name) as servo:
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        proc = ctx.Process(target=_stress_writer, args=(name, 2.0, ready))
        proc.start()
        assert ready.wait(30)
        out = new_cmd()
        accepted = rejected = 0
        last = 0
        deadline = time.monotonic() + 1.8
        while time.monotonic() < deadline:
            if servo.read_cmd(out, retries=1):
                accepted += 1
                n = int(out["state_seq"][0])
                assert int(out["t_mono_ns"][0]) == n
                assert int(out["episode"][0]) == n & 0xFFFFFFFF
                assert np.all(out["delta_ticks"][0] == np.float32(n % 1_000_000))
                assert n >= last  # monotone
                last = n
            else:
                rejected += 1
        proc.join(30)
        assert proc.exitcode == 0
        assert accepted > 100, (accepted, rejected)


# ---- cross-process lifetime and crashed-writer behaviour (review fixes) ------------------------------


def _attach_write_and_exit(name, exit_hard):
    """Policy-like child: attach, publish one command, exit WITHOUT closing (optionally via os._exit)."""
    pol = ServoShm.attach(name)
    rec = new_cmd()
    rec["episode"] = 1234
    pol.write_cmd(rec)
    if exit_hard:
        os._exit(0)


@pytest.mark.parametrize("exit_hard", [False, True])
def test_segment_survives_attacher_process_exit(name, exit_hard):
    with ServoShm.create(name) as servo:
        proc = mp.get_context("spawn").Process(target=_attach_write_and_exit, args=(name, exit_hard))
        proc.start()
        proc.join(60)
        assert proc.exitcode == 0
        out = new_cmd()
        assert servo.read_cmd(out) and int(out["episode"][0]) == 1234  # readable by the creator
        again = ServoShm.attach(name)  # and still attachable: the child's exit did not unlink it
        assert again.read_cmd(new_cmd())
        again.close()


def _die_mid_write(name):
    pol = ServoShm.attach(name)
    pol.write_cmd(new_cmd())
    pol._cmd._seq_word[0] = int(pol._cmd._seq_word[0]) | 1  # writer inside the critical section ...
    os._exit(0)  # ... and killed there


def test_writer_dies_with_odd_seq_read_returns_false_bounded(name):
    with ServoShm.create(name) as servo:
        proc = mp.get_context("spawn").Process(target=_die_mid_write, args=(name,))
        proc.start()
        proc.join(60)
        assert proc.exitcode == 0
        assert int(servo._cmd._seq_word[0]) & 1  # precondition: seq is odd
        out = new_cmd()
        out["episode"] = 9
        t0 = time.monotonic()
        for retries in (1, 3, 1000):
            assert servo.read_cmd(out, retries=retries) is False
        assert time.monotonic() - t0 < 1.0  # bounded retries, never blocks
        assert out["episode"][0] == 9  # untouched


def test_magic_is_written_last_so_racing_attach_sees_not_ready(name):
    with ServoShm.create(name) as servo:
        header = np.frombuffer(servo._shm.buf, dtype=HEADER_DTYPE, count=1)
        header["magic"] = 0  # what a racing attacher observes mid-create
        with pytest.raises(RuntimeError, match="not ready"):
            ServoShm.attach(name)
        header["magic"] = SHM_MAGIC
        del header
        ServoShm.attach(name).close()


def test_create_publishes_magic_after_the_rest_of_the_header(name, monkeypatch):
    seen = {}
    real_cls_init = ServoShm.__init__

    def spy(self, shm, *, owner):
        h = np.frombuffer(shm.buf, dtype=HEADER_DTYPE, count=1)
        seen["fields"] = (int(h["magic"][0]), int(h["version"][0]), int(h["cmd_size"][0]), int(h["state_size"][0]))
        del h
        real_cls_init(self, shm, owner=owner)

    monkeypatch.setattr(ServoShm, "__init__", spy)
    with ServoShm.create(name):
        pass
    assert seen["fields"][0] == SHM_MAGIC and all(seen["fields"][1:])
    # ordering itself is checked structurally: the magic store is the last header store in create()
    import inspect

    src = inspect.getsource(ServoShm.create)
    assert src.index('header["state_size"]') < src.index('header["magic"]')


def test_failed_attach_closes_mapping(name, monkeypatch):
    closed = []
    real = shm_mod._attach_untracked

    def spy(n):
        s = real(n)
        orig_close = s.close
        s.close = lambda: (closed.append(True), orig_close())[1]  # type: ignore[method-assign]
        return s

    monkeypatch.setattr(shm_mod, "_attach_untracked", spy)
    with ServoShm.create(name) as servo:
        header = np.frombuffer(servo._shm.buf, dtype=HEADER_DTYPE, count=1)
        header["version"] = 99
        with pytest.raises(RuntimeError, match="version"):
            ServoShm.attach(name)
        header["version"] = SHM_LAYOUT_VERSION
        del header
        assert closed == [True]

        def boom(self, shm, *, owner):
            raise MemoryError("view construction failed")

        monkeypatch.setattr(ServoShm, "__init__", boom)
        with pytest.raises(MemoryError):
            ServoShm.attach(name)  # a non-RuntimeError must not leak the mapping either
        assert closed == [True, True]


def test_resource_tracker_swap_is_locked_and_restored(name, monkeypatch):
    original = rt.register
    held = []
    real_shm = shared_memory.SharedMemory

    def checking(*args, **kwargs):
        held.append(shm_mod._RT_LOCK.locked() and rt.register is not original)
        return real_shm(*args, **kwargs)

    monkeypatch.setattr(shm_mod.shared_memory, "SharedMemory", checking)
    with ServoShm.create(name) as servo:  # create() also goes through `checking` but outside the lock
        held.clear()
        errors = []

        def worker():
            try:
                for _ in range(20):
                    ServoShm.attach(name).close()
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors and len(held) == 120 and all(held)
        assert rt.register is original  # never left as the no-op
        del servo
