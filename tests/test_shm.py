import multiprocessing as mp
import time
import uuid

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
