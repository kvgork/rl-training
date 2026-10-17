import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from so101_rl.perception.calibrate_hand_eye import farthest_point_subset, main
from so101_rl.perception.frames import load_hand_eye
from so101_rl.perception.object_table import (
    MAX_OBJECTS,
    OBJECT_DTYPE,
    OBJECT_TABLE_DTYPE,
    ObjectTableShm,
    new_object_table,
    set_object,
)


@pytest.fixture
def name():
    return f"so101_objtab_{uuid.uuid4().hex[:8]}"


def test_dtypes_pinned():
    assert OBJECT_DTYPE.itemsize == 48
    assert OBJECT_TABLE_DTYPE.itemsize % 8 == 0
    assert OBJECT_TABLE_DTYPE["objects"].shape == (MAX_OBJECTS,)


def test_write_read_roundtrip(name):
    with ObjectTableShm.create(name) as w:
        reader = ObjectTableShm.attach(name)
        try:
            out = new_object_table()
            assert not reader.read(out)  # never written
            rec = new_object_table()
            rec["t_mono_ns"][0] = 123456789
            rec["fiducial_ok"][0] = 1
            set_object(rec, 0, "die", np.array([0.2, -0.05, 0.02]), 0.9, True, 123456789)
            set_object(rec, 1, "gripper_tip", np.array([0.1, 0.0, 0.1]), 0.5, False, 123456000)
            w.write(rec)
            assert reader.read(out)
            assert int(out["n"][0]) == 2 and int(out["fiducial_ok"][0]) == 1
            o = out["objects"][0, 0]
            assert bytes(o["label"]) == b"die" and o["visible"] == 1
            assert np.allclose(o["xyz_base"], [0.2, -0.05, 0.02]) and np.isclose(o["confidence"], 0.9)
            assert out["objects"][0, 1]["visible"] == 0
            w.write(rec)
            assert reader.read(out)
        finally:
            reader.close()


def test_torn_read_rejected(name):
    with ObjectTableShm.create(name) as w:
        rec = new_object_table()
        set_object(rec, 0, "die", np.zeros(3), 1.0, True, 1)
        w.write(rec)
        out = new_object_table()
        assert w.read(out)
        block = w._block
        # writer stuck mid-write: odd seq
        block._seq_word[0] |= 1
        sentinel = new_object_table()
        sentinel["n"][0] = 77
        assert not w.read(sentinel, ) and int(sentinel["n"][0]) == 77  # out untouched on failure
        block._seq_word[0] += 1
        assert w.read(out)
        # corrupted payload: CRC mismatch
        block._raw[20] ^= 0xFF
        assert not w.read(sentinel)


def test_reader_cannot_write_and_does_not_unlink(name):
    with ObjectTableShm.create(name):
        reader = ObjectTableShm.attach(name)
        with pytest.raises(PermissionError):
            reader.write(new_object_table())
        reader.close()
        again = ObjectTableShm.attach(name)  # segment survived the reader closing
        again.close()
    with pytest.raises(FileNotFoundError):
        ObjectTableShm.attach(name)


def test_set_object_validation():
    rec = new_object_table()
    with pytest.raises(IndexError):
        set_object(rec, MAX_OBJECTS, "x", np.zeros(3), 1.0, True, 0)
    with pytest.raises(ValueError):
        set_object(rec, 0, "x" * 17, np.zeros(3), 1.0, True, 0)


def test_farthest_point_subset_spreads():
    pts = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    chosen = set(farthest_point_subset(pts, 4).tolist())
    assert {0, 2, 3, 4} == chosen
    with pytest.raises(ValueError):
        farthest_point_subset(pts, 9)


def test_dry_run_writes_yaml_with_low_residual(tmp_path):
    out = tmp_path / "hand_eye.yaml"
    assert main(["--dry-run", "--out", str(out)]) == 0
    he = load_hand_eye(out)
    assert he.residual_mm < 1.0 and he.n_poses >= 10


def test_dry_run_cli_subprocess_default_path(tmp_path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(
        [sys.executable, "-m", "so101_rl.perception.calibrate_hand_eye", "--dry-run"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert load_hand_eye(Path(tmp_path) / "runs/hand-eye-dry/hand_eye.yaml").residual_mm < 1.0


def test_mode_is_required_and_execute_never_moves(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    # --execute refuses before the camera or the servo is touched (tests/test_hand_eye_execute.py has the rest)
    assert main(["--execute", "--policy", "/nonexistent/policy.pt", "--wait-servo-s", "0"]) == 2
