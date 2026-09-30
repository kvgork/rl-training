"""Task 6.2: every gate in check_episode is shown to fail on a deliberately broken input (L12)."""

import os
import shutil
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_episode_logger import N, make_rec

from so101_rl.logging.checks import check_episode, check_run, disk_budget_gb, prune
from so101_rl.logging.episode_logger import EpisodeLogger

MAX_DELTA = 45
CFG = SimpleNamespace(action=SimpleNamespace(max_delta_ticks=np.full(6, MAX_DELTA, np.int32)))


def good_episode(root, *, applied=None, run="r", outcome="failure"):
    lg = EpisodeLogger(root, run)
    lg.start_episode({})
    for i in range(N):
        lg.log_step(make_rec(i, full=True, applied=applied))
    return lg.end_episode(outcome)


@pytest.fixture
def good(tmp_path):
    return good_episode(tmp_path)


def test_good_file_passes(good):
    rep = check_episode(good, CFG)
    assert rep.ok, rep.errors
    assert rep.n_steps == N and rep.errors == [] and rep.path == good
    assert check_episode(good).ok  # cfg optional


def _errors(path, cfg=CFG):
    rep = check_episode(path, cfg)
    assert not rep.ok
    return " | ".join(rep.errors)


def test_fails_on_tmp_file(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    lg.start_episode({})
    lg.log_step(make_rec(0))
    lg.close()
    tmp = next((tmp_path / "r" / "episodes").glob("*.tmp"))
    assert "temporary" in _errors(tmp)


def test_fails_on_missing_file(tmp_path):
    assert "does not exist" in _errors(tmp_path / "ep_000009.h5")


def test_fails_on_mismatched_lengths(good):
    with h5py.File(good, "r+") as f:
        f["reward"].resize(N - 1, axis=0)
    assert "mismatched lengths" in _errors(good)


def test_fails_on_nan_reward(good):
    with h5py.File(good, "r+") as f:
        f["reward"][2] = np.nan
    assert "reward" in _errors(good) and "NaN" in _errors(good)


def test_fails_on_nan_observation(good):
    with h5py.File(good, "r+") as f:
        f["obs/observation"][1, 4] = np.inf
    assert "obs/observation" in _errors(good)


def test_fails_on_applied_delta_above_max(tmp_path):
    p = good_episode(tmp_path, applied=np.full(6, MAX_DELTA + 1, np.float32))
    assert "max_delta_ticks" in _errors(p)
    assert check_episode(p).ok  # gate is only armed with a cfg


def test_applied_delta_at_max_passes(tmp_path):
    p = good_episode(tmp_path, applied=np.full(6, -MAX_DELTA, np.float32))
    assert check_episode(p, CFG).ok


def test_fails_on_target_4096(good):
    with h5py.File(good, "r+") as f:
        f["target_ticks"][3, 2] = 4096
    assert "target_ticks" in _errors(good)


def test_fails_on_negative_target(good):
    with h5py.File(good, "r+") as f:
        f["target_ticks"][0, 0] = -1
    assert "target_ticks" in _errors(good)


def test_target_4095_passes(good):
    with h5py.File(good, "r+") as f:
        f["target_ticks"][3, 2] = 4095
    assert check_episode(good, CFG).ok


@pytest.mark.parametrize("attr", ["run_id", "episode", "outcome", "config_dump", "git_sha", "created"])
def test_fails_on_missing_attr(good, attr):
    with h5py.File(good, "r+") as f:
        del f.attrs[attr]
    assert f"missing attr: {attr}" in _errors(good)


def test_fails_on_frame_count_mismatch(good):
    with h5py.File(good, "r+") as f:
        f["frames_small"].resize(N - 2, axis=0)
    assert "frame count mismatch" in _errors(good)


def test_fails_on_full_frame_count_mismatch(good):
    with h5py.File(good, "r+") as f:
        # frames_full is fixed-size; rebuild it with the wrong length
        del f["frames_full"]
        f.create_dataset("frames_full", data=np.zeros((N - 1, 480, 640, 3), np.uint8))
    assert "frames_full" in _errors(good)


def test_fails_on_wrong_dtype(good):
    with h5py.File(good, "r+") as f:
        data = f["target_ticks"][()].astype(np.float32)
        del f["target_ticks"]
        f.create_dataset("target_ticks", data=data)
    assert "target_ticks: dtype" in _errors(good)


def test_fails_on_missing_dataset(good):
    with h5py.File(good, "r+") as f:
        del f["present_ticks"]
    assert "missing dataset: present_ticks" in _errors(good)


def test_fails_on_truncated_file(good):
    size = good.stat().st_size
    with open(good, "r+b") as fh:
        fh.truncate(size // 2)
    assert "unreadable" in _errors(good) or "truncated" in _errors(good)


def test_fails_on_garbage_file(tmp_path):
    p = tmp_path / "ep_000000.h5"
    p.write_bytes(b"not hdf5 at all")
    assert "truncated" in _errors(p)


def test_fails_on_zero_steps(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    lg.start_episode({})
    p = lg.end_episode("failure")
    assert "zero steps" in _errors(p)


def test_check_run_reports_each_file(tmp_path):
    a = good_episode(tmp_path, run="r1")
    good_episode(tmp_path, run="r1")
    lg = EpisodeLogger(tmp_path, "r2")
    lg.start_episode({})
    lg.close()  # leaves a .tmp
    with h5py.File(a, "r+") as f:
        f["reward"][0] = np.nan
    reps = check_run(tmp_path)
    assert len(reps) == 3
    assert sorted(r.ok for r in reps) == [False, False, True]
    assert check_run(tmp_path / "r1")[1].ok


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_prune_dry_run_lists_but_does_not_delete(tmp_path):
    old = good_episode(tmp_path, run="r")
    new = good_episode(tmp_path, run="r")
    _age(old, 30)
    assert prune(tmp_path, 14) == [old]
    assert old.exists() and new.exists()
    assert prune(tmp_path, 14, dry_run=True) == [old]
    assert old.exists()


def test_prune_deletes_only_when_asked_and_only_episode_files(tmp_path):
    old = good_episode(tmp_path, run="r")
    new = good_episode(tmp_path, run="r")
    other = old.parent.parent / "notes.txt"
    other.write_text("keep")
    for p in (old, other):
        _age(p, 30)
    assert prune(tmp_path, 14, dry_run=False) == [old]
    assert not old.exists() and new.exists() and other.exists()


def test_prune_includes_stale_tmp(tmp_path):
    lg = EpisodeLogger(tmp_path, "r")
    lg.start_episode({})
    lg.close()
    tmp = next((tmp_path / "r" / "episodes").glob("*.tmp"))
    _age(tmp, 20)
    assert prune(tmp_path, 14) == [tmp]


def test_disk_budget_gb(tmp_path):
    assert disk_budget_gb(tmp_path) == 0.0
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"\0" * 3_000_000)
    (tmp_path / "y.bin").write_bytes(b"\0" * 2_000_000)
    assert disk_budget_gb(tmp_path) == pytest.approx(0.005)
    assert disk_budget_gb(tmp_path / "missing") == 0.0


def test_copy_of_good_file_still_passes(good, tmp_path):
    # the gates are content-based, not path-based
    dst = tmp_path / "copy" / "ep_000042.h5"
    dst.parent.mkdir()
    shutil.copy(good, dst)
    assert check_episode(dst, CFG).ok
