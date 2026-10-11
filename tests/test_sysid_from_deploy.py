"""``sim.sysid.from_deploy``: a sysid run directory from deploy-session servo tick logs."""

import json

import numpy as np
import pytest

from so101_rl.safety.types import SmState
from so101_rl.servo.core import PHASE_LOOP, TICK_LOG_DTYPE
from so101_rl.sim.sysid.dataset import ExcitationDataset
from so101_rl.sim.sysid.from_deploy import HELDOUT_KIND, TRAIN_KIND, build_run_dir, main

NS = 1_000_000_000
PERIOD = NS // 30


def _log(path, t0_ns, n, *, run=slice(None)):
    """``n`` loop ticks from servo tick 0; RUN with good read + write inside ``run``, BOOT elsewhere."""
    r = np.zeros(n, dtype=TICK_LOG_DTYPE)
    r["tick"] = np.arange(n)
    r["t_ns"] = t0_ns + np.arange(n) * PERIOD
    r["phase"] = PHASE_LOOP
    r["sm_state"] = int(SmState.BOOT)
    r["sm_state"][run] = int(SmState.RUN)
    r["ok"] = 1
    r["wrote"] = 1
    r["target"] = 2047 + np.arange(n)[:, None]
    r["present"] = 2047 + np.arange(n)[:, None]
    r.tofile(path)
    return path


def test_two_sessions_become_train_and_whole_heldout_segments(tmp_path):
    a = _log(tmp_path / "a.bin", 10 * NS, 300, run=slice(20, 280))
    b = _log(tmp_path / "b.bin", 100 * NS, 200, run=slice(10, 190))
    plan = build_run_dir(tmp_path / "run", [a], [b])
    assert [s["kind"] for s in plan["segments"]] == [TRAIN_KIND, HELDOUT_KIND]
    ds = ExcitationDataset.from_run(tmp_path / "run")
    assert sorted(len(s) for s in ds.segments) == [180, 260]  # only the RUN ticks
    train, held = ds.split(heldout_kinds={HELDOUT_KIND}, heldout_fraction_by_time=0.2, seed=0)
    assert sum(len(s) for s in held.segments if s.kind == HELDOUT_KIND) == 180  # the whole second session
    assert all(s.kind == TRAIN_KIND for s in train.segments)
    seg_b = next(s for s in ds.segments if s.kind == HELDOUT_KIND)
    assert seg_b.history_targets is not None and len(seg_b.history_targets) > 0  # from log b, not log a's tail
    assert int(seg_b.history_targets[-1, 0]) == 2047 + 9
    assert json.loads((tmp_path / "run" / "plan.json").read_text())["segments"][1]["sha256"]


def test_logs_out_of_time_order_or_not_tick_logs_are_refused(tmp_path):
    a = _log(tmp_path / "a.bin", 100 * NS, 50)
    b = _log(tmp_path / "b.bin", 10 * NS, 50)
    with pytest.raises(ValueError, match="time order"):
        build_run_dir(tmp_path / "r1", [a, b])
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * (TICK_LOG_DTYPE.itemsize + 1))
    with pytest.raises(ValueError, match="TICK_LOG_DTYPE"):
        build_run_dir(tmp_path / "r2", [bad])


def test_an_existing_run_directory_is_never_reused(tmp_path):
    a = _log(tmp_path / "a.bin", 10 * NS, 50)
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        build_run_dir(tmp_path / "run", [a])


def test_cli_prints_the_fit_command(tmp_path, capsys):
    a = _log(tmp_path / "a.bin", 10 * NS, 100)
    assert main(["--out", str(tmp_path / "run"), "--train", str(a)]) == 0
    assert f"--heldout-kinds {HELDOUT_KIND}" in capsys.readouterr().out
