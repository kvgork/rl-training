"""Task V.1: live viewer payload, selectors, publisher, headless refusal, overhead (plan Unit SUV)."""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import so101_rl
from so101_rl.viz import live_viewer as lv

SRC = str(Path(so101_rl.__file__).resolve().parents[1])
gpu = pytest.mark.gpu


def _env(**drop):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _fake_sim(n=16, nq=13, nv=12, device="cpu"):
    qpos = torch.arange(n, dtype=torch.float32, device=device)[:, None] * 100 + torch.arange(nq, device=device)
    qvel = -qpos[:, :nv].clone()
    return SimpleNamespace(qpos=qpos, qvel=qvel)


def test_viewer_payload_roundtrip():
    from multiprocessing import shared_memory

    from so101_rl.servo.shm import SeqlockBlock

    k, nq, nv = 3, 13, 12
    dt = lv.payload_dtype(k, nq, nv)
    shm = shared_memory.SharedMemory(create=True, size=dt.itemsize)
    reader_shm = None
    try:
        w = SeqlockBlock(shm.buf, dt)
        reader_shm = lv._attach_untracked(shm.name)
        r = SeqlockBlock(reader_shm.buf, dt)
        out = np.zeros(1, dtype=dt)
        assert not r.read(out)  # never written
        rec = np.zeros(1, dtype=dt)
        rec["env_step"], rec["dr_stage"], rec["n_slots"] = 1234, 2, k
        rec["slots"][0] = np.random.default_rng(0).normal(size=(k, lv.SlotLayout(nq, nv).width))
        w.write(rec)
        assert r.read(out)
        assert int(out["env_step"][0]) == 1234 and int(out["dr_stage"][0]) == 2 and int(out["n_slots"][0]) == k
        np.testing.assert_array_equal(out["slots"], rec["slots"])
        assert int(out["seq"][0]) % 2 == 0
        w.release()
        r.release()
    finally:
        if reader_shm is not None:
            reader_shm.close()
        shm.close()
        shm.unlink()


def test_slot_layout_is_dense_and_disjoint():
    L = lv.SlotLayout(13, 12)
    cols = [L.WORLD, L.EP_STEP, L.SUCCESS, L.RET, L.GHOST_VALID, *range(*L.GOAL.indices(99)), *range(*L.GHOST.indices(99))]
    cols += list(range(*L.qpos.indices(999))) + list(range(*L.qvel.indices(999)))
    assert sorted(cols) == list(range(L.width))


def test_selector_parsing():
    assert lv.parse_selector("0") == ("fixed", 0)
    assert lv.parse_selector(7) == ("fixed", 7)
    assert lv.parse_selector("Best") == ("best", 0)
    for bad in ("median", "-1", -3, True, ""):
        with pytest.raises((ValueError, TypeError)):
            lv.parse_selector(bad)
    with pytest.raises(ValueError, match="outside"):
        lv.WorldPicker(["0", "99"], 8, "cpu")


def test_selector_best_worst():
    ret = torch.tensor([-3.0, -1.0, -9.0, -2.0])
    p = lv.WorldPicker(["0", "best", "worst", "3"], 4, "cpu")
    assert p.update(ret).tolist() == [0, 1, 2, 3]
    # mid-episode: returns change but nobody's episode restarted -> the picks are sticky
    step = torch.tensor([5, 5, 5, 5])
    p2 = lv.WorldPicker(["best", "worst"], 4, "cpu")
    assert p2.update(ret, step).tolist() == [1, 2]
    ret2 = torch.tensor([-0.5, -4.0, -1.0, -2.0])
    assert p2.update(ret2, step + 1).tolist() == [1, 2]
    # the shown "best" world (1) restarts its episode: only that slot is re-picked
    step3 = torch.tensor([7, 0, 7, 7])
    assert p2.update(ret2, step3).tolist() == [0, 2]
    # now the shown "worst" world (2) restarts
    ret3 = torch.tensor([-5.0, -4.0, -0.1, -2.0])
    assert p2.update(ret3, torch.tensor([8, 1, 0, 8])).tolist() == [0, 0]


def test_selector_repicks_periodically_without_episode_step(monkeypatch):
    monkeypatch.setattr(lv, "REPICK_TICKS", 3)
    p = lv.WorldPicker(["best"], 3, "cpu")
    assert p.update(torch.tensor([0.0, 1.0, 0.0])).tolist() == [1]
    flip = torch.tensor([5.0, 1.0, 0.0])
    assert p.update(flip).tolist() == [1]
    assert p.update(flip).tolist() == [1]
    assert p.update(flip).tolist() == [0]


def test_build_viewer_model_slots():
    m, adr = lv.build_viewer_model(3)
    assert m.nq == 3 * 13 and m.nv == 3 * 12 and len(adr) == 3
    assert [a["qpos"] for a in adr] == [0, 13, 26] and [a["dof"] for a in adr] == [0, 12, 24]
    # slot s is offset along y by s * spacing
    import mujoco

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ys = [d.xpos[a["base"], 1] for a in adr]
    assert ys == pytest.approx([0.0, lv.SLOT_SPACING_M, 2 * lv.SLOT_SPACING_M])


def _read(pub, dtype):
    from so101_rl.servo.shm import SeqlockBlock

    out = np.zeros(1, dtype=dtype)
    ok = SeqlockBlock(pub._shm.buf, dtype).read(out)
    return ok, out


def test_publisher_publishes_selected_rows_only(tmp_path):
    sim = _fake_sim(16)
    ret = torch.arange(16, dtype=torch.float32)  # best = 15, worst = 0
    success = torch.zeros(16, dtype=torch.bool)
    success[3] = True
    ctx = lv.VizContext(dr_stage=2, episode_step=torch.arange(16), goal=torch.ones(16, 3) * 0.5)
    pub = lv.LivePublisher(sim, ["3", "best", "worst"], 1000.0, tmp_path / "viz", task="reach", spawn_viewer=False)
    try:
        assert (tmp_path / "viz" / lv.DESCRIPTOR).exists()
        assert pub.tick(42, success, ret, ctx, force=True)
        ok, out = _read(pub, pub.dtype)
        assert ok and int(out["env_step"][0]) == 42 and int(out["dr_stage"][0]) == 2
        L = pub.layout
        rows = out["slots"][0]
        assert rows[:, L.WORLD].tolist() == [3.0, 15.0, 0.0]
        np.testing.assert_array_equal(rows[:, L.qpos], sim.qpos[[3, 15, 0]].numpy())
        np.testing.assert_array_equal(rows[:, L.qvel], sim.qvel[[3, 15, 0]].numpy())
        assert rows[:, L.SUCCESS].tolist() == [1.0, 0.0, 0.0]
        assert rows[:, L.EP_STEP].tolist() == [3.0, 15.0, 0.0]
        assert rows[:, L.GOAL].tolist() == [[0.5] * 3] * 3
    finally:
        pub.close()
    assert not (tmp_path / "viz" / lv.DESCRIPTOR).exists()


def test_publisher_pauses_without_a_viewer(tmp_path, monkeypatch):
    sim = _fake_sim(4)
    pub = lv.LivePublisher(sim, ["0"], 1000.0, tmp_path / "viz", task="reach", spawn_viewer=False)
    try:
        args = (1, torch.zeros(4, dtype=torch.bool), torch.zeros(4), lv.VizContext())
        monkeypatch.setattr(pub, "_grace_until", 0.0)  # the viewer never started / was closed
        assert not pub.tick(*args, force=True) and pub.published == 0
        (tmp_path / "viz" / lv.VIEWER_PID).write_text(str(os.getpid()))  # a live viewer pid
        assert pub.tick(*args, force=True) and pub.published == 1
        (tmp_path / "viz" / lv.VIEWER_PID).write_text("999999999")  # window closed: stale pid
        assert not pub.tick(*args, force=True) and pub.published == 1
    finally:
        pub.close()


def test_publisher_rate_limit(tmp_path):
    sim = _fake_sim(4)
    pub = lv.LivePublisher(sim, ["0"], 5.0, tmp_path / "viz", task="reach", spawn_viewer=False)
    try:
        args = (1, torch.zeros(4, dtype=torch.bool), torch.zeros(4), lv.VizContext())
        assert pub.tick(*args)
        assert not any(pub.tick(*args) for _ in range(50))  # inside the 0.2 s period
    finally:
        pub.close()


def test_viewer_absent_display_is_clean_error(tmp_path):
    for cmd in (["attach", "--run", str(tmp_path / "run"), "--timeout", "1"], ["view", "--descriptor", str(tmp_path / "x.json")]):
        p = subprocess.run([sys.executable, "-m", "so101_rl.viz.live_viewer", *cmd], env=_env(), capture_output=True, text=True, timeout=60, check=False)
        assert p.returncode == 2, p.stderr
        assert "needs a display" in p.stderr and "Traceback" not in p.stderr
    assert not (tmp_path / "run" / "viz" / lv.ATTACH_REQUEST).exists()  # refused before touching the run


def test_require_display_accepts_either_variable():
    lv.require_display({"DISPLAY": ":0"})
    lv.require_display({"WAYLAND_DISPLAY": "wayland-0"})
    with pytest.raises(lv.NoDisplayError):
        lv.require_display({})


def test_descriptor_is_valid_json(tmp_path):
    pub = lv.LivePublisher(_fake_sim(4), ["0", "best"], 30.0, tmp_path / "viz", task="die_to_zone", spawn_viewer=False)
    try:
        d = json.loads((tmp_path / "viz" / lv.DESCRIPTOR).read_text())
        assert d["slots"] == 2 and d["nq"] == 13 and d["nv"] == 12 and d["task"] == "die_to_zone"
        assert d["trainer_pid"] == os.getpid() and d["shm"].startswith("so101viz-")
    finally:
        pub.close()


@gpu
def test_publisher_cuda_no_host_sync(tmp_path):
    """The training-side call path must not synchronise the device (pinned async copy + event poll)."""
    sim = _fake_sim(256, device="cuda")
    ret = torch.randn(256, device="cuda")
    succ = ret > 0
    ctx = lv.VizContext(episode_step=torch.zeros(256, dtype=torch.long, device="cuda"), goal=torch.zeros(256, 3, device="cuda"))
    pub = lv.LivePublisher(sim, ["0", "best", "worst"], 1000.0, tmp_path / "viz", task="reach", spawn_viewer=False)
    try:
        pub.tick(0, succ, ret, ctx, force=True)  # warm-up: allocations, first kernel loads
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for i in range(20):
                pub.tick(i, succ, ret, ctx, force=True)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        pub.tick(99, succ, ret, ctx, force=True)
        time.sleep(0.05)
        pub._flush()
        ok, out = _read(pub, pub.dtype)
        assert ok and pub.published >= 1
        L = pub.layout
        w = out["slots"][0][:, L.WORLD].astype(int)
        np.testing.assert_allclose(out["slots"][0][:, L.qpos], sim.qpos[w].cpu().numpy())
    finally:
        pub.close()


@gpu
def test_viewer_overhead_under_5pct(tmp_path):
    """Training steps/s with vs without the publisher (no window): the slowdown must stay under 5 %."""
    from so101_rl.viz.demo import RandomWalkEnv, random_policy

    env = RandomWalkEnv(256)
    obs = env._obs()

    def run(pub, seconds=2.5):
        nonlocal obs
        ctx = lv.VizContext()
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < seconds:
            obs = env.step(random_policy(obs))
            n += 1
            if pub is not None:
                ctx.goal, ctx.episode_step = env.goal, env.ep_step
                pub.tick(n, env.success, env.ep_return, ctx)
        torch.cuda.synchronize()
        return n / (time.perf_counter() - t0)

    run(None, 1.5)  # warm-up
    pub = lv.LivePublisher(env.sim, ["0", "best", "worst"], 30.0, tmp_path / "viz", task="reach", spawn_viewer=False)
    pub._grace_until = float("inf")  # publisher-only: act as if a viewer were attached
    try:
        base, with_pub = [], []
        for _ in range(5):
            base.append(run(None))
            with_pub.append(run(pub))
    finally:
        pub.close()
        env.close()
    overhead = 100.0 * (statistics.median(base) - statistics.median(with_pub)) / statistics.median(base)
    print(f"viewer publisher overhead: {overhead:+.2f} % (base {statistics.median(base):.1f}, pub {statistics.median(with_pub):.1f} steps/s)")
    assert pub.published > 5
    assert overhead < 5.0
