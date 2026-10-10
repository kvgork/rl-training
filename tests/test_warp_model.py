"""FS2 WarpSim (sim-first track S0.3). GPU tests run with `pixi run test-gpu` (default env, CUDA)."""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from so101_rl.config import JOINT_NAMES, SimWarpConfig

gpu = pytest.mark.gpu


@pytest.fixture(scope="module")
def scene():
    from so101_rl.sim.model import load_model

    return load_model()


def _sim(scene, cfg, nworld, **kw):
    from so101_rl.sim.warp_model import WarpSim

    return WarpSim(scene[0], cfg, nworld, **kw)


def _to_key(ws, model, key="rest"):
    import torch

    k = model.key(key)
    mask = torch.ones(ws.sizes.nworld, dtype=torch.bool, device=ws.device)
    ws.reset_worlds(
        mask,
        torch.as_tensor(k.qpos, device=ws.device),
        torch.zeros(model.nv, device=ws.device),
        torch.as_tensor(k.ctrl, device=ws.device),
    )


def _random_ctrl(ws, model, gen):
    import torch

    lo = torch.as_tensor(model.actuator_ctrlrange[:, 0], dtype=torch.float32, device=ws.device)
    hi = torch.as_tensor(model.actuator_ctrlrange[:, 1], dtype=torch.float32, device=ws.device)
    ws.ctrl.copy_(lo + (hi - lo) * torch.rand(ws.ctrl.shape, device=ws.device, generator=gen))


@dataclasses.dataclass
class _Params:  # duck-typed stand-in for FS7 BatchedSimParams (SU2)
    kp: object
    kv: object
    damping: object
    frictionloss: object
    armature: object
    forcerange: object
    die_mass: object
    die_friction: object
    table_friction: object


def _mjcf_params(model, n, device):
    import torch

    act = [model.actuator(j).id for j in JOINT_NAMES]
    dof = [int(model.jnt_dofadr[model.joint(j).id]) for j in JOINT_NAMES]

    def rows(v):
        return torch.as_tensor(np.tile(np.asarray(v, dtype=np.float32), (n, 1)), device=device)

    def one(v):
        return torch.full((n,), float(v), device=device)

    return _Params(
        kp=rows(model.actuator_gainprm[act, 0]),
        kv=rows(-model.actuator_biasprm[act, 2]),
        damping=rows(model.dof_damping[dof]),
        frictionloss=rows(model.dof_frictionloss[dof]),
        armature=rows(model.dof_armature[dof]),
        forcerange=rows(model.actuator_forcerange[act, 1]),
        die_mass=one(model.body_mass[model.body("die").id]),
        die_friction=one(model.geom_friction[model.geom("die_geom").id, 0]),
        table_friction=one(model.geom_friction[model.geom("table").id, 0]),
    )


# ---------------------------------------------------------------- CPU-only


def test_import_sim_does_not_load_warp():
    code = (
        "import sys, so101_rl.sim; "
        "bad = [m for m in ('warp', 'mujoco_warp', 'mujoco') if m in sys.modules]; "
        "assert not bad, bad; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_buffer_mask_excludes_solver_flags_only():
    import mujoco_warp as mjw

    from so101_rl.sim.warp_model import BUFFER_OVERFLOW_MASK

    for flag in (mjw.OverflowType.NEFC, mjw.OverflowType.BROADPHASE, mjw.OverflowType.NARROWPHASE):
        assert BUFFER_OVERFLOW_MASK & int(flag)
    for flag in (mjw.OverflowType.ITERATIONS, mjw.OverflowType.LS_ITERATIONS):
        assert not BUFFER_OVERFLOW_MASK & int(flag)


def test_warn_policy_refused_without_gpu(scene):
    from so101_rl.sim.warp_model import WarpSim

    bad = dataclasses.replace(SimWarpConfig(), overflow_policy="warn")
    with pytest.raises(ValueError, match="overflow_policy"):
        WarpSim(scene[0], bad, 1)


# ---------------------------------------------------------------- GPU


@gpu
def test_sizes_from_config(scene, cfg):
    ws = _sim(scene, cfg.sim_warp, 8)
    assert ws.sizes.nworld == 8
    assert ws.sizes.njmax == cfg.sim_warp.njmax == ws.d.njmax
    assert ws.sizes.nconmax == cfg.sim_warp.nconmax_per_world
    assert ws.sizes.naconmax == cfg.sim_warp.nconmax_per_world * 8 == ws.d.naconmax
    assert ws.vram_bytes > 0
    ws.close()


@gpu
def test_qpos_zero_copy(scene, cfg):
    import torch
    import warp as wp

    ws = _sim(scene, cfg.sim_warp, 4)
    model = scene[0]
    assert ws.qpos.shape == (4, model.nq) and ws.qpos.dtype == torch.float32 and ws.qpos.is_cuda
    assert ws.qpos.data_ptr() == ws.d.qpos.ptr
    assert ws.ctrl.data_ptr() == ws.d.ctrl.ptr
    ws.qpos[2, 0] = 0.25  # torch write -> visible to warp
    torch.cuda.synchronize()
    assert ws.d.qpos.numpy()[2, 0] == pytest.approx(0.25)
    ws.d.qvel.fill_(1.5)  # warp write -> visible to torch
    wp.synchronize()
    assert float(ws.qvel[1, 3]) == pytest.approx(1.5)
    ws.close()


def _cpu_vs_warp(scene, cfg, nworld, nsteps, ctrl_fn, **kw):
    import mujoco
    import torch

    model = scene[0]
    data = mujoco.MjData(model)
    # home, not rest: the measured rest (plan §10 2026-10-10) is folded onto the joint stops, and a sinusoid around it
    # would drive the arm into its limits instead of testing free-space stepping
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    ws = _sim(scene, cfg.sim_warp, nworld, **kw)
    _to_key(ws, model, "home")
    err = np.zeros((nsteps, model.nq))
    for t in range(nsteps):
        c = ctrl_fn(t)
        data.ctrl[:] = c
        mujoco.mj_step(model, data)
        ws.ctrl.copy_(torch.as_tensor(c, dtype=torch.float32, device=ws.device).expand_as(ws.ctrl))
        ws.step(1)
        q = ws.qpos.double().cpu().numpy()
        err[t] = np.abs(q - data.qpos).max(axis=0)
    ws.close()
    return err


def _sinusoid(model):
    home = np.array(model.key("home").ctrl)
    k = np.arange(1, 7) / 3

    def c(t):
        return home + 0.3 * np.sin(2 * np.pi * 0.5 * t * model.opt.timestep * k)

    return c


@gpu
def test_step_matches_cpu_300_steps(scene, cfg):
    """S0 gate: CPU float64 vs MJWarp float32, sinusoid around home, arm joints within 1e-6 rad over 300 steps."""
    err = _cpu_vs_warp(scene, cfg, 4, 300, _sinusoid(scene[0]))
    arm = err[:, :6].max()
    print(f"max |dqpos| arm {arm:.3e} rad, die {err[:, 6:].max():.3e}")
    assert arm <= 1e-6


@gpu
def test_graph_and_eager_paths_agree(scene, cfg):
    import torch

    model = scene[0]
    out = []
    for capture in (True, False):
        ws = _sim(scene, cfg.sim_warp, 4, capture_graph=capture)
        _to_key(ws, model, "home")
        gen = torch.Generator(device=ws.device).manual_seed(3)
        for _ in range(6):
            _random_ctrl(ws, model, gen)
            ws.step(17)
        out.append(ws.qpos.cpu().numpy().copy())
        ws.close()
    np.testing.assert_allclose(out[0], out[1], atol=1e-6)


@gpu
def test_overflow_detector_fails(scene, cfg):
    """The gate shown to fail: njmax 16 (below the scene's need) raises within 100 steps of random ctrl."""
    import torch

    from so101_rl.sim.warp_model import ContactBufferOverflow

    small = dataclasses.replace(cfg.sim_warp, njmax=16)
    ws = _sim(scene, small, 64)
    model = scene[0]
    _to_key(ws, model, "home")
    gen = torch.Generator(device=ws.device).manual_seed(0)
    with pytest.raises(ContactBufferOverflow, match=r"world \d+: (NEFC|nefc \d+ > njmax)"):
        for _ in range(10):
            _random_ctrl(ws, model, gen)
            ws.step(10)
            ws.check_overflow()


@gpu
def test_overflow_detector_quiet_at_256(scene, cfg):
    import torch

    ws = _sim(scene, cfg.sim_warp, 256)
    model = scene[0]
    _to_key(ws, model, "home")
    gen = torch.Generator(device=ws.device).manual_seed(1)
    steps = 0
    for k in range(118):  # 118 x 17 = 2006 mj_steps
        if k % 3 == 0:
            _random_ctrl(ws, model, gen)
        ws.step(17)
        steps += 17
        if k % cfg.sim_warp.overflow_check_every_steps == 0:
            ws.check_overflow()
    assert torch.isfinite(ws.qpos).all()
    ws.close()  # final check
    assert steps >= 2000
    assert 0 < ws.max_nefc_seen <= cfg.sim_warp.njmax
    print(f"max nefc {ws.max_nefc_seen}/{ws.sizes.njmax}, max nacon {ws.max_nacon_seen}/{ws.sizes.naconmax}")


@gpu
def test_overflow_between_checks_survives_reset(scene, cfg):
    """MJWarp's sticky flag is folded before a reset clears it, so a reset cannot hide an overflow."""
    import mujoco_warp as mjw

    from so101_rl.sim.warp_model import ContactBufferOverflow

    ws = _sim(scene, cfg.sim_warp, 8)  # njmax 256: no real overflow here
    model = scene[0]
    _to_key(ws, model, "rest")
    ws.step(17)
    ws.check_overflow()  # clean
    ws._overflow[3] = int(mjw.OverflowType.NEFC)  # an overflow MJWarp flagged on some step between two checks
    _to_key(ws, model, "rest")  # resets every world: clears MJWarp's own flags ...
    assert int(ws.d.overflow.numpy().max()) == 0
    with pytest.raises(ContactBufferOverflow, match=r"world 3: NEFC"):  # ... but the wrapper still raises
        ws.check_overflow()


@gpu
def test_close_checks_overflow(scene, cfg):
    from so101_rl.sim.warp_model import ContactBufferOverflow

    ws = _sim(scene, dataclasses.replace(cfg.sim_warp, njmax=16), 8)
    _to_key(ws, scene[0], "rest")
    ws.step(17)
    with pytest.raises(ContactBufferOverflow):
        ws.close()
    with pytest.raises(RuntimeError, match="closed"):
        ws.step(1)
    ws.close()  # idempotent


@gpu
def test_reset_worlds_touches_only_masked(scene, cfg):
    import torch

    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 3)
    _to_key(ws, model, "rest")
    ws.step(17)
    before, qvel_before, ws_before = ws.qpos.clone(), ws.qvel.clone(), ws._qacc_ws.clone()
    assert ws.qvel[1].abs().sum() > 0
    mask = torch.tensor([False, True, False], device=ws.device)
    home = torch.as_tensor(model.key("home").qpos, device=ws.device, dtype=torch.float32)
    ws.reset_worlds(mask, home.expand(3, -1), torch.zeros(3, model.nv, device=ws.device), ws.ctrl.clone())
    assert torch.equal(ws.qpos[0], before[0]) and torch.equal(ws.qpos[2], before[2])
    assert torch.equal(ws.qvel[0], qvel_before[0]) and torch.equal(ws.qvel[2], qvel_before[2])
    assert torch.equal(ws._qacc_ws[0], ws_before[0])
    assert torch.allclose(ws.qpos[1], home) and not ws.qvel[1].any() and not ws._qacc_ws[1].any()
    # forward ran: the reset world's tip follows its new qpos
    tip = ws.site_xpos("gripper_tip")
    assert not torch.allclose(tip[1], tip[0], atol=1e-3)
    assert float(ws._time[1]) == 0.0 and float(ws._time[0]) > 0.0
    ws.close()


@gpu
def test_site_body_views_match_cpu(scene, cfg):
    import mujoco

    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 2)
    _to_key(ws, model, "home")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(ws.site_xpos("gripper_tip")[1].cpu().numpy(), data.site("gripper_tip").xpos, atol=1e-5)
    np.testing.assert_allclose(ws.body_xpos("die")[0].cpu().numpy(), data.body("die").xpos, atol=1e-5)
    np.testing.assert_allclose(ws.body_xquat("die")[0].cpu().numpy(), data.body("die").xquat, atol=1e-5)
    copy = mujoco.MjData(model)
    ws.copy_world_to(1, copy)
    np.testing.assert_allclose(copy.qpos, data.qpos, atol=1e-6)
    ws.close()


@gpu
def test_contact_count_die_on_table_matches_cpu(scene, cfg):
    import mujoco
    import torch

    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 4)
    _to_key(ws, model, "rest")
    lifted = ws.qpos.clone()
    lifted[2, 8] += 0.05  # world 2: die 5 cm above the table (falls ~2 cm in 20 steps, no contact yet)
    ws.reset_worlds(torch.tensor([False, False, True, False], device=ws.device), lifted, ws.qvel.clone(), ws.ctrl.clone())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    die, table = model.geom("die_geom").id, model.geom("table").id
    for _ in range(20):
        data.ctrl[:] = model.key("rest").ctrl
        mujoco.mj_step(model, data)
        ws.step(1)
    cpu = sum(1 for c in data.contact[: data.ncon] if {int(c.geom1), int(c.geom2)} == {die, table})
    n = ws.contact_count([die], [table]).cpu().numpy()
    assert cpu >= 1
    np.testing.assert_array_equal(n, [cpu, cpu, 0, cpu])  # per-world attribution, not a global count
    np.testing.assert_array_equal(ws.contact_count([table], [die]).cpu().numpy(), n)  # order-independent
    assert ws.contact_count([die], [model.geom("jaw_fixed_tip").id]).sum() == 0
    ws.close()


@gpu
def test_set_world_params_per_world(scene, cfg):
    """kp 998 vs 300 in two worlds: the stiffer world tracks a shoulder_pan step faster; others unchanged."""
    import torch

    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 2)
    ptrs = {name: getattr(ws.m, name).ptr for name in ("body_mass", "body_subtreemass", "dof_invweight0", "actuator_gainprm")}
    p = _mjcf_params(model, 2, ws.device)
    p.kp[1, 0] = 300.0
    ws.set_world_params(p)
    assert {name: getattr(ws.m, name).ptr for name in ptrs} == ptrs  # graphs stay valid (no rebinding)
    g = ws._p["actuator_gainprm"]
    assert float(g[0, 0, 0]) == pytest.approx(998.22, rel=1e-5) and float(g[1, 0, 0]) == pytest.approx(300.0)
    assert float(ws._p["actuator_biasprm"][1, 0, 1]) == pytest.approx(-300.0)
    _to_key(ws, model, "home")
    target = ws.ctrl.clone()
    target[:, 0] += 0.3
    ws.ctrl.copy_(target)
    ws.step(51)  # one policy step, 0.1 s
    moved = (ws.qpos[:, 0] - torch.as_tensor(model.key("home").qpos[0], device=ws.device)).cpu().numpy()
    assert moved[0] > moved[1] > 0.0, moved
    ws.close()


@gpu
def test_set_world_params_die_mass_runs_set_const(scene, cfg):
    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 2)
    p = _mjcf_params(model, 2, ws.device)
    die = model.body("die").id
    p.die_mass[1] = 0.04
    ws.set_world_params(p)
    assert float(ws._p["body_mass"][1, die]) == pytest.approx(0.04)
    assert float(ws._p["body_subtreemass"][1, die]) == pytest.approx(0.04, rel=1e-5)  # derived by set_const
    assert float(ws._p["body_subtreemass"][0, die]) == pytest.approx(model.body_mass[die], rel=1e-5)
    ratio = ws._p["body_inertia"][1, die] / ws._p["body_inertia"][0, die]
    np.testing.assert_allclose(ratio.cpu().numpy(), 0.04 / model.body_mass[die], rtol=1e-5)
    with pytest.raises(ValueError, match="shape"):
        p.kp = p.kp[:1]
        ws.set_world_params(p)
    ws.close()


# ---------------------------------------------------------------- SU0 review fixes


@gpu
def test_params_changed_after_capture_are_replayed_per_world(scene, cfg):
    """Regression: set_world_params after the first CAPTURED step must reach the replayed graph (in-place writes,
    no rebinding), per world, and match an eager run with the same parameters set up front."""
    import torch

    model = scene[0]

    def run(capture: bool, change_after_capture: bool):
        ws = _sim(scene, cfg.sim_warp, 3, capture_graph=capture)
        p = _mjcf_params(model, 3, ws.device)
        if not change_after_capture:
            p.kp[1, 0] = 300.0
            p.die_mass[2] = 0.04
        ws.set_world_params(p)
        _to_key(ws, model, "home")
        ws.step(51)  # first call: captures the graph (capture=True)
        graph = ws._graphs.get(("step", 51))
        if change_after_capture:
            p.kp[1, 0] = 300.0
            p.die_mass[2] = 0.04
            ws.set_world_params(p)
        _to_key(ws, model, "home")
        target = ws.ctrl.clone()
        target[:, 0] += 0.3
        ws.ctrl.copy_(target)
        ws.step(51)
        if capture:
            assert graph is not None and ws._graphs[("step", 51)] is graph  # replayed, not re-captured
        q = ws.qpos.clone()
        mi = ws._meaninertia.clone()
        ws.close()
        return q, mi

    q_replay, mi = run(True, True)
    q_eager, _ = run(False, False)
    np.testing.assert_allclose(q_replay.cpu().numpy(), q_eager.cpu().numpy(), atol=1e-6)
    home0 = float(model.key("home").qpos[0])
    moved = (q_replay[:, 0] - home0).cpu().numpy()
    assert moved[0] > moved[1] > 0.0, moved  # world 1 got kp 300 after capture; world 0 untouched
    assert moved[2] == pytest.approx(moved[0], abs=1e-6)  # die mass does not change the arm's tracking here
    assert mi.shape == (3,) and torch.isfinite(mi).all()


@gpu
def test_meaninertia_is_per_world(scene, cfg):
    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 2)
    assert tuple(ws.m.stat.meaninertia.shape) == (2,)
    assert float(ws._meaninertia[0]) == pytest.approx(model.stat.meaninertia, rel=1e-5)
    p = _mjcf_params(model, 2, ws.device)
    p.armature[1] *= 10.0
    ws.set_world_params(p)
    mi = ws._meaninertia.cpu().numpy()
    assert mi[0] == pytest.approx(model.stat.meaninertia, rel=1e-3)  # set_const recomputed world 0 unchanged
    assert mi[1] > mi[0] * 1.5  # world 1's own value, not world 0's
    ws.close()


@gpu
@pytest.mark.parametrize(
    ("field", "row", "value"),
    [("kv", (0, 2), -0.1), ("die_mass", (1,), 0.0), ("die_mass", (0,), -1.0), ("kp", (1, 0), 0.0),
     ("damping", (0, 0), float("nan")), ("forcerange", (1, 3), float("inf"))],
)
def test_set_world_params_rejects_invalid(scene, cfg, field, row, value):
    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 2)
    p = _mjcf_params(model, 2, ws.device)
    before = ws._p["actuator_biasprm"].clone()
    getattr(p, field)[row] = value
    with pytest.raises(ValueError, match=field):
        ws.set_world_params(p)
    assert (ws._p["actuator_biasprm"] == before).all()  # nothing written
    # host-side params (the sampler's output) are validated without touching the device
    host = _mjcf_params(model, 2, "cpu")
    getattr(host, field)[row] = value
    with pytest.raises(ValueError, match=field):
        ws.set_world_params(host)
    ws.close()


@gpu
def test_contact_count_caches_geom_sets(scene, cfg):
    model = scene[0]
    ws = _sim(scene, cfg.sim_warp, 2)
    _to_key(ws, model, "rest")
    ws.step(17)
    die, table = model.geom("die_geom").id, model.geom("table").id
    a = ws.contact_count([die], [table])
    assert len(ws._geom_sets) == 1
    b = ws.contact_count((die,), (table,))
    assert len(ws._geom_sets) == 1 and (a == b).all()
    with pytest.raises(IndexError):
        ws.contact_count([model.ngeom], [table])
    ws.close()


@gpu
def test_exit_with_exception_is_not_masked_by_overflow(scene, cfg):
    """An exception inside ``with WarpSim(...)`` propagates unchanged; the overflow found by close() is a note."""
    small = dataclasses.replace(cfg.sim_warp, njmax=16)

    class Boom(Exception):
        pass

    with pytest.raises(Boom) as info, _sim(scene, small, 8) as ws:
        _to_key(ws, scene[0], "rest")
        ws.step(17)
        raise Boom("original")
    assert str(info.value) == "original"
    assert any("ContactBufferOverflow" in n for n in getattr(info.value, "__notes__", []))
    assert ws._closed
    # without an exception in flight the overflow is raised as before
    from so101_rl.sim.warp_model import ContactBufferOverflow

    with pytest.raises(ContactBufferOverflow), _sim(scene, small, 8) as ws2:
        _to_key(ws2, scene[0], "rest")
        ws2.step(17)
