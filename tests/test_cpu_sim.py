"""``sim/cpu_sim.py``: the CPU MuJoCo stand-in for WarpSim behind ``BatchedGoalEnv(device="cpu")`` (D2b)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from so101_rl.config import load_config
from so101_rl.env.batched_env import model_sim_params
from so101_rl.sim.cpu_sim import CpuSim
from so101_rl.sim.model import load_model

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def model():
    return load_model()[0]


def _home(model, n):
    k = model.key("home")
    return torch.as_tensor(k.qpos, dtype=torch.float32).expand(n, -1), torch.as_tensor(k.ctrl, dtype=torch.float32).expand(n, -1)


def test_matches_plain_mujoco_and_masks_resets(model):
    sim = CpuSim(model, load_config().sim_warp, 2)
    q, c = _home(model, 2)
    sim.reset_worlds(torch.ones(2, dtype=torch.bool), q, torch.zeros(2, model.nv), c)
    sim.ctrl[1, 1] += 0.2  # world 1 gets a shoulder_lift step
    sim.step(17)
    ref = mujoco.MjData(model)
    ref.qpos[:], ref.ctrl[:] = model.key("home").qpos, model.key("home").ctrl
    mujoco.mj_forward(model, ref)
    for _ in range(17):
        mujoco.mj_step(model, ref)
    np.testing.assert_allclose(sim.qpos[0].numpy(), ref.qpos.astype(np.float32), atol=1e-6)
    assert abs(float(sim.qpos[1, 1] - sim.qpos[0, 1])) > 1e-3
    before = sim.qpos[1].clone()
    sim.reset_worlds(torch.tensor([True, False]), q, torch.zeros(2, model.nv), c)  # world 1 untouched
    assert torch.equal(sim.qpos[1], before)
    assert torch.allclose(sim.qpos[0], q[0])
    assert torch.allclose(sim.body_xpos("die")[0], torch.as_tensor(model.key("home").qpos[6:9], dtype=torch.float32), atol=1e-6)


def test_contacts_and_world_params(model):
    sim = CpuSim(model, load_config().sim_warp, 1)
    q, c = _home(model, 1)
    sim.reset_worlds(torch.ones(1, dtype=torch.bool), q, torch.zeros(1, model.nv), c)
    sim.step(10)
    die, table = model.geom("die_geom").id, model.geom("table").id
    assert int(sim.contact_count([die], [table])[0]) > 0  # the die rests on the table
    p = model_sim_params(model)
    rows = type("P", (), {k: torch.tensor(np.array(getattr(p, k))).float().expand(1, 6) for k in
                          ("kp", "kv", "damping", "frictionloss", "armature", "forcerange")})()  # fmt: skip
    rows.die_mass, rows.die_friction, rows.table_friction = (torch.full((1,), float(getattr(p, k))) for k in
                                                             ("die_mass_kg", "die_friction", "table_friction"))  # fmt: skip
    rows.die_mass = rows.die_mass * 2.0
    sim.set_world_params(rows)
    m0, _ = sim.world_data(0)
    assert m0.body_mass[model.body("die").id] == pytest.approx(2 * model.body_mass[model.body("die").id], rel=1e-6)
    assert model.body_mass[model.body("die").id] == pytest.approx(p.die_mass_kg)  # the caller's model is untouched
    sim.check_overflow()
    assert sim.overflow_checks == 1 and sim.max_nefc_seen > 0
    sim.close()
    with pytest.raises(RuntimeError):
        sim.step(1)


def test_batched_env_on_cpu_uses_the_cpu_sim():
    from so101_rl.rl import eval_sim

    env = eval_sim.build_env("die_to_zone", 2, stage=0, seed=0, device="cpu",
                             calibration=str(ROOT / "tests/data/so101_follower.calib.json"),
                             sim_params=str(ROOT / "calib/sim_params.yaml"))  # fmt: skip
    try:
        assert isinstance(env.ws, CpuSim)
        obs, priv, _, _ = env.reset(phase_randomise=False)
        assert obs.shape == (2, 26) and priv.shape == (2, env.priv_dim)
        st = env.step(torch.zeros(2, 6))
        assert st.reward.shape == (2,) and not bool(st.info["tripped"].any())
        assert torch.allclose(st.desired, torch.as_tensor(env.task.zone.goal_xyz(), dtype=torch.float32).expand(2, 3))
    finally:
        env.close()
