"""The table height has ONE source: config ``safe_state.table_z_m`` (recalibration prep, 2026-10-10).

Every consumer follows it: the box tip floor (table + 5 mm) and gripper floor (table + 1 mm, loader-derived), the
transit floor (``safe_state``), the reach goal box and the die zone height (``env/tasks.py``), and the sim scene
(``sim/model.py``: table geom, zone / fiducial sites, tray, die, keyframe die z), which the camera code reads as
``TABLE_Z_M``. Before this, ``-0.02`` / ``-0.015`` were written in the yaml twice, in ``sim/model.py``, in
``env/tasks.py`` (zone) and implicitly through ``WorkspaceBox()`` schema defaults (goal box)."""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from so101_rl.config import DEFAULT_CONFIG_PATH, load_config, load_geometry, load_tasks_config
from so101_rl.env.tasks import DIE_HALF_EDGE_M, DIE_TO_ZONE, REACH, task_from_config
from so101_rl.sim.model import TABLE_Z_M, build_spec

SRC = Path(__file__).parent.parent / "src" / "so101_rl"
# a name containing table_z / z_table assigned (or defaulted) to a number literal
LITERAL = re.compile(r"(?i)\b\w*(?:table_z|z_table)\w*\b\s*(?::\s*[\w\[\]. |]+?)?\s*=\s*[-+]?\d")
WHAT_IF_Z = -0.005


def _code_lines():
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts[0] in {"config", "_vendor"}:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            yield f"{rel}:{i}: {line.strip()}", line.split("#", 1)[0]


def test_no_table_height_literal_outside_config():
    assert [w for w, code in _code_lines() if LITERAL.search(code)] == []


def test_no_default_workspace_box_outside_config():
    """``WorkspaceBox()`` uses the schema defaults, a silent second copy of the box (and its table-derived floors)."""
    assert [w for w, code in _code_lines() if "WorkspaceBox()" in code] == []


@pytest.mark.parametrize("line", ["TABLE_Z_M: Final[float] = -0.02", "z_table_m: float = -0.02", "table_z_m=-0.02"])
def test_literal_pattern_detects(line):
    assert LITERAL.search(line)


@pytest.mark.parametrize("line", ["TABLE_Z_M: Final[float] = load_geometry()[0]", "z_table_m=table_z_m", "table_z = float(t)"])
def test_literal_pattern_ignores_reads(line):
    assert not LITERAL.search(line)


def test_shipped_consumers_agree():
    table_z, box = load_geometry()
    cfg = load_config()
    assert table_z == cfg.safe_state.table_z_m == TABLE_Z_M
    assert box == cfg.box
    assert REACH.goal_bounds.box == cfg.box and DIE_TO_ZONE.goal_bounds.box == cfg.box
    assert DIE_TO_ZONE.zone.z_table_m == table_z
    model = build_spec(camera=False).compile()
    assert model.geom("table").pos[2] == pytest.approx(table_z)


@pytest.fixture
def what_if_yaml(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["safe_state"]["table_z_m"] = WHAT_IF_Z
    raw["kinematics_path"] = str(DEFAULT_CONFIG_PATH.parent / raw["kinematics_path"])
    p = tmp_path / "so101.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_one_edit_moves_every_floor(what_if_yaml):
    table_z, box = load_geometry(what_if_yaml)
    assert table_z == WHAT_IF_Z
    assert box.z[0] == pytest.approx(WHAT_IF_Z + 0.005) and box.gripper_floor_z_m == pytest.approx(WHAT_IF_Z + 0.001)
    die = task_from_config("die_to_zone", load_tasks_config(what_if_yaml), (table_z, box))
    assert die.zone.goal_xyz()[2] == pytest.approx(WHAT_IF_Z + DIE_HALF_EDGE_M)
    assert die.goal_bounds.box.z[0] == pytest.approx(WHAT_IF_Z + 0.005)


def test_sim_scene_follows_the_table():
    """The table and everything resting on it move together; the arm does not."""
    base = build_spec(camera=False).compile()
    moved = build_spec(camera=False, table_z_m=WHAT_IF_Z).compile()
    dz = WHAT_IF_Z - TABLE_Z_M
    assert moved.geom("table").pos[2] == pytest.approx(WHAT_IF_Z)
    for site in ("zone", "fiducial"):
        assert moved.site(site).pos[2] - base.site(site).pos[2] == pytest.approx(dz)
    for body in ("tray", "die"):
        assert moved.body(body).pos[2] - base.body(body).pos[2] == pytest.approx(dz)
    adr = int(moved.jnt_qposadr[moved.joint("die_free").id])
    for key in ("rest", "home"):
        assert moved.key(key).qpos[adr + 2] == pytest.approx(WHAT_IF_Z + DIE_HALF_EDGE_M)
        np.testing.assert_array_equal(moved.key(key).qpos[:adr], base.key(key).qpos[:adr])
    np.testing.assert_array_equal(moved.body_pos[moved.body("base").id], base.body_pos[base.body("base").id])
