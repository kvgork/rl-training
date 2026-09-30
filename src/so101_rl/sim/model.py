"""Scene loading for the MuJoCo backend (plan F10).

``assets/so101/so101_rl_scene.xml`` includes the untouched upstream arm and adds the table, die, zone, fiducial and
keyframes. MJCF cannot add geoms to an *included* body, so :func:`load_model` compiles the scene through ``MjSpec``
and adds the primitive jaw collision geoms (class ``jaw_col``) and the ``gripper_tip`` site there; the upstream mesh
collision geoms of both jaws are switched off (``contype = conaffinity = 0``) so grasp contact is primitive-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import mujoco
import numpy as np

ASSETS: Final[Path] = Path(__file__).resolve().parents[3] / "assets"
SCENE_XML: Final[Path] = ASSETS / "so101" / "so101_rl_scene.xml"

TABLE_Z_M: Final[float] = -0.02  # must match the table geom in so101_rl_scene.xml (test-enforced)
DIE_EDGE_M: Final[float] = 0.016
ZONE_EDGE_M: Final[float] = 0.060

FIXED_JAW_BODY: Final[str] = "gripper"
MOVING_JAW_BODY: Final[str] = "moving_jaw_so101_v1"
JAW_COL_CLASS: Final[str] = "jaw_col"

# (name, body, centre, half-size) in the owning body frame, fitted to the vertices of the upstream jaw meshes
# (distal finger + proximal segment per jaw). Placeholders until sim/real contact is compared in H-series.
_JAW_BOXES: Final[tuple[tuple[str, str, tuple[float, float, float], tuple[float, float, float]], ...]] = (
    ("jaw_fixed_tip", FIXED_JAW_BODY, (-0.0130, 0.0, -0.0950), (0.0050, 0.0062, 0.0094)),
    ("jaw_fixed_base", FIXED_JAW_BODY, (-0.0180, 0.0, -0.0700), (0.0070, 0.0090, 0.0200)),
    ("jaw_moving_tip", MOVING_JAW_BODY, (-0.0052, -0.0710, 0.0189), (0.0070, 0.0110, 0.0073)),
    ("jaw_moving_base", MOVING_JAW_BODY, (-0.0014, -0.0500, 0.0189), (0.0070, 0.0100, 0.0086)),
)


def build_spec(scene_xml: Path = SCENE_XML) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    jaw_default = spec.find_default(JAW_COL_CLASS)
    if jaw_default is None:
        raise ValueError(f"{scene_xml}: default class {JAW_COL_CLASS!r} missing")

    # Disable upstream mesh collisions on both jaws (visual geoms already have contype 0).
    for body_name in (FIXED_JAW_BODY, MOVING_JAW_BODY):
        body = spec.body(body_name)
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.contype = 0
                geom.conaffinity = 0

    for name, body_name, pos, half in _JAW_BOXES:
        spec.body(body_name).add_geom(name=name, default=jaw_default, pos=list(pos), size=list(half))

    gripper = spec.body(FIXED_JAW_BODY)
    frame = spec.site("gripperframe")
    gripper.add_site(name="gripper_tip", pos=list(frame.pos), quat=list(frame.quat), size=[0.003, 0.003, 0.003], group=3)
    return spec


def load_model(scene_xml: Path = SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the RL scene. Resolves mesh paths relative to ``assets/``; returns ``(model, data)`` at keyframe ``rest``."""
    model = build_spec(Path(scene_xml)).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    mujoco.mj_forward(model, data)
    return model, data


def jaw_collision_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.geom(n).id for n, *_ in _JAW_BOXES], dtype=np.int32)
