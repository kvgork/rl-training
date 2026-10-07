"""C.1: calib/camera.yaml is the ONLY source of the D435 intrinsics and pose. No hard-coded camera intrinsics,
camera pose or fovy remain in the sources (``src/``, the scene MJCF, the scripts)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [p for p in (ROOT / "src" / "so101_rl").rglob("*.py") if "_vendor" not in p.parts] + list((ROOT / "scripts").glob("*.py"))
SCENE = ROOT / "assets" / "so101" / "so101_rl_scene.xml"

# measured / formerly hard-coded values (colour fx fy cx cy, depth fx cx cy, the old nominal focal length and
# heights, the plane height) and the names of the removed constants
FORBIDDEN_NUMBERS = ["617.1", "617.5", "617.6", "327.9", "236.1", "388.3", "326.9", "240.8", "615.0", "0.614", "0.594", "345.7", "0.3457", "42.63"]
FORBIDDEN_NAMES = ["NOMINAL_CAM_POS", "NOMINAL_F_PX", "D435_FY_PX", "default_synthetic_camera_pose", "CameraModel.nominal", "HAND_EYE_PATH", "overhead_camera_pose"]
CAMERA_CTORS = {"Intrinsics", "CameraModel", "StreamIntrinsics"}


def _code_only(text: str) -> str:
    """The source without comments and docstrings (documentation may quote the measured values)."""
    tree = ast.parse(text)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            doc_lines.update(range(node.lineno, node.end_lineno + 1))
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if i in doc_lines:
            continue
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


def test_no_hard_coded_intrinsics_or_pose_in_the_sources():
    hits = []
    for p in SOURCES:
        code = _code_only(p.read_text())
        for tok in FORBIDDEN_NUMBERS + FORBIDDEN_NAMES:
            if tok in code:
                hits.append(f"{p.relative_to(ROOT)}: {tok}")
    assert not hits, hits


def test_no_camera_constructed_from_literals():
    hits = []
    for p in SOURCES:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                args = list(node.args) + [k.value for k in node.keywords]
                if name in CAMERA_CTORS and any(isinstance(a, ast.Constant) and isinstance(a.value, float) for a in args):
                    hits.append(f"{p.relative_to(ROOT)}:{node.lineno} {name}(...literal...)")
    assert not hits, hits


def test_no_camera_and_no_fovy_in_the_scene_xml():
    xml = re.sub(r"<!--.*?-->", "", SCENE.read_text(), flags=re.DOTALL)
    assert "<camera" not in xml and "fovy" not in xml


def test_the_shown_to_fail_twin_catches_a_literal():
    bad = 'from so101_rl.perception.camera import Intrinsics\nI = Intrinsics(615.0, 615.0, 320.0, 240.0, 640, 480, 0.001)\n'
    assert "615.0" in _code_only(bad)
    assert "615.0" not in _code_only('"""fx 615.0 in a docstring"""\nx = 1  # 615.0 in a comment\n')
