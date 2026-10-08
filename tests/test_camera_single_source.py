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


def _is_number(a: ast.AST) -> bool:
    if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub | ast.UAdd):
        a = a.operand
    return isinstance(a, ast.Constant) and isinstance(a.value, int | float) and not isinstance(a.value, bool)


def _literal_ctor_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """Camera / intrinsics constructors given a numeric literal (float OR int: a ``640, 480`` image size is as much
    a hard-coded intrinsic as ``fx``), also inside a literal list / tuple argument."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name not in CAMERA_CTORS:
                continue
            args = list(node.args) + [k.value for k in node.keywords]
            flat = []
            for a in args:
                flat.extend(a.elts if isinstance(a, ast.List | ast.Tuple) else [a])
            if any(_is_number(a) for a in flat):
                hits.append((node.lineno, name))
    return hits


def _pose_literal_hits(tree: ast.AST) -> list[int]:
    """A 4x4 numeric nested-list literal (a hand-written homogeneous pose / extrinsic), anywhere in the code."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List | ast.Tuple) and len(node.elts) == 4:
            rows = node.elts
            if all(isinstance(r, ast.List | ast.Tuple) and len(r.elts) == 4 and all(_is_number(e) for e in r.elts) for r in rows):
                hits.append(node.lineno)
    return hits


def test_no_camera_constructed_from_literals():
    hits = []
    for p in SOURCES:
        hits += [f"{p.relative_to(ROOT)}:{ln} {name}(...literal...)" for ln, name in _literal_ctor_hits(ast.parse(p.read_text()))]
    assert not hits, hits


def test_no_hand_written_4x4_pose_literal():
    hits = []
    for p in SOURCES:
        hits += [f"{p.relative_to(ROOT)}:{ln}" for ln in _pose_literal_hits(ast.parse(p.read_text()))]
    assert not hits, hits


def test_no_camera_and_no_fovy_in_the_scene_xml():
    xml = re.sub(r"<!--.*?-->", "", SCENE.read_text(), flags=re.DOTALL)
    assert "<camera" not in xml and "fovy" not in xml


def test_the_shown_to_fail_twin_catches_a_literal():
    bad = 'from so101_rl.perception.camera import Intrinsics\nI = Intrinsics(615.0, 615.0, 320.0, 240.0, 640, 480, 0.001)\n'
    assert "615.0" in _code_only(bad)
    assert "615.0" not in _code_only('"""fx 615.0 in a docstring"""\nx = 1  # 615.0 in a comment\n')


def test_shown_to_fail_twins_catch_int_intrinsics_and_4x4_pose_literals():
    ints = ast.parse("I = Intrinsics(fx, fy, cx, cy, 640, 480, scale)\nS = StreamIntrinsics(width=640, height=480)")
    assert [n for _, n in _literal_ctor_hits(ints)] == ["Intrinsics", "StreamIntrinsics"]
    listy = ast.parse("C = CameraModel([615, 615], 1)")
    assert _literal_ctor_hits(listy)
    assert not _literal_ctor_hits(ast.parse("I = Intrinsics(fx, fy, cx, cy, w, h, scale)"))
    pose = ast.parse("T = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]])")
    assert _pose_literal_hits(pose)
    assert not _pose_literal_hits(ast.parse("T = np.eye(4)\nR = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]"))
