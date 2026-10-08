"""Dependency rule: ``so101_rl.lab`` never imports the safety clamps, the servo core, the Feetech bus or serial
(static AST scan of every lab module, plus a runtime ``sys.modules`` check)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent / "src" / "so101_rl" / "lab"
FORBIDDEN = ("serial", "so101_rl.safety", "so101_rl.servo", "so101_rl.perception.camera", "pyrealsense2")


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def test_lab_modules_never_import_safety_servo_or_serial():
    files = sorted(LAB.glob("*.py"))
    assert {f.name for f in files} >= {"runs.py", "trainers.py", "job.py", "launcher.py", "api.py", "mcp_server.py"}
    bad = {f.name: sorted(m for m in _imports(f) if any(m == x or m.startswith(x + ".") for x in FORBIDDEN))
           for f in files}
    assert not any(bad.values()), bad


def test_importing_the_whole_lab_loads_no_safety_servo_or_serial_module():
    code = (
        "import sys, so101_rl.lab.api, so101_rl.lab.mcp_server, so101_rl.lab.cli, so101_rl.lab.job; "
        "bad = [m for m in sys.modules if m == 'serial' or m.startswith(('serial.', 'so101_rl.safety', "
        "'so101_rl.servo', 'so101_rl.perception'))]; print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]"
