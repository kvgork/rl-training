"""`pixi task list` must not advertise a broken task: every task's module, script or test path exists, and marker
expressions are not no-ops (SU0 review NIT)."""

import re
import shlex
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
TASKS = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pixi"]["tasks"]


def _cmd(task):
    return task if isinstance(task, str) else task.get("cmd")


def test_every_task_target_exists():
    broken = []
    for name, task in TASKS.items():
        cmd = _cmd(task)
        if cmd is None:  # depends-on only
            missing = [d for d in task.get("depends-on", []) if d not in TASKS]
            broken += [f"{name}: unknown dependency {d}" for d in missing]
            continue
        argv = shlex.split(cmd)
        if argv[:2] == ["python", "-m"]:
            rel = Path("src", *argv[2].split("."))
            if not (ROOT / rel.with_suffix(".py")).exists() and not (ROOT / rel / "__main__.py").exists():
                broken.append(f"{name}: module {argv[2]} does not exist")
        elif argv[0] == "python":
            if not (ROOT / argv[1]).exists():
                broken.append(f"{name}: script {argv[1]} does not exist")
        elif argv[0] == "pytest":
            for a in argv[1:]:
                if a.startswith("tests/") and not (ROOT / a.split("::")[0]).exists():
                    broken.append(f"{name}: test path {a} does not exist")
    assert not broken, broken


def test_marker_expressions_are_not_tautologies():
    for name, task in TASKS.items():
        cmd = _cmd(task) or ""
        for expr in re.findall(r"-m\s+\"([^\"]+)\"", cmd):
            for marker in re.findall(r"\b(\w+)\b", expr):
                if marker in ("and", "or", "not"):
                    continue
                assert not re.search(rf"\b{marker}\b\s+or\s+not\s+{marker}\b|\bnot\s+{marker}\s+or\s+{marker}\b", expr), (
                    name, expr)


def test_every_python_m_task_has_a_cli_entry_point():
    """`python -m <module>` on a module without a `__main__` guard imports it and exits 0 silently, which looks like
    a pass (the `calib-check` bug, 2026-10-09)."""
    silent = []
    for name, task in TASKS.items():
        argv = shlex.split(_cmd(task) or "")
        if argv[:2] != ["python", "-m"]:
            continue
        rel = ROOT / Path("src", *argv[2].split("."))
        if (rel / "__main__.py").exists():
            continue
        src = rel.with_suffix(".py").read_text()
        if not re.search(r"^if __name__ == [\"']__main__[\"']:", src, re.MULTILINE):
            silent.append(f"{name}: {argv[2]} has no `if __name__ == \"__main__\":`")
    assert not silent, silent
