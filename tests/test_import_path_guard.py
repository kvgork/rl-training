"""tests/conftest.py refuses to run when ``so101_rl`` is imported from outside ``<rootdir>/src``.

A shared editable install can point at another worktree; the session must then stop before any test runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = "tests/test_import_path_guard.py::test_target_noop"


def test_target_noop():
    """Collected by the subprocess runs below only; trivially passes."""


def _run(env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--rootdir", str(ROOT), TARGET],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )


def test_foreign_so101_rl_fails_the_session(tmp_path):
    fake = tmp_path / "elsewhere" / "so101_rl"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    r = _run({"PYTHONPATH": str(fake.parent)})
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "not inside this checkout's" in out and str(fake.resolve()) in out, out
    assert "1 passed" not in out  # nothing ran


def test_own_checkout_passes():
    r = _run({})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 passed" in r.stdout
