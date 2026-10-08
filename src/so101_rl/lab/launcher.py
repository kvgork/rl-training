"""Launch job segments so they outlive the MCP server (P12), and check whether they are still alive.

- ``systemd``: a transient user unit ``systemd-run --user --unit so101-job-<run_id> --collect`` with a pinned
  argv, the working directory set, stdout/stderr appended to ``<run_dir>/job.log`` and a clean environment
  (the user manager's, plus a few ``--setenv``).
- ``subprocess``: the fallback when ``systemd --user`` is unavailable (and in tests): a detached child
  (``start_new_session``, stdin from /dev/null, stdio to ``job.log``, a minimal environment).

Import pinning: every job gets ``PYTHONPATH=<repo>/src`` and ``SO101_LAB_REPO=<repo>`` (``<repo>`` = the checkout
this launcher was imported from), so it runs THIS checkout's code wherever the shared pixi editable install points;
the job refuses to start when ``so101_rl`` still comes from elsewhere (``so101_rl.lab.job.check_import_root``).

Liveness is re-checked by unit name (``systemctl --user show``) or by pid + process start time (pid reuse safe).
Environment overrides: ``SO101_LAB_LAUNCHER`` = ``auto|systemd|subprocess``; ``SO101_LAB_PYTHON`` = a python
executable used instead of ``pixi run ... python`` (tests).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from so101_rl.lab.errors import LabError
from so101_rl.lab.runs import REPO_ROOT

log = logging.getLogger(__name__)

ALIVE_UNIT_STATES = frozenset({"active", "activating", "deactivating", "reloading"})


def boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _proc_stat(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    # comm may contain spaces/parens: the fields after the last ')' start with the state (field 3)
    return raw[raw.rfind(")") + 2 :].split()


def proc_start_time(pid: int) -> int | None:
    f = _proc_stat(pid)
    return int(f[19]) if f and len(f) > 19 else None  # field 22 (starttime), 0-based 19 after the comm


def pid_alive(pid: int | None, start: int | None = None) -> bool:
    if not pid:
        return False
    f = _proc_stat(pid)
    if not f or f[0] in ("Z", "X", "x"):
        return False
    return start is None or (len(f) > 19 and int(f[19]) == int(start))


def _systemctl(*args: str) -> subprocess.CompletedProcess[str] | None:
    exe = shutil.which("systemctl")
    if not exe:
        return None
    try:
        return subprocess.run([exe, "--user", *args], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def systemd_user_available() -> bool:
    if not shutil.which("systemd-run"):
        return False
    r = _systemctl("is-system-running")
    return r is not None and r.stdout.strip() in ("running", "degraded", "starting")


def unit_state(unit: str) -> str | None:
    r = _systemctl("show", "-p", "ActiveState", "--value", unit)
    if r is None or r.returncode != 0:
        return None
    return r.stdout.strip() or None


def unit_main_pid(unit: str) -> int | None:
    r = _systemctl("show", "-p", "MainPID", "--value", unit)
    try:
        pid = int(r.stdout.strip()) if r is not None and r.returncode == 0 else 0
    except ValueError:
        pid = 0
    return pid or None


@dataclass(frozen=True)
class LaunchInfo:
    launcher: str  # "systemd" | "subprocess"
    unit: str | None
    pid: int | None
    pid_start: int | None


class Launcher:
    def __init__(self, mode: str | None = None, python: Sequence[str] | None = None) -> None:
        mode = mode or os.environ.get("SO101_LAB_LAUNCHER", "auto")
        if mode not in ("auto", "systemd", "subprocess"):
            raise ValueError(f"launcher mode {mode!r}")
        if mode == "auto":
            mode = "systemd" if systemd_user_available() else "subprocess"
        self.mode = mode
        env_py = os.environ.get("SO101_LAB_PYTHON")
        self.python: list[str] | None = list(python) if python else ([env_py] if env_py else None)

    def env(self, root: Path, unit: str | None) -> dict[str, str]:
        e = {"PYTHONNOUSERSITE": "1", "SO101_LAB_RUNS": str(root), "PYTHONPATH": str(REPO_ROOT / "src"),
             "SO101_LAB_REPO": str(REPO_ROOT)}
        if unit:
            e["SO101_LAB_UNIT"] = unit
        return e

    def launch(self, run_dir: Path, argv: Sequence[str], *, root: Path) -> LaunchInfo:
        run_id = run_dir.name
        logf = run_dir / "job.log"
        if self.mode == "systemd":
            unit = f"so101-job-{run_id}"
            cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet",
                   f"--working-directory={REPO_ROOT}", f"-pStandardOutput=append:{logf}",
                   f"-pStandardError=append:{logf}", "-pKillMode=mixed", "-pTimeoutStopSec=130"]
            cmd += [f"--setenv={k}={v}" for k, v in self.env(root, unit).items()]
            cmd += ["--", *argv]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            except (OSError, subprocess.SubprocessError) as e:
                raise LabError("LAUNCH_FAILED", "systemd-run failed", details={"error": str(e)[:200]}) from None
            if r.returncode != 0:
                raise LabError("LAUNCH_FAILED", "systemd-run refused the unit",
                               details={"rc": r.returncode, "stderr": r.stderr.strip()[-300:]})
            pid = unit_main_pid(unit)
            return LaunchInfo("systemd", unit, pid, proc_start_time(pid) if pid else None)
        base = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "USER", "LOGNAME") if k in os.environ}
        env: Mapping[str, str] = {**base, **self.env(root, None)}
        try:
            with logf.open("ab") as out:
                proc = subprocess.Popen(list(argv), cwd=REPO_ROOT, env=dict(env), stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        except OSError as e:
            raise LabError("LAUNCH_FAILED", "could not start the job process",
                           details={"error": str(e)[:200]}) from None
        threading.Thread(target=proc.wait, daemon=True, name=f"reap-{proc.pid}").start()  # no zombies
        return LaunchInfo("subprocess", None, proc.pid, proc_start_time(proc.pid))

    @staticmethod
    def alive(status: Mapping[str, Any]) -> bool:
        """Is the process that holds this run still there? By unit name first, else by pid + start time."""
        unit = status.get("unit")
        if unit:
            st = unit_state(str(unit))
            if st is not None:
                return st in ALIVE_UNIT_STATES
        return pid_alive(status.get("pid"), status.get("pid_start"))
