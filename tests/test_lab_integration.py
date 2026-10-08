"""End to end over stdio: start a fake run through the MCP server, SIGKILL the server, the run finishes on its
own, and a new server/client reads the result (acceptance of plans/mvp-plan.md M5).

The default variant uses the detached-subprocess launcher (CI); the ``integration`` variant launches the job as a
real transient ``systemd --user`` unit and is skipped when no user manager is available.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from so101_rl.lab import runs
from so101_rl.lab.launcher import pid_alive, systemd_user_available

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _params(root: Path, launcher: str) -> StdioServerParameters:
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "USER", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
           if k in os.environ}
    env.update(SO101_LAB_RUNS=str(root), SO101_LAB_LAUNCHER=launcher, SO101_LAB_PYTHON=sys.executable,
               PYTHONPATH=SRC, PYTHONNOUSERSITE="1")
    return StdioServerParameters(command=sys.executable, args=["-m", "so101_rl.lab.mcp_server"], env=env)


def _ppid(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rfind(")") + 2 :].split()[1])


async def _start_and_kill_server(params: StdioServerParameters, root: Path) -> tuple[str, int]:
    run_id, server_pid = "", 0
    with contextlib.suppress(BaseException):  # the session breaks once the server is killed
        async with Client(params) as c:
            r = await c.call_tool("start_run", {"trainer": "fake", "task": "reach", "idempotency_key": "it",
                                                "config": {"steps": 150, "step_s": 0.02}})
            out = json.loads(r.content[0].text)
            assert out["ok"], out
            run_id = out["run_id"]
            # the job is up and holding the run; find the server process (the job's parent in subprocess mode)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 30:
                st = runs.read_status(root / run_id)
                if st.get("state") == "running" and st.get("pid"):
                    break
                await anyio.sleep(0.1)
            server_pid = _server_pid(params)
            os.kill(server_pid, signal.SIGKILL)
            await anyio.sleep(0.3)
            await c.call_tool("list_runs", {})  # raises: the server is gone
    return run_id, server_pid


def _server_pid(params: StdioServerParameters) -> int:
    out = subprocess.run(["pgrep", "-f", "-n", "so101_rl.lab.mcp_server"], capture_output=True, text=True, check=False)
    pids = [int(p) for p in out.stdout.split()]
    mine = [p for p in pids if _ppid(p) == os.getpid()]
    assert mine, f"server process not found: {out.stdout}"
    return mine[0]


async def _read_back(params: StdioServerParameters, run_id: str) -> tuple[dict, dict]:
    async with Client(params) as c:
        got = json.loads((await c.call_tool("get_run", {"run_id": run_id, "wait_s": 60})).content[0].text)
        listed = json.loads((await c.call_tool("list_runs", {})).content[0].text)
    return got, listed


def _scenario(tmp_path: Path, launcher: str) -> tuple[dict, dict, int]:
    root = tmp_path / "runs"
    params = _params(root, launcher)
    run_id, server_pid = anyio.run(_start_and_kill_server, params, root)
    assert run_id and server_pid
    assert not pid_alive(server_pid)
    got, listed = anyio.run(_read_back, params, run_id)
    return got, listed, server_pid


def test_run_survives_mcp_server_kill_subprocess(tmp_path):
    got, listed, _ = _scenario(tmp_path, "subprocess")
    assert got["ok"] and got["status"]["state"] == "completed" and got["status"]["env_step"] == 150
    assert got["timed_out"] is False
    assert listed["runs"][0]["state"] == "completed"


@pytest.mark.integration
@pytest.mark.skipif(not systemd_user_available(), reason="no systemd --user manager")
def test_run_survives_mcp_server_kill_systemd(tmp_path):
    try:
        got, listed, _ = _scenario(tmp_path, "systemd")
        assert got["status"]["state"] == "completed" and got["status"]["env_step"] == 150
        assert listed["runs"][0]["state"] == "completed"
        run_dir = tmp_path / "runs" / got["run"]["run_id"]
        assert runs.read_status(run_dir)["unit"] == f"so101-job-{got['run']['run_id']}"
        assert runs.read_status(run_dir)["launcher"] == "systemd"
    finally:
        for d in runs.list_run_dirs(tmp_path / "runs"):
            subprocess.run(["systemctl", "--user", "stop", f"so101-job-{d.name}"], capture_output=True, check=False)
