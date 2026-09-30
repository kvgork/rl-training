import importlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
VENDOR = ROOT / "src" / "so101_rl" / "_vendor" / "tdmpc2"


def test_import_without_cuda():
    code = (
        "import so101_rl._vendor.tdmpc2.tdmpc2 as m, torch; "
        "import so101_rl._vendor.tdmpc2.common.buffer, so101_rl._vendor.tdmpc2.trainer.online_trainer; "
        "assert not torch.cuda.is_initialized(); print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90, check=False)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_module_importable_in_process():
    assert importlib.import_module("so101_rl._vendor.tdmpc2.tdmpc2").TDMPC2


def test_license_is_mit():
    text = (VENDOR / "LICENSE").read_text()
    assert "MIT License" in text and "Nicklas Hansen" in text


def test_upstream_metadata_pinned():
    meta = json.loads((VENDOR / "UPSTREAM.json").read_text())
    assert len(meta["sha"]) == 40 and meta["license"] == "MIT"
    assert "envs/" in meta["excluded"]
    assert not (VENDOR / "envs").exists()


def test_patches_md_lists_every_patched_file():
    text = (VENDOR / "PATCHES.md").read_text()
    for rel in ("tdmpc2.py", "common/buffer.py", "common/scale.py", "common/logger.py", "train.py"):
        assert f"### {rel}" in text, rel


def test_no_hard_coded_cuda_device_left():
    for path in VENDOR.rglob("*.py"):
        code = path.read_text()
        assert "device('cuda:0')" not in code and 'device="cuda:0"' not in code, path


def test_vendor_check_reproduces_tree():
    """Rebuilds the tree from the (cached) upstream tarball and compares byte-for-byte."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vendor_tdmpc2.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if out.returncode != 0 and ("urlopen" in out.stderr or "URLError" in out.stderr):
        pytest.skip("offline and upstream tarball not cached")
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.slow
def test_smoke_200_steps_cpu(tmp_path):
    from so101_rl.rl.smoke_tdmpc2 import run

    assert run(200, "cpu", 1, tmp_path) == 0
    (run_dir,) = list((tmp_path / "smoke-tdmpc2").iterdir())
    assert (run_dir / "model.pt").exists()
    assert len(pd.read_csv(run_dir / "metrics.csv")) >= 3
