"""Environment sanity: one torch (conda), numpy 2.2, python 3.12, lerobot import does not touch CUDA."""

import json
import os
import subprocess
import sys

import pytest


def _pixi_packages() -> list[dict]:
    env = os.environ.get("PIXI_ENVIRONMENT_NAME", "default")
    out = subprocess.run(
        ["pixi", "list", "-e", env, "--json"], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(out.stdout)


def test_single_torch():
    pkgs = _pixi_packages()
    torches = [p for p in pkgs if p["name"] in ("torch", "pytorch")]
    assert len(torches) == 1, torches
    assert torches[0]["name"] == "pytorch"
    assert torches[0]["kind"] == "conda"


def test_import_lerobot_motors_no_torch_init():
    code = (
        "import lerobot.motors.feetech, torch; "
        "assert not torch.cuda.is_initialized(); print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_numpy_version():
    import numpy

    assert numpy.__version__.startswith("2.2."), numpy.__version__


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.parametrize("mod", ["torchrl", "tensordict", "mujoco", "cv2", "stable_baselines3"])
def test_train_stack_imports(mod):
    __import__(mod)
