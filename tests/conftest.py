import os
import shutil
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
SRC_PKG = str((Path(__file__).parent.parent / "src" / "so101_rl").resolve())


def pytest_configure(config):
    for name, doc in (
        ("hardware", "needs the real arm (SO101_HW=1)"),
        ("gpu", "needs CUDA"),
        ("slow", "long-running test"),
        ("regression", "regression test"),
    ):
        config.addinivalue_line("markers", f"{name}: {doc}")
    # The stdlib `logging` name would be shadowed if src/so101_rl itself were importable as top level.
    on_path = [p for p in sys.path if p and str(Path(p).resolve()) == SRC_PKG]
    assert not on_path, f"src/so101_rl must not be on sys.path: {on_path}"


def pytest_collection_modifyitems(config, items):
    import torch

    skip_hw = pytest.mark.skip(reason="needs the arm: set SO101_HW=1")
    skip_gpu = pytest.mark.skip(reason="needs CUDA")
    has_hw = os.environ.get("SO101_HW") == "1"
    has_gpu = torch.cuda.is_available()
    for item in items:
        if "hardware" in item.keywords and not has_hw:
            item.add_marker(skip_hw)
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)


@pytest.fixture
def fake_clock():
    from so101_rl.servo.timing import FakeClock

    return FakeClock()


@pytest.fixture
def calibration():
    from so101_rl.config import load_calibration

    return load_calibration(DATA / "so101_follower.calib.json")


@pytest.fixture
def cfg(calibration):
    """Default so101.yaml with the real calibration JSON copied into tests/data."""
    from so101_rl.config import load_config

    return load_config(calibration=calibration)


@pytest.fixture
def calib_json_path(tmp_path):
    dst = tmp_path / "so101_follower.json"
    shutil.copy(DATA / "so101_follower.calib.json", dst)
    return dst
