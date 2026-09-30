import hashlib
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
VENDOR = ROOT / "src" / "so101_rl" / "_vendor" / "tdmpc2"


@pytest.fixture(scope="module")
def vendor_script():
    spec = importlib.util.spec_from_file_location("vendor_tdmpc2_script", ROOT / "scripts" / "vendor_tdmpc2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


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


def test_vendored_files_match_committed_manifest_offline(vendor_script):
    """No network: every vendored file (and no extra/missing one) hashes to the committed manifest, so a
    hand-edit of the generated tree is caught even when the upstream tarball is not cached."""
    assert vendor_script.verify_manifest() == []
    manifest = json.loads(vendor_script.MANIFEST.read_text())
    assert "tdmpc2.py" in manifest and "UPSTREAM.json" in manifest and "PATCHES.md" in manifest


def test_manifest_catches_hand_edit(vendor_script, tmp_path, monkeypatch):
    import shutil

    copy = tmp_path / "tdmpc2"
    shutil.copytree(VENDOR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(vendor_script, "OUT", copy)
    assert vendor_script.verify_manifest() == []
    (copy / "tdmpc2.py").write_text((copy / "tdmpc2.py").read_text() + "\n# hand edit\n")
    assert vendor_script.verify_manifest() == ["tdmpc2.py"]
    (copy / "tdmpc2.py").write_text((VENDOR / "tdmpc2.py").read_text())
    (copy / "extra.py").write_text("x = 1\n")
    assert vendor_script.verify_manifest() == ["extra.py"]


def test_verify_manifest_cli_offline():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vendor_tdmpc2.py"), "--verify-manifest"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_recorded_tarball_hash_is_pinned(vendor_script):
    meta = json.loads((VENDOR / "UPSTREAM.json").read_text())
    assert len(meta["tarball_sha256"]) == 64
    assert vendor_script.recorded_tarball_sha256(meta["sha"]) == meta["tarball_sha256"]
    assert vendor_script.recorded_tarball_sha256("0" * 40) is None


def test_cached_tarball_with_wrong_hash_is_an_error(vendor_script, tmp_path, monkeypatch):
    monkeypatch.setattr(vendor_script, "CACHE", tmp_path)
    good = _tar({"r/LICENSE": b"MIT"})
    (tmp_path / "tdmpc2-abc.tar.gz").write_bytes(good + b"tampered")
    with pytest.raises(SystemExit, match="corrupt or tampered"):
        vendor_script.fetch_tarball("abc", expected_sha256=hashlib.sha256(good).hexdigest())
    assert (tmp_path / "tdmpc2-abc.tar.gz").read_bytes() == good + b"tampered"  # not silently replaced


def test_cached_tarball_with_right_hash_is_returned_without_network(vendor_script, tmp_path, monkeypatch):
    monkeypatch.setattr(vendor_script, "CACHE", tmp_path)
    monkeypatch.setattr(vendor_script.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    good = _tar({"r/LICENSE": b"MIT"})
    (tmp_path / "tdmpc2-abc.tar.gz").write_bytes(good)
    assert vendor_script.fetch_tarball("abc", expected_sha256=hashlib.sha256(good).hexdigest()) == good


def test_refetch_flag_replaces_corrupt_cache_atomically(vendor_script, tmp_path, monkeypatch):
    monkeypatch.setattr(vendor_script, "CACHE", tmp_path)
    good = _tar({"r/LICENSE": b"MIT"})

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return good

    monkeypatch.setattr(vendor_script.urllib.request, "urlopen", lambda *a, **k: Resp())
    (tmp_path / "tdmpc2-abc.tar.gz").write_bytes(b"garbage")
    digest = hashlib.sha256(good).hexdigest()
    assert vendor_script.fetch_tarball("abc", expected_sha256=digest, refetch=True) == good
    assert (tmp_path / "tdmpc2-abc.tar.gz").read_bytes() == good
    assert [p.name for p in tmp_path.iterdir()] == ["tdmpc2-abc.tar.gz"]  # no temp file left behind
    # a download that does not match the recorded hash is refused and never cached
    (tmp_path / "tdmpc2-abc.tar.gz").unlink()
    with pytest.raises(SystemExit, match="!= recorded"):
        vendor_script.fetch_tarball("abc", expected_sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_cache_write_is_atomic_via_replace(vendor_script, tmp_path, monkeypatch):
    calls = []
    real = vendor_script.os.replace
    monkeypatch.setattr(vendor_script.os, "replace", lambda a, b: (calls.append((Path(a).name, Path(b).name)), real(a, b))[1])
    vendor_script._write_atomic(tmp_path / "f.tar.gz", b"x")
    assert len(calls) == 1 and calls[0][1] == "f.tar.gz" and calls[0][0] != "f.tar.gz"
    assert (tmp_path / "f.tar.gz").read_bytes() == b"x"


@pytest.mark.parametrize("bad", ["../evil", "r/../../evil", "/etc/passwd", "r/tdmpc2/../../x"])
def test_tar_members_with_traversal_or_absolute_paths_rejected(vendor_script, bad):
    tarball = _tar({"r/LICENSE": b"MIT", bad: b"x"})
    with pytest.raises(SystemExit, match="unsafe tar member"):
        vendor_script.read_upstream(tarball)


def test_read_upstream_accepts_well_formed_tarball(vendor_script):
    members = {"r/LICENSE": b"MIT"}
    members.update({f"r/tdmpc2/{f}": b"x = 1\r\n" for f in vendor_script.COPY_FILES})
    files, lic = vendor_script.read_upstream(_tar(members))
    assert lic == "MIT" and files["tdmpc2.py"] == "x = 1\n"


@pytest.mark.slow
def test_smoke_200_steps_cpu(tmp_path):
    from so101_rl.rl.smoke_tdmpc2 import run

    assert run(200, "cpu", 1, tmp_path) == 0
    (run_dir,) = list((tmp_path / "smoke-tdmpc2").iterdir())
    assert (run_dir / "model.pt").exists()
    assert len(pd.read_csv(run_dir / "metrics.csv")) >= 3
