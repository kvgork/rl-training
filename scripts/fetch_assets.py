"""Fetch the official SO-101 MJCF assets from TheRobotStudio/SO-ARM100 at a pinned commit.

Files land unmodified in assets/so101/upstream/ and their sha256 go to assets/so101/UPSTREAM.json.
Overrides live in assets/so101/so101_rl_scene.xml, never in upstream/.

    python scripts/fetch_assets.py [--sha <commit>] [--check]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "TheRobotStudio/SO-ARM100"
DEFAULT_SHA = "5f6d2b876a53a4872e405b991dd925556c9e38a4"
ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "assets" / "so101"
UPSTREAM_DIR = ASSET_DIR / "upstream"
SIM_PREFIX = "Simulation/SO101/"
FILES = ("so101_new_calib.xml", "scene.xml", "joints_properties.xml")
FETCH_ON_DEMAND_LIMIT_MB = 25


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "so101-rl-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def list_stl(sha: str) -> list[str]:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("git tree listing truncated")
    return sorted(
        e["path"]
        for e in tree["tree"]
        if e["path"].startswith(SIM_PREFIX + "assets/") and e["path"].endswith(".stl")
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(sha: str) -> dict:
    remote = [SIM_PREFIX + f for f in FILES] + list_stl(sha) + ["LICENSE"]
    files: dict[str, str] = {}
    for rel in remote:
        data = _get(f"https://raw.githubusercontent.com/{REPO}/{sha}/{rel}")
        # Simulation/SO101/x -> upstream/x ; LICENSE -> assets/so101/LICENSE (+ upstream/LICENSE)
        local = UPSTREAM_DIR / (rel.removeprefix(SIM_PREFIX))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        files[local.relative_to(UPSTREAM_DIR).as_posix()] = sha256(data)
        if rel == "LICENSE":
            (ASSET_DIR / "LICENSE").write_bytes(data)
    meta = {
        "repo": f"https://github.com/{REPO}",
        "sha": sha,
        "fetched": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "license": "Apache-2.0",
        "files": dict(sorted(files.items())),
    }
    (ASSET_DIR / "UPSTREAM.json").write_text(json.dumps(meta, indent=2) + "\n")
    (ASSET_DIR / "NOTICE").write_text(
        "SO-101 model (c) TheRobotStudio, Apache-2.0 (https://github.com/TheRobotStudio/SO-ARM100).\n"
        f"Fetched at commit {sha}.\n"
        "Modified: none (overrides live in so101_rl_scene.xml).\n"
    )
    total_mb = sum(p.stat().st_size for p in UPSTREAM_DIR.rglob("*") if p.is_file()) / 1e6
    print(f"fetched {len(files)} files ({total_mb:.1f} MB) at {sha}")
    if total_mb > FETCH_ON_DEMAND_LIMIT_MB:
        print(f"WARNING: > {FETCH_ON_DEMAND_LIMIT_MB} MB; switch to fetch-on-demand and gitignore upstream/")
    return meta


def check() -> int:
    meta = json.loads((ASSET_DIR / "UPSTREAM.json").read_text())
    bad = [
        rel
        for rel, digest in meta["files"].items()
        if not (UPSTREAM_DIR / rel).is_file() or sha256((UPSTREAM_DIR / rel).read_bytes()) != digest
    ]
    print("ok" if not bad else f"mismatch: {bad}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sha", default=DEFAULT_SHA)
    ap.add_argument("--check", action="store_true", help="verify local files against UPSTREAM.json")
    args = ap.parse_args()
    if args.check:
        return check()
    fetch(args.sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
