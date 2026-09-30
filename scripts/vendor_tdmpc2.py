"""Vendor nicklashansen/tdmpc2 (MIT) into src/so101_rl/_vendor/tdmpc2 with reproducible patches.

    python scripts/vendor_tdmpc2.py [--sha <commit>]   # (re)generate the tree
    python scripts/vendor_tdmpc2.py --check            # rebuild in memory, compare byte-for-byte
    python scripts/vendor_tdmpc2.py --verify-manifest  # OFFLINE: hash every vendored file vs the manifest
    python scripts/vendor_tdmpc2.py --refetch          # discard a corrupt cached tarball and download again

The vendored tree is generated: never edit it by hand. Add a patch to PATCHES below instead, re-run the
script and commit the result. PATCHES.md (generated) lists every changed file as a unified diff.
The upstream tarball is cached in ~/.cache/so101_rl/vendor/ and its sha256 is verified against the
`tarball_sha256` recorded in UPSTREAM.json on every read; a mismatch is an error (never a silent refetch).
scripts/vendor_tdmpc2.manifest.json holds the sha256 of every vendored file so that hand-edits are caught
offline (`--verify-manifest`, tests/test_vendor_tdmpc2.py).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "nicklashansen/tdmpc2"
DEFAULT_SHA = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "so101_rl" / "_vendor" / "tdmpc2"
CACHE = Path.home() / ".cache" / "so101_rl" / "vendor"
MANIFEST = Path(__file__).with_name("vendor_tdmpc2.manifest.json")

# Upstream paths (relative to tdmpc2/tdmpc2/) that are copied. envs/ is deliberately left out.
COPY_FILES = ("__init__.py", "tdmpc2.py", "train.py", "evaluate.py", "config.yaml")
COPY_DIRS = ("common", "trainer")
GENERATED = ("PATCHES.md", "UPSTREAM.json")

T = "\t"

# (file, old, new, description); each `old` must occur exactly once unless count is given.
PATCHES: list[tuple[str, str, str, str]] = [
    # --- device handling (b) ---------------------------------------------------------------------
    (
        "common/__init__.py",
        "TASK_SET = {",
        "def resolve_device(cfg):\n"
        f'{T}"""Device from `cfg.device` ("auto" -> cuda:0 if available else cpu)."""\n'
        f"{T}import torch\n"
        f"{T}dev = cfg.get('device', 'auto') if hasattr(cfg, 'get') else 'auto'\n"
        f"{T}if dev in (None, 'auto'):\n"
        f"{T}{T}dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n"
        f"{T}return torch.device(dev)\n\n\n"
        "TASK_SET = {",
        "add resolve_device(cfg) helper (device from cfg.device, auto-detect)",
    ),
]

PATCHES += [
    (
        "common/buffer.py",
        f"{T}{T}self._device = torch.device('cuda:0')\n",
        f"{T}{T}self._device = resolve_device(cfg)\n",
        "buffer: device from cfg",
    ),
    (
        "common/buffer.py",
        "from tensordict.tensordict import TensorDict\n",
        "from tensordict import TensorDict\nfrom . import resolve_device\n",
        "buffer: public tensordict import; import resolve_device",
    ),
    (
        "common/buffer.py",
        f"{T}{T}mem_free, _ = torch.cuda.mem_get_info()\n",
        f"{T}{T}mem_free = torch.cuda.mem_get_info()[0] if self._device.type == 'cuda' else 0\n",
        "buffer: no CUDA query on CPU",
    ),
    (
        "common/buffer.py",
        f"{T}{T}storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'\n",
        f"{T}{T}storage_device = 'cuda:0' if self._device.type == 'cuda' and 2.5*total_bytes < mem_free else 'cpu'\n",
        "buffer: CPU storage when the device is CPU",
    ),
    (
        "common/scale.py",
        "from torch.nn import Buffer\n",
        "from torch.nn import Buffer\nfrom . import resolve_device\n",
        "scale: import resolve_device",
    ),
    (
        "common/scale.py",
        f"{T}{T}self.value = Buffer(torch.ones(1, dtype=torch.float32, device=torch.device('cuda:0')))\n"
        f"{T}{T}self._percentiles = Buffer(torch.tensor([5, 95], dtype=torch.float32, device=torch.device('cuda:0')))\n",
        f"{T}{T}device = resolve_device(cfg)\n"
        f"{T}{T}self.value = Buffer(torch.ones(1, dtype=torch.float32, device=device))\n"
        f"{T}{T}self._percentiles = Buffer(torch.tensor([5, 95], dtype=torch.float32, device=device))\n",
        "scale: device from cfg",
    ),
    (
        "tdmpc2.py",
        "from .common.scale import RunningScale\n",
        "from .common.scale import RunningScale\nfrom .common import resolve_device\n",
        "agent: import resolve_device",
    ),
    (
        "tdmpc2.py",
        f"{T}{T}self.device = torch.device('cuda:0')\n",
        f"{T}{T}self.device = resolve_device(cfg)\n",
        "agent: device from cfg",
    ),
    (
        "tdmpc2.py",
        "], lr=self.cfg.lr, capturable=True)\n",
        "], lr=self.cfg.lr, capturable=self.device.type == 'cuda')\n",
        "agent: Adam capturable only on CUDA",
    ),
    (
        "tdmpc2.py",
        "eps=1e-5, capturable=True)\n",
        "eps=1e-5, capturable=self.device.type == 'cuda')\n",
        "agent: pi optimizer capturable only on CUDA",
    ),
    (
        "tdmpc2.py",
        "[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'\n",
        "[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device\n",
        "agent: multitask discount on cfg device",
    ),
    (
        "tdmpc2.py",
        f"{T}{T}torch.compiler.cudagraph_mark_step_begin()\n",
        f"{T}{T}if self.cfg.compile:\n{T}{T}{T}torch.compiler.cudagraph_mark_step_begin()\n",
        "agent: cudagraph step marker only when compiling (c)",
    ),
    (
        "trainer/online_trainer.py",
        f"{T}{T}{T}{T}torch.compiler.cudagraph_mark_step_begin()\n",
        f"{T}{T}{T}{T}if self.cfg.compile:\n{T}{T}{T}{T}{T}torch.compiler.cudagraph_mark_step_begin()\n",
        "online trainer: cudagraph step marker only when compiling (c)",
    ),
    (
        "trainer/online_trainer.py",
        "from tensordict.tensordict import TensorDict\n",
        "from tensordict import TensorDict\n",
        "online trainer: public tensordict import",
    ),
    (
        "trainer/offline_trainer.py",
        f"{T}{T}{T}{T}{T}torch.compiler.cudagraph_mark_step_begin()\n",
        f"{T}{T}{T}{T}{T}if self.cfg.compile:\n{T}{T}{T}{T}{T}{T}torch.compiler.cudagraph_mark_step_begin()\n",
        "offline trainer: cudagraph step marker only when compiling (c)",
    ),
    (
        "common/layers.py",
        f"{T}def __len__(self):\n{T}{T}return self._n\n",
        f"{T}def __len__(self):\n{T}{T}return self._n\n\n"
        f"{T}def _apply(self, fn, recurse=True):\n"
        f"{T}{T}# `module` is a stateless meta-device template for functional calls; torch>=2.9 refuses to move\n"
        f"{T}{T}# meta tensors, so hide it from Module._apply (.to/.cuda/.cpu) and move only the real params.\n"
        f"{T}{T}template = self._modules['module']\n"
        f"{T}{T}self._modules['module'] = None\n"
        f"{T}{T}try:\n"
        f"{T}{T}{T}super()._apply(fn, recurse)\n"
        f"{T}{T}finally:\n"
        f"{T}{T}{T}self._modules['module'] = template\n"
        f"{T}{T}return self\n",
        "layers.Ensemble: skip the meta-device template in Module._apply (.to() fails on torch 2.11 otherwise)",
    ),
    # --- parser: no hydra runtime needed (d/e) -------------------------------------------------------
    (
        "common/parser.py",
        "import hydra\nfrom omegaconf import OmegaConf\n",
        "from omegaconf import OmegaConf\n",
        "parser: drop hydra runtime dependency",
    ),
    (
        "common/parser.py",
        f"{T}cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name\n",
        f"{T}if OmegaConf.is_missing(cfg, 'work_dir'):\n"
        f"{T}{T}cfg.work_dir = Path.cwd() / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name\n"
        f"{T}else:\n"
        f"{T}{T}cfg.work_dir = Path(cfg.work_dir)\n",
        "parser: work_dir from cfg (or cwd) instead of hydra.utils.get_original_cwd()",
    ),
    # --- logger: CSV always, wandb guarded (d) -----------------------------------------------------
    (
        "common/logger.py",
        f"{T}{T}self._eval = []\n",
        f"{T}{T}self._eval = []\n{T}{T}self._rows = []\n",
        "logger: row buffer for metrics.csv",
    ),
    (
        "common/logger.py",
        f'{T}{T}if not cfg.enable_wandb or self.project == "none" or self.entity == "none":\n',
        f'{T}{T}if not cfg.enable_wandb or self.project in ("none", "???") or self.entity in ("none", "???"):\n',
        "logger: unset (???) wandb project/entity disables wandb",
    ),
    (
        "common/logger.py",
        f"{T}{T}{T}cfg.save_agent = False\n",
        "",
        "logger: keep save_agent when wandb is disabled (model.pt is a smoke-test artifact)",
    ),
    (
        "common/logger.py",
        f'{T}def log(self, d, category="train"):\n',
        f"{T}def _write_metrics(self, d, category):\n"
        f'{T}{T}"""Always write every logged row to <work_dir>/metrics.csv (with or without wandb)."""\n'
        f"{T}{T}row = {{'category': category}}\n"
        f"{T}{T}for k, v in d.items():\n"
        f"{T}{T}{T}try:\n"
        f"{T}{T}{T}{T}row[k] = float(v)\n"
        f"{T}{T}{T}except (TypeError, ValueError):\n"
        f"{T}{T}{T}{T}row[k] = str(v)\n"
        f"{T}{T}self._rows.append(row)\n"
        f'{T}{T}pd.DataFrame(self._rows).to_csv(self._log_dir / "metrics.csv", index=False)\n\n'
        f'{T}def log(self, d, category="train"):\n',
        "logger: metrics.csv writer",
    ),
    (
        "common/logger.py",
        f'{T}{T}assert category in CAT_TO_COLOR.keys(), f"invalid category: {{category}}"\n',
        f'{T}{T}assert category in CAT_TO_COLOR.keys(), f"invalid category: {{category}}"\n'
        f"{T}{T}self._write_metrics(d, category)\n",
        "logger: call metrics.csv writer",
    ),
    # --- train/evaluate: no CUDA assert, env factory by dotted path (e) ----------------------------
    (
        "train.py",
        f"{T}assert torch.cuda.is_available()\n",
        "",
        "train: CPU allowed",
    ),
    (
        "evaluate.py",
        f"{T}assert torch.cuda.is_available()\n",
        "",
        "evaluate: CPU allowed",
    ),
    (
        "train.py",
        "from .envs import make_env\n",
        "import importlib\n\n\n"
        "def make_env(cfg):\n"
        f'{T}"""Build the env from `cfg.env_factory` ("pkg.module.callable"); no DMControl/ManiSkill import."""\n'
        f"{T}module, _, name = cfg.env_factory.rpartition('.')\n"
        f"{T}return getattr(importlib.import_module(module), name)(cfg)\n\n",
        "train: env from cfg.env_factory dotted path",
    ),
    (
        "evaluate.py",
        "from .envs import make_env\n",
        "import importlib\n\n\n"
        "def make_env(cfg):\n"
        f'{T}"""Build the env from `cfg.env_factory` ("pkg.module.callable")."""\n'
        f"{T}module, _, name = cfg.env_factory.rpartition('.')\n"
        f"{T}return getattr(importlib.import_module(module), name)(cfg)\n\n",
        "evaluate: env from cfg.env_factory dotted path",
    ),
    # --- config.yaml ---------------------------------------------------------------------------------
    (
        "config.yaml",
        "defaults:\n    - override hydra/launcher: submitit_local\n\n",
        "",
        "config: drop the submitit launcher override (plugin not installed)",
    ),
    ("config.yaml", "compile: true\n", "compile: false\n", "config: torch.compile off by default (c)"),
    (
        "config.yaml",
        "# misc\n",
        "# misc\nenv_factory: ???  # dotted path 'pkg.module.callable' -> env(cfg)\ndevice: auto  # auto | cpu | cuda:0\n",
        "config: env_factory and device keys",
    ),
]


def rewrite_imports(rel: str, text: str) -> str:
    """(a) absolute upstream imports -> package-relative imports, by directory depth."""
    d = Path(rel).parts[0] if len(Path(rel).parts) > 1 else ""
    if d == "common":
        text = re.sub(r"^(\s*)from common import ", r"\1from . import ", text, flags=re.MULTILINE)
        text = re.sub(r"^(\s*)from common\.", r"\1from .", text, flags=re.MULTILINE)
    elif d == "trainer":
        text = re.sub(r"^(\s*)from common", r"\1from ..common", text, flags=re.MULTILINE)
        text = re.sub(r"^(\s*)from trainer\.", r"\1from .", text, flags=re.MULTILINE)
    else:
        text = re.sub(r"^(\s*)from (common|trainer|envs)\b", r"\1from .\2", text, flags=re.MULTILINE)
        text = re.sub(r"^(\s*)from tdmpc2 import ", r"\1from .tdmpc2 import ", text, flags=re.MULTILINE)
    return text


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recorded_tarball_sha256(sha: str) -> str | None:
    """``tarball_sha256`` from the committed UPSTREAM.json, if it pins this commit."""
    meta_path = OUT / "UPSTREAM.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return meta.get("tarball_sha256") if meta.get("sha") == sha else None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_tarball(sha: str, *, expected_sha256: str | None = None, refetch: bool = False) -> bytes:
    """Cached upstream tarball. The bytes are verified against ``expected_sha256`` on EVERY read.

    A cached file whose hash differs raises; it is only replaced when ``refetch`` is set. A freshly downloaded
    file must match too. With no expected hash (new pin) the download is trusted once and its hash recorded.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"tdmpc2-{sha}.tar.gz"
    if cached.exists() and not refetch:
        data = cached.read_bytes()
        if expected_sha256 is not None and sha256_hex(data) != expected_sha256:
            raise SystemExit(
                f"cached tarball {cached} has sha256 {sha256_hex(data)}, recorded {expected_sha256}: "
                "corrupt or tampered. Delete it or re-run with --refetch."
            )
        return data
    url = f"https://codeload.github.com/{REPO}/tar.gz/{sha}"
    req = urllib.request.Request(url, headers={"User-Agent": "so101-rl-vendor"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if expected_sha256 is not None and sha256_hex(data) != expected_sha256:
        raise SystemExit(f"downloaded tarball sha256 {sha256_hex(data)} != recorded {expected_sha256}")
    _write_atomic(cached, data)
    return data


def safe_member_parts(name: str) -> tuple[str, ...]:
    """Path parts of a tar member name; absolute paths and ``..`` components are refused."""
    path = Path(name)
    if path.is_absolute() or name.startswith(("/", "\\")) or ".." in path.parts or ".." in name.split("\\"):
        raise SystemExit(f"unsafe tar member name {name!r}")
    return path.parts


def read_upstream(tarball: bytes) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    license_text = ""
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:  # validate every member first, wanted or not
            safe_member_parts(member.name)
        for member in members:
            if not member.isfile():
                continue
            parts = safe_member_parts(member.name)[1:]  # strip <repo>-<sha>/
            if not parts:
                continue
            rel = "/".join(parts)
            sub = "/".join(parts[1:]) if parts[0] == "tdmpc2" else ""
            wanted = sub in COPY_FILES or sub.split("/")[0] in COPY_DIRS
            if rel == "LICENSE":
                license_text = tar.extractfile(member).read().decode()
            elif sub and wanted:
                files[sub] = tar.extractfile(member).read().decode().replace("\r\n", "\n")
    missing = [f for f in COPY_FILES if f not in files]
    if missing or not license_text:
        raise SystemExit(f"upstream layout changed: missing {missing or 'LICENSE'}")
    return files, license_text


def build(sha: str, *, refetch: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    """Returns (upstream files, patched files) keyed by path relative to the vendored dir."""
    tarball = fetch_tarball(sha, expected_sha256=recorded_tarball_sha256(sha), refetch=refetch)
    upstream, license_text = read_upstream(tarball)
    upstream["LICENSE"] = license_text
    patched = {rel: rewrite_imports(rel, text) for rel, text in upstream.items()}
    used = set()
    for rel, old, new, desc in PATCHES:
        text = patched[rel]
        if text.count(old) != 1:
            raise SystemExit(f"patch anchor for {rel!r} ({desc}) matches {text.count(old)} times, expected 1")
        patched[rel] = text.replace(old, new)
        used.add(rel)
    patched["UPSTREAM.json"] = (
        json.dumps(
            {
                "repo": f"https://github.com/{REPO}",
                "sha": sha,
                "license": "MIT",
                "tarball_sha256": sha256_hex(tarball),
                "copied": sorted(upstream),
                "excluded": ["envs/"],
                "generated_by": "scripts/vendor_tdmpc2.py",
            },
            indent=2,
        )
        + "\n"
    )
    patched["PATCHES.md"] = patches_md(sha, upstream, patched)
    return upstream, patched


def patches_md(sha: str, upstream: dict[str, str], patched: dict[str, str]) -> str:
    lines = [
        "# Patches applied to upstream TD-MPC2",
        "",
        f"Upstream: https://github.com/{REPO} @ `{sha}` (MIT, see `LICENSE`).",
        "Generated by `scripts/vendor_tdmpc2.py`; do not edit by hand. Re-run the script to reproduce this",
        "tree byte-for-byte (`--check` verifies it).",
        "",
        "## Patch list",
        "",
        "1. **Imports**: upstream absolute imports (`from common.`, `from trainer.`, `from tdmpc2 import`,",
        "   `from envs import`) become package-relative imports.",
        "0. Line endings: CRLF in upstream `common/logger.py` normalised to LF (diffs below are against the",
        "   normalised upstream).",
    ]
    lines += [f"{i}. `{rel}`: {desc}" for i, (rel, _, _, desc) in enumerate(PATCHES, start=2)]
    lines += ["", "## Diffs (upstream -> vendored)", ""]
    for rel in sorted(upstream):
        diff = list(
            difflib.unified_diff(
                upstream[rel].splitlines(keepends=True),
                patched[rel].splitlines(keepends=True),
                fromfile=f"upstream/{rel}",
                tofile=f"vendored/{rel}",
                n=1,
            )
        )
        if diff:
            lines += ["### " + rel, "", "```diff", "".join(d if d.endswith("\n") else d + "\n" for d in diff) + "```", ""]
    return "\n".join(lines)


def write(patched: dict[str, str]) -> None:
    for rel, text in patched.items():
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    write_manifest(current_tree())


def current_tree() -> dict[str, str]:
    return {
        p.relative_to(OUT).as_posix(): p.read_text()
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def manifest_of(tree: dict[str, str]) -> dict[str, str]:
    return {rel: sha256_hex(text.encode()) for rel, text in sorted(tree.items())}


def write_manifest(tree: dict[str, str]) -> None:
    MANIFEST.write_text(json.dumps(manifest_of(tree), indent=2, sort_keys=True) + "\n")


def verify_manifest() -> list[str]:
    """OFFLINE: differences between the vendored tree on disk and the committed manifest (empty = clean)."""
    want = json.loads(MANIFEST.read_text())
    have = manifest_of(current_tree())
    return sorted(set(want) ^ set(have)) + sorted(k for k in have if k in want and have[k] != want[k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sha", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify-manifest", action="store_true")
    ap.add_argument("--refetch", action="store_true", help="replace a cached tarball that fails its hash check")
    args = ap.parse_args()
    if args.verify_manifest:
        diff = verify_manifest()
        print("vendored tree matches the manifest" if not diff else f"MANIFEST MISMATCH: {diff}")
        return 1 if diff else 0
    if args.check:
        sha = json.loads((OUT / "UPSTREAM.json").read_text())["sha"]
        _, patched = build(sha, refetch=args.refetch)
        have = current_tree()
        diff = sorted(set(have) ^ set(patched)) + sorted(k for k in patched if k in have and have[k] != patched[k])
        print("vendored tree matches" if not diff else f"MISMATCH: {diff}")
        return 1 if diff else 0
    sha = args.sha or DEFAULT_SHA
    _, patched = build(sha, refetch=args.refetch)
    write(patched)
    (OUT.parent / "__init__.py").touch()
    print(f"vendored {len(patched)} files at {sha} into {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
