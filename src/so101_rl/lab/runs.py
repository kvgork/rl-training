"""Run registry: one directory per run under ``runs/lab/<run_id>/`` and the budget ledger derived from them.

Files per run (all JSON writes are atomic: tmp -> fsync -> ``os.replace`` -> fsync dir):

- ``run.json``      immutable manifest (task, trainer, kind, config, provenance); written LAST at creation, so a
                    directory without it is "not yet created" and is ignored.
- ``status.json``   mutable state (closed set ``STATES``), read-modify-written under ``.lock``.
- ``events.jsonl``  append-only ``{seq, t, kind, data}`` (seq monotonic per run, under ``.lock``).
- ``usage.json``    GPU-h as progress-holding time, attempts (P6).
- ``control.json``  the latest pause/stop request for the running job (written by the API, read by the job).
- ``metrics.csv``, ``heartbeat``, checkpoints: written by the job / trainer.

The ledger (P6) is derived from the run dirs every time; ``ledger.json`` only keeps the attempts high-water mark,
and a recomputation that yields fewer attempts than the mark writes ``HALT`` (a run dir went missing).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from so101_rl.lab.errors import LabError

REPO_ROOT = Path(__file__).resolve().parents[3]
STATES = ("queued", "running", "paused", "stopped", "completed", "failed", "interrupted")
TERMINAL = frozenset({"stopped", "completed", "failed"})  # "paused" and "interrupted" can be resumed
KINDS = ("attempt", "probe", "eval")
RUN_ID_RE = re.compile(r"^[a-z_]+-[a-z0-9_]+-[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
STALL_CAP_S = 30 * 60  # one heartbeat gap counts at most this much progress-holding time (P6)


@dataclass(frozen=True)
class Budget:
    """Per budget key (plan §10 P6). Probes are not attempts and draw from their own pooled GPU-h."""

    max_attempts: int
    gpu_h_max: float
    probe_max_min: float = 10.0
    probe_gpu_h_max: float = 1.0


BUDGETS: dict[str, Budget] = {"reach": Budget(max_attempts=3, gpu_h_max=6.0)}


def default_root() -> Path:
    env = os.environ.get("SO101_LAB_RUNS")
    return Path(env).resolve() if env else REPO_ROOT / "runs" / "lab"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# ----------------------------------------------------------------------------------------------- atomic files
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default


@contextlib.contextmanager
def flocked(path: Path) -> Iterator[None]:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- provenance
def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """git sha (+ dirty flag), config sha256, sim_params version and camera_pose_method. Never touches devices."""
    git_sha, dirty = None, None
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "git_sha": git_sha,
        "git_dirty": dirty,
        "config_sha256": canonical_sha256(config),
        "sim_params_version": _yaml_get(REPO_ROOT / "calib" / "sim_params.yaml", ("meta", "version")),
        "camera_pose_method": _yaml_get(REPO_ROOT / "calib" / "camera.yaml", ("T_base_color", "method")),
    }


def _yaml_get(path: Path, keys: tuple[str, ...]) -> Any:
    try:
        import yaml

        node: Any = yaml.safe_load(path.read_text())
        for k in keys:
            node = node[k]
        return node if isinstance(node, (int, float, str)) else None
    except Exception:  # noqa: BLE001 - provenance is best effort; a missing file is recorded as null
        return None


# ----------------------------------------------------------------------------------------------- run dirs
def new_run_id(trainer: str, task: str) -> str:
    return f"{trainer}-{task}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{secrets.token_hex(2)}"


def run_dir_for(root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise LabError("INVALID_ARGUMENT", "run_id does not match the run id grammar", details={"field": "run_id"})
    d = root / run_id
    if not (d / "run.json").is_file():
        raise LabError("NOT_FOUND", "no such run", details={"run_id": run_id})
    return d


def list_run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir() and (d / "run.json").is_file()), key=lambda d: d.name)


def create_run(
    root: Path,
    *,
    trainer: str,
    task: str,
    kind: str,
    config: dict[str, Any],
    budget_key: str,
    idempotency_key: str,
    extra: dict[str, Any] | None = None,
    request_sha: str | None = None,
) -> tuple[Path, bool]:
    """Create a run dir (``queued``) or replay an earlier one with the same idempotency key.

    Must be called with the registry lock held (see ``registry_lock``). Returns ``(run_dir, replayed)``.
    The idempotency record ``idem/<sha256(key)>`` is written atomically after ``run.json``; the same key with a
    different request is ``IDEMPOTENCY_MISMATCH``.
    """
    req_sha = request_sha or request_sha256(trainer=trainer, task=task, kind=kind, config=config, extra=extra)
    prior = lookup_idempotent(root, idempotency_key, req_sha)
    if prior is not None:
        return prior, True
    idem = _idem_path(root, idempotency_key)
    run_id = new_run_id(trainer, task)
    d = root / run_id
    d.mkdir(parents=False, exist_ok=False)  # O_EXCL semantics for the run dir itself
    t = now_iso()
    atomic_write_json(
        d / "status.json",
        {
            "schema": 1, "run_id": run_id, "state": "queued", "active": False, "segment": 0, "seq": 0,
            "pid": None, "pid_start": None, "unit": None, "launcher": None, "created_at": t, "updated_at": t,
            "heartbeat_at": None, "env_step": 0, "last_ckpt": None, "pause_reason": None, "stop_reason": None,
            "hold": None, "ai_pauses": 0, "error": None,
        },
    )
    atomic_write_json(
        d / "usage.json",
        {"schema": 1, "run_id": run_id, "kind": kind, "budget_key": budget_key, "gpu_h": 0.0, "wall_h": 0.0,
         "attempts": 1 if kind == "attempt" else 0, "segments": [], "updated_at": t},
    )
    append_event(d, "created", kind=kind, trainer=trainer, task=task)
    atomic_write_json(
        d / "run.json",
        {"schema": 1, "run_id": run_id, "trainer": trainer, "task": task, "kind": kind, "config": config,
         "budget_key": budget_key, "created_at": t, "provenance": provenance(config), **(extra or {})},
    )
    # The key is recorded last (atomically, under the registry lock), so a crash before this point leaves no key pointing at a half-made run.
    atomic_write_json(idem, {"run_id": run_id, "request_sha256": req_sha})
    return d, False


def request_sha256(**request: Any) -> str:
    return canonical_sha256(request)


def _idem_path(root: Path, key: str) -> Path:
    if not isinstance(key, str) or not key or len(key) > 128:
        raise LabError("INVALID_ARGUMENT", "idempotency_key must be 1..128 characters")
    d = root / "idem"
    d.mkdir(parents=True, exist_ok=True)
    return d / hashlib.sha256(key.encode()).hexdigest()


def lookup_idempotent(root: Path, key: str, req_sha: str) -> Path | None:
    """The run created earlier with this key and the same request; a different request is a mismatch."""
    prior = read_json(_idem_path(root, key))
    if prior is None:
        return None
    if prior.get("request_sha256") != req_sha:
        raise LabError(
            "IDEMPOTENCY_MISMATCH", "idempotency_key was already used for a different request",
            details={"run_id": prior.get("run_id")}, hint="use a new idempotency_key for a new request",
        )
    return root / prior["run_id"]


@contextlib.contextmanager
def registry_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with flocked(root / ".registry.lock"):
        yield


def read_run(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "run.json", {})


def read_status(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "status.json", {})


def update_status(run_dir: Path, **fields: Any) -> dict[str, Any]:
    """Locked read-modify-write; bumps ``seq`` and ``updated_at``. Unknown states are refused."""
    if "state" in fields and fields["state"] not in STATES:
        raise ValueError(f"state {fields['state']!r} not in {STATES}")
    with flocked(run_dir / ".lock"):
        st = read_status(run_dir)
        st.update(fields)
        st["seq"] = int(st.get("seq", 0)) + 1
        st["updated_at"] = now_iso()
        atomic_write_json(run_dir / "status.json", st)
        return st


def mark_interrupted_if_dead(run_dir: Path, alive: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    """Liveness re-check: an ``active`` run whose process is gone without a final status becomes ``interrupted``.

    Check and set happen under the run lock, so a job that wrote its final status just before exiting is never
    overwritten.
    """
    with flocked(run_dir / ".lock"):
        st = read_status(run_dir)
        if not st.get("active") or alive(st):
            return st
        st.update(state="interrupted", active=False, hold=None, seq=int(st.get("seq", 0)) + 1, updated_at=now_iso())
        atomic_write_json(run_dir / "status.json", st)
    append_event(run_dir, "interrupted", pid=st.get("pid"), unit=st.get("unit"))
    return st


def append_event(run_dir: Path, event: str, /, **data: Any) -> int:
    path = run_dir / "events.jsonl"
    with flocked(run_dir / ".events.lock"):
        seq = 0
        if path.exists():
            with path.open("rb") as f:
                lines = f.read().splitlines()
            for line in reversed(lines):
                with contextlib.suppress(json.JSONDecodeError):
                    seq = int(json.loads(line)["seq"])
                    break
        seq += 1
        torn = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
        with path.open("a") as f:
            if torn:  # a crash mid-line must not swallow the next event
                f.write("\n")
            f.write(json.dumps({"seq": seq, "t": now_iso(), "kind": event, "data": data}, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return seq


def read_events(run_dir: Path, last: int = 20) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines()[-last:]:
        with contextlib.suppress(json.JSONDecodeError):
            out.append(json.loads(line))
    return out


def read_usage(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "usage.json", {})


# ----------------------------------------------------------------------------------------------- ledger (P6)
def ledger(root: Path) -> dict[str, dict[str, float]]:
    """Per budget key: attempts used, GPU-h used by attempts, pooled probe GPU-h, eval GPU-h (recorded only).

    Derived from every run dir; an attempt counts from creation (a refused or crashed attempt still counts).
    """
    out: dict[str, dict[str, float]] = {}
    for d in list_run_dirs(root):
        run, usage = read_run(d), read_usage(d)
        key = run.get("budget_key")
        if not key:
            continue
        e = out.setdefault(key, {"attempts_used": 0, "gpu_h_used": 0.0, "probe_gpu_h_used": 0.0, "eval_gpu_h": 0.0})
        gpu_h = float(usage.get("gpu_h") or 0.0)
        kind = run.get("kind")
        if kind == "attempt":
            if read_status(d).get("launch_failed"):
                continue  # the job never started: not an attempt
            e["attempts_used"] += 1
            e["gpu_h_used"] += gpu_h
        elif kind == "probe":
            e["probe_gpu_h_used"] += gpu_h
        elif kind == "eval":
            e["eval_gpu_h"] += gpu_h
    return out


def check_halt(root: Path) -> None:
    halt = root / "HALT"
    if halt.exists():
        reason = (read_json(halt, {}) or {}).get("reason", "unknown")
        raise LabError(
            "HALTED", "the lab is halted", details={"reason": str(reason)[:200]},
            hint=f"a human must investigate and then delete {halt}",
        )


def checked_ledger(root: Path) -> dict[str, dict[str, float]]:
    """``ledger`` plus the attempts high-water check: fewer attempts than ever seen writes ``HALT`` and raises."""
    led = ledger(root)
    hw_path = root / "ledger.json"
    hw = read_json(hw_path, {}) or {}
    marks = dict(hw.get("attempts_high_water", {}))
    shrunk = {k: (v, int(led.get(k, {}).get("attempts_used", 0))) for k, v in marks.items()
              if int(led.get(k, {}).get("attempts_used", 0)) < int(v)}
    if shrunk:
        atomic_write_json(root / "HALT", {"reason": "ledger_shrink", "keys": shrunk, "t": now_iso()})
        check_halt(root)
    for k, e in led.items():
        marks[k] = max(int(marks.get(k, 0)), int(e["attempts_used"]))
    if marks != hw.get("attempts_high_water"):
        atomic_write_json(hw_path, {"attempts_high_water": marks, "updated_at": now_iso()})
    return led


def check_budget(led: dict[str, dict[str, float]], key: str, budget: Budget, kind: str, est_gpu_h: float) -> None:
    """Refuse a start that would exceed P6. Evals are recorded but never refused for budget."""
    e = led.get(key, {"attempts_used": 0, "gpu_h_used": 0.0, "probe_gpu_h_used": 0.0})
    if kind == "attempt":
        if e["attempts_used"] + 1 > budget.max_attempts:
            raise LabError(
                "BUDGET_EXHAUSTED", "no attempts left for this budget key",
                details={"budget_key": key, "attempts_used": e["attempts_used"], "max_attempts": budget.max_attempts},
                hint="review the earlier attempts with the human; the budget is set in plan §10 P6",
            )
        if e["gpu_h_used"] + est_gpu_h > budget.gpu_h_max + 1e-9:
            raise LabError(
                "BUDGET_EXHAUSTED", "this attempt would exceed the GPU-h budget",
                details={"budget_key": key, "gpu_h_used": round(e["gpu_h_used"], 3), "requested": est_gpu_h,
                         "gpu_h_max": budget.gpu_h_max},
                hint="request a smaller budget_min",
            )
    elif kind == "probe":
        if est_gpu_h * 60 > budget.probe_max_min + 1e-9:
            raise LabError(
                "CONFIG_INVALID", "a probe is limited in wall time",
                details={"budget_min": est_gpu_h * 60, "probe_max_min": budget.probe_max_min},
            )
        if e["probe_gpu_h_used"] + est_gpu_h > budget.probe_gpu_h_max + 1e-9:
            raise LabError(
                "BUDGET_EXHAUSTED", "the pooled probe GPU-h are used up",
                details={"budget_key": key, "probe_gpu_h_used": round(e["probe_gpu_h_used"], 3),
                         "probe_gpu_h_max": budget.probe_gpu_h_max},
            )


def budget_dict(b: Budget) -> dict[str, Any]:
    return asdict(b)
