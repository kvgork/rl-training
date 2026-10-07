"""Live 3D viewer of selected parallel worlds (plan Task V.1, frozen interface FS11 in ``viz/monitor.py``).

Two halves:

* **Publisher** (training process, :class:`LivePublisher`): every ``1/viewer_hz`` s it copies ONLY the selected worlds'
  rows (qpos, qvel, goal, ghost, return, flags) into one packed device tensor, starts one non-blocking device-to-host
  copy into pinned memory, and on the NEXT tick writes the finished copy into a shared-memory seqlock
  (:class:`~so101_rl.servo.shm.SeqlockBlock`). There is no host sync and no wait for the viewer: a copy that is not
  finished yet is simply picked up one tick later, and a missing viewer pauses publishing altogether.
* **Viewer** (a separate process, ``python -m so101_rl.viz.live_viewer view|attach``): a CPU ``MjModel`` built by the
  same ``sim.model.build_spec`` (the scene compiled once per slot, offset along y) and ``mujoco.viewer.launch_passive``.
  It never touches CUDA, so the GUI cannot hold the training process's GIL or its GPU.

World selectors (:func:`parse_selector`): ``"0"`` / any integer = a fixed world; ``"best"`` / ``"worst"`` = by the
current episode return, **re-picked at each episode boundary of the shown world** (device-side, no sync). The
boundary is detected as the shown world's ``episode_step`` decreasing; without an ``episode_step`` tensor the pick is
refreshed every :data:`REPICK_TICKS` publisher ticks instead.

Attach / detach: ``python -m so101_rl.viz.live_viewer attach --run runs/<run>`` drops ``viz/attach.request`` into the
run directory; the training process (polling once a second) then creates the segment and the descriptor
``viz/viewer.json`` if it did not already, and the attach process opens a window on it. Closing the window ends only
the viewer process; the publisher notices (``viz/viewer.pid`` is gone) and pauses.

This module imports ``sim`` and ``env`` read-only and never ``safety``. It never opens a serial device.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing.resource_tracker as _rt
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    import mujoco
    import torch

__all__ = [
    "REPICK_TICKS",
    "LivePublisher",
    "NoDisplayError",
    "SlotLayout",
    "VizContext",
    "WorldPicker",
    "attach_main",
    "build_viewer_model",
    "parse_selector",
    "payload_dtype",
    "require_display",
    "viewer_pid_alive",
]

REPICK_TICKS: Final[int] = 150  # best/worst refresh period (ticks, ~5 s at 30 Hz) when no episode_step is available
SLOT_SPACING_M: Final[float] = 0.70  # distance between the arms of two slots in the viewer scene (along y)
ATTACH_REQUEST: Final[str] = "attach.request"
DESCRIPTOR: Final[str] = "viewer.json"
VIEWER_PID: Final[str] = "viewer.pid"
GRACE_S: Final[float] = 20.0  # keep publishing this long after a viewer was requested (it needs time to start)
PAYLOAD_VERSION: Final[int] = 1


class NoDisplayError(RuntimeError):
    """No ``DISPLAY`` / ``WAYLAND_DISPLAY``: the live viewer cannot open a window."""


def require_display(env: dict[str, str] | None = None) -> None:
    e = os.environ if env is None else env
    if not (e.get("DISPLAY") or e.get("WAYLAND_DISPLAY")):
        raise NoDisplayError(
            "the live 3D viewer needs a display (DISPLAY / WAYLAND_DISPLAY is not set). Run it from a desktop "
            "session, or train without --viewer: rollout videos and TensorBoard work headless."
        )


# --------------------------------------------------------------------------------------------- selectors
def parse_selector(sel: str | int) -> tuple[str, int]:
    """``"best"`` / ``"worst"`` -> (kind, 0); an integer (or its string) -> ("fixed", world id)."""
    if isinstance(sel, bool):
        raise TypeError(f"invalid world selector {sel!r}")
    if isinstance(sel, int):
        if sel < 0:
            raise ValueError(f"world id must be >= 0, got {sel}")
        return "fixed", sel
    s = str(sel).strip().lower()
    if s in ("best", "worst"):
        return s, 0
    try:
        w = int(s)
    except ValueError:
        raise ValueError(f"invalid world selector {sel!r}: use an integer world id, 'best' or 'worst'") from None
    if w < 0:
        raise ValueError(f"world id must be >= 0, got {w}")
    return "fixed", w


class WorldPicker:
    """Maps slot selectors to world ids, entirely on the device (no host sync)."""

    def __init__(self, selectors: Sequence[str | int], nworld: int, device: torch.device | str) -> None:
        import torch

        if not selectors:
            raise ValueError("at least one world selector is needed")
        self.parsed = [parse_selector(s) for s in selectors]
        for kind, w in self.parsed:
            if kind == "fixed" and w >= nworld:
                raise ValueError(f"world {w} outside 0..{nworld - 1}")
        self.nworld = nworld
        self.device = torch.device(device)
        self.idx = torch.tensor([w for _, w in self.parsed], dtype=torch.long, device=self.device)
        self._dyn = torch.tensor([k != "fixed" for k, _ in self.parsed], device=self.device)
        self._is_best = torch.tensor([k == "best" for k, _ in self.parsed], device=self.device)
        self._prev_step = torch.zeros(len(self.parsed), dtype=torch.long, device=self.device)
        self._first = True
        self._since = 0

    @property
    def has_dynamic(self) -> bool:
        return any(k != "fixed" for k, _ in self.parsed)

    def update(self, ep_return: torch.Tensor, episode_step: torch.Tensor | None = None) -> torch.Tensor:
        """Return long[K] world ids for the K slots (a view of internal state: copy it if you keep it)."""
        import torch

        if not self.has_dynamic:
            return self.idx
        r = ep_return.reshape(-1)
        best, worst = torch.argmax(r), torch.argmin(r)
        cand = torch.where(self._is_best, best, worst)
        if self._first:
            repick: Any = True
            self._first = False
        elif episode_step is not None:
            cur = episode_step.reshape(-1)[self.idx].long()
            repick = cur < self._prev_step  # the shown world's episode restarted
        else:
            self._since += 1
            repick = self._since >= REPICK_TICKS
            if repick:
                self._since = 0
        pick = self._dyn & torch.as_tensor(repick, device=self.device)
        self.idx = torch.where(pick, cand, self.idx)
        if episode_step is not None:
            self._prev_step = episode_step.reshape(-1)[self.idx].long()
        return self.idx


# --------------------------------------------------------------------------------------------- payload
@dataclass(frozen=True)
class SlotLayout:
    """Float columns of one slot in the packed payload."""

    nq: int
    nv: int

    # fixed scalar columns first, then goal / ghost, then qpos / qvel
    WORLD = 0
    EP_STEP = 1
    SUCCESS = 2
    RET = 3
    GHOST_VALID = 4
    GOAL = slice(5, 8)
    GHOST = slice(8, 11)
    Q0 = 11

    @property
    def qpos(self) -> slice:
        return slice(self.Q0, self.Q0 + self.nq)

    @property
    def qvel(self) -> slice:
        return slice(self.Q0 + self.nq, self.Q0 + self.nq + self.nv)

    @property
    def width(self) -> int:
        return self.Q0 + self.nq + self.nv


def payload_dtype(slots: int, nq: int, nv: int) -> np.dtype:
    """Seqlock record: ``seq`` first, ``crc32`` last (SeqlockBlock's contract), packed slot floats in between."""
    w = SlotLayout(nq, nv).width
    return np.dtype(
        [
            ("seq", "<u8"),
            ("t_wall", "<f8"),
            ("env_step", "<i8"),
            ("dr_stage", "<i4"),
            ("n_slots", "<i4"),
            ("slots", "<f4", (slots, w)),
            ("crc32", "<u4"),
            ("_pad", "V4"),
        ],
        align=True,
    )


@dataclass
class VizContext:
    """Optional per-world overlay data the training loop may update by reference (no copies are made).

    All tensors are over ALL worlds (leading dim ``nworld``) and live wherever the training state lives.
    """

    dr_stage: int = 0
    episode_step: torch.Tensor | None = None  # int [N]
    goal: torch.Tensor | None = None  # float [N, 3] goal xyz (reach) / zone centre (die tasks), base frame
    tracker_pos: torch.Tensor | None = None  # float [N, 3] the tracker's LAST-SEEN die position (held when occluded)
    tracker_valid: torch.Tensor | None = None  # bool [N] a die position has been seen at all in this episode


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie answers kill(0); /proc tells the truth
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def viewer_pid_alive(viz_dir: Path) -> bool:
    try:
        return _alive(int((viz_dir / VIEWER_PID).read_text().strip()))
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------------------------- publisher
class LivePublisher:
    """Training-side half. ``tick`` is cheap when it is not time yet (one ``monotonic`` call)."""

    def __init__(
        self,
        sim: Any,
        selectors: Sequence[str | int],
        hz: float,
        viz_dir: Path,
        *,
        task: str,
        spawn_viewer: bool,
    ) -> None:
        import torch

        if hz <= 0:
            raise ValueError(f"viewer_hz must be > 0, got {hz}")
        self.sim = sim
        self.device = sim.qpos.device
        self.nworld = int(sim.qpos.shape[0])
        self.layout = SlotLayout(int(sim.qpos.shape[1]), int(sim.qvel.shape[1]))
        self.picker = WorldPicker(selectors, self.nworld, self.device)
        self.k = len(selectors)
        self.dtype = payload_dtype(self.k, self.layout.nq, self.layout.nv)
        self.period = 1.0 / hz
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.published = 0
        self.skipped_pending = 0
        self._next = 0.0
        self._rec = np.zeros(1, dtype=self.dtype)
        self._host = torch.empty((self.k, self.layout.width), dtype=torch.float32, pin_memory=self.device.type == "cuda")
        self._event: Any = torch.cuda.Event() if self.device.type == "cuda" else None
        self._pending: tuple[int, int] | None = None  # (env_step, dr_stage) of the copy in flight
        self._grace_until = 0.0
        self._proc: subprocess.Popen[bytes] | None = None

        self._name = f"so101viz-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=self.dtype.itemsize)
        self._block = _make_block(self._shm.buf, self.dtype)
        desc = {
            "version": PAYLOAD_VERSION,
            "shm": self._name,
            "slots": self.k,
            "nq": self.layout.nq,
            "nv": self.layout.nv,
            "nworld": self.nworld,
            "task": task,
            "selectors": [str(s) for s in selectors],
            "trainer_pid": os.getpid(),
        }
        tmp = self.viz_dir / (DESCRIPTOR + ".tmp")
        tmp.write_text(json.dumps(desc))
        tmp.replace(self.viz_dir / DESCRIPTOR)
        self.request_viewer()
        if spawn_viewer:
            require_display()
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "so101_rl.viz.live_viewer", "view", "--descriptor", str(self.viz_dir / DESCRIPTOR)],
                stdin=subprocess.DEVNULL,
            )

    # ---------------------------------------------------------------- control
    def request_viewer(self) -> None:
        """A viewer is (about to be) running: publish for GRACE_S even before it has written its pid file."""
        self._grace_until = time.monotonic() + GRACE_S

    @property
    def active(self) -> bool:
        return time.monotonic() < self._grace_until or viewer_pid_alive(self.viz_dir)

    def poll_child(self) -> int | None:
        """Exit code of a viewer process this publisher spawned (None while it runs or if there is none)."""
        return None if self._proc is None else self._proc.poll()

    # ---------------------------------------------------------------- publishing
    def tick(
        self,
        env_step: int,
        success: torch.Tensor,
        ep_return: torch.Tensor,
        ctx: VizContext,
        *,
        force: bool = False,
    ) -> bool:
        """Publish if due. True when a copy was started. Never syncs the device."""
        now = time.monotonic()
        if now < self._next and not force:
            return False
        self._next = now + self.period
        self._flush()  # write the previous copy if it has landed
        if self._pending is not None:  # still in flight: do not stack copies
            self.skipped_pending += 1
            return False
        if not self.active:
            return False
        import torch

        idx = self.picker.update(ep_return, ctx.episode_step)
        L = self.layout
        dev = self.device
        buf = torch.zeros((self.k, L.width), dtype=torch.float32, device=dev)
        buf[:, L.WORLD] = idx.to(torch.float32)
        buf[:, L.RET] = ep_return.reshape(-1)[idx].to(torch.float32)
        buf[:, L.SUCCESS] = success.reshape(-1)[idx].to(torch.float32)
        if ctx.episode_step is not None:
            buf[:, L.EP_STEP] = ctx.episode_step.reshape(-1)[idx].to(torch.float32)
        if ctx.goal is not None:
            buf[:, L.GOAL] = ctx.goal[idx].to(torch.float32)
        if ctx.tracker_pos is not None:
            buf[:, L.GHOST] = ctx.tracker_pos[idx].to(torch.float32)
            valid = torch.ones(self.k, device=dev) if ctx.tracker_valid is None else ctx.tracker_valid.reshape(-1)[idx]
            buf[:, L.GHOST_VALID] = valid.to(torch.float32)
        buf[:, L.qpos] = self.sim.qpos[idx]
        buf[:, L.qvel] = self.sim.qvel[idx]
        self._host.copy_(buf, non_blocking=True)
        if self._event is not None:
            self._event.record()
        self._pending = (int(env_step), int(ctx.dr_stage))
        if self._event is None:
            self._flush()
        return True

    def _flush(self) -> None:
        if self._pending is None:
            return
        if self._event is not None and not self._event.query():
            return
        env_step, dr = self._pending
        self._pending = None
        rec = self._rec
        rec["t_wall"] = time.time()
        rec["env_step"] = env_step
        rec["dr_stage"] = dr
        rec["n_slots"] = self.k
        rec["slots"][0] = self._host.numpy()
        self._block.write(rec)
        self.published += 1

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=5)
        self._pending = None
        try:
            self._block.release()
            self._shm.close()
            self._shm.unlink()
        except (OSError, BufferError, FileNotFoundError):
            pass
        for name in (DESCRIPTOR, VIEWER_PID):
            with contextlib.suppress(OSError):
                (self.viz_dir / name).unlink()


def _make_block(buf: memoryview, dtype: np.dtype):
    from so101_rl.servo.shm import SeqlockBlock

    return SeqlockBlock(buf, dtype)


_RT_LOCK = threading.Lock()


def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    """Attach without registering with this process's resource tracker (py3.12 has no ``track=False``): the
    trainer owns the segment, a viewer exiting must not unlink it. Same trick as ``servo.shm``."""
    with _RT_LOCK:
        original = _rt.register
        _rt.register = lambda *_a, **_k: None  # type: ignore[assignment]
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            _rt.register = original


# --------------------------------------------------------------------------------------------- viewer model
def build_viewer_model(slots: int, spacing: float = SLOT_SPACING_M) -> tuple[mujoco.MjModel, list[dict[str, int]]]:
    """The RL scene compiled ``slots`` times in one CPU model, slot ``s`` offset by ``s * spacing`` along y.

    Returns ``(model, adr)`` with, per slot, ``{"qpos": qposadr of its first arm joint, "base": body id}``. Every
    slot's qpos block is the 13 values of one :class:`WarpSim` world (arm 6 + die freejoint 7), in the same order.
    The scene's fixed target-zone sites are made invisible: the viewer draws the per-world goal itself.
    """
    from so101_rl.sim.model import build_spec

    parent = build_spec()
    for s in range(1, slots):
        child = build_spec(camera=False)  # one D435 (slot 0) is enough
        frame = parent.worldbody.add_frame(pos=[0.0, s * spacing, 0.0])
        parent.attach(child, prefix=f"w{s}_", frame=frame)
    model = parent.compile()
    adr: list[dict[str, int]] = []
    for s in range(slots):
        p = "" if s == 0 else f"w{s}_"
        jid = model.joint(f"{p}shoulder_pan").id
        adr.append({"qpos": int(model.jnt_qposadr[jid]), "dof": int(model.jnt_dofadr[jid]), "base": model.body(f"{p}base").id})
    for i in range(model.nsite):
        if model.site(i).name.endswith("zone"):
            model.site_rgba[i, 3] = 0.0
    return model, adr


def _fmt_slot(s: int, row: np.ndarray, L: SlotLayout) -> tuple[str, str]:
    ok = "SUCCESS" if row[L.SUCCESS] > 0.5 else "-"
    return (f"slot {s}  world {int(row[L.WORLD])}", f"step {int(row[L.EP_STEP])}  ret {row[L.RET]:+.2f}  {ok}")


def run_viewer(descriptor: Path, *, hz: float = 60.0) -> int:
    """Viewer process body. Returns the process exit code (0 on a normal window close)."""
    require_display()
    import mujoco
    import mujoco.viewer

    desc = json.loads(Path(descriptor).read_text())
    viz_dir = Path(descriptor).parent
    k, nq, nv = int(desc["slots"]), int(desc["nq"]), int(desc["nv"])
    task = str(desc.get("task", "reach"))
    layout = SlotLayout(nq, nv)
    dtype = payload_dtype(k, nq, nv)
    shm = _attach_untracked(str(desc["shm"]))
    block = _make_block(shm.buf, dtype)
    model, adr = build_viewer_model(k)
    data = mujoco.MjData(model)
    rec = np.zeros(1, dtype=dtype)
    pid_file = viz_dir / VIEWER_PID
    pid_file.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # the trainer terminates us: run the finally below
    trainer_pid = int(desc.get("trainer_pid", 0))
    next_alive_check = 0.0
    try:
        from so101_rl.sim.model import ZONE_EDGE_M

        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as handle:
            handle.cam.lookat[:] = [0.15, (k - 1) * SLOT_SPACING_M / 2, 0.0]
            handle.cam.distance = 0.9 + 0.55 * (k - 1)
            handle.cam.azimuth, handle.cam.elevation = 90.0, -35.0
            last_seq = -1
            period = 1.0 / hz
            while handle.is_running():
                t0 = time.monotonic()
                if t0 >= next_alive_check:
                    next_alive_check = t0 + 1.0
                    if trainer_pid and not _alive(trainer_pid):
                        break  # the training run ended: close the window
                if block.read(rec) and int(rec["seq"][0]) != last_seq:
                    last_seq = int(rec["seq"][0])
                    slots = rec["slots"][0]
                    for s in range(k):
                        row, a, d = slots[s], adr[s]["qpos"], adr[s]["dof"]
                        data.qpos[a : a + nq] = row[layout.qpos]
                        data.qvel[d : d + nv] = row[layout.qvel]
                    mujoco.mj_forward(model, data)
                    with handle.lock():
                        _draw_markers(handle.user_scn, slots, layout, task, ZONE_EDGE_M, k)
                        lines = [_fmt_slot(s, slots[s], layout) for s in range(k)]
                        lines.append((f"env step {int(rec['env_step'][0])}", f"DR stage {int(rec['dr_stage'][0])}"))
                        handle.set_texts(
                            (
                                mujoco.mjtFontScale.mjFONTSCALE_150,
                                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                "\n".join(a for a, _ in lines),
                                "\n".join(b for _, b in lines),
                            )
                        )
                    handle.sync()
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        return 0
    finally:
        with contextlib.suppress(OSError):
            pid_file.unlink()
        block.release()
        shm.close()


def _draw_markers(scn: Any, slots: np.ndarray, L: SlotLayout, task: str, zone_edge: float, k: int) -> None:
    """Goal marker (sphere; flat zone box for the die task), tracker ghost (translucent sphere), success lamp."""
    import mujoco

    scn.ngeom = 0

    def add(gtype: int, size: Sequence[float], pos: Sequence[float], rgba: Sequence[float]) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
        scn.ngeom += 1

    for s in range(k):
        row = slots[s]
        oy = s * SLOT_SPACING_M
        goal = row[L.GOAL]
        if np.any(goal != 0.0):
            gp = (float(goal[0]), float(goal[1]) + oy, float(goal[2]))
            if task == "die_to_zone":
                add(mujoco.mjtGeom.mjGEOM_BOX, (zone_edge / 2, zone_edge / 2, 0.0006), (gp[0], gp[1], gp[2] - 0.0085), (0.1, 0.8, 0.2, 0.5))
            else:
                add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.012, 0, 0), gp, (0.1, 0.9, 0.2, 0.6))
        if row[L.GHOST_VALID] > 0.5:
            g = row[L.GHOST]
            add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.010, 0, 0), (float(g[0]), float(g[1]) + oy, float(g[2])), (0.9, 0.2, 0.9, 0.35))
        ok = row[L.SUCCESS] > 0.5
        add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.015, 0, 0), (0.0, oy, 0.42), (0.1, 0.9, 0.1, 1.0) if ok else (0.6, 0.1, 0.1, 1.0))


# --------------------------------------------------------------------------------------------- CLI
def attach_main(run_dir: Path, *, timeout_s: float = 30.0) -> int:
    """``viz-attach``: ask the running trainer for a feed, then open a window on it."""
    require_display()
    viz = Path(run_dir) / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    (viz / ATTACH_REQUEST).write_text(str(os.getpid()))
    t0 = time.monotonic()
    desc = viz / DESCRIPTOR
    while not desc.exists():
        if time.monotonic() - t0 > timeout_s:
            with contextlib.suppress(OSError):
                (viz / ATTACH_REQUEST).unlink()
            print(f"no training process answered in {timeout_s:.0f} s ({viz}); is the run alive and using a TrainingMonitor?", file=sys.stderr)
            return 1
        time.sleep(0.2)
    return run_viewer(desc)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="so101_rl.viz.live_viewer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view", help="open a window on an existing descriptor")
    v.add_argument("--descriptor", type=Path, required=True)
    a = sub.add_parser("attach", help="attach to a running training run")
    a.add_argument("--run", type=Path, required=True, help="the run directory, e.g. runs/sac/2026-...")
    a.add_argument("--timeout", type=float, default=30.0)
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "view":
            return run_viewer(ns.descriptor)
        return attach_main(ns.run, timeout_s=ns.timeout)
    except NoDisplayError as e:
        print(f"live viewer: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
