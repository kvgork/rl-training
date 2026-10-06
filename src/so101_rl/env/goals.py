"""The ONE reach-goal sampler of the real-arm env AND the batched sim env (FS12 ``BoxGoalSampler``, FS3/FS4).

The bounds are the task's :class:`~so101_rl.env.tasks.GoalBounds` (``REACH.goal_bounds``): the workspace box
shrunk by ``margin_m``, z above the kinematic floor and at most ``radius_max_m`` from the base. On top of the
bounds the sampler rejects goals the arm cannot reach: a reachability table is built once from the FK of
uniformly sampled joint configurations inside the joint limits (voxels of ``voxel_m``; a goal is kept when its
voxel holds at least one FK sample). The table is deterministic in (bounds, limits, chain, seed) and cached per
process, so ``So101GoalEnv`` and ``BatchedGoalEnv`` built from the same config share one table object.

Two front ends over the same set (integration 2026-10-06; SU1's interim torch voxel grid is gone):

* numpy: ``sample`` / ``sample_n`` (rejection sampling with the env's ``np_random``) -- the real / reference env;
* torch: ``sample_torch`` (a fixed number of candidates per world, no host sync, fallback goal when none is
  accepted) -- the batched env. ``reachable_torch`` is the same voxel lookup as ``reachable``, bit for bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import numpy as np

from so101_rl.config import JointLimits
from so101_rl.safety.fk import N_ARM_JOINTS, So101FK

from .tasks import GoalBounds

if TYPE_CHECKING:
    import torch

__all__ = ["FALLBACK_GOAL", "BoxGoalSampler", "FixedGoalSampler", "GoalSampler", "fk_tip_batch"]

REACH_TABLE_SAMPLES: Final[int] = 500_000
VOXEL_M: Final[float] = 0.01
TORCH_CANDIDATES: Final[int] = 32  # per world and call of ``sample_torch``
FALLBACK_GOAL: Final[tuple[float, float, float]] = (0.20, 0.0, 0.15)  # FK-reachable, inside REACH.goal_bounds
_KEY_OFFSET: Final[int] = 1 << 20
_TABLE_CACHE: dict[tuple, np.ndarray] = {}  # reachability tables are deterministic in their inputs


@runtime_checkable
class GoalSampler(Protocol):
    """F7: ``sample(rng)`` -> xyz goal, base frame, metres (float64[3])."""

    def sample(self, rng: np.random.Generator) -> np.ndarray: ...


class FixedGoalSampler:
    """Always the same goal (the die task's zone centre, scripted tests)."""

    def __init__(self, goal: np.ndarray) -> None:
        self.goal = np.asarray(goal, dtype=np.float64).reshape(3).copy()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.goal.copy()


def fk_tip_batch(fk: So101FK, ticks: np.ndarray) -> np.ndarray:
    """Vectorised ``fk.gripper_xyz_from_ticks`` for ``ticks`` int[N, 6] -> float64[N, 3] (same chain, same signs;
    cross-checked against the scalar FK on the first rows, so it cannot drift from the safety FK)."""
    ticks = np.asarray(ticks)
    q = fk.ticks_to_rad(ticks.astype(np.float64), np.empty(ticks.shape))
    chain = fk._chain
    a = np.broadcast_to(chain.base_T, (len(q), 4, 4)).copy()
    rz = np.zeros((len(q), 4, 4))
    rz[:, 2, 2] = rz[:, 3, 3] = 1.0
    for i in range(N_ARM_JOINTS):
        c, s = np.cos(q[:, i]), np.sin(q[:, i])
        rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1] = c, -s, s, c
        a = a @ chain.parent_T[i] @ rz
    tip = chain.tip_T[:3, 3]
    out = np.einsum("nij,j->ni", a[:, :3, :3], tip) + a[:, :3, 3]
    ref = np.empty(3)
    for k in range(min(4, len(q))):
        if not np.allclose(fk.gripper_xyz_from_ticks(ticks[k], ref), out[k], atol=1e-9):
            raise AssertionError("batched FK disagrees with So101FK")
    return out


class BoxGoalSampler:
    """Rejection sampling inside ``bounds`` (``GoalBounds.contains_np``) intersected with the FK-reachable set."""

    def __init__(
        self,
        bounds: GoalBounds,
        *,
        fk: So101FK,
        limits: JointLimits,
        seed: int,
        table_samples: int = REACH_TABLE_SAMPLES,
        voxel_m: float = VOXEL_M,
    ) -> None:
        self.bounds = bounds
        self.voxel_m = float(voxel_m)
        lo = np.asarray(limits.min_ticks, dtype=np.int64)
        hi = np.asarray(limits.max_ticks, dtype=np.int64)
        chain = fk._chain
        ck = (
            repr(bounds), lo.tobytes(), hi.tobytes(), int(seed), int(table_samples), self.voxel_m,
            fk._signs.tobytes(), chain.base_T.tobytes(), chain.parent_T.tobytes(), chain.tip_T.tobytes(),
        )
        keys = _TABLE_CACHE.get(ck)
        if keys is None:
            rng = np.random.default_rng(seed)
            ticks = rng.integers(lo, hi + 1, size=(int(table_samples), len(lo)))
            xyz = fk_tip_batch(fk, ticks)
            keys = np.unique(self._key(xyz[bounds.contains_np(xyz)]))
            if keys.size == 0:
                raise ValueError("no FK-reachable point lies inside the goal bounds")
            keys.flags.writeable = False
            _TABLE_CACHE[ck] = keys
        self._keys = keys
        self._keys_t: dict[str, torch.Tensor] = {}
        fb = np.asarray(FALLBACK_GOAL)[None]
        if not (bounds.contains_np(fb).all() and self.reachable(fb).all()):
            raise ValueError(f"FALLBACK_GOAL {FALLBACK_GOAL} is outside these goal bounds or not FK-reachable")

    @property
    def keys(self) -> np.ndarray:
        """Sorted int64 voxel keys of the reachable set (read-only; shared through the per-process cache)."""
        return self._keys

    def _key(self, xyz: np.ndarray) -> np.ndarray:
        idx = np.floor(np.asarray(xyz, np.float64) / self.voxel_m).astype(np.int64) + _KEY_OFFSET
        return (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]

    def reachable(self, xyz: np.ndarray) -> np.ndarray:
        """bool[...]: the voxel of ``xyz`` holds an FK sample."""
        k = self._key(xyz)
        pos = np.clip(np.searchsorted(self._keys, k), 0, self._keys.size - 1)
        return self._keys[pos] == k

    def sample_n(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` goals, float64[n, 3]."""
        out = np.empty((0, 3))
        for _ in range(1000):
            if len(out) >= n:
                return out[:n]
            cand = self.bounds.sample_np(max(2 * (n - len(out)), 64), rng)
            out = np.concatenate([out, cand[self.reachable(cand)]])
        if len(out) >= n:
            return out[:n]
        raise RuntimeError("goal sampler: the reachable part of the bounds is too small to sample from")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.sample_n(1, rng)[0]

    # ------------------------------------------------------------------ torch front end (batched env)
    def _keys_on(self, device: torch.device) -> torch.Tensor:
        import torch

        k = self._keys_t.get(str(device))
        if k is None:
            k = self._keys_t[str(device)] = torch.as_tensor(self._keys, dtype=torch.int64, device=device)
        return k

    def reachable_torch(self, xyz: torch.Tensor) -> torch.Tensor:
        """bool[...] -- ``reachable`` on a torch tensor (float64 math, the same keys; no host sync)."""
        import torch

        keys = self._keys_on(xyz.device)
        idx = torch.floor(xyz.double() / self.voxel_m).to(torch.int64) + _KEY_OFFSET
        k = (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]
        pos = torch.searchsorted(keys, k.reshape(-1)).clamp_(0, keys.numel() - 1).reshape(k.shape)
        return keys[pos] == k

    def contains_torch(self, xyz: torch.Tensor) -> torch.Tensor:
        """bool[...] -- ``bounds.contains_np`` on a torch tensor (float64 math)."""
        import torch

        x = xyz.double()
        lo = torch.as_tensor(self.bounds.lo, dtype=torch.float64, device=x.device)
        hi = torch.as_tensor(self.bounds.hi, dtype=torch.float64, device=x.device)
        return ((x >= lo) & (x <= hi)).all(dim=-1) & (x.norm(dim=-1) <= self.bounds.radius_max_m)

    def sample_torch(
        self, n: int, generator: torch.Generator, *, candidates: int = TORCH_CANDIDATES, device: torch.device | None = None
    ) -> torch.Tensor:
        """``n`` goals float32[n, 3] on ``generator``'s device: per row the first of ``candidates`` uniform draws in
        the bounds' box that lies in the bounds AND in the reachable set (tested on the float32-rounded value
        that is returned), else :data:`FALLBACK_GOAL`. Fixed work, no host sync (the batched env's reset)."""
        import torch

        if candidates < 1:
            raise ValueError(f"sample_torch: candidates must be >= 1, got {candidates}")
        dev = torch.device(device) if device is not None else generator.device
        if dev.type != generator.device.type:
            raise ValueError(f"sample_torch: generator on {generator.device}, samples requested on {dev}")
        lo = torch.as_tensor(self.bounds.lo, dtype=torch.float64, device=dev)
        hi = torch.as_tensor(self.bounds.hi, dtype=torch.float64, device=dev)
        u = torch.rand((n, candidates, 3), generator=generator, device=dev, dtype=torch.float64)
        cand = (lo + u * (hi - lo)).float()
        c64 = cand.double()
        ok = self.contains_torch(c64) & self.reachable_torch(c64)
        first = torch.argmax(ok.to(torch.int8), dim=1)
        pick = cand[torch.arange(n, device=dev), first]
        fb = torch.tensor(FALLBACK_GOAL, dtype=torch.float32, device=dev)
        return torch.where(ok.any(dim=1)[:, None], pick, fb)
