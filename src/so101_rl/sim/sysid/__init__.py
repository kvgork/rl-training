"""System identification for the sim-first track (FS7, plan S2 "Sysid gate").

Modules:

* ``params``: :class:`SimParams` / :class:`SimParamsMeta` / :class:`BatchedSimParams` and ``calib/sim_params.yaml``.
* ``excitation``: excitation plans (multisine, chirp, step, ramp, random, gripper) as 10 Hz pre-clamp deltas, and
  the kinematic in-box preview.
* ``run_excitation``: the runner (``--backend sim`` drives a ``servo --backend mujoco`` child over shm;
  ``--backend real`` is hardware-gated: ``--attended`` plus the H2/H3/H5/H7 artifacts) and the in-process
  ``synthetic`` rig used for ground-truth recovery tests.
* ``dataset``: :class:`ExcitationDataset` (30 Hz tick-log segments of one run), time/kind held-out splits.
* ``replay``: multiple-shooting replay on CPU MuJoCo and MuJoCo Warp, ``rms_pct``.
* ``fit``: the CMA-ES fit (``pixi run sysid-fit``).

Imports are lazy, like ``so101_rl.sim``: importing this package loads neither mujoco nor torch nor warp.
"""

from __future__ import annotations

_EXPORTS = {
    "SIM_PARAMS_VERSION": "so101_rl.sim.sysid.params",
    "SimParams": "so101_rl.sim.sysid.params",
    "SimParamsMeta": "so101_rl.sim.sysid.params",
    "BatchedSimParams": "so101_rl.sim.sysid.params",
    "ExcitationPlan": "so101_rl.sim.sysid.excitation",
    "Segment": "so101_rl.sim.sysid.excitation",
    "default_plan": "so101_rl.sim.sysid.excitation",
    "generate_deltas": "so101_rl.sim.sysid.excitation",
    "check_plan_in_box": "so101_rl.sim.sysid.excitation",
    "ExcitationDataset": "so101_rl.sim.sysid.dataset",
    "TickSegment": "so101_rl.sim.sysid.dataset",
    "replay_segment_cpu": "so101_rl.sim.sysid.replay",
    "replay_segments_warp": "so101_rl.sim.sysid.replay",
    "rms_pct": "so101_rl.sim.sysid.replay",
    "FitResult": "so101_rl.sim.sysid.fit",
    "fit_sim_params": "so101_rl.sim.sysid.fit",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'so101_rl.sim.sysid' has no attribute {name!r}")
