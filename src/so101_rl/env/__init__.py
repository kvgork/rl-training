"""Goal-conditioned env (F7/FS3/FS4/FS12). Frozen contracts: ``obs`` (layout), ``tasks`` (success/termination) and
``reward`` (modular reward terms). ``backend`` / ``goals`` / ``goal_env``: the real-arm env and its CPU reference
rig (FS12).

Gymnasium registrations (``So101Reach-v0`` -> ``So101GoalEnv(task=REACH)``, ``So101DieToZone-v0`` ->
``So101GoalEnv(task=DIE_TO_ZONE)``). The entry point is :func:`make_goal_env`; the registered ids default to the
CPU reference backend (``backend="sim"``: in-process ServoCore + MujocoBus), never to the real arm: the arm needs
``backend="shm"`` (or a backend object) passed explicitly. Registration is lazy (entry-point strings), so importing
``so101_rl.env.obs`` imports gymnasium (for the registry) but never mujoco, torch or the servo stack.
"""

from __future__ import annotations

from typing import Any

ENV_IDS = {"So101Reach-v0": "reach", "So101DieToZone-v0": "die_to_zone"}


def make_goal_env(
    task: str = "reach",
    *,
    backend: Any = "sim",
    cfg: Any = None,
    calibration: Any = None,
    sim_kwargs: dict | None = None,
    **env_kwargs: Any,
):
    """Build a :class:`~so101_rl.env.goal_env.So101GoalEnv` for ``task`` (``"reach"`` / ``"die_to_zone"``).

    ``backend``: an ``EnvBackend`` object, ``"sim"`` (:class:`~so101_rl.env.backend.SimBackend`, ``sim_kwargs``
    forwarded) or ``"shm"`` (:class:`~so101_rl.env.backend.ShmBackend`: attaches to the running servo process).
    ``cfg`` defaults to the packaged ``so101.yaml``; ``calibration`` to ``cfg.calibration_path``.
    """
    from so101_rl.config import load_calibration, load_config

    from .goal_env import So101GoalEnv
    from .tasks import DIE_TO_ZONE, REACH

    specs = {"reach": REACH, "die_to_zone": DIE_TO_ZONE}
    if task not in specs:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(specs)}")
    if cfg is None:
        if calibration is None:
            calibration = load_calibration(load_config().calibration_path)
        cfg = load_config(calibration=calibration)
    elif calibration is None:
        calibration = load_calibration(cfg.calibration_path)
    if backend == "sim":
        from .backend import SimBackend

        backend = SimBackend(cfg, calibration, **(sim_kwargs or {}))
    elif backend == "shm":
        from .backend import ShmBackend

        backend = ShmBackend(cfg, calibration=calibration)
    elif isinstance(backend, str):
        raise ValueError(f"backend must be 'sim', 'shm' or an EnvBackend, got {backend!r}")
    try:
        return So101GoalEnv(backend, cfg, specs[task], **env_kwargs)
    except BaseException:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
        raise


def _register() -> None:
    try:
        import gymnasium
    except ImportError:  # an env without gymnasium (no train feature) can still use obs/tasks/reward
        return
    for env_id, task in ENV_IDS.items():
        if env_id not in gymnasium.registry:
            gymnasium.register(id=env_id, entry_point="so101_rl.env:make_goal_env", kwargs={"task": task})


_register()
