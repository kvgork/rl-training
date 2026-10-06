"""Training visualisation (sim-track unit SUV, frozen interface FS11).

``TrainingMonitor`` / ``VizConfig`` (monitor.py) are the interface the training loops call; ``live_viewer`` (a
separate process), ``video`` (rollout mp4s) and TensorBoard scalars sit behind it. Importing this package must stay
cheap (no torch / mujoco at import): everything is re-exported lazily.
"""

_EXPORTS = {
    "TrainingMonitor": "so101_rl.viz.monitor",
    "VizConfig": "so101_rl.viz.monitor",
    "VizContext": "so101_rl.viz.live_viewer",
    "RolloutEnv": "so101_rl.viz.video",
    "RolloutStatus": "so101_rl.viz.video",
    "record_rollout": "so101_rl.viz.video",
}

__all__ = ["RolloutEnv", "RolloutStatus", "TrainingMonitor", "VizConfig", "VizContext", "record_rollout"]


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'so101_rl.viz' has no attribute {name!r}")
