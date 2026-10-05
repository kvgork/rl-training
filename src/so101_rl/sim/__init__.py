


# Lazy re-exports (F10, FS2): importing so101_rl.sim must not import mujoco, mujoco_warp or warp eagerly.
_EXPORTS = {
    "load_model": "so101_rl.sim.model",
    "MujocoBus": "so101_rl.sim.mujoco_bus",
    "gripper_percent_to_rad": "so101_rl.sim.gripper_map",
    "gripper_rad_to_percent": "so101_rl.sim.gripper_map",
    "gripper_ticks_to_percent": "so101_rl.sim.gripper_map",
    "gripper_percent_to_ticks": "so101_rl.sim.gripper_map",
    "gripper_ticks_to_rad": "so101_rl.sim.gripper_map",
    "gripper_rad_to_ticks": "so101_rl.sim.gripper_map",
    "WarpSim": "so101_rl.sim.warp_model",
    "WarpSizes": "so101_rl.sim.warp_model",
    "ContactBufferOverflow": "so101_rl.sim.warp_model",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'so101_rl.sim' has no attribute {name!r}")
