# so101_rl

Goal-conditioned RL stack for one SO-101 arm with a policy-independent safety layer. See
`docs/ARCHITECTURE.md` (architecture), `so101-self-learning-rl-plan.md` (decisions) and
`plans/phase0-implementation-plan.md` (frozen interfaces, tasks).

## Environment

Everything runs through pixi. The NVIDIA driver on the dev host may be down, so acceptance uses the
`cpu` environment: `pixi run -e cpu <task>`.

Solver spike verdict (2026-09-30, `pixi install -e cpu`, linux-64):

| Package | Resolved | Source |
|---|---|---|
| python | 3.12.14 | conda-forge |
| numpy | 2.2.6 | conda-forge |
| pytorch (cpu) | 2.11.0 | conda-forge (`pytorch-cpu`); no PyPI `torch` |
| torchvision | 0.26.0 | conda-forge |
| torchrl | 0.14.0 | PyPI |
| tensordict | 0.14.2 | PyPI |
| lerobot | 0.6.1 (`feetech`, `grpcio-dep`) | PyPI |
| mujoco (`mujoco-python`) | 3.13.0 | conda-forge |
| gymnasium | 1.2.3 | conda-forge |
| stable-baselines3 | 2.9.0 | PyPI |
| opencv | 4.13.0 | conda-forge |

Pins that were needed so the conda solve does not fight lerobot 0.6.1 (pixi pins PyPI packages to the
conda-solved version): `setuptools<82`, `packaging<26`, `opencv<4.14`, `gymnasium<1.3`; plus
`[tool.pixi.system-requirements] cuda = "12"` so the `default` (gpu) environment locks on a host
without a driver. After H1 (driver restored) re-pin `cuda-version` to the driver's CUDA.

## Entry points

`pixi task list` shows all tasks; tasks pointing at modules that later units create fail until then.

Environments: `default` (sim+perception+train+gpu; locked but not installable while the driver is down),
`cpu` (same with the CPU torch build; used for acceptance) and `robot` (perception + CPU torch, for the
servo host process; it needs the `cpu` feature because the default conda `pytorch` solve otherwise
picks a CUDA build that requires `__cuda`).
