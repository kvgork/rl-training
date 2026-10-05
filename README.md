# so101_rl

Goal-conditioned RL stack for one SO-101 arm with a policy-independent safety layer. See
`docs/ARCHITECTURE.md` (architecture), `so101-self-learning-rl-plan.md` (decisions) and
`plans/phase0-implementation-plan.md` (frozen interfaces, tasks).

## Environment

Everything runs through pixi. The NVIDIA driver works on the dev host since H1 (2026-10-05: RTX 3080, driver
610.57), so both `pixi run -e cpu <task>` (CPU torch, CI and hardware-free acceptance) and `pixi run <task>` (the
`default` env, CUDA torch + MuJoCo Warp) are used. GPU tests: `pixi run test-gpu`.

Solved versions (2026-10-05, S0.1 of `plans/sim-track-implementation-plan.md`; `pixi install -e cpu`, `-e default`,
`-e robot`, linux-64):

| Package | `default` | `cpu` | `robot` | Source |
|---|---|---|---|---|
| python | 3.12.13 | 3.12.14 | 3.12.14 | conda-forge |
| numpy | 2.2.6 | 2.2.6 | 2.2.6 | conda-forge |
| pytorch | 2.11.0 (`cuda129_mkl`, `pytorch-gpu`) | 2.11.0 (`cpu_mkl`, `pytorch-cpu`) | 2.11.0 (`cpu_mkl`) | conda-forge; no PyPI `torch` |
| cuda-version | 12.9 | 12.9 (pulled by `warp-lang`; CPU torch) | — | conda-forge |
| torchvision | 0.26.0 (cuda129) | 0.26.0 (cpu) | 0.26.0 (cpu) | conda-forge |
| mujoco / `mujoco-python` | 3.14.0 | 3.14.0 | — | conda-forge |
| mujoco-warp | 3.14.0 | 3.14.0 | — | conda-forge |
| warp-lang | 1.17.0 (cuda129) | 1.17.0 (cuda129) | — | conda-forge |
| torchrl / tensordict | 0.14.0 / 0.14.2 | 0.14.0 / 0.14.2 | — | PyPI |
| stable-baselines3 | 2.9.0 | 2.9.0 | — | PyPI |
| cma | 4.5.0 | 4.5.0 | — | PyPI |
| lerobot | 0.6.1 (`feetech`, `grpcio-dep`) | 0.6.1 | 0.6.1 | PyPI |
| gymnasium | 1.2.3 | 1.2.3 | 1.2.3 | conda-forge |
| opencv | 4.13.0 | 4.13.0 | 4.13.0 | conda-forge |

Pin rationale: **lerobot 0.6.1 needs torch < 2.12 and numpy < 2.3**, so every env pins `pytorch 2.11.*` and
`numpy 2.2.*` (the MuJoCo Warp spike resolved torch 2.12 / numpy 2.5, which lerobot cannot use). `mujoco-python 3.14.*`,
`mujoco-warp 3.14.*` and `warp-lang >=1.17,<2` sit in the `sim` feature so `cpu` and `default` simulate the same
physics (until 2026-10-05 `cpu` had mujoco 3.13 / torch 2.11 and `default` 3.3.2 / torch 2.7.1). `cuda-version 12.9.*`
matches the cuda129 `warp-lang` build (decision D-CUDA; the documented fallback is 12.8 / 12.6 if a future solve
refuses 12.9 with torch 2.11 — not needed on 2026-10-05). Further pins so the conda solve does not fight lerobot
0.6.1 (pixi pins PyPI packages to the conda-solved version): `setuptools<82`, `packaging<26`, `opencv<4.14`,
`gymnasium<1.3`; plus `[tool.pixi.system-requirements] cuda = "12"` so the `default` environment also locks on a host
without a driver. `tests/test_env_sanity.py` enforces one conda torch 2.11, numpy 2.2, mujoco 3.14 and an importable
`mujoco_warp` / `warp`.

Environments: `default` (sim+perception+train+gpu; the GPU training env),
`cpu` (same with the CPU torch build; used for acceptance) and `robot` (perception + CPU torch, for the
servo host process; it needs the `cpu` feature because the default conda `pytorch` solve otherwise
picks a CUDA build that requires `__cuda`).

Spike verdict for TD-MPC2 (vendored at `nicklashansen/tdmpc2` `e9f5932`, patched by
`scripts/vendor_tdmpc2.py`): the state smoke train runs on CPU with the versions above
(`pixi run -e cpu smoke-tdmpc2 --steps 1000 --device cpu`, about 2 minutes). Two upstream assumptions
break on torch 2.11 and are patched (see `src/so101_rl/_vendor/tdmpc2/PATCHES.md`): hard-coded
`cuda:0` devices and moving the meta-device Q-function template with `Module.to()`.

## Entry points

- `pixi run -e cpu test` runs the suite without hardware or GPU; `pixi run -e cpu ci` adds lint. `pixi run test` runs the
  same selection in the `default` env; `pixi run test-gpu` runs the `gpu`-marked tests (CUDA); `pixi run ci-gpu` = lint +
  test + test-gpu.
- `pixi run -e cpu fetch-assets` / `extract-kinematics` regenerate `assets/so101/upstream/` and
  `src/so101_rl/config/so101_kinematics.yaml` (pinned commit in `assets/so101/UPSTREAM.json`).
- `pixi run -e cpu smoke-tdmpc2` state-only TD-MPC2 smoke train on the toy env.
- `pixi run -e cpu servo-fake --max-ticks 300` runs the servo process on the fake backend (a FakeMcu that boots
  latched, a simulated button press, and a FakeBus powered through the MCU's relay, so the real-arm start order
  runs: port without handshake, button + ARM, ping, re-energise). `pixi run servo --backend {feetech,fake,mujoco} ...` is the
  general CLI (`--heartbeat-port`, `--require-rt`, `--tick-log`, `--summary`, ...; see `docs/ARCHITECTURE.md` §3).
  The feetech backend drives the real arm: only attended, on the hardware track.
- `pixi run -e cpu soak-fake --minutes 0.1` runs a fake-backend soak (servo child + random-policy client, with the
  repository calibration, so it works on a clean host) and writes `runs/soak-<ts>/report.json`. `pixi run soak
  --backend B --minutes M [--calibration PATH]` is the general form. The gate is strict only when the servo got
  real-time scheduling (`rt_active`); a strict pass needs at least 30 minutes. A run with too few samples or policy
  steps, or one cut short, fails on any host.

`pixi task list` shows all tasks; tasks pointing at modules that later units create fail until then.
