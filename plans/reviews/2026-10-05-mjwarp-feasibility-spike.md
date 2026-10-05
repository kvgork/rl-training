# MJWarp S0 spike report (RTX 3080, driver 610.57 / CUDA 13.3 driver, toolkit 12.9)

## Env (conda-forge only, one solve, no conflicts)
pixi.toml: python 3.12.*, numpy, mujoco, mujoco-warp, warp-lang, pytorch-gpu, system-requirements cuda=12.
Resolved: mujoco / mujoco-python 3.14.0, mujoco-warp 3.14.0 (noarch; depends mujoco>=3.12.0, warp-lang>=1.15),
warp-lang 1.17.0 (cuda129), pytorch-gpu 2.12.0 (cuda129_mkl), libtorch 2.12.0, cuda-version 12.9, python 3.12.14, numpy 2.5.3.
conda-forge mujoco-warp version tracks mujoco version (3.14.0). Not tested: mujoco 3.13.0 / 3.3.2 with mujoco-warp.
Note repo pins numpy 2.2.* (spike resolved 2.5.3; not tested with 2.2).

## Model (repo load_model, imported verbatim)
integrator Euler (mjINT_EULER), solver Newton, cone pyramidal, iterations 100 / ls 50, dt 0.002, impratio 1.
36 geoms, 13 meshes, nq 13 nv 12 nu 6. 30 mesh geoms, of which 10 are collision meshes (contype=conaffinity=1; arm links) with
vertex counts 9526, 11253, 8800, 9526, 12976, 19720, 10481, 9526, 8652, 26967. Others: plane table, 4 jaw boxes, die box.

## MJWarp
put_model/put_data/step work with full model (collision meshes included, Euler/Newton OK).
One required change: jaw_col geoms have margin=0.0005, MJWarp raises NotImplementedError (box-box/box-mesh pairs with non-zero margin
under MULTICCD default). Fix: margin=0 on jaw_col default (I set geom_margin=0 on the 4 jaw boxes). Disabling MULTICCD instead fails
the same way (NATIVECCD check). Nothing else needed; no mesh simplification needed.
Pitfall: default njmax=64 overflows (nefc up to ~114-174 with random targets; pyramidal cone, condim 4). Use njmax=256 (or switch to elliptic cone).
nconmax=32/world fine (CPU max ncon 18). Overflow only prints a warning, results silently degrade.

## Throughput (random ctrl within ctrlrange resampled every 50 steps, CUDA graph, njmax=256, nconmax=32, dt=0.002)
| N | batch-steps/s | env-steps/s | VRAM delta MiB | max nefc/world seen | max nacon (total/naconmax) |
|---|---|---|---|---|---|
| 256 | 1171 | 299,876 | 439 | 138/256 | 1230/8192 |
| 1024 | 728 | 745,265 | 791 | 132/256 | 4875/32768 |
| 2048 | 530 | 1,085,321 | 1311 | 124/256 | 9662/65536 |
| 4096 | 410 | 1,678,084 | 2359 | 132/256 | 19530/131072 |
nefc/nacon sampled every 50 steps only (lower bound on peaks). No NaNs, no overflow at njmax=256. 8192 not run (not requested).
CPU mujoco single env (same model, random targets, after warmup): 77,525 steps/s (one earlier cold run read 20k). So GPU at N>=1024 ~10-20x one CPU core.
Run-to-run: first pass with njmax=160 gave 1.9M env-steps/s at 4096 (overflowed), so ~5-15% variance/cost from njmax.

## Parity (CPU vs MJWarp float32, 300 steps, same ctrl, margin=0 model)
Sinusoid around rest keyframe: max |dqpos| arm joints 3.4e-7 (N=1), 4.8e-7 (N=8 world 0); die (settling on table) 1.4e-4 (max over all).
Second sequence (home +/- 0.6 rad swing): arm 1.2e-6, die 1.4e-4. Contact counts matched (6 vs 6, 4 vs 4). Arm rarely hit table hard in these
sequences, so grasp-level contact parity is NOT yet tested (needs jaw/die scenario).

## Torch interop
wp.to_torch(dw.qpos): cuda:0, float32, shape (N,13), data_ptr == warp ptr (zero copy); writes visible both directions. Verified.

## Files
pixi.toml, inspect_model.py, t3.py (feature probes), bench.py (usage: bench.py NJMAX NCONMAX), parity.py, cpuefc.py.
