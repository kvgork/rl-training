"""TD-MPC2 state smoke train on the toy env (plan Task 0.4).

    python -m so101_rl.rl.smoke_tdmpc2 --steps 1000 --device {auto,cpu,cuda}

Writes ``runs/smoke-tdmpc2/<ts>/{metrics.csv, model.pt}`` and exits non-zero unless both artifacts are
there and ``metrics.csv`` has at least 3 rows (L12: check artifacts, not the return code).
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from so101_rl.rl.toy_env import ToyReachEnv

VENDOR_CONFIG = Path(__file__).resolve().parent.parent / "_vendor" / "tdmpc2" / "config.yaml"
MIN_ROWS = 3


class _FlatToyEnv:
    """Minimal TD-MPC2 env contract around ToyReachEnv: flat state = observation || desired_goal (6-D).

    The full adapter (`TdMpc2GoalEnv`) lands in U9. Success does not terminate: TD-MPC2's
    `episodic=false` mode treats termination as an error, so episodes always run to the time limit.
    """

    def __init__(self, seed: int = 0) -> None:
        self._env = ToyReachEnv()
        self._seed = seed
        self._first = True
        self.obs_dim = 6
        self.action_dim = 2
        self.max_episode_steps = self._env.episode_steps

    def _flat(self, obs: dict[str, np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32))

    def reset(self) -> torch.Tensor:
        obs, _ = self._env.reset(seed=self._seed if self._first else None)
        self._first = False
        return self._flat(obs)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = self._env.step(action.detach().cpu().numpy())
        done = bool(terminated or truncated) or self._env._t >= self.max_episode_steps
        # success keeps the episode running so that every episode has the same length
        if terminated:
            done = self._env._t >= self.max_episode_steps
        return (
            self._flat(obs),
            torch.tensor(reward, dtype=torch.float32),
            done,
            {"success": float(info["is_success"]), "terminated": torch.tensor(False)},
        )

    def rand_act(self) -> torch.Tensor:
        return torch.from_numpy(self._env.action_space.sample()).float()


def build_cfg(env: _FlatToyEnv, overrides: dict):
    from so101_rl._vendor.tdmpc2.common.parser import parse_cfg

    cfg = OmegaConf.load(VENDOR_CONFIG)
    cfg = OmegaConf.merge(
        cfg,
        {
            "obs_shape": {"state": [env.obs_dim]},
            "action_dim": env.action_dim,
            "episode_length": env.max_episode_steps,
            "episode_lengths": [env.max_episode_steps],
            "seed_steps": 200,
        },
        overrides,
    )
    cfg.data_dir = ""
    cfg.checkpoint = ""
    return parse_cfg(cfg)


def run(steps: int, device: str, seed: int, root: Path) -> int:
    from so101_rl._vendor.tdmpc2.common.buffer import Buffer
    from so101_rl._vendor.tdmpc2.common.logger import Logger
    from so101_rl._vendor.tdmpc2.common.seed import set_seed
    from so101_rl._vendor.tdmpc2.tdmpc2 import TDMPC2
    from so101_rl._vendor.tdmpc2.trainer.online_trainer import OnlineTrainer

    if device == "cuda":
        device = "cuda:0"
    out = root / "smoke-tdmpc2" / dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    env = _FlatToyEnv(seed=seed)
    cfg = build_cfg(
        env,
        {
            "task": "toy-reach",
            "steps": steps,
            "seed_steps": min(200, max(steps // 2, env.max_episode_steps + 10)),  # > 1 episode, or update() hits an empty buffer
            "buffer_size": 10_000,
            "batch_size": 64,
            "horizon": 3,
            "model_size": 1,
            "eval_freq": 10**9,
            "eval_episodes": 1,
            "save_video": False,
            "compile": False,
            "enable_wandb": False,
            "device": device,
            "seed": seed,
            "work_dir": str(out),
        },
    )
    set_seed(cfg.seed)
    trainer = OnlineTrainer(cfg=cfg, env=env, agent=TDMPC2(cfg), buffer=Buffer(cfg), logger=Logger(cfg))
    trainer.train()

    final = out / "models" / "final.pt"
    if final.exists():
        shutil.copy2(final, out / "model.pt")
    metrics = out / "metrics.csv"
    rows = len(pd.read_csv(metrics)) if metrics.exists() else 0
    ok = (out / "model.pt").exists() and rows >= MIN_ROWS
    print(f"smoke-tdmpc2: {'OK' if ok else 'FAILED'} dir={out} metrics_rows={rows} model={(out / 'model.pt').exists()}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", type=Path, default=Path("runs"))
    args = ap.parse_args(argv)
    return run(args.steps, args.device, args.seed, args.root)


if __name__ == "__main__":
    sys.exit(main())
