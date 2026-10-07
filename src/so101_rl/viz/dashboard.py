"""``pixi run dashboard [--logdir runs/] [--port 6006] [--host 127.0.0.1]``: TensorBoard over the training runs.

A thin wrapper around ``tensorboard.main`` (the ``train`` pixi feature ships ``tensorboard`` from conda-forge) so the
pixi task is a module of this package and the defaults are ours: every ``runs/<run>/tensorboard`` directory is
found recursively (``viz.monitor.TrainingMonitor`` writes there) and the server binds to localhost only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_LOGDIR = Path("runs")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6006


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dashboard", description=__doc__.splitlines()[0])
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR, help="scanned recursively (default: runs/)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    return ap


def tensorboard_argv(ns: argparse.Namespace) -> list[str]:
    """The ``tensorboard`` command line for the parsed arguments."""
    return ["tensorboard", "--logdir", str(ns.logdir), "--host", str(ns.host), "--port", str(int(ns.port))]


def main(argv: Sequence[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        from tensorboard import main as tb_main
    except ImportError:
        print("dashboard: tensorboard is not installed in this env (it is in the `train` pixi feature)", file=sys.stderr)
        return 2
    if not ns.logdir.exists():
        print(f"dashboard: {ns.logdir} does not exist (no runs yet?)", file=sys.stderr)
        return 2
    sys.argv = tensorboard_argv(ns)
    tb_main.run_main()  # blocks until Ctrl-C
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
