#!/usr/bin/env python3
"""Run unfixed actor-critic training with config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.algos.train_unfixed_ac import load_train_config, train_unfixed_ac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unfixed actor-critic training.")
    parser.add_argument("--config", type=str, default=None, help="Path to training config yaml/json.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--beta", type=float, default=None, help="Override tracking beta.")
    parser.add_argument("--p-mix", type=float, default=None, help="Override environment p_mix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.p_mix is not None:
        cfg.setdefault("env", {})["p_mix"] = args.p_mix

    result = train_unfixed_ac(cfg)
    print(f"Training complete. Logs at {result['csv_path']}")


if __name__ == "__main__":
    main()
