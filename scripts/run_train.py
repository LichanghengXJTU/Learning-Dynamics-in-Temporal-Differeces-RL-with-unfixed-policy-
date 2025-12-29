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
    parser.add_argument("--alpha-w", type=float, default=None, help="Override critic step size.")
    parser.add_argument("--alpha-pi", type=float, default=None, help="Override actor step size.")
    parser.add_argument("--sigma-mu", type=float, default=None, help="Override behavior policy sigma.")
    parser.add_argument("--sigma-pi", type=float, default=None, help="Override target policy sigma.")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor.")
    parser.add_argument("--theta-radius", type=float, default=None, help="Override policy parameter radius.")
    parser.add_argument("--outer-iters", type=int, default=None, help="Override outer training iterations.")
    parser.add_argument("--rho-clip", type=float, default=None, help="Override rho clip upper bound.")
    parser.add_argument("--disable-rho-clip", action="store_true", help="Disable rho clipping.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir.")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from explicit checkpoint path.")
    parser.add_argument("--report-every", type=int, default=None, help="Generate partial run report every N iters.")
    parser.add_argument(
        "--report-every-seconds",
        type=float,
        default=None,
        help="Generate partial run report every N seconds.",
    )
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
    if args.alpha_w is not None:
        cfg["alpha_w"] = args.alpha_w
    if args.alpha_pi is not None:
        cfg["alpha_pi"] = args.alpha_pi
    if args.sigma_mu is not None:
        cfg["sigma_mu"] = args.sigma_mu
    if args.sigma_pi is not None:
        cfg["sigma_pi"] = args.sigma_pi
    if args.gamma is not None:
        cfg["gamma"] = args.gamma
    if args.theta_radius is not None:
        cfg["theta_radius"] = args.theta_radius
    if args.outer_iters is not None:
        cfg["outer_iters"] = args.outer_iters
    if args.rho_clip is not None:
        cfg["rho_clip"] = args.rho_clip
    if args.disable_rho_clip:
        cfg["disable_rho_clip"] = True
    if args.resume:
        cfg["resume"] = True
    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from
    if args.report_every is not None:
        cfg["report_every"] = args.report_every
    if args.report_every_seconds is not None:
        cfg["report_every_seconds"] = args.report_every_seconds

    result = train_unfixed_ac(cfg)
    print(f"Training complete. Logs at {result['csv_path']}")


if __name__ == "__main__":
    main()
