#!/usr/bin/env python3
"""Simple rollout to exercise TorusGobletGhostEnv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke rollout for TorusGobletGhostEnv")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml/json")
    parser.add_argument("--steps", type=int, default=1000, help="Number of rollout steps")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reset and actions")
    parser.add_argument("--render", action="store_true", help="Enable pygame rendering")
    parser.add_argument("--record", type=str, default=None, help="Path to gif/mp4 recording")
    parser.add_argument("--fps", type=int, default=30, help="Renderer fps")
    parser.add_argument("--no-window", action="store_true", help="Render offscreen only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config) if args.config else None
    cfg = load_config(str(cfg_path)) if cfg_path else load_config()

    env = TorusGobletGhostEnv(config=cfg)
    obs, info = env.reset(seed=args.seed)

    rng = np.random.default_rng(args.seed if args.seed is not None else cfg.get("seed", None))
    renderer = None
    if args.render or args.record:
        from tdrl_unfixed_ac.envs.render import TorusRenderer

        renderer = TorusRenderer(env, fps=args.fps, show=not args.no_window, record_path=args.record)

    counts = {"caught": 0, "picked": 0, "restart": 0}
    rewards = []
    psi_norm_max = 0.0
    last_info = {}

    for _ in range(args.steps):
        action = rng.normal(loc=0.0, scale=1.0, size=2)
        obs, reward, terminated, truncated, info = env.step(action)
        if renderer is not None:
            renderer.render(action=action)
        rewards.append(reward)
        counts["caught"] += int(info.get("caught", False))
        counts["picked"] += int(info.get("picked", False))
        counts["restart"] += int(info.get("restart", False))
        psi_norm_max = max(psi_norm_max, float(np.linalg.norm(info.get("psi", 0.0))))
        last_info = info
        if terminated or truncated:
            raise RuntimeError("Environment should be continuing but returned a terminal flag.")

    rewards_arr = np.asarray(rewards, dtype=float)
    print(f"Ran {args.steps} steps. Total reward: {rewards_arr.sum():.2f}")
    print(
        "Reward stats: mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
            rewards_arr.mean(), rewards_arr.std(), rewards_arr.min(), rewards_arr.max()
        )
    )
    print(f"Max ||psi||: {psi_norm_max:.3f} (C_psi={env.c_psi:.3f})")
    if last_info.get("phi") is not None:
        print(f"Phi dim: {last_info['phi'].shape[0]}, reward_teacher: {last_info.get('reward_teacher', np.nan):.3f}")
    print(f"Caught events: {counts['caught']}, Goblets picked: {counts['picked']}, Restarts: {counts['restart']}")
    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
