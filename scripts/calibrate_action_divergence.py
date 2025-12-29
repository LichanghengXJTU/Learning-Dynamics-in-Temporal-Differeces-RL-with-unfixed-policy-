#!/usr/bin/env python3
"""Calibrate action divergence (KL/TV) under on-policy vs off-policy settings."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.algos.train_unfixed_ac import load_train_config
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv, load_config as load_env_config
from tdrl_unfixed_ac.probes.distribution_probe import run_distribution_probe


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p95": float("nan"), "p99": float("nan")}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_env_config(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    env_path = cfg.get("env_config_path")
    env_cfg = load_env_config(env_path) if env_path else load_env_config()
    env_cfg.update(cfg.get("env", {}))
    if env_cfg.get("seed") is None:
        env_cfg["seed"] = seed
    return env_cfg


def _load_dims(env_cfg: Dict[str, Any]) -> Dict[str, int]:
    env = TorusGobletGhostEnv(config=env_cfg)
    return {"actor_dim": int(env.actor_feature_dim), "action_dim": int(env.critic_features_map.action_dim)}


def _run_probe(
    *,
    env_cfg: Dict[str, Any],
    theta_mu: np.ndarray,
    theta_pi: np.ndarray,
    sigma_mu: float,
    sigma_pi: float,
    num_samples: int,
    action_samples: int,
    seed: int,
) -> Dict[str, Any]:
    result = run_distribution_probe(
        env_config=env_cfg,
        theta_mu=theta_mu,
        theta_pi=theta_pi,
        sigma_mu=sigma_mu,
        sigma_pi=sigma_pi,
        num_samples=num_samples,
        action_samples=action_samples,
        seed=seed,
    )
    return {
        "seed": seed,
        "num_samples": num_samples,
        "action_samples": action_samples,
        "dist_action_kl": float(result["dist_action_kl"]),
        "dist_action_tv": float(result["dist_action_tv"]),
    }


def _collect_rows(
    *,
    seeds: Iterable[int],
    num_samples_list: Sequence[int],
    env_cfg_base: Dict[str, Any],
    actor_dim: int,
    action_dim: int,
    theta_scale: float,
    sigma_mu: float,
    sigma_pi: float,
    action_samples: int,
    offpolicy_delta_scale: float,
    offpolicy: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        env_cfg = dict(env_cfg_base)
        env_cfg["seed"] = seed
        rng = np.random.default_rng(seed)
        theta_pi = rng.normal(loc=0.0, scale=theta_scale, size=(actor_dim, action_dim))
        if offpolicy:
            delta = rng.normal(loc=0.0, scale=theta_scale, size=(actor_dim, action_dim))
            theta_mu = theta_pi + offpolicy_delta_scale * delta
        else:
            theta_mu = np.array(theta_pi, copy=True)
        for num_samples in num_samples_list:
            rows.append(
                _run_probe(
                    env_cfg=env_cfg,
                    theta_mu=theta_mu,
                    theta_pi=theta_pi,
                    sigma_mu=sigma_mu,
                    sigma_pi=sigma_pi,
                    num_samples=num_samples,
                    action_samples=action_samples,
                    seed=seed,
                )
            )
    return rows


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _write_summary(
    path: Path,
    *,
    on_rows: List[Dict[str, Any]],
    off_rows: List[Dict[str, Any]],
    num_samples_list: Sequence[int],
    action_samples: int,
    theta_scale: float,
    sigma_pi: float,
    offpolicy_delta_scale: float,
) -> None:
    lines: List[str] = []
    lines.append("# Action divergence calibration")
    lines.append("")
    lines.append(f"- on-policy: theta_mu == theta_pi, sigma_mu == sigma_pi == {sigma_pi:g}")
    lines.append(f"- theta_init_scale: {theta_scale:g}")
    lines.append(f"- off-policy delta scale: {offpolicy_delta_scale:g}")
    lines.append(f"- action_samples: {action_samples:d}")
    lines.append("")

    metrics = ["dist_action_kl", "dist_action_tv"]
    eps_floors = {"dist_action_kl": 1e-6, "dist_action_tv": 1e-4}
    on_by_samples: Dict[int, List[Dict[str, Any]]] = {n: [] for n in num_samples_list}
    off_by_samples: Dict[int, List[Dict[str, Any]]] = {n: [] for n in num_samples_list}
    for row in on_rows:
        on_by_samples[int(row["num_samples"])].append(row)
    for row in off_rows:
        off_by_samples[int(row["num_samples"])].append(row)

    lines.append("## On-policy stats")
    lines.append("| num_samples | metric | count | mean | std | p95 | p99 | eps_floor | reco_threshold |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    reco_thresholds: Dict[int, Dict[str, Dict[str, float]]] = {n: {} for n in num_samples_list}
    for num_samples in num_samples_list:
        for metric in metrics:
            values = [float(row[metric]) for row in on_by_samples[num_samples]]
            stats = _summarize(values)
            eps_floor = eps_floors[metric]
            reco = max(eps_floor, stats["p99"] * 1.2)
            reco_thresholds[num_samples][metric] = {
                "p99": stats["p99"],
                "eps_floor": eps_floor,
                "reco_threshold": reco,
            }
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(num_samples),
                        metric,
                        str(len(values)),
                        _format_float(stats["mean"]),
                        _format_float(stats["std"]),
                        _format_float(stats["p95"]),
                        _format_float(stats["p99"]),
                        _format_float(eps_floor),
                        _format_float(reco),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Off-policy stats")
    lines.append("| num_samples | metric | count | mean | std | p95 | p99 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for num_samples in num_samples_list:
        for metric in metrics:
            values = [float(row[metric]) for row in off_by_samples[num_samples]]
            stats = _summarize(values)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(num_samples),
                        metric,
                        str(len(values)),
                        _format_float(stats["mean"]),
                        _format_float(stats["std"]),
                        _format_float(stats["p95"]),
                        _format_float(stats["p99"]),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Recommended thresholds (max(eps_floor, p99 * 1.2))")
    for num_samples in num_samples_list:
        kl_stats = reco_thresholds[num_samples].get("dist_action_kl", {})
        tv_stats = reco_thresholds[num_samples].get("dist_action_tv", {})
        lines.append(
            f"- num_samples={num_samples}: "
            f"dist_action_kl(p99={_format_float(kl_stats.get('p99', float('nan')))}, "
            f"eps_floor={_format_float(kl_stats.get('eps_floor', float('nan')))}, "
            f"reco_threshold={_format_float(kl_stats.get('reco_threshold', float('nan')))}), "
            f"dist_action_tv(p99={_format_float(tv_stats.get('p99', float('nan')))}, "
            f"eps_floor={_format_float(tv_stats.get('eps_floor', float('nan')))}, "
            f"reco_threshold={_format_float(tv_stats.get('reco_threshold', float('nan')))})"
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate action divergence (KL/TV) under on-policy vs off-policy.")
    parser.add_argument("--config", type=str, default="configs/train_sanity.yaml", help="Training config path.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/calibration_action_divergence",
        help="Output directory for calibration csv/summary.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First on-policy seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=49, help="Last on-policy seed (inclusive).")
    parser.add_argument("--offpolicy-seeds", type=int, default=30, help="Number of off-policy seeds.")
    parser.add_argument(
        "--num-samples",
        type=str,
        default="512",
        help="Comma-separated num_samples list.",
    )
    parser.add_argument("--action-samples", type=int, default=64, help="Action samples per state for TV estimate.")
    parser.add_argument(
        "--offpolicy-delta-scale",
        type=float,
        default=3.0,
        help="Scale for theta_mu - theta_pi perturbation in off-policy runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)

    num_samples_list = sorted(_parse_int_list(args.num_samples))
    on_seeds = list(range(args.seed_start, args.seed_end + 1))
    off_seeds = list(range(args.offpolicy_seeds))

    sigma_pi = float(cfg.get("sigma_pi", cfg.get("sigma_mu", 0.2)))
    sigma_mu = sigma_pi

    theta_scale = float(cfg.get("theta_init_scale", 0.1))

    env_cfg_seed0 = _build_env_config(cfg, seed=on_seeds[0] if on_seeds else 0)
    dims = _load_dims(env_cfg_seed0)
    actor_dim = dims["actor_dim"]
    action_dim = dims["action_dim"]

    on_rows = _collect_rows(
        seeds=on_seeds,
        num_samples_list=num_samples_list,
        env_cfg_base=env_cfg_seed0,
        actor_dim=actor_dim,
        action_dim=action_dim,
        theta_scale=theta_scale,
        sigma_mu=sigma_mu,
        sigma_pi=sigma_pi,
        action_samples=args.action_samples,
        offpolicy_delta_scale=args.offpolicy_delta_scale,
        offpolicy=False,
    )
    off_rows = _collect_rows(
        seeds=off_seeds,
        num_samples_list=num_samples_list,
        env_cfg_base=env_cfg_seed0,
        actor_dim=actor_dim,
        action_dim=action_dim,
        theta_scale=theta_scale,
        sigma_mu=sigma_mu,
        sigma_pi=sigma_pi,
        action_samples=args.action_samples,
        offpolicy_delta_scale=args.offpolicy_delta_scale,
        offpolicy=True,
    )

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    _write_csv(out_dir / "calibration.csv", on_rows)
    _write_csv(out_dir / "offpolicy_calibration.csv", off_rows)
    _write_summary(
        out_dir / "calibration_summary.md",
        on_rows=on_rows,
        off_rows=off_rows,
        num_samples_list=num_samples_list,
        action_samples=args.action_samples,
        theta_scale=theta_scale,
        sigma_pi=sigma_pi,
        offpolicy_delta_scale=args.offpolicy_delta_scale,
    )


if __name__ == "__main__":
    main()
