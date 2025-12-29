#!/usr/bin/env python3
"""Calibrate dist_mmd2 under on-policy vs off-policy settings."""

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


def _summarize_off(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
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
    seed: int,
) -> Dict[str, Any]:
    result = run_distribution_probe(
        env_config=env_cfg,
        theta_mu=theta_mu,
        theta_pi=theta_pi,
        sigma_mu=sigma_mu,
        sigma_pi=sigma_pi,
        num_samples=num_samples,
        seed=seed,
    )
    return {
        "seed": seed,
        "num_samples": num_samples,
        "dist_mmd2": float(result["mmd2"]),
        "dist_mean_l2": float(result["mean_l2"]),
        "mmd_sigma": float(result["mmd_sigma"]),
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
    theta_scale: float,
    sigma_pi: float,
    offpolicy_delta_scale: float,
) -> float:
    lines: List[str] = []
    lines.append("# On-policy dist_mmd2 calibration")
    lines.append("")
    lines.append(f"- on-policy: theta_mu == theta_pi, sigma_mu == sigma_pi == {sigma_pi:g}")
    lines.append(f"- theta_init_scale: {theta_scale:g}")
    lines.append(f"- off-policy delta scale: {offpolicy_delta_scale:g}")
    lines.append("")
    lines.append("## On-policy dist_mmd2 stats")
    lines.append("| num_samples | count | mean | std | p95 | p99 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    reco_threshold = float("nan")
    on_by_samples: Dict[int, List[float]] = {n: [] for n in num_samples_list}
    for row in on_rows:
        on_by_samples[int(row["num_samples"])].append(float(row["dist_mmd2"]))
    for num_samples in num_samples_list:
        stats = _summarize(on_by_samples[num_samples])
        if num_samples == num_samples_list[0]:
            reco_threshold = stats["p99"] * 1.2
        lines.append(
            "| "
            + " | ".join(
                [
                    str(num_samples),
                    str(len(on_by_samples[num_samples])),
                    _format_float(stats["mean"]),
                    _format_float(stats["std"]),
                    _format_float(stats["p95"]),
                    _format_float(stats["p99"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        f"Recommended threshold (num_samples={num_samples_list[0]}): "
        f"threshold_reco = p99 * 1.2 = {_format_float(reco_threshold)}"
    )
    lines.append("Reason: add 20% headroom over the on-policy p99 to reduce false positives.")
    lines.append("")
    lines.append("## Off-policy dist_mmd2 stats")
    lines.append("| num_samples | count | mean | std | p05 | p50 | p95 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    off_by_samples: Dict[int, List[float]] = {n: [] for n in num_samples_list}
    for row in off_rows:
        off_by_samples[int(row["num_samples"])].append(float(row["dist_mmd2"]))
    for num_samples in num_samples_list:
        stats = _summarize_off(off_by_samples[num_samples])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(num_samples),
                    str(len(off_by_samples[num_samples])),
                    _format_float(stats["mean"]),
                    _format_float(stats["std"]),
                    _format_float(stats["p05"]),
                    _format_float(stats["p50"]),
                    _format_float(stats["p95"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Separation (on-policy p99 vs off-policy p05)")
    for num_samples in num_samples_list:
        on_stats = _summarize(on_by_samples[num_samples])
        off_stats = _summarize_off(off_by_samples[num_samples])
        ratio = float("inf")
        if on_stats["p99"] > 0:
            ratio = off_stats["p05"] / on_stats["p99"]
        lines.append(
            f"- num_samples={num_samples}: on_p99={_format_float(on_stats['p99'])}, "
            f"off_p05={_format_float(off_stats['p05'])}, ratio={_format_float(ratio)}"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return reco_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate dist_mmd2 under on-policy vs off-policy probes.")
    parser.add_argument("--config", type=str, default="configs/train_sanity.yaml", help="Training config path.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/calibration_on_policy_mmd",
        help="Output directory for calibration csv/summary.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First on-policy seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=49, help="Last on-policy seed (inclusive).")
    parser.add_argument("--offpolicy-seeds", type=int, default=10, help="Number of off-policy seeds.")
    parser.add_argument(
        "--num-samples",
        type=str,
        default="512,1024",
        help="Comma-separated num_samples list.",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also include 2048 and 4096 samples (may be memory heavy).",
    )
    parser.add_argument(
        "--offpolicy-delta-scale",
        type=float,
        default=1.0,
        help="Scale for theta_mu - theta_pi perturbation in off-policy runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)

    num_samples_list = _parse_int_list(args.num_samples)
    if args.include_large:
        for extra in (2048, 4096):
            if extra not in num_samples_list:
                num_samples_list.append(extra)
    num_samples_list = sorted(num_samples_list)

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
        theta_scale=theta_scale,
        sigma_pi=sigma_pi,
        offpolicy_delta_scale=args.offpolicy_delta_scale,
    )


if __name__ == "__main__":
    main()
