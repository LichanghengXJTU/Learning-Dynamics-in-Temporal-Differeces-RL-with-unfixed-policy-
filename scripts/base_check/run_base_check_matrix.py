#!/usr/bin/env python3
"""Run base-check matrix for the unfixed actor-critic implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.algos.train_unfixed_ac import load_train_config, train_unfixed_ac


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _trend(values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return "flat"
    x = np.arange(arr.size)
    slope = float(np.polyfit(x, arr, 1)[0])
    tol = 1e-6 * max(1.0, float(np.mean(np.abs(arr))))
    if slope > tol:
        return "up"
    if slope < -tol:
        return "down"
    return "flat"


def _series_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"start": float("nan"), "end": float("nan"), "trend": "flat"}
    return {"start": float(arr[0]), "end": float(arr[-1]), "trend": _trend(arr.tolist())}


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_case(case: Dict[str, Any], base_cfg: Dict[str, Any], runs_dir: Path, seed: int) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["output_dir"] = str(runs_dir / case["name"])
    cfg["seed"] = seed
    cfg["log_contract_metrics"] = True
    cfg["squash_action"] = False
    cfg["require_teacher_reward"] = True
    cfg["disable_rho_clip"] = True
    cfg["rho_clip"] = None
    cfg["probes"] = {"enabled": False}
    cfg["check_name"] = case["name"]
    cfg.setdefault("env", {})
    cfg["env"]["clip_action"] = False
    cfg["env"].setdefault("p_mix", 0.0)
    cfg["env"].setdefault("sigma_env", 0.0)
    cfg["env"].setdefault("sigma_ghost", 0.0)

    _deep_update(cfg, case.get("overrides", {}))

    result = train_unfixed_ac(cfg)
    logs = result.get("logs", [])
    metrics_rows: List[Dict[str, Any]] = []
    for row in logs:
        metrics_rows.append(
            {
                "iter": row.get("iter"),
                "td_loss_est": row.get("td_loss_est", _float_or_nan(row.get("td_loss")) / 2.0),
                "cos_w_wr": row.get("cos_w_wr"),
                "w_dot_wr_over_n": row.get("w_dot_wr_over_n"),
                "w_norm": row.get("w_norm_contract"),
                "theta_pi_norm": row.get("theta_pi_norm"),
                "theta_mu_norm": row.get("theta_mu_norm"),
                "tracking_gap": row.get("tracking_gap_contract"),
                "mean_rho2": row.get("mean_rho2"),
                "p95_rho2": row.get("p95_rho2"),
                "action_mean": row.get("action_mean"),
                "action_var": row.get("action_var"),
                "dist_action_kl": row.get("dist_action_kl"),
            }
        )

    case_dir = Path(cfg["output_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = case_dir / "metrics.csv"
    fieldnames = list(metrics_rows[0].keys()) if metrics_rows else []
    if fieldnames:
        _write_csv(metrics_path, metrics_rows, fieldnames)

    return {
        "config": cfg,
        "metrics_rows": metrics_rows,
        "metrics_path": str(metrics_path),
    }


def _build_run_report(case: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = payload["config"]
    rows = payload["metrics_rows"]
    case_dir = Path(cfg["output_dir"])

    metrics = {
        "td_loss_est": _series_stats([_float_or_nan(r.get("td_loss_est")) for r in rows]),
        "w_dot_wr_over_n": _series_stats([_float_or_nan(r.get("w_dot_wr_over_n")) for r in rows]),
        "cos_w_wr": _series_stats([_float_or_nan(r.get("cos_w_wr")) for r in rows]),
        "mean_rho2": _series_stats([_float_or_nan(r.get("mean_rho2")) for r in rows]),
        "tracking_gap": _series_stats([_float_or_nan(r.get("tracking_gap")) for r in rows]),
        "theta_pi_norm": _series_stats([_float_or_nan(r.get("theta_pi_norm")) for r in rows]),
        "theta_mu_norm": _series_stats([_float_or_nan(r.get("theta_mu_norm")) for r in rows]),
    }

    td_start = metrics["td_loss_est"]["start"]
    td_end = metrics["td_loss_est"]["end"]
    w_start = metrics["w_dot_wr_over_n"]["start"]
    w_end = metrics["w_dot_wr_over_n"]["end"]
    rho_end = metrics["mean_rho2"]["end"]
    tracking_end = metrics["tracking_gap"]["end"]
    pi_norm_start = metrics["theta_pi_norm"]["start"]
    pi_norm_end = metrics["theta_pi_norm"]["end"]

    evidence_mode = case.get("evidence_mode", "none")
    evidence: Dict[str, Any] = {}
    evidence_pass = True
    if evidence_mode == "critic_only":
        evidence = {
            "td_loss_down": _finite(td_start) and _finite(td_end) and td_end < td_start,
            "w_alignment_up": _finite(w_start) and _finite(w_end) and w_end > w_start,
            "rho_finite": _finite(rho_end),
        }
        evidence_pass = all(evidence.values())
    elif evidence_mode == "actor_on":
        theta_pi_changed = _finite(pi_norm_start) and _finite(pi_norm_end) and abs(pi_norm_end - pi_norm_start) > 1e-6
        tracking_gap_ok = _finite(tracking_end) and tracking_end <= 1e-3
        rho_near_one = _finite(rho_end) and abs(rho_end - 1.0) <= 0.05
        evidence = {
            "theta_pi_changed": theta_pi_changed,
            "tracking_gap_small": tracking_gap_ok,
            "rho_near_one": rho_near_one,
        }
        evidence_pass = all(evidence.values())
    else:
        evidence = {"note": "evidence not enforced for this case"}

    sigma_mu = _float_or_nan(cfg.get("sigma_mu"))
    sigma_pi = _float_or_nan(cfg.get("sigma_pi"))
    sigma_condition = (
        _finite(sigma_mu)
        and _finite(sigma_pi)
        and sigma_mu > 0.0
        and sigma_pi > 0.0
        and (sigma_pi * sigma_pi < 2.0 * sigma_mu * sigma_mu)
    )

    summary = {
        "case": case["name"],
        "metrics": metrics,
        "evidence": evidence,
        "evidence_pass": evidence_pass,
        "sigma_condition": sigma_condition,
    }

    config_summary = {
        "seed": cfg.get("seed"),
        "N": cfg.get("env", {}).get("feature_dim", cfg.get("feature_dim")),
        "N_act": cfg.get("env", {}).get("actor_feature_dim", cfg.get("actor_feature_dim")),
        "B": cfg.get("trajectories"),
        "T": cfg.get("horizon"),
        "gamma": cfg.get("gamma"),
        "alpha_w": cfg.get("alpha_w"),
        "alpha_pi": cfg.get("alpha_pi"),
        "beta": cfg.get("beta"),
        "sigma_mu": cfg.get("sigma_mu"),
        "sigma_pi": cfg.get("sigma_pi"),
    }

    report = {
        "config": config_summary,
        "summary": summary,
        "metrics_path": payload["metrics_path"],
    }

    report_path = case_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    md_lines = ["# Base-Check Run Report", ""]
    md_lines.append("## Config")
    md_lines.append(
        "- "
        + ", ".join(
            f"{key}={config_summary.get(key)}"
            for key in ["N", "N_act", "B", "T", "gamma", "alpha_w", "alpha_pi", "beta", "sigma_mu", "sigma_pi", "seed"]
        )
    )
    md_lines.append("")
    md_lines.append("## Key Metrics (start -> end, trend)")
    for key in ["td_loss_est", "w_dot_wr_over_n", "cos_w_wr", "mean_rho2", "tracking_gap"]:
        stats = metrics[key]
        md_lines.append(
            f"- {key}: {_fmt(stats['start'])} -> {_fmt(stats['end'])} ({stats['trend']})"
        )
    md_lines.append("")
    md_lines.append("## Learning Evidence")
    if evidence_mode == "critic_only":
        md_lines.append(f"- td_loss_down: {evidence.get('td_loss_down')}")
        md_lines.append(f"- w_alignment_up: {evidence.get('w_alignment_up')}")
        md_lines.append(f"- rho_finite: {evidence.get('rho_finite')}")
        md_lines.append(f"- evidence_pass: {evidence_pass}")
    elif evidence_mode == "actor_on":
        md_lines.append(f"- theta_pi_changed: {evidence.get('theta_pi_changed')}")
        md_lines.append(f"- tracking_gap_small: {evidence.get('tracking_gap_small')}")
        md_lines.append(f"- rho_near_one: {evidence.get('rho_near_one')}")
        md_lines.append(f"- evidence_pass: {evidence_pass}")
    else:
        md_lines.append("- evidence_pass: not enforced for this case")
    md_lines.append("")
    md_lines.append("## Sigma Condition")
    md_lines.append(f"- sigma_pi^2 < 2 sigma_mu^2: {sigma_condition}")
    md_lines.append("")
    (case_dir / "run_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base-check matrix.")
    parser.add_argument("--base-dir", type=str, required=True, help="Base output dir (outputs/base_check/<TS>).")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--outer-iters", type=int, default=80, help="Outer iterations per case.")
    parser.add_argument("--horizon", type=int, default=30, help="Trajectory horizon T.")
    parser.add_argument("--trajectories", type=int, default=4, help="Trajectories per outer iter B.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_train_config(None)
    base_cfg["outer_iters"] = int(args.outer_iters)
    base_cfg["horizon"] = int(args.horizon)
    base_cfg["trajectories"] = int(args.trajectories)

    cases = [
        {
            "name": "C0_onpolicy_critic_only",
            "expect_on_policy": True,
            "evidence_mode": "critic_only",
            "overrides": {"alpha_pi": 0.0, "beta": 1.0, "sigma_mu": 0.2, "sigma_pi": 0.2},
        },
        {
            "name": "C1_onpolicy_actor_on",
            "expect_on_policy": True,
            "evidence_mode": "actor_on",
            "overrides": {"alpha_pi": 0.1, "beta": 1.0, "sigma_mu": 0.2, "sigma_pi": 0.2},
        },
        {
            "name": "C2_offpolicy_fixed_behavior",
            "expect_on_policy": False,
            "evidence_mode": "none",
            "overrides": {
                "beta": 0.0,
                "theta_mu_offset_scale": 0.5,
                "sigma_mu": 0.35,
                "sigma_pi": 0.2,
            },
        },
        {
            "name": "C3_offpolicy_tracking_beta_small",
            "expect_on_policy": False,
            "evidence_mode": "none",
            "overrides": {
                "beta": 0.1,
                "theta_mu_offset_scale": 0.5,
                "sigma_mu": 0.35,
                "sigma_pi": 0.2,
            },
        },
        {
            "name": "C4_offpolicy_tracking_beta_mid",
            "expect_on_policy": False,
            "evidence_mode": "none",
            "overrides": {
                "beta": 0.5,
                "theta_mu_offset_scale": 0.5,
                "sigma_mu": 0.35,
                "sigma_pi": 0.2,
            },
        },
        {
            "name": "C5_gamma_zero_sanity",
            "expect_on_policy": False,
            "evidence_mode": "none",
            "overrides": {"gamma": 0.0},
        },
        {
            "name": "C6_batch_scaling_check_B1",
            "expect_on_policy": False,
            "evidence_mode": "none",
            "overrides": {"trajectories": 1, "alpha_pi": 0.0, "beta": 1.0, "sigma_mu": 0.2, "sigma_pi": 0.2},
        },
        {
            "name": "C7_batch_scaling_check_B4",
            "expect_on_policy": False,
            "evidence_mode": "none",
            "overrides": {"trajectories": 4, "alpha_pi": 0.0, "beta": 1.0, "sigma_mu": 0.2, "sigma_pi": 0.2},
        },
    ]

    reports = []
    any_fail = False
    for idx, case in enumerate(cases):
        seed = args.seed + idx * 11
        payload = _run_case(case, base_cfg, runs_dir, seed)
        report = _build_run_report(case, payload)
        reports.append(report)
        if case.get("evidence_mode") in ("critic_only", "actor_on") and not report["summary"]["evidence_pass"]:
            any_fail = True

    summary_path = base_dir / "summary" / "base_check_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"cases": reports}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    raise SystemExit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
