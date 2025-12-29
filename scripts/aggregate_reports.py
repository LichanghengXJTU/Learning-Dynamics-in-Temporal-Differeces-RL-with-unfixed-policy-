#!/usr/bin/env python3
"""Aggregate run_report.json files into summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run reports into summary tables.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing runs.")
    parser.add_argument("--out", type=str, default=None, help="Output markdown summary path.")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(val):
        return val
    return None


def _last_finite_from_csv(csv_path: Path, columns: List[str]) -> Dict[str, Optional[float]]:
    results = {col: None for col in columns}
    if not csv_path.exists():
        return results
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for col in columns:
                raw = row.get(col)
                val = _to_float(raw)
                if val is not None:
                    results[col] = val
    return results


def _infer_label(run_dir: Path, report: Dict[str, Any]) -> str:
    key_hparams = report.get("config", {}).get("key_hparams", {})
    check_name = key_hparams.get("check_name")
    if check_name:
        return str(check_name)
    name = run_dir.name.lower()
    if "plateau" in name:
        return "plateau"
    if "instability" in name:
        return "instability"
    if "on_policy" in name:
        return "on_policy"
    return "unknown"


def _collect_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    metrics = {
        "mean_rho2": None,
        "stability_proxy": None,
        "fixed_point_drift": None,
        "dist_mmd2": None,
        "dist_action_kl": None,
        "dist_action_tv": None,
    }
    curves = _last_finite_from_csv(
        run_dir / "learning_curves.csv",
        ["mean_rho2", "stability_proxy", "fixed_point_drift", "dist_mmd2", "dist_action_kl", "dist_action_tv"],
    )
    metrics.update(curves)

    probes_dir = run_dir / "probes"
    if probes_dir.exists():
        stability = _last_finite_from_csv(probes_dir / "stability_probe.csv", ["stability_proxy"])
        if stability.get("stability_proxy") is not None:
            metrics["stability_proxy"] = stability["stability_proxy"]

        fixed_point = _last_finite_from_csv(probes_dir / "fixed_point_probe.csv", ["w_sharp_drift"])
        if fixed_point.get("w_sharp_drift") is not None:
            metrics["fixed_point_drift"] = fixed_point["w_sharp_drift"]

        dist = _last_finite_from_csv(probes_dir / "distribution_probe.csv", ["mmd2"])
        if dist.get("mmd2") is not None:
            metrics["dist_mmd2"] = dist["mmd2"]
        dist_actions = _last_finite_from_csv(
            probes_dir / "distribution_probe.csv",
            ["dist_action_kl", "dist_action_tv"],
        )
        if dist_actions.get("dist_action_kl") is not None:
            metrics["dist_action_kl"] = dist_actions["dist_action_kl"]
        if dist_actions.get("dist_action_tv") is not None:
            metrics["dist_action_tv"] = dist_actions["dist_action_tv"]

    return metrics


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


def _load_reports(root: Path) -> List[Dict[str, Any]]:
    reports = []
    for path in sorted(root.rglob("run_report.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        run_dir = Path(payload.get("meta", {}).get("run_dir") or path.parent)
        metrics = _collect_metrics(run_dir)
        health = payload.get("health_summary", {})
        reports.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "label": _infer_label(run_dir, payload),
                "status": health.get("status", "UNKNOWN"),
                "reasons": "; ".join(health.get("reasons", []) or []),
                "mean_rho2": metrics.get("mean_rho2"),
                "stability_proxy": metrics.get("stability_proxy"),
                "fixed_point_drift": metrics.get("fixed_point_drift"),
                "dist_mmd2": metrics.get("dist_mmd2"),
                "dist_action_kl": metrics.get("dist_action_kl"),
                "dist_action_tv": metrics.get("dist_action_tv"),
            }
        )
    return reports


def _write_markdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    headers = [
        "run",
        "label",
        "status",
        "mean_rho2",
        "stability_proxy",
        "fixed_point_drift",
        "dist_mmd2",
        "dist_action_kl",
        "dist_action_tv",
        "reasons",
        "run_dir",
    ]
    lines = ["# Summary", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_name", "-")),
                    str(row.get("label", "-")),
                    str(row.get("status", "-")),
                    _format_float(row.get("mean_rho2")),
                    _format_float(row.get("stability_proxy")),
                    _format_float(row.get("fixed_point_drift")),
                    _format_float(row.get("dist_mmd2")),
                    _format_float(row.get("dist_action_kl")),
                    _format_float(row.get("dist_action_tv")),
                    str(row.get("reasons", "-")),
                    str(row.get("run_dir", "-")),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines))


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "run_name",
        "label",
        "status",
        "mean_rho2",
        "stability_proxy",
        "fixed_point_drift",
        "dist_mmd2",
        "dist_action_kl",
        "dist_action_tv",
        "reasons",
        "run_dir",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = _load_reports(root)
    if not rows:
        raise SystemExit(f"No run_report.json files found under {root}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(rows, out_path)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, out_csv)

    if not args.out and not args.out_csv:
        _write_markdown(rows, Path("/dev/stdout"))


if __name__ == "__main__":
    main()
