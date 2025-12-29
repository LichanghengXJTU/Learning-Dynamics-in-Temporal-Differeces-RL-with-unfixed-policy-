#!/usr/bin/env python3
"""Summarize parameter update scales from learning_curves.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize update norms for plateau/instability runs.")
    parser.add_argument("--plateau-run", type=str, required=True, help="Plateau run directory")
    parser.add_argument("--instability-run", type=str, required=True, help="Instability run directory")
    parser.add_argument("--out", type=str, required=True, help="Output markdown path")
    return parser.parse_args()


def _parse_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        text = str(value).strip()
    except Exception:
        return math.nan
    if text == "":
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _load_rows(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float]] = []
        for row in reader:
            rows.append({key: _parse_float(val) for key, val in row.items()})
        return rows


def _finite(values: List[float]) -> List[float]:
    return [val for val in values if isinstance(val, (int, float)) and math.isfinite(float(val))]


def _series_stats(rows: List[Dict[str, float]], col: str) -> Dict[str, float]:
    vals = _finite([row.get(col, math.nan) for row in rows])
    if not vals:
        return {"count": 0.0, "first": math.nan, "last": math.nan, "mean": math.nan, "p95": math.nan}
    arr = np.asarray(vals, dtype=float)
    return {
        "count": float(arr.size),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "mean": float(np.mean(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _format(value: float) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{value:.4g}"


def _summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        "delta_theta_pi_norm": _series_stats(rows, "delta_theta_pi_norm"),
        "delta_w_norm": _series_stats(rows, "delta_w_norm"),
    }


def main() -> None:
    args = parse_args()
    plateau_run = Path(args.plateau_run)
    instability_run = Path(args.instability_run)

    plateau_rows = _load_rows(plateau_run / "learning_curves.csv")
    instability_rows = _load_rows(instability_run / "learning_curves.csv")

    plateau = _summarize(plateau_rows)
    instability = _summarize(instability_rows)

    lines = [
        "# updates scale summary",
        "",
        f"plateau: {plateau_run}",
        f"instability: {instability_run}",
        "",
        "| metric | plateau | instability |",
        "| --- | --- | --- |",
    ]

    for metric in ["delta_theta_pi_norm", "delta_w_norm"]:
        for stat_key in ["first", "last", "mean", "p95"]:
            plateau_val = plateau[metric][stat_key]
            instability_val = instability[metric][stat_key]
            lines.append(
                f"| {metric}_{stat_key} | {_format(plateau_val)} | {_format(instability_val)} |"
            )

    lines.append("")

    Path(args.out).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
