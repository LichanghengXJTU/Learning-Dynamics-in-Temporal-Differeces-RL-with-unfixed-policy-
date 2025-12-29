#!/usr/bin/env python3
"""Compare rho audit metrics across plateau/instability runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize rho_audit.csv for two runs.")
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
        return {"last": math.nan, "min": math.nan, "max": math.nan}
    return {
        "last": float(vals[-1]),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _format(value: float) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{value:.4g}"


def _summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "mean_rho2_last": _series_stats(rows, "mean_rho2")["last"],
        "mean_rho2_min": _series_stats(rows, "mean_rho2")["min"],
        "p99_rho_max": _series_stats(rows, "p99_rho")["max"],
        "max_rho_max": _series_stats(rows, "max_rho")["max"],
        "clip_fraction_last": _series_stats(rows, "clip_fraction")["last"],
        "clip_fraction_max": _series_stats(rows, "clip_fraction")["max"],
        "mean_abs_a_diff_last": _series_stats(rows, "mean_abs_a_diff")["last"],
        "mean_rho2_raw_last": _series_stats(rows, "mean_rho2_raw")["last"],
        "mean_rho2_exec_last": _series_stats(rows, "mean_rho2_exec")["last"],
    }


def _rho2_ok(value: float) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return "yes" if value >= 1.0 else "no"


def main() -> None:
    args = parse_args()
    plateau_run = Path(args.plateau_run)
    instability_run = Path(args.instability_run)

    plateau_rows = _load_rows(plateau_run / "rho_audit.csv")
    instability_rows = _load_rows(instability_run / "rho_audit.csv")

    plateau = _summarize(plateau_rows)
    instability = _summarize(instability_rows)

    lines = [
        "# rho audit summary",
        "",
        f"plateau: {plateau_run}",
        f"instability: {instability_run}",
        "",
        "| metric | plateau | instability |",
        "| --- | --- | --- |",
        f"| mean_rho2_last | {_format(plateau['mean_rho2_last'])} | {_format(instability['mean_rho2_last'])} |",
        f"| mean_rho2_min | {_format(plateau['mean_rho2_min'])} | {_format(instability['mean_rho2_min'])} |",
        f"| E[rho^2] >= 1 (min) | {_rho2_ok(plateau['mean_rho2_min'])} | {_rho2_ok(instability['mean_rho2_min'])} |",
        f"| p99_rho_max | {_format(plateau['p99_rho_max'])} | {_format(instability['p99_rho_max'])} |",
        f"| max_rho_max | {_format(plateau['max_rho_max'])} | {_format(instability['max_rho_max'])} |",
        f"| clip_fraction_last | {_format(plateau['clip_fraction_last'])} | {_format(instability['clip_fraction_last'])} |",
        f"| clip_fraction_max | {_format(plateau['clip_fraction_max'])} | {_format(instability['clip_fraction_max'])} |",
        f"| mean_abs_a_diff_last | {_format(plateau['mean_abs_a_diff_last'])} | {_format(instability['mean_abs_a_diff_last'])} |",
        f"| mean_rho2_raw_last | {_format(plateau['mean_rho2_raw_last'])} | {_format(instability['mean_rho2_raw_last'])} |",
        f"| mean_rho2_exec_last | {_format(plateau['mean_rho2_exec_last'])} | {_format(instability['mean_rho2_exec_last'])} |",
        "",
        "## conclusion",
        f"- plateau E[rho^2] >= 1 (min over iters): {_rho2_ok(plateau['mean_rho2_min'])}",
        f"- instability E[rho^2] >= 1 (min over iters): {_rho2_ok(instability['mean_rho2_min'])}",
        "",
    ]

    Path(args.out).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
