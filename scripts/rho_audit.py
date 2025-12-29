#!/usr/bin/env python3
"""Summarize rho statistics and action clipping diagnostics for a run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

REQUIRED_COLUMNS = [
    "iter",
    "mean_rho",
    "mean_rho2",
    "p95_rho",
    "p99_rho",
    "max_rho",
    "clip_fraction",
    "mean_abs_a_diff",
    "p95_abs_a_diff",
    "max_abs_a_diff",
    "mean_rho_raw",
    "mean_rho2_raw",
    "mean_rho_exec",
    "mean_rho2_exec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rho statistics and action clipping.")
    parser.add_argument("--run", type=str, required=True, help="Run directory containing learning_curves.csv")
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


def _load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _coerce_row(row: Dict[str, object]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for col in REQUIRED_COLUMNS:
        payload[col] = _parse_float(row.get(col))
    return payload


def _finite(values: List[float]) -> List[float]:
    return [val for val in values if isinstance(val, (int, float)) and math.isfinite(float(val))]


def _series_stats(values: List[float]) -> Dict[str, float]:
    finite = _finite(values)
    if not finite:
        return {"count": 0.0, "last": math.nan, "min": math.nan, "max": math.nan, "mean": math.nan, "p95": math.nan}
    arr = np.asarray(finite, dtype=float)
    return {
        "count": float(arr.size),
        "last": float(arr[-1]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _format(value: float) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{value:.4g}"


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, run_dir: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        path.write_text("# rho audit\n\nNo learning_curves.csv rows found.\n")
        return

    iters = [row["iter"] for row in rows]
    iter_finite = _finite(iters)
    iter_min = min(iter_finite) if iter_finite else math.nan
    iter_max = max(iter_finite) if iter_finite else math.nan

    mean_rho2_stats = _series_stats([row["mean_rho2"] for row in rows])
    p95_rho_stats = _series_stats([row["p95_rho"] for row in rows])
    p99_rho_stats = _series_stats([row["p99_rho"] for row in rows])
    max_rho_stats = _series_stats([row["max_rho"] for row in rows])
    clip_stats = _series_stats([row["clip_fraction"] for row in rows])
    mean_diff_stats = _series_stats([row["mean_abs_a_diff"] for row in rows])
    p95_diff_stats = _series_stats([row["p95_abs_a_diff"] for row in rows])
    max_diff_stats = _series_stats([row["max_abs_a_diff"] for row in rows])

    rho2_raw_stats = _series_stats([row["mean_rho2_raw"] for row in rows])
    rho2_exec_stats = _series_stats([row["mean_rho2_exec"] for row in rows])

    rho2_min = mean_rho2_stats["min"]
    rho2_below_one = isinstance(rho2_min, (int, float)) and math.isfinite(rho2_min) and rho2_min < 1.0

    lines = [
        "# rho audit",
        "",
        f"run: {run_dir}",
        f"rows: {int(mean_rho2_stats['count']) if mean_rho2_stats['count'] else 0} (iters { _format(iter_min) } -> { _format(iter_max) })",
        "",
        "## key metrics",
        f"- mean_rho2: last={_format(mean_rho2_stats['last'])}, min={_format(mean_rho2_stats['min'])}, max={_format(mean_rho2_stats['max'])}",
        f"- E[rho^2] < 1 observed: {'yes' if rho2_below_one else 'no'}",
        f"- rho tails (max over iters): p95={_format(p95_rho_stats['max'])}, p99={_format(p99_rho_stats['max'])}, max={_format(max_rho_stats['max'])}",
        f"- clip_fraction: last={_format(clip_stats['last'])}, max={_format(clip_stats['max'])}",
        f"- |a_exec-clip(a_exec)|: mean_last={_format(mean_diff_stats['last'])}, p95_last={_format(p95_diff_stats['last'])}, max_last={_format(max_diff_stats['last'])}",
    ]

    if math.isfinite(rho2_raw_stats["last"]) or math.isfinite(rho2_exec_stats["last"]):
        lines.append(
            f"- mean_rho2_raw vs mean_rho2_exec (last): {_format(rho2_raw_stats['last'])} vs {_format(rho2_exec_stats['last'])}"
        )
    else:
        lines.append("- mean_rho2_raw/mean_rho2_exec: unavailable (columns missing)")

    lines.append("")
    lines.append("## notes")
    lines.append("- mean_rho2 is based on the clipped rho used in training.")
    lines.append("- mean_rho2_raw/exec are unclipped rho from pre-squash vs exec actions; they should match when Jacobians cancel.")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    curves_path = run_dir / "learning_curves.csv"
    rows = _load_rows(curves_path)
    if not rows:
        _write_markdown(run_dir / "rho_audit.md", run_dir, [])
        raise SystemExit(f"No learning_curves.csv found at {curves_path}")

    coerced = [_coerce_row(row) for row in rows]
    # filter to rows with finite iter for output ordering
    coerced = [row for row in coerced if math.isfinite(row["iter"])]
    coerced.sort(key=lambda r: r["iter"])

    out_csv = run_dir / "rho_audit.csv"
    _write_csv(out_csv, coerced)
    _write_markdown(run_dir / "rho_audit.md", run_dir, coerced)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
