#!/usr/bin/env python3
"""Plot learning curves and probe diagnostics from a single run directory."""

from __future__ import annotations

import argparse
import csv
import math
import runpy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning curves + probes from outputs.")
    parser.add_argument("--run", type=str, required=True, help="Run directory containing learning_curves.csv")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots/tables")
    return parser.parse_args()


def _load_csv(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for field in reader.fieldnames:
            data[field] = []
        for row in reader:
            for field in reader.fieldnames:
                raw = row.get(field, "")
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = math.nan
                data[field].append(val)
    return data


def _last_finite(values: List[float]) -> Optional[float]:
    for value in reversed(values):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _plot_series(
    *,
    x: np.ndarray,
    series: Dict[str, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(6.5, 4.0))
    for label, y in series.items():
        plt.plot(x, y, linewidth=1.6, label=label)
    plt.title(title)
    plt.xlabel("iter")
    plt.grid(alpha=0.3)
    if len(series) > 1:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir.parent / "analysis" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    curves_path = run_dir / "learning_curves.csv"
    if curves_path.exists():
        curves = _load_csv(curves_path)
        plot_script = Path(__file__).resolve().parents[1] / "plots" / "plot_learning_curves.py"
        plot_args = ["--csv", str(curves_path), "--out", str(out_dir / "learning_curves.png")]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["plot_learning_curves.py"] + plot_args
            runpy.run_path(str(plot_script), run_name="__main__")
        finally:
            sys.argv = old_argv
    else:
        print(f"learning_curves.csv not found at {curves_path}")
        curves = {}

    probes_dir = run_dir / "probes"
    if probes_dir.exists():
        stability_path = probes_dir / "stability_probe.csv"
        if stability_path.exists():
            stab = _load_csv(stability_path)
            x = np.asarray(stab.get("iter", list(range(len(stab.get("stability_proxy", []))))), dtype=float)
            _plot_series(
                x=x,
                series={"stability_proxy": np.asarray(stab.get("stability_proxy", []), dtype=float)},
                title="Stability Proxy",
                out_path=out_dir / "stability_probe.png",
            )

        fixed_path = probes_dir / "fixed_point_probe.csv"
        if fixed_path.exists():
            fixed = _load_csv(fixed_path)
            x = np.asarray(fixed.get("iter", list(range(len(fixed.get("w_gap", []))))), dtype=float)
            series = {}
            if "w_gap" in fixed:
                series["w_gap"] = np.asarray(fixed.get("w_gap", []), dtype=float)
            if "w_sharp_drift" in fixed:
                series["w_sharp_drift"] = np.asarray(fixed.get("w_sharp_drift", []), dtype=float)
            if series:
                _plot_series(x=x, series=series, title="Fixed Point Probe", out_path=out_dir / "fixed_point_probe.png")

        dist_path = probes_dir / "distribution_probe.csv"
        if dist_path.exists():
            dist = _load_csv(dist_path)
            x = np.asarray(dist.get("iter", list(range(len(dist.get("mmd2", []))))), dtype=float)
            series = {}
            if "mmd2" in dist:
                series["mmd2"] = np.asarray(dist.get("mmd2", []), dtype=float)
            if "mean_l2" in dist:
                series["mean_l2"] = np.asarray(dist.get("mean_l2", []), dtype=float)
            if series:
                _plot_series(
                    x=x,
                    series=series,
                    title="Distribution Probe",
                    out_path=out_dir / "distribution_probe.png",
                )

    metrics = {
        "mean_rho2": None,
        "stability_proxy": None,
        "fixed_point_drift": None,
        "dist_mmd2": None,
    }
    if curves:
        if "mean_rho2" in curves:
            metrics["mean_rho2"] = _last_finite(curves["mean_rho2"])
        if "stability_proxy" in curves:
            metrics["stability_proxy"] = _last_finite(curves["stability_proxy"])
        if "fixed_point_drift" in curves:
            metrics["fixed_point_drift"] = _last_finite(curves["fixed_point_drift"])
        if "dist_mmd2" in curves:
            metrics["dist_mmd2"] = _last_finite(curves["dist_mmd2"])

    def fmt(val: Optional[float]) -> str:
        if val is None:
            return "-"
        return f"{val:.4g}"

    table = [
        "# Key Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| mean_rho2 | {fmt(metrics['mean_rho2'])} |",
        f"| stability_proxy | {fmt(metrics['stability_proxy'])} |",
        f"| fixed_point_drift | {fmt(metrics['fixed_point_drift'])} |",
        f"| dist_mmd2 | {fmt(metrics['dist_mmd2'])} |",
    ]
    (out_dir / "metrics_table.md").write_text("\n".join(table))
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
