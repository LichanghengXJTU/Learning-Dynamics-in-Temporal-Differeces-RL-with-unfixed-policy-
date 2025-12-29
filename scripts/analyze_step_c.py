#!/usr/bin/env python3
"""Analyze Step C short runs and generate plots + markdown evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting") from exc

STABILITY_EPS = 1e-3
PROBE_WINDOW = 5
TD_SLOPE_WINDOW = 20
TD_SLOPE_FLAT_MAX = 1e-6
TD_FLAT_RANGE_MAX = 1e-6
TD_INCREASE_MIN = 1e-6
W_NORM_INCREASE_MIN = 1e-4
W_GAP_MIN = 1e-3
SUSTAINED_WINDOW = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze plateau/instability short runs for Step C.")
    parser.add_argument("--plateau-run", type=str, required=True, help="Plateau run directory")
    parser.add_argument("--instability-run", type=str, required=True, help="Instability run directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for analysis artifacts")
    parser.add_argument("--last-k", type=int, default=5, help="Last-k window for slope annotations")
    parser.add_argument(
        "--diff-files",
        action="append",
        default=[],
        help="Repo diff files to include at top of the report (repeatable).",
    )
    parser.add_argument(
        "--key-paths",
        action="append",
        default=[],
        help="Key output file paths to include at top of the report (repeatable).",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    if not path.exists():
        return data
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return data
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


def _probe_map(path: Path) -> Dict[int, Dict[str, float]]:
    data = _load_csv(path)
    if not data or "iter" not in data:
        return {}
    iters = data.get("iter", [])
    mapping: Dict[int, Dict[str, float]] = {}
    for idx, raw_iter in enumerate(iters):
        if not math.isfinite(raw_iter):
            continue
        iter_key = int(raw_iter)
        row = {key: data[key][idx] for key in data.keys()}
        mapping[iter_key] = row
    return mapping


def _probe_iters(paths: List[Path]) -> List[float]:
    iters: set[int] = set()
    for path in paths:
        data = _load_csv(path)
        if not data or "iter" not in data:
            continue
        for raw_iter in data["iter"]:
            if math.isfinite(raw_iter):
                iters.add(int(raw_iter))
    return sorted(float(x) for x in iters)


def _load_config(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x[:0], y[:0]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _last_k_slope(x: np.ndarray, y: np.ndarray, k: int) -> float:
    if y.size < 2:
        return float("nan")
    k = max(2, min(int(k), y.size))
    xk = x[-k:]
    yk = y[-k:]
    dx = xk[-1] - xk[0]
    if dx == 0:
        return float("nan")
    return float((yk[-1] - yk[0]) / dx)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    denom = float(np.sum((x - mean_x) ** 2))
    if denom == 0.0:
        return 0.0
    num = float(np.sum((x - mean_x) * (y - mean_y)))
    return num / denom


def _last_k(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return values
    k = max(1, min(int(k), values.size))
    return values[-k:]


def _range_last_k(values: np.ndarray, k: int) -> float:
    window = _last_k(values, k)
    if window.size == 0:
        return float("nan")
    return float(np.max(window) - np.min(window))


def _sustained_increase(values: np.ndarray, window: int, min_increase: float) -> bool:
    window_vals = _last_k(values, window)
    if window_vals.size < 2:
        return False
    if float(window_vals[-1] - window_vals[0]) <= min_increase:
        return False
    return bool(np.all(window_vals[1:] > window_vals[:-1]))


def _find_sustained_windows(
    x: np.ndarray, y: np.ndarray, threshold: float, window: int
) -> List[Tuple[float, float]]:
    if y.size == 0:
        return []
    windows: List[Tuple[float, float]] = []
    start_idx: Optional[int] = None
    for idx, val in enumerate(y):
        if val > threshold:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None and idx - start_idx >= window:
                windows.append((float(x[start_idx]), float(x[idx - 1])))
            start_idx = None
    if start_idx is not None and y.size - start_idx >= window:
        windows.append((float(x[start_idx]), float(x[-1])))
    return windows


def _series_stats(x: np.ndarray, y: np.ndarray, last_k: int) -> Dict[str, float]:
    x_f, y_f = _finite_xy(x, y)
    if y_f.size == 0:
        return {"count": 0.0, "last": math.nan, "min": math.nan, "max": math.nan, "slope": math.nan}
    return {
        "count": float(y_f.size),
        "last": float(y_f[-1]),
        "min": float(np.min(y_f)),
        "max": float(np.max(y_f)),
        "slope": _last_k_slope(x_f, y_f, last_k),
    }


def _fmt(val: float) -> str:
    if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        return "-"
    return f"{val:.4g}"


def _annotation_text(
    series_stats: Dict[str, Dict[str, float]],
    *,
    last_k: int,
    probe_points: Optional[int] = None,
    outer_iters: Optional[int] = None,
) -> str:
    lines = []
    for name, stats in series_stats.items():
        lines.append(
            f"{name}: last={_fmt(stats['last'])}, min={_fmt(stats['min'])}, "
            f"max={_fmt(stats['max'])}, slope@{last_k}={_fmt(stats['slope'])}"
        )
    if probe_points is not None and outer_iters:
        coverage = probe_points / max(outer_iters, 1)
        lines.append(f"probe_points={probe_points} ({coverage:.1%} of {outer_iters})")
    return "\n".join(lines)


def _plot_series(
    *,
    x: np.ndarray,
    series: Dict[str, np.ndarray],
    title: str,
    out_path: Path,
    annotation: Optional[str] = None,
    hline: Optional[float] = None,
    sustained_windows: Optional[List[Tuple[float, float]]] = None,
    vlines: Optional[List[float]] = None,
) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for label, y in series.items():
        plt.plot(x, y, linewidth=1.6, label=label)
    if vlines:
        for xval in vlines:
            plt.axvline(xval, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
    if hline is not None:
        plt.axhline(hline, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    if sustained_windows:
        for idx, (start, end) in enumerate(sustained_windows):
            plt.axvspan(start, end, color="orange", alpha=0.15)
            if idx == 0:
                plt.gca().text(
                    start,
                    0.98,
                    "sustained",
                    transform=plt.gca().get_xaxis_transform(),
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="darkorange",
                )
    plt.title(title)
    plt.xlabel("iter")
    plt.grid(alpha=0.3)
    if len(series) > 1:
        plt.legend(fontsize=8)
    if annotation:
        plt.gca().text(
            0.02,
            0.98,
            annotation,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_td_loss_qhat(
    *,
    td_iters: np.ndarray,
    td_loss: np.ndarray,
    q_iters: np.ndarray,
    td_loss_from_q: np.ndarray,
    abs_diff: np.ndarray,
    rel_diff: np.ndarray,
    out_path: Path,
    vlines: Optional[List[float]] = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(td_iters, td_loss, label="td_loss", linewidth=1.6)
    axes[0].plot(
        q_iters,
        td_loss_from_q,
        label="td_loss_from_Q",
        linewidth=1.2,
        marker="o",
        markersize=3,
    )
    if vlines:
        for xval in vlines:
            axes[0].axvline(xval, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[0].set_ylabel("td_loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(q_iters, abs_diff, label="abs_diff", linewidth=1.2, marker="o", markersize=3)
    axes[1].plot(q_iters, rel_diff, label="rel_diff", linewidth=1.2, marker="o", markersize=3)
    if vlines:
        for xval in vlines:
            axes[1].axvline(xval, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[1].set_xlabel("iter")
    axes[1].set_ylabel("diff")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("td_loss vs td_loss_from_Q")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _probe_points(probe_csv: Path) -> int:
    data = _load_csv(probe_csv)
    if not data:
        return 0
    return len(data.get("iter", []))


def _write_merged_table(
    *,
    curves: Dict[str, List[float]],
    probes_dir: Path,
    out_path: Path,
) -> None:
    if not curves:
        return
    iters = curves.get("iter", list(range(len(next(iter(curves.values()))))))
    base_columns = [
        "iter",
        "td_loss",
        "w_norm",
        "tracking_gap",
        "teacher_error",
        "mean_rho",
        "mean_rho2",
        "min_rho",
        "max_rho",
        "p95_rho",
        "p99_rho",
        "p95_rho2",
        "p99_rho2",
        "max_rho2",
        "delta_mean",
        "delta_std",
        "delta_p95",
        "delta_p99",
        "delta_max",
    ]
    probe_maps = {
        "stability": _probe_map(probes_dir / "stability_probe.csv"),
        "fixed_point": _probe_map(probes_dir / "fixed_point_probe.csv"),
        "distribution": _probe_map(probes_dir / "distribution_probe.csv"),
        "q_kernel": _probe_map(probes_dir / "q_kernel_probe.csv"),
    }
    probe_columns: List[str] = []
    for mapping in probe_maps.values():
        for row in mapping.values():
            for key in row.keys():
                if key == "iter":
                    continue
                if key not in probe_columns:
                    probe_columns.append(key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_columns + probe_columns)
        writer.writeheader()
        for idx, raw_iter in enumerate(iters):
            if not math.isfinite(raw_iter):
                continue
            iter_key = int(raw_iter)
            row = {
                "iter": raw_iter,
                "td_loss": curves.get("td_loss", [math.nan] * len(iters))[idx],
                "w_norm": curves.get("w_norm", [math.nan] * len(iters))[idx],
                "tracking_gap": curves.get("tracking_gap", [math.nan] * len(iters))[idx],
                "teacher_error": curves.get("critic_teacher_error", [math.nan] * len(iters))[idx],
                "mean_rho": curves.get("mean_rho", [math.nan] * len(iters))[idx],
                "mean_rho2": curves.get("mean_rho2", [math.nan] * len(iters))[idx],
                "min_rho": curves.get("min_rho", [math.nan] * len(iters))[idx],
                "max_rho": curves.get("max_rho", [math.nan] * len(iters))[idx],
                "p95_rho": curves.get("p95_rho", [math.nan] * len(iters))[idx],
                "p99_rho": curves.get("p99_rho", [math.nan] * len(iters))[idx],
                "p95_rho2": curves.get("p95_rho2", [math.nan] * len(iters))[idx],
                "p99_rho2": curves.get("p99_rho2", [math.nan] * len(iters))[idx],
                "max_rho2": curves.get("max_rho2", [math.nan] * len(iters))[idx],
                "delta_mean": curves.get("delta_mean", [math.nan] * len(iters))[idx],
                "delta_std": curves.get("delta_std", [math.nan] * len(iters))[idx],
                "delta_p95": curves.get("delta_p95", [math.nan] * len(iters))[idx],
                "delta_p99": curves.get("delta_p99", [math.nan] * len(iters))[idx],
                "delta_max": curves.get("delta_max", [math.nan] * len(iters))[idx],
            }
            for mapping in probe_maps.values():
                probe_row = mapping.get(iter_key)
                if not probe_row:
                    continue
                for key, value in probe_row.items():
                    if key == "iter":
                        continue
                    row[key] = value
            writer.writerow(row)


def _metric_summary(name: str, stats: Dict[str, float], last_k: int) -> str:
    return (
        f"{name}: last={_fmt(stats['last'])}, min={_fmt(stats['min'])}, "
        f"max={_fmt(stats['max'])}, slope@{last_k}={_fmt(stats['slope'])}"
    )


def _evidence_metrics(curves: Dict[str, List[float]], probes_dir: Path) -> Dict[str, object]:
    td_loss = np.asarray(curves.get("td_loss", []), dtype=float)
    w_norm = np.asarray(curves.get("w_norm", []), dtype=float)
    iters = np.asarray(curves.get("iter", np.arange(td_loss.size)), dtype=float)

    stab = _load_csv(probes_dir / "stability_probe.csv")
    if "stability_proxy_mean" in stab:
        stability_series = np.asarray(stab["stability_proxy_mean"], dtype=float)
    else:
        stability_series = np.asarray(stab.get("stability_proxy", []), dtype=float)

    stability_margin = math.nan
    if stability_series.size:
        stab_window = _last_k(stability_series, PROBE_WINDOW)
        if stab_window.size:
            stability_margin = float(np.max(stab_window) - (1.0 + STABILITY_EPS))

    fixed = _load_csv(probes_dir / "fixed_point_probe.csv")
    fixed_iters = np.asarray(
        fixed.get("iter", np.arange(len(fixed.get("w_sharp_drift", [])))),
        dtype=float,
    )
    w_sharp_drift = np.asarray(fixed.get("w_sharp_drift", []), dtype=float)
    w_gap = np.asarray(fixed.get("w_gap", []), dtype=float)

    drift_slope = math.nan
    if w_sharp_drift.size and fixed_iters.size:
        fx = _last_k(fixed_iters, PROBE_WINDOW)
        fy = _last_k(w_sharp_drift, PROBE_WINDOW)
        drift_slope = _linear_slope(fx, fy) if fx.size and fy.size else math.nan

    w_gap_min = math.nan
    if w_gap.size:
        w_gap_window = _last_k(w_gap, PROBE_WINDOW)
        if w_gap_window.size:
            w_gap_min = float(np.min(w_gap_window))

    td_slope = math.nan
    if td_loss.size and iters.size:
        tx = _last_k(iters, TD_SLOPE_WINDOW)
        ty = _last_k(td_loss, TD_SLOPE_WINDOW)
        td_slope = _linear_slope(tx, ty) if tx.size and ty.size else math.nan

    td_flat = math.isfinite(td_slope) and abs(td_slope) <= TD_SLOPE_FLAT_MAX
    td_range = _range_last_k(td_loss, SUSTAINED_WINDOW)
    td_flat_sustained = math.isfinite(td_range) and td_range <= TD_FLAT_RANGE_MAX
    w_gap_ok = math.isfinite(w_gap_min) and w_gap_min >= W_GAP_MIN

    w_norm_increasing = _sustained_increase(w_norm, SUSTAINED_WINDOW, W_NORM_INCREASE_MIN)
    td_loss_increasing = _sustained_increase(td_loss, SUSTAINED_WINDOW, TD_INCREASE_MIN)

    plateau_score = td_flat and td_flat_sustained and w_gap_ok and math.isfinite(drift_slope) and drift_slope > 0
    instability_candidate = (math.isfinite(stability_margin) and stability_margin > 0) or w_norm_increasing or td_loss_increasing
    tracking_limited_plateau_candidate = plateau_score

    probe_counts = {
        "stability": _probe_points(probes_dir / "stability_probe.csv"),
        "fixed_point": _probe_points(probes_dir / "fixed_point_probe.csv"),
        "distribution": _probe_points(probes_dir / "distribution_probe.csv"),
        "q_kernel": _probe_points(probes_dir / "q_kernel_probe.csv"),
    }

    q_kernel = _load_csv(probes_dir / "q_kernel_probe.csv")
    td_q_abs = np.asarray(q_kernel.get("td_loss_from_Q_abs_diff", []), dtype=float)
    td_q_rel = np.asarray(q_kernel.get("td_loss_from_Q_rel_diff", []), dtype=float)
    td_q_abs_last = float(td_q_abs[-1]) if td_q_abs.size else math.nan
    td_q_rel_last = float(td_q_rel[-1]) if td_q_rel.size else math.nan

    return {
        "probe_counts": probe_counts,
        "stability_margin": stability_margin,
        "drift_slope": drift_slope,
        "td_slope": td_slope,
        "w_gap_min": w_gap_min,
        "td_flat_sustained": td_flat_sustained,
        "plateau_score": plateau_score,
        "instability_candidate": instability_candidate,
        "tracking_limited_plateau_candidate": tracking_limited_plateau_candidate,
        "w_norm_increasing": w_norm_increasing,
        "td_loss_increasing": td_loss_increasing,
        "td_loss_from_Q_abs_diff_last": td_q_abs_last,
        "td_loss_from_Q_rel_diff_last": td_q_rel_last,
    }


def _run_analysis(run_dir: Path, out_dir: Path, last_k: int) -> Dict[str, object]:
    curves = _load_csv(run_dir / "learning_curves.csv")
    probes_dir = run_dir / "probes"
    config = _load_config(run_dir)
    env_cfg = config.get("env", {}) if isinstance(config.get("env", {}), dict) else {}
    probe_iter_markers = _probe_iters(
        [
            probes_dir / "stability_probe.csv",
            probes_dir / "fixed_point_probe.csv",
            probes_dir / "distribution_probe.csv",
            probes_dir / "q_kernel_probe.csv",
        ]
    )

    outer_iters = int(config.get("outer_iters", 0) or len(curves.get("iter", [])))
    probes_cfg = config.get("probes", {}) if isinstance(config.get("probes", {}), dict) else {}
    probes_every = int(probes_cfg.get("every", 0) or 0)
    feature_dim = int(env_cfg.get("feature_dim", 0) or 0)
    trajectories = int(config.get("trajectories", 0) or 0)
    horizon = int(config.get("horizon", 0) or 0)
    alpha_w = float(config.get("alpha_w", 0.0) or 0.0)

    run_out = out_dir / run_dir.name
    run_out.mkdir(parents=True, exist_ok=True)
    _write_merged_table(curves=curves, probes_dir=probes_dir, out_path=run_out / "merged_metrics.csv")

    curve_metrics = ["td_loss", "w_norm", "mean_rho2", "tracking_gap", "critic_teacher_error"]
    curve_stats: Dict[str, Dict[str, float]] = {}
    for metric in curve_metrics:
        if metric not in curves:
            continue
        x = np.asarray(curves.get("iter", list(range(len(curves[metric])))), dtype=float)
        y = np.asarray(curves[metric], dtype=float)
        stats = _series_stats(x, y, last_k)
        curve_stats[metric] = stats
        annotation = _annotation_text({metric: stats}, last_k=last_k)
        _plot_series(
            x=x,
            series={metric: y},
            title=f"{run_dir.name}: {metric}",
            out_path=run_out / f"curves_{metric}.png",
            annotation=annotation,
            vlines=probe_iter_markers,
        )

    # Distribution probe
    dist_stats: Dict[str, Dict[str, float]] = {}
    dist_path = probes_dir / "distribution_probe.csv"
    if dist_path.exists():
        dist = _load_csv(dist_path)
        x = np.asarray(dist.get("iter", list(range(len(dist.get("mmd2", []))))), dtype=float)
        series = {}
        for key, label in [
            ("mmd2", "dist_mmd2"),
            ("mean_l2", "dist_mean_l2"),
            ("dist_action_kl", "dist_action_kl"),
            ("dist_action_tv", "dist_action_tv"),
        ]:
            if key in dist:
                y = np.asarray(dist[key], dtype=float)
                series[label] = y
                dist_stats[label] = _series_stats(x, y, last_k)
        if series:
            annotation = _annotation_text(
                dist_stats,
                last_k=last_k,
                probe_points=_probe_points(dist_path),
                outer_iters=outer_iters,
            )
            _plot_series(
                x=x,
                series=series,
                title=f"{run_dir.name}: distribution_probe",
                out_path=run_out / "probes_distribution.png",
                annotation=annotation,
            )

    # Fixed point probe
    fixed_stats: Dict[str, Dict[str, float]] = {}
    fixed_path = probes_dir / "fixed_point_probe.csv"
    if fixed_path.exists():
        fixed = _load_csv(fixed_path)
        x = np.asarray(fixed.get("iter", list(range(len(fixed.get("w_gap", []))))), dtype=float)
        series = {}
        for key, label in [("w_gap", "fixed_point_gap"), ("w_sharp_drift", "fixed_point_drift")]:
            if key in fixed:
                y = np.asarray(fixed[key], dtype=float)
                series[label] = y
                fixed_stats[label] = _series_stats(x, y, last_k)
        if series:
            annotation = _annotation_text(
                fixed_stats,
                last_k=last_k,
                probe_points=_probe_points(fixed_path),
                outer_iters=outer_iters,
            )
            _plot_series(
                x=x,
                series=series,
                title=f"{run_dir.name}: fixed_point_probe",
                out_path=run_out / "probes_fixed_point.png",
                annotation=annotation,
            )

    # Stability probe
    stab_stats: Dict[str, Dict[str, float]] = {}
    stab_path = probes_dir / "stability_probe.csv"
    if stab_path.exists():
        stab = _load_csv(stab_path)
        x = np.asarray(stab.get("iter", list(range(len(stab.get("stability_proxy", []))))), dtype=float)
        series = {}
        if "stability_proxy_mean" in stab:
            y = np.asarray(stab["stability_proxy_mean"], dtype=float)
            series["stability_proxy_mean"] = y
            stab_stats["stability_proxy_mean"] = _series_stats(x, y, last_k)
        elif "stability_proxy" in stab:
            y = np.asarray(stab["stability_proxy"], dtype=float)
            series["stability_proxy"] = y
            stab_stats["stability_proxy"] = _series_stats(x, y, last_k)
        if "stability_proxy_std" in stab:
            y_std = np.asarray(stab["stability_proxy_std"], dtype=float)
            series["stability_proxy_std"] = y_std
            stab_stats["stability_proxy_std"] = _series_stats(x, y_std, last_k)
        if series:
            main_key = "stability_proxy_mean" if "stability_proxy_mean" in stab_stats else "stability_proxy"
            series_main = np.asarray(stab.get(main_key, []), dtype=float)
            sustained = _find_sustained_windows(x, series_main, 1.0 + STABILITY_EPS, SUSTAINED_WINDOW)
            annotation = _annotation_text(
                stab_stats,
                last_k=last_k,
                probe_points=_probe_points(stab_path),
                outer_iters=outer_iters,
            )
            _plot_series(
                x=x,
                series=series,
                title=f"{run_dir.name}: stability_probe",
                out_path=run_out / "probes_stability.png",
                annotation=annotation,
                hline=1.0 + STABILITY_EPS,
                sustained_windows=sustained,
            )

    # Q-kernel probe
    q_kernel_stats: Dict[str, Dict[str, float]] = {}
    q_kernel_path = probes_dir / "q_kernel_probe.csv"
    if q_kernel_path.exists():
        q_kernel = _load_csv(q_kernel_path)
        q_iters = np.asarray(q_kernel.get("iter", []), dtype=float)
        td_loss = np.asarray(curves.get("td_loss", []), dtype=float)
        td_iters = np.asarray(curves.get("iter", np.arange(td_loss.size)), dtype=float)
        td_loss_from_q = np.asarray(q_kernel.get("td_loss_from_Q", []), dtype=float)
        abs_diff = np.asarray(q_kernel.get("td_loss_from_Q_abs_diff", []), dtype=float)
        rel_diff = np.asarray(q_kernel.get("td_loss_from_Q_rel_diff", []), dtype=float)
        for key, series in [
            ("td_loss_from_Q", td_loss_from_q),
            ("td_loss_from_Q_abs_diff", abs_diff),
            ("td_loss_from_Q_rel_diff", rel_diff),
        ]:
            if q_iters.size and series.size:
                q_kernel_stats[key] = _series_stats(q_iters, series, last_k)
        if td_iters.size and q_iters.size:
            _plot_td_loss_qhat(
                td_iters=td_iters,
                td_loss=td_loss,
                q_iters=q_iters,
                td_loss_from_q=td_loss_from_q,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
                out_path=run_out / "curves_td_loss_qhat.png",
                vlines=probe_iter_markers,
            )

    probe_counts = {
        "distribution": _probe_points(dist_path),
        "fixed_point": _probe_points(fixed_path),
        "stability": _probe_points(stab_path),
        "q_kernel": _probe_points(q_kernel_path),
    }
    evidence = _evidence_metrics(curves, probes_dir)

    return {
        "run_dir": str(run_dir),
        "outer_iters": outer_iters,
        "probes_every": probes_every,
        "feature_dim": feature_dim,
        "trajectories": trajectories,
        "horizon": horizon,
        "alpha_w": alpha_w,
        "curve_stats": curve_stats,
        "dist_stats": dist_stats,
        "fixed_stats": fixed_stats,
        "stability_stats": stab_stats,
        "q_kernel_stats": q_kernel_stats,
        "probe_counts": probe_counts,
        "evidence": evidence,
    }


def _scale_note(summary: Dict[str, object]) -> Tuple[str, float]:
    run_dir = Path(str(summary.get("run_dir", "")))
    alpha_w = float(summary.get("alpha_w", 0.0) or 0.0)
    trajectories = int(summary.get("trajectories", 0) or 0)
    horizon = int(summary.get("horizon", 0) or 0)
    total_steps = max(trajectories * horizon, 1)
    train_step_scale = alpha_w / total_steps if alpha_w > 0 else float("nan")
    stability_scale = float("nan")
    probe_path = run_dir / "probes" / "stability_probe.csv"
    if probe_path.exists():
        probe = _load_csv(probe_path)
        if "stability_probe_step_scale" in probe and probe["stability_probe_step_scale"]:
            stability_scale = float(probe["stability_probe_step_scale"][-1])
    ratio = stability_scale / train_step_scale if math.isfinite(stability_scale) and math.isfinite(train_step_scale) else float("nan")
    code = (
        "alpha_w = {alpha_w}\n"
        "trajectories = {trajectories}\n"
        "horizon = {horizon}\n"
        "train_step_scale = alpha_w / (trajectories * horizon)\n"
        "stability_probe_step_scale = {stability_scale}\n"
        "ratio = stability_probe_step_scale / train_step_scale\n"
    ).format(
        alpha_w=alpha_w,
        trajectories=trajectories,
        horizon=horizon,
        stability_scale=stability_scale,
    )
    return code, ratio


def main() -> None:
    args = parse_args()
    plateau_run = Path(args.plateau_run)
    instability_run = Path(args.instability_run)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if plateau_run.parent == instability_run.parent:
            out_dir = plateau_run.parent / "stepC_analysis"
        else:
            out_dir = Path.cwd() / "stepC_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    plateau_summary = _run_analysis(plateau_run, out_dir, args.last_k)
    instability_summary = _run_analysis(instability_run, out_dir, args.last_k)
    plateau_ev: Dict[str, object] = plateau_summary["evidence"]  # type: ignore[assignment]
    instability_ev: Dict[str, object] = instability_summary["evidence"]  # type: ignore[assignment]

    def _coverage(count: int, outer_iters: int) -> str:
        if outer_iters <= 0:
            return "-"
        return f"{count} ({count / outer_iters:.1%} of {outer_iters})"

    plateau_scale_code, plateau_ratio = _scale_note(plateau_summary)
    instability_scale_code, instability_ratio = _scale_note(instability_summary)

    def _rho_note(summary: Dict[str, object]) -> str:
        curves = _load_csv(Path(str(summary.get("run_dir", ""))) / "learning_curves.csv")
        mean_rho_series = np.asarray(curves.get("mean_rho", []), dtype=float)
        mean_rho_last = float(mean_rho_series[-1]) if mean_rho_series.size else math.nan
        clip_series = np.asarray(curves.get("clip_fraction", []), dtype=float)
        clip_last = float(clip_series[-1]) if clip_series.size else math.nan
        config = _load_config(Path(str(summary.get("run_dir", ""))))
        env_cfg = config.get("env", {}) if isinstance(config.get("env", {}), dict) else {}
        p_mix = float(env_cfg.get("p_mix", math.nan)) if env_cfg else math.nan
        rho_clip = config.get("rho_clip", None)
        disable_clip = bool(config.get("disable_rho_clip", False))
        clip_active = rho_clip is not None and float(rho_clip) > 0 and not disable_clip
        if math.isfinite(mean_rho_last) and abs(mean_rho_last - 1.0) <= 0.05:
            status = "mean_rho close to 1 (consistent with mu-sampled actions)."
        elif math.isfinite(mean_rho_last):
            status = f"mean_rho deviates from 1 (last={mean_rho_last:.4g})."
        else:
            status = "mean_rho unavailable."
        clip_note = "rho_clip active" if clip_active else "rho_clip inactive"
        if math.isfinite(clip_last) and clip_last > 0.0:
            action_note = f"env clipping still active (clip_fraction last={clip_last:.3g})."
        else:
            action_note = "action squashing consistent with log_prob."
        return (
            f"{status} {action_note} p_mix={p_mix:.3g} changes state distribution but not the per-state identity. ({clip_note})"
        )

    def _lines_for(summary: Dict[str, object]) -> List[str]:
        last_k = args.last_k
        curve_stats: Dict[str, Dict[str, float]] = summary["curve_stats"]  # type: ignore[assignment]
        dist_stats: Dict[str, Dict[str, float]] = summary["dist_stats"]  # type: ignore[assignment]
        fixed_stats: Dict[str, Dict[str, float]] = summary["fixed_stats"]  # type: ignore[assignment]
        stability_stats: Dict[str, Dict[str, float]] = summary["stability_stats"]  # type: ignore[assignment]
        q_kernel_stats: Dict[str, Dict[str, float]] = summary["q_kernel_stats"]  # type: ignore[assignment]
        probe_counts: Dict[str, int] = summary["probe_counts"]  # type: ignore[assignment]
        evidence: Dict[str, object] = summary["evidence"]  # type: ignore[assignment]
        outer_iters = int(summary.get("outer_iters", 0) or 0)
        probes_every = int(summary.get("probes_every", 0) or 0)

        lines = []
        lines.append("- Observations:")
        for metric in ["td_loss", "w_norm", "mean_rho2", "tracking_gap", "critic_teacher_error"]:
            if metric in curve_stats:
                lines.append(f"  - { _metric_summary(metric, curve_stats[metric], last_k) }")
        if stability_stats:
            stab_key = None
            if "stability_proxy_mean" in stability_stats:
                stab_key = "stability_proxy_mean"
            elif "stability_proxy" in stability_stats:
                stab_key = "stability_proxy"
            if stab_key:
                lines.append(
                    f"  - {stab_key} (probe): "
                    + _metric_summary(stab_key, stability_stats[stab_key], last_k)
                    + f", coverage={_coverage(probe_counts['stability'], outer_iters)}"
                )
        if fixed_stats:
            drift = fixed_stats.get("fixed_point_drift", {})
            lines.append(
                "  - fixed_point_drift (probe): "
                + _metric_summary("fixed_point_drift", drift, last_k)
                + f", coverage={_coverage(probe_counts['fixed_point'], outer_iters)}"
            )
        if dist_stats:
            for key in ["dist_mmd2", "dist_mean_l2", "dist_action_kl", "dist_action_tv"]:
                if key in dist_stats:
                    lines.append(
                        "  - "
                        + _metric_summary(key, dist_stats[key], last_k)
                        + f", coverage={_coverage(probe_counts['distribution'], outer_iters)}"
                    )
        if q_kernel_stats:
            for key in ["td_loss_from_Q", "td_loss_from_Q_abs_diff", "td_loss_from_Q_rel_diff"]:
                if key in q_kernel_stats:
                    lines.append(
                        "  - "
                        + _metric_summary(key, q_kernel_stats[key], last_k)
                        + f", coverage={_coverage(probe_counts.get('q_kernel', 0), outer_iters)}"
                    )

        lines.append("- Missing evidence:")
        probe_max = max(probe_counts.values()) if probe_counts else 0
        if probe_max < 3:
            lines.append(
                "  - probe coverage is sparse (<3 points), trends for drift/stability cannot be trusted."
            )
        elif probe_max < 5:
            lines.append("  - probe coverage is limited (<5 points), trend estimates remain weak.")
        elif probe_max < 10:
            lines.append("  - probe coverage is still light (<10 points); evidence chains are weaker.")
        if probe_counts.get("fixed_point", 0) < 2:
            lines.append("  - fixed_point_drift trend cannot be established without multiple probe points.")

        lines.append("- Evidence chain:")
        lines.append(
            "  - stability_margin="
            f"{_fmt(evidence.get('stability_margin'))} "
            f"(>0 -> instability candidate)"
        )
        lines.append(
            "  - drift_slope="
            f"{_fmt(evidence.get('drift_slope'))} "
            f"(>0 supports plateau drift)"
        )
        lines.append(
            "  - td_loss_slope="
            f"{_fmt(evidence.get('td_slope'))} "
            f"(|slope|<{TD_SLOPE_FLAT_MAX:g} -> flat)"
        )
        lines.append(
            "  - w_gap_min_last_window="
            f"{_fmt(evidence.get('w_gap_min'))} "
            f"(>= {W_GAP_MIN:g} -> tracking gap persists)"
        )
        return lines

    def _absence_reason(ev: Dict[str, object], probe_counts: Dict[str, int], mode: str) -> str:
        probe_min = min(probe_counts.values()) if probe_counts else 0
        if probe_min < 10:
            return "instrumentation: probe_points<10; increase probes.every or run longer."
        if mode == "instability":
            if probe_counts.get("stability", 0) < 2:
                return "instrumentation: stability_probe has <2 points; cannot resolve stability_margin."
            reasons = []
            stability_margin = ev.get("stability_margin")
            if not isinstance(stability_margin, (int, float)) or not math.isfinite(float(stability_margin)):
                reasons.append("stability_margin unavailable")
            elif float(stability_margin) <= 0:
                reasons.append("stability_margin<=0")
            if not ev.get("w_norm_increasing") and not ev.get("td_loss_increasing"):
                reasons.append("no sustained w_norm/td_loss increase")
            if reasons:
                return "parameter regime: " + "; ".join(reasons)
            return "instrumentation: insufficient probe coverage."
        if probe_counts.get("fixed_point", 0) < 2:
            return "instrumentation: fixed_point_probe has <2 points; drift_slope unreliable."
        reasons = []
        drift_slope = ev.get("drift_slope")
        if not isinstance(drift_slope, (int, float)) or not math.isfinite(float(drift_slope)) or float(drift_slope) <= 0:
            reasons.append("drift_slope<=0")
        w_gap_min = ev.get("w_gap_min")
        if not isinstance(w_gap_min, (int, float)) or not math.isfinite(float(w_gap_min)) or float(w_gap_min) < W_GAP_MIN:
            reasons.append("w_gap_min below threshold")
        td_slope = ev.get("td_slope")
        if not isinstance(td_slope, (int, float)) or not math.isfinite(float(td_slope)) or abs(float(td_slope)) > TD_SLOPE_FLAT_MAX:
            reasons.append("td_loss slope not flat")
        if reasons:
            return "parameter regime: " + "; ".join(reasons)
        return "instrumentation: insufficient probe coverage."

    plateau_present = bool(plateau_ev.get("tracking_limited_plateau_candidate"))
    instability_present = bool(instability_ev.get("instability_candidate"))
    plateau_probe_counts = plateau_ev.get("probe_counts", plateau_summary.get("probe_counts", {}))
    instability_probe_counts = instability_ev.get("probe_counts", instability_summary.get("probe_counts", {}))

    if plateau_present and instability_present:
        reason_line = "n/a"
    elif not plateau_present and not instability_present:
        reason_line = (
            "plateau: "
            + _absence_reason(plateau_ev, plateau_probe_counts, "plateau")
            + "; instability: "
            + _absence_reason(instability_ev, instability_probe_counts, "instability")
        )
    elif not plateau_present:
        reason_line = "plateau: " + _absence_reason(plateau_ev, plateau_probe_counts, "plateau")
    else:
        reason_line = "instability: " + _absence_reason(instability_ev, instability_probe_counts, "instability")

    diff_files = sorted({f for f in args.diff_files if f})
    key_paths = [p for p in args.key_paths if p]

    report_lines = [
        "# Step C analysis (plateau vs instability)",
        "",
        "## Repo diff (files)",
        *(["- " + path for path in diff_files] if diff_files else ["- (not provided)"]),
        "",
        "## Key outputs",
        *(["- " + path for path in key_paths] if key_paths else ["- (not provided)"]),
        "",
        f"Artifacts: {out_dir}",
        "",
        f"## plateau ({plateau_summary['run_dir']})",
        *_lines_for(plateau_summary),
        "",
        f"## instability ({instability_summary['run_dir']})",
        *_lines_for(instability_summary),
        "",
        "## Scale check (training vs stability_probe)",
        "",
        "plateau:",
        "```python",
        plateau_scale_code,
        "```",
        f"ratio (probe/train) = {_fmt(plateau_ratio)}",
        "",
        "instability:",
        "```python",
        instability_scale_code,
        "```",
        f"ratio (probe/train) = {_fmt(instability_ratio)}",
        "",
        "## Metric alignment notes",
        "",
        "- td_loss in training is mean(delta^2) over all steps; it matches (1/T) * sum_t E[Delta(t)^2].",
        "- td_loss_from_Q uses cached Delta to compute Q_hat(t,t') = E_b[Delta(t)Delta(t')] and",
        "  td_loss_from_Q = (1/(2T_cache)) * sum_t Q_hat(t,t); expect ~0.5 * td_loss.",
        "- rho consistency checks:",
        f"  - plateau: {_rho_note(plateau_summary)}",
        f"  - instability: {_rho_note(instability_summary)}",
        "",
        "## Verdict",
        "",
        "- instability evidence: "
        + ("present" if instability_present else "absent")
        + f" (stability_margin={_fmt(instability_ev.get('stability_margin'))}, "
        + f"w_norm_increasing={instability_ev.get('w_norm_increasing')}, "
        + f"td_loss_increasing={instability_ev.get('td_loss_increasing')})",
        "- plateau (tracking-limited) evidence: "
        + ("present" if plateau_present else "absent")
        + f" (drift_slope={_fmt(plateau_ev.get('drift_slope'))}, "
        + f"w_gap_min={_fmt(plateau_ev.get('w_gap_min'))}, "
        + f"td_loss_slope={_fmt(plateau_ev.get('td_slope'))})",
        "- if absent, likely reason: " + reason_line,
    ]

    (out_dir / "stepC_analysis.md").write_text("\n".join(report_lines))
    print(f"Saved Step C analysis to {out_dir}")


if __name__ == "__main__":
    main()
