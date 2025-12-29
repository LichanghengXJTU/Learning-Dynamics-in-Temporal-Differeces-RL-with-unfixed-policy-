#!/usr/bin/env python3
"""Inspect a run directory and suggest corrective actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

STABILITY_EPS = 1e-3
PROBE_WINDOW = 5
TD_SLOPE_WINDOW = 20
TD_SLOPE_FLAT_MAX = 1e-6
TD_FLAT_RANGE_MAX = 1e-6
TD_INCREASE_MIN = 1e-6
W_NORM_INCREASE_MIN = 1e-4
W_GAP_MIN = 1e-3
FIXED_DRIFT_SLOPE_MIN = 1e-3
SUSTAINED_WINDOW = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check run_report metrics and suggest fixes.")
    parser.add_argument("--run", type=str, required=True, help="Run directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="generic",
        choices=["generic", "on_policy", "plateau", "instability"],
        help="Expectation mode for checks",
    )
    parser.add_argument("--print-commands", action="store_true", help="Print suggested rerun commands")
    return parser.parse_args()


def _to_float(value: str) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(val):
        return val
    return None


def _last_finite_from_csv(csv_path: Path, column: str) -> Optional[float]:
    if not csv_path.exists():
        return None
    last_val = None
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get(column)
            if raw is None:
                continue
            val = _to_float(raw)
            if val is not None:
                last_val = val
    return last_val


def _range_from_csv(csv_path: Path, column: str) -> Optional[float]:
    if not csv_path.exists():
        return None
    values: List[float] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get(column)
            val = _to_float(raw)
            if val is not None:
                values.append(val)
    if not values:
        return None
    return max(values) - min(values)


def _count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _series_from_csv(csv_path: Path, column: str) -> List[float]:
    if not csv_path.exists():
        return []
    values: List[float] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get(column)
            val = _to_float(raw) if raw is not None else None
            if val is not None:
                values.append(val)
    return values


def _series_xy_from_csv(csv_path: Path, x_col: str, y_col: str) -> tuple[List[float], List[float]]:
    if not csv_path.exists():
        return [], []
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_raw = row.get(x_col)
            y_raw = row.get(y_col)
            x_val = _to_float(x_raw) if x_raw is not None else None
            y_val = _to_float(y_raw) if y_raw is not None else None
            if x_val is not None and y_val is not None:
                xs.append(x_val)
                ys.append(y_val)
    return xs, ys


def _last_k_pairs(xs: List[float], ys: List[float], k: int) -> tuple[List[float], List[float]]:
    if not xs or not ys:
        return [], []
    use_k = min(len(xs), len(ys), max(1, int(k)))
    return xs[-use_k:], ys[-use_k:]


def _first_last(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return values[0], values[-1]


def _slope(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    dx = xs[-1] - xs[0]
    if dx == 0:
        return None
    return (ys[-1] - ys[0]) / dx


def _linear_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return num / denom


def _last_k(values: List[float], k: int) -> List[float]:
    if not values:
        return []
    use_k = min(len(values), max(1, int(k)))
    return values[-use_k:]


def _range_last_k(values: List[float], k: int) -> Optional[float]:
    window = _last_k(values, k)
    if not window:
        return None
    return max(window) - min(window)


def _sustained_increase(values: List[float], window: int, min_increase: float) -> bool:
    window_vals = _last_k(values, window)
    if len(window_vals) < 2:
        return False
    if window_vals[-1] - window_vals[0] <= min_increase:
        return False
    return all(b > a for a, b in zip(window_vals, window_vals[1:]))


def _sustained_above(values: List[float], threshold: float, window: int) -> bool:
    if not values:
        return False
    use_window = min(window, len(values))
    return all(val > threshold for val in values[-use_window:])


def _load_run_config(run_dir: Path) -> Optional[Dict[str, object]]:
    path = run_dir / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_run_report(run_dir: Path) -> Optional[Dict[str, object]]:
    path = run_dir / "run_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _print_metrics(metrics: Dict[str, Optional[float]]) -> None:
    print("Metrics:")
    for key, val in metrics.items():
        if val is None:
            display = "-"
        else:
            display = f"{val:.4g}"
        print(f"  {key}: {display}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)

    curves_path = run_dir / "learning_curves.csv"
    probes_dir = run_dir / "probes"

    metrics = {
        "mean_rho2": _last_finite_from_csv(curves_path, "mean_rho2"),
        "stability_proxy": _last_finite_from_csv(curves_path, "stability_proxy"),
        "fixed_point_drift": _last_finite_from_csv(curves_path, "fixed_point_drift"),
        "dist_mmd2": _last_finite_from_csv(curves_path, "dist_mmd2"),
        "dist_action_kl": _last_finite_from_csv(curves_path, "dist_action_kl"),
        "dist_action_tv": _last_finite_from_csv(curves_path, "dist_action_tv"),
    }

    if probes_dir.exists():
        stability_probe = _last_finite_from_csv(probes_dir / "stability_probe.csv", "stability_proxy")
        if stability_probe is not None:
            metrics["stability_proxy"] = stability_probe
        fixed_point_probe = _last_finite_from_csv(probes_dir / "fixed_point_probe.csv", "w_sharp_drift")
        if fixed_point_probe is not None:
            metrics["fixed_point_drift"] = fixed_point_probe
        dist_probe = _last_finite_from_csv(probes_dir / "distribution_probe.csv", "mmd2")
        if dist_probe is not None:
            metrics["dist_mmd2"] = dist_probe
        dist_action_kl = _last_finite_from_csv(probes_dir / "distribution_probe.csv", "dist_action_kl")
        if dist_action_kl is not None:
            metrics["dist_action_kl"] = dist_action_kl
        dist_action_tv = _last_finite_from_csv(probes_dir / "distribution_probe.csv", "dist_action_tv")
        if dist_action_tv is not None:
            metrics["dist_action_tv"] = dist_action_tv

    print(f"Run: {run_dir}")
    print(f"stability_eps: {STABILITY_EPS:g}")
    _print_metrics(metrics)

    suggestions: List[str] = []
    rerun_cmds: List[str] = []
    warnings: List[str] = []

    run_cfg = _load_run_config(run_dir)
    outer_iters = None
    probes_every = None
    if run_cfg:
        outer_iters = int(run_cfg.get("outer_iters", 0) or 0)
        probes_cfg = run_cfg.get("probes", {})
        if isinstance(probes_cfg, dict):
            probes_every = int(probes_cfg.get("every", 0) or 0)
    if outer_iters is None or outer_iters <= 0:
        outer_iters = _count_rows(curves_path)

    probe_counts = {
        "stability_probe": _count_rows(probes_dir / "stability_probe.csv"),
        "fixed_point_probe": _count_rows(probes_dir / "fixed_point_probe.csv"),
        "distribution_probe": _count_rows(probes_dir / "distribution_probe.csv"),
    }
    probe_points = max(probe_counts.values()) if probe_counts else 0
    for name, count in probe_counts.items():
        if count < 10:
            warnings.append(f"WARNING: {name} has only {count} probe points (<10).")

    stability_series = _series_from_csv(probes_dir / "stability_probe.csv", "stability_proxy_mean")
    if not stability_series:
        stability_series = _series_from_csv(probes_dir / "stability_probe.csv", "stability_proxy")
    stability_last = stability_series[-1] if stability_series else None
    stability_window = _last_k(stability_series, PROBE_WINDOW)
    stability_margin = None
    if stability_window:
        stability_margin = max(stability_window) - (1.0 + STABILITY_EPS)

    fixed_x, fixed_y = _series_xy_from_csv(probes_dir / "fixed_point_probe.csv", "iter", "w_sharp_drift")
    fixed_x, fixed_y = _last_k_pairs(fixed_x, fixed_y, PROBE_WINDOW)
    drift_slope = _linear_slope(fixed_x, fixed_y) if fixed_x and fixed_y else None

    w_gap_series = _series_from_csv(probes_dir / "fixed_point_probe.csv", "w_gap")
    w_gap_window = _last_k(w_gap_series, PROBE_WINDOW)
    w_gap_min = min(w_gap_window) if w_gap_window else None
    w_gap_ok = w_gap_min is not None and w_gap_min >= W_GAP_MIN

    td_x, td_y = _series_xy_from_csv(curves_path, "iter", "td_loss")
    if not td_y:
        td_y = _series_from_csv(curves_path, "td_loss")
        td_x = list(range(len(td_y)))
    td_xk, td_yk = _last_k_pairs(td_x, td_y, TD_SLOPE_WINDOW)
    td_slope = _linear_slope(td_xk, td_yk) if td_xk and td_yk else None
    td_range = _range_last_k(td_y, SUSTAINED_WINDOW)
    td_flat = td_slope is not None and abs(td_slope) <= TD_SLOPE_FLAT_MAX
    td_flat_sustained = td_range is not None and td_range <= TD_FLAT_RANGE_MAX

    w_norm_series = _series_from_csv(curves_path, "w_norm")
    w_norm_increasing = _sustained_increase(w_norm_series, SUSTAINED_WINDOW, W_NORM_INCREASE_MIN)
    td_loss_increasing = _sustained_increase(td_y, SUSTAINED_WINDOW, TD_INCREASE_MIN)

    plateau_score = bool(td_flat and td_flat_sustained and w_gap_ok and (drift_slope is not None and drift_slope > 0))
    instability_candidate = bool((stability_margin is not None and stability_margin > 0) or w_norm_increasing or td_loss_increasing)
    tracking_limited_plateau_candidate = bool(
        (drift_slope is not None and drift_slope > 0) and w_gap_ok and td_flat and td_flat_sustained
    )

    print("\nEvidence chain:")
    print(f"  probe_points_count: {probe_counts}")
    print(f"  stability_margin: {stability_margin} (window={PROBE_WINDOW}, eps={STABILITY_EPS:g})")
    print(f"  drift_slope: {drift_slope} (window={PROBE_WINDOW})")
    print(f"  td_loss_slope_last_k: {td_slope} (k={TD_SLOPE_WINDOW})")
    print(f"  w_gap_min_last_window: {w_gap_min} (threshold={W_GAP_MIN:g})")
    print(f"  td_flat_sustained: {td_flat_sustained} (range<= {TD_FLAT_RANGE_MAX:g})")
    print(f"  plateau_score: {plateau_score}")
    print(f"  instability_candidate: {instability_candidate}")
    print(f"  tracking_limited_plateau_candidate: {tracking_limited_plateau_candidate}")

    report_payload = _load_run_report(run_dir) or {}
    report_payload["evidence_chain"] = {
        "probe_points_count": probe_counts,
        "stability_margin": stability_margin,
        "drift_slope": drift_slope,
        "td_loss_slope_last_k": td_slope,
        "w_gap_min_last_window": w_gap_min,
        "plateau_score": plateau_score,
        "instability_candidate": instability_candidate,
        "tracking_limited_plateau_candidate": tracking_limited_plateau_candidate,
        "stability_eps": STABILITY_EPS,
        "probe_window": PROBE_WINDOW,
        "td_slope_window": TD_SLOPE_WINDOW,
        "sustained_window": SUSTAINED_WINDOW,
        "w_gap_threshold": W_GAP_MIN,
        "td_slope_flat_max": TD_SLOPE_FLAT_MAX,
        "td_flat_range_max": TD_FLAT_RANGE_MAX,
        "td_increase_min": TD_INCREASE_MIN,
        "w_norm_increase_min": W_NORM_INCREASE_MIN,
    }
    (run_dir / "run_report.json").write_text(json.dumps(report_payload, indent=2, ensure_ascii=True))

    if any(val is not None and not math.isfinite(val) for val in metrics.values()):
        suggestions.append("Found NaN/Inf in metrics -> reduce alpha_w/alpha_pi by 10x and tighten theta_radius.")
        rerun_cmds.append(
            "python scripts/run_train.py --config configs/train_sanity.yaml --output-dir outputs/safe_rerun --alpha-w 0.01 --alpha-pi 0.005 --theta-radius 2.0"
        )

    if args.mode == "on_policy":
        rho = metrics.get("mean_rho2")
        dist_kl = metrics.get("dist_action_kl")
        dist_tv = metrics.get("dist_action_tv")
        if rho is None or abs(rho - 1.0) > 0.1:
            suggestions.append(
                "on_policy mean_rho2 not close to 1 -> verify mu=pi (beta=1), sigma_mu==sigma_pi, and on_policy config." 
            )
            rerun_cmds.append(
                "python scripts/run_train.py --config configs/train_sanity.yaml --output-dir outputs/on_policy_fix --beta 1.0 --sigma-mu 0.2 --sigma-pi 0.2"
            )
        if (dist_kl is not None and dist_kl > 1e-6) or (dist_tv is not None and dist_tv > 1e-4):
            suggestions.append("on_policy action divergence too large -> verify mu=pi and sigma_mu==sigma_pi.")
            rerun_cmds.append(
                "python scripts/run_train.py --config configs/train_sanity.yaml --output-dir outputs/on_policy_fix --beta 1.0 --sigma-mu 0.2 --sigma-pi 0.2"
            )

    if args.mode == "plateau":
        stability = stability_last if stability_last is not None else metrics.get("stability_proxy")
        if stability is not None and stability > 1.05:
            suggestions.append("plateau run unstable -> lower alpha_w/alpha_pi or increase beta.")
            rerun_cmds.append(
                "python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/short_runs/plateau_fix --alpha-w 0.04 --alpha-pi 0.03 --beta 0.1 --outer-iters 80"
            )

    if args.mode == "instability":
        td_series = td_y
        w_series = w_norm_series
        td_first, td_last = _first_last(td_series)
        w_first, w_last = _first_last(w_series)
        td_increase = td_last - td_first if td_first is not None and td_last is not None else None
        w_increase = w_last - w_first if w_first is not None and w_last is not None else None

        fp_slope = drift_slope
        stability_sustained = stability_margin is not None and stability_margin > 0

        td_ok = td_loss_increasing or (td_increase is not None and td_increase > TD_INCREASE_MIN)
        w_ok = w_norm_increasing or (w_increase is not None and w_increase > W_NORM_INCREASE_MIN)
        fp_ok = fp_slope is not None and fp_slope > FIXED_DRIFT_SLOPE_MIN

        if not (td_ok or w_ok or fp_ok or stability_sustained):
            def _fmt(val: Optional[float]) -> str:
                return "-" if val is None else f"{val:.4g}"

            suggestions.append(
                "instability run too stable -> increase alpha_w/alpha_pi or reduce beta/p_mix. "
                f"(td_increase={_fmt(td_increase)}>{TD_INCREASE_MIN:g}, "
                f"w_increase={_fmt(w_increase)}>{W_NORM_INCREASE_MIN:g}, "
                f"fixed_point_drift_slope={_fmt(fp_slope)}>{FIXED_DRIFT_SLOPE_MIN:g}, "
                f"stability_margin={_fmt(stability_margin)}>{0:g}, eps={STABILITY_EPS:g})"
            )
            rerun_cmds.append(
                "python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/short_runs/instability_fix --alpha-w 0.3 --alpha-pi 0.2 --beta 0.005 --p-mix 0.0 --outer-iters 80"
            )

    td_span = _range_from_csv(curves_path, "td_loss")
    w_span = _range_from_csv(curves_path, "w_norm")
    if td_span is not None and w_span is not None:
        if td_span < 1e-6 and w_span < 1e-6:
            suggestions.append(
                "metrics nearly constant -> increase lr or logging precision; verify you are reading the correct output dir."
            )
            rerun_cmds.append("head -n 5 " + str(curves_path))
            rerun_cmds.append("tail -n 5 " + str(curves_path))

    if suggestions:
        print("\nSuggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
    else:
        print("\nSuggestions:\n- No rule-based issues detected.")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if args.print_commands and rerun_cmds:
        print("\nRerun commands:")
        for cmd in rerun_cmds:
            print(cmd)


if __name__ == "__main__":
    main()
