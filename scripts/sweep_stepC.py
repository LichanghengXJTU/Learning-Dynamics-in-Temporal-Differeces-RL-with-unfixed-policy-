#!/usr/bin/env python3
"""Run a Step C sweep and summarize results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BETA_FACTORS = [0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 8.0]
DEFAULT_ALPHA_W_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
DEFAULT_THETA_MU_OFFSET_MIN = 0.0
DEFAULT_THETA_MU_OFFSET_MAX = 0.5


@dataclass
class RunSpec:
    run_id: str
    beta: float
    alpha_w: float
    theta_mu_offset_scale: float
    seed: Optional[int]
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Step C sweep (contract 10x10x10).")
    parser.add_argument("--base-config", type=str, default="configs/train_plateau.yaml")
    parser.add_argument("--out-root", type=str, default="outputs/base_check")
    parser.add_argument("--timestamp", type=str, default=None, help="Reuse an existing timestamp (YYYYmmdd_HHMMSS).")
    parser.add_argument("--run-root", type=str, default=None, help="Explicit sweep root (overrides out-root/timestamp).")
    parser.add_argument("--outer-iters", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--report-every-seconds", type=float, default=0.0)
    parser.add_argument("--grid", type=int, default=10, help="Points per axis.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds.")
    parser.add_argument("--beta-values", type=str, default=None, help="Comma-separated beta values.")
    parser.add_argument("--alpha-w-values", type=str, default=None, help="Comma-separated alpha_w values.")
    parser.add_argument(
        "--theta-mu-offset-values",
        type=str,
        default=None,
        help="Comma-separated theta_mu_offset_scale values.",
    )
    parser.add_argument("--eps-slope", type=float, default=1e-6, help="Plateau slope threshold.")
    parser.add_argument("--eps-drift", type=float, default=1e-4, help="Plateau drift threshold.")
    parser.add_argument("--offpolicy-threshold", type=float, default=1e-3, help="Off-policy score threshold.")
    parser.add_argument("--window", type=int, default=20, help="Window length for slope/range checks.")
    parser.add_argument("--stability-eps", type=float, default=1e-3, help="Stability probe epsilon.")
    parser.add_argument("--td-loss-blowup", type=float, default=1e6, help="TD loss blowup threshold.")
    parser.add_argument("--w-norm-blowup", type=float, default=1e4, help="W norm blowup threshold.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K entries per bucket.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan.")
    parser.add_argument("--summarize-only", action="store_true", help="Summarize an existing sweep root.")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume runs when output dir exists (default).",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume.")
    return parser.parse_args()


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    if not parts:
        return None
    values: List[float] = []
    for part in parts:
        values.append(float(part))
    return values


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    if yaml is not None:
        try:
            payload = yaml.safe_load(text)
            return payload or {}
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


def scaled_grid(
    base: float,
    factors: Sequence[float],
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> List[float]:
    values: List[float] = []
    for factor in factors:
        value = base * factor
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        values.append(round(float(value), 12))
    unique = sorted({v for v in values if v is not None})
    return unique


def pick_grid(values: List[float], grid: int) -> List[float]:
    if grid >= len(values):
        return values
    if grid <= 1:
        return [values[len(values) // 2]]
    indices = []
    span = len(values) - 1
    for i in range(grid):
        idx = int(round(i * span / (grid - 1)))
        indices.append(idx)
    chosen: List[float] = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            chosen.append(values[idx])
            seen.add(idx)
    if len(chosen) < grid:
        for idx, value in enumerate(values):
            if idx in seen:
                continue
            chosen.append(value)
            seen.add(idx)
            if len(chosen) >= grid:
                break
    return chosen


def linspace_values(min_val: float, max_val: float, num: int) -> List[float]:
    if num <= 1:
        return [round(float(min_val), 12)]
    step = (max_val - min_val) / (num - 1)
    values = [round(float(min_val + step * idx), 12) for idx in range(num)]
    if len(set(values)) != len(values):
        raise SystemExit("linspace produced duplicate values; adjust min/max or provide explicit values.")
    return values


def logspace_values(min_val: float, max_val: float, num: int) -> List[float]:
    if min_val <= 0 or max_val <= 0:
        raise SystemExit("logspace requires positive min/max values.")
    if num <= 1:
        return [round(float(min_val), 12)]
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    step = (log_max - log_min) / (num - 1)
    values = [round(float(10 ** (log_min + step * idx)), 12) for idx in range(num)]
    if len(set(values)) != len(values):
        raise SystemExit("logspace produced duplicate values; adjust min/max or provide explicit values.")
    return values


def format_tag(value: Optional[float]) -> str:
    if value is None:
        return "none"
    if value == 0:
        return "0"
    text = f"{value:.4g}"
    text = text.replace("-", "m").replace("+", "").replace(".", "p")
    return text


def run_id_for(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
    return (
        f"b{format_tag(params.get('beta'))}_aw{format_tag(params.get('alpha_w'))}_"
        f"tmos{format_tag(params.get('theta_mu_offset_scale'))}_"
        f"s{params.get('seed', 'na')}_{digest}"
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        info["commit"] = result.stdout.strip() or None
    except Exception:
        info["commit"] = None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        info["dirty"] = bool(result.stdout.strip())
    except Exception:
        info["dirty"] = None
    return info


def python_info() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.replace("\n", " ").strip(),
    }


def pip_freeze() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def git_text(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(args),
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def write_meta_artifacts(meta_dir: Path) -> None:
    ensure_dir(meta_dir)
    git_head = git_text(["git", "rev-parse", "HEAD"])
    (meta_dir / "git_head.txt").write_text((git_head or "") + "\n")
    git_status = git_text(["git", "status", "--porcelain"])
    (meta_dir / "git_status.txt").write_text((git_status or "") + "\n")
    git_diff = git_text(["git", "diff"])
    (meta_dir / "git_diff.patch").write_text((git_diff or "") + "\n")
    python_version = sys.version.replace("\n", " ").strip()
    (meta_dir / "python_version.txt").write_text(python_version + "\n")
    uname_text = " ".join(platform.uname())
    (meta_dir / "uname.txt").write_text(uname_text + "\n")
    (meta_dir / "requirements_freeze.txt").write_text(pip_freeze() + "\n")


def append_jsonl(path: Path, payload: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    with lock:
        with path.open("a") as handle:
            handle.write(line + "\n")


def run_complete(run_dir: Path) -> bool:
    report_path = run_dir / "run_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return False
    if report.get("incomplete"):
        return False
    return True


def build_command(
    spec: RunSpec,
    args: argparse.Namespace,
    base_config: Path,
    resume: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/run_train.py",
        "--config",
        str(base_config),
        "--output-dir",
        str(spec.output_dir),
        "--beta",
        str(spec.beta),
        "--alpha-w",
        str(spec.alpha_w),
        "--theta-mu-offset-scale",
        str(spec.theta_mu_offset_scale),
        "--outer-iters",
        str(args.outer_iters),
    ]
    if args.report_every is not None:
        cmd.extend(["--report-every", str(args.report_every)])
    if args.report_every_seconds is not None and args.report_every_seconds > 0:
        cmd.extend(["--report-every-seconds", str(args.report_every_seconds)])
    if spec.seed is not None:
        cmd.extend(["--seed", str(spec.seed)])
    if resume:
        cmd.append("--resume")
    return cmd


def run_one(
    spec: RunSpec,
    args: argparse.Namespace,
    base_config: Path,
    commands_path: Path,
    lock: threading.Lock,
) -> Dict[str, Any]:
    start_time = time.time()
    run_dir = spec.output_dir
    status: Dict[str, Any] = {
        "run_id": spec.run_id,
        "run_dir": str(run_dir),
        "status": "pending",
    }
    ensure_dir(run_dir)
    params = {
        "run_id": spec.run_id,
        "output_dir": str(run_dir),
        "base_config": str(base_config),
        "beta": spec.beta,
        "alpha_w": spec.alpha_w,
        "theta_mu_offset_scale": spec.theta_mu_offset_scale,
        "seed": spec.seed,
    }
    write_json(run_dir / "params.json", params)

    if run_complete(run_dir):
        status["status"] = "skipped"
        status["duration_sec"] = 0.0
        return status

    resume = args.resume and run_dir.exists()
    cmd = build_command(spec, args, base_config, resume=resume)
    log_path = run_dir / "stdout.log"
    with log_path.open("a") as handle:
        handle.write(f"Command: {shlex.join(cmd)}\n")
        handle.flush()
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.time() - start_time
    status["status"] = "ok" if result.returncode == 0 else "failed"
    status["returncode"] = result.returncode
    status["duration_sec"] = duration
    append_jsonl(
        commands_path,
        {
            "run_id": spec.run_id,
            "command": cmd,
            "returncode": result.returncode,
            "duration_sec": duration,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        lock,
    )
    return status


def parse_float(raw: Any) -> Optional[float]:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def read_csv_series(path: Path, columns: Sequence[str]) -> Tuple[Dict[str, List[float]], Dict[str, bool]]:
    series: Dict[str, List[float]] = {col: [] for col in columns}
    nonfinite: Dict[str, bool] = {col: False for col in columns}
    if not path.exists():
        return series, nonfinite
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for col in columns:
                raw = row.get(col)
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(val):
                    nonfinite[col] = True
                    continue
                series[col].append(val)
    return series, nonfinite


def last_value(values: Sequence[float]) -> Optional[float]:
    for val in reversed(values):
        if val is not None:
            return float(val)
    return None


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def linear_slope(values: Sequence[float], window: int) -> Optional[float]:
    if not values:
        return None
    k = min(len(values), max(1, int(window)))
    if k < 2:
        return None
    ys = list(values[-k:])
    xs = list(range(len(ys)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return num / denom


def range_last_k(values: Sequence[float], window: int) -> Optional[float]:
    if not values:
        return None
    k = min(len(values), max(1, int(window)))
    window_vals = values[-k:]
    return max(window_vals) - min(window_vals)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_report_metric(report: Optional[Dict[str, Any]], metric: str) -> Optional[float]:
    if not report:
        return None
    summary = report.get("summary_metrics", {}).get("learning_curves", {})
    entry = summary.get(metric)
    if isinstance(entry, dict):
        for key in ["mean_last_k", "last", "max"]:
            value = parse_float(entry.get(key))
            if value is not None:
                return value
    probes = report.get("summary_metrics", {}).get("probes", {})
    for probe_name in ["distribution_probe", "stability_probe", "fixed_point_probe"]:
        probe_summary = probes.get(probe_name, {})
        entry = probe_summary.get(metric)
        if isinstance(entry, dict):
            for key in ["mean_last_k", "last", "max"]:
                value = parse_float(entry.get(key))
                if value is not None:
                    return value
    return None


def compute_metrics(
    run_dir: Path,
    report: Optional[Dict[str, Any]],
    params: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    curves_path = run_dir / "learning_curves.csv"
    probes_dir = run_dir / "probes"

    curve_cols = [
        "td_loss",
        "w_norm",
        "tracking_gap",
        "mean_rho2",
        "fixed_point_drift",
        "stability_proxy",
        "dist_action_kl",
        "dist_action_tv",
    ]
    curves, curves_nonfinite = read_csv_series(curves_path, curve_cols)

    dist_probe, dist_nonfinite = read_csv_series(
        probes_dir / "distribution_probe.csv",
        ["dist_action_kl", "dist_action_tv"],
    )
    stability_probe, stability_nonfinite = read_csv_series(
        probes_dir / "stability_probe.csv",
        ["stability_proxy_mean", "stability_proxy"],
    )
    fixed_probe, fixed_nonfinite = read_csv_series(
        probes_dir / "fixed_point_probe.csv",
        ["w_sharp_drift"],
    )

    dist_action_kl = dist_probe.get("dist_action_kl") or curves.get("dist_action_kl") or []
    dist_action_tv = dist_probe.get("dist_action_tv") or curves.get("dist_action_tv") or []
    stability_series = stability_probe.get("stability_proxy_mean") or stability_probe.get("stability_proxy") or curves.get(
        "stability_proxy"
    ) or []
    fixed_drift_series = fixed_probe.get("w_sharp_drift") or curves.get("fixed_point_drift") or []

    td_loss_series = curves.get("td_loss") or []
    w_norm_series = curves.get("w_norm") or []
    tracking_gap_series = curves.get("tracking_gap") or []
    mean_rho2_series = curves.get("mean_rho2") or []

    td_loss_last = last_value(td_loss_series) or get_report_metric(report, "td_loss")
    w_norm_last = last_value(w_norm_series) or get_report_metric(report, "w_norm")
    tracking_gap_last = last_value(tracking_gap_series) or get_report_metric(report, "tracking_gap")
    mean_rho2_last = last_value(mean_rho2_series) or get_report_metric(report, "mean_rho2")

    td_loss_max = max(td_loss_series) if td_loss_series else None
    w_norm_max = max(w_norm_series) if w_norm_series else None
    stability_max = max(stability_series) if stability_series else None

    td_loss_slope = linear_slope(td_loss_series, args.window)
    w_norm_slope = linear_slope(w_norm_series, args.window)
    tracking_gap_slope = linear_slope(tracking_gap_series, args.window)
    mean_rho2_slope = linear_slope(mean_rho2_series, args.window)
    fixed_drift_slope = linear_slope(fixed_drift_series, args.window)

    td_loss_range = range_last_k(td_loss_series, args.window)

    tracking_gap_p95 = percentile(tracking_gap_series, 95) or get_report_metric(report, "tracking_gap")
    mean_rho2_p95 = percentile(mean_rho2_series, 95) or get_report_metric(report, "mean_rho2")
    dist_action_kl_p95 = percentile(dist_action_kl, 95) or get_report_metric(report, "dist_action_kl")
    dist_action_tv_p95 = percentile(dist_action_tv, 95) or get_report_metric(report, "dist_action_tv")

    offpolicy_score_candidates = [
        tracking_gap_p95 or 0.0,
        max(0.0, (mean_rho2_p95 or 1.0) - 1.0),
        dist_action_kl_p95 or 0.0,
        dist_action_tv_p95 or 0.0,
    ]
    offpolicy_score = max(offpolicy_score_candidates) if offpolicy_score_candidates else None

    drift_candidates = []
    for series, slope in [
        (w_norm_series, w_norm_slope),
        (tracking_gap_series, tracking_gap_slope),
        (mean_rho2_series, mean_rho2_slope),
        (fixed_drift_series, fixed_drift_slope),
    ]:
        if not series:
            continue
        slope_val = abs(slope) if slope is not None else 0.0
        range_val = range_last_k(series, args.window) or 0.0
        drift_candidates.append(max(slope_val, range_val))
    drift_score = max(drift_candidates) if drift_candidates else None

    health_status = None
    incomplete = None
    health_checks = {}
    exception = None
    if report:
        health_status = report.get("health_summary", {}).get("status")
        health_checks = report.get("health_checks", {}) or {}
        incomplete = report.get("incomplete")
        exception = report.get("exception")

    nonfinite_any = any(
        list(curves_nonfinite.values())
        + list(dist_nonfinite.values())
        + list(stability_nonfinite.values())
        + list(fixed_nonfinite.values())
    )

    instability_flag = False
    if incomplete:
        instability_flag = True
    if exception:
        instability_flag = True
    if health_status == "FAIL":
        instability_flag = True
    for entry in health_checks.values():
        if entry.get("applicable", True) and not entry.get("pass", False):
            instability_flag = True
            break
    if nonfinite_any:
        instability_flag = True
    if td_loss_max is not None and td_loss_max > args.td_loss_blowup:
        instability_flag = True
    if w_norm_max is not None and w_norm_max > args.w_norm_blowup:
        instability_flag = True
    stability_last = last_value(stability_series)
    if stability_last is not None and stability_last > 1.0 + args.stability_eps:
        instability_flag = True

    stable_flag = bool(health_status == "PASS") and not instability_flag
    plateau_flag = False
    if stable_flag and td_loss_slope is not None and drift_score is not None:
        plateau_flag = abs(td_loss_slope) < args.eps_slope and drift_score > args.eps_drift

    offpolicy_flag = bool(offpolicy_score is not None and offpolicy_score >= args.offpolicy_threshold)

    instability_score = 0.0
    if nonfinite_any:
        instability_score = float("inf")
    else:
        for val in [td_loss_max, w_norm_max, stability_max]:
            if val is not None:
                instability_score = max(instability_score, float(val))

    return {
        "run_id": params.get("run_id"),
        "run_dir": str(run_dir),
        "beta": params.get("beta"),
        "alpha_w": params.get("alpha_w"),
        "theta_mu_offset_scale": params.get("theta_mu_offset_scale"),
        "seed": params.get("seed"),
        "health_status": health_status,
        "stable_flag": stable_flag,
        "incomplete": incomplete,
        "instability_flag": instability_flag,
        "plateau_flag": plateau_flag,
        "offpolicy_flag": offpolicy_flag,
        "offpolicy_score": offpolicy_score,
        "instability_score": instability_score,
        "drift_score": drift_score,
        "td_loss_last": td_loss_last,
        "td_loss_slope": td_loss_slope,
        "td_loss_range": td_loss_range,
        "w_norm_last": w_norm_last,
        "w_norm_slope": w_norm_slope,
        "tracking_gap_last": tracking_gap_last,
        "tracking_gap_p95": tracking_gap_p95,
        "mean_rho2_last": mean_rho2_last,
        "mean_rho2_p95": mean_rho2_p95,
        "dist_action_kl_p95": dist_action_kl_p95,
        "dist_action_tv_p95": dist_action_tv_p95,
        "stability_proxy_last": stability_last,
    }


def load_params(run_dir: Path) -> Dict[str, Any]:
    params = load_json(run_dir / "params.json")
    if params:
        return params
    config = load_json(run_dir / "config.json") or {}
    env_cfg = config.get("env", {}) if isinstance(config.get("env"), dict) else {}
    return {
        "run_id": run_dir.name,
        "beta": config.get("beta"),
        "alpha_w": config.get("alpha_w"),
        "theta_mu_offset_scale": config.get("theta_mu_offset_scale"),
        "seed": config.get("seed"),
        "p_mix": env_cfg.get("p_mix"),
    }


def summarize_sweep(
    run_root: Path,
    args: argparse.Namespace,
    grid_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    runs_dir = run_root / "runs"
    summary_dir = run_root / "summary"
    buckets_dir = summary_dir / "buckets"
    ensure_dir(summary_dir)
    ensure_dir(buckets_dir)

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        params = load_params(run_dir)
        report = load_json(run_dir / "run_report.json")
        metrics = compute_metrics(run_dir, report, params, args)
        rows.append(metrics)

    summary_columns = [
        "run_id",
        "beta",
        "alpha_w",
        "theta_mu_offset_scale",
        "seed",
        "health_status",
        "stable_flag",
        "incomplete",
        "instability_flag",
        "plateau_flag",
        "offpolicy_flag",
        "offpolicy_score",
        "instability_score",
        "drift_score",
        "td_loss_last",
        "td_loss_slope",
        "td_loss_range",
        "w_norm_last",
        "w_norm_slope",
        "tracking_gap_last",
        "tracking_gap_p95",
        "mean_rho2_last",
        "mean_rho2_p95",
        "dist_action_kl_p95",
        "dist_action_tv_p95",
        "stability_proxy_last",
        "run_dir",
    ]

    with (summary_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in summary_columns})

    with (summary_dir / "summary_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    top_k = max(args.top_k, 10)
    offpolicy_stable = [
        r for r in rows if r.get("offpolicy_flag") and r.get("stable_flag") and not r.get("instability_flag")
    ]
    instability = [r for r in rows if r.get("instability_flag")]
    plateau = [r for r in rows if r.get("plateau_flag")]

    offpolicy_sorted = sorted(offpolicy_stable, key=lambda r: r.get("offpolicy_score") or 0.0, reverse=True)
    instability_sorted = sorted(instability, key=lambda r: r.get("instability_score") or 0.0, reverse=True)
    plateau_sorted = sorted(plateau, key=lambda r: r.get("drift_score") or 0.0, reverse=True)

    def write_bucket(path: Path, entries: List[Dict[str, Any]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_columns)
            writer.writeheader()
            for row in entries:
                writer.writerow({key: row.get(key) for key in summary_columns})

    write_bucket(buckets_dir / "offpolicy_stable_top.csv", offpolicy_sorted[:top_k])
    write_bucket(buckets_dir / "instability_top.csv", instability_sorted[:top_k])
    write_bucket(buckets_dir / "plateau_drift_top.csv", plateau_sorted[:top_k])

    write_json(
        summary_dir / "summary.json",
        {
            "run_root": str(run_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "grid": grid_info,
            "thresholds": {
                "offpolicy_threshold": args.offpolicy_threshold,
                "eps_slope": args.eps_slope,
                "eps_drift": args.eps_drift,
                "window": args.window,
                "stability_eps": args.stability_eps,
                "td_loss_blowup": args.td_loss_blowup,
                "w_norm_blowup": args.w_norm_blowup,
            },
            "counts": {
                "total_runs": len(rows),
                "offpolicy_stable": len(offpolicy_stable),
                "instability": len(instability),
                "plateau_drift": len(plateau),
            },
            "rows": rows,
        },
    )

    boundary_rows = []
    group_map: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("seed"), row.get("beta"), row.get("theta_mu_offset_scale"))
        group_map.setdefault(key, []).append(row)
    for items in group_map.values():
        items_sorted = sorted(items, key=lambda r: r.get("alpha_w") or 0.0)
        for idx in range(1, len(items_sorted)):
            prev = items_sorted[idx - 1]
            curr = items_sorted[idx]
            if prev.get("stable_flag") and not prev.get("instability_flag") and curr.get("instability_flag"):
                boundary_rows.append(
                    {
                        "run_id": prev.get("run_id"),
                        "beta": prev.get("beta"),
                        "alpha_w": prev.get("alpha_w"),
                        "theta_mu_offset_scale": prev.get("theta_mu_offset_scale"),
                        "offpolicy_score": prev.get("offpolicy_score"),
                        "td_loss_last": prev.get("td_loss_last"),
                        "w_norm_last": prev.get("w_norm_last"),
                        "next_alpha_w": curr.get("alpha_w"),
                        "next_run_id": curr.get("run_id"),
                    }
                )
                break

    summary_md = summary_dir / "summary.md"
    lines: List[str] = ["# Sweep Summary", ""]
    lines.append("## Sweep Info")
    lines.append(f"- run_root: {run_root}")
    lines.append(f"- total_runs: {len(rows)}")
    lines.append(f"- offpolicy_stable: {len(offpolicy_stable)}")
    lines.append(f"- instability: {len(instability)}")
    lines.append(f"- plateau_drift: {len(plateau)}")
    lines.append("")
    lines.append("## Grid (effective)")
    lines.append(f"- beta_values: {grid_info.get('beta_values')}")
    lines.append(f"- alpha_w_values: {grid_info.get('alpha_w_values')}")
    lines.append(f"- theta_mu_offset_scale_values: {grid_info.get('theta_mu_offset_scale_values')}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- offpolicy_threshold: {args.offpolicy_threshold}")
    lines.append(f"- eps_slope: {args.eps_slope}")
    lines.append(f"- eps_drift: {args.eps_drift}")
    lines.append(f"- window: {args.window}")
    lines.append(f"- stability_eps: {args.stability_eps}")
    lines.append(f"- td_loss_blowup: {args.td_loss_blowup}")
    lines.append(f"- w_norm_blowup: {args.w_norm_blowup}")
    lines.append("")
    lines.append("## Classification")
    lines.append(
        "- offpolicy_score = max(tracking_gap_p95, max(0, mean_rho2_p95 - 1), dist_action_kl_p95, dist_action_tv_p95)"
    )
    lines.append("- offpolicy_flag = offpolicy_score >= offpolicy_threshold")
    lines.append("- offpolicy_stable bucket = offpolicy_flag AND health_status PASS AND not instability_flag")
    lines.append(
        "- instability_flag = health_checks fail OR incomplete OR non-finite OR stability_proxy > 1+stability_eps "
        "OR td_loss_max > td_loss_blowup OR w_norm_max > w_norm_blowup"
    )
    lines.append("- plateau_flag = health_status PASS AND not instability AND |td_loss_slope| < eps_slope AND drift_score > eps_drift")
    lines.append("")

    def format_table(entries: List[Dict[str, Any]], top_k: int) -> List[str]:
        cols = [
            "run_id",
            "beta",
            "alpha_w",
            "theta_mu_offset_scale",
            "offpolicy_score",
            "td_loss_last",
            "w_norm_last",
            "tracking_gap_last",
            "mean_rho2_last",
            "health_status",
        ]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in entries[:top_k]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("run_id")),
                        str(row.get("beta")),
                        str(row.get("alpha_w")),
                        str(row.get("theta_mu_offset_scale")),
                        str(row.get("offpolicy_score")),
                        str(row.get("td_loss_last")),
                        str(row.get("w_norm_last")),
                        str(row.get("tracking_gap_last")),
                        str(row.get("mean_rho2_last")),
                        str(row.get("health_status")),
                    ]
                )
                + " |"
            )
        return lines

    lines.append(f"## Off-policy stable (top {min(top_k, len(offpolicy_sorted))})")
    lines.extend(format_table(offpolicy_sorted, min(top_k, len(offpolicy_sorted))))
    lines.append("")
    lines.append(f"## Instability (top {min(top_k, len(instability_sorted))})")
    lines.extend(format_table(instability_sorted, min(top_k, len(instability_sorted))))
    lines.append("")
    lines.append(f"## Plateau drift (top {min(top_k, len(plateau_sorted))})")
    lines.extend(format_table(plateau_sorted, min(top_k, len(plateau_sorted))))
    lines.append("")

    if boundary_rows:
        lines.append("## Stability boundary (closest stable points)")
        cols = [
            "run_id",
            "beta",
            "alpha_w",
            "theta_mu_offset_scale",
            "offpolicy_score",
            "td_loss_last",
            "w_norm_last",
            "next_alpha_w",
            "next_run_id",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in boundary_rows[:top_k]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("run_id")),
                        str(row.get("beta")),
                        str(row.get("alpha_w")),
                        str(row.get("theta_mu_offset_scale")),
                        str(row.get("offpolicy_score")),
                        str(row.get("td_loss_last")),
                        str(row.get("w_norm_last")),
                        str(row.get("next_alpha_w")),
                        str(row.get("next_run_id")),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Outputs")
    lines.append(f"- summary_csv: {summary_dir / 'summary.csv'}")
    lines.append(f"- summary_json: {summary_dir / 'summary.json'}")
    lines.append(f"- summary_rows: {summary_dir / 'summary_rows.jsonl'}")
    lines.append(f"- artifacts_index: {summary_dir / 'artifacts_index.md'}")
    lines.append(f"- buckets_dir: {buckets_dir}")

    summary_md.write_text("\n".join(lines))
    write_artifacts_index(run_root, summary_dir, buckets_dir)
    return rows


def write_artifacts_index(run_root: Path, summary_dir: Path, buckets_dir: Path) -> None:
    items: List[str] = []

    def add(path: Path) -> None:
        if path.exists():
            items.append(str(path))

    meta_dir = run_root / "meta"
    add(meta_dir / "git_head.txt")
    add(meta_dir / "git_status.txt")
    add(meta_dir / "git_diff.patch")
    add(meta_dir / "python_version.txt")
    add(meta_dir / "requirements_freeze.txt")
    add(meta_dir / "uname.txt")
    add(meta_dir / "commandline.txt")
    add(meta_dir / "meta.json")
    add(meta_dir / "grid_effective.yaml")
    add(meta_dir / "run_plan.jsonl")
    add(meta_dir / "run_status.json")
    add(meta_dir / "commands_executed.jsonl")

    add(summary_dir / "summary.csv")
    add(summary_dir / "summary.json")
    add(summary_dir / "summary.md")
    add(summary_dir / "summary_rows.jsonl")
    if buckets_dir.exists():
        for bucket in sorted(buckets_dir.glob("*.csv")):
            add(bucket)

    runs_dir = run_root / "runs"
    if runs_dir.exists():
        items.append(str(runs_dir))

    artifacts_path = summary_dir / "artifacts_index.md"
    lines = ["# Artifacts Index", ""]
    lines.extend(f"- {item}" for item in items)
    lines.append(f"- {artifacts_path}")
    artifacts_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.grid < 1:
        raise SystemExit("--grid must be >= 1.")

    base_config = Path(args.base_config)
    base_cfg = load_config(base_config)

    base_beta = parse_float(base_cfg.get("beta")) or 0.05
    base_alpha_w = parse_float(base_cfg.get("alpha_w")) or 0.08
    base_theta_mu_offset_scale = parse_float(base_cfg.get("theta_mu_offset_scale")) or 0.0

    beta_values = parse_float_list(args.beta_values)
    if beta_values is None:
        beta_min = max(1e-5, base_beta * min(DEFAULT_BETA_FACTORS))
        beta_max = min(1.0, base_beta * max(DEFAULT_BETA_FACTORS))
        beta_values = linspace_values(beta_min, beta_max, args.grid)

    alpha_values = parse_float_list(args.alpha_w_values)
    if alpha_values is None:
        alpha_min = max(1e-6, base_alpha_w * min(DEFAULT_ALPHA_W_FACTORS))
        alpha_max = max(alpha_min, base_alpha_w * max(DEFAULT_ALPHA_W_FACTORS))
        alpha_values = logspace_values(alpha_min, alpha_max, args.grid)

    theta_mu_offset_values = parse_float_list(args.theta_mu_offset_values)
    if theta_mu_offset_values is None:
        theta_min = DEFAULT_THETA_MU_OFFSET_MIN
        theta_max = max(theta_min, DEFAULT_THETA_MU_OFFSET_MAX, base_theta_mu_offset_scale)
        theta_mu_offset_values = linspace_values(theta_min, theta_max, args.grid)

    seeds = parse_float_list(args.seeds)
    seed_values: List[Optional[int]] = []
    if seeds:
        seed_values = [int(val) for val in seeds]
    else:
        seed_values = [int(base_cfg.get("seed", 0) or 0)]

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = Path(args.out_root) / timestamp / "sweep"

    grid_info = {
        "beta_values": beta_values,
        "alpha_w_values": alpha_values,
        "theta_mu_offset_scale_values": theta_mu_offset_values,
        "seed_values": seed_values,
        "base_config": str(base_config),
    }

    if args.dry_run:
        print("Dry run sweep plan:")
        print(f"  run_root: {run_root}")
        print(
            f"  total_grid: {len(beta_values) * len(alpha_values) * len(theta_mu_offset_values) * len(seed_values)}"
        )
        print(f"  beta_values: {beta_values}")
        print(f"  alpha_w_values: {alpha_values}")
        print(f"  theta_mu_offset_scale_values: {theta_mu_offset_values}")
        print(f"  seed_values: {seed_values}")
        return

    if args.summarize_only:
        meta_grid_path = run_root / "meta" / "grid_effective.yaml"
        if meta_grid_path.exists():
            existing_grid = load_config(meta_grid_path)
            if existing_grid:
                grid_info = existing_grid
        summarize_sweep(run_root, args, grid_info)
        return

    meta_dir = run_root / "meta"
    runs_dir = run_root / "runs"
    ensure_dir(meta_dir)
    ensure_dir(runs_dir)

    write_yaml(meta_dir / "grid_effective.yaml", grid_info)
    write_json(
        meta_dir / "meta.json",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "command": shlex.join(sys.argv),
            "git": git_info(),
            "python": python_info(),
        },
    )
    write_meta_artifacts(meta_dir)

    (meta_dir / "commandline.txt").write_text(shlex.join(sys.argv))

    run_specs: List[RunSpec] = []
    for beta in beta_values:
        for alpha_w in alpha_values:
            for theta_mu_offset_scale in theta_mu_offset_values:
                for seed in seed_values:
                    params = {
                        "beta": beta,
                        "alpha_w": alpha_w,
                        "theta_mu_offset_scale": theta_mu_offset_scale,
                        "seed": seed,
                    }
                    run_id = run_id_for(params)
                    run_dir = runs_dir / run_id
                    run_specs.append(
                        RunSpec(
                            run_id=run_id,
                            beta=beta,
                            alpha_w=alpha_w,
                            theta_mu_offset_scale=theta_mu_offset_scale,
                            seed=seed,
                            output_dir=run_dir,
                        )
                    )

    plan_path = meta_dir / "run_plan.jsonl"
    with plan_path.open("w") as handle:
        for spec in run_specs:
            payload = {
                "run_id": spec.run_id,
                "beta": spec.beta,
                "alpha_w": spec.alpha_w,
                "theta_mu_offset_scale": spec.theta_mu_offset_scale,
                "seed": spec.seed,
                "output_dir": str(spec.output_dir),
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    commands_path = meta_dir / "commands_executed.jsonl"
    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    if args.jobs <= 1:
        for spec in run_specs:
            result = run_one(spec, args, base_config, commands_path, lock)
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(run_one, spec, args, base_config, commands_path, lock) for spec in run_specs
            ]
            for future in as_completed(futures):
                results.append(future.result())

    write_json(meta_dir / "run_status.json", {"results": results})
    summarize_sweep(run_root, args, grid_info)


if __name__ == "__main__":
    main()
