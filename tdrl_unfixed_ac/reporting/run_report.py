"""Generate run_report.json and run_report.md from a training run directory."""

from __future__ import annotations

import csv
import json
import math
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

CORE_COLUMNS = [
    "td_loss",
    "w_norm",
    "mean_rho2",
    "tracking_gap",
    "critic_teacher_error",
]
PROBE_COLUMNS = [
    "fixed_point_gap",
    "fixed_point_drift",
    "fixed_point_drift_defined",
    "stability_proxy",
    "dist_mmd2",
    "dist_mean_l2",
    "dist_action_kl",
    "dist_action_tv",
    "td_loss_from_Q",
    "td_loss_from_Q_abs_diff",
    "td_loss_from_Q_rel_diff",
]
TIME_COLUMNS = {"iter", "step"}
DEFAULT_ON_POLICY_DIST_MMD2_THRESHOLD = 0.0783317
DEFAULT_ON_POLICY_ACTION_KL_THRESHOLD = 1e-6
DEFAULT_ON_POLICY_ACTION_TV_THRESHOLD = 1e-4
ON_POLICY_MEAN_RHO2_BOUNDS = (0.98, 1.02)
ON_POLICY_TRACKING_GAP_THRESHOLD = 1e-6


def _coerce_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _coerce_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_config(v) for v in value]
    return value


def _load_config(path: Optional[Path], fallback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if path and path.exists():
        try:
            text = path.read_text()
            if yaml is not None:
                payload = yaml.safe_load(text) or {}
            else:
                payload = json.loads(text)
        except Exception:
            payload = {}
    if fallback:
        fallback = _coerce_config(fallback)
    return payload or (fallback or {})


def _read_on_policy_dist_mmd2_threshold(cfg: Dict[str, Any]) -> float:
    reporting_cfg = cfg.get("reporting", {})
    if not isinstance(reporting_cfg, dict):
        return DEFAULT_ON_POLICY_DIST_MMD2_THRESHOLD
    raw_val = reporting_cfg.get("on_policy_dist_mmd2_threshold")
    if raw_val is None:
        return DEFAULT_ON_POLICY_DIST_MMD2_THRESHOLD
    try:
        return float(raw_val)
    except (TypeError, ValueError):
        return DEFAULT_ON_POLICY_DIST_MMD2_THRESHOLD


def _read_on_policy_action_kl_threshold(cfg: Dict[str, Any]) -> float:
    reporting_cfg = cfg.get("reporting", {})
    if not isinstance(reporting_cfg, dict):
        return DEFAULT_ON_POLICY_ACTION_KL_THRESHOLD
    raw_val = reporting_cfg.get("on_policy_action_kl_threshold")
    if raw_val is None:
        return DEFAULT_ON_POLICY_ACTION_KL_THRESHOLD
    try:
        return float(raw_val)
    except (TypeError, ValueError):
        return DEFAULT_ON_POLICY_ACTION_KL_THRESHOLD


def _read_on_policy_action_tv_threshold(cfg: Dict[str, Any]) -> float:
    reporting_cfg = cfg.get("reporting", {})
    if not isinstance(reporting_cfg, dict):
        return DEFAULT_ON_POLICY_ACTION_TV_THRESHOLD
    raw_val = reporting_cfg.get("on_policy_action_tv_threshold")
    if raw_val is None:
        return DEFAULT_ON_POLICY_ACTION_TV_THRESHOLD
    try:
        return float(raw_val)
    except (TypeError, ValueError):
        return DEFAULT_ON_POLICY_ACTION_TV_THRESHOLD


def _parse_csv(path: Path) -> Tuple[List[Dict[str, Any]], List[str], set[str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        raw_rows = [row for row in reader]

    if not raw_rows:
        return [], fieldnames, set()

    numeric_cols = set(fieldnames)
    for col in list(numeric_cols):
        for row in raw_rows:
            raw_val = row.get(col)
            if raw_val is None or str(raw_val).strip() == "":
                continue
            try:
                float(raw_val)
            except ValueError:
                numeric_cols.discard(col)
                break

    parsed_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        parsed: Dict[str, Any] = {}
        for col in fieldnames:
            raw_val = row.get(col)
            if col in numeric_cols:
                if raw_val is None or str(raw_val).strip() == "":
                    parsed[col] = None
                else:
                    try:
                        parsed[col] = float(raw_val)
                    except ValueError:
                        parsed[col] = None
            else:
                parsed[col] = raw_val
        parsed_rows.append(parsed)

    return parsed_rows, fieldnames, numeric_cols


def _iter_finite(values: Iterable[Any]) -> List[float]:
    finite: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            finite.append(float(value))
    return finite


def _split_columns(numeric_cols: set[str]) -> Tuple[List[str], List[str]]:
    probe_set = set(PROBE_COLUMNS)
    core_cols = sorted(col for col in numeric_cols if col not in probe_set and col not in TIME_COLUMNS)
    probe_cols = sorted(col for col in numeric_cols if col in probe_set)
    return core_cols, probe_cols


def _scan_nan_inf(
    rows: List[Dict[str, Any]],
    columns: Iterable[str],
) -> Tuple[int, int, List[str], List[str]]:
    nan_cols: List[str] = []
    inf_cols: List[str] = []
    nan_count = 0
    inf_count = 0
    for col in columns:
        for row in rows:
            value = row.get(col)
            if isinstance(value, (int, float)):
                val = float(value)
                if math.isnan(val):
                    nan_count += 1
                    if col not in nan_cols:
                        nan_cols.append(col)
                elif math.isinf(val):
                    inf_count += 1
                    if col not in inf_cols:
                        inf_cols.append(col)
    return nan_count, inf_count, nan_cols, inf_cols


def _last_finite(values: Iterable[Any]) -> Optional[float]:
    for value in reversed(list(values)):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _collect_last_points(
    rows: List[Dict[str, Any]],
    column: str,
    *,
    time_col: Optional[str],
    last_k: int,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for idx in range(len(rows) - 1, -1, -1):
        y_val = rows[idx].get(column)
        if not (isinstance(y_val, (int, float)) and math.isfinite(float(y_val))):
            continue
        if time_col:
            x_val = rows[idx].get(time_col)
            if not (isinstance(x_val, (int, float)) and math.isfinite(float(x_val))):
                x_val = idx
        else:
            x_val = idx
        xs.append(float(x_val))
        ys.append(float(y_val))
        if len(xs) >= last_k:
            break
    xs.reverse()
    ys.reverse()
    return xs, ys


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


def _summarize_columns(
    rows: List[Dict[str, Any]],
    numeric_cols: set[str],
    *,
    columns: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    last_k: int = 5,
) -> Dict[str, Dict[str, Optional[float]]]:
    if not rows:
        return {}
    target_cols = columns if columns is not None else sorted(numeric_cols)
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for col in target_cols:
        if col not in numeric_cols:
            continue
        values = [row.get(col) for row in rows]
        finite = _iter_finite(values)
        last_val = _last_finite(values)
        min_val = min(finite) if finite else None
        max_val = max(finite) if finite else None

        last_vals: List[float] = []
        for value in reversed(values):
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                last_vals.append(float(value))
            if len(last_vals) >= last_k:
                break
        mean_last_k = sum(last_vals) / len(last_vals) if last_vals else None

        xs, ys = _collect_last_points(rows, col, time_col=time_col, last_k=last_k)
        slope = _linear_slope(xs, ys) if xs and ys else None

        summary[col] = {
            "last": last_val,
            "min": min_val,
            "max": max_val,
            "mean_last_k": mean_last_k,
            "slope_last_k": slope,
        }
    return summary


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _sanitize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for row in rows:
        sanitized.append({k: _sanitize_value(v) for k, v in row.items()})
    return sanitized


def _extract_key_hparams(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_cfg = cfg.get("env", {}) if isinstance(cfg.get("env"), dict) else {}
    return {
        "seed": cfg.get("seed"),
        "outer_iters": cfg.get("outer_iters"),
        "horizon": cfg.get("horizon") or cfg.get("steps"),
        "gamma": cfg.get("gamma"),
        "alpha_w": cfg.get("alpha_w"),
        "alpha_pi": cfg.get("alpha_pi"),
        "beta": cfg.get("beta"),
        "sigma_pi": cfg.get("sigma_pi"),
        "sigma_mu": cfg.get("sigma_mu"),
        "feature_dim": cfg.get("feature_dim") or env_cfg.get("feature_dim"),
        "p_mix": env_cfg.get("p_mix") if env_cfg else cfg.get("p_mix"),
        "check_name": cfg.get("check_name"),
    }


def _train_step_scale(cfg: Dict[str, Any]) -> Optional[float]:
    try:
        alpha_w = float(cfg.get("alpha_w"))
        trajectories = int(cfg.get("trajectories", 0) or 0)
        horizon = int(cfg.get("horizon", 0) or 0)
    except (TypeError, ValueError):
        return None
    total_steps = max(trajectories * horizon, 0)
    if alpha_w <= 0.0 or total_steps <= 0:
        return None
    return alpha_w / total_steps


def _get_git_commit(run_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(run_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _is_on_policy_run(cfg: Dict[str, Any]) -> bool:
    if cfg.get("check_name") == "on_policy":
        return True
    sanity_cfg = cfg.get("sanity", {}) if isinstance(cfg.get("sanity"), dict) else {}
    if sanity_cfg.get("label") == "on_policy":
        return True
    policy_cfg = cfg.get("policy", {}) if isinstance(cfg.get("policy"), dict) else {}
    if bool(policy_cfg.get("on_policy")):
        return True
    return False


def _health_checks(
    *,
    rows: List[Dict[str, Any]],
    numeric_cols: set[str],
    cfg: Dict[str, Any],
    rho_threshold: float,
    dist_mmd2_value: Optional[float],
    dist_mmd2_meta: Optional[Dict[str, Optional[float]]],
    dist_action_kl_value: Optional[float],
    dist_action_tv_value: Optional[float],
    dist_action_meta: Optional[Dict[str, Optional[float]]],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    core_cols, probe_cols = _split_columns(numeric_cols)

    # no_nan_inf
    if not rows:
        checks["no_nan_inf"] = {
            "pass": False,
            "reason": "learning_curves.csv missing or empty",
            "observed": {"nan_count": None, "inf_count": None, "core_cols": core_cols},
            "applicable": True,
        }
    elif not core_cols:
        checks["no_nan_inf"] = {
            "pass": False,
            "reason": "no core columns found for NaN/Inf check",
            "observed": {"nan_count": None, "inf_count": None, "core_cols": core_cols},
            "applicable": True,
        }
    else:
        nan_count, inf_count, nan_cols, inf_cols = _scan_nan_inf(rows, core_cols)
        passed = nan_count == 0 and inf_count == 0
        reason = "all core columns finite" if passed else "found NaN/Inf in core columns"
        checks["no_nan_inf"] = {
            "pass": passed,
            "reason": reason,
            "observed": {
                "nan_count": nan_count,
                "inf_count": inf_count,
                "nan_cols": nan_cols,
                "inf_cols": inf_cols,
                "core_cols": core_cols,
            },
            "applicable": True,
        }

    # probe_missing_values
    probes_cfg = cfg.get("probes", {}) if isinstance(cfg.get("probes"), dict) else {}
    probes_enabled = bool(probes_cfg.get("enabled", False))
    try:
        probes_every = int(probes_cfg.get("every", 0) or 0)
    except Exception:
        probes_every = 0
    strict_probe = probes_enabled and probes_every == 1

    if not rows or not probe_cols:
        reason = "learning_curves.csv missing or empty" if not rows else "no probe columns found"
        checks["probe_missing_values"] = {
            "pass": True,
            "reason": reason,
            "observed": {
                "nan_count": None if not rows else 0,
                "inf_count": None if not rows else 0,
                "missing_count": None if not rows else 0,
                "nan_cols": [],
                "inf_cols": [],
                "probe_cols": probe_cols,
                "probes_enabled": probes_enabled,
                "probes_every": probes_every,
                "strict_mode": strict_probe,
            },
            "applicable": True,
        }
    else:
        coverage_map = {"fixed_point_drift": "fixed_point_drift_defined"}
        nan_cols: List[str] = []
        inf_cols: List[str] = []
        missing_cols: List[str] = []
        coverage_rows: Dict[str, int] = {}
        nan_count = 0
        inf_count = 0
        missing_count = 0

        for col in probe_cols:
            coverage_col = coverage_map.get(col)
            for row in rows:
                coverage_ok = True
                if coverage_col is not None:
                    cov_val = row.get(coverage_col)
                    coverage_ok = isinstance(cov_val, (int, float)) and cov_val > 0.0
                    if coverage_ok:
                        coverage_rows[col] = coverage_rows.get(col, 0) + 1
                if not coverage_ok:
                    continue
                value = row.get(col)
                if value is None:
                    if coverage_col is not None:
                        missing_count += 1
                        if col not in missing_cols:
                            missing_cols.append(col)
                    continue
                if isinstance(value, (int, float)):
                    val = float(value)
                    if math.isnan(val):
                        if coverage_col is not None:
                            nan_count += 1
                            if col not in nan_cols:
                                nan_cols.append(col)
                    elif math.isinf(val):
                        inf_count += 1
                        if col not in inf_cols:
                            inf_cols.append(col)

        missing_count_total = nan_count + inf_count + missing_count
        passed = missing_count_total == 0
        if missing_count_total == 0:
            reason = "probe columns finite where coverage indicates"
        else:
            reason = "probe columns contain non-finite values on covered rows"
        checks["probe_missing_values"] = {
            "pass": passed,
            "reason": reason,
            "observed": {
                "nan_count": nan_count,
                "inf_count": inf_count,
                "missing_count": missing_count,
                "nan_cols": nan_cols,
                "inf_cols": inf_cols,
                "missing_cols": missing_cols,
                "coverage_rows": coverage_rows,
                "coverage_map": coverage_map,
                "probe_cols": probe_cols,
                "probes_enabled": probes_enabled,
                "probes_every": probes_every,
                "strict_mode": strict_probe,
            },
            "applicable": True,
        }

    # monotone_time
    time_col = None
    if "step" in numeric_cols:
        time_col = "step"
    elif "iter" in numeric_cols:
        time_col = "iter"

    if not rows or time_col is None:
        checks["monotone_time"] = {
            "pass": False,
            "reason": "time column missing or no data",
            "observed": {"time_col": time_col, "num_rows": len(rows)},
            "applicable": True,
        }
    else:
        values = [row.get(time_col) for row in rows]
        finite = _iter_finite(values)
        if len(finite) < 2:
            checks["monotone_time"] = {
                "pass": True,
                "reason": "not enough rows to check monotonicity",
                "observed": {"time_col": time_col, "num_rows": len(rows)},
                "applicable": True,
            }
        else:
            passed = True
            first_bad = None
            prev = finite[0]
            for idx, value in enumerate(finite[1:], start=1):
                if value <= prev:
                    passed = False
                    first_bad = idx
                    break
                prev = value
            reason = "time is strictly increasing" if passed else "time column not strictly increasing"
            checks["monotone_time"] = {
                "pass": passed,
                "reason": reason,
                "observed": {"time_col": time_col, "first_violation_index": first_bad},
                "applicable": True,
            }

    # rho_sane
    rho_col = None
    if "mean_rho2" in numeric_cols:
        rho_col = "mean_rho2"
    else:
        for col in sorted(numeric_cols):
            if "rho" in col:
                rho_col = col
                break
    if rho_col is None:
        checks["rho_sane"] = {
            "pass": True,
            "reason": "rho statistics not found",
            "observed": {"rho_column": None, "threshold": rho_threshold},
            "applicable": False,
        }
    else:
        values = _iter_finite([row.get(rho_col) for row in rows])
        max_val = max(values) if values else None
        passed = max_val is not None and max_val <= rho_threshold
        reason = "rho within threshold" if passed else "rho exceeds threshold"
        checks["rho_sane"] = {
            "pass": passed,
            "reason": reason,
            "observed": {"rho_column": rho_col, "max": max_val, "threshold": rho_threshold},
            "applicable": True,
        }

    # on_policy_expected
    check_name = cfg.get("check_name")
    applicable = _is_on_policy_run(cfg)
    mean_rho2 = None
    if "mean_rho2" in numeric_cols:
        mean_rho2 = _last_finite([row.get("mean_rho2") for row in rows])
    tracking_gap = None
    if "tracking_gap" in numeric_cols:
        tracking_gap = _last_finite([row.get("tracking_gap") for row in rows])
    rho_ok = mean_rho2 is not None and ON_POLICY_MEAN_RHO2_BOUNDS[0] <= mean_rho2 <= ON_POLICY_MEAN_RHO2_BOUNDS[1]
    tracking_ok = tracking_gap is None or tracking_gap <= ON_POLICY_TRACKING_GAP_THRESHOLD
    action_kl_threshold = _read_on_policy_action_kl_threshold(cfg)
    action_tv_threshold = _read_on_policy_action_tv_threshold(cfg)
    action_kl_ok = dist_action_kl_value is not None and dist_action_kl_value <= action_kl_threshold
    action_tv_ok = dist_action_tv_value is not None and dist_action_tv_value <= action_tv_threshold
    if dist_action_kl_value is None and dist_action_tv_value is None:
        action_ok = True
    else:
        action_ok = action_kl_ok or action_tv_ok
    mmd_threshold = _read_on_policy_dist_mmd2_threshold(cfg)
    dist_meta = dist_mmd2_meta or {}
    action_meta = dist_action_meta or {}
    if applicable:
        passed = rho_ok and tracking_ok and action_ok
        if dist_action_kl_value is None and dist_action_tv_value is None:
            reason = "mean_rho2 close to 1; action divergence unavailable"
        else:
            reason = "on-policy expectations met" if passed else "on-policy expectations not met"
        if dist_mmd2_value is not None:
            reason = f"{reason} (dist_mmd2 diagnostic only)"
        checks["on_policy_expected"] = {
            "pass": passed,
            "reason": reason,
            "observed": {
                "mean_rho2": mean_rho2,
                "tracking_gap": tracking_gap,
                "dist_action_kl": dist_action_kl_value,
                "dist_action_tv": dist_action_tv_value,
                "action_kl_threshold": action_kl_threshold,
                "action_tv_threshold": action_tv_threshold,
                "dist_mmd2": dist_mmd2_value,
                "dist_mmd2_threshold": mmd_threshold,
                "num_samples": dist_meta.get("num_samples"),
                "mmd_sigma": dist_meta.get("mmd_sigma"),
                "action_samples": action_meta.get("action_samples"),
            },
            "applicable": True,
        }
    else:
        checks["on_policy_expected"] = {
            "pass": True,
            "reason": "not applicable",
            "observed": {
                "mean_rho2": mean_rho2,
                "tracking_gap": tracking_gap,
                "dist_action_kl": dist_action_kl_value,
                "dist_action_tv": dist_action_tv_value,
                "action_kl_threshold": action_kl_threshold,
                "action_tv_threshold": action_tv_threshold,
                "dist_mmd2": dist_mmd2_value,
                "dist_mmd2_threshold": mmd_threshold,
                "num_samples": dist_meta.get("num_samples"),
                "mmd_sigma": dist_meta.get("mmd_sigma"),
                "action_samples": action_meta.get("action_samples"),
            },
            "applicable": False,
        }

    # no_bootstrap_expected
    gamma = cfg.get("gamma")
    gamma_val = None
    try:
        gamma_val = float(gamma) if gamma is not None else None
    except Exception:
        gamma_val = None
    applicable = check_name == "no_bootstrap" or (gamma_val == 0.0)
    if applicable:
        passed = gamma_val == 0.0
        reason = "gamma=0 confirmed" if passed else "gamma is not zero"
        checks["no_bootstrap_expected"] = {
            "pass": passed,
            "reason": reason,
            "observed": {"gamma": gamma},
            "applicable": True,
        }
    else:
        checks["no_bootstrap_expected"] = {
            "pass": True,
            "reason": "not applicable",
            "observed": {"gamma": gamma},
            "applicable": False,
        }

    # fixed_pi_expected
    alpha_pi = cfg.get("alpha_pi")
    alpha_val = None
    try:
        alpha_val = float(alpha_pi) if alpha_pi is not None else None
    except Exception:
        alpha_val = None
    applicable = check_name == "fixed_pi" or (alpha_val == 0.0)
    if applicable:
        passed = alpha_val == 0.0
        reason = "alpha_pi=0 confirmed" if passed else "alpha_pi not zero"
        checks["fixed_pi_expected"] = {
            "pass": passed,
            "reason": reason,
            "observed": {"alpha_pi": alpha_pi},
            "applicable": True,
        }
    else:
        checks["fixed_pi_expected"] = {
            "pass": True,
            "reason": "not applicable",
            "observed": {"alpha_pi": alpha_pi},
            "applicable": False,
        }

    return checks


def _health_summary(health_checks: Dict[str, Dict[str, Any]], *, incomplete: bool) -> Dict[str, Any]:
    status = "PASS"
    reasons: List[str] = []
    warnings: List[str] = []

    if incomplete:
        status = "FAIL"
        reasons.append("incomplete run")

    for name, entry in health_checks.items():
        applicable = entry.get("applicable", True)
        passed = entry.get("pass", False)
        if applicable and not passed:
            status = "FAIL"
            reasons.append(f"{name}: {entry.get('reason')}")
        if not applicable and name not in {"on_policy_expected", "no_bootstrap_expected", "fixed_pi_expected"}:
            warnings.append(f"{name}: {entry.get('reason')}")

    if status != "FAIL" and warnings:
        status = "WARN"
        reasons.extend(warnings)

    if not reasons:
        reasons.append("all checks passed")

    return {"status": status, "reasons": reasons}


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


def _format_table(metrics: Dict[str, Dict[str, Optional[float]]]) -> List[str]:
    headers = ["metric", "last", "min", "max", "mean_last_k", "slope_last_k"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for metric, stats in metrics.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    _format_float(stats.get("last")),
                    _format_float(stats.get("min")),
                    _format_float(stats.get("max")),
                    _format_float(stats.get("mean_last_k")),
                    _format_float(stats.get("slope_last_k")),
                ]
            )
            + " |"
        )
    return lines


def _format_rows_table(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    headers = columns
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for col in headers:
            value = row.get(col)
            if isinstance(value, float):
                cells.append(_format_float(value))
            else:
                cells.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _pick_key_columns(fieldnames: List[str]) -> List[str]:
    preferred = [
        "iter",
        "step",
        "td_loss",
        "critic_teacher_error",
        "tracking_gap",
        "mean_rho2",
        "w_norm",
        "fixed_point_gap",
        "stability_proxy",
        "dist_mmd2",
    ]
    selected = [col for col in preferred if col in fieldnames]
    if not selected:
        selected = fieldnames[:6]
    return selected


def _metric_first_mean(rows: List[Dict[str, Any]], col: str, *, count: int = 5) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(col)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
        if len(values) >= count:
            break
    if not values:
        return None
    return sum(values) / len(values)


def _suggest_next_steps(
    *,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Dict[str, Optional[float]]],
    rho_threshold: float,
) -> List[str]:
    suggestions: List[str] = []
    triggered = False

    mean_rho2 = summary.get("mean_rho2", {})
    rho_max = mean_rho2.get("max")
    if rho_max is not None and rho_max > rho_threshold:
        triggered = True
        suggestions.append(
            "mean_rho2 is very large -> consider reducing sigma mismatch, increasing beta, increasing p_mix, or enabling rho_clip."
        )

    for metric in ["td_loss", "w_norm"]:
        stats = summary.get(metric, {})
        last_val = stats.get("last")
        slope = stats.get("slope_last_k")
        first_mean = _metric_first_mean(rows, metric)
        if (
            last_val is not None
            and first_mean is not None
            and first_mean > 0
            and last_val > 10 * first_mean
            and slope is not None
            and slope > 0
        ):
            triggered = True
            suggestions.append(
                f"{metric} is exploding -> instability likely; try on_policy / gamma=0 / alpha_pi=0 sanity runs to localize."
            )

    flat_metrics = 0
    for metric in ["td_loss", "w_norm"]:
        stats = summary.get(metric, {})
        min_val = stats.get("min")
        max_val = stats.get("max")
        mean_val = stats.get("mean_last_k")
        if min_val is None or max_val is None or mean_val is None:
            continue
        span = abs(max_val - min_val)
        scale = max(1e-8, abs(mean_val))
        if span / scale < 1e-3:
            flat_metrics += 1
    if flat_metrics >= 2:
        triggered = True
        suggestions.append(
            "metrics are nearly flat -> learning rate may be too small or CSV precision too low; increase lr or logging precision."
        )

    if not triggered:
        suggestions.append("no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.")
    return suggestions


def generate_run_report(
    *,
    run_dir: Union[Path, str],
    config: Optional[Dict[str, Any]] = None,
    curves_csv: Optional[Union[Path, str]] = None,
    probes_dir: Optional[Union[Path, str]] = None,
    stdout_log_path: Optional[Union[Path, str]] = None,
    config_resolved_path: Optional[Union[Path, str]] = None,
    incomplete: bool = False,
    exception: Optional[str] = None,
    last_k: int = 5,
    rho_threshold: float = 1e3,
    max_rows: int = 20,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    resolved_path = Path(config_resolved_path) if config_resolved_path else run_path / "config_resolved.yaml"
    cfg = _load_config(resolved_path if resolved_path.exists() else None, config)
    key_hparams = _extract_key_hparams(cfg)

    curves_path = Path(curves_csv) if curves_csv else run_path / "learning_curves.csv"
    curves_path = curves_path if curves_path.exists() else None

    probes_path = Path(probes_dir) if probes_dir else run_path / "probes"
    probe_paths = sorted(probes_path.glob("*.csv")) if probes_path.exists() else []

    stdout_path = Path(stdout_log_path) if stdout_log_path else run_path / "stdout.log"
    stdout_path = stdout_path if stdout_path.exists() else None

    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir_str = str(checkpoint_dir) if checkpoint_dir.exists() else None

    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    numeric_cols: set[str] = set()
    if curves_path is not None:
        rows, fieldnames, numeric_cols = _parse_csv(curves_path)

    dist_mmd2_value = None
    dist_action_kl_value = None
    dist_action_tv_value = None
    dist_mmd2_meta: Dict[str, Optional[float]] = {"num_samples": None, "mmd_sigma": None}
    dist_action_meta: Dict[str, Optional[float]] = {"action_samples": None}
    stability_probe_step_scale = None
    if "dist_mmd2" in numeric_cols:
        dist_mmd2_value = _last_finite([row.get("dist_mmd2") for row in rows])
    if "dist_action_kl" in numeric_cols:
        dist_action_kl_value = _last_finite([row.get("dist_action_kl") for row in rows])
    if "dist_action_tv" in numeric_cols:
        dist_action_tv_value = _last_finite([row.get("dist_action_tv") for row in rows])
    for probe_path in probe_paths:
        if probe_path.stem == "distribution_probe":
            probe_rows, _, probe_numeric = _parse_csv(probe_path)
            if dist_mmd2_value is None and "mmd2" in probe_numeric:
                dist_mmd2_value = _last_finite([row.get("mmd2") for row in probe_rows])
            if dist_action_kl_value is None and "dist_action_kl" in probe_numeric:
                dist_action_kl_value = _last_finite([row.get("dist_action_kl") for row in probe_rows])
            if dist_action_tv_value is None and "dist_action_tv" in probe_numeric:
                dist_action_tv_value = _last_finite([row.get("dist_action_tv") for row in probe_rows])
            if "num_samples" in probe_numeric:
                dist_mmd2_meta["num_samples"] = _last_finite([row.get("num_samples") for row in probe_rows])
            if "mmd_sigma" in probe_numeric:
                dist_mmd2_meta["mmd_sigma"] = _last_finite([row.get("mmd_sigma") for row in probe_rows])
            if "action_samples" in probe_numeric:
                dist_action_meta["action_samples"] = _last_finite([row.get("action_samples") for row in probe_rows])
            break
    for probe_path in probe_paths:
        if probe_path.stem != "stability_probe":
            continue
        probe_rows, _, probe_numeric = _parse_csv(probe_path)
        if "stability_probe_step_scale" in probe_numeric:
            stability_probe_step_scale = _last_finite(
                [row.get("stability_probe_step_scale") for row in probe_rows]
            )
        break

    time_col = "step" if "step" in numeric_cols else "iter" if "iter" in numeric_cols else None
    learning_summary = _summarize_columns(
        rows,
        numeric_cols,
        columns=[col for col in CORE_COLUMNS if col in numeric_cols],
        time_col=time_col,
        last_k=last_k,
    )

    if not learning_summary and numeric_cols:
        learning_summary = _summarize_columns(rows, numeric_cols, time_col=time_col, last_k=last_k)

    probes_summary: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for probe_path in probe_paths:
        probe_rows, _, probe_numeric = _parse_csv(probe_path)
        if not probe_rows:
            continue
        probe_time = "iter" if "iter" in probe_numeric else None
        probes_summary[probe_path.stem] = _summarize_columns(
            probe_rows,
            probe_numeric,
            time_col=probe_time,
            last_k=last_k,
        )

    head_rows = _sanitize_rows(rows[:max_rows]) if rows else []
    tail_rows = _sanitize_rows(rows[-max_rows:]) if rows else []

    health_checks = _health_checks(
        rows=rows,
        numeric_cols=numeric_cols,
        cfg=cfg,
        rho_threshold=rho_threshold,
        dist_mmd2_value=dist_mmd2_value,
        dist_mmd2_meta=dist_mmd2_meta,
        dist_action_kl_value=dist_action_kl_value,
        dist_action_tv_value=dist_action_tv_value,
        dist_action_meta=dist_action_meta,
    )
    health_summary = _health_summary(health_checks, incomplete=incomplete)

    suggestions = _suggest_next_steps(rows=rows, summary=learning_summary, rho_threshold=rho_threshold)

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_path),
            "host": socket.gethostname(),
            "git_commit": _get_git_commit(run_path),
        },
        "config": {"resolved": cfg, "key_hparams": key_hparams},
        "files": {
            "learning_curves_path": str(curves_path) if curves_path else None,
            "probes_paths": [str(path) for path in probe_paths],
            "stdout_log_path": str(stdout_path) if stdout_path else None,
            "checkpoint_dir": checkpoint_dir_str,
        },
        "health_checks": health_checks,
        "summary_metrics": {"learning_curves": learning_summary, "probes": probes_summary},
        "scale_checks": {
            "train_step_scale": _train_step_scale(cfg),
            "stability_probe_step_scale": stability_probe_step_scale,
            "stability_probe_step_scale_ratio": (
                None
                if _train_step_scale(cfg) in (None, 0) or stability_probe_step_scale is None
                else stability_probe_step_scale / _train_step_scale(cfg)
            ),
        },
        "samples": {"head_rows": head_rows, "tail_rows": tail_rows},
        "health_summary": health_summary,
        "recommendations": suggestions,
        "incomplete": incomplete,
        "exception": exception,
    }

    _write_report_json(report, run_path / "run_report.json")
    (run_path / "run_report.md").write_text(_render_markdown(report, fieldnames))
    return report


def _write_report_json(report: Dict[str, Any], path: Path) -> None:
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    sanitized = _sanitize(report)
    path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=True))


def _render_markdown(report: Dict[str, Any], fieldnames: List[str]) -> str:
    meta = report.get("meta", {})
    key_hparams = report.get("config", {}).get("key_hparams", {})
    health = report.get("health_summary", {})
    health_status = health.get("status", "UNKNOWN")
    reasons = health.get("reasons", [])
    summary = report.get("summary_metrics", {}).get("learning_curves", {})
    probes_summary = report.get("summary_metrics", {}).get("probes", {})
    samples = report.get("samples", {})
    head_rows = samples.get("head_rows", [])
    tail_rows = samples.get("tail_rows", [])
    recommendations = report.get("recommendations", [])
    scale_checks = report.get("scale_checks", {})

    lines: List[str] = ["# Run Report", ""]
    lines.append("## Run Info")
    lines.append(f"- run_dir: {meta.get('run_dir')}")
    lines.append(f"- timestamp: {meta.get('timestamp')}")
    lines.append(f"- seed: {key_hparams.get('seed')}")
    lines.append(
        "- key_hparams: "
        + ", ".join(
            f"{key}={key_hparams.get(key)}"
            for key in [
                "outer_iters",
                "horizon",
                "gamma",
                "alpha_w",
                "alpha_pi",
                "beta",
                "sigma_mu",
                "sigma_pi",
                "p_mix",
            ]
        )
    )
    lines.append("")

    lines.append("## Health")
    reason_text = "; ".join(reasons) if reasons else "no reasons available"
    lines.append(f"- status: {health_status} ({reason_text})")
    lines.append("")

    if scale_checks:
        lines.append("## Scale Checks")
        lines.append(f"- train_step_scale: {scale_checks.get('train_step_scale')}")
        lines.append(f"- stability_probe_step_scale: {scale_checks.get('stability_probe_step_scale')}")
        lines.append(
            "- stability_probe_step_scale_ratio: "
            f"{scale_checks.get('stability_probe_step_scale_ratio')} (expect ~1.0)"
        )
        lines.append("")

    lines.append("## Core Metrics")
    if summary:
        lines.extend(_format_table(summary))
    else:
        lines.append("No learning_curves metrics found.")
    lines.append("")

    if probes_summary:
        lines.append("## Probe Metrics")
        for name, metrics in probes_summary.items():
            lines.append(f"### {name}")
            if metrics:
                lines.extend(_format_table(metrics))
            else:
                lines.append("No probe metrics found.")
            lines.append("")

    lines.append("## Samples (Head)")
    if head_rows:
        key_cols = _pick_key_columns(fieldnames)
        lines.extend(_format_rows_table(head_rows[:8], key_cols))
    else:
        lines.append("No head rows available.")
    lines.append("")

    lines.append("## Samples (Tail)")
    if tail_rows:
        key_cols = _pick_key_columns(fieldnames)
        lines.extend(_format_rows_table(tail_rows[-8:], key_cols))
    else:
        lines.append("No tail rows available.")
    lines.append("")

    lines.append("## Next Steps")
    for suggestion in recommendations:
        lines.append(f"- {suggestion}")
    lines.append("")

    return "\n".join(lines)
