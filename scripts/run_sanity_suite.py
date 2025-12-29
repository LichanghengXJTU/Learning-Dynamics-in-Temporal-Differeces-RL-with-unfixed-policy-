#!/usr/bin/env python3
"""Run a short sanity suite for unfixed actor-critic training."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.algos.train_unfixed_ac import load_train_config, train_unfixed_ac
from tdrl_unfixed_ac.reporting import generate_run_report

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_BASE_CONFIG = ROOT / "configs" / "train_sanity.yaml"

CHECK_TEMPLATES = [
    ("on_policy", ROOT / "configs" / "sanity" / "on_policy.yaml"),
    ("no_bootstrap", ROOT / "configs" / "sanity" / "no_bootstrap.yaml"),
    ("fixed_pi", ROOT / "configs" / "sanity" / "fixed_pi.yaml"),
    ("full_triad_short", ROOT / "configs" / "sanity" / "full_triad_short.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity-suite checks for unfixed actor-critic.")
    parser.add_argument("--base", type=str, default=None, help="Path to base training config yaml/json.")
    parser.add_argument(
        "--out_root",
        type=str,
        default="outputs/sanity_suite",
        help="Output root for sanity suite runs.",
    )
    return parser.parse_args()


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_payload(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    return payload or {}


def _resolve_relative(path_str: str, base_dir: Path) -> str:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    options = [ROOT / candidate, base_dir / candidate, Path.cwd() / candidate]
    for option in options:
        if option.exists():
            return str(option.resolve())
    return str(candidate)


def _coerce_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _coerce_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_config(v) for v in value]
    return value


def _dump_config(path: Path, cfg: Dict[str, Any]) -> None:
    payload = _coerce_config(cfg)
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
    else:
        path.write_text(json.dumps(payload, indent=2))


def _ensure_run_reports(run_dir: Path, cfg: Optional[Dict[str, Any]], exception: Optional[str]) -> None:
    report_json = run_dir / "run_report.json"
    report_md = run_dir / "run_report.md"
    if report_json.exists() and report_md.exists():
        return
    try:
        generate_run_report(
            run_dir=run_dir,
            config=cfg,
            curves_csv=run_dir / "learning_curves.csv",
            probes_dir=run_dir / "probes",
            stdout_log_path=run_dir / "stdout.log",
            incomplete=exception is not None,
            exception=exception,
        )
    except Exception:
        report_json.write_text("{}\n")
        report_md.write_text("Run report pending.\n")


def _log_config_summary(name: str, base_path: Path, template_path: Optional[Path], cfg: Dict[str, Any]) -> None:
    print(f"== Sanity check: {name} ==")
    print(f"Base config: {base_path}")
    print(f"Template config: {template_path if template_path else 'None'}")
    print("Key hyperparameters:")
    keys = [
        "seed",
        "outer_iters",
        "trajectories",
        "horizon",
        "gamma",
        "alpha_w",
        "alpha_pi",
        "beta",
        "sigma_mu",
        "sigma_pi",
        "K_mc",
        "theta_radius",
        "checkpoint_every",
        "log_every",
    ]
    for key in keys:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")
    print(f"env_config_path: {cfg.get('env_config_path')}")
    env_overrides = cfg.get("env", {})
    if env_overrides:
        print(f"env overrides: {env_overrides}")
    probes_cfg = cfg.get("probes", {})
    if probes_cfg:
        print(f"probes.enabled: {probes_cfg.get('enabled', False)}")
        print(f"probes.every: {probes_cfg.get('every', 0)}")
    if name == "on_policy":
        print("Expected: rho ~= 1 (mu=pi, sigma_mu=sigma_pi).")
    if name == "no_bootstrap":
        print("Expected: gamma=0 (bootstrap disabled).")
    if name == "fixed_pi":
        print("Expected: alpha_pi=0 (actor update disabled).")


def _apply_required_overrides(name: str, cfg: Dict[str, Any]) -> None:
    if name == "on_policy":
        cfg["beta"] = 1.0
        sigma_pi = float(cfg.get("sigma_pi", 1.0))
        cfg["sigma_mu"] = sigma_pi
    elif name == "no_bootstrap":
        cfg["gamma"] = 0.0
    elif name == "fixed_pi":
        cfg["alpha_pi"] = 0.0


def _resolve_run_config(base_path: Path, template_path: Path, run_dir: Path, name: str) -> Dict[str, Any]:
    cfg = load_train_config(str(base_path))
    if template_path.exists():
        overrides = _load_payload(template_path)
        if overrides:
            _deep_update(cfg, overrides)
    _apply_required_overrides(name, cfg)
    cfg["check_name"] = name
    cfg["output_dir"] = str(run_dir)
    env_path = cfg.get("env_config_path")
    if env_path:
        cfg["env_config_path"] = _resolve_relative(str(env_path), base_path.parent)
    return cfg


def _run_preflight(suite_dir: Path, env_config_path: Optional[str]) -> None:
    preflight_dir = suite_dir / "preflight_smoke"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    log_path = preflight_dir / "stdout.log"
    smoke_script = ROOT / "scripts" / "smoke_rollout.py"

    with log_path.open("w") as log:
        log.write("== Preflight: compileall ==\n")
        try:
            subprocess.run(
                [sys.executable, "-m", "compileall", "."],
                cwd=str(ROOT),
                stdout=log,
                stderr=log,
                check=False,
            )
        except Exception:
            traceback.print_exc(file=log)

        log.write("\n== Preflight: smoke_rollout ==\n")
        if smoke_script.exists():
            cmd = [sys.executable, str(smoke_script), "--steps", "300"]
            if env_config_path:
                cmd.extend(["--config", env_config_path])
            try:
                subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=log, check=False)
            except Exception:
                traceback.print_exc(file=log)
        else:
            log.write("smoke_rollout.py not found; skipping.\n")


def _run_check(name: str, base_path: Path, template_path: Path, suite_dir: Path) -> None:
    run_dir = suite_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "stdout.log"
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        cfg: Optional[Dict[str, Any]] = None
        exception: Optional[str] = None
        try:
            cfg = _resolve_run_config(base_path, template_path, run_dir, name)
            _dump_config(run_dir / "config_resolved.yaml", cfg)
            _log_config_summary(name, base_path, template_path, cfg)
            result = train_unfixed_ac(cfg)
            print(f"Training complete. Logs at {result['csv_path']}")
        except Exception:
            print("Run failed with exception:")
            exception = traceback.format_exc()
            print(exception)
        finally:
            _ensure_run_reports(run_dir, cfg, exception)


def main() -> None:
    args = parse_args()
    base_path = Path(args.base) if args.base else DEFAULT_BASE_CONFIG
    if not base_path.is_absolute():
        base_path = (ROOT / base_path).resolve()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (ROOT / out_root).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = out_root / timestamp
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_train_config(str(base_path))
    env_config_path = base_cfg.get("env_config_path")
    if env_config_path:
        env_config_path = _resolve_relative(str(env_config_path), base_path.parent)

    print(f"Sanity suite root: {suite_dir}")
    _run_preflight(suite_dir, env_config_path)

    for name, template_path in CHECK_TEMPLATES:
        _run_check(name, base_path, template_path, suite_dir)


if __name__ == "__main__":
    main()
