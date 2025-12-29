#!/usr/bin/env python3
"""Check that probes do not perturb training RNG state."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdrl_unfixed_ac.algos.train_unfixed_ac import load_train_config, train_unfixed_ac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test probes are side-effect free.")
    parser.add_argument("--config", type=str, required=True, help="Training config path.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root directory.")
    parser.add_argument("--outer-iters", type=int, default=120, help="Number of outer iters to run.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--tol", type=float, default=1e-12, help="Absolute tolerance for matching metrics.")
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        for row in reader:
            parsed: Dict[str, float] = {}
            for field in reader.fieldnames:
                raw = row.get(field, "")
                try:
                    parsed[field] = float(raw)
                except (TypeError, ValueError):
                    parsed[field] = math.nan
            rows.append(parsed)
    return rows


def _probe_iters(probes_dir: Path) -> set[int]:
    probe_iters: set[int] = set()
    for path in probes_dir.glob("*_probe.csv"):
        rows = _load_rows(path)
        for row in rows:
            raw_iter = row.get("iter")
            if raw_iter is None or not math.isfinite(raw_iter):
                continue
            probe_iters.add(int(raw_iter))
    return probe_iters


def _compare_rows(
    rows_a: List[Dict[str, float]],
    rows_b: List[Dict[str, float]],
    metrics: List[str],
    tol: float,
) -> Tuple[bool, str]:
    if len(rows_a) != len(rows_b):
        return False, f"Row count mismatch: {len(rows_a)} vs {len(rows_b)}"
    for idx, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        iter_a = row_a.get("iter")
        iter_b = row_b.get("iter")
        if iter_a != iter_b:
            return False, f"Iter mismatch at row {idx}: {iter_a} vs {iter_b}"
        for metric in metrics:
            val_a = row_a.get(metric, math.nan)
            val_b = row_b.get(metric, math.nan)
            if math.isnan(val_a) and math.isnan(val_b):
                continue
            if not math.isfinite(val_a) or not math.isfinite(val_b):
                return False, f"Non-finite {metric} at iter {iter_a}: {val_a} vs {val_b}"
            if abs(val_a - val_b) > tol:
                return False, f"{metric} mismatch at iter {iter_a}: {val_a} vs {val_b}"
    return True, ""


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output root {output_root} is not empty; choose a new directory.")

    cfg = load_train_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["outer_iters"] = int(args.outer_iters)

    probes_on = deepcopy(cfg)
    probes_on.setdefault("probes", {})["enabled"] = True
    probes_on["output_dir"] = str(output_root / "probes_on")

    probes_off = deepcopy(cfg)
    probes_off.setdefault("probes", {})["enabled"] = False
    probes_off["output_dir"] = str(output_root / "probes_off")

    train_unfixed_ac(probes_on)
    train_unfixed_ac(probes_off)

    rows_on = _load_rows(Path(probes_on["output_dir"]) / "learning_curves.csv")
    rows_off = _load_rows(Path(probes_off["output_dir"]) / "learning_curves.csv")

    metrics = ["td_loss", "w_norm", "critic_teacher_error"]
    ok, message = _compare_rows(rows_on, rows_off, metrics, args.tol)
    probe_iters = _probe_iters(Path(probes_on["output_dir"]) / "probes")
    if not ok:
        if message.startswith("td_loss mismatch") or message.startswith("w_norm mismatch") or message.startswith("critic_teacher_error mismatch"):
            parts = message.split("iter ")
            iter_val = None
            if len(parts) > 1:
                try:
                    iter_val = int(float(parts[1].split(":")[0]))
                except ValueError:
                    iter_val = None
            if iter_val is not None:
                triggered = iter_val in probe_iters
                print(f"{message}; probe_triggered={triggered}")
                return
        print(message)
        return

    print("Probes side-effect free: metrics match within tolerance.")


if __name__ == "__main__":
    main()
