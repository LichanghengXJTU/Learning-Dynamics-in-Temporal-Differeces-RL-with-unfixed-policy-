#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

TS=20251229_000304
RUN_ROOT="outputs/short_runs/${TS}/stepC_v3_rho_audit"
META="$RUN_ROOT/meta"
PLATEAU="$RUN_ROOT/plateau"
INSTAB="$RUN_ROOT/instability"
ANALYSIS="$RUN_ROOT/analysis"
mkdir -p "$META" "$PLATEAU" "$INSTAB" "$ANALYSIS"

# Repo state capture
pwd
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status -sb
git log -1 --oneline
python --version || true
python3 --version || true
pip freeze | head -n 60

# Note: previous attempts timed out at outputs/short_runs/20251228_234540/stepC_v3_rho_audit and outputs/short_runs/20251228_235106/stepC_v3_rho_audit; rerun in this timestamp.

# Step C baseline runs
python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "$PLATEAU" --outer-iters 80 --report-every 20 | tee "$PLATEAU/stdout.log"
python scripts/run_train.py --config configs/train_instability.yaml --output-dir "$INSTAB" --outer-iters 80 --report-every 20 | tee "$INSTAB/stdout.log"

# Run report checks
python scripts/check_run_report.py --run "$PLATEAU" --mode plateau --print-commands | tee "$PLATEAU/check_run_report.log"
python scripts/check_run_report.py --run "$INSTAB" --mode instability --print-commands | tee "$INSTAB/check_run_report.log"

# Step C analysis
python scripts/analyze_step_c.py --plateau-run "$PLATEAU" --instability-run "$INSTAB" --out-dir "$ANALYSIS" | tee "$ANALYSIS/analyze_step_c.log"

# rho audit
python scripts/rho_audit.py --run "$PLATEAU" | tee "$ANALYSIS/rho_audit_plateau.log"
python scripts/rho_audit.py --run "$INSTAB" | tee "$ANALYSIS/rho_audit_instability.log"
python scripts/rho_audit_summary.py --plateau-run "$PLATEAU" --instability-run "$INSTAB" --out "$ANALYSIS/rho_audit_summary.md"

# update norms summary
python scripts/summarize_updates.py --plateau-run "$PLATEAU" --instability-run "$INSTAB" --out "$ANALYSIS/updates_scale_summary.md"

# Optional sweep config copies
python3 - <<'PY'
import json
from pathlib import Path

base = json.loads(Path("configs/train_instability.yaml").read_text())

variants = {
    "instab_alpha_boost.yaml": {
        "alpha_w": 0.3,
        "alpha_pi": 0.2,
        "beta": 0.005,
    },
    "instab_sigma_mismatch.yaml": {
        "sigma_mu": 0.2,
        "sigma_pi": 0.6,
    },
    "instab_pmix0.yaml": {
        "env": {**base.get("env", {}), "p_mix": 0.0},
    },
    "instab_theta_radius12.yaml": {
        "theta_radius": 12.0,
    },
    "instab_rho_clip_on.yaml": {
        "rho_clip": 5.0,
        "disable_rho_clip": False,
    },
}

out_dir = Path("outputs/short_runs/20251229_000304/stepC_v3_rho_audit/meta/sweeps")
out_dir.mkdir(parents=True, exist_ok=True)
for name, updates in variants.items():
    cfg = json.loads(json.dumps(base))
    for key, value in updates.items():
        if key == "env" and isinstance(value, dict):
            cfg.setdefault("env", {}).update(value)
        else:
            cfg[key] = value
    out_dir.joinpath(name).write_text(json.dumps(cfg, indent=2))
PY

# Health summary quick check
python3 - <<'PY'
import json
from pathlib import Path
for label, path in [
    ("plateau", Path("outputs/short_runs/20251229_000304/stepC_v3_rho_audit/plateau")),
    ("instability", Path("outputs/short_runs/20251229_000304/stepC_v3_rho_audit/instability")),
]:
    data = json.loads((path / "run_report.json").read_text())
    summary = data.get("health_summary") or data.get("health") or {}
    print(label, summary)
PY
