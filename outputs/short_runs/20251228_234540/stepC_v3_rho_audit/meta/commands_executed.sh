#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

TS=20251228_234540
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
