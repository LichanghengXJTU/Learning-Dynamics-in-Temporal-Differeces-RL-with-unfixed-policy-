#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUTPUTS="$ROOT/outputs"
FAIL_LOG="$OUTPUTS/pipeline_failures.log"
mkdir -p "$OUTPUTS"

LONG_ONLY=0
SKIP_LONG=0
LONG_DIR=""
LONG_CONFIG="${LONG_CONFIG:-configs/train_plateau.yaml}"
LONG_ITERS="${LONG_ITERS:-2000}"
REPORT_EVERY="${REPORT_EVERY:-50}"
REPORT_EVERY_SECONDS="${REPORT_EVERY_SECONDS:-600}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --long-only)
      LONG_ONLY=1
      shift
      ;;
    --skip-long)
      SKIP_LONG=1
      shift
      ;;
    --long-dir)
      LONG_DIR="$2"
      shift 2
      ;;
    --long-config)
      LONG_CONFIG="$2"
      shift 2
      ;;
    --long-iters)
      LONG_ITERS="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

log_fail() {
  local name="$1"
  local status="$2"
  local cmd="$3"
  printf "%s | %s | exit=%s | %s\n" "$(date '+%F %T')" "$name" "$status" "$cmd" >> "$FAIL_LOG"
}

run_cmd() {
  local name="$1"
  shift
  local cmd="$*"
  echo "== $name =="
  bash -lc "set -o pipefail; $cmd"
  local status=$?
  if [[ $status -ne 0 ]]; then
    log_fail "$name" "$status" "$cmd"
  fi
  return 0
}

run_resume_loop() {
  local name="$1"
  shift
  local cmd="$*"
  local attempt=1
  local max_restarts="${MAX_RESTARTS:-0}"
  while true; do
    echo "== $name (attempt $attempt) =="
    bash -lc "set -o pipefail; $cmd"
    local status=$?
    if [[ $status -eq 0 ]]; then
      break
    fi
    log_fail "$name" "$status" "$cmd"
    if [[ $max_restarts -gt 0 && $attempt -ge $max_restarts ]]; then
      echo "Reached MAX_RESTARTS=$max_restarts; stopping resume loop."
      break
    fi
    attempt=$((attempt + 1))
    sleep 5
  done
}

TS="$(date +%Y%m%d_%H%M%S)"

if [[ $LONG_ONLY -eq 0 ]]; then
  PREFLIGHT_SMOKE="$OUTPUTS/preflight_smoke/$TS"
  PREFLIGHT_TRAIN="$OUTPUTS/preflight_train/$TS"
  SHORT_ROOT="$OUTPUTS/short_runs/$TS"

  run_cmd "preflight_compile" "python -m compileall ."

  run_cmd "preflight_smoke" "mkdir -p $PREFLIGHT_SMOKE && python scripts/smoke_rollout.py --steps 300 --config configs/default.yaml | tee $PREFLIGHT_SMOKE/stdout.log"

  run_cmd "preflight_train" "mkdir -p $PREFLIGHT_TRAIN && python scripts/run_train.py --config configs/train_sanity.yaml --output-dir $PREFLIGHT_TRAIN --outer-iters 5 --report-every 1 | tee $PREFLIGHT_TRAIN/stdout.log"

  run_cmd "sanity_suite" "python tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite | tee outputs/sanity_suite/sanity_suite_${TS}.log"

  SANITY_DIR=$(ls -td outputs/sanity_suite/*/ 2>/dev/null | head -1 | tr -d '\n')
  if [[ -n "$SANITY_DIR" ]]; then
    run_cmd "sanity_summary" "python scripts/aggregate_reports.py --root $SANITY_DIR --out ${SANITY_DIR%/}/SUMMARY.md --out-csv ${SANITY_DIR%/}/SUMMARY.csv"
  fi

  SHORT_PLATEAU="$SHORT_ROOT/plateau"
  SHORT_INSTABILITY="$SHORT_ROOT/instability"

  run_cmd "short_plateau" "mkdir -p $SHORT_PLATEAU && python scripts/run_train.py --config configs/train_plateau.yaml --output-dir $SHORT_PLATEAU --outer-iters 80 --report-every 20 | tee $SHORT_PLATEAU/stdout.log"
  run_cmd "check_plateau" "python scripts/check_run_report.py --run $SHORT_PLATEAU --mode plateau --print-commands"

  run_cmd "short_instability" "mkdir -p $SHORT_INSTABILITY && python scripts/run_train.py --config configs/train_instability.yaml --output-dir $SHORT_INSTABILITY --outer-iters 80 --report-every 20 | tee $SHORT_INSTABILITY/stdout.log"
  run_cmd "check_instability" "python scripts/check_run_report.py --run $SHORT_INSTABILITY --mode instability --print-commands"

  run_cmd "analysis_plateau" "python scripts/plot_from_outputs.py --run $SHORT_PLATEAU --out-dir outputs/analysis/${TS}_plateau"
  run_cmd "analysis_instability" "python scripts/plot_from_outputs.py --run $SHORT_INSTABILITY --out-dir outputs/analysis/${TS}_instability"
  run_cmd "analysis_short_summary" "python scripts/aggregate_reports.py --root $SHORT_ROOT --out outputs/analysis/${TS}_short_runs.md --out-csv outputs/analysis/${TS}_short_runs.csv"
fi

if [[ $SKIP_LONG -eq 0 ]]; then
  if [[ -z "$LONG_DIR" ]]; then
    LONG_DIR="$OUTPUTS/long_runs/${TS}_long"
  fi
  mkdir -p "$LONG_DIR"
  run_resume_loop "long_run" "python scripts/run_train.py --config $LONG_CONFIG --output-dir $LONG_DIR --outer-iters $LONG_ITERS --resume --report-every $REPORT_EVERY --report-every-seconds $REPORT_EVERY_SECONDS | tee -a $LONG_DIR/stdout.log"
fi
