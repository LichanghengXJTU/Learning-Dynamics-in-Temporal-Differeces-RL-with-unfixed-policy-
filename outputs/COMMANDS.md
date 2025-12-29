# COMMANDS

All commands assume you are at the repo root. Outputs are written under `outputs/`.

## Step A: Preflight (code + environment correctness)

### A1) Dependencies (no requirements file found)
```bash
python -m pip install --upgrade pip
python -m pip install numpy pyyaml matplotlib
```
Expected: packages installed; no output files.

### A2) Syntax check
```bash
python -m compileall .
```
Expected: no errors; console output only.

### A3) Smoke rollout (reward/event/feature stats)
```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=outputs/preflight_smoke/$TS
mkdir -p $OUT
python scripts/smoke_rollout.py --steps 300 --config configs/default.yaml | tee $OUT/stdout.log
```
Expected: `outputs/preflight_smoke/<ts>/stdout.log` with reward stats, events counts, and feature dims.

### A4) Short training run (outer_iters=5)
```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=outputs/preflight_train/$TS
mkdir -p $OUT
python scripts/run_train.py --config configs/train_sanity.yaml --output-dir $OUT --outer-iters 5 --report-every 1 | tee $OUT/stdout.log
```
Expected: `outputs/preflight_train/<ts>/learning_curves.csv`, `outputs/preflight_train/<ts>/run_report.json`, `outputs/preflight_train/<ts>/checkpoints/`.

## Step B: Sanity suite (4 checks + summary)

Run the suite (note: wrapper exists at `tools/run_sanity_suite.py`):
```bash
python tools/run_sanity_suite.py --base configs/train_sanity.yaml --out_root outputs/sanity_suite
```
Expected: `outputs/sanity_suite/<timestamp>/{on_policy,no_bootstrap,fixed_pi,full_triad_short}/` + `preflight_smoke/`.

Aggregate reports into `SUMMARY.md`:
```bash
LATEST=$(ls -td outputs/sanity_suite/*/ | head -1)
python scripts/aggregate_reports.py --root $LATEST --out ${LATEST%/}/SUMMARY.md --out-csv ${LATEST%/}/SUMMARY.csv
```
Expected: `outputs/sanity_suite/<timestamp>/SUMMARY.md` with PASS/WARN/FAIL table.

## Step C: Short-run tuning (plateau vs instability)

Short plateau-leaning run:
```bash
TS=$(date +%Y%m%d_%H%M%S)
ROOT=outputs/short_runs/$TS
OUT=$ROOT/plateau
mkdir -p $OUT
python scripts/run_train.py --config configs/train_plateau.yaml --output-dir $OUT --outer-iters 80 --report-every 20 | tee $OUT/stdout.log
python scripts/check_run_report.py --run $OUT --mode plateau --print-commands
```
Expected: `outputs/short_runs/<ts>/plateau/run_report.json` and probes in `outputs/short_runs/<ts>/plateau/probes/`.

Short instability-leaning run:
```bash
TS=$(date +%Y%m%d_%H%M%S)
ROOT=outputs/short_runs/$TS
OUT=$ROOT/instability
mkdir -p $OUT
python scripts/run_train.py --config configs/train_instability.yaml --output-dir $OUT --outer-iters 80 --report-every 20 | tee $OUT/stdout.log
python scripts/check_run_report.py --run $OUT --mode instability --print-commands
```
Expected: `outputs/short_runs/<ts>/instability/run_report.json` and probes in `outputs/short_runs/<ts>/instability/probes/`.

How to check mean_rho2 / stability_proxy / fixed_point_drift / dist_mmd2:
```bash
python scripts/check_run_report.py --run outputs/short_runs/<ts>/plateau --mode plateau --print-commands
python scripts/check_run_report.py --run outputs/short_runs/<ts>/instability --mode instability --print-commands
```
If metrics are off, follow the printed rerun commands. Typical manual overrides:
- Plateau too unstable: lower `--alpha-w`, `--alpha-pi`, increase `--beta`.
- Instability too mild: raise `--alpha-w`, `--alpha-pi`, lower `--beta`, reduce `--p-mix`.

## Step D: Long run (background + resume + self-heal)

Training writes checkpoints to `outputs/<run>/checkpoints/` and keeps `latest.pt` for resume.
Periodic reports are generated with `--report-every` and/or `--report-every-seconds`.

### tmux/screen option
```bash
RUN=outputs/long_runs/plateau_$(date +%Y%m%d_%H%M%S)
tmux new -s tdrl_long "bash scripts/run_pipeline.sh --long-only --long-dir $RUN --long-config configs/train_plateau.yaml --long-iters 2000"
```
Expected: `outputs/long_runs/<run>/checkpoints/latest.pt` and partial `run_report.json` updates.

### nohup option
```bash
RUN=outputs/long_runs/plateau_$(date +%Y%m%d_%H%M%S)
nohup bash scripts/run_pipeline.sh --long-only --long-dir $RUN --long-config configs/train_plateau.yaml --long-iters 2000 > $RUN/nohup.log 2>&1 &
```
Expected: `outputs/long_runs/<run>/nohup.log`, `outputs/long_runs/<run>/checkpoints/latest.pt`.

## Step E: Probe analysis and plots

Single run analysis (learning curves + probes + metrics table):
```bash
python scripts/plot_from_outputs.py --run outputs/short_runs/<ts>/plateau --out-dir outputs/analysis/plateau_<ts>
```
Expected: `outputs/analysis/plateau_<ts>/{learning_curves.png,stability_probe.png,fixed_point_probe.png,distribution_probe.png,metrics_table.md}`.

Multi-run summary (CSV + Markdown):
```bash
python scripts/aggregate_reports.py --root outputs/short_runs/<ts> --out outputs/analysis/short_runs_<ts>.md --out-csv outputs/analysis/short_runs_<ts>.csv
```
Expected: `outputs/analysis/short_runs_<ts>.md` with a label column (plateau/instability).

## One-shot pipeline

Run everything in order (A->E). Failures are logged to `outputs/pipeline_failures.log` and the pipeline continues:
```bash
bash scripts/run_pipeline.sh
```
Expected: outputs under `outputs/preflight_*`, `outputs/sanity_suite/`, `outputs/short_runs/`, `outputs/analysis/`, `outputs/long_runs/`.

## Step F: On-policy MMD calibration

Run the calibration sweep (uses local venv):
```bash
.venv/bin/python scripts/calibrate_on_policy_mmd.py
```
Expected: `outputs/calibration_on_policy_mmd/{calibration.csv,offpolicy_calibration.csv,calibration_summary.md}`.
