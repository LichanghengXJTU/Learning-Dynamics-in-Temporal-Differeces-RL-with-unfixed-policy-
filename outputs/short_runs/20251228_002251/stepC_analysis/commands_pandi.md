# Commands: Step C closed loop

## 0) Optional: local venv for plotting (numpy/matplotlib)
python3 -m venv outputs/short_runs/20251228_002251/stepC_analysis/.venv
outputs/short_runs/20251228_002251/stepC_analysis/.venv/bin/pip install numpy matplotlib
PY=outputs/short_runs/20251228_002251/stepC_analysis/.venv/bin/python

## 1) Re-run plateau short-run (80 iters) + report
python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/short_runs/20251228_002251/plateau_rerun --outer-iters 80
python scripts/check_run_report.py --run outputs/short_runs/20251228_002251/plateau_rerun --mode plateau --print-commands

## 2) Re-run instability_fix short-run (80 iters) + report
python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/short_runs/20251228_002251/instability_fix --alpha-w 0.3 --alpha-pi 0.2 --beta 0.005 --p-mix 0.0 --outer-iters 80
python scripts/check_run_report.py --run outputs/short_runs/20251228_002251/instability_fix --mode instability --print-commands

## 3) Step C analysis (plots + markdown)
${PY:-python3} scripts/analyze_step_c.py \
  --plateau-run outputs/short_runs/20251228_002251/plateau_rerun \
  --instability-run outputs/short_runs/20251228_002251/instability_fix \
  --out-dir outputs/short_runs/20251228_002251/stepC_analysis_rerun

## 4) If instability_fix is still stable, try a boosted run
python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/short_runs/20251228_002251/instability_boost --alpha-w 0.35 --alpha-pi 0.25 --beta 0.0025 --p-mix 0.0 --outer-iters 200
python scripts/check_run_report.py --run outputs/short_runs/20251228_002251/instability_boost --mode instability --print-commands

## 5) Optional candidate: increase E[rho^2] via sigma tuning (not default)
python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/short_runs/20251228_002251/instability_rho2_candidate --sigma-mu 0.45 --sigma-pi 0.25 --outer-iters 80
python scripts/check_run_report.py --run outputs/short_runs/20251228_002251/instability_rho2_candidate --mode instability --print-commands
