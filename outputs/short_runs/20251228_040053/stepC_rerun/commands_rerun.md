# Step C rerun commands (20251228_040053)

python3 scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251228_040053/stepC_rerun/plateau" --outer-iters 200 --report-every 20 | tee "outputs/short_runs/20251228_040053/stepC_rerun/plateau/stdout.log"
python3 scripts/check_run_report.py --run "outputs/short_runs/20251228_040053/stepC_rerun/plateau" --mode plateau --print-commands

python3 scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251228_040053/stepC_rerun/instability" --outer-iters 200 --report-every 20 | tee "outputs/short_runs/20251228_040053/stepC_rerun/instability/stdout.log"
python3 scripts/check_run_report.py --run "outputs/short_runs/20251228_040053/stepC_rerun/instability" --mode instability --print-commands

python3 scripts/analyze_step_c.py --plateau-run "outputs/short_runs/20251228_040053/stepC_rerun/plateau" --instability-run "outputs/short_runs/20251228_040053/stepC_rerun/instability" --out-dir "outputs/short_runs/20251228_040053/stepC_rerun/stepC_analysis"
