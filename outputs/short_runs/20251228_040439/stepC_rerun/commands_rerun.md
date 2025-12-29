# Step C rerun commands (segmented for long runtime)

# plateau run (segment to 90 iters, then resume to 130/170/200)
.venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/plateau" --outer-iters 90 --report-every 20 | tee "outputs/short_runs/20251228_040439/stepC_rerun/plateau/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/plateau" --outer-iters 130 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/plateau/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/plateau" --outer-iters 170 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/plateau/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/plateau" --outer-iters 200 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/plateau/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251228_040439/stepC_rerun/plateau" --mode plateau --print-commands

# instability run (segment to 70 iters, then resume to 110/150/190/200)
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/instability" --outer-iters 70 --report-every 20 | tee "outputs/short_runs/20251228_040439/stepC_rerun/instability/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/instability" --outer-iters 110 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/instability/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/instability" --outer-iters 150 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/instability/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/instability" --outer-iters 190 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/instability/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251228_040439/stepC_rerun/instability" --outer-iters 200 --resume --report-every 20 | tee -a "outputs/short_runs/20251228_040439/stepC_rerun/instability/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251228_040439/stepC_rerun/instability" --mode instability --print-commands

# analysis
.venv/bin/python scripts/analyze_step_c.py --plateau-run "outputs/short_runs/20251228_040439/stepC_rerun/plateau" --instability-run "outputs/short_runs/20251228_040439/stepC_rerun/instability" --out-dir "outputs/short_runs/20251228_040439/stepC_rerun/stepC_analysis"
