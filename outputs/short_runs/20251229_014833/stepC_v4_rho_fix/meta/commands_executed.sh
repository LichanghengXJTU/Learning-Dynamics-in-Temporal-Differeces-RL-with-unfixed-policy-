#!/bin/sh
set -e

date +%Y%m%d_%H%M%S
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta" "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis"
ls
rg -n "log_prob|rho|sample_action|score" tdrl_unfixed_ac scripts configs
sed -n '1,220p' tdrl_unfixed_ac/algos/unfixed_ac.py
rg -n "clip_action|action_clip|v_max|tanh|atanh" tdrl_unfixed_ac
sed -n '1,220p' tdrl_unfixed_ac/envs/torus_gg.py
sed -n '220,420p' tdrl_unfixed_ac/envs/torus_gg.py
sed -n '1,220p' tdrl_unfixed_ac/algos/train_unfixed_ac.py
sed -n '220,420p' tdrl_unfixed_ac/algos/train_unfixed_ac.py
sed -n '1,200p' tdrl_unfixed_ac/probes/common.py
rg -n "LinearGaussianPolicy" -S tdrl_unfixed_ac scripts
sed -n '1,160p' tdrl_unfixed_ac/probes/fixed_point_probe.py
sed -n '1,240p' tdrl_unfixed_ac/probes/distribution_probe.py
sed -n '1,120p' tdrl_unfixed_ac/probes/stability_probe.py
rg -n "LinearGaussianPolicy\(" -S tdrl_unfixed_ac | cat
sed -n '180,240p' tdrl_unfixed_ac/envs/render.py
sed -n '740,820p' scripts/analyze_step_c.py
sed -n '1,220p' scripts/rho_audit.py
cat > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/rho_fix_design.md" <<'EOF'
# rho consistency fix design

## Code reading (pre-fix)
1) Action clipping
- env: `TorusGobletGhostEnv.step()` applies `_clip_action` (L2 ball of radius `v_max`) before computing features/reward.
- train loop: `train_unfixed_ac` also applied `_clip_action(a_raw, env.v_max)` before `env.step`, so the env clip could still fire (double clipping).

2) Policy form
- `LinearGaussianPolicy` used a diagonal Gaussian with constant scalar sigma.
- mean parameterization: `mean = theta^T psi / sqrt(actor_dim)` via `policy_mean`.
- `sample_action`: draws `a ~ Normal(mean, sigma)`.
- `log_prob`: isotropic Gaussian log density at `a`.
- `score`: grad wrt theta: `outer(psi, (a-mean)) / (sigma^2 * sqrt(actor_dim))`.

3) rho + score action source
- `rho_raw` computed from `log_prob(a_raw)`; `rho_exec` from `log_prob(a_exec)` where `a_exec` is clipped.
- training uses `rho_exec` for updates; `score` uses `a_exec` (post-clip).

## Fix plan (implemented)
- Replace the policy with tanh-squashed Gaussian actions so the executed actions lie in-bounds and match the distribution assumed by `log_prob`/`score`.
- Sampling: draw `u ~ Normal(mean, sigma)`, then `a_exec = v_max * tanh(u)`.
- Log-prob for executed action uses change-of-variables:
  - `u = atanh(a_exec / v_max)`
  - `logp(a_exec) = logN(u; mean, sigma) - sum_i[log(v_max) + log(1 - tanh(u_i)^2)]`
  - `log(1 - tanh^2)` computed with a stable `logaddexp` form.
- Score: compute `u = atanh(a_exec / v_max)` then reuse Gaussian score on `u`:
  - `∂/∂mean logN(u; mean, sigma) = (u-mean)/sigma^2` and chain via mean=theta^T psi / sqrt(actor_dim).
- Training loop uses only `a_exec`:
  - `env.step(a_exec)` (no extra clipping in the loop).
  - `rho` uses `log_prob(a_exec)` for both pi and mu.
  - `score` uses `a_exec`.
- For audit: `rho_raw` is computed in pre-squash space using `u = atanh(a_exec / v_max)`; `rho_exec` uses `log_prob(a_exec)`. These should match (Jacobian cancels in the ratio).
- Action clipping helper + env clip updated to per-component `[-v_max, v_max]` so env clipping is a no-op for tanh-squashed actions.

## Expected consistency
- `a_exec` is the only action used for env transitions, rho, and score.
- `mean_rho2_raw` and `mean_rho2_exec` should converge (no raw/exec mismatch).
- `clip_fraction` and `|a_exec-clip(a_exec)|` should be ~0 if squashing matches env bounds.
# NOTE: a previous attempt to append this here-doc log failed with zsh: command not found: EOF
cat > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/rho_fix_design.md" <<'EOF_RHO'
# rho consistency fix design

## Code reading (pre-fix)
1) Action clipping
- env: `TorusGobletGhostEnv.step()` applies `_clip_action` (L2 ball of radius `v_max`) before computing features/reward.
- train loop: `train_unfixed_ac` also applied `_clip_action(a_raw, env.v_max)` before `env.step`, so the env clip could still fire (double clipping).

2) Policy form
- `LinearGaussianPolicy` used a diagonal Gaussian with constant scalar sigma.
- mean parameterization: `mean = theta^T psi / sqrt(actor_dim)` via `policy_mean`.
- `sample_action`: draws `a ~ Normal(mean, sigma)`.
- `log_prob`: isotropic Gaussian log density at `a`.
- `score`: grad wrt theta: `outer(psi, (a-mean)) / (sigma^2 * sqrt(actor_dim))`.

3) rho + score action source
- `rho_raw` computed from `log_prob(a_raw)`; `rho_exec` from `log_prob(a_exec)` where `a_exec` is clipped.
- training uses `rho_exec` for updates; `score` uses `a_exec` (post-clip).

## Fix plan (implemented)
- Replace the policy with tanh-squashed Gaussian actions so the executed actions lie in-bounds and match the distribution assumed by `log_prob`/`score`.
- Sampling: draw `u ~ Normal(mean, sigma)`, then `a_exec = v_max * tanh(u)`.
- Log-prob for executed action uses change-of-variables:
  - `u = atanh(a_exec / v_max)`
  - `logp(a_exec) = logN(u; mean, sigma) - sum_i[log(v_max) + log(1 - tanh(u_i)^2)]`
  - `log(1 - tanh^2)` computed with a stable `logaddexp` form.
- Score: compute `u = atanh(a_exec / v_max)` then reuse Gaussian score on `u`:
  - `∂/∂mean logN(u; mean, sigma) = (u-mean)/sigma^2` and chain via mean=theta^T psi / sqrt(actor_dim).
- Training loop uses only `a_exec`:
  - `env.step(a_exec)` (no extra clipping in the loop).
  - `rho` uses `log_prob(a_exec)` for both pi and mu.
  - `score` uses `a_exec`.
- For audit: `rho_raw` is computed in pre-squash space using `u = atanh(a_exec / v_max)`; `rho_exec` uses `log_prob(a_exec)`. These should match (Jacobian cancels in the ratio).
- Action clipping helper + env clip updated to per-component `[-v_max, v_max]` so env clipping is a no-op for tanh-squashed actions.

## Expected consistency
- `a_exec` is the only action used for env transitions, rho, and score.
- `mean_rho2_raw` and `mean_rho2_exec` should converge (no raw/exec mismatch).
- `clip_fraction` and `|a_exec-clip(a_exec)|` should be ~0 if squashing matches env bounds.
EOF_RHO
cat > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/change_rationale.md" <<'EOF_RATIONALE'
# change rationale

The instability runs showed E[rho^2] < 1 because the executed actions were clipped but the log_prob/score assumed an unbounded Gaussian. That mismatch makes rho a ratio of *different* densities (pre-clip vs post-clip), which biases rho downward and can even violate E_mu[rho^2] >= 1.

The fix makes the executed action distribution match the density used in log_prob/score by:
- sampling u ~ Normal(mean, sigma) and executing a_exec = v_max * tanh(u),
- computing log_prob(a_exec) with the exact change-of-variables (Gaussian log-prob on u minus the Jacobian term), and
- using the same a_exec everywhere (env.step, rho, and score).

With this change, rho is computed on the correct density ratio for the actually executed actions, and the Jacobian term cancels between pi and mu. That removes the systematic underestimation that previously produced E[rho^2] < 1.
EOF_RATIONALE
git diff > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
# NOTE: initial repo_state capture failed because `python` was not found.
{
  echo "## git_branch";
  git rev-parse --abbrev-ref HEAD;
  echo "## git_commit";
  git rev-parse HEAD;
  echo "## git_status";
  git status -sb;
  echo "## python_version";
  python --version;
  echo "## pip_freeze_head_60";
  python -m pip freeze | head -n 60;
} > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/repo_state.txt"
{
  echo "## git_branch";
  git rev-parse --abbrev-ref HEAD;
  echo "## git_commit";
  git rev-parse HEAD;
  echo "## git_status";
  git status -sb;
  echo "## python_version";
  python3 --version;
  echo "## pip_freeze_head_60";
  python3 -m pip freeze | head -n 60;
} > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/repo_state.txt"
python3 scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau/stdout.log"
rm -rf "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" && mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau"
rg -n "numpy" -S requirements* pyproject.toml setup.cfg setup.py
ls -a
ls ".venv/bin"
tail -n 5 "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau/stdout.log"
ls -lh "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau"
wc -l "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau/learning_curves.csv"
sed -n '1,120p' scripts/run_train.py
.venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --outer-iters 80 --report-every 20 --resume | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau/stdout.log"
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/stdout.log"
wc -l "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/learning_curves.csv"
.venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" --outer-iters 80 --report-every 20 --resume | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --mode plateau --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau/check_run_report.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" --mode instability --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/check_run_report.log"
.venv/bin/python scripts/analyze_step_c.py --plateau-run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --instability-run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" --out-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/analyze_step_c.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_plateau.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_instability.log"
.venv/bin/python scripts/rho_audit_summary.py --plateau-run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --instability-run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" --out "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_summary.md"
.venv/bin/python scripts/summarize_updates.py --plateau-run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau" --instability-run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability" --out "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/updates_scale_summary.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/check_run_report.log"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/run_report.json"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_summary.md"
.venv/bin/python - <<'PY'
import csv
import math
from pathlib import Path

path = Path('outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/rho_audit.csv')
rows = []
with path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def col_min(col):
    vals = []
    for row in rows:
        try:
            v = float(row.get(col, 'nan'))
        except ValueError:
            v = math.nan
        if math.isfinite(v):
            vals.append(v)
    return min(vals) if vals else math.nan

print('mean_rho2_exec_min', col_min('mean_rho2_exec'))
print('mean_rho2_raw_min', col_min('mean_rho2_raw'))
print('mean_rho2_min', col_min('mean_rho2'))
PY
.venv/bin/python - <<'PY'
import csv
from pathlib import Path

path = Path('outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/learning_curves.csv')
rows = list(csv.DictReader(path.open()))
last = rows[-1]
print('delta_theta_pi_norm_last', last.get('delta_theta_pi_norm'))
PY
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability/run_report.json')
report = json.loads(path.read_text())
print('health_summary_status', report.get('health_summary', {}).get('status'))
PY
cat > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/gate_check.md" <<'EOF_GATE'
# gate check

status: PASS

- condition1 (instability mean_rho2_exec min >= 1): 2.5106 -> PASS
- condition2 (raw vs exec no order-of-magnitude diff): mean_rho2_raw_min=2.5106, mean_rho2_exec_min=2.5106 (ratio=1.00), last ratio=1.00 -> PASS
- condition3 (health_summary PASS): PASS
- condition4 (delta_theta_pi_norm > 0): 3.49e-06 -> PASS
EOF_GATE
ls "meta/sweeps"
rg -n "instab_.*yaml" -S
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps" && cp "outputs/short_runs/20251229_000304/stepC_v3_rho_audit/meta/sweeps/"instab_*.yaml "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/"
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps"
for f in outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/*.yaml; do echo "--- $f"; sed -n '1,200p' "$f"; done
.venv/bin/python scripts/run_train.py --config "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/instab_alpha_boost.yaml" --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_alpha_boost" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_alpha_boost/stdout.log"
ls "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_alpha_boost"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_alpha_boost" --mode instability --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_alpha_boost/check_run_report.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_alpha_boost" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_instab_alpha_boost.log"
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch"
.venv/bin/python scripts/run_train.py --config "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/instab_sigma_mismatch.yaml" --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch" --mode instability --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch/check_run_report.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_instab_sigma_mismatch.log"
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12"
.venv/bin/python scripts/run_train.py --config "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/instab_theta_radius12.yaml" --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12" --mode instability --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12/check_run_report.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_instab_theta_radius12.log"
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0"
.venv/bin/python scripts/run_train.py --config "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/instab_pmix0.yaml" --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0" --mode instability --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0/check_run_report.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_instab_pmix0.log"
mkdir -p "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on"
.venv/bin/python scripts/run_train.py --config "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/sweeps/instab_rho_clip_on.yaml" --output-dir "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on" --outer-iters 80 --report-every 20 | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on/stdout.log"
.venv/bin/python scripts/check_run_report.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on" --mode instability --print-commands | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on/check_run_report.log"
.venv/bin/python scripts/rho_audit.py --run "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on" | tee "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_instab_rho_clip_on.log"
.venv/bin/python - <<'PY'
import csv
import json
import math
from pathlib import Path

run_root = Path('outputs/short_runs/20251229_014833/stepC_v4_rho_fix')
sweeps = [
    'instab_alpha_boost',
    'instab_sigma_mismatch',
    'instab_theta_radius12',
    'instab_pmix0',
    'instab_rho_clip_on',
]

def _read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _last_val(rows, key):
    for row in reversed(rows):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except ValueError:
            continue
    return math.nan


def _min_val(rows, key):
    vals = []
    for row in rows:
        try:
            v = float(row.get(key, 'nan'))
        except ValueError:
            v = math.nan
        if math.isfinite(v):
            vals.append(v)
    return min(vals) if vals else math.nan


def _max_val(rows, key):
    vals = []
    for row in rows:
        try:
            v = float(row.get(key, 'nan'))
        except ValueError:
            v = math.nan
        if math.isfinite(v):
            vals.append(v)
    return max(vals) if vals else math.nan


def _fmt(val):
    if not isinstance(val, (int, float)) or not math.isfinite(val):
        return '-'
    if abs(val) >= 1e3 or abs(val) < 1e-3:
        return f"{val:.3g}"
    return f"{val:.4g}"

rows_out = []
for name in sweeps:
    run_dir = run_root / 'sweeps' / name
    curves = _read_csv_rows(run_dir / 'learning_curves.csv')
    rho_rows = _read_csv_rows(run_dir / 'rho_audit.csv')
    report_path = run_dir / 'run_report.json'
    health = '-'
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            health = report.get('health_summary', {}).get('status', '-')
        except Exception:
            health = '-'
    rows_out.append({
        'sweep': name,
        'mean_rho2_last': _last_val(curves, 'mean_rho2'),
        'mean_rho2_min': _min_val(rho_rows, 'mean_rho2_exec') if rho_rows else _min_val(curves, 'mean_rho2'),
        'p99_rho_max': _max_val(rho_rows, 'p99_rho'),
        'max_rho_max': _max_val(rho_rows, 'max_rho'),
        'clip_fraction_max': _max_val(rho_rows, 'clip_fraction'),
        'delta_theta_pi_norm_last': _last_val(curves, 'delta_theta_pi_norm'),
        'tracking_gap_last': _last_val(curves, 'tracking_gap'),
        'health': health,
    })

out_path = run_root / 'analysis' / 'sweeps_summary.md'
lines = [
    '# sweeps summary',
    '',
    f'run_root: {run_root}',
    '',
    '| sweep | mean_rho2_last | mean_rho2_min | p99_rho_max | max_rho_max | clip_fraction_max | delta_theta_pi_norm_last | tracking_gap_last | health |',
    '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
]
for row in rows_out:
    lines.append(
        '| {sweep} | {mean_rho2_last} | {mean_rho2_min} | {p99_rho_max} | {max_rho_max} | {clip_fraction_max} | {delta_theta_pi_norm_last} | {tracking_gap_last} | {health} |'.format(
            sweep=row['sweep'],
            mean_rho2_last=_fmt(row['mean_rho2_last']),
            mean_rho2_min=_fmt(row['mean_rho2_min']),
            p99_rho_max=_fmt(row['p99_rho_max']),
            max_rho_max=_fmt(row['max_rho_max']),
            clip_fraction_max=_fmt(row['clip_fraction_max']),
            delta_theta_pi_norm_last=_fmt(row['delta_theta_pi_norm_last']),
            tracking_gap_last=_fmt(row['tracking_gap_last']),
            health=row['health'],
        )
    )

out_path.write_text('\n'.join(lines) + '\n')
print(f'Wrote {out_path}')
PY
ls "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/stepC_analysis.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/sweeps_summary.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/updates_scale_summary.md"
cat > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/diagnosis.md" <<'EOF_DIAG'
# diagnosis

## 1) plateau / instability
- plateau: not observed. `plateau_score` is false; td_loss and w_norm are flat (slopes ~0), and tracking gap stays well above threshold (w_gap_min_last_window=0.089).
- instability: not observed. `instability_candidate` is false; td_loss/w_norm do not increase, and stability_margin is negative.

## 2) rho consistency (instability run)
- E[rho^2] lower bound: mean_rho2_exec min = 2.5106 (>= 1).
- raw vs exec: mean_rho2_raw_min = 2.5106 and mean_rho2_exec_min = 2.5106 (ratio=1.00); last raw/exec both 5.697.
- tails: p99_rho_max = 8.526, max_rho_max = 323.7.
- clip diagnostics: clip_fraction_max = 0 and |a_exec-clip(a_exec)| = 0 (no residual clipping).

## 3) on-policy / weak off-policy
- tracking_gap_last = 3.818e-11 and mean_rho ~ 1, with clip_fraction=0.
- This is effectively on-policy / weak off-policy despite moderate tails (p99 ~ 8.5).

## 4) why no plateau/instability (likely order)
1) Updates are tiny: delta_theta_pi_norm_last = 3.49e-06, delta_w_norm_last = 1.39e-04.
2) mu tracks pi too closely (tracking_gap ~ 1e-11), so off-policy drive is weak.
3) rho tail is not extreme (p99 ~ 8.5) and only occasional large maxima; 80 iters may be too short to accumulate drift.
4) Environment/teacher dynamics are stable; w_norm is flat and probes cover only ~23/80 iterations.

## 5) sweep takeaways
- instab_alpha_boost / instab_theta_radius12 / instab_pmix0: mean_rho2_last ~ 3.657 and still stable (no instability candidate).
- instab_sigma_mismatch: much heavier tails (max_rho_max ~ 2.01e3) but health FAILED and still no instability.
- instab_rho_clip_on: rho tails bounded (p99/max = 5), most stable as expected.

## 6) next steps
- Long-run only if the gate remains PASS (mean_rho2_exec min >= 1, raw/exec within 2x, health PASS, delta_theta_pi_norm > 0). Otherwise adjust step sizes / mismatch first.
- Long-run template (5000 iters, report_every 50, seeds 0/1/2):
  .venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/plateau_seed0 --outer-iters 5000 --report-every 50 --seed 0
  .venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/plateau_seed1 --outer-iters 5000 --report-every 50 --seed 1
  .venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/plateau_seed2 --outer-iters 5000 --report-every 50 --seed 2
  .venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/instability_seed0 --outer-iters 5000 --report-every 50 --seed 0
  .venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/instability_seed1 --outer-iters 5000 --report-every 50 --seed 1
  .venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/instability_seed2 --outer-iters 5000 --report-every 50 --seed 2
EOF_DIAG
cat > "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/ARTIFACT_INDEX.md" <<'EOF_INDEX'
# artifact index

run_root: outputs/short_runs/20251229_014833/stepC_v4_rho_fix

## meta
- meta/commands_executed.sh
- meta/repo_state.txt
- meta/rho_fix_design.md
- meta/change_rationale.md
- meta/patch_rho_fix_tanh_squash.patch
- meta/sweeps/instab_alpha_boost.yaml
- meta/sweeps/instab_sigma_mismatch.yaml
- meta/sweeps/instab_theta_radius12.yaml
- meta/sweeps/instab_pmix0.yaml
- meta/sweeps/instab_rho_clip_on.yaml

## baseline runs
- plateau/learning_curves.csv
- plateau/check_run_report.log
- plateau/run_report.json
- plateau/run_report.md
- plateau/stdout.log
- instability/learning_curves.csv
- instability/check_run_report.log
- instability/run_report.json
- instability/run_report.md
- instability/stdout.log

## analysis
- analysis/stepC_analysis.md
- analysis/diagnosis.md
- analysis/gate_check.md
- analysis/rho_audit_summary.md
- analysis/updates_scale_summary.md
- analysis/sweeps_summary.md
- analysis/analyze_step_c.log
- analysis/rho_audit_plateau.log
- analysis/rho_audit_instability.log
- analysis/rho_audit_instab_alpha_boost.log
- analysis/rho_audit_instab_sigma_mismatch.log
- analysis/rho_audit_instab_theta_radius12.log
- analysis/rho_audit_instab_pmix0.log
- analysis/rho_audit_instab_rho_clip_on.log

## sweeps (instability)
- sweeps/instab_alpha_boost/learning_curves.csv
- sweeps/instab_alpha_boost/check_run_report.log
- sweeps/instab_alpha_boost/run_report.json
- sweeps/instab_alpha_boost/run_report.md
- sweeps/instab_sigma_mismatch/learning_curves.csv
- sweeps/instab_sigma_mismatch/check_run_report.log
- sweeps/instab_sigma_mismatch/run_report.json
- sweeps/instab_sigma_mismatch/run_report.md
- sweeps/instab_theta_radius12/learning_curves.csv
- sweeps/instab_theta_radius12/check_run_report.log
- sweeps/instab_theta_radius12/run_report.json
- sweeps/instab_theta_radius12/run_report.md
- sweeps/instab_pmix0/learning_curves.csv
- sweeps/instab_pmix0/check_run_report.log
- sweeps/instab_pmix0/run_report.json
- sweeps/instab_pmix0/run_report.md
- sweeps/instab_rho_clip_on/learning_curves.csv
- sweeps/instab_rho_clip_on/check_run_report.log
- sweeps/instab_rho_clip_on/run_report.json
- sweeps/instab_rho_clip_on/run_report.md
EOF_INDEX
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/diagnosis.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/gate_check.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/rho_audit_summary.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/updates_scale_summary.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis/stepC_analysis.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/rho_fix_design.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/change_rationale.md"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
cat "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/ARTIFACT_INDEX.md"
wc -l "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '1,200p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '201,400p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '401,600p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '601,800p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '801,1000p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '1001,1200p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '1201,1400p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '1401,1600p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '1601,1800p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '1801,2000p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '2001,2200p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '2201,2400p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '2401,2600p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '2601,2800p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '2801,3000p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '3001,3200p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '3201,3400p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '3401,3600p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '3601,3800p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
sed -n '3801,3913p' "outputs/short_runs/20251229_014833/stepC_v4_rho_fix/meta/patch_rho_fix_tanh_squash.patch"
