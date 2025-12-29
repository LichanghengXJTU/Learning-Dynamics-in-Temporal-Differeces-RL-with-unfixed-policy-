# Health check policy (probe NaN handling)

## Core vs probe columns
- Core columns for `no_nan_inf` are defined as: all numeric columns in `learning_curves.csv` **excluding** `PROBE_COLUMNS` and time columns (`iter`, `step`).
- `PROBE_COLUMNS` (optional diagnostics): `fixed_point_gap`, `fixed_point_drift`, `fixed_point_drift_defined`, `stability_proxy`, `dist_mmd2`, `dist_mean_l2`, `dist_action_kl`, `dist_action_tv`, `td_loss_from_Q`, `td_loss_from_Q_abs_diff`, `td_loss_from_Q_rel_diff`.
- The summary table in `run_report.md` still uses the explicit `CORE_COLUMNS` list (`td_loss`, `w_norm`, `mean_rho2`, `tracking_gap`, `critic_teacher_error`) for display, but health checks use the rule above.

## Probe NaN/Inf handling
- `no_nan_inf` now checks **core columns only**.
- Probe columns allow NaN/Inf when probes do not run (sparse coverage).
- Coverage-aware check:
  - `fixed_point_drift` is only required to be finite when `fixed_point_drift_defined > 0` on that row.
  - Other probe columns are checked for Inf only when a value is present; NaN is treated as "missing" and allowed.

## Rationale
- Probe outputs are optional and sparse by design (e.g., `probes.every > 1` or plateau-triggered probes), so NaN placeholders should not fail health checks.
- Core metrics must remain finite to validate the run.
