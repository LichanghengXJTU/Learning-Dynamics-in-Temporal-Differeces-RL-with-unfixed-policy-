# Change rationale

## rho audit + update norms logging
- Added per-iteration logging for action clipping (`clip_fraction`, `mean_abs_a_diff`, `p95_abs_a_diff`, `max_abs_a_diff`) and raw vs executed rho (`mean_rho2_raw`, `mean_rho2_exec`) to audit whether rho corresponds to the intended density ratio and whether action clipping distorts log-prob calculations.
- Added update norms (`delta_theta_pi_norm`, `delta_w_norm`) to quantify policy nonstationarity and critic update scale without changing the training update logic.
- Implemented `scripts/rho_audit.py` and summary helpers to aggregate these diagnostics into `rho_audit.csv/.md` and cross-run summaries.

## health check probe NaN fix
- Updated health checks to only enforce finiteness on core metrics; probe columns are allowed to be NaN when probes do not run.
- Added coverage-aware logic for `fixed_point_drift` using `fixed_point_drift_defined` to avoid false FAILs when drift is undefined.
- Included q-kernel probe columns in `PROBE_COLUMNS` to keep optional probe NaNs out of the core check.

These changes add auditing/recording only and do not alter the training algorithm or update equations.
