# rho audit

run: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12
rows: 80 (iters 0 -> 79)

## key metrics
- mean_rho2: last=3.657, min=1.862, max=42.71
- E[rho^2] < 1 observed: no
- rho tails (max over iters): p95=2.675, p99=8.526, max=323.7
- clip_fraction: last=0, max=0
- |a_exec-clip(a_exec)|: mean_last=0, p95_last=0, max_last=0
- mean_rho2_raw vs mean_rho2_exec (last): 3.657 vs 3.657

## notes
- mean_rho2 is based on the clipped rho used in training.
- mean_rho2_raw/exec are unclipped rho from pre-squash vs exec actions; they should match when Jacobians cancel.
