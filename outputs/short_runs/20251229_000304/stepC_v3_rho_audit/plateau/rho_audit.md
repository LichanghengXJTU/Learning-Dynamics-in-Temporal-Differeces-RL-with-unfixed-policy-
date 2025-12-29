# rho audit

run: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/plateau
rows: 80 (iters 0 -> 79)

## key metrics
- mean_rho2: last=1.331, min=1.319, max=1.337
- E[rho^2] < 1 observed: no
- rho tails (max over iters): p95=1.341, p99=1.358, max=1.361
- clip_fraction: last=0.5272, max=0.5403
- |a_raw-a_exec|: mean_last=0.1112, p95_last=0.4532, max_last=0.9764
- mean_rho2_raw vs mean_rho2_exec (last): 1.078 vs 1.331

## notes
- mean_rho2 is based on the clipped rho used in training.
- mean_rho2_raw/exec are unclipped rho computed on raw vs clipped actions.
