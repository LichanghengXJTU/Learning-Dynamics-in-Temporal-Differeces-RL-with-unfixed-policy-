# rho audit

run: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/instability
rows: 80 (iters 0 -> 79)

## key metrics
- mean_rho2: last=0.4259, min=0.4191, max=0.4365
- E[rho^2] < 1 observed: yes
- rho tails (max over iters): p95=0.885, p99=0.9133, max=0.9632
- clip_fraction: last=0.2753, max=0.2978
- |a_raw-a_exec|: mean_last=0.03468, p95_last=0.2077, max_last=0.585
- mean_rho2_raw vs mean_rho2_exec (last): 3.657 vs 0.4259

## notes
- mean_rho2 is based on the clipped rho used in training.
- mean_rho2_raw/exec are unclipped rho computed on raw vs clipped actions.
