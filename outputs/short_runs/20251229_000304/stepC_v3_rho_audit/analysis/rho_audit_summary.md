# rho audit summary

plateau: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/plateau
instability: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/instability

| metric | plateau | instability |
| --- | --- | --- |
| mean_rho2_last | 1.331 | 0.4259 |
| mean_rho2_min | 1.319 | 0.4191 |
| E[rho^2] >= 1 (min) | yes | no |
| p99_rho_max | 1.358 | 0.9133 |
| max_rho_max | 1.361 | 0.9632 |
| clip_fraction_last | 0.5272 | 0.2753 |
| clip_fraction_max | 0.5403 | 0.2978 |
| mean_abs_a_diff_last | 0.1112 | 0.03468 |
| mean_rho2_raw_last | 1.078 | 3.657 |
| mean_rho2_exec_last | 1.331 | 0.4259 |

## conclusion
- plateau E[rho^2] >= 1 (min over iters): yes
- instability E[rho^2] >= 1 (min over iters): no
