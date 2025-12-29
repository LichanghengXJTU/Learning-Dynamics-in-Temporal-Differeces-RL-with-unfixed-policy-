# rho audit summary

plateau: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau
instability: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability

| metric | plateau | instability |
| --- | --- | --- |
| mean_rho2_last | 1.068 | 5.697 |
| mean_rho2_min | 1.058 | 2.511 |
| E[rho^2] >= 1 (min) | yes | yes |
| p99_rho_max | 1.358 | 8.526 |
| max_rho_max | 1.361 | 323.7 |
| clip_fraction_last | 0 | 0 |
| clip_fraction_max | 0 | 0 |
| mean_abs_a_diff_last | 0 | 0 |
| mean_rho2_raw_last | 1.068 | 5.697 |
| mean_rho2_exec_last | 1.068 | 5.697 |

## conclusion
- plateau E[rho^2] >= 1 (min over iters): yes
- instability E[rho^2] >= 1 (min over iters): yes
