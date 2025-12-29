# Run Report

## Run Info
- run_dir: outputs/base_check/20251229_172736/runs/_tmp_c0
- timestamp: 2025-12-29T17:55:05
- seed: 0
- key_hparams: outer_iters=40, horizon=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.0, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 0.0033333333333333335
- stability_probe_step_scale: None
- stability_probe_step_scale_ratio: None (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 0.002999 | 0.001227 | 0.005849 | 0.003874 | -9.687e-05 |
| w_norm | 11.59 | 11.59 | 11.62 | 11.59 | -0.001073 |
| mean_rho2 | 1 | 1 | 1 | 1 | 0 |
| tracking_gap | 0 | 0 | 0 | 0 | 0 |
| critic_teacher_error | 1.072 | 1.072 | 1.078 | 1.072 | -0.0002026 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.001944 | 1.078 | 0 | 1 | 11.62 |
| 1 | 0.002585 | 1.078 | 0 | 1 | 11.62 |
| 2 | 0.001489 | 1.078 | 0 | 1 | 11.62 |
| 3 | 0.001227 | 1.078 | 0 | 1 | 11.62 |
| 4 | 0.002388 | 1.078 | 0 | 1 | 11.62 |
| 5 | 0.002268 | 1.078 | 0 | 1 | 11.62 |
| 6 | 0.002304 | 1.078 | 0 | 1 | 11.62 |
| 7 | 0.002617 | 1.077 | 0 | 1 | 11.62 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.004426 | 1.073 | 0 | 1 | 11.6 |
| 33 | 0.002307 | 1.073 | 0 | 1 | 11.6 |
| 34 | 0.003839 | 1.073 | 0 | 1 | 11.6 |
| 35 | 0.003364 | 1.073 | 0 | 1 | 11.59 |
| 36 | 0.003699 | 1.073 | 0 | 1 | 11.59 |
| 37 | 0.005849 | 1.072 | 0 | 1 | 11.59 |
| 38 | 0.00346 | 1.072 | 0 | 1 | 11.59 |
| 39 | 0.002999 | 1.072 | 0 | 1 | 11.59 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
