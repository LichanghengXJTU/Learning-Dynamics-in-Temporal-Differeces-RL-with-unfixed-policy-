# Run Report

## Run Info
- run_dir: outputs/base_check/20251229_172736/runs/_tmp_c0b
- timestamp: 2025-12-29T17:55:31
- seed: 0
- key_hparams: outer_iters=40, horizon=30, gamma=0.95, alpha_w=1.0, alpha_pi=0.0, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 0.016666666666666666
- stability_probe_step_scale: None
- stability_probe_step_scale_ratio: None (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 0.0001928 | 4.059e-05 | 0.0004595 | 0.0001816 | -3.494e-06 |
| w_norm | 1.154 | 1.154 | 1.162 | 1.154 | -0.0003231 |
| mean_rho2 | 1 | 1 | 1 | 1 | 0 |
| tracking_gap | 0 | 0 | 0 | 0 | 0 |
| critic_teacher_error | 0.0336 | 0.0336 | 0.0363 | 0.03372 | -5.884e-05 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm |
| --- | --- | --- | --- | --- | --- |
| 0 | 4.059e-05 | 0.0363 | 0 | 1 | 1.162 |
| 1 | 8.663e-05 | 0.03627 | 0 | 1 | 1.162 |
| 2 | 0.0001029 | 0.03622 | 0 | 1 | 1.161 |
| 3 | 9.883e-05 | 0.03619 | 0 | 1 | 1.161 |
| 4 | 0.0001361 | 0.03615 | 0 | 1 | 1.16 |
| 5 | 4.599e-05 | 0.03613 | 0 | 1 | 1.16 |
| 6 | 0.0003912 | 0.036 | 0 | 1 | 1.16 |
| 7 | 0.0002706 | 0.03591 | 0 | 1 | 1.16 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.0002491 | 0.03409 | 0 | 1 | 1.154 |
| 33 | 0.0002165 | 0.03402 | 0 | 1 | 1.155 |
| 34 | 0.0003466 | 0.03391 | 0 | 1 | 1.155 |
| 35 | 0.0002077 | 0.03384 | 0 | 1 | 1.155 |
| 36 | 0.000181 | 0.03378 | 0 | 1 | 1.154 |
| 37 | 0.0001508 | 0.03373 | 0 | 1 | 1.154 |
| 38 | 0.0001758 | 0.03367 | 0 | 1 | 1.154 |
| 39 | 0.0001928 | 0.0336 | 0 | 1 | 1.154 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
