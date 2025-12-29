# Run Report

## Run Info
- run_dir: outputs/base_check/20251229_172736/runs/_tmp_c0c
- timestamp: 2025-12-29T17:56:04
- seed: 0
- key_hparams: outer_iters=80, horizon=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.0, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, p_mix=0.0

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 0.0033333333333333335
- stability_probe_step_scale: None
- stability_probe_step_scale_ratio: None (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 0.0002854 | 2.385e-05 | 0.0005401 | 0.0003273 | 2.415e-05 |
| w_norm | 1.158 | 1.158 | 1.162 | 1.158 | -2.895e-05 |
| mean_rho2 | 1 | 1 | 1 | 1 | 0 |
| tracking_gap | 0 | 0 | 0 | 0 | 0 |
| critic_teacher_error | 0.03484 | 0.03484 | 0.03628 | 0.03489 | -2.44e-05 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0004524 | 0.03628 | 0 | 1 | 1.162 |
| 1 | 0.0002021 | 0.03626 | 0 | 1 | 1.162 |
| 2 | 8.917e-05 | 0.03626 | 0 | 1 | 1.162 |
| 3 | 0.0001613 | 0.03625 | 0 | 1 | 1.162 |
| 4 | 0.0001811 | 0.03623 | 0 | 1 | 1.162 |
| 5 | 0.0001199 | 0.03622 | 0 | 1 | 1.162 |
| 6 | 0.0001738 | 0.03621 | 0 | 1 | 1.161 |
| 7 | 0.0001739 | 0.0362 | 0 | 1 | 1.161 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm |
| --- | --- | --- | --- | --- | --- |
| 72 | 0.0002206 | 0.03499 | 0 | 1 | 1.158 |
| 73 | 0.0004226 | 0.03496 | 0 | 1 | 1.158 |
| 74 | 0.0001081 | 0.03496 | 0 | 1 | 1.158 |
| 75 | 0.0002352 | 0.03494 | 0 | 1 | 1.158 |
| 76 | 0.0003576 | 0.03491 | 0 | 1 | 1.158 |
| 77 | 0.0002598 | 0.0349 | 0 | 1 | 1.158 |
| 78 | 0.0004987 | 0.03486 | 0 | 1 | 1.158 |
| 79 | 0.0002854 | 0.03484 | 0 | 1 | 1.158 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
