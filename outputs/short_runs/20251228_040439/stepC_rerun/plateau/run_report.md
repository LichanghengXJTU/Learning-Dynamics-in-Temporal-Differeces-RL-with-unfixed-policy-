# Run Report

## Run Info
- run_dir: outputs/short_runs/20251228_040439/stepC_rerun/plateau
- timestamp: 2025-12-28T04:47:20
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.08, alpha_pi=0.06, beta=0.05, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 2.5e-05
- stability_probe_step_scale: 2.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.561e-05 | 1.358e-05 | 1.941e-05 | 1.571e-05 | -3.297e-07 |
| w_norm | 4.581 | 4.581 | 4.582 | 4.581 | -9.858e-07 |
| mean_rho2 | 1.326 | 1.319 | 1.337 | 1.326 | 0.0003483 |
| tracking_gap | 2.45e-13 | 1.928e-15 | 2.724e-13 | 2.25e-13 | 7.49e-15 |
| critic_teacher_error | 0.01393 | 0.01393 | 0.01395 | 0.01393 | -5.986e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | 5.846e-16 |
| dist_action_tv | 0.1131 | 0.1125 | 0.1134 | 0.113 | -1.27e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.3817 | 0.1518 | 0.454 | 0.2649 | 0.009205 |
| mmd2 | 0.01184 | 0.00319 | 0.01617 | 0.007117 | 0.0003704 |
| mmd_sigma | 2.371 | 2.345 | 2.377 | 2.362 | 0.001716 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.1013 | 0.08887 | 0.1183 | 0.1055 | 0.0005552 |
| w_sharp_drift | 0.02831 | 0 | 0.03744 | 0.02414 | 0.0003993 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 2.493e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 2.493e-08 |
| stability_proxy_std | 1.55e-06 | 7.096e-07 | 4.207e-06 | 1.563e-06 | 3.625e-10 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.688e-05 | 0.01395 | 1.928e-15 | 1.329 | 4.582 | - | - | - |
| 1 | 1.648e-05 | 0.01395 | 4.481e-15 | 1.326 | 4.582 | - | - | - |
| 2 | 1.879e-05 | 0.01395 | 6.773e-15 | 1.327 | 4.582 | - | - | - |
| 3 | 1.609e-05 | 0.01395 | 1.388e-14 | 1.332 | 4.582 | - | - | - |
| 4 | 1.701e-05 | 0.01395 | 1.889e-14 | 1.326 | 4.582 | - | - | - |
| 5 | 1.812e-05 | 0.01395 | 2.082e-14 | 1.325 | 4.582 | 0.1021 | 1 | 0.00594 |
| 6 | 1.61e-05 | 0.01395 | 2.366e-14 | 1.323 | 4.582 | - | - | - |
| 7 | 1.569e-05 | 0.01395 | 2.792e-14 | 1.329 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 1.665e-05 | 0.01393 | 2.214e-13 | 1.328 | 4.581 | - | - | - |
| 193 | 1.518e-05 | 0.01393 | 2.199e-13 | 1.334 | 4.581 | - | - | - |
| 194 | 1.655e-05 | 0.01393 | 2.196e-13 | 1.324 | 4.581 | - | - | - |
| 195 | 1.558e-05 | 0.01393 | 2.169e-13 | 1.325 | 4.581 | 0.1183 | 1 | 0.005448 |
| 196 | 1.789e-05 | 0.01393 | 2.167e-13 | 1.325 | 4.581 | - | - | - |
| 197 | 1.495e-05 | 0.01393 | 2.112e-13 | 1.331 | 4.581 | - | - | - |
| 198 | 1.453e-05 | 0.01393 | 2.354e-13 | 1.325 | 4.581 | - | - | - |
| 199 | 1.561e-05 | 0.01393 | 2.45e-13 | 1.326 | 4.581 | 0.1013 | 1 | 0.01184 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
