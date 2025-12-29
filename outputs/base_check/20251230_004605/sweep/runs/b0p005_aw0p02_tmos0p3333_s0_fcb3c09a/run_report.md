# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p3333_s0_fcb3c09a
- timestamp: 2025-12-30T05:28:23
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.02, alpha_pi=0.06, beta=0.005, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 2.5e-05
- stability_probe_step_scale: 2.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.778e-05 | 2.059e-05 | 3.178e-05 | 2.683e-05 | 2.094e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.064e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003124 |
| tracking_gap | 0.009563 | 0.009563 | 0.0703 | 0.009757 | -9.781e-05 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.338e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04304 | 0.04304 | 0.04337 | 0.04305 | -6.625e-07 |
| dist_action_tv | 0.1131 | 0.1128 | 0.1138 | 0.1131 | -1.273e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2315 | 0.1086 | 0.4368 | 0.268 | -4.443e-05 |
| mmd2 | 0.005272 | 0.002795 | 0.01621 | 0.00695 | -4.801e-05 |
| mmd_sigma | 2.349 | 2.338 | 2.375 | 2.365 | -0.0009707 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001816 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.369 | 1.362 | -3.942e-05 |
| rho_mean | 1 | 0.9919 | 1.009 | 0.999 | -9.253e-05 |
| rho_min | 0.1233 | 0.01532 | 0.1233 | 0.0825 | 0.003559 |
| rho_p95 | 1.338 | 1.332 | 1.34 | 1.336 | 0.0001482 |
| rho_p99 | 1.356 | 1.354 | 1.359 | 1.356 | 7.89e-06 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.077 | -6.314e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.37 | 1.362 | -1.814e-05 |
| rho_mean | 1.001 | 0.9912 | 1.008 | 1.001 | -1.304e-05 |
| rho_min | 0.04101 | 0.01917 | 0.1078 | 0.0795 | -0.003397 |
| rho_p95 | 1.337 | 1.334 | 1.34 | 1.336 | 2.221e-05 |
| rho_p99 | 1.356 | 1.354 | 1.359 | 1.356 | 3.093e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03194 | 0.02534 | 0.03501 | 0.03118 | 0.0002735 |
| w_sharp_drift | 0.006675 | 0 | 0.009349 | 0.006454 | 5.59e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.778e-05 | 2.191e-05 | 3.034e-05 | 2.754e-05 | -3.844e-08 |
| td_loss_from_Q | 1.254e-05 | 9.742e-06 | 1.652e-05 | 1.295e-05 | -5.51e-08 |
| td_loss_from_Q_abs_diff | 1.524e-05 | 9.85e-06 | 1.59e-05 | 1.459e-05 | 1.666e-08 |
| td_loss_from_Q_rel_diff | 0.5486 | 0.4137 | 0.5977 | 0.5301 | 0.001356 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 2.968e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.362 | 1.371 | 1.362 | -2.706e-05 |
| rho_mean | 1 | 0.9972 | 1.003 | 1.001 | 6.877e-05 |
| rho_min | 0.01953 | 0.006724 | 0.05261 | 0.02693 | -0.001418 |
| rho_p95 | 1.336 | 1.335 | 1.339 | 1.336 | 2.599e-05 |
| rho_p99 | 1.356 | 1.356 | 1.358 | 1.356 | -9.442e-06 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.243e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.243e-07 |
| stability_proxy_std | 1.899e-06 | 6.454e-07 | 5.778e-06 | 2.256e-06 | -1.928e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.711e-05 | 0.01399 | 0.0703 | 1.083 | 4.53 | - | - | - |
| 1 | 2.65e-05 | 0.01399 | 0.0696 | 1.076 | 4.53 | - | - | - |
| 2 | 2.661e-05 | 0.01399 | 0.06891 | 1.077 | 4.53 | - | - | - |
| 3 | 2.614e-05 | 0.01399 | 0.06822 | 1.084 | 4.53 | - | - | - |
| 4 | 2.788e-05 | 0.01399 | 0.06754 | 1.068 | 4.53 | - | - | - |
| 5 | 2.897e-05 | 0.01399 | 0.06687 | 1.068 | 4.53 | 0.03214 | 1 | 0.008649 |
| 6 | 2.606e-05 | 0.01399 | 0.0662 | 1.073 | 4.53 | - | - | - |
| 7 | 2.466e-05 | 0.01399 | 0.06554 | 1.086 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.713e-05 | 0.01398 | 0.01026 | 1.065 | 4.53 | - | - | - |
| 193 | 2.872e-05 | 0.01398 | 0.01016 | 1.073 | 4.53 | - | - | - |
| 194 | 2.672e-05 | 0.01398 | 0.01005 | 1.08 | 4.53 | - | - | - |
| 195 | 2.658e-05 | 0.01398 | 0.009954 | 1.068 | 4.53 | 0.03477 | 1 | 0.007252 |
| 196 | 2.615e-05 | 0.01398 | 0.009855 | 1.104 | 4.53 | - | - | - |
| 197 | 2.779e-05 | 0.01398 | 0.009756 | 1.066 | 4.53 | - | - | - |
| 198 | 2.585e-05 | 0.01398 | 0.009659 | 1.068 | 4.53 | - | - | - |
| 199 | 2.778e-05 | 0.01398 | 0.009563 | 1.07 | 4.53 | 0.03194 | 1 | 0.005272 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
