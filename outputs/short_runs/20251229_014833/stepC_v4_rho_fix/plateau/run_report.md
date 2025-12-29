# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau
- timestamp: 2025-12-29T02:22:27
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.08, alpha_pi=0.06, beta=0.05, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 2.5e-05
- stability_probe_step_scale: 2.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.537e-05 | 1.367e-05 | 1.83e-05 | 1.572e-05 | -3.666e-07 |
| w_norm | 4.582 | 4.582 | 4.582 | 4.582 | -6.011e-07 |
| mean_rho2 | 1.068 | 1.058 | 1.095 | 1.075 | -0.001895 |
| tracking_gap | 3.964e-14 | 9.334e-16 | 5.444e-14 | 3.932e-14 | 2.162e-16 |
| critic_teacher_error | 0.01394 | 0.01394 | 0.01395 | 0.01394 | -5.377e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | 2.867e-16 |
| dist_action_tv | 0.1133 | 0.1125 | 0.1134 | 0.113 | 1.709e-05 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| mean_l2 | 0.473 | 0.1737 | 0.473 | 0.3326 | 0.009092 |
| mmd2 | 0.01831 | 0.004201 | 0.01831 | 0.01032 | 0.0005777 |
| mmd_sigma | 2.355 | 2.349 | 2.375 | 2.356 | -0.0006849 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.074 | 1.064 | 1.089 | 1.075 | -0.000309 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | -5.328e-07 |
| rho_mean | 0.9991 | 0.9941 | 1.006 | 0.9993 | -0.0001471 |
| rho_min | 0.06457 | 0.03642 | 0.08462 | 0.06706 | 0.001213 |
| rho_p95 | 1.336 | 1.332 | 1.338 | 1.337 | -3.374e-05 |
| rho_p99 | 1.356 | 1.354 | 1.357 | 1.356 | -4.026e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.078 | 1.062 | 1.089 | 1.077 | 0.0008202 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | -2.259e-06 |
| rho_mean | 1 | 0.9931 | 1.006 | 0.9995 | 0.0004399 |
| rho_min | 0.05178 | 0.02084 | 0.08878 | 0.04851 | 0.002186 |
| rho_p95 | 1.339 | 1.333 | 1.339 | 1.337 | 0.0002887 |
| rho_p99 | 1.356 | 1.355 | 1.357 | 1.356 | 8.087e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.09915 | 0.07518 | 0.1129 | 0.0934 | 0.0003638 |
| w_sharp_drift | 0.02058 | 0 | 0.04004 | 0.01941 | 6.101e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| td_loss | 1.537e-05 | 1.415e-05 | 1.83e-05 | 1.558e-05 | -9.001e-08 |
| td_loss_from_Q | 8.122e-06 | 6.72e-06 | 9.496e-06 | 8.071e-06 | -1.235e-07 |
| td_loss_from_Q_abs_diff | 7.249e-06 | 6.42e-06 | 9.714e-06 | 7.514e-06 | 3.344e-08 |
| td_loss_from_Q_rel_diff | 0.4716 | 0.404 | 0.5537 | 0.4825 | 0.005047 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.078 | 1.07 | 1.082 | 1.074 | 0.0001403 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 1.138e-06 |
| rho_mean | 1.001 | 0.9973 | 1.003 | 0.9992 | 9.527e-05 |
| rho_min | 0.02564 | 0.007364 | 0.04471 | 0.03456 | -0.001126 |
| rho_p95 | 1.336 | 1.334 | 1.338 | 1.336 | -0.0001374 |
| rho_p99 | 1.356 | 1.356 | 1.357 | 1.356 | -6.278e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -2.281e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | -2.281e-08 |
| stability_proxy_std | 1.807e-06 | 6.122e-07 | 1.92e-06 | 1.223e-06 | 6.615e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.547e-05 | 0.01395 | 9.334e-16 | 1.082 | 4.582 | - | - | - |
| 1 | 1.492e-05 | 0.01395 | 2.581e-15 | 1.075 | 4.582 | - | - | - |
| 2 | 1.447e-05 | 0.01395 | 3.071e-15 | 1.076 | 4.582 | - | - | - |
| 3 | 1.409e-05 | 0.01395 | 3.597e-15 | 1.084 | 4.582 | - | - | - |
| 4 | 1.725e-05 | 0.01395 | 4.291e-15 | 1.067 | 4.582 | - | - | - |
| 5 | 1.566e-05 | 0.01395 | 5.056e-15 | 1.068 | 4.582 | 0.09045 | 1 | 0.006419 |
| 6 | 1.405e-05 | 0.01395 | 7.129e-15 | 1.072 | 4.582 | - | - | - |
| 7 | 1.66e-05 | 0.01395 | 6.845e-15 | 1.086 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 1.712e-05 | 0.01394 | 3.745e-14 | 1.081 | 4.582 | - | - | - |
| 73 | 1.67e-05 | 0.01394 | 3.749e-14 | 1.069 | 4.582 | 0.09196 | 1 | 0.005411 |
| 74 | 1.487e-05 | 0.01394 | 4.067e-14 | 1.07 | 4.582 | - | - | - |
| 75 | 1.689e-05 | 0.01394 | 3.938e-14 | 1.076 | 4.582 | - | - | - |
| 76 | 1.494e-05 | 0.01394 | 3.898e-14 | 1.081 | 4.582 | - | - | - |
| 77 | 1.706e-05 | 0.01394 | 3.8e-14 | 1.073 | 4.582 | - | - | - |
| 78 | 1.432e-05 | 0.01394 | 4.062e-14 | 1.077 | 4.582 | 0.09191 | 1 | 0.009938 |
| 79 | 1.537e-05 | 0.01394 | 3.964e-14 | 1.068 | 4.582 | 0.09915 | 1 | 0.01831 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
