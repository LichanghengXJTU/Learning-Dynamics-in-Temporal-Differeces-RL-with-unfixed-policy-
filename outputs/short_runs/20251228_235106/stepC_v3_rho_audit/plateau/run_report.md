# Run Report

## Run Info
- run_dir: outputs/short_runs/20251228_235106/stepC_v3_rho_audit/plateau
- timestamp: 2025-12-29T00:00:21
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.08, alpha_pi=0.06, beta=0.05, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: FAIL (incomplete run)

## Scale Checks
- train_step_scale: 2.5e-05
- stability_probe_step_scale: 2.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.632e-05 | 1.443e-05 | 1.941e-05 | 1.725e-05 | -6.167e-07 |
| w_norm | 4.582 | 4.582 | 4.582 | 4.582 | -1.095e-06 |
| mean_rho2 | 1.328 | 1.321 | 1.335 | 1.33 | 0.0009058 |
| tracking_gap | 1.76e-13 | 1.928e-15 | 1.932e-13 | 1.8e-13 | 1.04e-16 |
| critic_teacher_error | 0.01394 | 0.01394 | 0.01395 | 0.01394 | -6.642e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | 1.774e-15 |
| dist_action_tv | 0.1128 | 0.1125 | 0.1134 | 0.1128 | 1.794e-05 |
| iter | 39 | 5 | 39 | 31.6 | 1 |
| mean_l2 | 0.2113 | 0.2014 | 0.3064 | 0.245 | -0.005361 |
| mmd2 | 0.004852 | 0.004334 | 0.009515 | 0.006306 | -0.0002491 |
| mmd_sigma | 2.363 | 2.349 | 2.377 | 2.359 | -0.0004262 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.337 | 1.323 | 1.337 | 1.327 | 0.0008184 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | -8.283e-06 |
| rho_mean | 1.152 | 1.146 | 1.152 | 1.148 | 0.000337 |
| rho_min | 1.057 | 1.056 | 1.06 | 1.057 | -1.938e-06 |
| rho_p95 | 1.337 | 1.333 | 1.338 | 1.335 | 0.0002773 |
| rho_p99 | 1.357 | 1.354 | 1.357 | 1.356 | 5.766e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 39 | 5 | 39 | 31.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.326 | 1.322 | 1.333 | 1.327 | 7.863e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 3.575e-07 |
| rho_mean | 1.148 | 1.146 | 1.151 | 1.148 | 2.934e-05 |
| rho_min | 1.058 | 1.056 | 1.06 | 1.058 | 7.895e-05 |
| rho_p95 | 1.337 | 1.333 | 1.338 | 1.337 | 0.0001657 |
| rho_p99 | 1.355 | 1.355 | 1.357 | 1.357 | -4.38e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.1024 | 0.09419 | 0.11 | 0.1024 | -0.0005717 |
| w_sharp_drift | 0.0226 | 0 | 0.02593 | 0.02341 | -8.092e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 39 | 5 | 39 | 31.6 | 1 |
| td_loss | 1.632e-05 | 1.443e-05 | 1.816e-05 | 1.64e-05 | 7.918e-08 |
| td_loss_from_Q | 8.166e-06 | 7.531e-06 | 9.737e-06 | 8.385e-06 | 2.785e-08 |
| td_loss_from_Q_abs_diff | 8.159e-06 | 6.568e-06 | 1.003e-05 | 8.013e-06 | 5.133e-08 |
| td_loss_from_Q_rel_diff | 0.4998 | 0.4386 | 0.5536 | 0.4891 | 0.0009447 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 39 | 5 | 39 | 31.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.33 | 1.326 | 1.331 | 1.328 | 0.0002156 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 2.791e-06 |
| rho_mean | 1.149 | 1.147 | 1.15 | 1.149 | 9.181e-05 |
| rho_min | 1.057 | 1.054 | 1.057 | 1.056 | 3.625e-05 |
| rho_p95 | 1.336 | 1.335 | 1.338 | 1.336 | 4.986e-05 |
| rho_p99 | 1.357 | 1.356 | 1.357 | 1.356 | 5.169e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -1.708e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | -1.708e-08 |
| stability_proxy_std | 1.439e-06 | 9.617e-07 | 2.73e-06 | 1.665e-06 | -7.257e-09 |

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
| 32 | 1.68e-05 | 0.01394 | 1.798e-13 | 1.322 | 4.582 | - | - | - |
| 33 | 1.519e-05 | 0.01394 | 1.932e-13 | 1.326 | 4.582 | - | - | - |
| 34 | 1.497e-05 | 0.01394 | 1.892e-13 | 1.33 | 4.582 | - | - | - |
| 35 | 1.766e-05 | 0.01394 | 1.779e-13 | 1.33 | 4.582 | 0.09443 | 1 | 0.004334 |
| 36 | 1.941e-05 | 0.01394 | 1.777e-13 | 1.323 | 4.582 | - | - | - |
| 37 | 1.695e-05 | 0.01394 | 1.857e-13 | 1.332 | 4.582 | - | - | - |
| 38 | 1.591e-05 | 0.01394 | 1.825e-13 | 1.335 | 4.582 | - | - | - |
| 39 | 1.632e-05 | 0.01394 | 1.76e-13 | 1.328 | 4.582 | 0.1024 | 1 | 0.004852 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
