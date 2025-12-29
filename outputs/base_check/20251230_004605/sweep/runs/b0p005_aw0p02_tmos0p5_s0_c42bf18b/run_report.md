# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p5_s0_c42bf18b
- timestamp: 2025-12-30T06:55:46
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
| td_loss | 2.419e-05 | 2.298e-05 | 3.133e-05 | 2.507e-05 | -6.511e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -3.866e-06 |
| mean_rho2 | 1.076 | 1.058 | 1.097 | 1.07 | 0.001498 |
| tracking_gap | 0.05441 | 0.05441 | 0.07651 | 0.05552 | -0.0005565 |
| critic_teacher_error | 0.01399 | 0.01399 | 0.01399 | 0.01399 | -7.67e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04332 | 0.0433 | 0.04343 | 0.04332 | -2.762e-06 |
| dist_action_tv | 0.1133 | 0.1129 | 0.1139 | 0.1134 | -4.162e-05 |
| iter | 30 | 5 | 30 | 24.6 | 1 |
| mean_l2 | 0.2501 | 0.1724 | 0.3919 | 0.2339 | -0.003373 |
| mmd2 | 0.007071 | 0.004643 | 0.0132 | 0.006265 | -5.243e-05 |
| mmd_sigma | 2.354 | 2.351 | 2.373 | 2.36 | 1.754e-05 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.079 | 1.067 | 1.084 | 1.075 | 0.0004353 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.364 | 1.364 | 1.371 | 1.366 | -8.915e-05 |
| rho_mean | 1.003 | 0.995 | 1.005 | 1 | 0.0003248 |
| rho_min | 0.05595 | 0.02918 | 0.08456 | 0.06213 | -0.002636 |
| rho_p95 | 1.335 | 1.334 | 1.34 | 1.336 | -0.0002243 |
| rho_p99 | 1.357 | 1.356 | 1.358 | 1.357 | -3.175e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 35 | 5 | 35 | 27.8 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.081 | 1.064 | 1.087 | 1.075 | 0.0007765 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.364 | 1.364 | 1.371 | 1.365 | -0.0001785 |
| rho_mean | 1.003 | 0.993 | 1.006 | 0.9997 | 0.0004334 |
| rho_min | 0.06968 | 0.02068 | 0.06968 | 0.04147 | 0.003267 |
| rho_p95 | 1.339 | 1.335 | 1.339 | 1.337 | 0.0002508 |
| rho_p99 | 1.357 | 1.357 | 1.36 | 1.358 | 6.147e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.02664 | 0.02596 | 0.03114 | 0.02882 | -0.0002453 |
| w_sharp_drift | 0.006947 | 0 | 0.007255 | 0.006438 | 0.0001285 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 30 | 5 | 30 | 24.6 | 1 |
| td_loss | 2.63e-05 | 2.298e-05 | 2.962e-05 | 2.526e-05 | -1.949e-07 |
| td_loss_from_Q | 1.545e-05 | 1.004e-05 | 1.545e-05 | 1.269e-05 | -6.976e-08 |
| td_loss_from_Q_abs_diff | 1.085e-05 | 1.085e-05 | 1.645e-05 | 1.256e-05 | -1.251e-07 |
| td_loss_from_Q_rel_diff | 0.4126 | 0.4126 | 0.5633 | 0.5007 | -0.0008956 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 30 | 5 | 30 | 24.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.071 | 1.071 | 1.082 | 1.074 | -0.0001383 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.369 | 1.366 | 1.372 | 1.369 | -0.0002917 |
| rho_mean | 0.9972 | 0.9972 | 1.003 | 0.9989 | -4.991e-05 |
| rho_min | 0.03073 | 0.009418 | 0.03986 | 0.0296 | 6.643e-05 |
| rho_p95 | 1.336 | 1.336 | 1.339 | 1.337 | -0.0001281 |
| rho_p99 | 1.357 | 1.357 | 1.358 | 1.358 | -5.02e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.072e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.072e-07 |
| stability_proxy_std | 2.688e-06 | 6.894e-07 | 3.868e-06 | 2.341e-06 | -1.323e-07 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.496e-05 | 0.01399 | 0.07651 | 1.084 | 4.53 | - | - | - |
| 1 | 2.72e-05 | 0.01399 | 0.07574 | 1.077 | 4.53 | - | - | - |
| 2 | 2.66e-05 | 0.01399 | 0.07499 | 1.077 | 4.53 | - | - | - |
| 3 | 2.634e-05 | 0.01399 | 0.07424 | 1.085 | 4.53 | - | - | - |
| 4 | 2.754e-05 | 0.01399 | 0.0735 | 1.068 | 4.53 | - | - | - |
| 5 | 2.962e-05 | 0.01399 | 0.07277 | 1.068 | 4.53 | 0.02995 | 1 | 0.007632 |
| 6 | 2.658e-05 | 0.01399 | 0.07204 | 1.073 | 4.53 | - | - | - |
| 7 | 2.524e-05 | 0.01399 | 0.07132 | 1.086 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 2.433e-05 | 0.01399 | 0.05837 | 1.083 | 4.53 | - | - | - |
| 28 | 2.573e-05 | 0.01399 | 0.05778 | 1.072 | 4.53 | - | - | - |
| 29 | 2.346e-05 | 0.01399 | 0.05721 | 1.078 | 4.53 | 0.02596 | 1 | 0.005521 |
| 30 | 2.63e-05 | 0.01399 | 0.05664 | 1.068 | 4.53 | 0.03041 | 1 | 0.007071 |
| 31 | 2.581e-05 | 0.01399 | 0.05607 | 1.073 | 4.53 | - | - | - |
| 32 | 2.552e-05 | 0.01399 | 0.05551 | 1.058 | 4.53 | - | - | - |
| 33 | 2.352e-05 | 0.01399 | 0.05496 | 1.074 | 4.53 | - | - | - |
| 34 | 2.419e-05 | 0.01399 | 0.05441 | 1.076 | 4.53 | - | - | - |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
