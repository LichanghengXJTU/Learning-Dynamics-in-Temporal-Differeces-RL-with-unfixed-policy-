# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p2778_s0_4bd77e92
- timestamp: 2025-12-30T04:48:05
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
| td_loss | 2.777e-05 | 2.058e-05 | 3.133e-05 | 2.682e-05 | 2.186e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.063e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003134 |
| tracking_gap | 0.009082 | 0.009082 | 0.06677 | 0.009267 | -9.29e-05 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.338e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04304 | 0.04304 | 0.04336 | 0.04305 | -6.2e-07 |
| dist_action_tv | 0.1131 | 0.1127 | 0.1138 | 0.1131 | -1.261e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2322 | 0.1515 | 0.4119 | 0.2661 | 0.0002134 |
| mmd2 | 0.005297 | 0.003203 | 0.01403 | 0.006874 | -3.771e-05 |
| mmd_sigma | 2.349 | 2.345 | 2.379 | 2.366 | -0.001039 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001756 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.366 | 1.361 | -2.608e-05 |
| rho_mean | 1 | 0.992 | 1.009 | 0.999 | -8.975e-05 |
| rho_min | 0.1177 | 0.01591 | 0.1177 | 0.08157 | 0.00321 |
| rho_p95 | 1.337 | 1.331 | 1.34 | 1.336 | 0.0001155 |
| rho_p99 | 1.356 | 1.354 | 1.358 | 1.356 | 1.986e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.077 | -9.057e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.369 | 1.362 | -1.444e-05 |
| rho_mean | 1.001 | 0.9913 | 1.008 | 1.001 | -3.037e-05 |
| rho_min | 0.04109 | 0.01714 | 0.1077 | 0.0795 | -0.003397 |
| rho_p95 | 1.337 | 1.334 | 1.34 | 1.336 | 1.41e-05 |
| rho_p99 | 1.357 | 1.355 | 1.359 | 1.356 | 5.009e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03195 | 0.02536 | 0.03478 | 0.03097 | 0.0002814 |
| w_sharp_drift | 0.00668 | 0 | 0.009573 | 0.006722 | 3.072e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.777e-05 | 2.191e-05 | 3.045e-05 | 2.749e-05 | -3.538e-08 |
| td_loss_from_Q | 1.254e-05 | 9.743e-06 | 1.656e-05 | 1.291e-05 | -5.04e-08 |
| td_loss_from_Q_abs_diff | 1.524e-05 | 9.805e-06 | 1.59e-05 | 1.458e-05 | 1.502e-08 |
| td_loss_from_Q_rel_diff | 0.5486 | 0.4142 | 0.5984 | 0.5308 | 0.001237 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 3.674e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.362 | 1.371 | 1.362 | -2.137e-05 |
| rho_mean | 1 | 0.9971 | 1.003 | 1.001 | 7.254e-05 |
| rho_min | 0.01961 | 0.006789 | 0.05258 | 0.02701 | -0.001422 |
| rho_p95 | 1.336 | 1.335 | 1.338 | 1.336 | 1.639e-05 |
| rho_p99 | 1.356 | 1.356 | 1.358 | 1.356 | -6.858e-06 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.189e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.189e-07 |
| stability_proxy_std | 1.899e-06 | 7.295e-07 | 5.637e-06 | 2.284e-06 | -2.589e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.734e-05 | 0.01399 | 0.06677 | 1.083 | 4.53 | - | - | - |
| 1 | 2.656e-05 | 0.01399 | 0.06611 | 1.076 | 4.53 | - | - | - |
| 2 | 2.664e-05 | 0.01399 | 0.06545 | 1.077 | 4.53 | - | - | - |
| 3 | 2.624e-05 | 0.01399 | 0.06479 | 1.084 | 4.53 | - | - | - |
| 4 | 2.737e-05 | 0.01399 | 0.06415 | 1.069 | 4.53 | - | - | - |
| 5 | 2.897e-05 | 0.01399 | 0.06351 | 1.068 | 4.53 | 0.03215 | 1 | 0.008647 |
| 6 | 2.598e-05 | 0.01399 | 0.06288 | 1.073 | 4.53 | - | - | - |
| 7 | 2.512e-05 | 0.01399 | 0.06225 | 1.086 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.713e-05 | 0.01398 | 0.009742 | 1.065 | 4.53 | - | - | - |
| 193 | 2.902e-05 | 0.01398 | 0.009645 | 1.073 | 4.53 | - | - | - |
| 194 | 2.7e-05 | 0.01398 | 0.009549 | 1.08 | 4.53 | - | - | - |
| 195 | 2.653e-05 | 0.01398 | 0.009454 | 1.068 | 4.53 | 0.03478 | 1 | 0.007251 |
| 196 | 2.615e-05 | 0.01398 | 0.00936 | 1.104 | 4.53 | - | - | - |
| 197 | 2.779e-05 | 0.01398 | 0.009266 | 1.066 | 4.53 | - | - | - |
| 198 | 2.585e-05 | 0.01398 | 0.009174 | 1.068 | 4.53 | - | - | - |
| 199 | 2.777e-05 | 0.01398 | 0.009082 | 1.07 | 4.53 | 0.03195 | 1 | 0.005297 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
