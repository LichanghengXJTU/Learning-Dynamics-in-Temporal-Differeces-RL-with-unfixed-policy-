# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p4444_s0_c866ce21
- timestamp: 2025-12-30T06:49:02
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
| td_loss | 2.932e-05 | 2.115e-05 | 3.132e-05 | 2.709e-05 | 4.968e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.085e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003089 |
| tracking_gap | 0.01019 | 0.01019 | 0.07492 | 0.0104 | -0.0001042 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.43e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04304 | 0.04304 | 0.0434 | 0.04305 | -7.264e-07 |
| dist_action_tv | 0.1131 | 0.1128 | 0.1138 | 0.1131 | -1.294e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2319 | 0.1489 | 0.4607 | 0.2573 | 0.001152 |
| mmd2 | 0.005285 | 0.003438 | 0.01688 | 0.006716 | -2.073e-05 |
| mmd_sigma | 2.349 | 2.345 | 2.375 | 2.364 | -0.0009075 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001882 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.37 | 1.362 | -3.173e-05 |
| rho_mean | 1 | 0.9919 | 1.009 | 0.999 | -9.519e-05 |
| rho_min | 0.1234 | 0.0152 | 0.1234 | 0.0822 | 0.003583 |
| rho_p95 | 1.338 | 1.332 | 1.34 | 1.336 | 0.0001662 |
| rho_p99 | 1.356 | 1.354 | 1.359 | 1.356 | -7.821e-06 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.077 | -7.712e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.37 | 1.362 | -9.915e-06 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -1.987e-05 |
| rho_min | 0.04091 | 0.01898 | 0.1081 | 0.07951 | -0.003399 |
| rho_p95 | 1.337 | 1.334 | 1.34 | 1.336 | 4.077e-05 |
| rho_p99 | 1.357 | 1.355 | 1.359 | 1.356 | 4.866e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03242 | 0.02534 | 0.03579 | 0.03134 | 0.0002968 |
| w_sharp_drift | 0.006469 | 0 | 0.009373 | 0.006324 | 5.115e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.932e-05 | 2.255e-05 | 3.034e-05 | 2.78e-05 | 6.49e-08 |
| td_loss_from_Q | 1.26e-05 | 9.742e-06 | 1.69e-05 | 1.294e-05 | -5.108e-08 |
| td_loss_from_Q_abs_diff | 1.672e-05 | 9.851e-06 | 1.672e-05 | 1.486e-05 | 1.16e-07 |
| td_loss_from_Q_rel_diff | 0.5703 | 0.4126 | 0.5977 | 0.5344 | 0.002821 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.071 | 1.082 | 1.077 | 2.075e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.362 | 1.372 | 1.362 | -2.543e-05 |
| rho_mean | 1 | 0.9973 | 1.003 | 1.001 | 6.414e-05 |
| rho_min | 0.01943 | 0.00664 | 0.05264 | 0.02653 | -0.00133 |
| rho_p95 | 1.336 | 1.335 | 1.339 | 1.336 | 3.861e-05 |
| rho_p99 | 1.356 | 1.356 | 1.358 | 1.356 | -1.9e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.574e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.574e-07 |
| stability_proxy_std | 1.782e-06 | 6.788e-07 | 5.359e-06 | 2.308e-06 | -4.093e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.397e-05 | 0.01399 | 0.07492 | 1.084 | 4.53 | - | - | - |
| 1 | 2.663e-05 | 0.01399 | 0.07418 | 1.076 | 4.53 | - | - | - |
| 2 | 2.66e-05 | 0.01399 | 0.07344 | 1.077 | 4.53 | - | - | - |
| 3 | 2.64e-05 | 0.01399 | 0.0727 | 1.085 | 4.53 | - | - | - |
| 4 | 2.754e-05 | 0.01399 | 0.07198 | 1.068 | 4.53 | - | - | - |
| 5 | 2.963e-05 | 0.01399 | 0.07126 | 1.068 | 4.53 | 0.03058 | 1 | 0.006733 |
| 6 | 2.653e-05 | 0.01399 | 0.07055 | 1.072 | 4.53 | - | - | - |
| 7 | 2.524e-05 | 0.01399 | 0.06985 | 1.086 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.768e-05 | 0.01398 | 0.01093 | 1.065 | 4.53 | - | - | - |
| 193 | 2.873e-05 | 0.01398 | 0.01082 | 1.073 | 4.53 | - | - | - |
| 194 | 2.672e-05 | 0.01398 | 0.01071 | 1.08 | 4.53 | - | - | - |
| 195 | 2.658e-05 | 0.01398 | 0.01061 | 1.068 | 4.53 | 0.03477 | 1 | 0.007266 |
| 196 | 2.615e-05 | 0.01398 | 0.0105 | 1.104 | 4.53 | - | - | - |
| 197 | 2.78e-05 | 0.01398 | 0.0104 | 1.066 | 4.53 | - | - | - |
| 198 | 2.564e-05 | 0.01398 | 0.01029 | 1.068 | 4.53 | - | - | - |
| 199 | 2.932e-05 | 0.01398 | 0.01019 | 1.07 | 4.53 | 0.03242 | 1 | 0.005285 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
