# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p2222_s0_d86bfa49
- timestamp: 2025-12-30T04:07:48
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
| td_loss | 2.844e-05 | 2.103e-05 | 3.109e-05 | 2.689e-05 | 3.49e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.046e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003147 |
| tracking_gap | 0.008405 | 0.008405 | 0.06179 | 0.008576 | -8.597e-05 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.343e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04304 | 0.04304 | 0.04333 | 0.04305 | -4.161e-07 |
| dist_action_tv | 0.1131 | 0.1127 | 0.1138 | 0.1131 | -1.297e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.3186 | 0.1496 | 0.4704 | 0.2557 | 0.007443 |
| mmd2 | 0.008631 | 0.003228 | 0.01573 | 0.006472 | 0.0002412 |
| mmd_sigma | 2.355 | 2.345 | 2.379 | 2.367 | -0.0006537 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001668 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.366 | 1.361 | -2.963e-05 |
| rho_mean | 1 | 0.992 | 1.009 | 0.999 | -8.677e-05 |
| rho_min | 0.1176 | 0.01613 | 0.1176 | 0.08213 | 0.003298 |
| rho_p95 | 1.337 | 1.332 | 1.34 | 1.336 | 6.087e-05 |
| rho_p99 | 1.356 | 1.354 | 1.358 | 1.356 | 9.04e-06 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.078 | -7.937e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.368 | 1.362 | -4.169e-06 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -2.54e-05 |
| rho_min | 0.0412 | 0.02089 | 0.1075 | 0.0795 | -0.003399 |
| rho_p95 | 1.337 | 1.334 | 1.34 | 1.336 | 1.385e-05 |
| rho_p99 | 1.356 | 1.355 | 1.359 | 1.356 | 3.548e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03276 | 0.02404 | 0.03521 | 0.03104 | 0.0003388 |
| w_sharp_drift | 0.006416 | 0 | 0.0104 | 0.006701 | 3.004e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.844e-05 | 2.239e-05 | 3.044e-05 | 2.768e-05 | -1.47e-09 |
| td_loss_from_Q | 1.324e-05 | 9.747e-06 | 1.656e-05 | 1.304e-05 | -3.037e-09 |
| td_loss_from_Q_abs_diff | 1.519e-05 | 9.811e-06 | 1.573e-05 | 1.465e-05 | 1.567e-09 |
| td_loss_from_Q_rel_diff | 0.5343 | 0.417 | 0.6001 | 0.5293 | 9.922e-05 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.071 | 1.082 | 1.077 | 2.749e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.369 | 1.362 | -2.878e-05 |
| rho_mean | 1 | 0.9971 | 1.003 | 1.001 | 6.728e-05 |
| rho_min | 0.01844 | 0.007089 | 0.05255 | 0.02666 | -0.00145 |
| rho_p95 | 1.335 | 1.335 | 1.338 | 1.336 | -5.522e-06 |
| rho_p99 | 1.356 | 1.356 | 1.358 | 1.356 | -7.137e-06 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.244e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.244e-07 |
| stability_proxy_std | 2.032e-06 | 8.203e-07 | 5.697e-06 | 2.325e-06 | -1.508e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.569e-05 | 0.01399 | 0.06179 | 1.082 | 4.53 | - | - | - |
| 1 | 2.567e-05 | 0.01399 | 0.06118 | 1.077 | 4.53 | - | - | - |
| 2 | 2.688e-05 | 0.01399 | 0.06057 | 1.077 | 4.53 | - | - | - |
| 3 | 2.64e-05 | 0.01399 | 0.05996 | 1.084 | 4.53 | - | - | - |
| 4 | 2.734e-05 | 0.01399 | 0.05936 | 1.068 | 4.53 | - | - | - |
| 5 | 2.895e-05 | 0.01399 | 0.05877 | 1.068 | 4.53 | 0.02964 | 1 | 0.008644 |
| 6 | 2.682e-05 | 0.01399 | 0.05819 | 1.072 | 4.53 | - | - | - |
| 7 | 2.517e-05 | 0.01399 | 0.05761 | 1.087 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.716e-05 | 0.01398 | 0.009016 | 1.065 | 4.53 | - | - | - |
| 193 | 2.902e-05 | 0.01398 | 0.008926 | 1.073 | 4.53 | - | - | - |
| 194 | 2.702e-05 | 0.01398 | 0.008837 | 1.08 | 4.53 | - | - | - |
| 195 | 2.653e-05 | 0.01398 | 0.008749 | 1.068 | 4.53 | 0.03478 | 1 | 0.007259 |
| 196 | 2.614e-05 | 0.01398 | 0.008661 | 1.104 | 4.53 | - | - | - |
| 197 | 2.754e-05 | 0.01398 | 0.008575 | 1.066 | 4.53 | - | - | - |
| 198 | 2.582e-05 | 0.01398 | 0.00849 | 1.068 | 4.53 | - | - | - |
| 199 | 2.844e-05 | 0.01398 | 0.008405 | 1.07 | 4.53 | 0.03276 | 1 | 0.008631 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
