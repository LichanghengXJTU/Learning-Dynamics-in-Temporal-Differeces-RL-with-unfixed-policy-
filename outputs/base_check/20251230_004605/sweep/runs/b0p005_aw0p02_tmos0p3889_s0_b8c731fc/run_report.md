# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p3889_s0_b8c731fc
- timestamp: 2025-12-30T06:08:41
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
| td_loss | 2.778e-05 | 2.06e-05 | 3.143e-05 | 2.679e-05 | 1.892e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.059e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003116 |
| tracking_gap | 0.009918 | 0.009918 | 0.07292 | 0.01012 | -0.0001015 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.348e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04304 | 0.04304 | 0.04338 | 0.04305 | -6.97e-07 |
| dist_action_tv | 0.1131 | 0.1128 | 0.1138 | 0.1131 | -1.287e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2315 | 0.1081 | 0.4101 | 0.2572 | 0.001147 |
| mmd2 | 0.005272 | 0.002774 | 0.01405 | 0.006711 | -2.073e-05 |
| mmd_sigma | 2.349 | 2.34 | 2.375 | 2.364 | -0.000909 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001843 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.37 | 1.362 | -3.503e-05 |
| rho_mean | 1 | 0.9919 | 1.009 | 0.999 | -9.36e-05 |
| rho_min | 0.1234 | 0.01525 | 0.1234 | 0.08237 | 0.00357 |
| rho_p95 | 1.338 | 1.332 | 1.34 | 1.336 | 0.0001653 |
| rho_p99 | 1.356 | 1.354 | 1.359 | 1.356 | -5.359e-06 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.077 | -6.229e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.368 | 1.362 | -2.077e-05 |
| rho_mean | 1.001 | 0.9913 | 1.008 | 1.001 | -1.27e-05 |
| rho_min | 0.04095 | 0.01906 | 0.108 | 0.0795 | -0.003398 |
| rho_p95 | 1.337 | 1.334 | 1.34 | 1.336 | 1.22e-05 |
| rho_p99 | 1.357 | 1.355 | 1.359 | 1.356 | 3.531e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03184 | 0.02534 | 0.03597 | 0.03115 | 0.0002677 |
| w_sharp_drift | 0.00672 | 0 | 0.009346 | 0.006451 | 6.135e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.778e-05 | 2.258e-05 | 3.034e-05 | 2.749e-05 | -2.923e-08 |
| td_loss_from_Q | 1.254e-05 | 9.742e-06 | 1.653e-05 | 1.292e-05 | -5.325e-08 |
| td_loss_from_Q_abs_diff | 1.524e-05 | 9.851e-06 | 1.654e-05 | 1.457e-05 | 2.401e-08 |
| td_loss_from_Q_rel_diff | 0.5486 | 0.4128 | 0.5979 | 0.5302 | 0.001444 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 1.969e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.371 | 1.362 | -3.25e-05 |
| rho_mean | 1 | 0.9972 | 1.003 | 1.001 | 6.43e-05 |
| rho_min | 0.01948 | 0.006676 | 0.05263 | 0.02662 | -0.001346 |
| rho_p95 | 1.336 | 1.335 | 1.339 | 1.336 | 3.946e-05 |
| rho_p99 | 1.356 | 1.356 | 1.358 | 1.356 | -1.46e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.555e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.555e-07 |
| stability_proxy_std | 1.778e-06 | 6.389e-07 | 5.795e-06 | 2.286e-06 | -4.647e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.405e-05 | 0.01399 | 0.07292 | 1.083 | 4.53 | - | - | - |
| 1 | 2.65e-05 | 0.01399 | 0.07219 | 1.077 | 4.53 | - | - | - |
| 2 | 2.659e-05 | 0.01399 | 0.07147 | 1.077 | 4.53 | - | - | - |
| 3 | 2.621e-05 | 0.01399 | 0.07076 | 1.085 | 4.53 | - | - | - |
| 4 | 2.749e-05 | 0.01399 | 0.07005 | 1.068 | 4.53 | - | - | - |
| 5 | 2.974e-05 | 0.01399 | 0.06935 | 1.068 | 4.53 | 0.03055 | 1 | 0.00814 |
| 6 | 2.649e-05 | 0.01399 | 0.06866 | 1.072 | 4.53 | - | - | - |
| 7 | 2.476e-05 | 0.01399 | 0.06798 | 1.086 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.713e-05 | 0.01398 | 0.01064 | 1.065 | 4.53 | - | - | - |
| 193 | 2.873e-05 | 0.01398 | 0.01053 | 1.073 | 4.53 | - | - | - |
| 194 | 2.672e-05 | 0.01398 | 0.01043 | 1.08 | 4.53 | - | - | - |
| 195 | 2.658e-05 | 0.01398 | 0.01032 | 1.068 | 4.53 | 0.03477 | 1 | 0.007276 |
| 196 | 2.615e-05 | 0.01398 | 0.01022 | 1.104 | 4.53 | - | - | - |
| 197 | 2.779e-05 | 0.01398 | 0.01012 | 1.066 | 4.53 | - | - | - |
| 198 | 2.564e-05 | 0.01398 | 0.01002 | 1.068 | 4.53 | - | - | - |
| 199 | 2.778e-05 | 0.01398 | 0.009918 | 1.07 | 4.53 | 0.03184 | 1 | 0.005272 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
