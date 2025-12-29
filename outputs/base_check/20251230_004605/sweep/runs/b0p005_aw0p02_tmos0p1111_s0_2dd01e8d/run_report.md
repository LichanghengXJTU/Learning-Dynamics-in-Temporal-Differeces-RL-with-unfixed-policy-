# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p1111_s0_2dd01e8d
- timestamp: 2025-12-30T02:47:06
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
| td_loss | 2.772e-05 | 2.094e-05 | 3.113e-05 | 2.685e-05 | 1.854e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.061e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003201 |
| tracking_gap | 0.003463 | 0.003463 | 0.02546 | 0.003533 | -3.542e-05 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.325e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04301 | 0.04301 | 0.04314 | 0.04302 | -1.817e-07 |
| dist_action_tv | 0.1131 | 0.1126 | 0.1136 | 0.1131 | -1.271e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2728 | 0.1368 | 0.3998 | 0.251 | 0.004313 |
| mmd2 | 0.006927 | 0.003309 | 0.01365 | 0.006312 | 0.0001223 |
| mmd_sigma | 2.348 | 2.338 | 2.379 | 2.364 | -0.001067 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001934 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.364 | 1.361 | 1.879e-06 |
| rho_mean | 1 | 0.9919 | 1.009 | 0.9991 | -0.0001033 |
| rho_min | 0.1184 | 0.01885 | 0.1184 | 0.08216 | 0.003377 |
| rho_p95 | 1.337 | 1.332 | 1.339 | 1.336 | 5.646e-05 |
| rho_p99 | 1.355 | 1.354 | 1.358 | 1.356 | -2.342e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.078 | -7.771e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.363 | 1.361 | -5.633e-06 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -2.212e-05 |
| rho_min | 0.04094 | 0.01939 | 0.1086 | 0.08 | -0.003541 |
| rho_p95 | 1.337 | 1.332 | 1.339 | 1.336 | 5.049e-05 |
| rho_p99 | 1.356 | 1.355 | 1.358 | 1.356 | 5.477e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03247 | 0.02488 | 0.03451 | 0.03092 | 0.0002312 |
| w_sharp_drift | 0.006218 | 0 | 0.009116 | 0.006053 | 2.529e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.772e-05 | 2.25e-05 | 3.021e-05 | 2.736e-05 | -3.927e-08 |
| td_loss_from_Q | 1.254e-05 | 9.183e-06 | 1.681e-05 | 1.315e-05 | -8.565e-08 |
| td_loss_from_Q_abs_diff | 1.518e-05 | 1.044e-05 | 1.621e-05 | 1.422e-05 | 4.638e-08 |
| td_loss_from_Q_rel_diff | 0.5475 | 0.4217 | 0.5919 | 0.5197 | 0.002453 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 3.534e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.364 | 1.361 | 1.7e-06 |
| rho_mean | 1 | 0.9972 | 1.003 | 1.001 | 7.121e-05 |
| rho_min | 0.01892 | 0.007036 | 0.05332 | 0.02712 | -0.001431 |
| rho_p95 | 1.336 | 1.335 | 1.338 | 1.336 | 2.163e-05 |
| rho_p99 | 1.356 | 1.356 | 1.357 | 1.356 | -1.128e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.475e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.475e-07 |
| stability_proxy_std | 1.947e-06 | 7.521e-07 | 5.784e-06 | 2.423e-06 | -2.54e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.617e-05 | 0.01399 | 0.02546 | 1.082 | 4.53 | - | - | - |
| 1 | 2.517e-05 | 0.01399 | 0.02521 | 1.076 | 4.53 | - | - | - |
| 2 | 2.832e-05 | 0.01399 | 0.02495 | 1.076 | 4.53 | - | - | - |
| 3 | 2.501e-05 | 0.01399 | 0.0247 | 1.084 | 4.53 | - | - | - |
| 4 | 2.782e-05 | 0.01399 | 0.02446 | 1.068 | 4.53 | - | - | - |
| 5 | 3.021e-05 | 0.01399 | 0.02421 | 1.068 | 4.53 | 0.0303 | 1 | 0.004332 |
| 6 | 2.732e-05 | 0.01399 | 0.02397 | 1.072 | 4.53 | - | - | - |
| 7 | 2.461e-05 | 0.01399 | 0.02373 | 1.087 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.721e-05 | 0.01398 | 0.003715 | 1.065 | 4.53 | - | - | - |
| 193 | 3.05e-05 | 0.01398 | 0.003678 | 1.074 | 4.53 | - | - | - |
| 194 | 2.641e-05 | 0.01398 | 0.003641 | 1.08 | 4.53 | - | - | - |
| 195 | 2.665e-05 | 0.01398 | 0.003605 | 1.068 | 4.53 | 0.03245 | 1 | 0.007309 |
| 196 | 2.61e-05 | 0.01398 | 0.003569 | 1.104 | 4.53 | - | - | - |
| 197 | 2.795e-05 | 0.01398 | 0.003533 | 1.066 | 4.53 | - | - | - |
| 198 | 2.583e-05 | 0.01398 | 0.003498 | 1.068 | 4.53 | - | - | - |
| 199 | 2.772e-05 | 0.01398 | 0.003463 | 1.07 | 4.53 | 0.03247 | 1 | 0.006927 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
