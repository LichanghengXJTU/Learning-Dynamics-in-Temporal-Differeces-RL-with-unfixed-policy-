# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p1667_s0_f2503c1d
- timestamp: 2025-12-30T03:27:24
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
| td_loss | 2.843e-05 | 2.078e-05 | 3.218e-05 | 2.697e-05 | 2.819e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.08e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.00314 |
| tracking_gap | 0.007402 | 0.007402 | 0.05442 | 0.007553 | -7.572e-05 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.357e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.04303 | 0.04303 | 0.0433 | 0.04304 | -4.344e-07 |
| dist_action_tv | 0.1131 | 0.1127 | 0.1138 | 0.1131 | -1.292e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.3199 | 0.1284 | 0.4576 | 0.2433 | 0.009121 |
| mmd2 | 0.008612 | 0.003245 | 0.01508 | 0.006091 | 0.0002912 |
| mmd_sigma | 2.354 | 2.345 | 2.379 | 2.366 | -0.000891 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001638 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.366 | 1.361 | -3.563e-05 |
| rho_mean | 1 | 0.992 | 1.009 | 0.999 | -9.134e-05 |
| rho_min | 0.1175 | 0.01646 | 0.1175 | 0.08237 | 0.003216 |
| rho_p95 | 1.337 | 1.331 | 1.34 | 1.335 | 6.587e-05 |
| rho_p99 | 1.356 | 1.354 | 1.359 | 1.356 | -4.54e-06 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.058 | 1.091 | 1.077 | -6.847e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.368 | 1.361 | -5.421e-06 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -1.802e-05 |
| rho_min | 0.04133 | 0.02053 | 0.1073 | 0.07952 | -0.003408 |
| rho_p95 | 1.337 | 1.334 | 1.34 | 1.336 | 6.136e-05 |
| rho_p99 | 1.356 | 1.354 | 1.36 | 1.356 | 3.415e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03277 | 0.02425 | 0.03571 | 0.03071 | 0.000362 |
| w_sharp_drift | 0.006132 | 0 | 0.009633 | 0.006648 | -3.15e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.843e-05 | 2.239e-05 | 3.038e-05 | 2.765e-05 | 1.788e-08 |
| td_loss_from_Q | 1.324e-05 | 9.532e-06 | 1.672e-05 | 1.309e-05 | 1.216e-08 |
| td_loss_from_Q_abs_diff | 1.519e-05 | 9.808e-06 | 1.552e-05 | 1.457e-05 | 5.724e-09 |
| td_loss_from_Q_rel_diff | 0.5343 | 0.4219 | 0.6011 | 0.527 | -0.0001376 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.071 | 1.082 | 1.077 | 3.476e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.362 | 1.361 | 1.368 | 1.362 | -9.101e-06 |
| rho_mean | 1 | 0.9972 | 1.003 | 1.001 | 7.095e-05 |
| rho_min | 0.01859 | 0.007865 | 0.0525 | 0.02621 | -0.001289 |
| rho_p95 | 1.336 | 1.335 | 1.339 | 1.336 | 2.108e-05 |
| rho_p99 | 1.356 | 1.356 | 1.358 | 1.356 | 2.19e-06 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.213e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.213e-07 |
| stability_proxy_std | 2.023e-06 | 5.281e-07 | 5.773e-06 | 2.338e-06 | -8.162e-09 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.59e-05 | 0.01399 | 0.05442 | 1.082 | 4.53 | - | - | - |
| 1 | 2.572e-05 | 0.01399 | 0.05388 | 1.077 | 4.53 | - | - | - |
| 2 | 2.662e-05 | 0.01399 | 0.05334 | 1.077 | 4.53 | - | - | - |
| 3 | 2.64e-05 | 0.01399 | 0.05281 | 1.084 | 4.53 | - | - | - |
| 4 | 2.634e-05 | 0.01399 | 0.05228 | 1.068 | 4.53 | - | - | - |
| 5 | 2.995e-05 | 0.01399 | 0.05176 | 1.068 | 4.53 | 0.03124 | 1 | 0.004685 |
| 6 | 2.708e-05 | 0.01399 | 0.05125 | 1.072 | 4.53 | - | - | - |
| 7 | 2.431e-05 | 0.01399 | 0.05073 | 1.087 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.716e-05 | 0.01398 | 0.00794 | 1.065 | 4.53 | - | - | - |
| 193 | 3.033e-05 | 0.01398 | 0.007861 | 1.074 | 4.53 | - | - | - |
| 194 | 2.651e-05 | 0.01398 | 0.007783 | 1.08 | 4.53 | - | - | - |
| 195 | 2.685e-05 | 0.01398 | 0.007705 | 1.067 | 4.53 | 0.03408 | 1 | 0.007246 |
| 196 | 2.614e-05 | 0.01398 | 0.007628 | 1.104 | 4.53 | - | - | - |
| 197 | 2.764e-05 | 0.01398 | 0.007552 | 1.066 | 4.53 | - | - | - |
| 198 | 2.58e-05 | 0.01398 | 0.007477 | 1.068 | 4.53 | - | - | - |
| 199 | 2.843e-05 | 0.01398 | 0.007402 | 1.07 | 4.53 | 0.03277 | 1 | 0.008612 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
