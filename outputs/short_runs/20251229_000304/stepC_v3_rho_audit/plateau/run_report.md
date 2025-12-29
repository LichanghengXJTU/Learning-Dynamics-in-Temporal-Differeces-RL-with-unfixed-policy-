# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/plateau
- timestamp: 2025-12-29T00:21:11
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
| td_loss | 1.548e-05 | 1.358e-05 | 1.941e-05 | 1.658e-05 | -1.884e-07 |
| w_norm | 4.582 | 4.582 | 4.582 | 4.582 | -1.127e-06 |
| mean_rho2 | 1.331 | 1.319 | 1.337 | 1.329 | 0.001279 |
| tracking_gap | 2.435e-13 | 1.928e-15 | 2.65e-13 | 2.369e-13 | -7.73e-16 |
| critic_teacher_error | 0.01394 | 0.01394 | 0.01395 | 0.01394 | -6.551e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | -1.483e-15 |
| dist_action_tv | 0.1133 | 0.1125 | 0.1134 | 0.1129 | 4.787e-05 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| mean_l2 | 0.2996 | 0.1591 | 0.4377 | 0.2614 | 0.001456 |
| mmd2 | 0.00835 | 0.00348 | 0.01532 | 0.006733 | 7.089e-05 |
| mmd_sigma | 2.37 | 2.345 | 2.377 | 2.365 | -4.809e-05 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.328 | 1.319 | 1.337 | 1.329 | -0.0002932 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | -2.659e-06 |
| rho_mean | 1.149 | 1.145 | 1.152 | 1.149 | -0.0001252 |
| rho_min | 1.058 | 1.055 | 1.06 | 1.058 | -4.09e-05 |
| rho_p95 | 1.336 | 1.332 | 1.338 | 1.337 | -0.0001277 |
| rho_p99 | 1.356 | 1.354 | 1.357 | 1.356 | 2.109e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.331 | 1.32 | 1.333 | 1.328 | 2.813e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 1.339e-05 |
| rho_mean | 1.15 | 1.145 | 1.151 | 1.149 | 1.152e-05 |
| rho_min | 1.057 | 1.056 | 1.06 | 1.057 | -7.604e-05 |
| rho_p95 | 1.339 | 1.333 | 1.339 | 1.336 | 4.004e-06 |
| rho_p99 | 1.356 | 1.355 | 1.357 | 1.356 | -5.409e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.09999 | 0.09419 | 0.11 | 0.1041 | -0.0002325 |
| w_sharp_drift | 0.02164 | 0 | 0.02593 | 0.02205 | -7.721e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| td_loss | 1.548e-05 | 1.443e-05 | 1.865e-05 | 1.611e-05 | -7.933e-08 |
| td_loss_from_Q | 6.413e-06 | 6.413e-06 | 9.888e-06 | 7.661e-06 | -1.039e-07 |
| td_loss_from_Q_abs_diff | 9.069e-06 | 6.568e-06 | 1.003e-05 | 8.444e-06 | 2.454e-08 |
| td_loss_from_Q_rel_diff | 0.5858 | 0.4386 | 0.5858 | 0.5245 | 0.004298 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.329 | 1.326 | 1.331 | 1.327 | 0.0002262 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 1.636e-07 |
| rho_mean | 1.149 | 1.147 | 1.15 | 1.148 | 9.724e-05 |
| rho_min | 1.057 | 1.054 | 1.058 | 1.057 | 2.278e-06 |
| rho_p95 | 1.336 | 1.335 | 1.338 | 1.336 | -2.253e-05 |
| rho_p99 | 1.356 | 1.356 | 1.357 | 1.356 | -3.096e-05 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 3.713e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 3.713e-08 |
| stability_proxy_std | 2.3e-06 | 7.096e-07 | 3.211e-06 | 1.657e-06 | 3.236e-08 |

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
| 72 | 1.681e-05 | 0.01394 | 2.65e-13 | 1.323 | 4.582 | - | - | - |
| 73 | 1.73e-05 | 0.01394 | 2.572e-13 | 1.331 | 4.582 | - | - | - |
| 74 | 1.614e-05 | 0.01394 | 2.503e-13 | 1.327 | 4.582 | - | - | - |
| 75 | 1.617e-05 | 0.01394 | 2.501e-13 | 1.321 | 4.582 | 0.107 | 1 | 0.006991 |
| 76 | 1.682e-05 | 0.01394 | 2.277e-13 | 1.335 | 4.582 | - | - | - |
| 77 | 1.813e-05 | 0.01394 | 2.301e-13 | 1.329 | 4.582 | - | - | - |
| 78 | 1.631e-05 | 0.01394 | 2.333e-13 | 1.328 | 4.582 | - | - | - |
| 79 | 1.548e-05 | 0.01394 | 2.435e-13 | 1.331 | 4.582 | 0.09999 | 1 | 0.00835 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
