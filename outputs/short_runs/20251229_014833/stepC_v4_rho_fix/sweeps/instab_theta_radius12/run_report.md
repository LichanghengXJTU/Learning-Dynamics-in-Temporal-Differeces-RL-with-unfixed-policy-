# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_theta_radius12
- timestamp: 2025-12-29T03:38:34
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.2, alpha_pi=0.12, beta=0.01, sigma_mu=0.25, sigma_pi=0.4, p_mix=0.01

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 6.25e-05
- stability_probe_step_scale: 6.25e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.086e-05 | 2.082e-05 | 2.612e-05 | 2.24e-05 | 1.78e-07 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -1.792e-06 |
| mean_rho2 | 3.657 | 1.862 | 42.71 | 5.093 | -0.9542 |
| tracking_gap | 2.379e-11 | 4.853e-14 | 2.655e-11 | 2.41e-11 | -2.247e-13 |
| critic_teacher_error | 0.04454 | 0.04454 | 0.04455 | 0.04454 | -1.643e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | -9.801e-14 |
| dist_action_tv | 0.3338 | 0.3331 | 0.334 | 0.3338 | -7.574e-06 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| mean_l2 | 0.7595 | 0.3021 | 0.8321 | 0.5433 | 0.008756 |
| mmd2 | 0.04631 | 0.01231 | 0.05497 | 0.02811 | 0.000945 |
| mmd_sigma | 2.38 | 2.292 | 2.38 | 2.358 | 0.001705 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 4.648 | 3.253 | 16.43 | 6.06 | 0.02335 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 66.96 | 35.41 | 176 | 92.85 | 0.6165 |
| rho_mean | 0.9759 | 0.9364 | 1.033 | 0.981 | 0.001262 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | 1.287e-06 |
| rho_p95 | 2.385 | 2.157 | 2.513 | 2.403 | 0.00024 |
| rho_p99 | 6.065 | 4.936 | 7.484 | 6.406 | -0.001202 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 5.224 | 2.963 | 53.05 | 9.783 | -0.4294 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 97.18 | 37.09 | 451.3 | 97.93 | 0.8747 |
| rho_mean | 0.9838 | 0.9335 | 1.141 | 1.004 | -0.001892 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | -6.488e-06 |
| rho_p95 | 2.446 | 2.264 | 2.564 | 2.366 | 0.004759 |
| rho_p99 | 7.148 | 5.233 | 8.156 | 6.528 | -0.006819 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.4824 | 0.2298 | 0.4824 | 0.3492 | 0.01145 |
| w_sharp_drift | 0.2503 | 0 | 0.2858 | 0.2038 | 0.0008365 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| td_loss | 2.086e-05 | 2.086e-05 | 2.611e-05 | 2.29e-05 | -3.221e-07 |
| td_loss_from_Q | 1.064e-05 | 9.348e-06 | 1.336e-05 | 1.137e-05 | -1.959e-07 |
| td_loss_from_Q_abs_diff | 1.023e-05 | 1.02e-05 | 1.353e-05 | 1.153e-05 | -1.263e-07 |
| td_loss_from_Q_rel_diff | 0.4902 | 0.4329 | 0.5759 | 0.5034 | 0.001697 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 10.94 | 4.317 | 229 | 53.32 | -2.61 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 318.1 | 146.4 | 2668 | 724.1 | -22.04 |
| rho_mean | 0.9901 | 0.9676 | 1.091 | 1.013 | -0.0005456 |
| rho_min | 0.3906 | 0.3906 | 0.3906 | 0.3906 | -7.924e-08 |
| rho_p95 | 2.361 | 2.361 | 2.491 | 2.393 | -0.002912 |
| rho_p99 | 6.01 | 6.01 | 6.757 | 6.354 | -0.02011 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -6.133e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | -6.133e-08 |
| stability_proxy_std | 6.768e-06 | 2.758e-06 | 9.505e-06 | 6.069e-06 | -3.181e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.366e-05 | 0.04455 | 4.853e-14 | 4.83 | 9.163 | - | - | - |
| 1 | 2.31e-05 | 0.04455 | 1.313e-13 | 5.428 | 9.163 | - | - | - |
| 2 | 2.32e-05 | 0.04455 | 3.06e-12 | 12.23 | 9.163 | - | - | - |
| 3 | 2.304e-05 | 0.04455 | 3.042e-12 | 6.873 | 9.163 | - | - | - |
| 4 | 2.372e-05 | 0.04455 | 4.142e-12 | 5.39 | 9.163 | - | - | - |
| 5 | 2.314e-05 | 0.04455 | 3.955e-12 | 9.615 | 9.163 | 0.2463 | 1 | 0.02751 |
| 6 | 2.523e-05 | 0.04455 | 3.923e-12 | 19.05 | 9.163 | - | - | - |
| 7 | 2.355e-05 | 0.04455 | 3.898e-12 | 3.497 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 2.245e-05 | 0.04454 | 2.498e-11 | 34.81 | 9.163 | - | - | - |
| 73 | 2.479e-05 | 0.04454 | 2.416e-11 | 3.97 | 9.163 | - | - | - |
| 74 | 2.134e-05 | 0.04454 | 2.477e-11 | 3.893 | 9.163 | - | - | - |
| 75 | 2.204e-05 | 0.04454 | 2.49e-11 | 8.411 | 9.163 | 0.2871 | 1 | 0.01639 |
| 76 | 2.097e-05 | 0.04454 | 2.398e-11 | 5.227 | 9.163 | - | - | - |
| 77 | 2.302e-05 | 0.04454 | 2.386e-11 | 2.978 | 9.163 | - | - | - |
| 78 | 2.511e-05 | 0.04454 | 2.396e-11 | 5.193 | 9.163 | - | - | - |
| 79 | 2.086e-05 | 0.04454 | 2.379e-11 | 3.657 | 9.163 | 0.4824 | 1 | 0.04631 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
