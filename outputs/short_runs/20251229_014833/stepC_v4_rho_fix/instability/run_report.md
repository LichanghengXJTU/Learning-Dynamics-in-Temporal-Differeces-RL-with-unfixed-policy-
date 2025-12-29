# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability
- timestamp: 2025-12-29T02:40:32
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
| td_loss | 2.199e-05 | 2.068e-05 | 2.612e-05 | 2.268e-05 | -9.567e-07 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -1.491e-06 |
| mean_rho2 | 5.697 | 2.511 | 42.38 | 4.681 | 0.4833 |
| tracking_gap | 3.818e-11 | 4.853e-14 | 4.149e-11 | 3.909e-11 | -5.362e-13 |
| critic_teacher_error | 0.04454 | 0.04454 | 0.04455 | 0.04454 | -1.656e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | 4.273e-13 |
| dist_action_tv | 0.3338 | 0.3331 | 0.3341 | 0.3337 | -3.226e-05 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| mean_l2 | 0.7595 | 0.2771 | 0.8321 | 0.4961 | 0.01582 |
| mmd2 | 0.04631 | 0.01142 | 0.05497 | 0.02452 | 0.001401 |
| mmd_sigma | 2.38 | 2.301 | 2.38 | 2.351 | 0.001036 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 4.648 | 3.253 | 16.43 | 5.066 | -0.1641 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 66.96 | 42.42 | 176 | 72.06 | -3.43 |
| rho_mean | 0.9759 | 0.9364 | 1.033 | 0.9824 | -0.0001987 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | 2.555e-07 |
| rho_p95 | 2.385 | 2.157 | 2.588 | 2.462 | -0.006096 |
| rho_p99 | 6.065 | 4.936 | 7.6 | 6.328 | 0.02828 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 5.224 | 2.983 | 53.05 | 11.22 | -1.384 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 97.18 | 39.12 | 451.3 | 123.1 | -9.056 |
| rho_mean | 0.9838 | 0.9548 | 1.141 | 1.027 | -0.008879 |
| rho_min | 0.3907 | 0.3906 | 0.3909 | 0.3907 | 1.097e-06 |
| rho_p95 | 2.446 | 2.264 | 2.578 | 2.491 | 0.006688 |
| rho_p99 | 7.148 | 5.703 | 8.156 | 7.018 | 0.03675 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.483 | 0.2302 | 0.483 | 0.3573 | 0.00479 |
| w_sharp_drift | 0.2407 | 0 | 0.2637 | 0.1872 | -0.0003818 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| td_loss | 2.199e-05 | 2.068e-05 | 2.436e-05 | 2.242e-05 | -1.501e-07 |
| td_loss_from_Q | 1.148e-05 | 1.044e-05 | 1.336e-05 | 1.167e-05 | -9.556e-08 |
| td_loss_from_Q_abs_diff | 1.051e-05 | 9.144e-06 | 1.336e-05 | 1.074e-05 | -5.455e-08 |
| td_loss_from_Q_rel_diff | 0.478 | 0.4327 | 0.559 | 0.4797 | 0.0007002 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 73.4 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 10.94 | 4.317 | 327.3 | 8.337 | 0.3156 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 318.1 | 124.5 | 2612 | 211.9 | 11.6 |
| rho_mean | 0.9901 | 0.9676 | 1.112 | 0.9923 | 0.0001573 |
| rho_min | 0.3906 | 0.3906 | 0.3906 | 0.3906 | -5.515e-07 |
| rho_p95 | 2.361 | 2.361 | 2.491 | 2.404 | -0.004217 |
| rho_p99 | 6.01 | 6.01 | 6.781 | 6.382 | -0.03904 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 5.742e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 5.742e-08 |
| stability_proxy_std | 6.768e-06 | 2.161e-06 | 1.041e-05 | 4.433e-06 | -4.661e-08 |

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
| 72 | 2.254e-05 | 0.04454 | 4.067e-11 | 3.516 | 9.163 | - | - | - |
| 73 | 2.314e-05 | 0.04454 | 4.053e-11 | 3.719 | 9.163 | 0.2765 | 1 | 0.01478 |
| 74 | 2.412e-05 | 0.04454 | 4.003e-11 | 4.91 | 9.163 | - | - | - |
| 75 | 2.585e-05 | 0.04454 | 3.987e-11 | 2.804 | 9.163 | - | - | - |
| 76 | 2.283e-05 | 0.04454 | 4.018e-11 | 4.389 | 9.163 | - | - | - |
| 77 | 2.173e-05 | 0.04454 | 3.902e-11 | 7.077 | 9.163 | - | - | - |
| 78 | 2.099e-05 | 0.04454 | 3.819e-11 | 3.437 | 9.163 | 0.2966 | 1 | 0.01961 |
| 79 | 2.199e-05 | 0.04454 | 3.818e-11 | 5.697 | 9.163 | 0.483 | 1 | 0.04631 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
