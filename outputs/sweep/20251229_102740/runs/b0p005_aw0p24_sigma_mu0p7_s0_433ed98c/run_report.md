# Run Report

## Run Info
- run_dir: outputs/sweep/20251229_102740/runs/b0p005_aw0p24_sigma_mu0p7_s0_433ed98c
- timestamp: 2025-12-29T13:27:55
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.24, alpha_pi=0.06, beta=0.005, sigma_mu=0.7, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 7.5e-05
- stability_probe_step_scale: 7.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.562e-05 | 1.313e-05 | 2.031e-05 | 1.583e-05 | 5.082e-08 |
| w_norm | 4.581 | 4.581 | 4.582 | 4.581 | -9.564e-07 |
| mean_rho2 | 3.135 | 2.695 | 3.287 | 3.054 | 0.005846 |
| tracking_gap | 2.814e-12 | 1.987e-15 | 2.83e-12 | 2.764e-12 | 4.104e-14 |
| critic_teacher_error | 0.01391 | 0.01391 | 0.01395 | 0.01391 | -1.582e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.8783 | 0.8783 | 0.8783 | 0.8783 | 2.478e-15 |
| dist_action_tv | 0.6028 | 0.4966 | 1.181 | 0.5505 | 0.005459 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2549 | 0.1452 | 0.4412 | 0.2908 | -0.01113 |
| mmd2 | 0.006213 | 0.003825 | 0.016 | 0.008616 | -0.0005822 |
| mmd_sigma | 2.354 | 2.337 | 2.382 | 2.362 | 0.0005575 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 3.051 | 2.758 | 3.246 | 2.97 | 0.001819 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 5.442 | 5.418 | 5.444 | 5.441 | 0.0002061 |
| rho_mean | 1.006 | 0.9419 | 1.06 | 0.9932 | -0.0002037 |
| rho_min | 5.993e-13 | 2.813e-23 | 5.993e-13 | 1.226e-13 | 3.703e-14 |
| rho_p95 | 4.376 | 4.156 | 4.455 | 4.315 | 0.00215 |
| rho_p99 | 5.184 | 5.092 | 5.27 | 5.194 | 0.0006748 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 3.139 | 2.809 | 3.19 | 2.987 | 0.003899 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 5.436 | 5.423 | 5.444 | 5.437 | -1.58e-05 |
| rho_mean | 1.034 | 0.9508 | 1.048 | 1.001 | 0.0005576 |
| rho_min | 7.784e-19 | 2.501e-22 | 2.373e-13 | 6.659e-14 | -6.581e-15 |
| rho_p95 | 4.396 | 4.213 | 4.45 | 4.312 | 0.003921 |
| rho_p99 | 5.215 | 5.132 | 5.26 | 5.18 | 0.001392 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.2716 | 0.1596 | 0.3223 | 0.2812 | -0.003827 |
| w_sharp_drift | 0.07999 | 0 | 0.1585 | 0.08825 | -0.001193 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 1.562e-05 | 1.35e-05 | 2.031e-05 | 1.55e-05 | -1.369e-08 |
| td_loss_from_Q | 7.5e-06 | 6.236e-06 | 1.132e-05 | 7.746e-06 | -3.509e-08 |
| td_loss_from_Q_abs_diff | 8.123e-06 | 6.299e-06 | 1.161e-05 | 7.753e-06 | 2.14e-08 |
| td_loss_from_Q_rel_diff | 0.5199 | 0.4255 | 0.587 | 0.5001 | 0.001806 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 2.988 | 2.921 | 3.068 | 2.984 | 0.002414 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 5.444 | 5.441 | 5.444 | 5.443 | 9.224e-05 |
| rho_mean | 0.9997 | 0.9827 | 1.019 | 0.9989 | 0.0003436 |
| rho_min | 1.198e-22 | 5.894e-28 | 2.042e-17 | 3.869e-20 | -6.239e-21 |
| rho_p95 | 4.302 | 4.285 | 4.398 | 4.319 | 0.001165 |
| rho_p99 | 5.181 | 5.174 | 5.237 | 5.2 | -0.0002104 |
| stability_probe_step_scale | 7.5e-05 | 7.5e-05 | 7.5e-05 | 7.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -3.909e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | -3.909e-08 |
| stability_proxy_std | 3.805e-06 | 9.514e-07 | 1.064e-05 | 3.469e-06 | 6.189e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.734e-05 | 0.01395 | 1.987e-15 | 3.116 | 4.582 | - | - | - |
| 1 | 1.695e-05 | 0.01395 | 3.779e-15 | 2.81 | 4.582 | - | - | - |
| 2 | 1.643e-05 | 0.01395 | 5.53e-15 | 2.976 | 4.582 | - | - | - |
| 3 | 1.674e-05 | 0.01395 | 6.198e-15 | 3.124 | 4.582 | - | - | - |
| 4 | 1.588e-05 | 0.01394 | 7.705e-15 | 3.001 | 4.582 | - | - | - |
| 5 | 1.754e-05 | 0.01394 | 1.542e-14 | 2.754 | 4.582 | 0.2809 | 1 | 0.005639 |
| 6 | 1.508e-05 | 0.01394 | 1.577e-14 | 2.895 | 4.582 | - | - | - |
| 7 | 1.593e-05 | 0.01394 | 1.436e-14 | 3.032 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 1.593e-05 | 0.01392 | 2.643e-12 | 2.905 | 4.581 | - | - | - |
| 193 | 1.731e-05 | 0.01392 | 2.67e-12 | 2.921 | 4.581 | - | - | - |
| 194 | 1.576e-05 | 0.01392 | 2.656e-12 | 3.144 | 4.581 | - | - | - |
| 195 | 1.577e-05 | 0.01392 | 2.668e-12 | 2.953 | 4.581 | 0.2536 | 1 | 0.006288 |
| 196 | 1.623e-05 | 0.01391 | 2.713e-12 | 3.287 | 4.581 | - | - | - |
| 197 | 1.452e-05 | 0.01391 | 2.796e-12 | 2.915 | 4.581 | - | - | - |
| 198 | 1.703e-05 | 0.01391 | 2.83e-12 | 2.982 | 4.581 | - | - | - |
| 199 | 1.562e-05 | 0.01391 | 2.814e-12 | 3.135 | 4.581 | 0.2716 | 1 | 0.006213 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
