# Run Report

## Run Info
- run_dir: outputs/base_check/20251229_192835/stepC_v2/instability
- timestamp: 2025-12-29T20:57:08
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.2, alpha_pi=0.12, beta=0.01, sigma_mu=0.25, sigma_pi=0.4, p_mix=0.01

## Health
- status: FAIL (sigma_condition: sigma_pi^2 >= 2 sigma_mu^2)

## Scale Checks
- train_step_scale: 0.00025
- stability_probe_step_scale: 0.00025
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.295e-05 | 2.011e-05 | 2.873e-05 | 2.281e-05 | -6.678e-08 |
| w_norm | 9.159 | 9.159 | 9.163 | 9.159 | -1.515e-05 |
| mean_rho2 | 7.391 | 1.862 | 168.3 | 5.513 | 0.4338 |
| tracking_gap | 1.019e-09 | 3.94e-12 | 1.102e-09 | 1.063e-09 | -1.825e-11 |
| critic_teacher_error | 0.04437 | 0.04437 | 0.04455 | 0.04437 | -7.571e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | 4.91e-12 |
| dist_action_tv | 0.3339 | 0.3331 | 0.3343 | 0.3338 | -1.975e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.6769 | 0.159 | 1.022 | 0.6352 | 0.001914 |
| mmd2 | 0.0348 | 0.005541 | 0.076 | 0.03144 | 9.902e-05 |
| mmd_sigma | 2.351 | 2.31 | 2.393 | 2.35 | -0.0003105 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 2.448 | 2.448 | 96.95 | 3.637 | -0.117 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 23.36 | 23.36 | 609 | 50.51 | -3.273 |
| rho_mean | 0.9353 | 0.9281 | 1.145 | 0.9592 | -0.001243 |
| rho_min | 0.3907 | 0.3906 | 0.3909 | 0.3907 | -2.038e-06 |
| rho_p95 | 2.488 | 2.157 | 2.622 | 2.432 | 0.0006994 |
| rho_p99 | 6.259 | 4.936 | 8.2 | 6.287 | 0.03108 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 8.491 | 2.322 | 53.04 | 4.096 | 0.2963 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 149.8 | 26.52 | 451.3 | 62.9 | 7.421 |
| rho_mean | 0.9785 | 0.9216 | 1.141 | 0.9477 | 0.0003771 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | 1.639e-07 |
| rho_p95 | 2.243 | 2.179 | 2.664 | 2.358 | -0.01109 |
| rho_p99 | 5.977 | 5.063 | 8.156 | 6.069 | -0.03259 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.2522 | 0.2239 | 0.5084 | 0.2759 | -0.001852 |
| w_sharp_drift | 0.1192 | 0 | 0.2839 | 0.1129 | 0.0009354 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.295e-05 | 2.082e-05 | 2.74e-05 | 2.328e-05 | -1.833e-08 |
| td_loss_from_Q | 1.141e-05 | 9.546e-06 | 1.45e-05 | 1.146e-05 | 7.644e-09 |
| td_loss_from_Q_abs_diff | 1.154e-05 | 1.06e-05 | 1.416e-05 | 1.182e-05 | -2.597e-08 |
| td_loss_from_Q_rel_diff | 0.5028 | 0.4376 | 0.5475 | 0.5069 | -0.0006831 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 14.22 | 4.317 | 229 | 12.42 | 0.75 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 499.3 | 95.72 | 2668 | 362.1 | 30.71 |
| rho_mean | 0.9893 | 0.9627 | 1.091 | 0.9979 | -0.0006013 |
| rho_min | 0.3906 | 0.3906 | 0.3907 | 0.3906 | -9.111e-07 |
| rho_p95 | 2.367 | 2.346 | 2.492 | 2.414 | -0.009584 |
| rho_p99 | 6.253 | 6.01 | 6.878 | 6.479 | -0.02169 |
| stability_probe_step_scale | 0.00025 | 0.00025 | 0.00025 | 0.00025 | 0 |
| stability_proxy | 0.9998 | 0.9998 | 1.036 | 0.9998 | 9.639e-08 |
| stability_proxy_mean | 0.9998 | 0.9998 | 1.036 | 0.9998 | 9.639e-08 |
| stability_proxy_std | 8.888e-06 | 8.122e-06 | 0.0945 | 1.517e-05 | -1.376e-07 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.873e-05 | 0.04455 | 3.94e-12 | 4.83 | 9.163 | - | - | - |
| 1 | 2.515e-05 | 0.04455 | 4.934e-12 | 5.428 | 9.163 | - | - | - |
| 2 | 2.508e-05 | 0.04455 | 9.304e-12 | 12.23 | 9.163 | - | - | - |
| 3 | 2.429e-05 | 0.04455 | 1.604e-11 | 6.873 | 9.163 | - | - | - |
| 4 | 2.572e-05 | 0.04455 | 2.483e-11 | 5.39 | 9.163 | - | - | - |
| 5 | 2.74e-05 | 0.04455 | 2.857e-11 | 9.615 | 9.163 | 0.3421 | 0.9999 | 0.03787 |
| 6 | 2.583e-05 | 0.04455 | 3.963e-11 | 19.05 | 9.163 | - | - | - |
| 7 | 2.61e-05 | 0.04454 | 4.144e-11 | 3.497 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.311e-05 | 0.04438 | 1.086e-09 | 22.13 | 9.159 | - | - | - |
| 193 | 2.159e-05 | 0.04438 | 1.09e-09 | 2.874 | 9.159 | - | - | - |
| 194 | 2.348e-05 | 0.04438 | 1.077e-09 | 4.164 | 9.159 | - | - | - |
| 195 | 2.273e-05 | 0.04437 | 1.097e-09 | 4.526 | 9.159 | 0.2593 | 0.9998 | 0.02911 |
| 196 | 2.336e-05 | 0.04437 | 1.077e-09 | 5.514 | 9.159 | - | - | - |
| 197 | 2.279e-05 | 0.04437 | 1.071e-09 | 6.016 | 9.159 | - | - | - |
| 198 | 2.224e-05 | 0.04437 | 1.05e-09 | 4.121 | 9.159 | - | - | - |
| 199 | 2.295e-05 | 0.04437 | 1.019e-09 | 7.391 | 9.159 | 0.2522 | 0.9998 | 0.0348 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
