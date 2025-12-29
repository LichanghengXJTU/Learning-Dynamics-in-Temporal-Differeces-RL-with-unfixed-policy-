# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_sigma_mismatch
- timestamp: 2025-12-29T03:20:40
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.2, alpha_pi=0.12, beta=0.01, sigma_mu=0.2, sigma_pi=0.6, p_mix=0.01

## Health
- status: FAIL (rho_sane: rho exceeds threshold)

## Scale Checks
- train_step_scale: 6.25e-05
- stability_probe_step_scale: 6.25e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.195e-05 | 2.087e-05 | 2.591e-05 | 2.258e-05 | 5.751e-08 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -1.733e-06 |
| mean_rho2 | 10.66 | 1.68 | 1381 | 27.11 | -12.85 |
| tracking_gap | 1.165e-11 | 2.772e-14 | 2.053e-11 | 1.201e-11 | -1.928e-13 |
| critic_teacher_error | 0.04454 | 0.04454 | 0.04455 | 0.04454 | -8.922e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 5.803 | 5.803 | 5.803 | 5.803 | 2.186e-13 |
| dist_action_tv | 0.6743 | 0.6737 | 0.6768 | 0.6755 | -0.0001545 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| mean_l2 | 0.352 | 0.2684 | 0.786 | 0.4607 | -0.01068 |
| mmd2 | 0.01367 | 0.009776 | 0.04762 | 0.0214 | -0.0007355 |
| mmd_sigma | 2.352 | 2.29 | 2.376 | 2.347 | -5.213e-06 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 19.65 | 8.131 | 286.2 | 46.86 | -0.345 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 201.6 | 79.61 | 825.5 | 340 | 1.29 |
| rho_mean | 0.649 | 0.5851 | 0.9494 | 0.6849 | 0.00215 |
| rho_min | 0.1111 | 0.1111 | 0.1112 | 0.1111 | 5.341e-07 |
| rho_p95 | 1.556 | 1.344 | 1.679 | 1.573 | 0.0002265 |
| rho_p99 | 6.07 | 4.495 | 8.248 | 6.606 | -0.002545 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 34.26 | 5.497 | 2603 | 132 | -10.07 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 347.1 | 85.16 | 3261 | 383.1 | -0.06497 |
| rho_mean | 0.6633 | 0.5746 | 1.43 | 0.7873 | -0.01109 |
| rho_min | 0.1111 | 0.1111 | 0.1112 | 0.1111 | -2.694e-06 |
| rho_p95 | 1.614 | 1.441 | 1.729 | 1.538 | 0.004547 |
| rho_p99 | 7.713 | 4.894 | 9.35 | 6.774 | -0.008435 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.2353 | 0.1334 | 0.536 | 0.2728 | -0.004244 |
| w_sharp_drift | 0.1348 | 0 | 0.67 | 0.2653 | 0.0006194 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| td_loss | 2.195e-05 | 2.087e-05 | 2.562e-05 | 2.275e-05 | -1.928e-07 |
| td_loss_from_Q | 1.087e-05 | 9.487e-06 | 1.297e-05 | 1.098e-05 | -1.005e-07 |
| td_loss_from_Q_abs_diff | 1.108e-05 | 9.855e-06 | 1.405e-05 | 1.177e-05 | -9.229e-08 |
| td_loss_from_Q_rel_diff | 0.5048 | 0.4551 | 0.5757 | 0.516 | 0.0005964 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 208.7 | 24.51 | 5.807e+04 | 1.173e+04 | -767.1 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 1958 | 631.2 | 4.354e+04 | 9758 | -484.8 |
| rho_mean | 0.7791 | 0.6222 | 2.149 | 1.036 | -0.01143 |
| rho_min | 0.1111 | 0.1111 | 0.1111 | 0.1111 | -3.304e-08 |
| rho_p95 | 1.533 | 1.533 | 1.658 | 1.563 | -0.002773 |
| rho_p99 | 5.989 | 5.989 | 7.106 | 6.498 | -0.02968 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.946e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.946e-07 |
| stability_proxy_std | 6.228e-06 | 3.117e-06 | 5.53e-05 | 5.734e-06 | 5.228e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.427e-05 | 0.04455 | 2.772e-14 | 21.23 | 9.163 | - | - | - |
| 1 | 2.301e-05 | 0.04455 | 3.099e-14 | 29.81 | 9.163 | - | - | - |
| 2 | 2.442e-05 | 0.04455 | 1.135e-13 | 206 | 9.163 | - | - | - |
| 3 | 2.262e-05 | 0.04455 | 1.142e-13 | 62.84 | 9.163 | - | - | - |
| 4 | 2.424e-05 | 0.04455 | 4.802e-13 | 25.53 | 9.163 | - | - | - |
| 5 | 2.209e-05 | 0.04455 | 4.717e-13 | 138.2 | 9.163 | 0.2141 | 1 | 0.01974 |
| 6 | 2.455e-05 | 0.04455 | 1.333e-12 | 285.5 | 9.163 | - | - | - |
| 7 | 2.312e-05 | 0.04455 | 1.323e-12 | 9.218 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 2.149e-05 | 0.04454 | 1.295e-11 | 1214 | 9.163 | - | - | - |
| 73 | 2.591e-05 | 0.04454 | 1.262e-11 | 14.7 | 9.163 | - | - | - |
| 74 | 2.239e-05 | 0.04454 | 1.241e-11 | 11.66 | 9.163 | - | - | - |
| 75 | 2.236e-05 | 0.04454 | 1.247e-11 | 69.27 | 9.163 | 0.1374 | 1 | 0.02294 |
| 76 | 2.225e-05 | 0.04454 | 1.212e-11 | 30.49 | 9.163 | - | - | - |
| 77 | 2.272e-05 | 0.04454 | 1.199e-11 | 5.896 | 9.163 | - | - | - |
| 78 | 2.363e-05 | 0.04454 | 1.183e-11 | 19.23 | 9.163 | - | - | - |
| 79 | 2.195e-05 | 0.04454 | 1.165e-11 | 10.66 | 9.163 | 0.2353 | 1 | 0.01367 |

## Next Steps
- mean_rho2 is very large -> consider reducing sigma mismatch, increasing beta, increasing p_mix, or enabling rho_clip.
