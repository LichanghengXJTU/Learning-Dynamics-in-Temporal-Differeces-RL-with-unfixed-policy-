# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_rho_clip_on
- timestamp: 2025-12-29T04:14:28
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
| td_loss | 2.092e-05 | 2.087e-05 | 2.616e-05 | 2.246e-05 | 1.792e-07 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -1.446e-06 |
| mean_rho2 | 1.443 | 1.291 | 1.573 | 1.416 | 0.01587 |
| tracking_gap | 4.588e-12 | 9.466e-15 | 4.696e-12 | 4.586e-12 | -2.261e-14 |
| critic_teacher_error | 0.04454 | 0.04454 | 0.04455 | 0.04454 | -1.588e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | 6.651e-14 |
| dist_action_tv | 0.3338 | 0.3331 | 0.334 | 0.3338 | -7.574e-06 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| mean_l2 | 0.7595 | 0.3021 | 0.8321 | 0.5138 | 0.01074 |
| mmd2 | 0.04631 | 0.01231 | 0.05497 | 0.02611 | 0.001079 |
| mmd_sigma | 2.38 | 2.292 | 2.38 | 2.36 | 0.001554 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.419 | 1.267 | 1.468 | 1.41 | 0.002955 |
| rho_clip | 5 | 5 | 5 | 5 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 5 | 5 | 5 | 5 | 0 |
| rho_mean | 0.8805 | 0.852 | 0.8924 | 0.8785 | 0.0006843 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | 1.287e-06 |
| rho_p95 | 2.385 | 2.157 | 2.513 | 2.403 | 0.0002395 |
| rho_p99 | 5 | 4.936 | 5 | 5 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.47 | 1.33 | 1.513 | 1.403 | 0.002961 |
| rho_clip | 5 | 5 | 5 | 5 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 5 | 5 | 5 | 5 | 0 |
| rho_mean | 0.888 | 0.8631 | 0.9004 | 0.8776 | 0.0007033 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | -6.49e-06 |
| rho_p95 | 2.446 | 2.264 | 2.564 | 2.366 | 0.00476 |
| rho_p99 | 5 | 5 | 5 | 5 | 0 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.32 | 0.2207 | 0.32 | 0.2757 | 0.005427 |
| w_sharp_drift | 0.08539 | 0 | 0.09465 | 0.08319 | -7.348e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| td_loss | 2.092e-05 | 2.092e-05 | 2.616e-05 | 2.296e-05 | -3.225e-07 |
| td_loss_from_Q | 1.066e-05 | 9.371e-06 | 1.337e-05 | 1.14e-05 | -1.963e-07 |
| td_loss_from_Q_abs_diff | 1.025e-05 | 1.021e-05 | 1.356e-05 | 1.156e-05 | -1.262e-07 |
| td_loss_from_Q_rel_diff | 0.4903 | 0.4329 | 0.5759 | 0.5034 | 0.001704 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.391 | 1.391 | 1.462 | 1.415 | -0.001998 |
| rho_clip | 5 | 5 | 5 | 5 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 5 | 5 | 5 | 5 | 0 |
| rho_mean | 0.8749 | 0.8749 | 0.8889 | 0.8804 | -0.0006245 |
| rho_min | 0.3906 | 0.3906 | 0.3906 | 0.3906 | -7.942e-08 |
| rho_p95 | 2.361 | 2.361 | 2.491 | 2.393 | -0.002913 |
| rho_p99 | 5 | 5 | 5 | 5 | 0 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 2.84e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 2.84e-08 |
| stability_proxy_std | 5.801e-06 | 1.863e-06 | 7.688e-06 | 4.548e-06 | 2.222e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.366e-05 | 0.04455 | 9.466e-15 | 1.376 | 9.163 | - | - | - |
| 1 | 2.31e-05 | 0.04455 | 2.265e-14 | 1.46 | 9.163 | - | - | - |
| 2 | 2.321e-05 | 0.04455 | 4.637e-14 | 1.382 | 9.163 | - | - | - |
| 3 | 2.305e-05 | 0.04455 | 6.985e-14 | 1.371 | 9.163 | - | - | - |
| 4 | 2.373e-05 | 0.04455 | 1.21e-13 | 1.45 | 9.163 | - | - | - |
| 5 | 2.315e-05 | 0.04455 | 1.194e-13 | 1.499 | 9.163 | 0.2401 | 1 | 0.02751 |
| 6 | 2.524e-05 | 0.04455 | 1.311e-13 | 1.396 | 9.163 | - | - | - |
| 7 | 2.356e-05 | 0.04455 | 1.541e-13 | 1.433 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 2.25e-05 | 0.04454 | 4.46e-12 | 1.443 | 9.163 | - | - | - |
| 73 | 2.486e-05 | 0.04454 | 4.415e-12 | 1.382 | 9.163 | - | - | - |
| 74 | 2.139e-05 | 0.04454 | 4.607e-12 | 1.412 | 9.163 | - | - | - |
| 75 | 2.21e-05 | 0.04454 | 4.696e-12 | 1.462 | 9.163 | 0.2794 | 1 | 0.01639 |
| 76 | 2.103e-05 | 0.04454 | 4.554e-12 | 1.291 | 9.163 | - | - | - |
| 77 | 2.308e-05 | 0.04454 | 4.55e-12 | 1.397 | 9.163 | - | - | - |
| 78 | 2.518e-05 | 0.04454 | 4.544e-12 | 1.488 | 9.163 | - | - | - |
| 79 | 2.092e-05 | 0.04454 | 4.588e-12 | 1.443 | 9.163 | 0.32 | 1 | 0.04631 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
