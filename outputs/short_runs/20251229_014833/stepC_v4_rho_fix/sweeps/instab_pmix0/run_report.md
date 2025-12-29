# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/sweeps/instab_pmix0
- timestamp: 2025-12-29T03:56:28
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.2, alpha_pi=0.12, beta=0.01, sigma_mu=0.25, sigma_pi=0.4, p_mix=0.0

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 6.25e-05
- stability_probe_step_scale: 6.25e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.321e-05 | 2.173e-05 | 2.763e-05 | 2.472e-05 | -3.295e-07 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -2.53e-06 |
| mean_rho2 | 3.657 | 1.862 | 42.71 | 5.093 | -0.9542 |
| tracking_gap | 1.596e-11 | 7.34e-14 | 1.596e-11 | 1.524e-11 | 4.31e-13 |
| critic_teacher_error | 0.04454 | 0.04454 | 0.04455 | 0.04454 | -2.161e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | -1.446e-13 |
| dist_action_tv | 0.3338 | 0.3331 | 0.334 | 0.3338 | -7.575e-06 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| mean_l2 | 2.115 | 0.3242 | 2.819 | 1.557 | 0.0736 |
| mmd2 | 0.3339 | 0.02183 | 0.5345 | 0.2103 | 0.01646 |
| mmd_sigma | 2.33 | 2.029 | 2.406 | 2.245 | 0.01125 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 4.648 | 3.253 | 16.43 | 6.06 | 0.02334 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 66.96 | 35.41 | 176 | 92.85 | 0.6164 |
| rho_mean | 0.9759 | 0.9364 | 1.033 | 0.981 | 0.001262 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | 1.288e-06 |
| rho_p95 | 2.385 | 2.157 | 2.513 | 2.403 | 0.0002385 |
| rho_p99 | 6.065 | 4.936 | 7.484 | 6.406 | -0.001205 |

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
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | -6.491e-06 |
| rho_p95 | 2.446 | 2.264 | 2.564 | 2.366 | 0.004761 |
| rho_p99 | 7.148 | 5.233 | 8.156 | 6.528 | -0.006821 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.3273 | 0.2556 | 0.4181 | 0.3152 | -0.001584 |
| w_sharp_drift | 0.1228 | 0 | 0.2078 | 0.1334 | -0.004035 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| td_loss | 2.321e-05 | 2.276e-05 | 2.689e-05 | 2.461e-05 | -1.164e-07 |
| td_loss_from_Q | 1.194e-05 | 1.029e-05 | 1.32e-05 | 1.229e-05 | -5.433e-08 |
| td_loss_from_Q_abs_diff | 1.128e-05 | 1.082e-05 | 1.434e-05 | 1.232e-05 | -6.207e-08 |
| td_loss_from_Q_rel_diff | 0.4858 | 0.4668 | 0.548 | 0.5004 | -0.0002305 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 10.94 | 4.317 | 229 | 53.32 | -2.61 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 318.1 | 146.4 | 2667 | 724.1 | -22.04 |
| rho_mean | 0.9901 | 0.9676 | 1.091 | 1.013 | -0.0005456 |
| rho_min | 0.3906 | 0.3906 | 0.3906 | 0.3906 | -7.926e-08 |
| rho_p95 | 2.361 | 2.361 | 2.491 | 2.393 | -0.002914 |
| rho_p99 | 6.009 | 6.009 | 6.757 | 6.354 | -0.02011 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -3.972e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | -3.972e-08 |
| stability_proxy_std | 8.722e-06 | 1.63e-06 | 1.035e-05 | 6.396e-06 | 2.68e-07 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.425e-05 | 0.04455 | 7.511e-14 | 4.83 | 9.163 | - | - | - |
| 1 | 2.386e-05 | 0.04455 | 7.34e-14 | 5.428 | 9.163 | - | - | - |
| 2 | 2.408e-05 | 0.04455 | 3.113e-12 | 12.23 | 9.163 | - | - | - |
| 3 | 2.547e-05 | 0.04455 | 3.146e-12 | 6.873 | 9.163 | - | - | - |
| 4 | 2.587e-05 | 0.04455 | 4.298e-12 | 5.39 | 9.163 | - | - | - |
| 5 | 2.546e-05 | 0.04455 | 4.17e-12 | 9.615 | 9.163 | 0.2947 | 1 | 0.1736 |
| 6 | 2.763e-05 | 0.04455 | 4.617e-12 | 19.05 | 9.163 | - | - | - |
| 7 | 2.243e-05 | 0.04455 | 4.56e-12 | 3.497 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 2.297e-05 | 0.04454 | 1.461e-11 | 34.81 | 9.163 | - | - | - |
| 73 | 2.428e-05 | 0.04454 | 1.45e-11 | 3.97 | 9.163 | - | - | - |
| 74 | 2.253e-05 | 0.04454 | 1.449e-11 | 3.893 | 9.163 | - | - | - |
| 75 | 2.523e-05 | 0.04454 | 1.449e-11 | 8.411 | 9.163 | 0.27 | 1 | 0.3493 |
| 76 | 2.521e-05 | 0.04454 | 1.456e-11 | 5.227 | 9.163 | - | - | - |
| 77 | 2.401e-05 | 0.04454 | 1.529e-11 | 2.978 | 9.163 | - | - | - |
| 78 | 2.594e-05 | 0.04454 | 1.592e-11 | 5.193 | 9.163 | - | - | - |
| 79 | 2.321e-05 | 0.04454 | 1.596e-11 | 3.657 | 9.163 | 0.3273 | 1 | 0.3339 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
