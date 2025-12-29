# Run Report

## Run Info
- run_dir: outputs/sweep/20251229_102740/runs/b0p005_aw0p02_sigma_mu0p175_s0_eb4a791b
- timestamp: 2025-12-29T11:12:48
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.02, alpha_pi=0.06, beta=0.005, sigma_mu=0.175, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 6.25e-06
- stability_probe_step_scale: 6.25e-06
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.501e-05 | 1.188e-05 | 1.857e-05 | 1.539e-05 | -3.116e-07 |
| w_norm | 4.582 | 4.582 | 4.582 | 4.582 | -1.504e-07 |
| mean_rho2 | 2.013 | 1.687 | 2.522 | 2.05 | -0.02031 |
| tracking_gap | 1.529e-12 | 6.328e-15 | 1.639e-12 | 1.527e-12 | -2.332e-15 |
| critic_teacher_error | 0.01394 | 0.01394 | 0.01395 | 0.01394 | -1.305e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.8608 | 0.8608 | 0.8608 | 0.8608 | -4.28e-15 |
| dist_action_tv | 0.3787 | 0.3778 | 0.3791 | 0.3786 | -1.948e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.3784 | 0.143 | 0.4227 | 0.2891 | 0.002107 |
| mmd2 | 0.01211 | 0.003544 | 0.01521 | 0.009067 | 7.023e-05 |
| mmd_sigma | 2.353 | 2.337 | 2.376 | 2.357 | 0.0004261 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 2.002 | 1.726 | 2.369 | 2.046 | 0.0004397 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 10 | 10 | 10 | 10 | 0 |
| rho_mean | 0.8771 | 0.8393 | 0.908 | 0.8806 | 0.0001692 |
| rho_min | 0.3403 | 0.3403 | 0.3405 | 0.3403 | -1.914e-06 |
| rho_p95 | 2.526 | 2.164 | 2.673 | 2.464 | 0.0007652 |
| rho_p99 | 6.857 | 5.302 | 9.186 | 6.891 | 0.03683 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.981 | 1.794 | 2.419 | 1.988 | -0.01039 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 10 | 10 | 10 | 10 | 0 |
| rho_mean | 0.8719 | 0.8524 | 0.9182 | 0.8731 | -0.001044 |
| rho_min | 0.3404 | 0.3403 | 0.3405 | 0.3403 | 1.474e-07 |
| rho_p95 | 2.258 | 2.188 | 2.719 | 2.384 | -0.01212 |
| rho_p99 | 6.524 | 5.449 | 9.132 | 6.632 | -0.03864 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.01836 | 0.01558 | 0.02794 | 0.01955 | 4.347e-05 |
| w_sharp_drift | 0.005231 | 0 | 0.01011 | 0.006151 | -5.154e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 1.501e-05 | 1.314e-05 | 1.857e-05 | 1.517e-05 | -8.897e-09 |
| td_loss_from_Q | 7.575e-06 | 6.078e-06 | 1.02e-05 | 7.358e-06 | 4.921e-08 |
| td_loss_from_Q_abs_diff | 7.432e-06 | 6.264e-06 | 8.984e-06 | 7.812e-06 | -5.811e-08 |
| td_loss_from_Q_rel_diff | 0.4952 | 0.4286 | 0.5792 | 0.5151 | -0.0034 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 2.04 | 1.979 | 2.186 | 2.09 | -0.008451 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 10 | 10 | 10 | 10 | 0 |
| rho_mean | 0.879 | 0.871 | 0.8955 | 0.8829 | -0.001118 |
| rho_min | 0.3403 | 0.3403 | 0.3403 | 0.3403 | -8.56e-07 |
| rho_p95 | 2.393 | 2.369 | 2.53 | 2.445 | -0.01051 |
| rho_p99 | 6.85 | 6.561 | 7.593 | 7.118 | -0.0258 |
| stability_probe_step_scale | 6.25e-06 | 6.25e-06 | 6.25e-06 | 6.25e-06 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 3.955e-09 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 3.955e-09 |
| stability_proxy_std | 1.928e-07 | 1.193e-07 | 8.049e-07 | 4.093e-07 | -2.262e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.554e-05 | 0.01395 | 6.328e-15 | 1.922 | 4.582 | - | - | - |
| 1 | 1.39e-05 | 0.01395 | 1.688e-14 | 2.282 | 4.582 | - | - | - |
| 2 | 1.598e-05 | 0.01395 | 2.177e-14 | 2.034 | 4.582 | - | - | - |
| 3 | 1.53e-05 | 0.01395 | 2.775e-14 | 1.973 | 4.582 | - | - | - |
| 4 | 1.627e-05 | 0.01395 | 4.254e-14 | 2.085 | 4.582 | - | - | - |
| 5 | 1.621e-05 | 0.01395 | 5.26e-14 | 2.293 | 4.582 | 0.02284 | 1 | 0.005581 |
| 6 | 1.592e-05 | 0.01395 | 5.88e-14 | 1.929 | 4.582 | - | - | - |
| 7 | 1.43e-05 | 0.01395 | 4.574e-14 | 2.1 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 1.549e-05 | 0.01394 | 1.497e-12 | 2.252 | 4.582 | - | - | - |
| 193 | 1.765e-05 | 0.01394 | 1.539e-12 | 1.84 | 4.582 | - | - | - |
| 194 | 1.408e-05 | 0.01394 | 1.536e-12 | 2.211 | 4.582 | - | - | - |
| 195 | 1.625e-05 | 0.01394 | 1.561e-12 | 2.228 | 4.582 | 0.0197 | 1 | 0.007236 |
| 196 | 1.622e-05 | 0.01394 | 1.507e-12 | 1.966 | 4.582 | - | - | - |
| 197 | 1.387e-05 | 0.01394 | 1.491e-12 | 1.852 | 4.582 | - | - | - |
| 198 | 1.559e-05 | 0.01394 | 1.548e-12 | 2.193 | 4.582 | - | - | - |
| 199 | 1.501e-05 | 0.01394 | 1.529e-12 | 2.013 | 4.582 | 0.01836 | 1 | 0.01211 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
