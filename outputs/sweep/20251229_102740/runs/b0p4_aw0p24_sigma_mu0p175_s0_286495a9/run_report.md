# Run Report

## Run Info
- run_dir: outputs/sweep/20251229_102740/runs/b0p4_aw0p24_sigma_mu0p175_s0_286495a9
- timestamp: 2025-12-29T15:43:36
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.24, alpha_pi=0.06, beta=0.4, sigma_mu=0.175, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 7.5e-05
- stability_probe_step_scale: 7.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.452e-05 | 1.16e-05 | 1.833e-05 | 1.489e-05 | -3.053e-07 |
| w_norm | 4.581 | 4.581 | 4.582 | 4.581 | -1.167e-06 |
| mean_rho2 | 2.013 | 1.687 | 2.522 | 2.05 | -0.02031 |
| tracking_gap | 3.04e-15 | 5.389e-16 | 5.892e-15 | 2.557e-15 | 1.622e-16 |
| critic_teacher_error | 0.01392 | 0.01392 | 0.01395 | 0.01392 | -1.51e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.8608 | 0.8608 | 0.8608 | 0.8608 | -1.595e-16 |
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
| rho_p95 | 2.526 | 2.164 | 2.673 | 2.464 | 0.0007648 |
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
| rho_min | 0.3404 | 0.3403 | 0.3405 | 0.3403 | 1.472e-07 |
| rho_p95 | 2.258 | 2.188 | 2.719 | 2.384 | -0.01212 |
| rho_p99 | 6.524 | 5.449 | 9.132 | 6.632 | -0.03863 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.1967 | 0.1705 | 0.2893 | 0.208 | 0.0004241 |
| w_sharp_drift | 0.05845 | 0 | 0.09912 | 0.06684 | -0.0005299 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 1.452e-05 | 1.28e-05 | 1.833e-05 | 1.47e-05 | -1.123e-08 |
| td_loss_from_Q | 7.322e-06 | 6.002e-06 | 1.006e-05 | 7.128e-06 | 4.603e-08 |
| td_loss_from_Q_abs_diff | 7.198e-06 | 6.109e-06 | 8.976e-06 | 7.573e-06 | -5.726e-08 |
| td_loss_from_Q_rel_diff | 0.4957 | 0.4288 | 0.5793 | 0.5152 | -0.003376 |

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
| rho_min | 0.3403 | 0.3403 | 0.3403 | 0.3403 | -8.561e-07 |
| rho_p95 | 2.393 | 2.369 | 2.53 | 2.445 | -0.01051 |
| rho_p99 | 6.85 | 6.561 | 7.593 | 7.118 | -0.0258 |
| stability_probe_step_scale | 7.5e-05 | 7.5e-05 | 7.5e-05 | 7.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 2.292e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 2.292e-08 |
| stability_proxy_std | 1.784e-06 | 1.35e-06 | 7.737e-06 | 4.065e-06 | -2.332e-07 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.554e-05 | 0.01395 | 2.301e-15 | 1.922 | 4.582 | - | - | - |
| 1 | 1.39e-05 | 0.01395 | 5.853e-15 | 2.282 | 4.582 | - | - | - |
| 2 | 1.597e-05 | 0.01395 | 3.744e-15 | 2.034 | 4.582 | - | - | - |
| 3 | 1.53e-05 | 0.01395 | 2.585e-15 | 1.973 | 4.582 | - | - | - |
| 4 | 1.626e-05 | 0.01394 | 4.904e-15 | 2.085 | 4.582 | - | - | - |
| 5 | 1.62e-05 | 0.01394 | 3.914e-15 | 2.293 | 4.582 | 0.2442 | 1 | 0.005581 |
| 6 | 1.59e-05 | 0.01394 | 3.079e-15 | 1.929 | 4.582 | - | - | - |
| 7 | 1.428e-05 | 0.01394 | 2.173e-15 | 2.1 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 1.499e-05 | 0.01392 | 2.069e-15 | 2.252 | 4.581 | - | - | - |
| 193 | 1.709e-05 | 0.01392 | 1.261e-15 | 1.84 | 4.581 | - | - | - |
| 194 | 1.365e-05 | 0.01392 | 1.423e-15 | 2.211 | 4.581 | - | - | - |
| 195 | 1.573e-05 | 0.01392 | 3.102e-15 | 2.228 | 4.581 | 0.2097 | 1 | 0.007236 |
| 196 | 1.57e-05 | 0.01392 | 1.681e-15 | 1.966 | 4.581 | - | - | - |
| 197 | 1.342e-05 | 0.01392 | 1.534e-15 | 1.852 | 4.581 | - | - | - |
| 198 | 1.508e-05 | 0.01392 | 3.427e-15 | 2.193 | 4.581 | - | - | - |
| 199 | 1.452e-05 | 0.01392 | 3.04e-15 | 2.013 | 4.581 | 0.1967 | 1 | 0.01211 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
