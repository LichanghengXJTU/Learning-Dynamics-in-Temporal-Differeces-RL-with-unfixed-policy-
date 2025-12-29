# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0p05556_s0_eb49491a
- timestamp: 2025-12-30T02:06:45
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.02, alpha_pi=0.06, beta=0.005, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 2.5e-05
- stability_probe_step_scale: 2.5e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.889e-05 | 2.089e-05 | 3.139e-05 | 2.694e-05 | 3.248e-07 |
| w_norm | 4.53 | 4.53 | 4.53 | 4.53 | -4.034e-06 |
| mean_rho2 | 1.07 | 1.056 | 1.104 | 1.075 | -0.003241 |
| tracking_gap | 0.0008657 | 0.0008657 | 0.006365 | 0.0008834 | -8.855e-06 |
| critic_teacher_error | 0.01398 | 0.01398 | 0.01399 | 0.01398 | -8.341e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.04303 | 0.043 | -4.602e-08 |
| dist_action_tv | 0.1131 | 0.1125 | 0.1135 | 0.1131 | -1.248e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.2522 | 0.1379 | 0.4685 | 0.2696 | 0.002464 |
| mmd2 | 0.00648 | 0.00258 | 0.01743 | 0.007207 | 9.573e-05 |
| mmd_sigma | 2.361 | 2.341 | 2.375 | 2.366 | -0.000149 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0001934 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 9.887e-06 |
| rho_mean | 1 | 0.992 | 1.009 | 0.9991 | -0.0001026 |
| rho_min | 0.1195 | 0.01712 | 0.1195 | 0.0828 | 0.003429 |
| rho_p95 | 1.337 | 1.332 | 1.34 | 1.336 | 4.821e-05 |
| rho_p99 | 1.355 | 1.354 | 1.358 | 1.356 | 2.885e-06 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.057 | 1.091 | 1.077 | -6.575e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.362 | 1.361 | -7.521e-06 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -1.526e-05 |
| rho_min | 0.0405 | 0.0209 | 0.1102 | 0.08037 | -0.003677 |
| rho_p95 | 1.338 | 1.333 | 1.339 | 1.335 | 0.0001229 |
| rho_p99 | 1.356 | 1.354 | 1.357 | 1.355 | 1.713e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.03303 | 0.02434 | 0.0339 | 0.03146 | 0.0002894 |
| w_sharp_drift | 0.005975 | 0 | 0.009375 | 0.006043 | 2.061e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 2.889e-05 | 2.208e-05 | 3.051e-05 | 2.755e-05 | 4.469e-08 |
| td_loss_from_Q | 1.254e-05 | 9.644e-06 | 1.671e-05 | 1.305e-05 | -6.975e-08 |
| td_loss_from_Q_abs_diff | 1.635e-05 | 1.038e-05 | 1.635e-05 | 1.45e-05 | 1.144e-07 |
| td_loss_from_Q_rel_diff | 0.566 | 0.4253 | 0.5785 | 0.5258 | 0.003209 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 3.253e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.362 | 1.361 | -1.363e-06 |
| rho_mean | 1 | 0.9973 | 1.003 | 1.001 | 6.921e-05 |
| rho_min | 0.01973 | 0.007316 | 0.05278 | 0.0271 | -0.001337 |
| rho_p95 | 1.336 | 1.335 | 1.338 | 1.336 | 4.064e-05 |
| rho_p99 | 1.356 | 1.356 | 1.357 | 1.356 | 9.065e-06 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.319e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.319e-07 |
| stability_proxy_std | 1.644e-06 | 5.645e-07 | 5.046e-06 | 2.148e-06 | -1.825e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.675e-05 | 0.01399 | 0.006365 | 1.081 | 4.53 | - | - | - |
| 1 | 2.582e-05 | 0.01399 | 0.006301 | 1.076 | 4.53 | - | - | - |
| 2 | 2.792e-05 | 0.01399 | 0.006238 | 1.076 | 4.53 | - | - | - |
| 3 | 2.574e-05 | 0.01399 | 0.006176 | 1.084 | 4.53 | - | - | - |
| 4 | 2.81e-05 | 0.01399 | 0.006115 | 1.067 | 4.53 | - | - | - |
| 5 | 2.91e-05 | 0.01399 | 0.006054 | 1.067 | 4.53 | 0.03128 | 1 | 0.00426 |
| 6 | 2.735e-05 | 0.01399 | 0.005993 | 1.072 | 4.53 | - | - | - |
| 7 | 2.516e-05 | 0.01399 | 0.005933 | 1.086 | 4.53 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 2.665e-05 | 0.01398 | 0.0009287 | 1.065 | 4.53 | - | - | - |
| 193 | 2.961e-05 | 0.01398 | 0.0009194 | 1.074 | 4.53 | - | - | - |
| 194 | 2.706e-05 | 0.01398 | 0.0009102 | 1.08 | 4.53 | - | - | - |
| 195 | 2.69e-05 | 0.01398 | 0.0009012 | 1.068 | 4.53 | 0.03361 | 1 | 0.009533 |
| 196 | 2.61e-05 | 0.01398 | 0.0008922 | 1.104 | 4.53 | - | - | - |
| 197 | 2.742e-05 | 0.01398 | 0.0008833 | 1.066 | 4.53 | - | - | - |
| 198 | 2.537e-05 | 0.01398 | 0.0008745 | 1.068 | 4.53 | - | - | - |
| 199 | 2.889e-05 | 0.01398 | 0.0008657 | 1.07 | 4.53 | 0.03303 | 1 | 0.00648 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
