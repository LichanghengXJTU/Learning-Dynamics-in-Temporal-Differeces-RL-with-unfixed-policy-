# Run Report

## Run Info
- run_dir: outputs/base_check/20251229_192835/stepC_v2/plateau
- timestamp: 2025-12-29T20:12:54
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.08, alpha_pi=0.06, beta=0.05, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 0.0001
- stability_probe_step_scale: 0.0001
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.863e-05 | 1.444e-05 | 2.04e-05 | 1.767e-05 | 1.628e-07 |
| w_norm | 4.581 | 4.581 | 4.582 | 4.581 | -3.7e-06 |
| mean_rho2 | 1.07 | 1.057 | 1.104 | 1.075 | -0.003296 |
| tracking_gap | 7.435e-12 | 4.72e-14 | 8.866e-12 | 7.328e-12 | 1.009e-13 |
| critic_teacher_error | 0.0139 | 0.0139 | 0.01395 | 0.0139 | -2.278e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | -1.302e-14 |
| dist_action_tv | 0.1131 | 0.1125 | 0.1134 | 0.113 | -1.27e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.3079 | 0.1185 | 0.417 | 0.276 | 0.005084 |
| mmd2 | 0.008532 | 0.002824 | 0.01428 | 0.007425 | 0.0001993 |
| mmd_sigma | 2.351 | 2.342 | 2.374 | 2.367 | -0.001164 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0002066 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 4.19e-06 |
| rho_mean | 1 | 0.992 | 1.009 | 0.9991 | -0.0001092 |
| rho_min | 0.1205 | 0.01745 | 0.1205 | 0.08284 | 0.003546 |
| rho_p95 | 1.337 | 1.332 | 1.339 | 1.336 | 5.353e-05 |
| rho_p99 | 1.356 | 1.354 | 1.358 | 1.356 | 1.456e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.057 | 1.091 | 1.077 | -7.518e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | -3.119e-07 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -1.789e-05 |
| rho_min | 0.04006 | 0.02084 | 0.1118 | 0.08062 | -0.003773 |
| rho_p95 | 1.338 | 1.333 | 1.339 | 1.336 | 9.726e-05 |
| rho_p99 | 1.356 | 1.355 | 1.357 | 1.356 | 2.942e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.1051 | 0.08305 | 0.1136 | 0.09643 | 0.001752 |
| w_sharp_drift | 0.02037 | 0 | 0.02944 | 0.02122 | 1.929e-05 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 1.863e-05 | 1.553e-05 | 1.995e-05 | 1.808e-05 | -2.467e-08 |
| td_loss_from_Q | 8.022e-06 | 7.224e-06 | 1.074e-05 | 8.65e-06 | -9.12e-08 |
| td_loss_from_Q_abs_diff | 1.061e-05 | 7.038e-06 | 1.077e-05 | 9.433e-06 | 6.652e-08 |
| td_loss_from_Q_rel_diff | 0.5694 | 0.4299 | 0.5694 | 0.5211 | 0.00438 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 3.535e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 1.877e-06 |
| rho_mean | 1 | 0.9973 | 1.003 | 1.001 | 7.088e-05 |
| rho_min | 0.01963 | 0.007273 | 0.05225 | 0.02667 | -0.001267 |
| rho_p95 | 1.335 | 1.335 | 1.338 | 1.336 | 2.917e-05 |
| rho_p99 | 1.356 | 1.355 | 1.357 | 1.356 | -4.442e-06 |
| stability_probe_step_scale | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0 |
| stability_proxy | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 4.645e-07 |
| stability_proxy_mean | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 4.645e-07 |
| stability_proxy_std | 5.133e-06 | 2.813e-06 | 1.473e-05 | 6.589e-06 | -8.597e-09 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.817e-05 | 0.01395 | 4.72e-14 | 1.082 | 4.582 | - | - | - |
| 1 | 1.805e-05 | 0.01395 | 1.076e-13 | 1.075 | 4.582 | - | - | - |
| 2 | 1.896e-05 | 0.01394 | 1.576e-13 | 1.076 | 4.582 | - | - | - |
| 3 | 1.742e-05 | 0.01394 | 2.56e-13 | 1.084 | 4.582 | - | - | - |
| 4 | 1.897e-05 | 0.01394 | 3.335e-13 | 1.067 | 4.582 | - | - | - |
| 5 | 1.961e-05 | 0.01394 | 5.158e-13 | 1.068 | 4.582 | 0.09803 | 0.9999 | 0.004019 |
| 6 | 1.76e-05 | 0.01394 | 6.637e-13 | 1.072 | 4.582 | - | - | - |
| 7 | 1.721e-05 | 0.01394 | 8.288e-13 | 1.086 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 1.779e-05 | 0.0139 | 7.19e-12 | 1.065 | 4.581 | - | - | - |
| 193 | 1.777e-05 | 0.0139 | 7.203e-12 | 1.074 | 4.581 | - | - | - |
| 194 | 1.764e-05 | 0.0139 | 7.235e-12 | 1.08 | 4.581 | - | - | - |
| 195 | 1.742e-05 | 0.0139 | 7.171e-12 | 1.068 | 4.581 | 0.1083 | 0.9999 | 0.009157 |
| 196 | 1.763e-05 | 0.0139 | 7.104e-12 | 1.104 | 4.581 | - | - | - |
| 197 | 1.784e-05 | 0.0139 | 7.344e-12 | 1.066 | 4.581 | - | - | - |
| 198 | 1.683e-05 | 0.0139 | 7.585e-12 | 1.068 | 4.581 | - | - | - |
| 199 | 1.863e-05 | 0.0139 | 7.435e-12 | 1.07 | 4.581 | 0.1051 | 0.9999 | 0.008532 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
