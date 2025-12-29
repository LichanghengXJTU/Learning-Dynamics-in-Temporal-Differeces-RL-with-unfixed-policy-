# Run Report

## Run Info
- run_dir: outputs/base_check/20251230_004605/sweep/runs/b0p005_aw0p02_tmos0_s0_b9094ba9
- timestamp: 2025-12-30T01:26:26
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
| td_loss | 1.94e-05 | 1.493e-05 | 2.078e-05 | 1.838e-05 | 1.785e-07 |
| w_norm | 4.581 | 4.581 | 4.582 | 4.581 | -1.221e-06 |
| mean_rho2 | 1.07 | 1.057 | 1.104 | 1.075 | -0.003296 |
| tracking_gap | 3.14e-10 | 5.178e-14 | 3.14e-10 | 3.098e-10 | 2.396e-12 |
| critic_teacher_error | 0.01393 | 0.01393 | 0.01395 | 0.01393 | -5.938e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | 1.112e-12 |
| dist_action_tv | 0.1131 | 0.1125 | 0.1134 | 0.113 | -1.27e-05 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| mean_l2 | 0.3079 | 0.1185 | 0.417 | 0.276 | 0.005084 |
| mmd2 | 0.008532 | 0.002824 | 0.01428 | 0.007425 | 0.0001993 |
| mmd_sigma | 2.351 | 2.342 | 2.374 | 2.367 | -0.001163 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 1.076 | 1.061 | 1.093 | 1.073 | -0.0002065 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 4.195e-06 |
| rho_mean | 1 | 0.992 | 1.009 | 0.9991 | -0.0001092 |
| rho_min | 0.1205 | 0.01745 | 0.1205 | 0.08284 | 0.003545 |
| rho_p95 | 1.337 | 1.332 | 1.339 | 1.336 | 5.337e-05 |
| rho_p99 | 1.356 | 1.354 | 1.358 | 1.356 | 1.445e-05 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 1.079 | 1.057 | 1.091 | 1.077 | -7.519e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | -2.682e-07 |
| rho_mean | 1.001 | 0.9911 | 1.008 | 1.001 | -1.789e-05 |
| rho_min | 0.04007 | 0.02084 | 0.1118 | 0.08062 | -0.003774 |
| rho_p95 | 1.338 | 1.333 | 1.339 | 1.336 | 9.741e-05 |
| rho_p99 | 1.356 | 1.355 | 1.357 | 1.356 | 2.914e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.02796 | 0.02183 | 0.02965 | 0.02555 | 0.0004811 |
| w_sharp_drift | 0.005241 | 0 | 0.007857 | 0.005482 | 5.777e-06 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| td_loss | 1.94e-05 | 1.577e-05 | 2.058e-05 | 1.878e-05 | -1.929e-08 |
| td_loss_from_Q | 8.352e-06 | 7.29e-06 | 1.098e-05 | 8.987e-06 | -9.208e-08 |
| td_loss_from_Q_abs_diff | 1.105e-05 | 7.078e-06 | 1.105e-05 | 9.798e-06 | 7.279e-08 |
| td_loss_from_Q_rel_diff | 0.5694 | 0.4298 | 0.5694 | 0.521 | 0.004389 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 191.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 1.076 | 1.07 | 1.082 | 1.077 | 3.538e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 1 | 1 | 1 | 1 | 0 |
| rho_max | 1.361 | 1.361 | 1.361 | 1.361 | 1.887e-06 |
| rho_mean | 1 | 0.9973 | 1.003 | 1.001 | 7.089e-05 |
| rho_min | 0.01963 | 0.007272 | 0.05224 | 0.02667 | -0.001266 |
| rho_p95 | 1.335 | 1.335 | 1.338 | 1.336 | 2.901e-05 |
| rho_p99 | 1.356 | 1.355 | 1.357 | 1.356 | -4.49e-06 |
| stability_probe_step_scale | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.653e-07 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 1.653e-07 |
| stability_proxy_std | 1.596e-06 | 8.036e-07 | 5.434e-06 | 2.274e-06 | -3.312e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.817e-05 | 0.01395 | 5.178e-14 | 1.082 | 4.582 | - | - | - |
| 1 | 1.805e-05 | 0.01395 | 1.24e-13 | 1.075 | 4.582 | - | - | - |
| 2 | 1.896e-05 | 0.01395 | 1.911e-13 | 1.076 | 4.582 | - | - | - |
| 3 | 1.743e-05 | 0.01395 | 3.219e-13 | 1.084 | 4.582 | - | - | - |
| 4 | 1.898e-05 | 0.01395 | 4.405e-13 | 1.067 | 4.582 | - | - | - |
| 5 | 1.963e-05 | 0.01395 | 7.01e-13 | 1.068 | 4.582 | 0.02537 | 1 | 0.004019 |
| 6 | 1.762e-05 | 0.01395 | 9.425e-13 | 1.072 | 4.582 | - | - | - |
| 7 | 1.724e-05 | 0.01395 | 1.227e-12 | 1.086 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | 1.85e-05 | 0.01393 | 2.998e-10 | 1.065 | 4.581 | - | - | - |
| 193 | 1.844e-05 | 0.01393 | 3.019e-10 | 1.074 | 4.581 | - | - | - |
| 194 | 1.83e-05 | 0.01393 | 3.04e-10 | 1.08 | 4.581 | - | - | - |
| 195 | 1.811e-05 | 0.01393 | 3.054e-10 | 1.068 | 4.581 | 0.02878 | 1 | 0.009157 |
| 196 | 1.829e-05 | 0.01393 | 3.066e-10 | 1.104 | 4.581 | - | - | - |
| 197 | 1.858e-05 | 0.01393 | 3.098e-10 | 1.066 | 4.581 | - | - | - |
| 198 | 1.751e-05 | 0.01393 | 3.133e-10 | 1.068 | 4.581 | - | - | - |
| 199 | 1.94e-05 | 0.01393 | 3.14e-10 | 1.07 | 4.581 | 0.02796 | 1 | 0.008532 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
