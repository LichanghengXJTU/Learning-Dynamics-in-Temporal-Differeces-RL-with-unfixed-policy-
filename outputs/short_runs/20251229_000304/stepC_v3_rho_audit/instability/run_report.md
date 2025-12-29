# Run Report

## Run Info
- run_dir: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/instability
- timestamp: 2025-12-29T00:38:30
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
| td_loss | 2.405e-05 | 2.154e-05 | 2.819e-05 | 2.549e-05 | 2.072e-07 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -2.762e-06 |
| mean_rho2 | 0.4259 | 0.4191 | 0.4365 | 0.4266 | -0.001011 |
| tracking_gap | 2.766e-12 | 1.586e-15 | 2.766e-12 | 2.688e-12 | 3.728e-14 |
| critic_teacher_error | 0.04454 | 0.04454 | 0.04455 | 0.04454 | -1.377e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | 6.448e-14 |
| dist_action_tv | 0.3338 | 0.3331 | 0.334 | 0.3338 | -7.575e-06 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| mean_l2 | 0.6243 | 0.2645 | 0.9258 | 0.5792 | 0.01035 |
| mmd2 | 0.03143 | 0.01104 | 0.06195 | 0.02959 | 0.0007143 |
| mmd_sigma | 2.372 | 2.309 | 2.372 | 2.351 | 0.003094 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |
| rho2_mean | 0.4297 | 0.4222 | 0.4332 | 0.4275 | 0.0003505 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 0.9446 | 0.9308 | 0.9613 | 0.9484 | -0.0004196 |
| rho_mean | 0.6319 | 0.6259 | 0.6353 | 0.6303 | 0.0002729 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | 1.287e-06 |
| rho_p95 | 0.8805 | 0.8764 | 0.8836 | 0.8797 | 7.778e-05 |
| rho_p99 | 0.9047 | 0.9011 | 0.9116 | 0.9068 | -0.0004006 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| rho2_mean | 0.426 | 0.4223 | 0.4367 | 0.4275 | 4.307e-05 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 0.9305 | 0.9305 | 0.96 | 0.9379 | -0.0005062 |
| rho_mean | 0.6291 | 0.6265 | 0.6376 | 0.6303 | 2.674e-05 |
| rho_min | 0.3907 | 0.3906 | 0.3908 | 0.3907 | -6.493e-06 |
| rho_p95 | 0.8797 | 0.875 | 0.8833 | 0.8788 | 0.0001482 |
| rho_p99 | 0.9017 | 0.9017 | 0.911 | 0.9044 | -8.376e-05 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.21 | 0.1801 | 0.2201 | 0.201 | 0.001276 |
| w_sharp_drift | 0.03875 | 0 | 0.05437 | 0.04746 | -0.0005009 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### q_kernel_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| cache_batch_size | 8 | 8 | 8 | 8 | 0 |
| cache_horizon | 200 | 200 | 200 | 200 | 0 |
| cache_valid_t | 200 | 200 | 200 | 200 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| td_loss | 2.405e-05 | 2.154e-05 | 2.705e-05 | 2.446e-05 | -2.726e-08 |
| td_loss_from_Q | 1.19e-05 | 1.035e-05 | 1.361e-05 | 1.194e-05 | -6.614e-08 |
| td_loss_from_Q_abs_diff | 1.215e-05 | 9.96e-06 | 1.387e-05 | 1.252e-05 | 3.888e-08 |
| td_loss_from_Q_rel_diff | 0.5053 | 0.4624 | 0.5409 | 0.5122 | 0.002107 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 5 | 79 | 71.6 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| rho2_mean | 0.427 | 0.4246 | 0.4302 | 0.4283 | -0.000224 |
| rho_clip | 10 | 10 | 10 | 10 | 0 |
| rho_clip_active | 0 | 0 | 0 | 0 | 0 |
| rho_max | 0.9608 | 0.9496 | 0.9688 | 0.9572 | 5.293e-05 |
| rho_mean | 0.6299 | 0.628 | 0.6325 | 0.631 | -0.0001823 |
| rho_min | 0.3906 | 0.3906 | 0.3906 | 0.3906 | -7.968e-08 |
| rho_p95 | 0.8797 | 0.8778 | 0.8815 | 0.88 | -2.45e-05 |
| rho_p99 | 0.9049 | 0.9032 | 0.9077 | 0.9057 | -8.13e-05 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 3.176e-09 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 3.176e-09 |
| stability_proxy_std | 4.052e-06 | 1.169e-06 | 6.119e-06 | 2.731e-06 | 7.097e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5e-05 | 0.04455 | 1.586e-15 | 0.4245 | 9.163 | - | - | - |
| 1 | 2.532e-05 | 0.04455 | 4.881e-15 | 0.4263 | 9.163 | - | - | - |
| 2 | 2.541e-05 | 0.04455 | 8.559e-15 | 0.4272 | 9.163 | - | - | - |
| 3 | 2.386e-05 | 0.04455 | 1.488e-14 | 0.425 | 9.163 | - | - | - |
| 4 | 2.465e-05 | 0.04455 | 2.338e-14 | 0.4322 | 9.163 | - | - | - |
| 5 | 2.602e-05 | 0.04455 | 3.052e-14 | 0.431 | 9.163 | 0.19 | 1 | 0.01245 |
| 6 | 2.595e-05 | 0.04455 | 4.112e-14 | 0.4271 | 9.163 | - | - | - |
| 7 | 2.402e-05 | 0.04455 | 5.324e-14 | 0.4214 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 2.469e-05 | 0.04454 | 2.525e-12 | 0.4351 | 9.163 | - | - | - |
| 73 | 2.74e-05 | 0.04454 | 2.544e-12 | 0.4274 | 9.163 | - | - | - |
| 74 | 2.463e-05 | 0.04454 | 2.599e-12 | 0.4284 | 9.163 | - | - | - |
| 75 | 2.482e-05 | 0.04454 | 2.607e-12 | 0.4345 | 9.163 | 0.1941 | 1 | 0.03732 |
| 76 | 2.398e-05 | 0.04454 | 2.67e-12 | 0.4194 | 9.163 | - | - | - |
| 77 | 2.7e-05 | 0.04454 | 2.675e-12 | 0.4266 | 9.163 | - | - | - |
| 78 | 2.761e-05 | 0.04454 | 2.724e-12 | 0.4265 | 9.163 | - | - | - |
| 79 | 2.405e-05 | 0.04454 | 2.766e-12 | 0.4259 | 9.163 | 0.21 | 1 | 0.03143 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
