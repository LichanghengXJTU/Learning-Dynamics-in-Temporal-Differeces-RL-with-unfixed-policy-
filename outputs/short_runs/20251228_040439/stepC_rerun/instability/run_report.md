# Run Report

## Run Info
- run_dir: outputs/short_runs/20251228_040439/stepC_rerun/instability
- timestamp: 2025-12-28T05:30:11
- seed: 0
- key_hparams: outer_iters=200, horizon=200, gamma=0.99, alpha_w=0.2, alpha_pi=0.12, beta=0.01, sigma_mu=0.25, sigma_pi=0.4, p_mix=0.01

## Health
- status: PASS (all checks passed)

## Scale Checks
- train_step_scale: 6.25e-05
- stability_probe_step_scale: 6.25e-05
- stability_probe_step_scale_ratio: 1.0 (expect ~1.0)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 2.585e-05 | 2.114e-05 | 2.819e-05 | 2.484e-05 | -2.949e-08 |
| w_norm | 9.163 | 9.163 | 9.163 | 9.163 | -2.315e-06 |
| mean_rho2 | 0.4315 | 0.4191 | 0.4365 | 0.4274 | 1.555e-06 |
| tracking_gap | 5.646e-12 | 1.586e-15 | 5.646e-12 | 5.614e-12 | 1.39e-14 |
| critic_teacher_error | 0.04453 | 0.04453 | 0.04455 | 0.04453 | -1.256e-07 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | 1.967e-14 |
| dist_action_tv | 0.3339 | 0.3331 | 0.3343 | 0.3337 | 6.509e-06 |
| iter | 199 | 5 | 199 | 189.4 | 1 |
| mean_l2 | 0.5386 | 0.2066 | 0.9258 | 0.4476 | 0.005345 |
| mmd2 | 0.02398 | 0.007235 | 0.06195 | 0.02039 | 0.0002972 |
| mmd_sigma | 2.356 | 2.285 | 2.372 | 2.336 | 0.0001521 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 189.4 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.1957 | 0.1694 | 0.2201 | 0.1966 | 0.000883 |
| w_sharp_drift | 0.04351 | 0 | 0.05478 | 0.03833 | -0.001376 |
| w_sharp_drift_defined | 1 | 0 | 1 | 0.8 | -0.0208 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 199 | 5 | 199 | 189.4 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_probe_step_scale | 6.25e-05 | 6.25e-05 | 6.25e-05 | 6.25e-05 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 2.792e-08 |
| stability_proxy_mean | 1 | 1 | 1 | 1 | 2.792e-08 |
| stability_proxy_std | 2.608e-06 | 1.145e-06 | 7.22e-06 | 3.138e-06 | -7.723e-08 |

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
| 192 | 2.479e-05 | 0.04453 | 5.456e-12 | 0.4272 | 9.163 | - | - | - |
| 193 | 2.333e-05 | 0.04453 | 5.524e-12 | 0.425 | 9.163 | - | - | - |
| 194 | 2.41e-05 | 0.04453 | 5.57e-12 | 0.4322 | 9.163 | 0.202 | 1 | 0.02533 |
| 195 | 2.541e-05 | 0.04453 | 5.586e-12 | 0.431 | 9.163 | - | - | - |
| 196 | 2.534e-05 | 0.04453 | 5.598e-12 | 0.4271 | 9.163 | - | - | - |
| 197 | 2.345e-05 | 0.04453 | 5.623e-12 | 0.4214 | 9.163 | - | - | - |
| 198 | 2.417e-05 | 0.04453 | 5.616e-12 | 0.4261 | 9.163 | - | - | - |
| 199 | 2.585e-05 | 0.04453 | 5.646e-12 | 0.4315 | 9.163 | 0.1957 | 1 | 0.02398 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
