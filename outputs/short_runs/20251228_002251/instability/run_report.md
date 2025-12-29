# Run Report

## Run Info
- run_dir: outputs/short_runs/20251228_002251/instability
- timestamp: 2025-12-28T00:28:10
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.2, alpha_pi=0.12, beta=0.01, sigma_mu=0.25, sigma_pi=0.4, p_mix=0.01

## Health
- status: PASS (all checks passed)

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
| dist_action_kl | 0.62 | 0.62 | 0.62 | 0.62 | 5.816e-14 |
| dist_action_tv | 0.3338 | 0.3337 | 0.3338 | 0.3337 | 4.249e-07 |
| iter | 79 | 19 | 79 | 49 | 1 |
| mean_l2 | 0.6243 | 0.4908 | 0.8016 | 0.6156 | 0.0007232 |
| mmd2 | 0.03143 | 0.02147 | 0.05181 | 0.03254 | 1.758e-05 |
| mmd_sigma | 2.372 | 2.329 | 2.372 | 2.35 | 0.0007317 |
| num_samples | 4096 | 4096 | 4096 | 4096 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 19 | 79 | 49 | 1 |
| num_iters | 2000 | 2000 | 2000 | 2000 | 0 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | 0 |
| w_gap | 0.1875 | 0.1875 | 0.2004 | 0.1952 | -0.0001194 |
| w_sharp_drift | 0.04711 | 0 | 0.04711 | 0.0324 | 0.0007119 |
| w_sharp_drift_defined | 1 | 0 | 1 | 0.75 | 0.015 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | 0 |
| iter | 79 | 19 | 79 | 49 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -1.554e-09 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5e-05 | 0.04455 | 1.586e-15 | 0.4245 | 9.163 | - | - | - |
| 1 | 2.532e-05 | 0.04455 | 4.881e-15 | 0.4263 | 9.163 | - | - | - |
| 2 | 2.541e-05 | 0.04455 | 8.559e-15 | 0.4272 | 9.163 | - | - | - |
| 3 | 2.386e-05 | 0.04455 | 1.488e-14 | 0.425 | 9.163 | - | - | - |
| 4 | 2.465e-05 | 0.04455 | 2.338e-14 | 0.4322 | 9.163 | - | - | - |
| 5 | 2.602e-05 | 0.04455 | 3.052e-14 | 0.431 | 9.163 | - | - | - |
| 6 | 2.595e-05 | 0.04455 | 4.112e-14 | 0.4271 | 9.163 | - | - | - |
| 7 | 2.402e-05 | 0.04455 | 5.324e-14 | 0.4214 | 9.163 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 2.469e-05 | 0.04454 | 2.525e-12 | 0.4351 | 9.163 | - | - | - |
| 73 | 2.74e-05 | 0.04454 | 2.544e-12 | 0.4274 | 9.163 | - | - | - |
| 74 | 2.463e-05 | 0.04454 | 2.599e-12 | 0.4284 | 9.163 | - | - | - |
| 75 | 2.482e-05 | 0.04454 | 2.607e-12 | 0.4345 | 9.163 | - | - | - |
| 76 | 2.398e-05 | 0.04454 | 2.67e-12 | 0.4194 | 9.163 | - | - | - |
| 77 | 2.7e-05 | 0.04454 | 2.675e-12 | 0.4266 | 9.163 | - | - | - |
| 78 | 2.761e-05 | 0.04454 | 2.724e-12 | 0.4265 | 9.163 | - | - | - |
| 79 | 2.405e-05 | 0.04454 | 2.766e-12 | 0.4259 | 9.163 | 0.1875 | 1 | 0.03143 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
