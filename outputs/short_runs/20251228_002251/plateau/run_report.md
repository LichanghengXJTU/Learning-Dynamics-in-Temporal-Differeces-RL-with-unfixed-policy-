# Run Report

## Run Info
- run_dir: outputs/short_runs/20251228_002251/plateau
- timestamp: 2025-12-28T00:25:05
- seed: 0
- key_hparams: outer_iters=80, horizon=200, gamma=0.99, alpha_w=0.08, alpha_pi=0.06, beta=0.05, sigma_mu=0.35, sigma_pi=0.3, p_mix=0.05

## Health
- status: PASS (all checks passed)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 1.548e-05 | 1.358e-05 | 1.941e-05 | 1.658e-05 | -1.884e-07 |
| w_norm | 4.582 | 4.582 | 4.582 | 4.582 | -1.127e-06 |
| mean_rho2 | 1.331 | 1.319 | 1.337 | 1.329 | 0.001279 |
| tracking_gap | 2.435e-13 | 1.928e-15 | 2.65e-13 | 2.369e-13 | -7.73e-16 |
| critic_teacher_error | 0.01394 | 0.01394 | 0.01395 | 0.01394 | -6.551e-08 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | - |
| dist_action_kl | 0.043 | 0.043 | 0.043 | 0.043 | - |
| dist_action_tv | 0.113 | 0.113 | 0.113 | 0.113 | - |
| iter | 49 | 49 | 49 | 49 | - |
| mean_l2 | 0.3155 | 0.3155 | 0.3155 | 0.3155 | - |
| mmd2 | 0.008802 | 0.008802 | 0.008802 | 0.008802 | - |
| mmd_sigma | 2.36 | 2.36 | 2.36 | 2.36 | - |
| num_samples | 4096 | 4096 | 4096 | 4096 | - |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | - |
| iter | 49 | 49 | 49 | 49 | - |
| num_iters | 2000 | 2000 | 2000 | 2000 | - |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 | - |
| w_gap | 0.1186 | 0.1186 | 0.1186 | 0.1186 | - |
| w_sharp_drift | 0 | 0 | 0 | 0 | - |
| w_sharp_drift_defined | 0 | 0 | 0 | 0 | - |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 4096 | 4096 | 4096 | 4096 | - |
| iter | 49 | 49 | 49 | 49 | - |
| power_iters | 20 | 20 | 20 | 20 | - |
| stability_proxy | 1 | 1 | 1 | 1 | - |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.688e-05 | 0.01395 | 1.928e-15 | 1.329 | 4.582 | - | - | - |
| 1 | 1.648e-05 | 0.01395 | 4.481e-15 | 1.326 | 4.582 | - | - | - |
| 2 | 1.879e-05 | 0.01395 | 6.773e-15 | 1.327 | 4.582 | - | - | - |
| 3 | 1.609e-05 | 0.01395 | 1.388e-14 | 1.332 | 4.582 | - | - | - |
| 4 | 1.701e-05 | 0.01395 | 1.889e-14 | 1.326 | 4.582 | - | - | - |
| 5 | 1.812e-05 | 0.01395 | 2.082e-14 | 1.325 | 4.582 | - | - | - |
| 6 | 1.61e-05 | 0.01395 | 2.366e-14 | 1.323 | 4.582 | - | - | - |
| 7 | 1.569e-05 | 0.01395 | 2.792e-14 | 1.329 | 4.582 | - | - | - |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 1.681e-05 | 0.01394 | 2.65e-13 | 1.323 | 4.582 | - | - | - |
| 73 | 1.73e-05 | 0.01394 | 2.572e-13 | 1.331 | 4.582 | - | - | - |
| 74 | 1.614e-05 | 0.01394 | 2.503e-13 | 1.327 | 4.582 | - | - | - |
| 75 | 1.617e-05 | 0.01394 | 2.501e-13 | 1.321 | 4.582 | - | - | - |
| 76 | 1.682e-05 | 0.01394 | 2.277e-13 | 1.335 | 4.582 | - | - | - |
| 77 | 1.813e-05 | 0.01394 | 2.301e-13 | 1.329 | 4.582 | - | - | - |
| 78 | 1.631e-05 | 0.01394 | 2.333e-13 | 1.328 | 4.582 | - | - | - |
| 79 | 1.548e-05 | 0.01394 | 2.435e-13 | 1.331 | 4.582 | - | - | - |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
