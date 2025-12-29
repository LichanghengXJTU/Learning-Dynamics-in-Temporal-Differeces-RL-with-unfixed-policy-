# Run Report

## Run Info
- run_dir: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_181544/fixed_pi
- timestamp: 2025-12-27T18:15:50
- seed: 0
- key_hparams: outer_iters=5, horizon=50, gamma=0.95, alpha_w=0.1, alpha_pi=0.0, beta=0.2, sigma_mu=0.3, sigma_pi=0.2, p_mix=0.1

## Health
- status: PASS (all checks passed)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 0.0001515 | 3.987e-05 | 0.0001515 | 8.688e-05 | 1.772e-05 |
| w_norm | 1.548 | 1.548 | 1.548 | 1.548 | -7.561e-06 |
| mean_rho2 | 1.48 | 1.429 | 1.78 | 1.607 | -0.04446 |
| tracking_gap | 2.059e-34 | 1.705e-34 | 2.059e-34 | 1.988e-34 | 7.081e-36 |
| critic_teacher_error | 0.02238 | 0.02238 | 0.02239 | 0.02238 | -1.186e-06 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| iter | 4 | 0 | 4 | 2 | 1 |
| mean_l2 | 0.3205 | 0.3205 | 0.7841 | 0.5959 | -0.1073 |
| mmd2 | 0.01805 | 0.01805 | 0.05279 | 0.03721 | -0.008141 |
| mmd_sigma | 2.371 | 2.354 | 2.371 | 2.359 | 0.00341 |
| num_samples | 512 | 512 | 512 | 512 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 4 | 0 | 4 | 2 | 1 |
| num_iters | 200 | 200 | 200 | 200 | 0 |
| tol | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 0 |
| w_gap | 0.02161 | 0.01602 | 0.02681 | 0.02207 | -0.001264 |
| w_sharp_drift | 0.01721 | 0 | 0.0212 | 0.01521 | 0.003241 |
| w_sharp_drift_defined | 1 | 0 | 1 | 0.8 | 0.2 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 4 | 0 | 4 | 2 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.934e-07 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.987e-05 | 0.02239 | 1.705e-34 | 1.78 | 1.548 | 0.02499 | 1 | 0.05279 |
| 1 | 0.0001023 | 0.02238 | 2.059e-34 | 1.429 | 1.548 | 0.02681 | 1 | 0.04285 |
| 2 | 8.443e-05 | 0.02238 | 2.059e-34 | 1.76 | 1.548 | 0.01602 | 1 | 0.04147 |
| 3 | 5.625e-05 | 0.02238 | 2.059e-34 | 1.585 | 1.548 | 0.02093 | 1 | 0.0309 |
| 4 | 0.0001515 | 0.02238 | 2.059e-34 | 1.48 | 1.548 | 0.02161 | 1 | 0.01805 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.987e-05 | 0.02239 | 1.705e-34 | 1.78 | 1.548 | 0.02499 | 1 | 0.05279 |
| 1 | 0.0001023 | 0.02238 | 2.059e-34 | 1.429 | 1.548 | 0.02681 | 1 | 0.04285 |
| 2 | 8.443e-05 | 0.02238 | 2.059e-34 | 1.76 | 1.548 | 0.01602 | 1 | 0.04147 |
| 3 | 5.625e-05 | 0.02238 | 2.059e-34 | 1.585 | 1.548 | 0.02093 | 1 | 0.0309 |
| 4 | 0.0001515 | 0.02238 | 2.059e-34 | 1.48 | 1.548 | 0.02161 | 1 | 0.01805 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
