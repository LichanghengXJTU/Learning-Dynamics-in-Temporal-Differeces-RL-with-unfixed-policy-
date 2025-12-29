# Run Report

## Run Info
- run_dir: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_181544/full_triad_short
- timestamp: 2025-12-27T18:15:52
- seed: 0
- key_hparams: outer_iters=8, horizon=50, gamma=0.95, alpha_w=0.1, alpha_pi=0.05, beta=0.2, sigma_mu=0.3, sigma_pi=0.2, p_mix=0.1

## Health
- status: PASS (all checks passed)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 8.235e-05 | 2.704e-05 | 0.0001515 | 7.431e-05 | -4.493e-06 |
| w_norm | 1.548 | 1.548 | 1.548 | 1.548 | -1.026e-05 |
| mean_rho2 | 1.67 | 1.429 | 1.78 | 1.571 | 0.02724 |
| tracking_gap | 6.382e-11 | 1.044e-11 | 7.946e-11 | 5.872e-11 | 9.029e-13 |
| critic_teacher_error | 0.02238 | 0.02238 | 0.02239 | 0.02238 | -1.134e-06 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| iter | 7 | 0 | 7 | 5 | 1 |
| mean_l2 | 0.6361 | 0.3205 | 0.7841 | 0.4698 | 0.0403 |
| mmd2 | 0.03616 | 0.01805 | 0.05279 | 0.02742 | 0.002013 |
| mmd_sigma | 2.346 | 2.345 | 2.371 | 2.354 | -0.003695 |
| num_samples | 512 | 512 | 512 | 512 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 7 | 0 | 7 | 5 | 1 |
| num_iters | 200 | 200 | 200 | 200 | 0 |
| tol | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 0 |
| w_gap | 0.02922 | 0.01602 | 0.02922 | 0.02273 | 0.001829 |
| w_sharp_drift | 0.02569 | 0 | 0.02569 | 0.01951 | 0.001744 |
| w_sharp_drift_defined | 1 | 0 | 1 | 1 | 0 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 7 | 0 | 7 | 5 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 8.571e-08 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.987e-05 | 0.02239 | 1.044e-11 | 1.78 | 1.548 | 0.02499 | 1 | 0.05279 |
| 1 | 0.0001023 | 0.02238 | 4.106e-11 | 1.429 | 1.548 | 0.02681 | 1 | 0.04285 |
| 2 | 8.443e-05 | 0.02238 | 6.344e-11 | 1.76 | 1.548 | 0.01602 | 1 | 0.04147 |
| 3 | 5.625e-05 | 0.02238 | 4.525e-11 | 1.585 | 1.548 | 0.02093 | 1 | 0.0309 |
| 4 | 0.0001515 | 0.02238 | 7.946e-11 | 1.48 | 1.548 | 0.02161 | 1 | 0.01805 |
| 5 | 2.704e-05 | 0.02238 | 5.373e-11 | 1.537 | 1.548 | 0.01857 | 1 | 0.02432 |
| 6 | 5.44e-05 | 0.02238 | 5.136e-11 | 1.581 | 1.548 | 0.02331 | 1 | 0.02768 |
| 7 | 8.235e-05 | 0.02238 | 6.382e-11 | 1.67 | 1.548 | 0.02922 | 1 | 0.03616 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.987e-05 | 0.02239 | 1.044e-11 | 1.78 | 1.548 | 0.02499 | 1 | 0.05279 |
| 1 | 0.0001023 | 0.02238 | 4.106e-11 | 1.429 | 1.548 | 0.02681 | 1 | 0.04285 |
| 2 | 8.443e-05 | 0.02238 | 6.344e-11 | 1.76 | 1.548 | 0.01602 | 1 | 0.04147 |
| 3 | 5.625e-05 | 0.02238 | 4.525e-11 | 1.585 | 1.548 | 0.02093 | 1 | 0.0309 |
| 4 | 0.0001515 | 0.02238 | 7.946e-11 | 1.48 | 1.548 | 0.02161 | 1 | 0.01805 |
| 5 | 2.704e-05 | 0.02238 | 5.373e-11 | 1.537 | 1.548 | 0.01857 | 1 | 0.02432 |
| 6 | 5.44e-05 | 0.02238 | 5.136e-11 | 1.581 | 1.548 | 0.02331 | 1 | 0.02768 |
| 7 | 8.235e-05 | 0.02238 | 6.382e-11 | 1.67 | 1.548 | 0.02922 | 1 | 0.03616 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
