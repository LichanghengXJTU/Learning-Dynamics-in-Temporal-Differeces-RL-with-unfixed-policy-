# Run Report

## Run Info
- run_dir: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_173439/no_bootstrap
- timestamp: 2025-12-27T17:34:44
- seed: 0
- key_hparams: outer_iters=5, horizon=50, gamma=0.0, alpha_w=0.1, alpha_pi=0.05, beta=0.2, sigma_mu=0.3, sigma_pi=0.2, p_mix=0.1

## Health
- status: PASS (all checks passed)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 0.0001647 | 6.76e-05 | 0.0001647 | 0.0001018 | 1.764e-05 |
| w_norm | 1.548 | 1.548 | 1.548 | 1.548 | -3.296e-05 |
| mean_rho2 | 1.48 | 1.429 | 1.78 | 1.607 | -0.04446 |
| tracking_gap | 8.499e-11 | 2.871e-11 | 8.499e-11 | 5.544e-11 | 1.122e-11 |
| critic_teacher_error | 0.02238 | 0.02238 | 0.02239 | 0.02238 | -1.493e-06 |

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
| w_gap | 0.03235 | 0.02369 | 0.03557 | 0.02959 | -0.0001689 |
| w_sharp_drift | 0.02956 | 0 | 0.03002 | 0.02283 | 0.005507 |
| w_sharp_drift_defined | 1 | 0 | 1 | 0.8 | 0.2 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 4 | 0 | 4 | 2 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | -1.68e-06 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6.76e-05 | 0.02239 | 2.871e-11 | 1.78 | 1.548 | 0.02906 | 1 | 0.05279 |
| 1 | 9.725e-05 | 0.02238 | 4.719e-11 | 1.429 | 1.548 | 0.03557 | 1 | 0.04285 |
| 2 | 9.986e-05 | 0.02238 | 6.95e-11 | 1.76 | 1.548 | 0.02369 | 1 | 0.04147 |
| 3 | 7.934e-05 | 0.02238 | 4.681e-11 | 1.585 | 1.548 | 0.02731 | 1 | 0.0309 |
| 4 | 0.0001647 | 0.02238 | 8.499e-11 | 1.48 | 1.548 | 0.03235 | 1 | 0.01805 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6.76e-05 | 0.02239 | 2.871e-11 | 1.78 | 1.548 | 0.02906 | 1 | 0.05279 |
| 1 | 9.725e-05 | 0.02238 | 4.719e-11 | 1.429 | 1.548 | 0.03557 | 1 | 0.04285 |
| 2 | 9.986e-05 | 0.02238 | 6.95e-11 | 1.76 | 1.548 | 0.02369 | 1 | 0.04147 |
| 3 | 7.934e-05 | 0.02238 | 4.681e-11 | 1.585 | 1.548 | 0.02731 | 1 | 0.0309 |
| 4 | 0.0001647 | 0.02238 | 8.499e-11 | 1.48 | 1.548 | 0.03235 | 1 | 0.01805 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
