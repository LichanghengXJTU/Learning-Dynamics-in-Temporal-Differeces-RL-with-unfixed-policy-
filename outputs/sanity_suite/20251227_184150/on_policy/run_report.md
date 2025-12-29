# Run Report

## Run Info
- run_dir: /Users/enhuili/Desktop/Learning Dynamics in Temporal Differences Reinforcement Learning with Unfixed Policy/Learning-Dynamics-in-Temporal-Differeces-RL-with-unfixed-policy-/outputs/sanity_suite/20251227_184150/on_policy
- timestamp: 2025-12-27T18:41:52
- seed: 0
- key_hparams: outer_iters=5, horizon=50, gamma=0.95, alpha_w=0.1, alpha_pi=0.05, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, p_mix=0.1

## Health
- status: PASS (all checks passed)

## Core Metrics
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| td_loss | 0.0001756 | 3.742e-05 | 0.0001756 | 8.754e-05 | 2.331e-05 |
| w_norm | 1.548 | 1.548 | 1.548 | 1.548 | -7.368e-06 |
| mean_rho2 | 1 | 1 | 1 | 1 | 0 |
| tracking_gap | 0 | 0 | 0 | 0 | 0 |
| critic_teacher_error | 0.02238 | 0.02238 | 0.02239 | 0.02238 | -1.088e-06 |

## Probe Metrics
### distribution_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| action_samples | 64 | 64 | 64 | 64 | 0 |
| dist_action_kl | 0 | 0 | 0 | 0 | 0 |
| dist_action_tv | 0 | 0 | 0 | 0 | 0 |
| iter | 4 | 0 | 4 | 2 | 1 |
| mean_l2 | 0.346 | 0.346 | 0.8864 | 0.6042 | -0.1133 |
| mmd2 | 0.02035 | 0.02035 | 0.06438 | 0.03844 | -0.009162 |
| mmd_sigma | 2.367 | 2.345 | 2.367 | 2.358 | 0.0001506 |
| num_samples | 512 | 512 | 512 | 512 | 0 |

### fixed_point_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 4 | 0 | 4 | 2 | 1 |
| num_iters | 200 | 200 | 200 | 200 | 0 |
| tol | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 0 |
| w_gap | 0.02501 | 0.01433 | 0.02501 | 0.02108 | 0.001141 |
| w_sharp_drift | 0.01524 | 0 | 0.01868 | 0.0131 | 0.003377 |
| w_sharp_drift_defined | 1 | 0 | 1 | 0.8 | 0.2 |

### stability_probe
| metric | last | min | max | mean_last_k | slope_last_k |
| --- | --- | --- | --- | --- | --- |
| batch_size | 512 | 512 | 512 | 512 | 0 |
| iter | 4 | 0 | 4 | 2 | 1 |
| power_iters | 20 | 20 | 20 | 20 | 0 |
| stability_proxy | 1 | 1 | 1 | 1 | 1.339e-07 |

## Samples (Head)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.742e-05 | 0.02239 | 0 | 1 | 1.548 | 0.02052 | 1 | 0.06438 |
| 1 | 9.133e-05 | 0.02239 | 0 | 1 | 1.548 | 0.02156 | 1 | 0.03639 |
| 2 | 8.529e-05 | 0.02238 | 0 | 1 | 1.548 | 0.01433 | 1 | 0.03825 |
| 3 | 4.81e-05 | 0.02238 | 0 | 1 | 1.548 | 0.02398 | 1 | 0.03283 |
| 4 | 0.0001756 | 0.02238 | 0 | 1 | 1.548 | 0.02501 | 1 | 0.02035 |

## Samples (Tail)
| iter | td_loss | critic_teacher_error | tracking_gap | mean_rho2 | w_norm | fixed_point_gap | stability_proxy | dist_mmd2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.742e-05 | 0.02239 | 0 | 1 | 1.548 | 0.02052 | 1 | 0.06438 |
| 1 | 9.133e-05 | 0.02239 | 0 | 1 | 1.548 | 0.02156 | 1 | 0.03639 |
| 2 | 8.529e-05 | 0.02238 | 0 | 1 | 1.548 | 0.01433 | 1 | 0.03825 |
| 3 | 4.81e-05 | 0.02238 | 0 | 1 | 1.548 | 0.02398 | 1 | 0.03283 |
| 4 | 0.0001756 | 0.02238 | 0 | 1 | 1.548 | 0.02501 | 1 | 0.02035 |

## Next Steps
- no major rule-based issues detected; consider longer runs or new probes if results are inconclusive.
