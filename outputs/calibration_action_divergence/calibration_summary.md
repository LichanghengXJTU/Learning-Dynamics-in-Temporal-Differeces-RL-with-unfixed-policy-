# Action divergence calibration

- on-policy: theta_mu == theta_pi, sigma_mu == sigma_pi == 0.2
- theta_init_scale: 0.1
- off-policy delta scale: 3
- action_samples: 64

## On-policy stats
| num_samples | metric | count | mean | std | p95 | p99 | eps_floor | reco_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | dist_action_kl | 50 | 0 | 0 | 0 | 0 | 1e-06 | 1e-06 |
| 512 | dist_action_tv | 50 | 0 | 0 | 0 | 0 | 0.0001 | 0.0001 |

## Off-policy stats
| num_samples | metric | count | mean | std | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | dist_action_kl | 30 | 0.0397007 | 0.0130387 | 0.066456 | 0.0738987 |
| 512 | dist_action_tv | 30 | 0.100215 | 0.0164999 | 0.131963 | 0.140865 |

## Recommended thresholds (max(eps_floor, p99 * 1.2))
- num_samples=512: dist_action_kl(p99=0, eps_floor=1e-06, reco_threshold=1e-06), dist_action_tv(p99=0, eps_floor=0.0001, reco_threshold=0.0001)
