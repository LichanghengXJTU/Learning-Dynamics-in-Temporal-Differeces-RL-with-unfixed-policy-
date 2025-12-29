# On-policy dist_mmd2 calibration

- on-policy: theta_mu == theta_pi, sigma_mu == sigma_pi == 0.2
- theta_init_scale: 0.1
- off-policy delta scale: 5

## On-policy dist_mmd2 stats
| num_samples | count | mean | std | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| 512 | 10 | 0.0391654 | 0.0121063 | 0.0545527 | 0.0549276 |

Recommended threshold (num_samples=512): threshold_reco = p99 * 1.2 = 0.0659132
Reason: add 20% headroom over the on-policy p99 to reduce false positives.

## Off-policy dist_mmd2 stats
| num_samples | count | mean | std | p05 | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 10 | 0.0389696 | 0.0131101 | 0.0213724 | 0.0394227 | 0.0566301 |

## Separation (on-policy p99 vs off-policy p05)
- num_samples=512: on_p99=0.0549276, off_p05=0.0213724, ratio=0.389101

