# On-policy dist_mmd2 calibration

- on-policy: theta_mu == theta_pi, sigma_mu == sigma_pi == 0.2
- theta_init_scale: 0.1
- off-policy delta scale: 1

## On-policy dist_mmd2 stats
| num_samples | count | mean | std | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| 512 | 50 | 0.0384928 | 0.0110865 | 0.0572361 | 0.0652764 |
| 1024 | 50 | 0.0183991 | 0.00697082 | 0.032465 | 0.0382144 |

Recommended threshold (num_samples=512): threshold_reco = p99 * 1.2 = 0.0783317
Reason: add 20% headroom over the on-policy p99 to reduce false positives.

## Off-policy dist_mmd2 stats
| num_samples | count | mean | std | p05 | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 10 | 0.0387198 | 0.0117572 | 0.0222895 | 0.0391458 | 0.0540718 |
| 1024 | 10 | 0.0167858 | 0.00442333 | 0.00997032 | 0.0183958 | 0.0218162 |

## Separation (on-policy p99 vs off-policy p05)
- num_samples=512: on_p99=0.0652764, off_p05=0.0222895, ratio=0.341463
- num_samples=1024: on_p99=0.0382144, off_p05=0.00997032, ratio=0.260904

