# Sweep Summary

## Sweep Info
- run_root: outputs/sweep/20251229_102740
- total_runs: 8
- offpolicy_stable: 0
- instability: 8
- plateau_drift: 0

## Grid (effective)
- beta_values: [0.005, 0.4]
- alpha_w_values: [0.02, 0.24]
- offpolicy_axis: sigma_mu
- offpolicy_values: [0.175, 0.7]

## Thresholds
- offpolicy_threshold: 0.001
- eps_slope: 1e-06
- eps_drift: 0.0001
- window: 20
- stability_eps: 0.001
- td_loss_blowup: 1000000.0
- w_norm_blowup: 10000.0

## Classification
- offpolicy_score = max(tracking_gap_p95, max(0, mean_rho2_p95 - 1), dist_action_kl_p95, dist_action_tv_p95)
- offpolicy_flag = offpolicy_score >= offpolicy_threshold
- offpolicy_stable bucket = offpolicy_flag AND health_status PASS AND not instability_flag
- instability_flag = health_checks fail OR incomplete OR non-finite OR stability_proxy > 1+stability_eps OR td_loss_max > td_loss_blowup OR w_norm_max > w_norm_blowup
- plateau_flag = health_status PASS AND not instability AND |td_loss_slope| < eps_slope AND drift_score > eps_drift

## Off-policy stable (top 0)
| run_id | beta | alpha_w | offpolicy_value | offpolicy_score | td_loss_last | w_norm_last | tracking_gap_last | mean_rho2_last | health_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Instability (top 8)
| run_id | beta | alpha_w | offpolicy_value | offpolicy_score | td_loss_last | w_norm_last | tracking_gap_last | mean_rho2_last | health_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0p005_aw0p02_sigma_mu0p175_s0_eb4a791b | 0.005 | 0.02 | 0.175 | 1.3648828910576438 | 1.5006752132538168e-05 | 4.5816598862111615 | 1.529111173778993e-12 | 2.0131317233576307 | PASS |
| b0p005_aw0p02_sigma_mu0p7_s0_013c6677 | 0.005 | 0.02 | 0.7 | 2.173097761314948 | 1.6163422770641704e-05 | 4.581656278035281 | 2.810498925491517e-12 | 3.1346005672116286 | PASS |
| b0p005_aw0p24_sigma_mu0p175_s0_92b1233d | 0.005 | 0.24 | 0.175 | 1.3648828909682478 | 1.451984441863901e-05 | 4.581428465304897 | 1.5237551103599163e-12 | 2.013131719392947 | PASS |
| b0p005_aw0p24_sigma_mu0p7_s0_433ed98c | 0.005 | 0.24 | 0.7 | 2.1730977615093074 | 1.562254542232203e-05 | 4.58139874024597 | 2.814245764075215e-12 | 3.1346005644539843 | PASS |
| b0p4_aw0p02_sigma_mu0p175_s0_5f7e73cd | 0.4 | 0.02 | 0.175 | 1.3648830994174554 | 1.5006751930071132e-05 | 4.58165988621011 | 3.1084141343898228e-15 | 2.0131315402996894 | PASS |
| b0p4_aw0p02_sigma_mu0p7_s0_8115b1ab | 0.4 | 0.02 | 0.7 | 2.173097837782597 | 1.6163421991386432e-05 | 4.581656278035907 | 1.297631474557065e-15 | 3.134600929685311 | PASS |
| b0p4_aw0p24_sigma_mu0p175_s0_286495a9 | 0.4 | 0.24 | 0.175 | 1.364883099322579 | 1.4519844211053673e-05 | 4.5814284652925625 | 3.0397565008259434e-15 | 2.013131540922987 | PASS |
| b0p4_aw0p24_sigma_mu0p7_s0_d8de6485 | 0.4 | 0.24 | 0.7 | 2.17309783780422 | 1.562254465613087e-05 | 4.5813987402498535 | 1.2678340856850287e-15 | 3.134600929859055 | PASS |

## Plateau drift (top 0)
| run_id | beta | alpha_w | offpolicy_value | offpolicy_score | td_loss_last | w_norm_last | tracking_gap_last | mean_rho2_last | health_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Outputs
- summary_csv: outputs/sweep/20251229_102740/summary/summary.csv
- buckets_dir: outputs/sweep/20251229_102740/summary/buckets