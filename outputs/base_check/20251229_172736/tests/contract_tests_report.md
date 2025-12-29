# Contract Tests Report

| test | status | error | threshold | details |
| --- | --- | --- | --- | --- |
| T1_score_autograd | PASS | 4.441e-16 | 1e-06 | actor_dim=5, action_dim=3 |
| T2_rho_ratio | PASS | 0 | 1e-06 | rho=31.19, log_rho=3.44 |
| T3_td_error | PASS | 1.11e-16 | 1e-07 | delta_code=-0.8278, delta_manual=-0.8278 |
| T4_critic_update_scaling | PASS | 0 | 1e-07 | scale_ratio=0.5, ratio_err=0 |
| T5_semi_gradient | PASS | 0 | 0 | numpy update; no autograd frameworks imported |
