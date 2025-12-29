# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.0, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, seed=77

## Key Metrics (start -> end, trend)
- td_loss_est: 1.585e-05 -> 0.0001069 (flat)
- w_dot_wr_over_n: -0.0005251 -> 4.133e-05 (up)
- cos_w_wr: -0.03833 -> 0.003046 (up)
- mean_rho2: 1 -> 1 (flat)
- tracking_gap: 0 -> 0 (flat)

## Learning Evidence
- td_loss_down: False
- w_alignment_up: True
- rho_finite: True
- evidence_pass: False

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

