# Base-Check Run Report

## Config
- N=None, N_act=None, B=1, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.0, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, seed=66

## Key Metrics (start -> end, trend)
- td_loss_est: 0.0003476 -> 0.0001819 (flat)
- w_dot_wr_over_n: 0.000933 -> 0.001181 (up)
- cos_w_wr: 0.05625 -> 0.07143 (up)
- mean_rho2: 1 -> 1 (flat)
- tracking_gap: 0 -> 0 (flat)

## Learning Evidence
- td_loss_down: True
- w_alignment_up: True
- rho_finite: True
- evidence_pass: True

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

