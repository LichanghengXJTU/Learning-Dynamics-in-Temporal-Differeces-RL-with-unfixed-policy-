# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.1, beta=1.0, sigma_mu=0.2, sigma_pi=0.2, seed=11

## Key Metrics (start -> end, trend)
- td_loss_est: 4.357e-05 -> 8.238e-05 (flat)
- w_dot_wr_over_n: 0.001206 -> 0.001672 (up)
- cos_w_wr: 0.07582 -> 0.1051 (up)
- mean_rho2: 1 -> 1 (flat)
- tracking_gap: 0 -> 0 (flat)

## Learning Evidence
- td_loss_down: False
- w_alignment_up: True
- rho_finite: True
- evidence_pass: False

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

