# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.0, alpha_w=0.2, alpha_pi=0.1, beta=0.2, sigma_mu=0.2, sigma_pi=0.2, seed=55

## Key Metrics (start -> end, trend)
- td_loss_est: 0.0001031 -> 0.0001048 (flat)
- w_dot_wr_over_n: -0.001976 -> -0.001435 (up)
- cos_w_wr: -0.1186 -> -0.08769 (up)
- mean_rho2: 1 -> 1 (flat)
- tracking_gap: 7.771e-05 -> 0.0001017 (flat)

## Learning Evidence
- evidence_pass: not enforced for this case

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

