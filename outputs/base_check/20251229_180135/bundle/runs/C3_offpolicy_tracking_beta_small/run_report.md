# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.1, beta=0.1, sigma_mu=0.35, sigma_pi=0.2, seed=33

## Key Metrics (start -> end, trend)
- td_loss_est: 1.78e-05 -> 7.987e-05 (flat)
- w_dot_wr_over_n: 3.396e-05 -> 0.0005508 (up)
- cos_w_wr: 0.002118 -> 0.03433 (up)
- mean_rho2: 1.731 -> 2.06 (up)
- tracking_gap: 0.6833 -> 0.0001787 (down)

## Learning Evidence
- evidence_pass: not enforced for this case

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

