# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.1, beta=0.1, sigma_mu=0.35, sigma_pi=0.2, seed=33

## Key Metrics (start -> end, trend)
- td_loss_est: 1.885e-05 -> 7.208e-05 (flat)
- w_dot_wr_over_n: 3.424e-05 -> 0.0005655 (up)
- cos_w_wr: 0.002136 -> 0.03527 (up)
- mean_rho2: 1.727 -> 2.06 (up)
- tracking_gap: 0.6833 -> 0.0001951 (down)

## Learning Evidence
- evidence_pass: not enforced for this case

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

