# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.1, beta=0.0, sigma_mu=0.35, sigma_pi=0.2, seed=22

## Key Metrics (start -> end, trend)
- td_loss_est: 0.0001149 -> 0.0001004 (flat)
- w_dot_wr_over_n: 0.0003333 -> 0.0008157 (up)
- cos_w_wr: 0.02045 -> 0.04989 (up)
- mean_rho2: 1.856 -> 2.242 (up)
- tracking_gap: 0.6696 -> 0.6697 (up)

## Learning Evidence
- evidence_pass: not enforced for this case

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

