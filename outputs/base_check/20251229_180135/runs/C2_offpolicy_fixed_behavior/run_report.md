# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.1, beta=0.0, sigma_mu=0.35, sigma_pi=0.2, seed=22

## Key Metrics (start -> end, trend)
- td_loss_est: 0.0001195 -> 0.0001054 (flat)
- w_dot_wr_over_n: 0.0003364 -> 0.0007753 (up)
- cos_w_wr: 0.02064 -> 0.04742 (up)
- mean_rho2: 1.893 -> 2.307 (up)
- tracking_gap: 0.6696 -> 0.6696 (flat)

## Learning Evidence
- evidence_pass: not enforced for this case

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

