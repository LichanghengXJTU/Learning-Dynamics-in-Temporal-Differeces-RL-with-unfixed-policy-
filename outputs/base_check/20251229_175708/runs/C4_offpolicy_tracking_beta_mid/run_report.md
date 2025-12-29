# Base-Check Run Report

## Config
- N=None, N_act=None, B=4, T=30, gamma=0.95, alpha_w=0.2, alpha_pi=0.1, beta=0.5, sigma_mu=0.35, sigma_pi=0.2, seed=44

## Key Metrics (start -> end, trend)
- td_loss_est: 9.512e-05 -> 6.918e-05 (flat)
- w_dot_wr_over_n: -0.002229 -> -0.001674 (up)
- cos_w_wr: -0.1292 -> -0.09772 (up)
- mean_rho2: 1.757 -> 1.835 (down)
- tracking_gap: 0.3879 -> 4.189e-05 (down)

## Learning Evidence
- evidence_pass: not enforced for this case

## Sigma Condition
- sigma_pi^2 < 2 sigma_mu^2: True

