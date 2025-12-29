# gate check

status: PASS

- condition1 (instability mean_rho2_exec min >= 1): 2.5106 -> PASS
- condition2 (raw vs exec no order-of-magnitude diff): mean_rho2_raw_min=2.5106, mean_rho2_exec_min=2.5106 (ratio=1.00), last ratio=1.00 -> PASS
- condition3 (health_summary PASS): PASS
- condition4 (delta_theta_pi_norm > 0): 3.49e-06 -> PASS
