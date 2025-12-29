# change rationale

The instability runs showed E[rho^2] < 1 because the executed actions were clipped but the log_prob/score assumed an unbounded Gaussian. That mismatch makes rho a ratio of *different* densities (pre-clip vs post-clip), which biases rho downward and can even violate E_mu[rho^2] >= 1.

The fix makes the executed action distribution match the density used in log_prob/score by:
- sampling u ~ Normal(mean, sigma) and executing a_exec = v_max * tanh(u),
- computing log_prob(a_exec) with the exact change-of-variables (Gaussian log-prob on u minus the Jacobian term), and
- using the same a_exec everywhere (env.step, rho, and score).

With this change, rho is computed on the correct density ratio for the actually executed actions, and the Jacobian term cancels between pi and mu. That removes the systematic underestimation that previously produced E[rho^2] < 1.
