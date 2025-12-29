# rho consistency fix design

## Code reading (pre-fix)
1) Action clipping
- env: `TorusGobletGhostEnv.step()` applies `_clip_action` (L2 ball of radius `v_max`) before computing features/reward.
- train loop: `train_unfixed_ac` also applied `_clip_action(a_raw, env.v_max)` before `env.step`, so the env clip could still fire (double clipping).

2) Policy form
- `LinearGaussianPolicy` used a diagonal Gaussian with constant scalar sigma.
- mean parameterization: `mean = theta^T psi / sqrt(actor_dim)` via `policy_mean`.
- `sample_action`: draws `a ~ Normal(mean, sigma)`.
- `log_prob`: isotropic Gaussian log density at `a`.
- `score`: grad wrt theta: `outer(psi, (a-mean)) / (sigma^2 * sqrt(actor_dim))`.

3) rho + score action source
- `rho_raw` computed from `log_prob(a_raw)`; `rho_exec` from `log_prob(a_exec)` where `a_exec` is clipped.
- training uses `rho_exec` for updates; `score` uses `a_exec` (post-clip).

## Fix plan (implemented)
- Replace the policy with tanh-squashed Gaussian actions so the executed actions lie in-bounds and match the distribution assumed by `log_prob`/`score`.
- Sampling: draw `u ~ Normal(mean, sigma)`, then `a_exec = v_max * tanh(u)`.
- Log-prob for executed action uses change-of-variables:
  - `u = atanh(a_exec / v_max)`
  - `logp(a_exec) = logN(u; mean, sigma) - sum_i[log(v_max) + log(1 - tanh(u_i)^2)]`
  - `log(1 - tanh^2)` computed with a stable `logaddexp` form.
- Score: compute `u = atanh(a_exec / v_max)` then reuse Gaussian score on `u`:
  - `∂/∂mean logN(u; mean, sigma) = (u-mean)/sigma^2` and chain via mean=theta^T psi / sqrt(actor_dim).
- Training loop uses only `a_exec`:
  - `env.step(a_exec)` (no extra clipping in the loop).
  - `rho` uses `log_prob(a_exec)` for both pi and mu.
  - `score` uses `a_exec`.
- For audit: `rho_raw` is computed in pre-squash space using `u = atanh(a_exec / v_max)`; `rho_exec` uses `log_prob(a_exec)`. These should match (Jacobian cancels in the ratio).
- Action clipping helper + env clip updated to per-component `[-v_max, v_max]` so env clipping is a no-op for tanh-squashed actions.

## Expected consistency
- `a_exec` is the only action used for env transitions, rho, and score.
- `mean_rho2_raw` and `mean_rho2_exec` should converge (no raw/exec mismatch).
- `clip_fraction` and `|a_exec-clip(a_exec)|` should be ~0 if squashing matches env bounds.
