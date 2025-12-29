# Step C analysis (plateau vs instability)

## Repo diff (files)
- (not provided)

## Key outputs
- (not provided)

Artifacts: outputs/short_runs/20251229_014833/stepC_v4_rho_fix/analysis

## plateau (outputs/short_runs/20251229_014833/stepC_v4_rho_fix/plateau)
- Observations:
  - td_loss: last=1.537e-05, min=1.367e-05, max=1.83e-05, slope@5=-3.808e-07
  - w_norm: last=4.582, min=4.582, max=4.582, slope@5=-5.81e-07
  - mean_rho2: last=1.068, min=1.058, max=1.095, slope@5=-0.001903
  - tracking_gap: last=3.964e-14, min=9.334e-16, max=5.444e-14, slope@5=6.52e-17
  - critic_teacher_error: last=0.01394, min=0.01394, max=0.01395, slope@5=-5.338e-08
  - stability_proxy_mean (probe): stability_proxy_mean: last=1, min=1, max=1, slope@5=-4.02e-08, coverage=23 (28.7% of 80)
  - fixed_point_drift (probe): fixed_point_drift: last=0.02058, min=0, max=0.04004, slope@5=0.0001649, coverage=23 (28.7% of 80)
  - dist_mmd2: last=0.01831, min=0.004201, max=0.01831, slope@5=0.0008789, coverage=23 (28.7% of 80)
  - dist_mean_l2: last=0.473, min=0.1737, max=0.473, slope@5=0.01415, coverage=23 (28.7% of 80)
  - dist_action_kl: last=0.043, min=0.043, max=0.043, slope@5=4.239e-16, coverage=23 (28.7% of 80)
  - dist_action_tv: last=0.1133, min=0.1125, max=0.1134, slope@5=1.066e-05, coverage=23 (28.7% of 80)
  - td_loss_from_Q: last=8.122e-06, min=6.72e-06, max=9.496e-06, slope@5=-3.449e-09, coverage=23 (28.7% of 80)
  - td_loss_from_Q_abs_diff: last=7.249e-06, min=6.42e-06, max=9.714e-06, slope@5=-2.132e-08, coverage=23 (28.7% of 80)
  - td_loss_from_Q_rel_diff: last=0.4716, min=0.404, max=0.5537, slope@5=-0.0006162, coverage=23 (28.7% of 80)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001008 (>0 -> instability candidate)
  - drift_slope=6.101e-05 (>0 supports plateau drift)
  - td_loss_slope=1.642e-08 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.08908 (>= 0.001 -> tracking gap persists)

## instability (outputs/short_runs/20251229_014833/stepC_v4_rho_fix/instability)
- Observations:
  - td_loss: last=2.199e-05, min=2.068e-05, max=2.612e-05, slope@5=-9.667e-07
  - w_norm: last=9.163, min=9.163, max=9.163, slope@5=-1.473e-06
  - mean_rho2: last=5.697, min=2.511, max=42.38, slope@5=0.7232
  - tracking_gap: last=3.818e-11, min=4.853e-14, max=4.149e-11, slope@5=-4.222e-13
  - critic_teacher_error: last=0.04454, min=0.04454, max=0.04455, slope@5=-1.674e-07
  - stability_proxy_mean (probe): stability_proxy_mean: last=1, min=1, max=1, slope@5=-2.043e-07, coverage=23 (28.7% of 80)
  - fixed_point_drift (probe): fixed_point_drift: last=0.2407, min=0, max=0.2637, slope@5=0.01328, coverage=23 (28.7% of 80)
  - dist_mmd2: last=0.04631, min=0.01142, max=0.05497, slope@5=0.002631, coverage=23 (28.7% of 80)
  - dist_mean_l2: last=0.7595, min=0.2771, max=0.8321, slope@5=0.03139, coverage=23 (28.7% of 80)
  - dist_action_kl: last=0.62, min=0.62, max=0.62, slope@5=2.745e-13, coverage=23 (28.7% of 80)
  - dist_action_tv: last=0.3338, min=0.3331, max=0.3341, slope@5=-2.899e-05, coverage=23 (28.7% of 80)
  - td_loss_from_Q: last=1.148e-05, min=1.044e-05, max=1.336e-05, slope@5=-1.617e-07, coverage=23 (28.7% of 80)
  - td_loss_from_Q_abs_diff: last=1.051e-05, min=9.144e-06, max=1.336e-05, slope@5=3.659e-08, coverage=23 (28.7% of 80)
  - td_loss_from_Q_rel_diff: last=0.478, min=0.4327, max=0.559, slope@5=0.004126, coverage=23 (28.7% of 80)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001017 (>0 -> instability candidate)
  - drift_slope=-0.0003818 (>0 supports plateau drift)
  - td_loss_slope=-5.065e-08 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.2601 (>= 0.001 -> tracking gap persists)

## Scale check (training vs stability_probe)

plateau:
```python
alpha_w = 0.08
trajectories = 16
horizon = 200
train_step_scale = alpha_w / (trajectories * horizon)
stability_probe_step_scale = 2.5e-05
ratio = stability_probe_step_scale / train_step_scale

```
ratio (probe/train) = 1

instability:
```python
alpha_w = 0.2
trajectories = 16
horizon = 200
train_step_scale = alpha_w / (trajectories * horizon)
stability_probe_step_scale = 6.25e-05
ratio = stability_probe_step_scale / train_step_scale

```
ratio (probe/train) = 1

## Metric alignment notes

- td_loss in training is mean(delta^2) over all steps; it matches (1/T) * sum_t E[Delta(t)^2].
- td_loss_from_Q uses cached Delta to compute Q_hat(t,t') = E_b[Delta(t)Delta(t')] and
  td_loss_from_Q = (1/(2T_cache)) * sum_t Q_hat(t,t); expect ~0.5 * td_loss.
- rho consistency checks:
  - plateau: mean_rho close to 1 (consistent with mu-sampled actions). action squashing consistent with log_prob. p_mix=0.05 changes state distribution but not the per-state identity. (rho_clip active)
  - instability: mean_rho close to 1 (consistent with mu-sampled actions). action squashing consistent with log_prob. p_mix=0.01 changes state distribution but not the per-state identity. (rho_clip inactive)

## Verdict

- instability evidence: absent (stability_margin=-0.001017, w_norm_increasing=False, td_loss_increasing=False)
- plateau (tracking-limited) evidence: absent (drift_slope=6.101e-05, w_gap_min=0.08908, td_loss_slope=1.642e-08)
- if absent, likely reason: plateau: instrumentation: insufficient probe coverage.; instability: parameter regime: stability_margin<=0; no sustained w_norm/td_loss increase