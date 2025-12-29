# Step C analysis (plateau vs instability)

## Repo diff (files)
- (not provided)

## Key outputs
- (not provided)

Artifacts: outputs/base_check/20251229_192835/stepC_v2/analysis

## plateau (outputs/base_check/20251229_192835/stepC_v2/plateau)
- Observations:
  - td_loss: last=1.863e-05, min=1.444e-05, max=2.04e-05, slope@5=3.031e-07
  - w_norm: last=4.581, min=4.581, max=4.582, slope@5=-3.639e-06
  - mean_rho2: last=1.07, min=1.057, max=1.104, slope@5=0.0004322
  - tracking_gap: last=7.435e-12, min=4.72e-14, max=8.866e-12, slope@5=6.594e-14
  - critic_teacher_error: last=0.0139, min=0.0139, max=0.01395, slope@5=-2.296e-07
  - stability_proxy_mean (probe): stability_proxy_mean: last=0.9999, min=0.9999, max=0.9999, slope@5=4.119e-07, coverage=59 (29.5% of 200)
  - fixed_point_drift (probe): fixed_point_drift: last=0.02037, min=0, max=0.02944, slope@5=-0.0001246, coverage=59 (29.5% of 200)
  - dist_mmd2: last=0.008532, min=0.002824, max=0.01428, slope@5=0.0001305, coverage=59 (29.5% of 200)
  - dist_mean_l2: last=0.3079, min=0.1185, max=0.417, slope@5=0.002683, coverage=59 (29.5% of 200)
  - dist_action_kl: last=0.043, min=0.043, max=0.043, slope@5=-3.747e-15, coverage=59 (29.5% of 200)
  - dist_action_tv: last=0.1131, min=0.1125, max=0.1134, slope@5=-1.309e-05, coverage=59 (29.5% of 200)
  - td_loss_from_Q: last=8.022e-06, min=7.224e-06, max=1.074e-05, slope@5=-7.292e-08, coverage=59 (29.5% of 200)
  - td_loss_from_Q_abs_diff: last=1.061e-05, min=7.038e-06, max=1.077e-05, slope@5=3.23e-08, coverage=59 (29.5% of 200)
  - td_loss_from_Q_rel_diff: last=0.5694, min=0.4299, max=0.5694, slope@5=0.002887, coverage=59 (29.5% of 200)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001062 (>0 -> instability candidate)
  - drift_slope=1.929e-05 (>0 supports plateau drift)
  - td_loss_slope=-1.615e-09 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.08362 (>= 0.001 -> tracking gap persists)

## instability (outputs/base_check/20251229_192835/stepC_v2/instability)
- Observations:
  - td_loss: last=2.295e-05, min=2.011e-05, max=2.873e-05, slope@5=5.637e-08
  - w_norm: last=9.159, min=9.159, max=9.163, slope@5=-1.554e-05
  - mean_rho2: last=7.391, min=1.862, max=168.3, slope@5=0.7165
  - tracking_gap: last=1.019e-09, min=3.94e-12, max=1.102e-09, slope@5=-1.943e-11
  - critic_teacher_error: last=0.04437, min=0.04437, max=0.04455, slope@5=-7.526e-07
  - stability_proxy_mean (probe): stability_proxy_mean: last=0.9998, min=0.9998, max=1.036, slope@5=9.243e-08, coverage=59 (29.5% of 200)
  - fixed_point_drift (probe): fixed_point_drift: last=0.1192, min=0, max=0.2839, slope@5=0.0005039, coverage=59 (29.5% of 200)
  - dist_mmd2: last=0.0348, min=0.005541, max=0.076, slope@5=0.0001389, coverage=59 (29.5% of 200)
  - dist_mean_l2: last=0.6769, min=0.159, max=1.022, slope@5=0.001893, coverage=59 (29.5% of 200)
  - dist_action_kl: last=0.62, min=0.62, max=0.62, slope@5=9.521e-13, coverage=59 (29.5% of 200)
  - dist_action_tv: last=0.3339, min=0.3331, max=0.3343, slope@5=-1.14e-05, coverage=59 (29.5% of 200)
  - td_loss_from_Q: last=1.141e-05, min=9.546e-06, max=1.45e-05, slope@5=1.325e-08, coverage=59 (29.5% of 200)
  - td_loss_from_Q_abs_diff: last=1.154e-05, min=1.06e-05, max=1.416e-05, slope@5=6.345e-09, coverage=59 (29.5% of 200)
  - td_loss_from_Q_rel_diff: last=0.5028, min=0.4376, max=0.5475, slope@5=-0.0001547, coverage=59 (29.5% of 200)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001146 (>0 -> instability candidate)
  - drift_slope=0.0009354 (>0 supports plateau drift)
  - td_loss_slope=-5.047e-08 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.2522 (>= 0.001 -> tracking gap persists)

## Scale check (training vs stability_probe)

plateau:
```python
alpha_w = 0.08
trajectories = 16
horizon = 200
train_step_scale = alpha_w / (trajectories * horizon)
stability_probe_step_scale = 0.0001
ratio = stability_probe_step_scale / train_step_scale

```
ratio (probe/train) = 4

instability:
```python
alpha_w = 0.2
trajectories = 16
horizon = 200
train_step_scale = alpha_w / (trajectories * horizon)
stability_probe_step_scale = 0.00025
ratio = stability_probe_step_scale / train_step_scale

```
ratio (probe/train) = 4

## Metric alignment notes

- td_loss in training is mean(delta^2) over all steps; it matches (1/T) * sum_t E[Delta(t)^2].
- td_loss_from_Q uses cached Delta to compute Q_hat(t,t') = E_b[Delta(t)Delta(t')] and
  td_loss_from_Q = (1/(2T_cache)) * sum_t Q_hat(t,t); expect ~0.5 * td_loss.
- rho consistency checks:
  - plateau: mean_rho close to 1 (consistent with mu-sampled actions). action squashing consistent with log_prob. p_mix=0.05 changes state distribution but not the per-state identity. (rho_clip active)
  - instability: mean_rho close to 1 (consistent with mu-sampled actions). action squashing consistent with log_prob. p_mix=0.01 changes state distribution but not the per-state identity. (rho_clip inactive)

## Verdict

- instability evidence: absent (stability_margin=-0.001146, w_norm_increasing=False, td_loss_increasing=False)
- plateau (tracking-limited) evidence: absent (drift_slope=1.929e-05, w_gap_min=0.08362, td_loss_slope=-1.615e-09)
- if absent, likely reason: plateau: instrumentation: insufficient probe coverage.; instability: parameter regime: stability_margin<=0; no sustained w_norm/td_loss increase