# Step C analysis (plateau vs instability)

## Repo diff (files)
- (not provided)

## Key outputs
- (not provided)

Artifacts: outputs/short_runs/20251229_000304/stepC_v3_rho_audit/analysis

## plateau (outputs/short_runs/20251229_000304/stepC_v3_rho_audit/plateau)
- Observations:
  - td_loss: last=1.548e-05, min=1.358e-05, max=1.941e-05, slope@5=-1.715e-07
  - w_norm: last=4.582, min=4.582, max=4.582, slope@5=-1.082e-06
  - mean_rho2: last=1.331, min=1.319, max=1.337, slope@5=0.002561
  - tracking_gap: last=2.435e-13, min=1.928e-15, max=2.65e-13, slope@5=-1.663e-15
  - critic_teacher_error: last=0.01394, min=0.01394, max=0.01395, slope@5=-6.495e-08
  - stability_proxy_mean (probe): stability_proxy_mean: last=1, min=1, max=1, slope@5=4.076e-08, coverage=23 (28.7% of 80)
  - fixed_point_drift (probe): fixed_point_drift: last=0.02164, min=0, max=0.02593, slope@5=-5.645e-05, coverage=23 (28.7% of 80)
  - dist_mmd2: last=0.00835, min=0.00348, max=0.01532, slope@5=-1.883e-05, coverage=23 (28.7% of 80)
  - dist_mean_l2: last=0.2996, min=0.1591, max=0.4377, slope@5=-0.001471, coverage=23 (28.7% of 80)
  - dist_action_kl: last=0.043, min=0.043, max=0.043, slope@5=-1.565e-15, coverage=23 (28.7% of 80)
  - dist_action_tv: last=0.1133, min=0.1125, max=0.1134, slope@5=5.819e-05, coverage=23 (28.7% of 80)
  - td_loss_from_Q: last=6.413e-06, min=6.413e-06, max=9.888e-06, slope@5=-9.21e-08, coverage=23 (28.7% of 80)
  - td_loss_from_Q_abs_diff: last=9.069e-06, min=6.568e-06, max=1.003e-05, slope@5=-5.545e-09, coverage=23 (28.7% of 80)
  - td_loss_from_Q_rel_diff: last=0.5858, min=0.4386, max=0.5858, slope@5=0.003066, coverage=23 (28.7% of 80)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001015 (>0 -> instability candidate)
  - drift_slope=-7.721e-05 (>0 supports plateau drift)
  - td_loss_slope=-1.135e-08 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.09969 (>= 0.001 -> tracking gap persists)

## instability (outputs/short_runs/20251229_000304/stepC_v3_rho_audit/instability)
- Observations:
  - td_loss: last=2.405e-05, min=2.154e-05, max=2.819e-05, slope@5=-1.936e-07
  - w_norm: last=9.163, min=9.163, max=9.163, slope@5=-2.744e-06
  - mean_rho2: last=0.4259, min=0.4191, max=0.4365, slope@5=-0.00214
  - tracking_gap: last=2.766e-12, min=1.586e-15, max=2.766e-12, slope@5=3.979e-14
  - critic_teacher_error: last=0.04454, min=0.04454, max=0.04455, slope@5=-1.364e-07
  - stability_proxy_mean (probe): stability_proxy_mean: last=1, min=1, max=1, slope@5=5.128e-09, coverage=23 (28.7% of 80)
  - fixed_point_drift (probe): fixed_point_drift: last=0.03875, min=0, max=0.05437, slope@5=-0.0005227, coverage=23 (28.7% of 80)
  - dist_mmd2: last=0.03143, min=0.01104, max=0.06195, slope@5=0.0004076, coverage=23 (28.7% of 80)
  - dist_mean_l2: last=0.6243, min=0.2645, max=0.9258, slope@5=0.006309, coverage=23 (28.7% of 80)
  - dist_action_kl: last=0.62, min=0.62, max=0.62, slope@5=8.357e-14, coverage=23 (28.7% of 80)
  - dist_action_tv: last=0.3338, min=0.3331, max=0.334, slope@5=6.031e-06, coverage=23 (28.7% of 80)
  - td_loss_from_Q: last=1.19e-05, min=1.035e-05, max=1.361e-05, slope@5=-4.9e-08, coverage=23 (28.7% of 80)
  - td_loss_from_Q_abs_diff: last=1.215e-05, min=9.96e-06, max=1.387e-05, slope@5=3.741e-08, coverage=23 (28.7% of 80)
  - td_loss_from_Q_rel_diff: last=0.5053, min=0.4624, max=0.5409, slope@5=0.001787, coverage=23 (28.7% of 80)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001018 (>0 -> instability candidate)
  - drift_slope=-0.0005009 (>0 supports plateau drift)
  - td_loss_slope=4.892e-08 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.1801 (>= 0.001 -> tracking gap persists)

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
  - plateau: mean_rho deviates from 1 (last=1.15). Actions are clipped before log_prob; this truncation biases E_mu[rho] away from 1. p_mix=0.05 changes state distribution but not the per-state identity. (rho_clip active)
  - instability: mean_rho deviates from 1 (last=0.6291). Actions are clipped before log_prob; this truncation biases E_mu[rho] away from 1. p_mix=0.01 changes state distribution but not the per-state identity. (rho_clip inactive)

## Verdict

- instability evidence: absent (stability_margin=-0.001018, w_norm_increasing=False, td_loss_increasing=False)
- plateau (tracking-limited) evidence: absent (drift_slope=-7.721e-05, w_gap_min=0.09969, td_loss_slope=-1.135e-08)
- if absent, likely reason: plateau: parameter regime: drift_slope<=0; instability: parameter regime: stability_margin<=0; no sustained w_norm/td_loss increase