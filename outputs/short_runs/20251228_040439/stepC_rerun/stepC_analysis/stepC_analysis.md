# Step C analysis (plateau vs instability)

Artifacts: outputs/short_runs/20251228_040439/stepC_rerun/stepC_analysis

## plateau (outputs/short_runs/20251228_040439/stepC_rerun/plateau)
- Observations:
  - td_loss: last=1.561e-05, min=1.358e-05, max=1.941e-05, slope@5=8.755e-09
  - w_norm: last=4.581, min=4.581, max=4.582, slope@5=-9.868e-07
  - mean_rho2: last=1.326, min=1.319, max=1.337, slope@5=0.000446
  - tracking_gap: last=2.45e-13, min=1.928e-15, max=2.724e-13, slope@5=7.032e-15
  - critic_teacher_error: last=0.01393, min=0.01393, max=0.01395, slope@5=-6.058e-08
  - stability_proxy_mean (probe): stability_proxy_mean: last=1, min=1, max=1, slope@5=-2.254e-08, coverage=56 (28.0% of 200)
  - fixed_point_drift (probe): fixed_point_drift: last=0.02831, min=0, max=0.03744, slope@5=0.0003192, coverage=57 (28.5% of 200)
  - dist_mmd2: last=0.01184, min=0.00319, max=0.01617, slope@5=0.0004814, coverage=56 (28.0% of 200)
  - dist_mean_l2: last=0.3817, min=0.1518, max=0.454, slope@5=0.01266, coverage=56 (28.0% of 200)
  - dist_action_kl: last=0.043, min=0.043, max=0.043, slope@5=5.055e-16, coverage=56 (28.0% of 200)
  - dist_action_tv: last=0.1131, min=0.1125, max=0.1134, slope@5=-1.309e-05, coverage=56 (28.0% of 200)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001014 (>0 -> instability candidate)
  - drift_slope=0.0003993 (>0 supports plateau drift)
  - td_loss_slope=-5.307e-08 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.09805 (>= 0.001 -> tracking gap persists)

## instability (outputs/short_runs/20251228_040439/stepC_rerun/instability)
- Observations:
  - td_loss: last=2.585e-05, min=2.114e-05, max=2.819e-05, slope@5=1.092e-07
  - w_norm: last=9.163, min=9.163, max=9.163, slope@5=-2.353e-06
  - mean_rho2: last=0.4315, min=0.4191, max=0.4365, slope@5=0.0001227
  - tracking_gap: last=5.646e-12, min=1.586e-15, max=5.646e-12, slope@5=1.51e-14
  - critic_teacher_error: last=0.04453, min=0.04453, max=0.04455, slope@5=-1.268e-07
  - stability_proxy_mean (probe): stability_proxy_mean: last=1, min=1, max=1, slope@5=9.2e-08, coverage=56 (28.0% of 200)
  - fixed_point_drift (probe): fixed_point_drift: last=0.04351, min=0, max=0.05478, slope@5=-0.0003048, coverage=56 (28.0% of 200)
  - dist_mmd2: last=0.02398, min=0.007235, max=0.06195, slope@5=-1.038e-05, coverage=55 (27.5% of 200)
  - dist_mean_l2: last=0.5386, min=0.2066, max=0.9258, slope@5=0.000949, coverage=55 (27.5% of 200)
  - dist_action_kl: last=0.62, min=0.62, max=0.62, slope@5=2.145e-14, coverage=55 (27.5% of 200)
  - dist_action_tv: last=0.3339, min=0.3331, max=0.3343, slope@5=2.111e-05, coverage=55 (27.5% of 200)
- Missing evidence:
- Evidence chain:
  - stability_margin=-0.001018 (>0 -> instability candidate)
  - drift_slope=-0.001376 (>0 supports plateau drift)
  - td_loss_slope=-3.907e-09 (|slope|<1e-06 -> flat)
  - w_gap_min_last_window=0.1741 (>= 0.001 -> tracking gap persists)

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

## Verdict

- instability evidence: absent (stability_margin=-0.001018, w_norm_increasing=False, td_loss_increasing=False)
- plateau (tracking-limited) evidence: absent (drift_slope=0.0003993, w_gap_min=0.09805, td_loss_slope=-5.307e-08)
- if absent, likely reason: plateau: instrumentation: insufficient probe coverage.; instability: parameter regime: stability_margin<=0; no sustained w_norm/td_loss increase