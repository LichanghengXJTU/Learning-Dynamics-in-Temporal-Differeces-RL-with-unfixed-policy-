# Step C analysis (plateau vs instability)

Artifacts: outputs/short_runs/20251228_002251/stepC_analysis

## plateau (outputs/short_runs/20251228_002251/plateau)
- Observations:
  - td_loss: last=1.548e-05, min=1.358e-05, max=1.941e-05, slope@5=-1.715e-07
  - w_norm: last=4.582, min=4.582, max=4.582, slope@5=-1.082e-06
  - mean_rho2: last=1.331, min=1.319, max=1.337, slope@5=0.002561
  - tracking_gap: last=2.435e-13, min=1.928e-15, max=2.65e-13, slope@5=-1.663e-15
  - critic_teacher_error: last=0.01394, min=0.01394, max=0.01395, slope@5=-6.495e-08
  - stability_proxy (probe): stability_proxy: last=1, min=1, max=1, slope@5=-, coverage=1 (1.2% of 80)
  - fixed_point_drift (probe): fixed_point_drift: last=0, min=0, max=0, slope@5=-, coverage=1 (1.2% of 80)
  - dist_mmd2: last=0.008802, min=0.008802, max=0.008802, slope@5=-, coverage=1 (1.2% of 80)
  - dist_mean_l2: last=0.3155, min=0.3155, max=0.3155, slope@5=-, coverage=1 (1.2% of 80)
  - dist_action_kl: last=0.043, min=0.043, max=0.043, slope@5=-, coverage=1 (1.2% of 80)
  - dist_action_tv: last=0.113, min=0.113, max=0.113, slope@5=-, coverage=1 (1.2% of 80)
- Missing evidence:
  - probes.every=50 with outer_iters=80 -> only 1 probe point(s) (too sparse for trends).
  - fixed_point_drift trend cannot be established without multiple probe points.
- Why the mechanism is not supported yet:
  - stability_proxy ~ 1 can be explained by probe step scaling alpha_w/sqrt(feature_dim) suppressing growth relative to training.
  - distribution and fixed-point probes are too sparse to confirm plateau/instability trends.

## instability (outputs/short_runs/20251228_002251/instability)
- Observations:
  - td_loss: last=2.405e-05, min=2.154e-05, max=2.819e-05, slope@5=-1.936e-07
  - w_norm: last=9.163, min=9.163, max=9.163, slope@5=-2.744e-06
  - mean_rho2: last=0.4259, min=0.4191, max=0.4365, slope@5=-0.00214
  - tracking_gap: last=2.766e-12, min=1.586e-15, max=2.766e-12, slope@5=3.979e-14
  - critic_teacher_error: last=0.04454, min=0.04454, max=0.04455, slope@5=-1.364e-07
  - stability_proxy (probe): stability_proxy: last=1, min=1, max=1, slope@5=-1.723e-09, coverage=4 (5.0% of 80)
  - fixed_point_drift (probe): fixed_point_drift: last=0.04711, min=0, max=0.04711, slope@5=0.0007852, coverage=4 (5.0% of 80)
  - dist_mmd2: last=0.03143, min=0.02147, max=0.05181, slope@5=0.0001659, coverage=4 (5.0% of 80)
  - dist_mean_l2: last=0.6243, min=0.4908, max=0.8016, slope@5=0.002226, coverage=4 (5.0% of 80)
  - dist_action_kl: last=0.62, min=0.62, max=0.62, slope@5=5.587e-14, coverage=4 (5.0% of 80)
  - dist_action_tv: last=0.3338, min=0.3337, max=0.3338, slope@5=9.807e-07, coverage=4 (5.0% of 80)
- Missing evidence:
  - probe coverage is limited (<5 points), trend estimates remain weak.
- Why the mechanism is not supported yet:
  - stability_proxy ~ 1 can be explained by probe step scaling alpha_w/sqrt(feature_dim) suppressing growth relative to training.
  - distribution and fixed-point probes are too sparse to confirm plateau/instability trends.

## Scale check (training vs stability_probe)

plateau:
```python
feature_dim = 2048
alpha_w = 0.08
train_scale = alpha_w
probe_scale = alpha_w / np.sqrt(feature_dim)  # pre-fix stability probe
ratio = probe_scale / train_scale

```
ratio (probe/train) = 0.0221

instability:
```python
feature_dim = 2048
alpha_w = 0.2
train_scale = alpha_w
probe_scale = alpha_w / np.sqrt(feature_dim)  # pre-fix stability probe
ratio = probe_scale / train_scale

```
ratio (probe/train) = 0.0221
