# diagnosis

## 1) plateau / instability
- plateau: not observed. `plateau_score` is false; td_loss and w_norm are flat (slopes ~0), and tracking gap stays well above threshold (w_gap_min_last_window=0.089).
- instability: not observed. `instability_candidate` is false; td_loss/w_norm do not increase, and stability_margin is negative.

## 2) rho consistency (instability run)
- E[rho^2] lower bound: mean_rho2_exec min = 2.5106 (>= 1).
- raw vs exec: mean_rho2_raw_min = 2.5106 and mean_rho2_exec_min = 2.5106 (ratio=1.00); last raw/exec both 5.697.
- tails: p99_rho_max = 8.526, max_rho_max = 323.7.
- clip diagnostics: clip_fraction_max = 0 and |a_exec-clip(a_exec)| = 0 (no residual clipping).

## 3) on-policy / weak off-policy
- tracking_gap_last = 3.818e-11 and mean_rho ~ 1, with clip_fraction=0.
- This is effectively on-policy / weak off-policy despite moderate tails (p99 ~ 8.5).

## 4) why no plateau/instability (likely order)
1) Updates are tiny: delta_theta_pi_norm_last = 3.49e-06, delta_w_norm_last = 1.39e-04.
2) mu tracks pi too closely (tracking_gap ~ 1e-11), so off-policy drive is weak.
3) rho tail is not extreme (p99 ~ 8.5) and only occasional large maxima; 80 iters may be too short to accumulate drift.
4) Environment/teacher dynamics are stable; w_norm is flat and probes cover only ~23/80 iterations.

## 5) sweep takeaways
- instab_alpha_boost / instab_theta_radius12 / instab_pmix0: mean_rho2_last ~ 3.657 and still stable (no instability candidate).
- instab_sigma_mismatch: much heavier tails (max_rho_max ~ 2.01e3) but health FAILED and still no instability.
- instab_rho_clip_on: rho tails bounded (p99/max = 5), most stable as expected.

## 6) next steps
- Long-run only if the gate remains PASS (mean_rho2_exec min >= 1, raw/exec within 2x, health PASS, delta_theta_pi_norm > 0). Otherwise adjust step sizes / mismatch first.
- Long-run template (5000 iters, report_every 50, seeds 0/1/2):
  .venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/plateau_seed0 --outer-iters 5000 --report-every 50 --seed 0
  .venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/plateau_seed1 --outer-iters 5000 --report-every 50 --seed 1
  .venv/bin/python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/plateau_seed2 --outer-iters 5000 --report-every 50 --seed 2
  .venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/instability_seed0 --outer-iters 5000 --report-every 50 --seed 0
  .venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/instability_seed1 --outer-iters 5000 --report-every 50 --seed 1
  .venv/bin/python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/instability_seed2 --outer-iters 5000 --report-every 50 --seed 2
