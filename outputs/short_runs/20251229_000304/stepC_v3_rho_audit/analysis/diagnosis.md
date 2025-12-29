# Step C diagnosis (rho audit + health check + short-run validation)

## 7.1 Conclusion: plateau / instability?
- Plateau run: **no plateau evidence**. `check_run_report` plateau_score=false; `stepC_analysis.md` shows td_loss and w_norm flat (slope@5 ~ -1.7e-7, w_norm flat), tracking_gap near 0, stability_proxy ~1.
- Instability run: **no instability evidence**. `check_run_report` says "too stable"; `stepC_analysis.md` shows td_loss and w_norm flat, stability_margin < 0, no sustained increases.

## 7.2 Is rho effectively on-policy (or distorted by clipping)?
Evidence from `rho_audit.md`:
- Plateau: max_rho ~1.361 vs rho_clip=10 -> rho_clip not binding; clip_fraction ~0.53; mean_rho2_raw vs mean_rho2_exec = **1.078 vs 1.331** -> action clipping changes the log_prob action.
- Instability: disable_rho_clip=true, yet mean_rho2_exec ~0.426 **(<1)** and p99_rho<1; clip_fraction ~0.28; mean_rho2_raw vs mean_rho2_exec = **3.657 vs 0.426** -> strong raw/exec mismatch.
- E_mu[rho^2] < 1 occurs in instability, which should not happen for a correct density ratio. This indicates rho is **not** the correct pi(a|s)/mu(a|s) on executed actions (action clipping / mismatch dominates).

## 7.3 Hypothesis & assumptions vs evidence
- Hypothesis (plateau + instability should appear): **not observed** in this short run.
- Type-II nonstationarity (theta_pi changes): **present but tiny** (delta_theta_pi_norm ~1e-6); policy moves, but very slowly.
- Effective off-policy signal: **weak/ distorted**.
  - Plateau: rho tail small (p99 ~1.358), max_rho ~1.361.
  - Instability: rho tail <1 and mean_rho2_exec <1; raw/exec mismatch suggests distorted rho.
- Update scale sufficient to cross stability boundary? **No**: stability_proxy ~1 with negative stability margin; w_norm and td_loss flat; delta_w_norm ~5e-5 (plateau) / ~1e-4 (instability) are small.

## 7.4 Next-step actions (two branches)
### A) If rho audit shows distortion / action mismatch (applies now)
Minimal fix routes:
1) **Tanh-squashed Gaussian with Jacobian correction**
   - Replace hard clipping with tanh-squash; compute log_prob for the squashed action with the Jacobian term.
   - Aligns executed action and log_prob, preserves bounded action while keeping a valid density ratio.
2) **Torus env: wrap rather than clip (consistency fix)**
   - Execute a wrapped action transform in env and compute log_prob on the same transformed action.
   - Better if action is treated as angular on the torus; avoids hard truncation artifacts.

Recommendation: given v_max is a speed cap, tanh-squash + Jacobian is the more principled bounded-action fix; wrapping is suitable if action is interpreted as angular control.

### B) If rho is acceptable but dynamics too stable (conditional)
Sweep suggestions (<=10 short runs) with expected effect; configs saved under `meta/sweeps/`:
- `instab_alpha_boost.yaml`: higher alpha_w/alpha_pi, lower beta -> larger updates, more instability.
- `instab_sigma_mismatch.yaml`: widen sigma_pi vs sigma_mu -> stronger off-policy mismatch (larger rho tail).
- `instab_pmix0.yaml`: remove p_mix -> reduce resets, allow drift accumulation.
- `instab_theta_radius12.yaml`: larger theta radius -> allow bigger policy movement.
- `instab_rho_clip_on.yaml`: enable rho_clip=5 -> test stability with bounded rho.

Run template: `python scripts/run_train.py --config <sweep_yaml> --output-dir <out> --outer-iters 80 --report-every 20`.

## 7.5 Long-run gate
Gate condition: health_summary PASS (core), rho audit not anomalous (no E[rho^2] < 1 or raw/exec mismatch explained), and delta_theta_pi_norm > 0 (if Type-II needed).
- Current status: **FAIL gate** due to rho distortion in instability (mean_rho2_exec < 1, large raw/exec mismatch).

If gate passes, long-run template (5000 iters, seeds 0/1/2, report_every 50):
```
python scripts/run_train.py --config configs/train_plateau.yaml --output-dir outputs/long_runs/<ts>/plateau_seed{0,1,2} --outer-iters 5000 --report-every 50 --seed <seed>
python scripts/run_train.py --config configs/train_instability.yaml --output-dir outputs/long_runs/<ts>/instability_seed{0,1,2} --outer-iters 5000 --report-every 50 --seed <seed>
```
