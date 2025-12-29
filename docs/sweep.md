# Step C sweep

## Quick start
Run a 10x10x10 sweep (alpha_w logspace, beta linspace, theta_mu_offset_scale linspace) and summarize:

```bash
python scripts/sweep_stepC.py --base-config configs/train_plateau.yaml --outer-iters 200 --grid 10 --jobs 1
```

Outputs are written under `outputs/base_check/<timestamp>/sweep/`.

## Resume
Reuse a prior sweep root:

```bash
python scripts/sweep_stepC.py --timestamp <YYYYmmdd_HHMMSS> --resume --jobs 1
```

Or point to the sweep root directly:

```bash
python scripts/sweep_stepC.py --run-root outputs/base_check/<timestamp>/sweep --resume --jobs 1
```

## Pilot (2x2x2)
Quick pilot grid:

```bash
python scripts/sweep_stepC.py --base-config configs/train_plateau.yaml --outer-iters 200 --grid 2 --jobs 1
```

## Summary outputs
After the sweep finishes:

- `outputs/base_check/<timestamp>/sweep/summary/summary.md` includes bucketed top-k tables and thresholds.
- `outputs/base_check/<timestamp>/sweep/summary/summary.csv` has the full grid.
- `outputs/base_check/<timestamp>/sweep/summary/summary.json` has the full grid in JSON.
- `outputs/base_check/<timestamp>/sweep/summary/buckets/` contains per-bucket CSVs.

To bias selection toward true off-policy points, pick from `offpolicy_stable_top.csv`. For instability boundary points, check the "Stability boundary" table in `summary.md`, then rerun those configurations with longer `--outer-iters`.

## Common tweaks
- Adjust thresholds: `--eps-slope`, `--eps-drift`, `--offpolicy-threshold`.
- Override grid values: `--beta-values`, `--alpha-w-values`, `--theta-mu-offset-values`.
- Print the plan only: `--dry-run`.
