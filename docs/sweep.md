# Step C sweep

## Quick start
Run a 7x7x7 sweep (default axes) and summarize:

```bash
python scripts/sweep_stepC.py --base-config configs/train_plateau.yaml --outer-iters 1000 --grid 7 --jobs 1
```

Outputs are written under `outputs/sweep/<timestamp>/`.

## Resume
Reuse a prior sweep root:

```bash
python scripts/sweep_stepC.py --timestamp <YYYYmmdd_HHMMSS> --resume --jobs 1
```

Or point to the sweep root directly:

```bash
python scripts/sweep_stepC.py --run-root outputs/sweep/<timestamp> --resume --jobs 1
```

## Pilot (2x2x2)
Quick pilot grid:

```bash
python scripts/sweep_stepC.py --base-config configs/train_plateau.yaml --outer-iters 200 --grid 2 --jobs 1
```

## Summary outputs
After the sweep finishes:

- `outputs/sweep/<timestamp>/summary/summary.md` includes bucketed top-k tables and thresholds.
- `outputs/sweep/<timestamp>/summary/summary.csv` has the full grid.
- `outputs/sweep/<timestamp>/summary/buckets/` contains per-bucket CSVs.

To bias selection toward true off-policy points, pick from `offpolicy_stable_top.csv`. For instability boundary points, check the "Stability boundary" table in `summary.md`, then rerun those configurations with longer `--outer-iters`.

## Common tweaks
- Adjust thresholds: `--eps-slope`, `--eps-drift`, `--offpolicy-threshold`.
- Change off-policy axis: `--offpolicy-axis sigma_mu|sigma_pi|p_mix`.
- Print the plan only: `--dry-run`.
