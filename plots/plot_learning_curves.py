#!/usr/bin/env python3
"""Plot learning curves from training CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot unfixed AC learning curves.")
    parser.add_argument("--csv", type=str, required=True, help="Path to learning_curves.csv")
    parser.add_argument("--out", type=str, default=None, help="Output image path (png).")
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing header fields.")
        data = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                value = row.get(field, "")
                data[field].append(float(value))
    return {key: np.asarray(vals, dtype=float) for key, vals in data.items()}


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "learning_curves.png"

    data = load_csv(csv_path)
    iters = data.get("iter", np.arange(len(next(iter(data.values())))))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    ax0 = axes[0]
    if "td_loss" in data:
        ax0.plot(iters, data["td_loss"], linewidth=1.5, label="TD Loss")
    ax0.set_title("TD Loss + Stability")
    ax0.set_xlabel("iter")
    ax0.grid(alpha=0.3)

    if "stability_proxy" in data:
        ax1 = ax0.twinx()
        ax1.plot(iters, data["stability_proxy"], linewidth=1.2, color="tab:orange", label="Stability Proxy")
        ax1.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.0, label="Stability=1")
        ax1.set_ylabel("stability")
        handles, labels = ax0.get_legend_handles_labels()
        handles2, labels2 = ax1.get_legend_handles_labels()
        ax0.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=8)
    else:
        ax0.legend(loc="upper right", fontsize=8)

    plot_axes = axes[1:]
    candidates = [
        ("critic_teacher_error", "Critic Teacher Error"),
        ("tracking_gap", "Tracking Gap"),
        ("mean_rho2", "Mean rho^2"),
        ("w_norm", "Critic ||w||"),
        ("fixed_point_gap", "||w - w_sharp||"),
        ("dist_mmd2", "MMD^2(d_mu, d_pi)"),
    ]
    idx = 0
    for key, title in candidates:
        if key not in data:
            continue
        if idx >= len(plot_axes):
            break
        ax = plot_axes[idx]
        idx += 1
        ax.plot(iters, data[key], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.3)

    for ax in plot_axes[idx:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
