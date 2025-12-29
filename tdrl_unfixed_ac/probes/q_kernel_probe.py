"""Q-kernel probe from cached TD errors."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def run_q_kernel_probe(
    *,
    delta_cache: np.ndarray,
    td_loss: float,
    iteration: int,
) -> Dict[str, Any]:
    """Compute empirical Q_hat from cached TD errors."""
    delta = np.asarray(delta_cache, dtype=float)
    if delta.ndim != 2 or delta.size == 0:
        td_loss_from_q = float("nan")
        abs_diff = float("nan")
        rel_diff = float("nan")
        t_count = 0
        b_count = int(delta.shape[0]) if delta.ndim >= 1 else 0
        t_cache = int(delta.shape[1]) if delta.ndim >= 2 else 0
    else:
        b_count, t_cache = delta.shape
        finite_mask = np.isfinite(delta)
        if np.all(finite_mask):
            q_hat = (delta.T @ delta) / max(b_count, 1)
            q_diag = np.diag(q_hat)
        else:
            q_diag = np.full(t_cache, np.nan, dtype=float)
            for t_idx in range(t_cache):
                vals = delta[:, t_idx]
                mask = finite_mask[:, t_idx]
                if np.any(mask):
                    q_diag[t_idx] = float(np.mean(vals[mask] * vals[mask]))
        t_mask = np.isfinite(q_diag)
        t_count = int(np.sum(t_mask))
        if t_count > 0:
            td_loss_from_q = 0.5 * float(np.mean(q_diag[t_mask]))
        else:
            td_loss_from_q = float("nan")
        if np.isfinite(td_loss_from_q) and np.isfinite(td_loss):
            abs_diff = float(abs(td_loss_from_q - td_loss))
            rel_diff = abs_diff / abs(td_loss) if abs(td_loss) > 0 else float("nan")
        else:
            abs_diff = float("nan")
            rel_diff = float("nan")

    return {
        "iter": int(iteration),
        "td_loss": float(td_loss),
        "td_loss_from_Q": float(td_loss_from_q),
        "td_loss_from_Q_abs_diff": float(abs_diff),
        "td_loss_from_Q_rel_diff": float(rel_diff),
        "cache_batch_size": int(b_count),
        "cache_horizon": int(t_cache),
        "cache_valid_t": int(t_count),
    }
