"""Interpretable observation vector construction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from tdrl_unfixed_ac.utils.geometry import torus_delta


def _encode_position(pos: np.ndarray, torus_size: float) -> np.ndarray:
    """Encode a 2D position with sin/cos to remove torus discontinuity."""
    pos = np.asarray(pos, dtype=float).reshape(2)
    angle = 2.0 * np.pi * pos / torus_size
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=0)


def build_event_flags(raw_obs: Dict[str, Any]) -> np.ndarray:
    """Return event one-hots: caught, picked_pos, picked_neg, restart."""
    picked_type = float(raw_obs.get("picked_type", 0.0))
    picked = bool(raw_obs.get("picked", False))
    return np.array(
        [
            float(raw_obs.get("caught", False)),
            float(picked and picked_type > 0.0),
            float(picked and picked_type < 0.0),
            float(raw_obs.get("restart", False)),
        ],
        dtype=float,
    )


def build_observation_vector(raw_obs: Dict[str, Any], torus_size: float) -> np.ndarray:
    """
    Build a low-dimensional observation vector o(s) from raw env observation.

    Components (all continuous and fixed-size):
        - sin/cos of adventurer position (4)
        - sin/cos of ghost position (4)
        - relative vector to ghost (2, scaled by torus size)
        - relative vector to nearest goblet (2, scaled by torus size)
        - nearest goblet type (1) and mean goblet type (1)
        - event flags caught/pick_pos/pick_neg/restart (4)
    """
    adventurer = np.asarray(raw_obs["adventurer"], dtype=float).reshape(2)
    ghost = np.asarray(raw_obs["ghost"], dtype=float).reshape(2)
    goblets_pos = np.asarray(raw_obs["goblets_pos"], dtype=float)
    goblets_type = np.asarray(raw_obs["goblets_type"], dtype=float)

    rel_ghost = torus_delta(adventurer, ghost, torus_size) / torus_size

    if goblets_pos.size > 0:
        deltas = torus_delta(adventurer, goblets_pos, torus_size)
        dists = np.linalg.norm(deltas, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_delta = deltas[nearest_idx] / torus_size
        nearest_type = float(goblets_type[nearest_idx])
        mean_type = float(np.mean(goblets_type))
    else:
        nearest_delta = np.zeros(2, dtype=float)
        nearest_type = 0.0
        mean_type = 0.0

    events = build_event_flags(raw_obs)

    obs_vec = np.concatenate(
        [
            _encode_position(adventurer, torus_size),
            _encode_position(ghost, torus_size),
            rel_ghost,
            nearest_delta,
            np.array([nearest_type, mean_type], dtype=float),
            events,
        ],
        axis=0,
    )
    return obs_vec
