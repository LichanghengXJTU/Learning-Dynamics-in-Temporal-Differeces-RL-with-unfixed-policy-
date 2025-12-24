"""Geometry helpers for torus domains."""

from __future__ import annotations

import numpy as np


def wrap_torus(x: np.ndarray, size: float) -> np.ndarray:
    """Wrap coordinates onto a torus of given size."""
    return np.mod(x, size)


def torus_delta(a: np.ndarray, b: np.ndarray, size: float) -> np.ndarray:
    """Shortest signed delta on a torus from a to b."""
    half = 0.5 * size
    return (b - a + half) % size - half


def torus_distance(a: np.ndarray, b: np.ndarray, size: float) -> float:
    """Euclidean distance using torus-aware shortest delta."""
    delta = torus_delta(a, b, size)
    return float(np.linalg.norm(delta))

