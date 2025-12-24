"""Actor feature map producing bounded psi(s)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ActorFeatureMap:
    """Fixed random projection with norm clipping to bound psi(s)."""

    def __init__(self, obs_dim: int, dim: int, c_psi: float = 1.0, rng: Optional[np.random.Generator] = None) -> None:
        self.obs_dim = int(obs_dim)
        self.dim = int(dim)
        self.c_psi = float(c_psi)
        self.rng = rng if rng is not None else np.random.default_rng()

        scale = 1.0 / max(np.sqrt(self.obs_dim), 1.0)
        self.W = self.rng.normal(loc=0.0, scale=scale, size=(self.dim, self.obs_dim))
        self.b = self.rng.normal(loc=0.0, scale=0.1, size=self.dim)

    def __call__(self, obs_vec: np.ndarray) -> np.ndarray:
        obs_vec = np.asarray(obs_vec, dtype=float).reshape(self.obs_dim)
        z = self.W @ obs_vec + self.b
        psi = np.tanh(z)
        norm = float(np.linalg.norm(psi))
        if norm > self.c_psi:
            psi = psi * (self.c_psi / (norm + 1e-12))
        return psi
