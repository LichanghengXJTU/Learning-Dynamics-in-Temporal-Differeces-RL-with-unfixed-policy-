"""Critic feature map using random Fourier features plus explicit events."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.features.observations import build_event_flags


class CriticFeatureMap:
    """Compute phi(s, a) with random Fourier features and appended event features."""

    def __init__(
        self,
        obs_dim: int,
        dim: int,
        *,
        action_dim: int = 2,
        sigma: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.event_dim = 4  # caught, pick_pos, pick_neg, action_penalty
        self.base_dim = max(self.dim - self.event_dim, 1)
        self.input_dim = self.obs_dim + self.action_dim + 1  # +1 for bias term in x
        self.rng = rng if rng is not None else np.random.default_rng()

        self.W = self.rng.normal(loc=0.0, scale=sigma, size=(self.base_dim, self.input_dim))
        self.b = self.rng.uniform(low=0.0, high=2.0 * np.pi, size=self.base_dim)

    def __call__(self, obs_vec: np.ndarray, action: np.ndarray, raw_obs: Dict[str, Any]) -> np.ndarray:
        obs_vec = np.asarray(obs_vec, dtype=float).reshape(self.obs_dim)
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        x = np.concatenate([obs_vec, action, np.array([1.0], dtype=float)], axis=0)

        proj = self.W @ x + self.b
        base_features = np.sqrt(2.0 / self.base_dim) * np.cos(proj)

        events = build_event_flags(raw_obs)
        action_penalty = float(np.dot(action, action))
        event_features = np.concatenate([events[:3], np.array([action_penalty], dtype=float)])

        phi_full = np.concatenate([base_features, event_features], axis=0)
        if phi_full.shape[0] < self.dim:
            phi_full = np.pad(phi_full, (0, self.dim - phi_full.shape[0]))
        return phi_full[: self.dim]
