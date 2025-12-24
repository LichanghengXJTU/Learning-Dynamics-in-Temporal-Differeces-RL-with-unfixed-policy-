"""Teacher reward parameterization over critic features."""

from __future__ import annotations

from typing import Optional

import numpy as np

from tdrl_unfixed_ac.features.critic_features import CriticFeatureMap


class TeacherReward:
    """Fixed teacher vector w_R with interpretable event weights."""

    def __init__(
        self,
        critic_features: CriticFeatureMap,
        *,
        lambda_action: float = 0.1,
        base_scale: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.critic_features = critic_features
        self.lambda_action = float(lambda_action)
        self.base_scale = float(base_scale)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.w_R = self._init_teacher_vector()

    def _init_teacher_vector(self) -> np.ndarray:
        base = self.rng.normal(loc=0.0, scale=self.base_scale, size=self.critic_features.base_dim)
        event_weights = np.array([-1.0, 1.0, -1.0, -self.lambda_action], dtype=float)
        w_full = np.concatenate([base, event_weights])
        if w_full.shape[0] < self.critic_features.dim:
            w_full = np.pad(w_full, (0, self.critic_features.dim - w_full.shape[0]))
        return w_full[: self.critic_features.dim]

    def __call__(self, phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        scale = np.sqrt(self.critic_features.dim)
        return float(np.dot(self.w_R, phi) / scale)
