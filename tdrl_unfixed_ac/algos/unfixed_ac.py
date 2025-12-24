"""Unfixed actor-critic primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def policy_mean(theta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Compute m(s)=theta^T psi(s)/sqrt(N_act)."""
    theta = np.asarray(theta, dtype=float)
    psi = np.asarray(psi, dtype=float).reshape(theta.shape[0])
    return (theta.T @ psi) / np.sqrt(theta.shape[0])


@dataclass
class LinearGaussianPolicy:
    """Linear-Gaussian policy with diagonal covariance."""

    theta: np.ndarray
    sigma: float

    def mean(self, psi: np.ndarray) -> np.ndarray:
        return policy_mean(self.theta, psi)

    @property
    def action_dim(self) -> int:
        return int(self.theta.shape[1])

    @property
    def actor_dim(self) -> int:
        return int(self.theta.shape[0])

    def sample_action(self, psi: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        mean = self.mean(psi)
        return rng.normal(loc=mean, scale=self.sigma, size=self.action_dim)

    def log_prob(self, action: np.ndarray, psi: np.ndarray) -> float:
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        mean = self.mean(psi)
        diff = action - mean
        var = self.sigma * self.sigma
        log_norm = -0.5 * self.action_dim * np.log(2.0 * np.pi * var)
        return float(log_norm - 0.5 * np.dot(diff, diff) / var)

    def score(self, action: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Return grad_theta log pi(a|s) with shape theta."""
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        psi = np.asarray(psi, dtype=float).reshape(self.actor_dim)
        mean = self.mean(psi)
        diff = action - mean
        scale = 1.0 / (self.sigma * self.sigma * np.sqrt(self.actor_dim))
        return np.outer(psi, diff) * scale


def importance_ratio(logpi: float, logmu: float) -> float:
    """Compute rho = exp(logpi - logmu)."""
    return float(np.exp(logpi - logmu))


def critic_value(w: np.ndarray, phi: np.ndarray) -> float:
    """Compute Q_w(s,a) = (w @ phi)/sqrt(N)."""
    w = np.asarray(w, dtype=float).reshape(-1)
    phi = np.asarray(phi, dtype=float).reshape(-1)
    return float(np.dot(w, phi) / np.sqrt(w.shape[0]))


def project_to_ball(theta: np.ndarray, radius: float) -> np.ndarray:
    """Project theta onto L2 ball of given radius."""
    if radius <= 0.0:
        return theta
    norm = float(np.linalg.norm(theta))
    if norm > radius:
        return theta * (radius / (norm + 1e-12))
    return theta
