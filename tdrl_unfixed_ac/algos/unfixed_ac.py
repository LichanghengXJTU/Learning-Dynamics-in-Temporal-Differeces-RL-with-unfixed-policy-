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


def _squash_action(u: np.ndarray, v_max: float) -> np.ndarray:
    return v_max * np.tanh(u)


def _unsquash_action(action: np.ndarray, v_max: float, eps: float = 1e-6) -> np.ndarray:
    if v_max <= 0.0:
        raise ValueError("v_max must be positive.")
    scaled = np.clip(action / v_max, -1.0 + eps, 1.0 - eps)
    return np.arctanh(scaled)


def _log1m_tanh2(u: np.ndarray) -> np.ndarray:
    return 2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))


def _gaussian_log_prob(u: np.ndarray, mean: np.ndarray, sigma: float) -> float:
    diff = u - mean
    var = sigma * sigma
    log_norm = -0.5 * diff.size * np.log(2.0 * np.pi * var)
    return float(log_norm - 0.5 * np.dot(diff, diff) / var)


@dataclass
class LinearGaussianPolicy:
    """Linear-Gaussian policy with diagonal covariance."""

    theta: np.ndarray
    sigma: float
    v_max: float

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
        u = rng.normal(loc=mean, scale=self.sigma, size=self.action_dim)
        return _squash_action(u, self.v_max)

    def pre_squash(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        return _unsquash_action(action, self.v_max)

    def log_prob_pre_squash(self, u: np.ndarray, psi: np.ndarray) -> float:
        u = np.asarray(u, dtype=float).reshape(self.action_dim)
        mean = self.mean(psi)
        return _gaussian_log_prob(u, mean, self.sigma)

    def log_prob(self, action: np.ndarray, psi: np.ndarray) -> float:
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        mean = self.mean(psi)
        u = _unsquash_action(action, self.v_max)
        log_base = _gaussian_log_prob(u, mean, self.sigma)
        log_det = float(np.sum(np.log(self.v_max) + _log1m_tanh2(u)))
        return log_base - log_det

    def score(self, action: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Return grad_theta log pi(a|s) with shape theta."""
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        psi = np.asarray(psi, dtype=float).reshape(self.actor_dim)
        mean = self.mean(psi)
        u = _unsquash_action(action, self.v_max)
        diff = u - mean
        scale = 1.0 / (self.sigma * self.sigma * np.sqrt(self.actor_dim))
        return np.outer(psi, diff) * scale


def importance_ratio(logpi: float, logmu: float) -> float:
    """Compute rho = exp(logpi - logmu)."""
    return float(np.exp(logpi - logmu))


def apply_rho_clip(rho: float, rho_clip: Optional[float], *, disable: bool = False) -> float:
    """Optionally clip rho to an upper bound."""
    if disable:
        return float(rho)
    if rho_clip is None:
        return float(rho)
    try:
        rho_clip_val = float(rho_clip)
    except (TypeError, ValueError):
        return float(rho)
    if rho_clip_val <= 0.0:
        return float(rho)
    return float(min(rho, rho_clip_val))


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
