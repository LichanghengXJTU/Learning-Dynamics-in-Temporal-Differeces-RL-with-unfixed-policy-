"""Distribution discrepancy probe between d_mu(s) and d_pi(s)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv
from tdrl_unfixed_ac.probes.common import clip_action


def _collect_obs_vectors(
    env: TorusGobletGhostEnv,
    policy: LinearGaussianPolicy,
    rng: np.random.Generator,
    num_samples: int,
) -> np.ndarray:
    action_dim = int(policy.action_dim)
    zero_action = np.zeros(action_dim, dtype=float)
    seed_max = np.iinfo(np.int32).max

    env.reset(seed=int(rng.integers(0, seed_max)))
    obs_vecs = []
    for _ in range(num_samples):
        features = env.compute_features(zero_action)
        obs_vecs.append(features["obs_vec"])

        psi = features["psi"]
        action = policy.sample_action(psi, rng)
        action = clip_action(action, env.v_max)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Environment should be continuing but returned a terminal flag.")

    return np.stack(obs_vecs, axis=0)


def _median_bandwidth(x: np.ndarray, max_samples: int, rng: np.random.Generator) -> float:
    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
    diffs = x[:, None, :] - x[None, :, :]
    dist_sq = np.sum(diffs * diffs, axis=-1)
    median = float(np.median(dist_sq))
    if median <= 1e-12:
        return 1.0
    return np.sqrt(0.5 * median)


def _mmd_rbf(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    combined = np.concatenate([x, y], axis=0)
    sigma = _median_bandwidth(combined, max_samples=300, rng=rng)
    denom = 2.0 * sigma * sigma

    diff_xx = x[:, None, :] - x[None, :, :]
    diff_yy = y[:, None, :] - y[None, :, :]
    diff_xy = x[:, None, :] - y[None, :, :]

    k_xx = np.exp(-np.sum(diff_xx * diff_xx, axis=-1) / denom)
    k_yy = np.exp(-np.sum(diff_yy * diff_yy, axis=-1) / denom)
    k_xy = np.exp(-np.sum(diff_xy * diff_xy, axis=-1) / denom)

    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return mmd2, sigma


def run_distribution_probe(
    *,
    env_config: Dict[str, Any],
    theta_mu: np.ndarray,
    theta_pi: np.ndarray,
    sigma_mu: float,
    sigma_pi: float,
    num_samples: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Compare visitation distributions using MMD over observation vectors."""
    base_seed = int(seed) if seed is not None else 0
    rng_mu = np.random.default_rng(base_seed + 1)
    rng_pi = np.random.default_rng(base_seed + 2)

    env_mu = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 11))
    env_pi = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 13))

    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu))
    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi))

    obs_mu = _collect_obs_vectors(env_mu, mu_policy, rng_mu, num_samples)
    obs_pi = _collect_obs_vectors(env_pi, pi_policy, rng_pi, num_samples)

    mmd2, sigma = _mmd_rbf(obs_mu, obs_pi, rng_mu)
    mean_l2 = float(np.linalg.norm(obs_mu.mean(axis=0) - obs_pi.mean(axis=0)))

    return {
        "mmd2": float(mmd2),
        "mmd_sigma": float(sigma),
        "mean_l2": float(mean_l2),
        "num_samples": int(num_samples),
    }
