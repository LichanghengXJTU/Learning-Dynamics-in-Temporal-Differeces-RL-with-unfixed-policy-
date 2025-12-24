"""Stability probe via power iteration on linearized critic updates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv
from tdrl_unfixed_ac.probes.common import collect_critic_batch


def run_stability_probe(
    *,
    env_config: Dict[str, Any],
    theta_mu: np.ndarray,
    theta_pi: np.ndarray,
    sigma_mu: float,
    sigma_pi: float,
    alpha_w: float,
    gamma: float,
    k_mc: int,
    batch_size: int,
    power_iters: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Estimate local amplification (spectral radius proxy) for critic updates."""
    rng = np.random.default_rng(seed)
    env = TorusGobletGhostEnv(config=env_config, rng=rng)

    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu))
    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi))

    batch = collect_critic_batch(env, mu_policy, pi_policy, rng, batch_size, k_mc)

    phi = batch["phi"]
    diff = gamma * batch["bar_phi"] - phi
    rho = batch["rho"]
    feature_dim = phi.shape[1]
    scale = np.sqrt(feature_dim)

    v = rng.normal(size=feature_dim)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        v = np.ones(feature_dim, dtype=float) / np.sqrt(feature_dim)
    else:
        v = v / v_norm

    spectral = float("nan")
    for _ in range(int(power_iters)):
        dot_term = diff @ v
        update = (rho * dot_term)[:, None] * phi
        v_next = v + (alpha_w / scale) * update.mean(axis=0)
        norm_next = float(np.linalg.norm(v_next))
        if norm_next <= 1e-12:
            spectral = 0.0
            v = v_next
            break
        spectral = norm_next
        v = v_next / norm_next

    return {
        "stability_proxy": float(spectral),
        "power_iters": int(power_iters),
        "batch_size": int(batch_size),
    }
