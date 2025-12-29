"""Stability probe via power iteration on linearized critic updates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv
from tdrl_unfixed_ac.probes.common import collect_critic_batch, rho_clip_metadata, summarize_rho


def run_stability_probe(
    *,
    env_config: Dict[str, Any],
    theta_mu: np.ndarray,
    theta_pi: np.ndarray,
    sigma_mu: float,
    sigma_pi: float,
    alpha_w: float,
    train_step_scale: float,
    gamma: float,
    k_mc: int,
    batch_size: int,
    power_iters: int,
    rho_clip: Optional[float],
    disable_rho_clip: bool,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Estimate local amplification (spectral radius proxy) for critic updates."""
    rng = np.random.default_rng(seed)
    v_max = float(env_config.get("v_max", 1.0))
    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu), v_max=v_max)
    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi), v_max=v_max)

    def _estimate_proxy(batch: Dict[str, np.ndarray], local_rng: np.random.Generator) -> float:
        phi = batch["phi"]
        diff = gamma * batch["bar_phi"] - phi
        rho = batch["rho"]
        feature_dim = phi.shape[1]

        v = local_rng.normal(size=feature_dim)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-12:
            v = np.ones(feature_dim, dtype=float) / np.sqrt(feature_dim)
        else:
            v = v / v_norm

        spectral = float("nan")
        for _ in range(int(power_iters)):
            dot_term = diff @ v
            update = (rho * dot_term)[:, None] * phi
            # Align probe step size with training critic updates.
            v_next = v + train_step_scale * update.sum(axis=0)
            norm_next = float(np.linalg.norm(v_next))
            if norm_next <= 1e-12:
                spectral = 0.0
                v = v_next
                break
            spectral = norm_next
            v = v_next / norm_next
        return float(spectral)

    reps = max(1, int(k_mc))
    proxies: list[float] = []
    rho_samples = []
    for _ in range(reps):
        env = TorusGobletGhostEnv(config=env_config, rng=rng)
        batch = collect_critic_batch(
            env,
            mu_policy,
            pi_policy,
            rng,
            batch_size,
            k_mc,
            rho_clip=rho_clip,
            disable_rho_clip=disable_rho_clip,
        )
        rho_samples.append(batch["rho"])
        proxies.append(_estimate_proxy(batch, rng))

    proxies_arr = np.asarray(proxies, dtype=float)
    stability_proxy_mean = float(np.mean(proxies_arr))
    stability_proxy_std = float(np.std(proxies_arr))
    rho_all = np.concatenate(rho_samples, axis=0) if rho_samples else np.asarray([], dtype=float)
    rho_stats = summarize_rho(rho_all)
    rho_meta = rho_clip_metadata(rho_clip, disable_rho_clip)

    return {
        "stability_proxy": stability_proxy_mean,
        "stability_proxy_mean": stability_proxy_mean,
        "stability_proxy_std": stability_proxy_std,
        "power_iters": int(power_iters),
        "batch_size": int(batch_size),
        "stability_probe_step_scale": float(train_step_scale),
        **rho_stats,
        **rho_meta,
    }
