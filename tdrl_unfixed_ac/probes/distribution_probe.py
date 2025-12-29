"""Distribution discrepancy probe between d_mu(s) and d_pi(s)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy, apply_rho_clip
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv
from tdrl_unfixed_ac.probes.common import clip_action, rho_clip_metadata, summarize_rho


def _collect_state_features(
    env: TorusGobletGhostEnv,
    policy: LinearGaussianPolicy,
    rng: np.random.Generator,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    action_dim = int(policy.action_dim)
    zero_action = np.zeros(action_dim, dtype=float)
    seed_max = np.iinfo(np.int32).max

    env.reset(seed=int(rng.integers(0, seed_max)))
    obs_vecs = []
    psi_vecs = []
    for _ in range(num_samples):
        features = env.compute_features(zero_action)
        obs_vecs.append(features["obs_vec"])

        psi = features["psi"]
        psi_vecs.append(psi)
        action = policy.sample_action(psi, rng)
        action = clip_action(action, env.v_max, env.clip_action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Environment should be continuing but returned a terminal flag.")

    return np.stack(obs_vecs, axis=0), np.stack(psi_vecs, axis=0)


def _collect_rho_samples(
    env: TorusGobletGhostEnv,
    mu_policy: LinearGaussianPolicy,
    pi_policy: LinearGaussianPolicy,
    rng: np.random.Generator,
    num_samples: int,
    *,
    rho_clip: Optional[float],
    disable_rho_clip: bool,
) -> np.ndarray:
    action_dim = int(mu_policy.action_dim)
    zero_action = np.zeros(action_dim, dtype=float)
    seed_max = np.iinfo(np.int32).max

    env.reset(seed=int(rng.integers(0, seed_max)))
    feat0 = env.compute_features(zero_action)
    psi = feat0["psi"]

    rhos = []
    for _ in range(num_samples):
        action = mu_policy.sample_action(psi, rng)
        action = clip_action(action, env.v_max, env.clip_action)
        logp_pi = pi_policy.log_prob(action, psi)
        logp_mu = mu_policy.log_prob(action, psi)
        rho_raw = float(np.exp(logp_pi - logp_mu))
        rho = apply_rho_clip(rho_raw, rho_clip, disable=disable_rho_clip)
        rhos.append(rho)

        _, _, terminated, truncated, info = env.step(action)
        psi = info["psi_next"]
        if terminated or truncated:
            env.reset(seed=int(rng.integers(0, seed_max)))
            feat0 = env.compute_features(zero_action)
            psi = feat0["psi"]

    return np.asarray(rhos, dtype=float)


def _policy_mean_batch(theta: np.ndarray, psi_batch: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    psi_batch = np.asarray(psi_batch, dtype=float)
    scale = np.sqrt(theta.shape[0])
    return (psi_batch @ theta) / scale


def _gaussian_kl_isotropic(
    mean_p: np.ndarray,
    sigma_p: float,
    mean_q: np.ndarray,
    sigma_q: float,
) -> np.ndarray:
    mean_p = np.asarray(mean_p, dtype=float)
    mean_q = np.asarray(mean_q, dtype=float)
    var_p = float(sigma_p) ** 2
    var_q = float(sigma_q) ** 2
    diff = mean_q - mean_p
    diff_norm_sq = np.sum(diff * diff, axis=-1)
    dim = mean_p.shape[-1]
    log_ratio = np.log(var_q / var_p)
    return 0.5 * (dim * (var_p / var_q) + diff_norm_sq / var_q - dim + dim * log_ratio)


def _estimate_action_tv(
    mean_pi: np.ndarray,
    sigma_pi: float,
    mean_mu: np.ndarray,
    sigma_mu: float,
    rng: np.random.Generator,
    num_action_samples: int,
    *,
    clip_log_ratio: float = 50.0,
) -> float:
    if num_action_samples <= 0:
        return float("nan")
    mean_pi = np.asarray(mean_pi, dtype=float)
    mean_mu = np.asarray(mean_mu, dtype=float)
    num_states, action_dim = mean_pi.shape
    if num_states == 0:
        return float("nan")
    actions = rng.normal(
        loc=mean_pi[:, None, :],
        scale=float(sigma_pi),
        size=(num_states, num_action_samples, action_dim),
    )
    var_pi = float(sigma_pi) ** 2
    var_mu = float(sigma_mu) ** 2
    log_norm_pi = -0.5 * action_dim * np.log(2.0 * np.pi * var_pi)
    log_norm_mu = -0.5 * action_dim * np.log(2.0 * np.pi * var_mu)
    diff_pi = actions - mean_pi[:, None, :]
    diff_mu = actions - mean_mu[:, None, :]
    logp_pi = log_norm_pi - 0.5 * np.sum(diff_pi * diff_pi, axis=-1) / var_pi
    logp_mu = log_norm_mu - 0.5 * np.sum(diff_mu * diff_mu, axis=-1) / var_mu
    log_ratio = np.clip(logp_mu - logp_pi, -clip_log_ratio, clip_log_ratio)
    ratio = np.exp(log_ratio)
    tv_per_state = 0.5 * np.mean(np.abs(1.0 - ratio), axis=-1)
    return float(np.mean(tv_per_state))


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
    squash_action: bool,
    num_samples: int,
    seed: Optional[int],
    action_samples: int = 64,
    rho_clip: Optional[float] = None,
    disable_rho_clip: bool = False,
) -> Dict[str, Any]:
    """Compare visitation distributions using MMD and action divergence."""
    base_seed = int(seed) if seed is not None else 0
    rng_mu = np.random.default_rng(base_seed + 1)
    rng_pi = np.random.default_rng(base_seed + 2)

    env_mu = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 11))
    env_pi = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 13))
    env_rho = TorusGobletGhostEnv(config=env_config, rng=np.random.default_rng(base_seed + 17))

    mu_policy = LinearGaussianPolicy(
        theta=np.array(theta_mu, copy=True),
        sigma=float(sigma_mu),
        v_max=env_mu.v_max,
        squash_action=bool(squash_action),
    )
    pi_policy = LinearGaussianPolicy(
        theta=np.array(theta_pi, copy=True),
        sigma=float(sigma_pi),
        v_max=env_mu.v_max,
        squash_action=bool(squash_action),
    )

    obs_mu, psi_mu = _collect_state_features(env_mu, mu_policy, rng_mu, num_samples)
    obs_pi, psi_pi = _collect_state_features(env_pi, pi_policy, rng_pi, num_samples)

    mmd2, sigma = _mmd_rbf(obs_mu, obs_pi, rng_mu)
    mean_l2 = float(np.linalg.norm(obs_mu.mean(axis=0) - obs_pi.mean(axis=0)))

    psi_samples = np.concatenate([psi_mu, psi_pi], axis=0)
    mean_pi = _policy_mean_batch(theta_pi, psi_samples)
    mean_mu = _policy_mean_batch(theta_mu, psi_samples)
    kl_vals = _gaussian_kl_isotropic(mean_pi, sigma_pi, mean_mu, sigma_mu)
    dist_action_kl = float(np.mean(kl_vals)) if kl_vals.size > 0 else float("nan")
    dist_action_tv = _estimate_action_tv(
        mean_pi,
        sigma_pi,
        mean_mu,
        sigma_mu,
        rng_pi,
        int(action_samples),
    )
    rho_samples = _collect_rho_samples(
        env_rho,
        mu_policy,
        pi_policy,
        rng_mu,
        num_samples,
        rho_clip=rho_clip,
        disable_rho_clip=disable_rho_clip,
    )
    rho_stats = summarize_rho(rho_samples)
    rho_meta = rho_clip_metadata(rho_clip, disable_rho_clip)

    return {
        "mmd2": float(mmd2),
        "mmd_sigma": float(sigma),
        "mean_l2": float(mean_l2),
        "num_samples": int(num_samples),
        "dist_action_kl": float(dist_action_kl),
        "dist_action_tv": float(dist_action_tv),
        "action_samples": int(action_samples),
        **rho_stats,
        **rho_meta,
    }
