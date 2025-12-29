"""Shared utilities for probe computations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy, apply_rho_clip
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv


def clip_action(action: np.ndarray, v_max: float) -> np.ndarray:
    """Clip action per component to [-v_max, v_max]."""
    if v_max <= 0.0:
        return action
    return np.clip(action, -v_max, v_max)


def mc_bar_phi(
    env: TorusGobletGhostEnv,
    policy: LinearGaussianPolicy,
    psi: np.ndarray,
    rng: np.random.Generator,
    k_mc: int,
) -> np.ndarray:
    """Monte Carlo estimate of E_pi[phi(s', a')] with frozen policy."""
    if k_mc <= 0:
        return env.compute_features(np.zeros(policy.action_dim, dtype=float))["phi"]
    phis = []
    for _ in range(k_mc):
        action = policy.sample_action(psi, rng)
        action = clip_action(action, env.v_max)
        phis.append(env.compute_features(action)["phi"])
    return np.mean(np.stack(phis, axis=0), axis=0)


def collect_critic_batch(
    env: TorusGobletGhostEnv,
    mu_policy: LinearGaussianPolicy,
    pi_policy: LinearGaussianPolicy,
    rng: np.random.Generator,
    batch_size: int,
    k_mc: int,
    rho_clip: Optional[float] = None,
    disable_rho_clip: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect a batch of critic transitions under behavior policy mu."""
    zero_action = np.zeros(2, dtype=float)
    seed_max = np.iinfo(np.int32).max

    env.reset(seed=int(rng.integers(0, seed_max)))
    feat0 = env.compute_features(zero_action)
    psi = feat0["psi"]

    obs_vecs = []
    phis = []
    bar_phis = []
    rhos = []
    rewards = []

    for _ in range(batch_size):
        a = mu_policy.sample_action(psi, rng)
        a = clip_action(a, env.v_max)

        logp_pi = pi_policy.log_prob(a, psi)
        logp_mu = mu_policy.log_prob(a, psi)
        rho_raw = float(np.exp(logp_pi - logp_mu))
        rho = apply_rho_clip(rho_raw, rho_clip, disable=disable_rho_clip)

        obs, reward, terminated, truncated, info = env.step(a)

        phi = info["phi"]
        psi_next = info["psi_next"]

        bar_phi = mc_bar_phi(env, pi_policy, psi_next, rng, k_mc=k_mc)

        obs_vecs.append(info["obs_vec"])
        phis.append(phi)
        bar_phis.append(bar_phi)
        rewards.append(float(reward))
        rhos.append(rho)

        psi = psi_next
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
            feat0 = env.compute_features(zero_action)
            psi = feat0["psi"]

    return {
        "obs_vec": np.stack(obs_vecs, axis=0),
        "phi": np.stack(phis, axis=0),
        "bar_phi": np.stack(bar_phis, axis=0),
        "rho": np.asarray(rhos, dtype=float),
        "reward": np.asarray(rewards, dtype=float),
    }


def summarize_rho(rho: np.ndarray) -> Dict[str, float]:
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.size == 0 or not np.isfinite(rho_arr).any():
        return {
            "rho_mean": float("nan"),
            "rho2_mean": float("nan"),
            "rho_min": float("nan"),
            "rho_max": float("nan"),
            "rho_p95": float("nan"),
            "rho_p99": float("nan"),
        }
    rho_arr = rho_arr[np.isfinite(rho_arr)]
    rho2_arr = rho_arr * rho_arr
    return {
        "rho_mean": float(np.mean(rho_arr)),
        "rho2_mean": float(np.mean(rho2_arr)),
        "rho_min": float(np.min(rho_arr)),
        "rho_max": float(np.max(rho_arr)),
        "rho_p95": float(np.quantile(rho_arr, 0.95)),
        "rho_p99": float(np.quantile(rho_arr, 0.99)),
    }


def rho_clip_metadata(rho_clip: Optional[float], disable_rho_clip: bool) -> Dict[str, float]:
    active = rho_clip is not None and float(rho_clip) > 0.0 and not disable_rho_clip
    return {
        "rho_clip": float(rho_clip) if rho_clip is not None else float("nan"),
        "rho_clip_active": 1.0 if active else 0.0,
    }
