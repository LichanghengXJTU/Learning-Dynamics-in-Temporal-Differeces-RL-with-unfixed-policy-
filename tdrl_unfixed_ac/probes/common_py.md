"""Shared utilities for probe computations."""

from __future__ import annotations

from typing import Dict

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy, importance_ratio
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv


def clip_action(action: np.ndarray, v_max: float) -> np.ndarray:
    """Clip action to L2 ball of radius v_max."""
    norm = float(np.linalg.norm(action))
    if norm > v_max and norm > 0.0:
        return action / norm * v_max
    return action


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
) -> Dict[str, np.ndarray]:
    """Collect a batch of critic transitions under behavior policy mu."""
    action_dim = int(mu_policy.action_dim)
    zero_action = np.zeros(action_dim, dtype=float)
    seed_max = np.iinfo(np.int32).max

    env.reset(seed=int(rng.integers(0, seed_max)))

    phis = []
    bar_phis = []
    rhos = []
    rewards = []

    for _ in range(batch_size):
        psi = env.compute_features(zero_action)["psi"]
        action = mu_policy.sample_action(psi, rng)
        action = clip_action(action, env.v_max)
        phi = env.compute_features(action)["phi"]

        logmu = mu_policy.log_prob(action, psi)
        logpi = pi_policy.log_prob(action, psi)
        rho = importance_ratio(logpi, logmu)

        _, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Environment should be continuing but returned a terminal flag.")

        psi_next = env.compute_features(zero_action)["psi"]
        bar_phi = mc_bar_phi(env, pi_policy, psi_next, rng, k_mc)

        phis.append(phi)
        bar_phis.append(bar_phi)
        rhos.append(rho)
        rewards.append(reward)

    return {
        "phi": np.stack(phis, axis=0),
        "bar_phi": np.stack(bar_phis, axis=0),
        "rho": np.asarray(rhos, dtype=float),
        "reward": np.asarray(rewards, dtype=float),
    }
