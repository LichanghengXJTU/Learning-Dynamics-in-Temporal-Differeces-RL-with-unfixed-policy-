"""Fixed point drift probe for the critic under frozen policies."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv
from tdrl_unfixed_ac.probes.common import collect_critic_batch


def run_fixed_point_probe(
    *,
    env_config: Dict[str, Any],
    theta_mu: np.ndarray,
    theta_pi: np.ndarray,
    w_init: np.ndarray,
    sigma_mu: float,
    sigma_pi: float,
    alpha_w: float,
    gamma: float,
    k_mc: int,
    batch_size: int,
    max_iters: int,
    tol: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Estimate the TD fixed point w_sharp for frozen (mu, pi)."""
    rng = np.random.default_rng(seed)
    env = TorusGobletGhostEnv(config=env_config, rng=rng)

    mu_policy = LinearGaussianPolicy(theta=np.array(theta_mu, copy=True), sigma=float(sigma_mu))
    pi_policy = LinearGaussianPolicy(theta=np.array(theta_pi, copy=True), sigma=float(sigma_pi))

    batch = collect_critic_batch(env, mu_policy, pi_policy, rng, batch_size, k_mc)

    w = np.array(w_init, copy=True)
    feature_dim = w.shape[0]
    scale = np.sqrt(feature_dim)
    diff = gamma * batch["bar_phi"] - batch["phi"]

    converged = False
    steps = 0
    for step in range(int(max_iters)):
        delta = batch["reward"] + (diff @ w) / scale
        grad = (batch["rho"] * delta)[:, None] * batch["phi"]
        w_next = w + alpha_w * grad.mean(axis=0)
        steps = step + 1
        if float(np.linalg.norm(w_next - w)) <= tol:
            converged = True
            w = w_next
            break
        w = w_next

    return {
        "w_sharp": w,
        "converged": converged,
        "num_iters": steps,
        "batch_size": int(batch_size),
        "tol": float(tol),
    }
