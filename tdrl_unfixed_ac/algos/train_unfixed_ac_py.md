"""Training loop for unfixed actor-critic."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import LinearGaussianPolicy, critic_value, importance_ratio, project_to_ball
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv, load_config as load_env_config
from tdrl_unfixed_ac.probes import ProbeManager
from tdrl_unfixed_ac.utils.seeding import Seeder

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_TRAIN_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train_unfixed_ac.yaml"

DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "outer_iters": 50,
    "trajectories": 6,
    "horizon": 200,
    "gamma": 0.95,
    "alpha_w": 0.2,
    "alpha_pi": 0.1,
    "beta": 0.2,
    "sigma_mu": 0.2,
    "sigma_pi": 0.2,
    "K_mc": 4,
    "theta_radius": 4.0,
    "theta_init_scale": 0.1,
    "w_init_scale": 0.1,
    "checkpoint_every": 5,
    "log_every": 1,
    "output_dir": "outputs/unfixed_ac",
    "env_config_path": None,
    "env": {},
    "probes": {
        "enabled": False,
        "every": 0,
        "plateau": {
            "enabled": False,
            "window": 5,
            "tol": 1e-3,
            "cooldown": 5,
            "min_iter": 5,
        },
        "fixed_point": {
            "enabled": True,
            "batch_size": 4096,
            "max_iters": 200,
            "tol": 1e-4,
        },
        "stability": {
            "enabled": True,
            "batch_size": 4096,
            "power_iters": 20,
        },
        "distribution": {
            "enabled": True,
            "num_samples": 512,
        },
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_train_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load training config from JSON/YAML file with defaults."""
    config = deepcopy(DEFAULT_TRAIN_CONFIG)
    if path is None and DEFAULT_TRAIN_CONFIG_PATH.exists():
        path = str(DEFAULT_TRAIN_CONFIG_PATH)
    if path is None:
        return config
    text = Path(path).read_text()
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if payload:
        _deep_update(config, payload)
    return config


def _clip_action(action: np.ndarray, v_max: float) -> np.ndarray:
    norm = float(np.linalg.norm(action))
    if norm > v_max and norm > 0.0:
        return action / norm * v_max
    return action


def _mc_bar_phi(
    env: TorusGobletGhostEnv,
    policy: LinearGaussianPolicy,
    psi: np.ndarray,
    rng: np.random.Generator,
    k_mc: int,
) -> np.ndarray:
    if k_mc <= 0:
        return env.compute_features(np.zeros(policy.action_dim, dtype=float))["phi"]
    phis = []
    for _ in range(k_mc):
        action = policy.sample_action(psi, rng)
        action = _clip_action(action, env.v_max)
        phi = env.compute_features(action)["phi"]
        phis.append(phi)
    return np.mean(np.stack(phis, axis=0), axis=0)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def train_unfixed_ac(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    seed = int(cfg.get("seed", 0))
    outer_iters = int(cfg.get("outer_iters", 1))
    trajectories = int(cfg.get("trajectories", 1))
    horizon = int(cfg.get("horizon", 1))
    gamma = float(cfg.get("gamma", 0.0))
    alpha_w = float(cfg.get("alpha_w", 0.0))
    alpha_pi = float(cfg.get("alpha_pi", 0.0))
    beta = float(cfg.get("beta", 0.0))
    sigma_mu = float(cfg.get("sigma_mu", 1.0))
    sigma_pi = float(cfg.get("sigma_pi", 1.0))
    k_mc = int(cfg.get("K_mc", 1))
    theta_radius = float(cfg.get("theta_radius", 0.0))
    theta_init_scale = float(cfg.get("theta_init_scale", 0.1))
    w_init_scale = float(cfg.get("w_init_scale", 0.1))
    checkpoint_every = int(cfg.get("checkpoint_every", 0))
    log_every = int(cfg.get("log_every", 1))
    output_dir = Path(cfg.get("output_dir", "outputs/unfixed_ac"))

    env_cfg_path = cfg.get("env_config_path", None)
    env_cfg = load_env_config(env_cfg_path) if env_cfg_path else load_env_config()
    env_cfg.update(cfg.get("env", {}))
    if env_cfg.get("seed") is None:
        env_cfg["seed"] = seed
    env = TorusGobletGhostEnv(config=env_cfg)

    seeder = Seeder(seed)
    init_rng = seeder.spawn()
    rollout_rng = seeder.spawn()

    action_dim = int(env.critic_features_map.action_dim)
    actor_dim = int(env.actor_feature_dim)
    feature_dim = int(env.feature_dim)

    theta_pi = init_rng.normal(loc=0.0, scale=theta_init_scale, size=(actor_dim, action_dim))
    theta_mu = np.array(theta_pi, copy=True)
    w = init_rng.normal(loc=0.0, scale=w_init_scale, size=feature_dim)

    teacher_w = np.array(env.teacher_reward.w_R, copy=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    probe_manager = ProbeManager(
        cfg.get("probes", {}),
        output_dir=output_dir,
        env_config=env_cfg,
        seed=seed,
        alpha_w=alpha_w,
        gamma=gamma,
        k_mc=k_mc,
        sigma_mu=sigma_mu,
        sigma_pi=sigma_pi,
    )
    probe_defaults = probe_manager.log_defaults()

    with (output_dir / "config.json").open("w") as handle:
        json.dump({k: _json_ready(v) for k, v in cfg.items()}, handle, indent=2)

    logs = []
    total_steps = max(trajectories * horizon, 1)
    zero_action = np.zeros(action_dim, dtype=float)
    seed_max = np.iinfo(np.int32).max

    for n in range(outer_iters):
        mu_policy = LinearGaussianPolicy(theta=theta_mu, sigma=sigma_mu)
        pi_policy = LinearGaussianPolicy(theta=theta_pi, sigma=sigma_pi)

        grad_w = np.zeros_like(w)
        grad_theta = np.zeros_like(theta_pi)
        td_errors = []
        rho_sq = []

        for _ in range(trajectories):
            env.reset(seed=int(rollout_rng.integers(0, seed_max)))
            for _ in range(horizon):
                psi = env.compute_features(zero_action)["psi"]

                action = mu_policy.sample_action(psi, rollout_rng)
                action = _clip_action(action, env.v_max)
                phi = env.compute_features(action)["phi"]

                logmu = mu_policy.log_prob(action, psi)
                logpi = pi_policy.log_prob(action, psi)
                rho = importance_ratio(logpi, logmu)

                _, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("Environment should be continuing but returned a terminal flag.")

                psi_next = env.compute_features(zero_action)["psi"]
                bar_phi = _mc_bar_phi(env, pi_policy, psi_next, rollout_rng, k_mc)

                delta = reward + gamma * critic_value(w, bar_phi) - critic_value(w, phi)
                g = pi_policy.score(action, psi)

                grad_w += rho * delta * phi
                grad_theta += rho * delta * g

                td_errors.append(delta)
                rho_sq.append(rho * rho)

        scale = 1.0 / total_steps
        w = w + alpha_w * scale * grad_w
        theta_pi = theta_pi + alpha_pi * scale * grad_theta
        theta_mu = (1.0 - beta) * theta_mu + beta * theta_pi

        theta_pi = project_to_ball(theta_pi, theta_radius)
        theta_mu = project_to_ball(theta_mu, theta_radius)

        td_loss = float(np.mean(np.square(td_errors))) if td_errors else float("nan")
        critic_teacher_error = float(np.dot(w - teacher_w, w - teacher_w) / feature_dim)
        tracking_gap = float(np.linalg.norm(theta_pi - theta_mu) ** 2 / actor_dim)
        mean_rho2 = float(np.mean(rho_sq)) if rho_sq else float("nan")
        w_norm = float(np.linalg.norm(w))

        log_row = {
            "iter": n,
            "td_loss": td_loss,
            "critic_teacher_error": critic_teacher_error,
            "tracking_gap": tracking_gap,
            "mean_rho2": mean_rho2,
            "w_norm": w_norm,
            **probe_defaults,
        }
        probe_updates = probe_manager.maybe_run(
            iteration=n, td_loss=td_loss, w=w, theta_mu=theta_mu, theta_pi=theta_pi
        )
        if probe_updates:
            log_row.update(probe_updates)
        logs.append(log_row)

        if log_every > 0 and (n % log_every == 0):
            print(
                "iter {:03d} | td_loss {:.4f} | teacher_err {:.4f} | gap {:.4f} | rho2 {:.4f} | w_norm {:.3f}".format(
                    n, td_loss, critic_teacher_error, tracking_gap, mean_rho2, w_norm
                )
            )

        if checkpoint_every > 0 and (n + 1) % checkpoint_every == 0:
            np.savez(
                checkpoint_dir / f"iter_{n:04d}.npz",
                theta_mu=theta_mu,
                theta_pi=theta_pi,
                w=w,
                iter=n,
            )

    csv_path = output_dir / "learning_curves.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(logs[0].keys()) if logs else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)

    np.savez(
        checkpoint_dir / "final.npz",
        theta_mu=theta_mu,
        theta_pi=theta_pi,
        w=w,
        iter=outer_iters - 1,
    )

    return {"output_dir": str(output_dir), "csv_path": str(csv_path), "logs": logs}
