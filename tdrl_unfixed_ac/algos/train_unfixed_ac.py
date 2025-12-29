"""Training loop for unfixed actor-critic."""

from __future__ import annotations

import csv
import json
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.algos.unfixed_ac import (
    LinearGaussianPolicy,
    apply_rho_clip,
    critic_value,
    project_to_ball,
)
from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv, load_config as load_env_config
from tdrl_unfixed_ac.probes import ProbeManager
from tdrl_unfixed_ac.reporting import generate_run_report
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
    "rho_clip": None,
    "disable_rho_clip": False,
    "checkpoint_every": 5,
    "log_every": 1,
    "report_every": 0,
    "report_every_seconds": 0,
    "output_dir": "outputs/unfixed_ac",
    "resume": False,
    "resume_from": None,
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
        "q_kernel": {
            "enabled": False,
            "batch_size": 8,
            "max_horizon": 200,
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
    if v_max <= 0.0:
        return action
    return np.clip(action, -v_max, v_max)


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
        phi = env.compute_features(action)["phi"]
        phis.append(phi)
    return np.mean(np.stack(phis, axis=0), axis=0)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_existing_logs(csv_path: Path) -> list[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _last_logged_iter(logs: list[Dict[str, Any]]) -> Optional[int]:
    for row in reversed(logs):
        raw = row.get("iter")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(float(raw))
        except ValueError:
            continue
    return None


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
    rho_clip = cfg.get("rho_clip", None)
    disable_rho_clip = bool(cfg.get("disable_rho_clip", False))
    checkpoint_every = int(cfg.get("checkpoint_every", 0))
    log_every = int(cfg.get("log_every", 1))
    report_every = int(cfg.get("report_every", 0) or 0)
    report_every_seconds = float(cfg.get("report_every_seconds", 0) or 0.0)
    output_dir = Path(cfg.get("output_dir", "outputs/unfixed_ac"))
    resume = bool(cfg.get("resume", False))
    resume_from = cfg.get("resume_from", None)
    total_steps = max(trajectories * horizon, 1)
    train_step_scale = alpha_w / total_steps

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "learning_curves.csv"
    exception: Optional[str] = None

    csv_handle = None
    try:
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

        logs = _load_existing_logs(csv_path)
        last_logged = _last_logged_iter(logs)
        start_iter = 0
        resume_path = Path(resume_from) if resume_from else None
        if resume_path is None and resume:
            resume_path = checkpoint_dir / "latest.pt"
        if resume_path and resume_path.exists():
            ckpt = _load_checkpoint(resume_path)
            theta_mu = np.array(ckpt.get("theta_mu", theta_mu), copy=True)
            theta_pi = np.array(ckpt.get("theta_pi", theta_pi), copy=True)
            w = np.array(ckpt.get("w", w), copy=True)
            ckpt_iter = ckpt.get("iter")
            if ckpt_iter is not None:
                try:
                    start_iter = int(ckpt_iter) + 1
                except (TypeError, ValueError):
                    start_iter = 0
            print(f"Resuming from checkpoint: {resume_path}")
        if last_logged is not None:
            start_iter = max(start_iter, last_logged + 1)

        teacher_w = np.array(env.teacher_reward.w_R, copy=True)

        probe_manager = ProbeManager(
            cfg.get("probes", {}),
            output_dir=output_dir,
            env_config=env_cfg,
            seed=seed,
            alpha_w=alpha_w,
            train_step_scale=train_step_scale,
            gamma=gamma,
            k_mc=k_mc,
            sigma_mu=sigma_mu,
            sigma_pi=sigma_pi,
            rho_clip=rho_clip,
            disable_rho_clip=disable_rho_clip,
        )
        probe_defaults = probe_manager.log_defaults()

        with (output_dir / "config.json").open("w") as handle:
            json.dump({k: _json_ready(v) for k, v in cfg.items()}, handle, indent=2)

        csv_fieldnames = None
        csv_writer = None
        if csv_path.exists():
            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                csv_fieldnames = reader.fieldnames
            if csv_fieldnames:
                csv_handle = csv_path.open("a", newline="")
                csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fieldnames)

        seed_max = np.iinfo(np.int32).max
        last_report_time = time.time()

        for n in range(start_iter, outer_iters):
            mu_policy = LinearGaussianPolicy(theta=theta_mu, sigma=sigma_mu, v_max=env.v_max)
            pi_policy = LinearGaussianPolicy(theta=theta_pi, sigma=sigma_pi, v_max=env.v_max)

            grad_w = np.zeros_like(w)
            grad_theta = np.zeros_like(theta_pi)
            td_errors = []
            rho_vals = []
            rho_raw_vals = []
            rho_exec_vals = []
            a_diff_vals = []
            clip_count = 0

            delta_cache = None
            if probe_manager.enabled and probe_manager.q_kernel_enabled:
                b_cache = probe_manager.q_kernel_batch_size
                t_cache = min(probe_manager.q_kernel_max_horizon, horizon)
                delta_cache = np.full((b_cache, t_cache), np.nan, dtype=float)

            for traj_idx in range(trajectories):
                env.reset(seed=int(rollout_rng.integers(0, seed_max)))
                zero_action = np.zeros(2, dtype=float)
                feat0 = env.compute_features(zero_action)
                psi = feat0["psi"]
                for t_idx in range(horizon):
                    # ---- sample action from behavior policy mu ----
                    a_exec = mu_policy.sample_action(psi, rollout_rng)
                    a_clip = _clip_action(a_exec, env.v_max)
                    a_diff = float(np.linalg.norm(a_exec - a_clip))
                    a_diff_vals.append(a_diff)
                    if a_diff > 1e-12:
                        clip_count += 1

                    # ---- importance ratio rho = pi(a|s) / mu(a|s) ----
                    u_exec = mu_policy.pre_squash(a_exec)
                    logp_pi_raw = pi_policy.log_prob_pre_squash(u_exec, psi)
                    logp_mu_raw = mu_policy.log_prob_pre_squash(u_exec, psi)
                    rho_raw = float(np.exp(logp_pi_raw - logp_mu_raw))

                    logp_pi_exec = pi_policy.log_prob(a_exec, psi)
                    logp_mu_exec = mu_policy.log_prob(a_exec, psi)
                    rho_exec = float(np.exp(logp_pi_exec - logp_mu_exec))
                    rho = apply_rho_clip(rho_exec, rho_clip, disable=disable_rho_clip)

                    # ---- step env (reward + phi are consistent via info["phi"]) ----
                    obs, reward, terminated, truncated, info = env.step(a_exec)

                    phi = info["phi"]  # phi(s_t, a_t) used for reward + TD
                    psi_next = info["psi_next"]  # psi(s_{t+1})

                    # ---- compute bar_phi(s_{t+1}) = E_{a'~pi}[phi(s_{t+1},a')] ----
                    bar_phi = _mc_bar_phi(env, pi_policy, psi_next, rollout_rng, k_mc=k_mc)

                    # ---- TD error ----
                    q_sa = critic_value(w, phi)
                    q_next = critic_value(w, bar_phi)
                    delta = float(reward + gamma * q_next - q_sa)

                    # ---- actor score grad (target policy) ----
                    g = pi_policy.score(a_exec, psi)  # grad_theta log pi(a|s)

                    # ---- accumulate gradients ----
                    grad_w += rho * delta * phi
                    grad_theta += rho * delta * g

                    td_errors.append(delta)
                    rho_vals.append(rho)
                    rho_raw_vals.append(rho_raw)
                    rho_exec_vals.append(rho_exec)
                    if delta_cache is not None and traj_idx < delta_cache.shape[0] and t_idx < delta_cache.shape[1]:
                        delta_cache[traj_idx, t_idx] = delta

                    # advance
                    psi = psi_next
                    if terminated or truncated:
                        break

            w_prev = np.array(w, copy=True)
            theta_pi_prev = np.array(theta_pi, copy=True)

            scale = 1.0 / total_steps
            w = w + alpha_w * scale * grad_w
            theta_pi = theta_pi + alpha_pi * scale * grad_theta
            theta_mu = (1.0 - beta) * theta_mu + beta * theta_pi

            theta_pi = project_to_ball(theta_pi, theta_radius)
            theta_mu = project_to_ball(theta_mu, theta_radius)
            delta_theta_pi_norm = float(np.linalg.norm(theta_pi - theta_pi_prev))
            delta_w_norm = float(np.linalg.norm(w - w_prev))

            td_loss = float(np.mean(np.square(td_errors))) if td_errors else float("nan")
            if rho_vals:
                rho_arr = np.asarray(rho_vals, dtype=float)
                rho2_arr = rho_arr * rho_arr
                mean_rho = float(np.mean(rho_arr))
                mean_rho2 = float(np.mean(rho2_arr))
                min_rho = float(np.min(rho_arr))
                max_rho = float(np.max(rho_arr))
                p95_rho = float(np.quantile(rho_arr, 0.95))
                p99_rho = float(np.quantile(rho_arr, 0.99))
                p95_rho2 = float(np.quantile(rho2_arr, 0.95))
                p99_rho2 = float(np.quantile(rho2_arr, 0.99))
                max_rho2 = float(np.max(rho2_arr))
            else:
                mean_rho = float("nan")
                mean_rho2 = float("nan")
                min_rho = float("nan")
                max_rho = float("nan")
                p95_rho = float("nan")
                p99_rho = float("nan")
                p95_rho2 = float("nan")
                p99_rho2 = float("nan")
                max_rho2 = float("nan")

            if rho_raw_vals:
                rho_raw_arr = np.asarray(rho_raw_vals, dtype=float)
                mean_rho_raw = float(np.mean(rho_raw_arr))
                mean_rho2_raw = float(np.mean(rho_raw_arr * rho_raw_arr))
            else:
                mean_rho_raw = float("nan")
                mean_rho2_raw = float("nan")

            if rho_exec_vals:
                rho_exec_arr = np.asarray(rho_exec_vals, dtype=float)
                mean_rho_exec = float(np.mean(rho_exec_arr))
                mean_rho2_exec = float(np.mean(rho_exec_arr * rho_exec_arr))
            else:
                mean_rho_exec = float("nan")
                mean_rho2_exec = float("nan")

            if td_errors:
                delta_arr = np.asarray(td_errors, dtype=float)
                delta_mean = float(np.mean(delta_arr))
                delta_std = float(np.std(delta_arr))
                delta_p95 = float(np.quantile(delta_arr, 0.95))
                delta_p99 = float(np.quantile(delta_arr, 0.99))
                delta_max = float(np.max(delta_arr))
            else:
                delta_mean = float("nan")
                delta_std = float("nan")
                delta_p95 = float("nan")
                delta_p99 = float("nan")
                delta_max = float("nan")

            if a_diff_vals:
                diff_arr = np.asarray(a_diff_vals, dtype=float)
                mean_abs_a_diff = float(np.mean(diff_arr))
                p95_abs_a_diff = float(np.quantile(diff_arr, 0.95))
                max_abs_a_diff = float(np.max(diff_arr))
                clip_fraction = float(clip_count / diff_arr.size)
            else:
                mean_abs_a_diff = float("nan")
                p95_abs_a_diff = float("nan")
                max_abs_a_diff = float("nan")
                clip_fraction = float("nan")
            critic_teacher_error = float(np.dot(w - teacher_w, w - teacher_w) / feature_dim)
            tracking_gap = float(np.linalg.norm(theta_pi - theta_mu) ** 2 / actor_dim)
            w_norm = float(np.linalg.norm(w))

            log_row = {
                "iter": n,
                "td_loss": td_loss,
                "critic_teacher_error": critic_teacher_error,
                "tracking_gap": tracking_gap,
                "mean_rho": mean_rho,
                "mean_rho2": mean_rho2,
                "mean_rho_raw": mean_rho_raw,
                "mean_rho2_raw": mean_rho2_raw,
                "mean_rho_exec": mean_rho_exec,
                "mean_rho2_exec": mean_rho2_exec,
                "min_rho": min_rho,
                "max_rho": max_rho,
                "p95_rho": p95_rho,
                "p99_rho": p99_rho,
                "p95_rho2": p95_rho2,
                "p99_rho2": p99_rho2,
                "max_rho2": max_rho2,
                "clip_fraction": clip_fraction,
                "mean_abs_a_diff": mean_abs_a_diff,
                "p95_abs_a_diff": p95_abs_a_diff,
                "max_abs_a_diff": max_abs_a_diff,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "delta_p95": delta_p95,
                "delta_p99": delta_p99,
                "delta_max": delta_max,
                "w_norm": w_norm,
                "delta_theta_pi_norm": delta_theta_pi_norm,
                "delta_w_norm": delta_w_norm,
                **probe_defaults,
            }
            probe_updates = probe_manager.maybe_run(
                iteration=n,
                td_loss=td_loss,
                w=w,
                theta_mu=theta_mu,
                theta_pi=theta_pi,
                delta_cache=delta_cache,
            )
            if probe_updates:
                log_row.update(probe_updates)
            logs.append(log_row)
            if csv_writer is None:
                csv_fieldnames = list(log_row.keys())
                csv_handle = csv_path.open("w", newline="")
                csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fieldnames)
                csv_writer.writeheader()
            csv_writer.writerow(log_row)
            csv_handle.flush()

            if log_every > 0 and (n % log_every == 0):
                print(
                    "iter {:03d} | td_loss {:.4f} | teacher_err {:.4f} | gap {:.4f} | rho2 {:.4f} | w_norm {:.3f}".format(
                        n, td_loss, critic_teacher_error, tracking_gap, mean_rho2, w_norm
                    )
                )

            if checkpoint_every > 0 and (n + 1) % checkpoint_every == 0:
                payload = {"theta_mu": theta_mu, "theta_pi": theta_pi, "w": w, "iter": n}
                _save_checkpoint(checkpoint_dir / f"iter_{n:04d}.npz", **payload)
                _save_checkpoint(checkpoint_dir / "latest.pt", **payload)

            should_report = False
            if report_every > 0 and (n + 1) % report_every == 0:
                should_report = True
            if report_every_seconds > 0 and (time.time() - last_report_time) >= report_every_seconds:
                should_report = True
            if should_report:
                try:
                    generate_run_report(
                        run_dir=output_dir,
                        config=cfg,
                        curves_csv=csv_path,
                        probes_dir=output_dir / "probes",
                        stdout_log_path=output_dir / "stdout.log",
                        incomplete=True,
                        exception=None,
                    )
                    last_report_time = time.time()
                except Exception:
                    print("Failed to generate periodic run report.")
                    traceback.print_exc()

        payload = {"theta_mu": theta_mu, "theta_pi": theta_pi, "w": w, "iter": outer_iters - 1}
        _save_checkpoint(checkpoint_dir / "final.npz", **payload)
        _save_checkpoint(checkpoint_dir / "latest.pt", **payload)

        if csv_handle is not None:
            csv_handle.close()

        return {"output_dir": str(output_dir), "csv_path": str(csv_path), "logs": logs}
    except Exception:
        exception = traceback.format_exc()
        raise
    finally:
        try:
            generate_run_report(
                run_dir=output_dir,
                config=cfg,
                curves_csv=csv_path,
                probes_dir=output_dir / "probes",
                stdout_log_path=output_dir / "stdout.log",
                incomplete=exception is not None,
                exception=exception,
            )
        except Exception:
            print("Failed to generate run report.")
            traceback.print_exc()
        finally:
            if csv_handle is not None:
                csv_handle.close()
