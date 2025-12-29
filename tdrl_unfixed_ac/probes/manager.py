"""Probe manager for running diagnostics at intervals or plateaus."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tdrl_unfixed_ac.probes.distribution_probe import run_distribution_probe
from tdrl_unfixed_ac.probes.fixed_point_probe import run_fixed_point_probe
from tdrl_unfixed_ac.probes.q_kernel_probe import run_q_kernel_probe
from tdrl_unfixed_ac.probes.stability_probe import run_stability_probe
from tdrl_unfixed_ac.utils.seeding import save_restore_rng_state


class ProbeManager:
    """Trigger probes on schedule or upon TD loss plateau."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        *,
        output_dir: Path,
        env_config: Dict[str, Any],
        seed: Optional[int],
        alpha_w: float,
        train_step_scale: float,
        gamma: float,
        k_mc: int,
        sigma_mu: float,
        sigma_pi: float,
        squash_action: bool,
        rho_clip: Optional[float],
        disable_rho_clip: bool,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.every = int(cfg.get("every", 0))

        self.plateau_cfg = deepcopy(cfg.get("plateau", {}))
        self.plateau_enabled = bool(self.plateau_cfg.get("enabled", False))
        self.plateau_window = int(self.plateau_cfg.get("window", 5))
        self.plateau_tol = float(self.plateau_cfg.get("tol", 1e-3))
        self.plateau_cooldown = int(self.plateau_cfg.get("cooldown", self.plateau_window))
        self.plateau_min_iter = int(self.plateau_cfg.get("min_iter", self.plateau_window))

        self.fixed_cfg = deepcopy(cfg.get("fixed_point", {}))
        self.fixed_enabled = bool(self.fixed_cfg.get("enabled", True))
        self.stability_cfg = deepcopy(cfg.get("stability", {}))
        self.stability_enabled = bool(self.stability_cfg.get("enabled", True))
        self.dist_cfg = deepcopy(cfg.get("distribution", {}))
        self.dist_enabled = bool(self.dist_cfg.get("enabled", True))
        self.q_kernel_cfg = deepcopy(cfg.get("q_kernel", {}))
        self.q_kernel_enabled = bool(self.q_kernel_cfg.get("enabled", False))
        self.q_kernel_batch_size = int(self.q_kernel_cfg.get("batch_size", 8))
        self.q_kernel_max_horizon = int(self.q_kernel_cfg.get("max_horizon", 200))

        self.env_config = deepcopy(env_config)
        self.seed = seed
        self.alpha_w = float(alpha_w)
        self.train_step_scale = float(train_step_scale)
        self.gamma = float(gamma)
        self.k_mc = int(k_mc)
        self.sigma_mu = float(sigma_mu)
        self.sigma_pi = float(sigma_pi)
        self.squash_action = bool(squash_action)
        self.rho_clip = rho_clip
        self.disable_rho_clip = bool(disable_rho_clip)

        self.output_dir = Path(output_dir) / "probes"
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._td_history: list[float] = []
        self._prev_w_sharp: Optional[np.ndarray] = None
        self._last_probe_iter: Optional[int] = None
        self._last_plateau_iter = -self.plateau_cooldown - 1

    def log_defaults(self) -> Dict[str, float]:
        defaults: Dict[str, float] = {}
        if not self.enabled:
            return defaults
        if self.fixed_enabled:
            defaults["fixed_point_gap"] = float("nan")
            defaults["fixed_point_drift"] = float("nan")
            defaults["fixed_point_drift_defined"] = 0.0
        if self.stability_enabled:
            defaults["stability_proxy"] = float("nan")
        if self.dist_enabled:
            defaults["dist_mmd2"] = float("nan")
            defaults["dist_mean_l2"] = float("nan")
            defaults["dist_action_kl"] = float("nan")
            defaults["dist_action_tv"] = float("nan")
        if self.q_kernel_enabled:
            defaults["td_loss_from_Q"] = float("nan")
            defaults["td_loss_from_Q_abs_diff"] = float("nan")
            defaults["td_loss_from_Q_rel_diff"] = float("nan")
        return defaults

    def maybe_run(
        self,
        *,
        iteration: int,
        td_loss: float,
        w: np.ndarray,
        theta_mu: np.ndarray,
        theta_pi: np.ndarray,
        delta_cache: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if not self.enabled:
            return {}

        self._td_history.append(float(td_loss))
        if self._last_probe_iter == iteration:
            return {}

        should_run = self._should_run(iteration)
        if not should_run:
            return {}

        self._last_probe_iter = iteration
        results: Dict[str, float] = {}

        with save_restore_rng_state():
            if self.fixed_enabled:
                fixed_cfg = self._with_defaults(
                    self.fixed_cfg,
                    {
                        "batch_size": 4096,
                        "max_iters": 200,
                        "tol": 1e-4,
                        "alpha_w": self.alpha_w,
                        "gamma": self.gamma,
                        "k_mc": self.k_mc,
                    },
                )
                fixed_out = run_fixed_point_probe(
                    env_config=self.env_config,
                    theta_mu=theta_mu,
                    theta_pi=theta_pi,
                    w_init=w,
                    sigma_mu=self.sigma_mu,
                    sigma_pi=self.sigma_pi,
                    squash_action=self.squash_action,
                    alpha_w=float(fixed_cfg["alpha_w"]),
                    gamma=float(fixed_cfg["gamma"]),
                    k_mc=int(fixed_cfg["k_mc"]),
                    batch_size=int(fixed_cfg["batch_size"]),
                    max_iters=int(fixed_cfg["max_iters"]),
                    tol=float(fixed_cfg["tol"]),
                    rho_clip=self.rho_clip,
                    disable_rho_clip=self.disable_rho_clip,
                    seed=self._seed_for(iteration, 1),
                )
                w_sharp = fixed_out["w_sharp"]
                gap = float(np.linalg.norm(w - w_sharp))
                if self._prev_w_sharp is not None:
                    drift = float(np.linalg.norm(w_sharp - self._prev_w_sharp))
                    drift_defined = 1.0
                else:
                    drift = 0.0
                    drift_defined = 0.0
                self._prev_w_sharp = np.array(w_sharp, copy=True)
                results["fixed_point_gap"] = gap
                results["fixed_point_drift"] = drift
                results["fixed_point_drift_defined"] = drift_defined
                self._append_probe(
                    "fixed_point_probe",
                    {
                        "iter": iteration,
                        "w_gap": gap,
                        "w_sharp_drift": drift,
                        "w_sharp_drift_defined": drift_defined,
                        "converged": fixed_out["converged"],
                        "num_iters": fixed_out["num_iters"],
                        "batch_size": fixed_out["batch_size"],
                        "tol": fixed_out["tol"],
                        "rho_mean": fixed_out["rho_mean"],
                        "rho2_mean": fixed_out["rho2_mean"],
                        "rho_min": fixed_out["rho_min"],
                        "rho_max": fixed_out["rho_max"],
                        "rho_p95": fixed_out["rho_p95"],
                        "rho_p99": fixed_out["rho_p99"],
                        "rho_clip": fixed_out["rho_clip"],
                        "rho_clip_active": fixed_out["rho_clip_active"],
                    },
                )

            if self.stability_enabled:
                stability_cfg = self._with_defaults(
                    self.stability_cfg,
                    {
                        "batch_size": 4096,
                        "power_iters": 20,
                        "alpha_w": self.alpha_w,
                        "gamma": self.gamma,
                        "k_mc": self.k_mc,
                    },
                )
                stability_out = run_stability_probe(
                    env_config=self.env_config,
                    theta_mu=theta_mu,
                    theta_pi=theta_pi,
                    sigma_mu=self.sigma_mu,
                    sigma_pi=self.sigma_pi,
                    squash_action=self.squash_action,
                    alpha_w=float(stability_cfg["alpha_w"]),
                    train_step_scale=self.train_step_scale,
                    gamma=float(stability_cfg["gamma"]),
                    k_mc=int(stability_cfg["k_mc"]),
                    batch_size=int(stability_cfg["batch_size"]),
                    power_iters=int(stability_cfg["power_iters"]),
                    rho_clip=self.rho_clip,
                    disable_rho_clip=self.disable_rho_clip,
                    seed=self._seed_for(iteration, 2),
                )
                results["stability_proxy"] = float(stability_out["stability_proxy"])
                self._append_probe(
                    "stability_probe",
                    {
                        "iter": iteration,
                        "stability_proxy": stability_out["stability_proxy"],
                        "stability_proxy_mean": stability_out["stability_proxy_mean"],
                        "stability_proxy_std": stability_out["stability_proxy_std"],
                        "power_iters": stability_out["power_iters"],
                        "batch_size": stability_out["batch_size"],
                        "stability_probe_step_scale": stability_out["stability_probe_step_scale"],
                        "rho_mean": stability_out["rho_mean"],
                        "rho2_mean": stability_out["rho2_mean"],
                        "rho_min": stability_out["rho_min"],
                        "rho_max": stability_out["rho_max"],
                        "rho_p95": stability_out["rho_p95"],
                        "rho_p99": stability_out["rho_p99"],
                        "rho_clip": stability_out["rho_clip"],
                        "rho_clip_active": stability_out["rho_clip_active"],
                    },
                )

            if self.dist_enabled:
                dist_cfg = self._with_defaults(self.dist_cfg, {"num_samples": 512, "action_samples": 64})
                dist_out = run_distribution_probe(
                    env_config=self.env_config,
                    theta_mu=theta_mu,
                    theta_pi=theta_pi,
                    sigma_mu=self.sigma_mu,
                    sigma_pi=self.sigma_pi,
                    squash_action=self.squash_action,
                    num_samples=int(dist_cfg["num_samples"]),
                    action_samples=int(dist_cfg["action_samples"]),
                    rho_clip=self.rho_clip,
                    disable_rho_clip=self.disable_rho_clip,
                    seed=self._seed_for(iteration, 3),
                )
                results["dist_mmd2"] = float(dist_out["mmd2"])
                results["dist_mean_l2"] = float(dist_out["mean_l2"])
                results["dist_action_kl"] = float(dist_out["dist_action_kl"])
                results["dist_action_tv"] = float(dist_out["dist_action_tv"])
                self._append_probe(
                    "distribution_probe",
                    {
                        "iter": iteration,
                        "mmd2": dist_out["mmd2"],
                        "mmd_sigma": dist_out["mmd_sigma"],
                        "mean_l2": dist_out["mean_l2"],
                        "num_samples": dist_out["num_samples"],
                        "dist_action_kl": dist_out["dist_action_kl"],
                        "dist_action_tv": dist_out["dist_action_tv"],
                        "action_samples": dist_out["action_samples"],
                        "rho_mean": dist_out["rho_mean"],
                        "rho2_mean": dist_out["rho2_mean"],
                        "rho_min": dist_out["rho_min"],
                        "rho_max": dist_out["rho_max"],
                        "rho_p95": dist_out["rho_p95"],
                        "rho_p99": dist_out["rho_p99"],
                        "rho_clip": dist_out["rho_clip"],
                        "rho_clip_active": dist_out["rho_clip_active"],
                    },
                )

            if self.q_kernel_enabled and delta_cache is not None:
                q_out = run_q_kernel_probe(
                    delta_cache=delta_cache,
                    td_loss=td_loss,
                    iteration=iteration,
                )
                results["td_loss_from_Q"] = float(q_out["td_loss_from_Q"])
                results["td_loss_from_Q_abs_diff"] = float(q_out["td_loss_from_Q_abs_diff"])
                results["td_loss_from_Q_rel_diff"] = float(q_out["td_loss_from_Q_rel_diff"])
                self._append_probe("q_kernel_probe", q_out)

        return results

    def _should_run(self, iteration: int) -> bool:
        if self.every > 0 and (iteration + 1) % self.every == 0:
            return True
        if self.plateau_enabled and iteration >= self.plateau_min_iter:
            if iteration - self._last_plateau_iter < self.plateau_cooldown:
                return False
            if len(self._td_history) < self.plateau_window:
                return False
            window = np.asarray(self._td_history[-self.plateau_window :], dtype=float)
            if np.isnan(window).any():
                return False
            spread = float(np.max(window) - np.min(window))
            mean = float(np.mean(window))
            tol = self.plateau_tol * max(1.0, abs(mean))
            if spread <= tol:
                self._last_plateau_iter = iteration
                return True
        return False

    def _seed_for(self, iteration: int, offset: int) -> int:
        base = int(self.seed) if self.seed is not None else 0
        return int((base + 10007 * (iteration + 1) + 97 * offset) % (2**31 - 1))

    def _append_probe(self, stem: str, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        csv_path = self.output_dir / f"{stem}.csv"
        json_path = self.output_dir / f"{stem}.jsonl"

        if not csv_path.exists():
            fieldnames = list(row.keys())
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
        else:
            with csv_path.open("r", newline="") as handle:
                reader = csv.reader(handle)
                fieldnames = next(reader, list(row.keys()))
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(row)

        with json_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    @staticmethod
    def _with_defaults(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        merged.update(cfg)
        return merged
