"""Continuing-control torus Goblet&Ghost environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tdrl_unfixed_ac.features import ActorFeatureMap, CriticFeatureMap, TeacherReward, build_observation_vector
from tdrl_unfixed_ac.utils.geometry import torus_delta, torus_distance, wrap_torus
from tdrl_unfixed_ac.utils.seeding import Seeder

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON/YAML file."""
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    text = target.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


class TorusGobletGhostEnv:
    """
    Headless Goblet&Ghost environment on a continuous torus.

    State:
        - adventurer position in R^2
        - ghost position in R^2
        - K goblet positions in R^{K x 2} and types in {+1, -1}
    Events:
        - caught: ghost within eps_catch of adventurer, ghost respawns
        - picked: adventurer within eps_pick of goblet j, goblet respawns
        - p_mix restart: with probability p_mix replace next state with sample from nu
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        base_config = load_config()
        if config_path:
            base_config.update(load_config(config_path))
        if config:
            base_config.update(config)
        self.cfg = base_config

        self.torus_size = float(self.cfg["torus_size"])
        self.dt = float(self.cfg["dt"])
        self.v_max = float(self.cfg["v_max"])
        self.v_ghost = float(self.cfg["v_ghost"])
        self.sigma_env = float(self.cfg["sigma_env"])
        self.sigma_ghost = float(self.cfg["sigma_ghost"])
        self.eps_catch = float(self.cfg["eps_catch"])
        self.eps_pick = float(self.cfg["eps_pick"])
        self.num_goblets = int(self.cfg["num_goblets"])
        self.p_mix = float(self.cfg["p_mix"])
        self.p_type_positive = float(self.cfg["p_type_positive"])
        self.type_resample_p = float(self.cfg["type_resample_p"])
        self.use_teacher_reward = bool(self.cfg.get("use_teacher_reward", True))
        self.clip_action = bool(self.cfg.get("clip_action", False))
        self.feature_dim = int(self.cfg.get("feature_dim", 128))
        self.actor_feature_dim = int(self.cfg.get("actor_feature_dim", 32))
        self.c_psi = float(self.cfg.get("c_psi", 1.0))
        self.feature_sigma = float(self.cfg.get("feature_sigma", 1.0))
        self.lambda_action = float(self.cfg.get("lambda_action_penalty", 0.1))
        self.teacher_base_scale = float(self.cfg.get("teacher_base_scale", 0.05))

        self.seeder = Seeder(self.cfg.get("seed"))
        self.rng = rng if rng is not None else self.seeder.rng
        self._init_feature_maps()

        self.adventurer: np.ndarray
        self.ghost: np.ndarray
        self.goblets_pos: np.ndarray
        self.goblets_type: np.ndarray
        self._step_count = 0
        self.last_events: Dict[str, Any] = {}

    def _init_feature_maps(self) -> None:
        dummy_obs = {
            "adventurer": np.zeros(2, dtype=float),
            "ghost": np.zeros(2, dtype=float),
            "goblets_pos": np.zeros((self.num_goblets, 2), dtype=float),
            "goblets_type": np.zeros(self.num_goblets, dtype=float),
            "caught": False,
            "picked": False,
            "picked_type": 0.0,
            "restart": False,
        }
        self.obs_dim = build_observation_vector(dummy_obs, self.torus_size).shape[0]

        actor_rng = self.seeder.spawn()
        critic_rng = self.seeder.spawn()
        teacher_rng = self.seeder.spawn()

        self.actor_features_map = ActorFeatureMap(
            obs_dim=self.obs_dim, dim=self.actor_feature_dim, c_psi=self.c_psi, rng=actor_rng
        )
        self.critic_features_map = CriticFeatureMap(
            obs_dim=self.obs_dim, dim=self.feature_dim, sigma=self.feature_sigma, rng=critic_rng
        )
        self.teacher_reward = TeacherReward(
            critic_features=self.critic_features_map,
            lambda_action=self.lambda_action,
            base_scale=self.teacher_base_scale,
            rng=teacher_rng,
        )

    # API ----------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = self.seeder.reseed(seed)
        self._step_count = 0
        self._sample_state(resample_types=True)
        self.last_events = {
            "caught": False,
            "picked": False,
            "picked_type": 0.0,
            "restart": False,
        }
        return self._get_obs(), {"seed": seed}
        
    def step(self, action):
        """
        Gym-style step.

        IMPORTANT SEMANTICS (for DMFT / teacher-student consistency):
        - reward is computed from the SAME feature vector phi(s_t, a_t) returned in info["phi"].
        - events (caught/picked/restart) describe what happened during THIS transition and
          are written into self.last_events for the NEXT observation.
        - if restart happens (p-mix), we ignore caught/picked from the regular dynamics step.
        """
        # -------------------------
        # 0) sanitize + clip action
        # -------------------------
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError(f"Expected action dim 2, got shape {action.shape}")
        clipped_action = self._clip_action(action)

        # ---------------------------------------------------------
        # 1) compute features on CURRENT state (s_t) for TD + reward
        # ---------------------------------------------------------
        obs_t = self._get_obs()
        obs_vec_t, psi_t, phi_t = self._compute_features(obs_t, clipped_action)

        reward = 0.0
        if self.use_teacher_reward:
            reward = float(self.teacher_reward(phi_t))

        # -------------------------
        # 2) advance the environment
        # -------------------------
        # adv dynamics
        env_noise = self.rng.normal(loc=0.0, scale=self.sigma_env, size=(2,))
        self.adventurer = self._wrap(self.adventurer + self.dt * clipped_action + env_noise)

        # ghost dynamics
        self._ghost_step()

        # events under regular dynamics
        caught = self._check_caught()
        picked, picked_type = self._check_picked()

        # p-mix restart
        restart = False
        if self.rng.random() < self.p_mix:
            self._sample_state(resample_types=True)
            restart = True
            # By definition of restart ~ nu, ignore events from the regular transition
            caught = False
            picked = False
            picked_type = 0.0

        # bookkeeping + write events into NEXT obs
        self._step_count += 1
        self.last_events = {
            "caught": bool(caught),
            "picked": bool(picked),
            "picked_type": float(picked_type),
            "restart": bool(restart),
        }

        # ---------------------------------------------------------
        # 3) build next observation (s_{t+1}) + next-state features
        # ---------------------------------------------------------
        obs_next = self._get_obs()
        obs_vec_next, psi_next, phi_next = self._compute_features(obs_next, clipped_action)

        terminated = False
        truncated = False

        info = {
            # transition events (THIS step)
            "caught": bool(caught),
            "picked": bool(picked),
            "picked_type": float(picked_type),
            "restart": bool(restart),
            "step": int(self._step_count),

            # features for CURRENT (s_t, a_t) used for reward + TD
            "obs_vec": obs_vec_t,
            "psi": psi_t,
            "phi": phi_t,
            "clipped_action": clipped_action,
            "reward_teacher": float(reward),

            # convenience: next-state features (s_{t+1})
            "obs_vec_next": obs_vec_next,
            "psi_next": psi_next,
            "phi_next": phi_next,
        }

        return obs_next, reward, terminated, truncated, info

    # Internal helpers ---------------------------------------------------- #
    def _sample_state(self, resample_types: bool = True) -> None:
        self.adventurer = self.rng.uniform(0.0, self.torus_size, size=2)
        self.ghost = self.rng.uniform(0.0, self.torus_size, size=2)
        self.goblets_pos = self.rng.uniform(0.0, self.torus_size, size=(self.num_goblets, 2))
        if resample_types or not hasattr(self, "goblets_type"):
            self.goblets_type = self._sample_goblet_types(self.num_goblets)

    def _sample_goblet_types(self, n: int) -> np.ndarray:
        mask = self.rng.random(size=n) < self.p_type_positive
        return np.where(mask, 1.0, -1.0)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        if not self.clip_action or self.v_max <= 0.0:
            return action
        return np.clip(action, -self.v_max, self.v_max)

    def _wrap(self, pos: np.ndarray) -> np.ndarray:
        return wrap_torus(pos, self.torus_size)

    def _safe_unit(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.zeros_like(vec)
        return vec / norm

    def _ghost_step(self) -> None:
        ghost_delta = torus_delta(self.ghost, self.adventurer, self.torus_size)
        ghost_dir = self._safe_unit(ghost_delta)
        ghost_noise = self.rng.normal(loc=0.0, scale=self.sigma_ghost, size=2)
        self.ghost = self._wrap(self.ghost + self.dt * self.v_ghost * ghost_dir + ghost_noise)

    def _check_caught(self) -> bool:
        dist = torus_distance(self.ghost, self.adventurer, self.torus_size)
        if dist <= self.eps_catch:
            self.ghost = self.rng.uniform(0.0, self.torus_size, size=2)
            return True
        return False

    def _check_picked(self) -> Tuple[bool, float]:
        deltas = torus_delta(self.goblets_pos, self.adventurer, self.torus_size)
        dists = np.linalg.norm(deltas, axis=1)
        close_indices = np.nonzero(dists <= self.eps_pick)[0]
        if close_indices.size == 0:
            return False, 0.0

        idx = int(close_indices[0])
        picked_type = float(self.goblets_type[idx])
        self._respawn_goblet(idx)
        return True, picked_type

    def _respawn_goblet(self, idx: int) -> None:
        self.goblets_pos[idx] = self.rng.uniform(0.0, self.torus_size, size=2)
        if self.rng.random() < self.type_resample_p:
            self.goblets_type[idx] = self._sample_goblet_types(1)[0]

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "adventurer": np.array(self.adventurer, copy=True),
            "ghost": np.array(self.ghost, copy=True),
            "goblets_pos": np.array(self.goblets_pos, copy=True),
            "goblets_type": np.array(self.goblets_type, copy=True),
            "caught": bool(self.last_events.get("caught", False)),
            "picked": bool(self.last_events.get("picked", False)),
            "picked_type": float(self.last_events.get("picked_type", 0.0)),
            "restart": bool(self.last_events.get("restart", False)),
        }

    def _compute_features(
        self, raw_obs: Dict[str, Any], clipped_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_vec = build_observation_vector(raw_obs, self.torus_size)
        psi = self.actor_features_map(obs_vec)
        phi = self.critic_features_map(obs_vec, clipped_action, raw_obs)
        return obs_vec, psi, phi

    def compute_features(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Public helper to obtain o(s), psi(s), and phi(s, a) for diagnostics."""
        action = np.asarray(action, dtype=float).reshape(2)
        clipped_action = self._clip_action(action)
        raw_obs = self._get_obs()
        obs_vec, psi, phi = self._compute_features(raw_obs, clipped_action)
        return {"obs_vec": obs_vec, "psi": psi, "phi": phi}
