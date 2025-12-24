import unittest

import numpy as np

from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv


class TestFeatureMaps(unittest.TestCase):
    def test_actor_psi_bounded(self) -> None:
        env = TorusGobletGhostEnv(config={"actor_feature_dim": 12, "c_psi": 0.8})
        env.reset(seed=7)
        norms = []
        for _ in range(10):
            _, reward, terminated, truncated, info = env.step(np.array([0.3, -0.2]))
            self.assertFalse(terminated or truncated)
            psi = info["psi"]
            self.assertEqual(psi.shape[0], env.actor_feature_dim)
            norm = float(np.linalg.norm(psi))
            norms.append(norm)
            self.assertLessEqual(norm, env.c_psi + 1e-6)
        self.assertLessEqual(max(norms), env.c_psi + 1e-6)

    def test_phi_stability(self) -> None:
        env = TorusGobletGhostEnv(config={"feature_dim": 64})
        env.reset(seed=1)
        for _ in range(10):
            _, reward, terminated, truncated, info = env.step(np.array([0.1, -0.05]))
            self.assertFalse(terminated or truncated)
            phi = info["phi"]
            self.assertEqual(phi.shape[0], env.feature_dim)
            self.assertFalse(np.isnan(phi).any())
            self.assertLess(np.linalg.norm(phi), 50.0)

    def test_reward_determinism_with_seed(self) -> None:
        cfg = {"feature_dim": 48, "actor_feature_dim": 16}
        env1 = TorusGobletGhostEnv(config=cfg)
        env2 = TorusGobletGhostEnv(config=cfg)

        env1.reset(seed=123)
        env2.reset(seed=123)

        actions = [
            np.array([0.2, -0.1]),
            np.array([0.0, 0.0]),
            np.array([-0.15, 0.05]),
        ]
        rewards1 = []
        rewards2 = []
        for action in actions:
            _, r1, _, _, info1 = env1.step(action)
            _, r2, _, _, info2 = env2.step(action)
            rewards1.append(r1)
            rewards2.append(r2)

            expected_r1 = np.dot(env1.teacher_reward.w_R, info1["phi"]) / np.sqrt(env1.critic_features_map.dim)
            self.assertAlmostEqual(r1, expected_r1)
            self.assertAlmostEqual(r1, info1["reward_teacher"])
            self.assertEqual(info1["phi"].shape[0], env1.critic_features_map.dim)
            self.assertEqual(info2["phi"].shape[0], env2.critic_features_map.dim)

        np.testing.assert_allclose(rewards1, rewards2)


if __name__ == "__main__":
    unittest.main()
