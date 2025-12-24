import unittest

import numpy as np

from tdrl_unfixed_ac.envs.torus_gg import TorusGobletGhostEnv


class TestEnvSmoke(unittest.TestCase):
    def test_step_shapes_and_finiteness(self) -> None:
        env = TorusGobletGhostEnv()
        obs, info = env.reset(seed=123)

        self.assertEqual(obs["adventurer"].shape, (2,))
        self.assertEqual(obs["ghost"].shape, (2,))
        self.assertEqual(obs["goblets_pos"].shape, (env.num_goblets, 2))
        self.assertEqual(obs["goblets_type"].shape, (env.num_goblets,))

        for _ in range(20):
            action = np.zeros(2, dtype=float)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertFalse(np.isnan(obs["adventurer"]).any())
            self.assertFalse(np.isnan(obs["ghost"]).any())
            self.assertFalse(np.isnan(obs["goblets_pos"]).any())
            self.assertFalse(np.isnan(obs["goblets_type"]).any())
            self.assertEqual(terminated, False)
            self.assertEqual(truncated, False)

    def test_p_mix_restart_triggers(self) -> None:
        env = TorusGobletGhostEnv(config={"p_mix": 1.0, "sigma_env": 0.0, "sigma_ghost": 0.0})
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        self.assertTrue(info["restart"])
        self.assertEqual(terminated, False)
        self.assertEqual(truncated, False)


if __name__ == "__main__":
    unittest.main()
