"""Deterministic seeding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Seeder:
    """Wrapper around numpy SeedSequence to simplify reproducible RNG."""

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.seed_sequence = np.random.SeedSequence(self.seed)
        self.rng = np.random.default_rng(self.seed_sequence)

    def spawn(self) -> np.random.Generator:
        """Spawn a child generator deterministically."""
        child_seq = self.seed_sequence.spawn(1)[0]
        return np.random.default_rng(child_seq)

    def reseed(self, seed: Optional[int]) -> np.random.Generator:
        """Reset to a new base seed and return the generator."""
        self.seed = seed
        self.seed_sequence = np.random.SeedSequence(self.seed)
        self.rng = np.random.default_rng(self.seed_sequence)
        return self.rng

