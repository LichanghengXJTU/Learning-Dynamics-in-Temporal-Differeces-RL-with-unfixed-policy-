"""Deterministic seeding utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

try:  # pragma: no cover - torch is optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


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


@contextmanager
def save_restore_rng_state() -> Generator[None, None, None]:
    """Save and restore global RNG state for numpy/torch (if available)."""
    np_state = np.random.get_state()
    torch_state = None
    torch_cuda_state = None
    if torch is not None:
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_state = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_state)
