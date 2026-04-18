# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deterministic pseudo-random sequencer implementation."""

# Third Party
import numpy as np

# CuRobo
from .sequencer_base import BaseSequencer


class RandomSequencer(BaseSequencer):
    """Deterministic pseudo-random sequencer.

    Provides deterministic pseudo-random sequences using numpy's RandomState.
    This solves the non-deterministic behavior and reset issues present in
    PyTorch's MultivariateNormal which uses global random state.

    Unlike the Halton sequencer which provides low-discrepancy sequences,
    this generates standard pseudo-random uniform samples with proper
    seed control and reset functionality.

    Example:
        >>> sequencer = RandomSequencer(ndims=7, seed=123)
        >>> samples1 = sequencer.random(1000)  # (1000, 7) array in [0,1]^7
        >>> sequencer.reset()
        >>> samples2 = sequencer.random(1000)  # Identical to samples1
        >>> assert np.allclose(samples1, samples2)
    """

    def __init__(self, ndims: int, seed: int = 123):
        """Initialize random sequencer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Random seed for deterministic behavior
        """
        super().__init__(ndims, seed)
        self.rng = np.random.RandomState(seed)
        self._initial_state = self.rng.get_state()

    def random(self, n_samples: int) -> np.ndarray:
        """Generate n uniform pseudo-random samples in [0,1]^d.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, ndims) with values in [0,1]
        """
        return self.rng.uniform(0.0, 1.0, size=(n_samples, self.ndims))

    def reset(self) -> None:
        """Reset random sequencer to initial seed state."""
        self.rng.set_state(self._initial_state)

    def fast_forward(self, steps: int) -> None:
        """Skip ahead by generating and discarding samples.

        Args:
            steps: Number of samples to skip ahead
        """
        if steps > 0:
            # Generate and discard samples to advance the RNG state
            self.rng.uniform(0.0, 1.0, size=(steps, self.ndims))
