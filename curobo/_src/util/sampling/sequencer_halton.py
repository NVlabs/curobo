# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Halton quasi-Monte Carlo sequencer implementation."""

# Third Party
import numpy as np
from scipy.stats.qmc import Halton

# CuRobo
from .sequencer_base import BaseSequencer


class HaltonSequencer(BaseSequencer):
    """Halton quasi-Monte Carlo sequencer.

    Provides low-discrepancy Halton sequences for improved sampling coverage
    compared to pseudo-random sequences. Uses scipy's QMC implementation with
    scrambling for better distribution properties.

    Example:
        >>> sequencer = HaltonSequencer(ndims=7, seed=123)
        >>> samples = sequencer.random(1000)  # (1000, 7) array in [0,1]^7
        >>> sequencer.reset()  # Start sequence over
    """

    def __init__(self, ndims: int, seed: int = 123, scramble: bool = True):
        """Initialize Halton sequencer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Random seed for deterministic scrambling
            scramble: Whether to use scrambled Halton sequence for better properties
        """
        super().__init__(ndims, seed)
        self.scramble = scramble
        self.qmc = Halton(d=ndims, seed=seed, scramble=scramble)

    def random(self, n_samples: int) -> np.ndarray:
        """Generate n uniform Halton samples in [0,1]^d.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, ndims) with values in [0,1]
        """
        return self.qmc.random(n_samples)

    def reset(self) -> None:
        """Reset Halton sequencer to initial state."""
        self.qmc.reset()

    def fast_forward(self, steps: int) -> None:
        """Skip ahead in the Halton sequence.

        Args:
            steps: Number of samples to skip ahead
        """
        if steps > 0:
            self.qmc.fast_forward(steps)
