# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Roberts sequence quasi-Monte Carlo sequencer implementation."""

# Third Party
import numpy as np

# CuRobo
from .sequencer_base import BaseSequencer


def _roberts_root(dim: int) -> float:
    """Compute the generalized golden ratio for d-dimensional Roberts sequence.

    Solves the equation: x^(d+1) - x - 1 = 0
    For d=1, this gives the golden ratio φ ≈ 1.618033988749

    Args:
        dim: Dimensionality of the sequence

    Returns:
        The generalized golden ratio for the given dimension
    """
    x = 1.5  # Initial guess
    for _ in range(100):  # Newton-Raphson iterations
        f = x ** (dim + 1) - x - 1.0
        df = (dim + 1) * x**dim - 1.0
        x_next = x - f / df
        if abs(x_next - x) < 1.0e-12:
            break
        x = x_next
    return x


class RobertsSequencer(BaseSequencer):
    """Roberts sequence quasi-Monte Carlo sequencer.

    Implements the R-sequence developed by Martin Roberts (2018), which provides
    superior low-discrepancy properties compared to traditional sequences like
    Halton and Sobol. The Roberts sequence offers:

    - Better uniform distribution of points
    - No parameter tuning required
    - Optimal discrepancy using generalized golden ratios
    - Improved numerical integration performance

    The sequence is based on additive recurrence with irrational basis vectors
    derived from generalized golden ratios that solve x^(d+1) - x - 1 = 0.

    References:
        Roberts, M. (2018). "The Unreasonable Effectiveness of Quasirandom Sequences"
        https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    Example:
        >>> sequencer = RobertsSequencer(ndims=7, seed=123)
        >>> samples = sequencer.random(1000)  # (1000, 7) array in [0,1]^7
        >>> sequencer.reset()  # Start sequence over
        >>> samples2 = sequencer.random(1000)  # Identical to samples
        >>> assert np.allclose(samples, samples2)
    """

    def __init__(self, ndims: int, seed: int = 123):
        """Initialize Roberts sequencer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Random seed for deterministic behavior (used for toroidal shifts)
        """
        super().__init__(ndims, seed)

        # Compute generalized golden ratio for this dimensionality
        self.root = _roberts_root(ndims)

        # Compute basis vectors using improved formulation from Marty's Mods
        # Using 1 - 1/r^(1+i) instead of 1/r^(1+i) for better floating-point precision
        self.basis = 1.0 - 1.0 / self.root ** (1 + np.arange(ndims))

        # Track current position in sequence
        self.current_index = 0

        # Optional toroidal shift for additional randomization
        # This doesn't affect sequence properties but can be useful
        # self.rng = np.random.RandomState(seed)
        # self.shift = self.rng.uniform(0.0, 1.0, size=ndims)

    def random(self, n_samples: int) -> np.ndarray:
        """Generate n uniform Roberts sequence samples in [0,1]^d.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, ndims) with values in [0,1]
        """
        # Generate indices for this batch
        indices = np.arange(self.current_index, self.current_index + n_samples)

        # Apply Roberts sequence formula: (n * basis + shift) mod 1
        samples = (indices[:, None] * self.basis[None, :]) % 1.0

        # Update current index
        self.current_index += n_samples

        return samples.astype(np.float32)

    def reset(self) -> None:
        """Reset Roberts sequencer to initial state."""
        self.current_index = 0

    def fast_forward(self, steps: int) -> None:
        """Skip ahead in the Roberts sequence.

        Args:
            steps: Number of samples to skip ahead
        """
        if steps > 0:
            self.current_index += steps
