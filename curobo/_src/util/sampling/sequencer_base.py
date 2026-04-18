# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Base protocol and interfaces for all sequencers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSequencer(ABC):
    """Abstract base class for sequencer implementations.

    Provides common functionality and enforces the interface.
    """

    def __init__(self, ndims: int, seed: int = 123):
        """Initialize base sequencer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Random seed for deterministic behavior
        """
        self.ndims = ndims
        self.seed = seed

    @abstractmethod
    def random(self, n_samples: int) -> np.ndarray:
        """Generate n uniform random samples in [0,1]^d."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset sequencer to initial state."""
        pass

    @abstractmethod
    def fast_forward(self, steps: int) -> None:
        """Skip ahead in the sequence."""
        pass
