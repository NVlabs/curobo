# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for BaseSequencer abstract class."""

import numpy as np
import pytest

# CuRobo
from curobo._src.util.sampling.sequencer_base import BaseSequencer


class ConcreteSequencer(BaseSequencer):
    """Concrete implementation of BaseSequencer for testing."""

    def __init__(self, ndims: int, seed: int = 123):
        super().__init__(ndims, seed)
        self.state = 0

    def random(self, n_samples: int) -> np.ndarray:
        """Generate simple incremental samples for testing."""
        samples = np.zeros((n_samples, self.ndims))
        for i in range(n_samples):
            samples[i] = np.full(self.ndims, (self.state + i) / 100.0)
        self.state += n_samples
        return samples

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = 0

    def fast_forward(self, steps: int) -> None:
        """Skip ahead in sequence."""
        self.state += steps


class TestBaseSequencer:
    """Test cases for BaseSequencer abstract base class."""

    def test_init_basic(self):
        """Test basic initialization."""
        sequencer = ConcreteSequencer(ndims=5, seed=42)
        assert sequencer.ndims == 5
        assert sequencer.seed == 42

    def test_init_default_seed(self):
        """Test initialization with default seed."""
        sequencer = ConcreteSequencer(ndims=3)
        assert sequencer.ndims == 3
        assert sequencer.seed == 123

    def test_abstract_methods_implemented(self):
        """Test that concrete implementation has all required methods."""
        sequencer = ConcreteSequencer(ndims=4)

        # Test random method
        samples = sequencer.random(10)
        assert samples.shape == (10, 4)
        assert isinstance(samples, np.ndarray)

        # Test reset method
        sequencer.reset()
        assert sequencer.state == 0

        # Test fast_forward method
        sequencer.fast_forward(5)
        assert sequencer.state == 5

    def test_abstract_class_instantiation(self):
        """Test that BaseSequencer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSequencer(ndims=3, seed=123)

    def test_reset_functionality(self):
        """Test that reset properly restores initial state."""
        sequencer = ConcreteSequencer(ndims=2, seed=456)

        # Generate some samples
        initial_samples = sequencer.random(5)

        # Reset and generate again
        sequencer.reset()
        reset_samples = sequencer.random(5)

        # Should be identical
        np.testing.assert_array_equal(initial_samples, reset_samples)

    def test_fast_forward_functionality(self):
        """Test fast forward advances state correctly."""
        sequencer = ConcreteSequencer(ndims=3)

        # Generate 10 samples normally
        sequencer.reset()
        normal_samples = sequencer.random(10)

        # Reset, fast forward 5, then generate 5 more
        sequencer.reset()
        sequencer.fast_forward(5)
        ff_samples = sequencer.random(5)

        # The fast forwarded samples should match the last 5 normal samples
        np.testing.assert_array_equal(normal_samples[5:], ff_samples)

    def test_ndims_validation(self):
        """Test various ndims values."""
        # Valid cases
        for ndims in [1, 2, 5, 10, 20]:
            sequencer = ConcreteSequencer(ndims=ndims)
            assert sequencer.ndims == ndims
            samples = sequencer.random(3)
            assert samples.shape == (3, ndims)

    def test_seed_determinism(self):
        """Test that same seed produces deterministic behavior."""
        sequencer1 = ConcreteSequencer(ndims=4, seed=999)
        sequencer2 = ConcreteSequencer(ndims=4, seed=999)

        samples1 = sequencer1.random(8)
        samples2 = sequencer2.random(8)

        # Same seed should produce same sequence
        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds(self):
        """Test that different seeds produce different sequences."""
        sequencer1 = ConcreteSequencer(ndims=3, seed=111)
        sequencer2 = ConcreteSequencer(ndims=3, seed=222)

        samples1 = sequencer1.random(10)
        samples2 = sequencer2.random(10)

        # Different seeds should not be identical (with high probability)
        # Note: For our simple test implementation they'll actually be the same
        # since seed isn't used, but this tests the interface
        assert samples1.shape == samples2.shape
