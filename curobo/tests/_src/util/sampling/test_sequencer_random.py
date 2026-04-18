# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RandomSequencer."""

import numpy as np

# CuRobo
from curobo._src.util.sampling.sequencer_random import RandomSequencer


class TestRandomSequencer:
    """Test cases for RandomSequencer implementation."""

    def test_init_basic(self):
        """Test basic initialization."""
        sequencer = RandomSequencer(ndims=5, seed=42)
        assert sequencer.ndims == 5
        assert sequencer.seed == 42
        assert hasattr(sequencer, "rng")
        assert hasattr(sequencer, "_initial_state")

    def test_init_default_seed(self):
        """Test initialization with default seed."""
        sequencer = RandomSequencer(ndims=3)
        assert sequencer.ndims == 3
        assert sequencer.seed == 123

    def test_random_output_shape(self):
        """Test that random() returns correct shape."""
        sequencer = RandomSequencer(ndims=7)

        samples = sequencer.random(100)
        assert samples.shape == (100, 7)
        assert isinstance(samples, np.ndarray)

    def test_random_output_range(self):
        """Test that random() returns values in [0, 1] range."""
        sequencer = RandomSequencer(ndims=4)

        samples = sequencer.random(1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_random_different_sizes(self):
        """Test random() with different sample sizes."""
        sequencer = RandomSequencer(ndims=3)

        for n_samples in [1, 5, 10, 50, 100]:
            samples = sequencer.random(n_samples)
            assert samples.shape == (n_samples, 3)

    def test_reset_functionality(self):
        """Test that reset restores initial state."""
        sequencer = RandomSequencer(ndims=5, seed=456)

        # Generate initial samples
        initial_samples = sequencer.random(10)

        # Generate more samples to advance state
        sequencer.random(20)

        # Reset and generate again
        sequencer.reset()
        reset_samples = sequencer.random(10)

        # Should be identical to initial samples
        np.testing.assert_array_equal(initial_samples, reset_samples)

    def test_fast_forward_functionality(self):
        """Test fast forward advances RNG state correctly."""
        sequencer = RandomSequencer(ndims=3, seed=789)

        # Generate 15 samples normally
        sequencer.reset()
        normal_samples = sequencer.random(15)

        # Reset, fast forward 10, then generate 5 more
        sequencer.reset()
        sequencer.fast_forward(10)
        ff_samples = sequencer.random(5)

        # The fast forwarded samples should match the last 5 normal samples
        np.testing.assert_array_equal(normal_samples[10:], ff_samples)

    def test_seed_determinism(self):
        """Test that same seed produces deterministic sequences."""
        seed = 999
        sequencer1 = RandomSequencer(ndims=4, seed=seed)
        sequencer2 = RandomSequencer(ndims=4, seed=seed)

        samples1 = sequencer1.random(50)
        samples2 = sequencer2.random(50)

        # Same seed should produce identical sequences
        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        sequencer1 = RandomSequencer(ndims=3, seed=111)
        sequencer2 = RandomSequencer(ndims=3, seed=222)

        samples1 = sequencer1.random(100)
        samples2 = sequencer2.random(100)

        # Different seeds should produce different sequences
        # (with very high probability for random sequences)
        assert not np.array_equal(samples1, samples2)

    def test_statistical_properties(self):
        """Test basic statistical properties of random samples."""
        sequencer = RandomSequencer(ndims=3, seed=123)

        # Generate many samples for statistical testing
        samples = sequencer.random(10000)

        # Check basic statistical properties for each dimension
        for dim in range(3):
            dim_samples = samples[:, dim]

            # Mean should be close to 0.5 for uniform [0,1] distribution
            mean = np.mean(dim_samples)
            assert abs(mean - 0.5) < 0.05, f"Mean {mean} too far from 0.5"

            # Standard deviation should be close to 1/sqrt(12) ≈ 0.289 for uniform [0,1]
            std = np.std(dim_samples)
            expected_std = 1.0 / np.sqrt(12)
            assert abs(std - expected_std) < 0.05, f"Std {std} too far from {expected_std}"

            # Should cover the full [0,1] range reasonably well
            assert np.min(dim_samples) < 0.05
            assert np.max(dim_samples) > 0.95

    def test_ndims_edge_cases(self):
        """Test edge cases for number of dimensions."""
        # Single dimension
        sequencer = RandomSequencer(ndims=1)
        samples = sequencer.random(10)
        assert samples.shape == (10, 1)

        # Many dimensions
        sequencer = RandomSequencer(ndims=20)
        samples = sequencer.random(5)
        assert samples.shape == (5, 20)

    def test_zero_samples(self):
        """Test requesting zero samples."""
        sequencer = RandomSequencer(ndims=3)
        samples = sequencer.random(0)
        assert samples.shape == (0, 3)

    def test_rng_state_management(self):
        """Test that RNG state is properly managed."""
        sequencer = RandomSequencer(ndims=2, seed=555)

        # Get initial state
        initial_state = sequencer.rng.get_state()

        # Generate some samples
        sequencer.random(5)

        # State should have changed
        current_state = sequencer.rng.get_state()
        assert not np.array_equal(initial_state[1], current_state[1])

        # Reset should restore initial state
        sequencer.reset()
        reset_state = sequencer.rng.get_state()
        np.testing.assert_array_equal(initial_state[1], reset_state[1])

    def test_inheritance_from_base(self):
        """Test that RandomSequencer properly inherits from BaseSequencer."""
        from curobo._src.util.sampling.sequencer_base import BaseSequencer

        sequencer = RandomSequencer(ndims=3)
        assert isinstance(sequencer, BaseSequencer)

        # Check that all abstract methods are implemented
        assert hasattr(sequencer, "random")
        assert hasattr(sequencer, "reset")
        assert hasattr(sequencer, "fast_forward")

        # Check that they are callable
        assert callable(sequencer.random)
        assert callable(sequencer.reset)
        assert callable(sequencer.fast_forward)

    def test_fast_forward_zero_steps(self):
        """Test fast forward with zero steps."""
        sequencer = RandomSequencer(ndims=3, seed=123)

        # Get initial samples
        sequencer.reset()
        initial_samples = sequencer.random(5)

        # Reset and fast forward zero steps
        sequencer.reset()
        sequencer.fast_forward(0)
        ff_samples = sequencer.random(5)

        # Should be identical
        np.testing.assert_array_equal(initial_samples, ff_samples)

    def test_fast_forward_negative_steps(self):
        """Test fast forward with negative steps (should do nothing)."""
        sequencer = RandomSequencer(ndims=3, seed=123)

        # Get initial samples
        sequencer.reset()
        initial_samples = sequencer.random(5)

        # Reset and fast forward negative steps
        sequencer.reset()
        sequencer.fast_forward(-5)
        ff_samples = sequencer.random(5)

        # Should be identical (negative steps ignored)
        np.testing.assert_array_equal(initial_samples, ff_samples)

    def test_reproducibility_across_instances(self):
        """Test that separate instances with same seed are reproducible."""
        seed = 777

        # Create two separate instances
        sequencer1 = RandomSequencer(ndims=4, seed=seed)
        sequencer2 = RandomSequencer(ndims=4, seed=seed)

        # Generate samples from both
        samples1 = sequencer1.random(20)
        samples2 = sequencer2.random(20)

        # Should be identical
        np.testing.assert_array_equal(samples1, samples2)

        # Reset both and test again
        sequencer1.reset()
        sequencer2.reset()

        samples1_reset = sequencer1.random(20)
        samples2_reset = sequencer2.random(20)

        np.testing.assert_array_equal(samples1_reset, samples2_reset)
        np.testing.assert_array_equal(samples1, samples1_reset)

    def test_uniform_distribution_properties(self):
        """Test that samples follow uniform distribution properties."""
        sequencer = RandomSequencer(ndims=1, seed=12345)

        # Generate large sample for statistical testing
        samples = sequencer.random(50000)[:, 0]  # Take first dimension

        # Test uniformity using Kolmogorov-Smirnov test approximation
        # Divide [0,1] into bins and check distribution
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(samples, bins=bins)

        # Each bin should have roughly equal counts (±10% tolerance)
        expected_count = len(samples) / 10
        for count in hist:
            assert abs(count - expected_count) / expected_count < 0.1
