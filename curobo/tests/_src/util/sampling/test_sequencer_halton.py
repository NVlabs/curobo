# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for HaltonSequencer."""

import numpy as np

# CuRobo
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer


class TestHaltonSequencer:
    """Test cases for HaltonSequencer implementation."""

    def test_init_basic(self):
        """Test basic initialization."""
        sequencer = HaltonSequencer(ndims=5, seed=42)
        assert sequencer.ndims == 5
        assert sequencer.seed == 42
        assert sequencer.scramble is True  # Default scrambling

    def test_init_no_scrambling(self):
        """Test initialization without scrambling."""
        sequencer = HaltonSequencer(ndims=3, seed=123, scramble=False)
        assert sequencer.ndims == 3
        assert sequencer.seed == 123
        assert sequencer.scramble is False

    def test_random_output_shape(self):
        """Test that random() returns correct shape."""
        sequencer = HaltonSequencer(ndims=7)

        samples = sequencer.random(100)
        assert samples.shape == (100, 7)
        assert isinstance(samples, np.ndarray)

    def test_random_output_range(self):
        """Test that random() returns values in [0, 1] range."""
        sequencer = HaltonSequencer(ndims=4)

        samples = sequencer.random(1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_random_different_sizes(self):
        """Test random() with different sample sizes."""
        sequencer = HaltonSequencer(ndims=3)

        for n_samples in [1, 5, 10, 50, 100]:
            samples = sequencer.random(n_samples)
            assert samples.shape == (n_samples, 3)

    def test_reset_functionality(self):
        """Test that reset restores initial state."""
        sequencer = HaltonSequencer(ndims=5, seed=456)

        # Generate initial samples
        initial_samples = sequencer.random(10)

        # Generate more samples to advance state
        sequencer.random(20)

        # Reset and generate again
        sequencer.reset()
        reset_samples = sequencer.random(10)

        # Should be identical to initial samples
        np.testing.assert_array_almost_equal(initial_samples, reset_samples, decimal=6)

    def test_fast_forward_functionality(self):
        """Test fast forward advances sequence correctly."""
        sequencer = HaltonSequencer(ndims=3, seed=789)

        # Generate 15 samples normally
        sequencer.reset()
        normal_samples = sequencer.random(15)

        # Reset, fast forward 10, then generate 5 more
        sequencer.reset()
        sequencer.fast_forward(10)
        ff_samples = sequencer.random(5)

        # The fast forwarded samples should match the last 5 normal samples
        np.testing.assert_array_almost_equal(normal_samples[10:], ff_samples, decimal=6)

    def test_seed_determinism(self):
        """Test that same seed produces deterministic sequences."""
        seed = 999
        sequencer1 = HaltonSequencer(ndims=4, seed=seed)
        sequencer2 = HaltonSequencer(ndims=4, seed=seed)

        samples1 = sequencer1.random(50)
        samples2 = sequencer2.random(50)

        # Same seed should produce identical sequences
        np.testing.assert_array_almost_equal(samples1, samples2, decimal=6)

    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        sequencer1 = HaltonSequencer(ndims=3, seed=111)
        sequencer2 = HaltonSequencer(ndims=3, seed=222)

        samples1 = sequencer1.random(100)
        samples2 = sequencer2.random(100)

        # Different seeds should produce different sequences
        assert not np.allclose(samples1, samples2)

    def test_scrambling_vs_no_scrambling(self):
        """Test that scrambling affects the sequence."""
        seed = 555
        sequencer_scrambled = HaltonSequencer(ndims=3, seed=seed, scramble=True)
        sequencer_unscrambled = HaltonSequencer(ndims=3, seed=seed, scramble=False)

        samples_scrambled = sequencer_scrambled.random(50)
        samples_unscrambled = sequencer_unscrambled.random(50)

        # Scrambled and unscrambled should be different
        assert not np.allclose(samples_scrambled, samples_unscrambled)

    def test_low_discrepancy_properties(self):
        """Test basic low-discrepancy properties of Halton sequence."""
        sequencer = HaltonSequencer(ndims=2, scramble=False)

        # Generate many samples
        samples = sequencer.random(1000)

        # Check that samples are well-distributed across dimensions
        for dim in range(2):
            dim_samples = samples[:, dim]

            # Should have reasonable coverage of [0,1] interval
            assert np.min(dim_samples) < 0.1
            assert np.max(dim_samples) > 0.9

            # Mean should be close to 0.5 for uniform distribution
            assert abs(np.mean(dim_samples) - 0.5) < 0.1

    def test_ndims_edge_cases(self):
        """Test edge cases for number of dimensions."""
        # Single dimension
        sequencer = HaltonSequencer(ndims=1)
        samples = sequencer.random(10)
        assert samples.shape == (10, 1)

        # Many dimensions
        sequencer = HaltonSequencer(ndims=20)
        samples = sequencer.random(5)
        assert samples.shape == (5, 20)

    def test_zero_samples(self):
        """Test requesting zero samples."""
        sequencer = HaltonSequencer(ndims=3)
        samples = sequencer.random(0)
        assert samples.shape == (0, 3)

    def test_qmc_interface_access(self):
        """Test that internal QMC object is accessible."""
        sequencer = HaltonSequencer(ndims=4, seed=123)
        assert hasattr(sequencer, "qmc")
        assert sequencer.qmc.d == 4

    def test_inheritance_from_base(self):
        """Test that HaltonSequencer properly inherits from BaseSequencer."""
        from curobo._src.util.sampling.sequencer_base import BaseSequencer

        sequencer = HaltonSequencer(ndims=3)
        assert isinstance(sequencer, BaseSequencer)

        # Check that all abstract methods are implemented
        assert hasattr(sequencer, "random")
        assert hasattr(sequencer, "reset")
        assert hasattr(sequencer, "fast_forward")

        # Check that they are callable
        assert callable(sequencer.random)
        assert callable(sequencer.reset)
        assert callable(sequencer.fast_forward)

    def test_state_consistency_after_operations(self):
        """Test that state remains consistent after various operations."""
        sequencer = HaltonSequencer(ndims=2, seed=100)

        # Generate some samples
        samples1 = sequencer.random(5)

        # Fast forward and generate more
        sequencer.fast_forward(3)
        samples2 = sequencer.random(2)

        # Reset and verify we can reproduce initial sequence
        sequencer.reset()
        samples1_repeat = sequencer.random(5)

        np.testing.assert_array_almost_equal(samples1, samples1_repeat, decimal=6)
