# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobertsSequencer and related functions."""

import numpy as np

# CuRobo
from curobo._src.util.sampling.sequencer_roberts import RobertsSequencer, _roberts_root


class TestRobertsRoot:
    """Test cases for _roberts_root helper function."""

    def test_golden_ratio_1d(self):
        """Test that 1D case gives the golden ratio."""
        root = _roberts_root(1)
        golden_ratio = (1 + np.sqrt(5)) / 2
        np.testing.assert_allclose(root, golden_ratio, atol=1e-10)

    def test_convergence_different_dims(self):
        """Test convergence for different dimensions."""
        for dim in [1, 2, 3, 5, 7, 10]:
            root = _roberts_root(dim)

            # Verify the root satisfies the equation: x^(d+1) - x - 1 = 0
            equation_value = root ** (dim + 1) - root - 1.0
            np.testing.assert_allclose(equation_value, 0.0, atol=1e-10)

            # Root should be greater than 1
            assert root > 1.0

    def test_monotonicity(self):
        """Test that roots decrease with dimension."""
        roots = [_roberts_root(dim) for dim in range(1, 8)]

        # Roots should be monotonically decreasing for higher dimensions
        for i in range(1, len(roots)):
            assert roots[i] < roots[i - 1]

    def test_numerical_precision(self):
        """Test high numerical precision of root computation."""
        # Test a few specific dimensions for very high precision
        root_2d = _roberts_root(2)
        root_3d = _roberts_root(3)

        # Should satisfy equation to very high precision
        eq_2d = root_2d**3 - root_2d - 1.0
        eq_3d = root_3d**4 - root_3d - 1.0

        assert abs(eq_2d) < 1e-12
        assert abs(eq_3d) < 1e-12


class TestRobertsSequencer:
    """Test cases for RobertsSequencer implementation."""

    def test_init_basic(self):
        """Test basic initialization."""
        sequencer = RobertsSequencer(ndims=5, seed=42)
        assert sequencer.ndims == 5
        assert sequencer.seed == 42
        assert sequencer.current_index == 0
        assert len(sequencer.basis) == 5

    def test_init_default_seed(self):
        """Test initialization with default seed."""
        sequencer = RobertsSequencer(ndims=3)
        assert sequencer.ndims == 3
        assert sequencer.seed == 123

    def test_basis_computation(self):
        """Test that basis vectors are computed correctly."""
        sequencer = RobertsSequencer(ndims=4, seed=100)

        # Check basis vector properties
        assert len(sequencer.basis) == 4

        # All basis values should be in (0, 1)
        assert np.all(sequencer.basis > 0)
        assert np.all(sequencer.basis < 1)

        # Basis vectors should be increasing (1 - 1/r^(1+i) increases as i increases)
        for i in range(1, len(sequencer.basis)):
            assert sequencer.basis[i] > sequencer.basis[i - 1]

    def test_random_output_shape(self):
        """Test that random() returns correct shape."""
        sequencer = RobertsSequencer(ndims=7)

        samples = sequencer.random(100)
        assert samples.shape == (100, 7)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32

    def test_random_output_range(self):
        """Test that random() returns values in [0, 1] range."""
        sequencer = RobertsSequencer(ndims=4)

        samples = sequencer.random(1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_random_different_sizes(self):
        """Test random() with different sample sizes."""
        sequencer = RobertsSequencer(ndims=3)

        for n_samples in [1, 5, 10, 50, 100]:
            samples = sequencer.random(n_samples)
            assert samples.shape == (n_samples, 3)

    def test_reset_functionality(self):
        """Test that reset restores initial state."""
        sequencer = RobertsSequencer(ndims=5, seed=456)

        # Generate initial samples
        initial_samples = sequencer.random(10)
        assert sequencer.current_index == 10

        # Generate more samples to advance state
        sequencer.random(20)
        assert sequencer.current_index == 30

        # Reset and generate again
        sequencer.reset()
        assert sequencer.current_index == 0
        reset_samples = sequencer.random(10)

        # Should be identical to initial samples
        np.testing.assert_array_equal(initial_samples, reset_samples)

    def test_fast_forward_functionality(self):
        """Test fast forward advances sequence correctly."""
        sequencer = RobertsSequencer(ndims=3, seed=789)

        # Generate 15 samples normally
        sequencer.reset()
        normal_samples = sequencer.random(15)

        # Reset, fast forward 10, then generate 5 more
        sequencer.reset()
        sequencer.fast_forward(10)
        assert sequencer.current_index == 10
        ff_samples = sequencer.random(5)

        # The fast forwarded samples should match the last 5 normal samples
        np.testing.assert_array_equal(normal_samples[10:], ff_samples)

    def test_deterministic_behavior(self):
        """Test that sequences are deterministic."""
        sequencer1 = RobertsSequencer(ndims=4, seed=999)
        sequencer2 = RobertsSequencer(ndims=4, seed=999)

        samples1 = sequencer1.random(50)
        samples2 = sequencer2.random(50)

        # Same parameters should produce identical sequences
        np.testing.assert_array_equal(samples1, samples2)

    def test_different_dimensions_different_sequences(self):
        """Test that different dimensions produce different sequences."""
        sequencer_3d = RobertsSequencer(ndims=3, seed=111)
        sequencer_4d = RobertsSequencer(ndims=4, seed=111)

        samples_3d = sequencer_3d.random(100)
        samples_4d = sequencer_4d.random(100)

        # Different dimensions should have different basis vectors
        assert not np.array_equal(sequencer_3d.basis, sequencer_4d.basis[:3])

        # First 3 dimensions should still be different due to different roots
        assert not np.allclose(samples_3d, samples_4d[:, :3])

    def test_low_discrepancy_properties(self):
        """Test basic low-discrepancy properties of Roberts sequence."""
        sequencer = RobertsSequencer(ndims=2, seed=123)

        # Generate many samples for discrepancy testing
        samples = sequencer.random(1000)

        # Check that samples are well-distributed
        for dim in range(2):
            dim_samples = samples[:, dim]

            # Should have good coverage of [0,1] interval
            assert np.min(dim_samples) < 0.05
            assert np.max(dim_samples) > 0.95

            # Mean should be close to 0.5 for uniform distribution over [0,1]
            mean = np.mean(dim_samples)
            assert abs(mean - 0.5) < 0.1

    def test_equidistribution_property(self):
        """Test basic equidistribution property."""
        sequencer = RobertsSequencer(ndims=1, seed=0)

        # Generate samples
        n_samples = 1000
        samples = sequencer.random(n_samples)[:, 0]

        # Divide [0,1] into bins and check distribution
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        hist, _ = np.histogram(samples, bins=bins)

        # For a good quasi-random sequence, bins should be roughly equally filled
        expected_count = n_samples / n_bins
        max_deviation = np.max(np.abs(hist - expected_count))

        # Allow some deviation but should be much better than random
        assert max_deviation < expected_count * 0.3

    def test_ndims_edge_cases(self):
        """Test edge cases for number of dimensions."""
        # Single dimension
        sequencer = RobertsSequencer(ndims=1)
        samples = sequencer.random(10)
        assert samples.shape == (10, 1)

        # Many dimensions
        sequencer = RobertsSequencer(ndims=15)
        samples = sequencer.random(5)
        assert samples.shape == (5, 15)

    def test_zero_samples(self):
        """Test requesting zero samples."""
        sequencer = RobertsSequencer(ndims=3)
        samples = sequencer.random(0)
        assert samples.shape == (0, 3)

    def test_current_index_tracking(self):
        """Test that current_index is properly tracked."""
        sequencer = RobertsSequencer(ndims=2)

        assert sequencer.current_index == 0

        # Generate samples and check index
        sequencer.random(5)
        assert sequencer.current_index == 5

        sequencer.random(10)
        assert sequencer.current_index == 15

        # Reset should restore index
        sequencer.reset()
        assert sequencer.current_index == 0

    def test_inheritance_from_base(self):
        """Test that RobertsSequencer properly inherits from BaseSequencer."""
        from curobo._src.util.sampling.sequencer_base import BaseSequencer

        sequencer = RobertsSequencer(ndims=3)
        assert isinstance(sequencer, BaseSequencer)

        # Check that all abstract methods are implemented
        assert hasattr(sequencer, "random")
        assert hasattr(sequencer, "reset")
        assert hasattr(sequencer, "fast_forward")

    def test_fast_forward_zero_steps(self):
        """Test fast forward with zero steps."""
        sequencer = RobertsSequencer(ndims=3, seed=123)

        initial_index = sequencer.current_index
        sequencer.fast_forward(0)
        assert sequencer.current_index == initial_index

    def test_fast_forward_negative_steps(self):
        """Test fast forward with negative steps (should do nothing)."""
        sequencer = RobertsSequencer(ndims=3, seed=123)

        initial_index = sequencer.current_index
        sequencer.fast_forward(-5)
        assert sequencer.current_index == initial_index

    def test_sequential_generation_consistency(self):
        """Test that sequential generation is consistent with batch generation."""
        sequencer1 = RobertsSequencer(ndims=3, seed=555)
        sequencer2 = RobertsSequencer(ndims=3, seed=555)

        # Generate samples one by one
        sequential_samples = []
        for _ in range(10):
            sample = sequencer1.random(1)
            sequential_samples.append(sample[0])
        sequential_samples = np.array(sequential_samples)

        # Generate all at once
        batch_samples = sequencer2.random(10)

        # Should be identical
        np.testing.assert_array_equal(sequential_samples, batch_samples)

    def test_roberts_formula_implementation(self):
        """Test that the Roberts sequence formula is correctly implemented."""
        sequencer = RobertsSequencer(ndims=2, seed=0)

        # Manually compute first few samples using the formula
        basis = sequencer.basis
        manual_samples = []

        for i in range(5):
            sample = (i * basis) % 1.0
            manual_samples.append(sample)

        manual_samples = np.array(manual_samples, dtype=np.float32)

        # Generate samples using sequencer
        generated_samples = sequencer.random(5)

        # Should match manual computation
        np.testing.assert_array_almost_equal(manual_samples, generated_samples, decimal=6)
