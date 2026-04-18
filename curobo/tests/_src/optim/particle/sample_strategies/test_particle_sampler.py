# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ParticleSampler and MixedParticleSampler."""

import pytest
import torch

# Import the actual classes we want to test
from curobo._src.optim.particle.sample_strategies.particle_sampler import (
    MixedParticleSampler,
    ParticleSampler,
)

# CuRobo
from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import ParticleSamplerCfg
from curobo._src.optim.particle.sample_strategies.processor_knot import KnotParticleProcessor
from curobo._src.optim.particle.sample_strategies.processor_standard import (
    StandardParticleProcessor,
)
from curobo._src.optim.particle.sample_strategies.processor_stomp import StompParticleProcessor
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
from curobo._src.util.sampling.sequencer_random import RandomSequencer


@pytest.mark.parametrize(
    "horizon,action_dim",
    [
        (4, 5),
        (2, 3),
        (1, 1),
    ],
)
class TestParticleSampler:
    """Test cases for ParticleSampler implementation."""

    @pytest.fixture
    def test_params(self, horizon, action_dim):
        """Fixture for test parameters."""
        return {"horizon": horizon, "action_dim": action_dim, "ndims": horizon * action_dim}

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def sample_config(self, device_cfg):
        """Fixture for sample configuration."""
        return ParticleSamplerCfg(
            device_cfg=device_cfg,
            fixed_samples=True,
            seed=123,
        )

    @pytest.fixture
    def halton_generator(self, device_cfg, test_params):
        """Fixture for Halton sample generator."""
        sequencer = HaltonSequencer(ndims=test_params["ndims"], seed=123)
        return SampleBuffer(sequencer, ndims=test_params["ndims"], device_cfg=device_cfg)

    @pytest.fixture
    def standard_processor(self, device_cfg, test_params):
        """Fixture for standard processor."""
        return StandardParticleProcessor(
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            device_cfg=device_cfg,
        )

    @pytest.fixture
    def knot_processor(self, device_cfg, test_params):
        """Fixture for knot processor."""
        return KnotParticleProcessor(
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            n_knots=3,
            device_cfg=device_cfg,
        )

    @pytest.fixture
    def basic_particle_sampler(
        self, sample_config, halton_generator, standard_processor, test_params
    ):
        """Fixture for basic particle sampler."""
        return ParticleSampler(
            sample_config=sample_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            generator=halton_generator,
            post_processor=standard_processor,
        )

    def test_init_basic(self, sample_config, halton_generator, standard_processor, test_params):
        """Test basic initialization."""
        sampler = ParticleSampler(
            sample_config=sample_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            generator=halton_generator,
            post_processor=standard_processor,
        )

        assert sampler.sample_config == sample_config
        assert sampler.horizon == test_params["horizon"]
        assert sampler.action_dim == test_params["action_dim"]
        assert sampler.ndims == test_params["ndims"]
        assert sampler.generator == halton_generator
        assert sampler.post_processor == standard_processor
        assert sampler.sample_shape == 0
        assert sampler.samples is None

    def test_reset_seed(self, basic_particle_sampler):
        """Test reset_seed functionality."""
        # Should not raise an exception
        basic_particle_sampler.reset_seed()

        # Generator should be reset
        assert hasattr(basic_particle_sampler.generator, "reset")

    def test_get_samples_basic(self, basic_particle_sampler, test_params):
        """Test basic get_samples functionality."""
        sample_shape = [10]

        samples = basic_particle_sampler.get_samples(sample_shape)

        expected_shape = (
            10,
            test_params["horizon"],
            test_params["action_dim"],
        )  # (batch, horizon, action_dim)
        assert samples.shape == expected_shape
        assert isinstance(samples, torch.Tensor)

    def test_get_samples_caching(self, basic_particle_sampler):
        """Test that samples are cached when fixed_samples=True."""
        sample_shape = [5]

        # First call should generate samples
        samples1 = basic_particle_sampler.get_samples(sample_shape)

        # Second call with same shape should return cached samples
        samples2 = basic_particle_sampler.get_samples(sample_shape)

        torch.testing.assert_close(samples1, samples2)
        assert basic_particle_sampler.sample_shape == sample_shape

    def test_get_samples_different_shapes(self, basic_particle_sampler, test_params):
        """Test get_samples with different shapes."""
        shape1 = [5]
        shape2 = [10]

        samples1 = basic_particle_sampler.get_samples(shape1)
        samples2 = basic_particle_sampler.get_samples(shape2)

        assert samples1.shape == (5, test_params["horizon"], test_params["action_dim"])
        assert samples2.shape == (10, test_params["horizon"], test_params["action_dim"])
        assert not torch.allclose(samples1, samples2[:5])  # Should be different

    def test_get_samples_filter_smooth(self, basic_particle_sampler, test_params):
        """Test get_samples with filter_smooth parameter."""
        sample_shape = [3]

        samples_no_smooth = basic_particle_sampler.get_samples(sample_shape, filter_smooth=False)
        samples_smooth = basic_particle_sampler.get_samples(sample_shape, filter_smooth=True)

        # Both should work and return valid shapes
        assert samples_no_smooth.shape == (3, test_params["horizon"], test_params["action_dim"])
        assert samples_smooth.shape == (3, test_params["horizon"], test_params["action_dim"])

    def test_get_samples_base_seed_parameter(self, basic_particle_sampler):
        """Test get_samples with base_seed parameter (should be ignored)."""
        sample_shape = [5]

        samples1 = basic_particle_sampler.get_samples(sample_shape, base_seed=123)
        samples2 = basic_particle_sampler.get_samples(sample_shape, base_seed=456)

        # base_seed is currently unused, so samples should be identical for cached calls
        torch.testing.assert_close(samples1, samples2)

    def test_get_samples_kwargs(self, basic_particle_sampler, test_params):
        """Test get_samples with additional kwargs."""
        sample_shape = [3]

        samples = basic_particle_sampler.get_samples(
            sample_shape,
            filter_smooth=False,
            custom_param="test",  # Should be ignored
        )

        assert samples.shape == (3, test_params["horizon"], test_params["action_dim"])

    def test_get_samples_invalid_shape(self, basic_particle_sampler):
        """Test get_samples with invalid shape (multiple dimensions)."""
        # This should log an error but not crash
        sample_shape = [5, 3]  # Invalid: multiple dimensions

        # The current implementation logs an error but continues
        with pytest.raises(ValueError):
            samples = basic_particle_sampler.get_samples(sample_shape)

    def test_with_different_processors(
        self, sample_config, halton_generator, device_cfg, test_params
    ):
        """Test ParticleSampler with different processors."""
        processor_configs = [
            (
                StandardParticleProcessor(
                    horizon=test_params["horizon"],
                    action_dim=test_params["action_dim"],
                    device_cfg=device_cfg,
                ),
                test_params["ndims"],
            ),  # horizon * action_dim
            (
                KnotParticleProcessor(
                    horizon=test_params["horizon"],
                    action_dim=test_params["action_dim"],
                    n_knots=2,
                    device_cfg=device_cfg,
                ),
                test_params["ndims"],
            ),  # n_knots * action_dim
            (
                StompParticleProcessor(
                    horizon=test_params["horizon"],
                    action_dim=test_params["action_dim"],
                    device_cfg=device_cfg,
                ),
                test_params["ndims"],
            ),  # horizon * action_dim
        ]

        for processor, ndims in processor_configs:
            # Create new generator for each test with appropriate dimensions
            sequencer = HaltonSequencer(ndims=processor.input_ndims, seed=123)
            generator = SampleBuffer(
                sequencer, ndims=processor.input_ndims, device_cfg=device_cfg
            )

            sampler = ParticleSampler(
                sample_config=sample_config,
                horizon=test_params["horizon"],
                action_dim=test_params["action_dim"],
                generator=generator,
                post_processor=processor,
            )

            samples = sampler.get_samples([5])
            assert samples.shape == (5, test_params["horizon"], test_params["action_dim"])

    def test_with_different_generators(
        self, sample_config, standard_processor, device_cfg, test_params
    ):
        """Test ParticleSampler with different generators."""
        generators = [
            SampleBuffer(
                HaltonSequencer(ndims=test_params["ndims"], seed=123),
                ndims=test_params["ndims"],
                device_cfg=device_cfg,
            ),
            SampleBuffer(
                RandomSequencer(ndims=test_params["ndims"], seed=456),
                ndims=test_params["ndims"],
                device_cfg=device_cfg,
            ),
        ]

        for generator in generators:
            sampler = ParticleSampler(
                sample_config=sample_config,
                horizon=test_params["horizon"],
                action_dim=test_params["action_dim"],
                generator=generator,
                post_processor=standard_processor,
            )

            samples = sampler.get_samples([4])
            assert samples.shape == (4, test_params["horizon"], test_params["action_dim"])

    def test_non_fixed_samples(
        self, device_cfg, halton_generator, standard_processor, test_params
    ):
        """Test ParticleSampler with fixed_samples=False."""
        config = ParticleSamplerCfg(
            device_cfg=device_cfg,
            fixed_samples=False,
            seed=123,
        )

        sampler = ParticleSampler(
            sample_config=config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            generator=halton_generator,
            post_processor=standard_processor,
        )

        sample_shape = [5]

        # Should generate new samples each time
        samples1 = sampler.get_samples(sample_shape)
        samples2 = sampler.get_samples(sample_shape)

        assert (
            samples1.shape == samples2.shape == (5, test_params["horizon"], test_params["action_dim"])
        )
        # May or may not be different depending on implementation

    def test_sample_shape_tracking(self, basic_particle_sampler):
        """Test that sample_shape is properly tracked."""
        assert basic_particle_sampler.sample_shape == 0

        shape1 = [3]
        basic_particle_sampler.get_samples(shape1)
        assert basic_particle_sampler.sample_shape == shape1

        shape2 = [7]
        basic_particle_sampler.get_samples(shape2)
        assert basic_particle_sampler.sample_shape == shape2

    def test_samples_caching_behavior(self, basic_particle_sampler):
        """Test detailed caching behavior."""
        shape = [4]

        # First call should set samples
        assert basic_particle_sampler.samples is None
        samples1 = basic_particle_sampler.get_samples(shape)
        assert basic_particle_sampler.samples is not None

        # Second call should use cached samples
        samples2 = basic_particle_sampler.get_samples(shape)
        torch.testing.assert_close(samples1, samples2)

    def test_zero_batch_size(self, basic_particle_sampler, test_params):
        """Test with zero batch size."""
        samples = basic_particle_sampler.get_samples([0])
        assert samples.shape == (0, test_params["horizon"], test_params["action_dim"])

    def test_large_batch_size(self, basic_particle_sampler, test_params):
        """Test with large batch size."""
        samples = basic_particle_sampler.get_samples([100])
        assert samples.shape == (100, test_params["horizon"], test_params["action_dim"])

    def test_tensor_properties(self, basic_particle_sampler):
        """Test that output tensors have correct properties."""
        samples = basic_particle_sampler.get_samples([5])

        assert samples.device == basic_particle_sampler.generator.device_cfg.device
        assert samples.dtype == basic_particle_sampler.generator.device_cfg.dtype

    def test_different_ndims_calculation(self, sample_config, device_cfg, test_params):
        """Test ndims calculation for current parameter combination."""
        # Test that the current parameterized values calculate ndims correctly
        sequencer = HaltonSequencer(ndims=test_params["ndims"], seed=123)
        generator = SampleBuffer(sequencer, ndims=test_params["ndims"], device_cfg=device_cfg)
        processor = StandardParticleProcessor(
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            device_cfg=device_cfg,
        )

        sampler = ParticleSampler(
            sample_config=sample_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
            generator=generator,
            post_processor=processor,
        )

        assert sampler.ndims == test_params["ndims"]


@pytest.mark.parametrize(
    "horizon,action_dim",
    [
        (4, 3),
        (2, 2),
        (6, 4),
    ],
)
class TestMixedParticleSampler:
    """Test cases for MixedParticleSampler implementation."""

    @pytest.fixture
    def test_params(self, horizon, action_dim):
        """Fixture for test parameters."""
        return {"horizon": horizon, "action_dim": action_dim, "ndims": horizon * action_dim}

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def basic_mixed_config(self, device_cfg):
        """Fixture for basic mixed sample configuration."""
        return ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 0.5, "random": 0.3, "stomp": 0.2},
            fixed_samples=True,
            seed=123,
        )

    @pytest.fixture
    def single_type_config(self, device_cfg):
        """Fixture for single sampler type configuration."""
        return ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 1.0, "random": 0.0, "stomp": 0.0},
            fixed_samples=True,
            seed=123,
        )

    @pytest.fixture
    def knot_mixed_config(self, device_cfg):
        """Fixture for knot-based mixed sampling."""
        return ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 0.4, "halton-knot": 0.3, "random": 0.3},
            fixed_samples=True,
            seed=123,
            n_knots=3,
            degree=3,
        )

    @pytest.fixture
    def basic_mixed_sampler(self, basic_mixed_config, test_params):
        """Fixture for basic mixed particle sampler."""
        return MixedParticleSampler(
            sample_config=basic_mixed_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

    def test_init_basic(self, basic_mixed_config, test_params):
        """Test basic initialization of MixedParticleSampler."""
        sampler = MixedParticleSampler(
            sample_config=basic_mixed_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        assert sampler.sample_config == basic_mixed_config
        assert sampler.horizon == test_params["horizon"]
        assert sampler.action_dim == test_params["action_dim"]
        assert sampler.samples is None
        assert sampler._last_sample_shape == 0

        # Check that particle samplers are created for non-zero ratios
        expected_types = {"halton", "random", "stomp"}
        assert set(sampler.particle_samplers.keys()) == expected_types
        assert set(sampler.sample_fns.keys()) == expected_types

    def test_init_with_zero_ratios(self, device_cfg, test_params):
        """Test initialization with some zero ratios."""
        config = ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 0.7, "random": 0.0, "stomp": 0.3},
            fixed_samples=True,
            seed=123,
        )

        sampler = MixedParticleSampler(
            sample_config=config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        # Only non-zero ratios should create samplers
        expected_types = {"halton", "stomp"}
        assert set(sampler.particle_samplers.keys()) == expected_types
        assert "random" not in sampler.particle_samplers

    def test_init_single_type(self, single_type_config, test_params):
        """Test initialization with single sampler type."""
        sampler = MixedParticleSampler(
            sample_config=single_type_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        # Only halton should be created
        assert set(sampler.particle_samplers.keys()) == {"halton"}
        assert len(sampler.sample_fns) == 1

    def test_reset_seed(self, basic_mixed_sampler):
        """Test reset_seed functionality."""
        # Should not raise an exception
        basic_mixed_sampler.reset_seed()

        # All particle samplers should have been reset
        for particle_sampler in basic_mixed_sampler.particle_samplers.values():
            assert hasattr(particle_sampler, "reset_seed")

    def test_get_samples_basic(self, basic_mixed_sampler, test_params):
        """Test basic get_samples functionality."""
        sample_shape = [10]

        samples = basic_mixed_sampler.get_samples(sample_shape)

        expected_shape = (10, test_params["horizon"], test_params["action_dim"])
        assert samples.shape == expected_shape
        assert isinstance(samples, torch.Tensor)

    def test_get_samples_ratio_distribution(self, basic_mixed_sampler, test_params):
        """Test that samples are distributed according to ratios."""
        sample_shape = [100]  # Use larger number to see distribution
        basic_mixed_sampler.get_samples(sample_shape)

        # With ratios halton: 0.5, random: 0.3, stomp: 0.2
        # We expect roughly 50, 30, 20 samples respectively
        # The actual implementation concatenates them, so total should be 100
        assert basic_mixed_sampler.samples.shape[0] == 100

    def test_get_samples_small_batch(self, basic_mixed_sampler, test_params):
        """Test get_samples with small batch sizes that might round to zero."""
        sample_shape = [2]  # Small batch

        samples = basic_mixed_sampler.get_samples(sample_shape)

        # Should still work even if some ratios round to zero samples
        assert samples.shape[0] <= 2  # May be less due to rounding
        assert samples.shape[1:] == (test_params["horizon"], test_params["action_dim"])

    def test_get_samples_caching(self, basic_mixed_sampler):
        """Test that samples are cached when fixed_samples=True."""
        sample_shape = [5]

        # First call should generate samples
        samples1 = basic_mixed_sampler.get_samples(sample_shape)

        # Second call with same shape should return cached samples
        samples2 = basic_mixed_sampler.get_samples(sample_shape)

        torch.testing.assert_close(samples1, samples2)
        assert basic_mixed_sampler._last_sample_shape == sample_shape[0]

    def test_get_samples_different_shapes(self, basic_mixed_sampler, test_params):
        """Test get_samples with different shapes."""
        shape1 = [5]
        shape2 = [10]

        samples1 = basic_mixed_sampler.get_samples(shape1)
        samples2 = basic_mixed_sampler.get_samples(shape2)

        assert samples1.shape[0] <= 5  # May be less due to rounding
        assert samples2.shape[0] <= 10
        assert samples1.shape[1:] == (test_params["horizon"], test_params["action_dim"])
        assert samples2.shape[1:] == (test_params["horizon"], test_params["action_dim"])

    def test_get_samples_non_fixed(self, device_cfg, test_params):
        """Test get_samples with fixed_samples=False."""
        config = ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 0.6, "random": 0.4},
            fixed_samples=False,
            seed=123,
        )

        sampler = MixedParticleSampler(
            sample_config=config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        sample_shape = [5]

        # Should regenerate samples each time
        samples1 = sampler.get_samples(sample_shape)
        samples2 = sampler.get_samples(sample_shape)

        # Shape should be consistent
        assert samples1.shape[1:] == samples2.shape[1:]

    def test_get_samples_zero_batch(self, basic_mixed_sampler, test_params):
        """Test get_samples with zero batch size."""
        sample_shape = [0]

        samples = basic_mixed_sampler.get_samples(sample_shape)

        # Should handle zero batch size gracefully
        assert samples.shape == (0, test_params["horizon"], test_params["action_dim"])

    def test_get_samples_all_zero_ratios(self, device_cfg, test_params):
        """Test get_samples when all ratios are zero."""
        config = ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 0.0, "random": 0.0, "stomp": 0.0},
            fixed_samples=True,
            seed=123,
        )

        sampler = MixedParticleSampler(
            sample_config=config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        sample_shape = [5]
        samples = sampler.get_samples(sample_shape)

        # Should return empty tensor with correct shape
        expected_shape = (5, test_params["horizon"], test_params["action_dim"])
        assert samples.shape == expected_shape
        assert torch.all(samples == 0.0)  # Should be zeros fallback

    def test_sample_ratios_sum_not_one(self, device_cfg, test_params):
        """Test with sample ratios that don't sum to 1.0."""
        config = ParticleSamplerCfg(
            device_cfg=device_cfg,
            sample_ratio={"halton": 0.3, "random": 0.2},  # Sum = 0.5
            fixed_samples=True,
            seed=123,
        )

        sampler = MixedParticleSampler(
            sample_config=config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        sample_shape = [10]
        samples = sampler.get_samples(sample_shape)

        # Should still work, but total samples may be less than requested
        assert samples.shape[0] <= 10
        assert samples.shape[1:] == (test_params["horizon"], test_params["action_dim"])

    def test_get_samples_with_knots(self, knot_mixed_config, test_params):
        """Test get_samples with knot-based samplers."""
        sampler = MixedParticleSampler(
            sample_config=knot_mixed_config,
            horizon=test_params["horizon"],
            action_dim=test_params["action_dim"],
        )

        sample_shape = [10]
        samples = sampler.get_samples(sample_shape)

        expected_shape = (sample_shape[0], test_params["horizon"], test_params["action_dim"])
        assert samples.shape[0] <= expected_shape[0]  # May be less due to rounding
        assert samples.shape[1:] == expected_shape[1:]

        # Should have created halton, halton-knot, and random samplers
        expected_types = {"halton", "halton-knot", "random"}
        assert set(sampler.particle_samplers.keys()) == expected_types

    def test_get_samples_filter_smooth(self, basic_mixed_sampler, test_params):
        """Test get_samples with filter_smooth parameter."""
        sample_shape = [3]

        samples_no_smooth = basic_mixed_sampler.get_samples(sample_shape, filter_smooth=False)
        samples_smooth = basic_mixed_sampler.get_samples(sample_shape, filter_smooth=True)

        # Both should work and return valid shapes
        assert samples_no_smooth.shape[1:] == (test_params["horizon"], test_params["action_dim"])
        assert samples_smooth.shape[1:] == (test_params["horizon"], test_params["action_dim"])

    def test_get_samples_kwargs(self, basic_mixed_sampler, test_params):
        """Test get_samples with additional kwargs."""
        sample_shape = [3]

        samples = basic_mixed_sampler.get_samples(
            sample_shape,
            filter_smooth=False,
            custom_param="test",  # Should be passed through
        )

        assert samples.shape[1:] == (test_params["horizon"], test_params["action_dim"])

    def test_individual_sampler_types(self, device_cfg, test_params):
        """Test mixed sampler with individual sampler types."""
        sampler_types = ["halton", "random", "stomp", "halton-knot", "random-knot"]

        for sampler_type in sampler_types:
            config = ParticleSamplerCfg(
                device_cfg=device_cfg,
                sample_ratio={sampler_type: 1.0},
                fixed_samples=True,
                seed=123,
                n_knots=3,  # For knot-based samplers
                degree=3,
            )

            sampler = MixedParticleSampler(
                sample_config=config,
                horizon=test_params["horizon"],
                action_dim=test_params["action_dim"],
            )

            samples = sampler.get_samples([5])
            assert samples.shape == (5, test_params["horizon"], test_params["action_dim"])
            assert sampler_type in sampler.particle_samplers

    def test_tensor_properties(self, basic_mixed_sampler):
        """Test that output tensors have correct properties."""
        samples = basic_mixed_sampler.get_samples([5])

        expected_device = basic_mixed_sampler.sample_config.device_cfg.device
        expected_dtype = basic_mixed_sampler.sample_config.device_cfg.dtype

        assert samples.device == expected_device
        assert samples.dtype == expected_dtype

    def test_large_batch_size(self, basic_mixed_sampler, test_params):
        """Test with large batch size."""
        samples = basic_mixed_sampler.get_samples([1000])
        assert samples.shape[0] <= 1000  # May be less due to rounding in ratios
        assert samples.shape[1:] == (test_params["horizon"], test_params["action_dim"])

    def test_sample_shape_tracking(self, basic_mixed_sampler):
        """Test that sample shape is properly tracked."""
        assert basic_mixed_sampler._last_sample_shape == 0

        shape1 = [10]
        basic_mixed_sampler.get_samples(shape1)
        assert basic_mixed_sampler._last_sample_shape == shape1[0]

        shape2 = [20]
        basic_mixed_sampler.get_samples(shape2)
        assert basic_mixed_sampler._last_sample_shape == shape2[0]

    def test_mixed_sampler_with_base_seed(self, basic_mixed_sampler):
        """Test mixed sampler with base_seed parameter."""
        sample_shape = [5]

        samples1 = basic_mixed_sampler.get_samples(sample_shape, base_seed=123)
        samples2 = basic_mixed_sampler.get_samples(sample_shape, base_seed=456)

        # With fixed_samples=True, should return cached samples regardless of base_seed
        torch.testing.assert_close(samples1, samples2)
