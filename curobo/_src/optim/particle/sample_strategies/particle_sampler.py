# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unified particle sampler that eliminates code duplication."""

from typing import Any

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
from curobo._src.util.sampling.sequencer_random import RandomSequencer

# Local imports
from .particle_sampler_cfg import ParticleSamplerCfg
from .processor_knot import KnotParticleProcessor
from .processor_standard import StandardParticleProcessor
from .processor_stomp import StompParticleProcessor


class ParticleSampler:
    """Unified particle sampler that works with any sequencer and post-processor.

    This class eliminates code duplication by providing a single implementation
    that can be configured with different sequencers (Halton, Random) and
    post-processors (Standard, Knot, STOMP).

    Example:
        >>> sequencer = HaltonSequencer(ndims=7, seed=123)
        >>> buffer = SampleBuffer(sequencer, ...)
        >>> processor = StandardParticleProcessor(...)
        >>> particle_sampler = ParticleSampler(sample_config, horizon, action_dim, generator, processor)
        >>> samples = particle_sampler.get_samples([1000])
    """

    def __init__(
        self,
        sample_config: ParticleSamplerCfg,
        horizon: int,
        action_dim: int,
        generator: SampleBuffer,
        post_processor: Any,  # StandardParticleProcessor, KnotParticleProcessor, or StompParticleProcessor
    ):
        """Initialize unified particle sampler.

        Args:
            sample_config: Configuration for sampling parameters
            horizon: Time horizon for trajectory sampling
            action_dim: Number of action dimensions (DOF)
            generator: Universal generator with configured sequencer
            post_processor: Post-processor for domain-specific transformations
        """
        self.sample_config = sample_config
        self.sample_shape = 0
        self.samples = None

        self.horizon = horizon
        self.action_dim = action_dim
        self.ndims = self.horizon * self.action_dim
        self.generator = generator
        self.post_processor = post_processor
        if self.post_processor.input_ndims != self.generator.ndims:
            log_and_raise(
                "input_ndims of post_processor {} and generator {} do not match".format(
                    self.post_processor.input_ndims, self.generator.ndims
                )
            )

    def reset_seed(self):
        """Reset the generator and post-processor to initial state."""
        self.generator.reset()

    @profiler.record_function("ParticleSampler/get_samples")
    def get_samples(self, sample_shape, base_seed=None, filter_smooth=False, **kwargs):
        """Get processed samples using configured generator and post-processor.

        Args:
            sample_shape: Shape specification for number of samples
            base_seed: Optional seed override (currently unused, for compatibility)
            filter_smooth: Whether to apply smooth filtering
            **kwargs: Additional arguments passed to post-processor

        Returns:
            Processed samples with shape (sample_shape[0], horizon, action_dim)
        """
        if self.sample_shape != sample_shape or not self.sample_config.fixed_samples:
            if len(sample_shape) > 1:
                log_and_raise("sample shape should be a single value")

            self.sample_shape = sample_shape

            # Generate Gaussian samples from universal generator
            raw_samples = self.generator.get_gaussian_samples(sample_shape[0])

            # Reshape to (batch, horizon, action_dim) for other processors
            raw_samples = raw_samples.view(
                sample_shape[0], self.post_processor.input_horizon, self.action_dim
            )

            # Apply post-processing
            self.samples = self.post_processor.process_samples(raw_samples, filter_smooth)

        if self.samples.shape[0] != sample_shape[0]:
            log_and_raise("sampling failed")

        return self.samples

    @classmethod
    def create_halton_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int
    ) -> "ParticleSampler":
        """Create a Halton-based particle sampler with standard filtering.

        Args:
            sample_config: Sampling configuration
            horizon: Time horizon
            action_dim: Action dimensions

        Returns:
            Configured ParticleSampler with Halton sequencer and standard processing
        """
        sequencer = HaltonSequencer(ndims=horizon * action_dim, seed=sample_config.seed)
        generator = SampleBuffer(
            sequencer=sequencer,
            ndims=horizon * action_dim,
            device_cfg=sample_config.device_cfg,
            store_buffer=2000,
        )
        post_processor = StandardParticleProcessor(
            horizon=horizon,
            action_dim=action_dim,
            device_cfg=sample_config.device_cfg,
            filter_coeffs=getattr(sample_config, "filter_coeffs", None),
        )

        return cls(sample_config, horizon, action_dim, generator, post_processor)

    @classmethod
    def create_random_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int
    ) -> "ParticleSampler":
        """Create a Random-based particle sampler with standard filtering.

        Args:
            sample_config: Sampling configuration
            horizon: Time horizon
            action_dim: Action dimensions

        Returns:
            Configured ParticleSampler with Random sequencer and standard processing
        """
        sequencer = RandomSequencer(ndims=horizon * action_dim, seed=sample_config.seed)
        generator = SampleBuffer(
            sequencer=sequencer,
            ndims=horizon * action_dim,
            device_cfg=sample_config.device_cfg,
            store_buffer=None,  # Random doesn't benefit from buffering as much
        )
        post_processor = StandardParticleProcessor(
            horizon=horizon,
            action_dim=action_dim,
            device_cfg=sample_config.device_cfg,
            filter_coeffs=getattr(sample_config, "filter_coeffs", None),
        )

        return cls(sample_config, horizon, action_dim, generator, post_processor)

    @classmethod
    def create_knot_particle_sampler(
        cls,
        sample_config: ParticleSamplerCfg,
        horizon: int,
        action_dim: int,
        sequencer_type: str = "halton",
    ) -> "ParticleSampler":
        """Create a knot-based particle sampler with B-spline interpolation.

        Args:
            sample_config: Sampling configuration (must have n_knots and degree attributes)
            horizon: Time horizon
            action_dim: Action dimensions
            sequencer_type: "halton" or "random"

        Returns:
            Configured ParticleSampler with specified sequencer and knot processing
        """
        n_knots = getattr(sample_config, "n_knots", 5)
        degree = getattr(sample_config, "degree", 3)

        if sequencer_type == "halton":
            sequencer = HaltonSequencer(ndims=n_knots * action_dim, seed=sample_config.seed)
        else:
            sequencer = RandomSequencer(ndims=n_knots * action_dim, seed=sample_config.seed)

        generator = SampleBuffer(
            sequencer=sequencer,
            ndims=n_knots * action_dim,
            device_cfg=sample_config.device_cfg,
            store_buffer=2000 if sequencer_type == "halton" else None,
        )
        post_processor = KnotParticleProcessor(
            horizon=horizon,
            action_dim=action_dim,
            n_knots=n_knots,
            degree=degree,
            device_cfg=sample_config.device_cfg,
        )

        return cls(
            sample_config,
            horizon,
            action_dim,
            generator,
            post_processor,
        )

    @classmethod
    def create_stomp_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int
    ) -> "ParticleSampler":
        """Create a STOMP-based particle sampler with smooth trajectory generation.

        Args:
            sample_config: Sampling configuration (must have cov_mode attribute)
            horizon: Time horizon
            action_dim: Action dimensions

        Returns:
            Configured ParticleSampler with Halton sequencer and STOMP processing
        """
        # STOMP uses Halton sequencer in original implementation
        sequencer = HaltonSequencer(ndims=horizon * action_dim, seed=sample_config.seed)
        generator = SampleBuffer(
            sequencer=sequencer,
            ndims=horizon * action_dim,
            device_cfg=sample_config.device_cfg,
            store_buffer=2000,
        )
        post_processor = StompParticleProcessor(
            horizon=horizon,
            action_dim=action_dim,
            device_cfg=sample_config.device_cfg,
            stencil_type=getattr(sample_config, "stencil_type", "3point"),
        )

        return cls(sample_config, horizon, action_dim, generator, post_processor)


def create_particle_sampler(
    sample_type: str, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int, **kwargs
) -> ParticleSampler:
    """Factory function to create any particle sampler type.

    Args:
        sample_type: Type of particle sampler ("halton", "random", "halton-knot", "random-knot", "stomp")
        sample_config: Sampling configuration
        horizon: Time horizon
        action_dim: Action dimensions
        **kwargs: Additional arguments for specific sample types

    Returns:
        Configured ParticleSampler of the specified type
    """
    if sample_type == "halton":
        return ParticleSampler.create_halton_particle_sampler(sample_config, horizon, action_dim)
    elif sample_type == "random":
        return ParticleSampler.create_random_particle_sampler(sample_config, horizon, action_dim)
    elif sample_type == "halton-knot":
        return ParticleSampler.create_knot_particle_sampler(
            sample_config, horizon, action_dim, "halton"
        )
    elif sample_type == "random-knot":
        return ParticleSampler.create_knot_particle_sampler(
            sample_config, horizon, action_dim, "random"
        )
    elif sample_type == "stomp":
        return ParticleSampler.create_stomp_particle_sampler(sample_config, horizon, action_dim)
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")


class MixedParticleSampler:
    """Unified replacement for SampleLib that manages multiple particle sample types.

    This class replaces the original SampleLib and eliminates all the duplicated
    sample library instances by using the unified architecture.
    """

    def __init__(self, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int):
        """Initialize unified multi-particle sampler.

        Args:
            sample_config: Configuration with sample_ratio dict
            horizon: Time horizon
            action_dim: Action dimensions
        """
        self.sample_config = sample_config
        self.horizon = horizon
        self.action_dim = action_dim

        self.particle_samplers = {}
        self.sample_fns = {}
        self.samples = None
        self._last_sample_shape = 0

        sample_ratio = getattr(sample_config, "sample_ratio", {})

        for sample_type, ratio in sample_ratio.items():
            if ratio > 0.0:
                self.particle_samplers[sample_type] = create_particle_sampler(
                    sample_type, sample_config, horizon, action_dim
                )
                self.sample_fns[sample_type] = self.particle_samplers[sample_type].get_samples

    def reset_seed(self):
        """Reset all active particle samplers."""
        for particle_sampler in self.particle_samplers.values():
            particle_sampler.reset_seed()

    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        """Get mixed samples according to configured ratios.

        Args:
            sample_shape: Shape specification for number of samples
            base_seed: Optional seed override (for compatibility)
            **kwargs: Additional arguments passed to individual samplers

        Returns:
            Mixed samples concatenated according to configured ratios
        """
        # Check if we need to regenerate samples
        if (
            (not self.sample_config.fixed_samples)
            or self.samples is None
            or sample_shape[0] != getattr(self, "_last_sample_shape", 0)
        ):
            cat_list = []
            sample_shape = list(sample_shape)
            sample_ratio = getattr(self.sample_config, "sample_ratio", {})

            for sample_type, ratio in sample_ratio.items():
                if ratio == 0.0 or sample_type not in self.sample_fns:
                    continue

                # Calculate number of samples for this type
                n_samples = round(sample_shape[0] * ratio)
                if n_samples == 0:
                    continue

                s_shape = [n_samples]

                # Get samples from the corresponding particle sampler
                samples = self.sample_fns[sample_type](s_shape, base_seed=base_seed, **kwargs)
                cat_list.append(samples)

            if cat_list:
                self.samples = torch.cat(cat_list, dim=0)
            else:
                # Fallback: create empty tensor with correct shape
                self.samples = torch.zeros(
                    (sample_shape[0], self.horizon, self.action_dim),
                    dtype=self.sample_config.device_cfg.dtype,
                    device=self.sample_config.device_cfg.device,
                )

            self._last_sample_shape = sample_shape[0]

        return self.samples
