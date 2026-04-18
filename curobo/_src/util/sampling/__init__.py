# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unified sampling module for CuRobo.

This module provides a generalized sampling architecture that separates:
- Sequencers: Generate uniform random sequences (Halton, Random, Roberts, etc.)
- Sample Buffers: Transform sequences to bounded/Gaussian samples
- Processors: Apply domain-specific post-processing (Standard, Knot, STOMP)
- Libraries: High-level sampling APIs with backwards compatibility

The new architecture eliminates code duplication while maintaining full
backwards compatibility with existing SampleLib implementations.

Key Benefits:
- Single SampleBuffer class works with any sequencer
- Deterministic RandomSequencer fixes reset issues
- Post-processors enable different trajectory types
- Factory functions provide easy migration path

Examples:
    Basic usage with any sequencer:
    >>> from curobo.util.sampling import SampleBuffer, HaltonSequencer, RobertsSequencer
    >>> sequencer = HaltonSequencer(ndims=7, seed=123)  # or RobertsSequencer(ndims=7, seed=123)
    >>> buffer = SampleBuffer(sequencer, ndims=7)
    >>> samples = buffer.get_gaussian_samples(1000)

    High-level sample library usage:
    >>> from curobo.util.sampling import create_halton_sample_lib
    >>> sample_lib = create_halton_sample_lib(sample_config, horizon=50, action_dim=7)
    >>> samples = sample_lib.get_samples([1000])

    Custom composition:
    >>> from curobo.util.sampling import SampleBuffer, RandomSequencer, StandardPostProcessor
    >>> sequencer = RandomSequencer(ndims=7, seed=42)
    >>> buffer = SampleBuffer(sequencer, ndims=7)
    >>> processor = StandardPostProcessor(horizon=50, action_dim=7, device_cfg=device_cfg)
    >>> # Use buffer + processor in custom workflow
"""

# Core interfaces
# Universal generator
from .sample_buffer import SampleBuffer
from .sequencer_base import BaseSequencer

# Sequencer implementations
from .sequencer_halton import HaltonSequencer
from .sequencer_random import RandomSequencer
from .sequencer_roberts import RobertsSequencer

# Note: Post-processors are located in optim.particle.sample_strategies
# from .processor_standard import StandardPostProcessor
# from .processor_knot import KnotPostProcessor
# from .processor_stomp import StompPostProcessor

# Note: Unified libraries may not be implemented yet
# from .pipeline_cfg import SamplePipelineCfg
# from .pipeline import SamplePipeline, MultiSamplePipeline

__all__ = [
    # Core interfaces
    "BaseSequencer",
    "SampleBuffer",
    # Sequencer implementations
    "HaltonSequencer",
    "RandomSequencer",
    "RobertsSequencer",
    # Note: Post-processors and unified libraries are in optim.particle.sample_strategies
]
