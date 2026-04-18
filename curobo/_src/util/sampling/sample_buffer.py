# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Sample buffer that works with any sequencer."""

# Standard Library
from typing import List, Optional

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sequencer_base import BaseSequencer
from curobo._src.util.torch_util import get_torch_jit_decorator


class SampleBuffer:
    """Sample buffer that works with any sequencer.

    This class generalizes the HaltonGenerator to work with any sequencer
    (Halton, Random, Sobol, etc.) while providing the same transformation
    and buffering capabilities.

    The SampleBuffer handles:
    - Sequencer management (any SequencerBase implementation)
    - Uniform to bounded sample transformation
    - Uniform to Gaussian sample transformation
    - Optional sample buffering for performance
    - Device and dtype management for PyTorch

    Example:
        >>> from .sequencer_halton import HaltonSequencer
        >>> sequencer = HaltonSequencer(ndims=7, seed=123)
        >>> buffer = SampleBuffer(sequencer, ndims=7)
        >>> samples = buffer.get_gaussian_samples(1000)
    """

    def __init__(
        self,
        sequencer: BaseSequencer,
        ndims: int,
        device_cfg: DeviceCfg = DeviceCfg(),
        up_bounds: List[float] = [1],
        low_bounds: List[float] = [0],
        store_buffer: Optional[int] = 2000,
    ):
        """Initialize sample buffer.

        Args:
            sequencer: Any sequencer implementing BaseSequencer
            ndims: Number of dimensions for sample generation
            device_cfg: Device and dtype configuration for PyTorch tensors
            up_bounds: Upper bounds for bounded sampling
            low_bounds: Lower bounds for bounded sampling
            store_buffer: Size of sample buffer for performance (None to disable)
        """
        self.sequencer = sequencer
        self.device_cfg = device_cfg
        if self.sequencer.ndims != ndims:
            log_and_raise(f"Sequencer ndims ({self.sequencer.ndims}) does not match ndims ({ndims})")
        self.ndims = ndims

        # Scale samples by joint range
        up_bounds = self.device_cfg.to_device(up_bounds)
        low_bounds = self.device_cfg.to_device(low_bounds)
        self.range_b = up_bounds - low_bounds
        self.up_bounds = up_bounds
        self.low_bounds = low_bounds

        # Matrices for Gaussian transformation
        self.proj_mat = torch.sqrt(
            torch.tensor([2.0], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        )
        self.i_mat = torch.eye(
            self.ndims, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

        # Optional sample buffering for performance
        self._sample_buffer = None
        self._store_buffer = store_buffer
        self.fixed_samples = store_buffer is not None

        if store_buffer is not None:
            # Pre-generate samples and randomly sample from buffer
            self._sample_buffer = torch.tensor(
                self.sequencer.random(store_buffer),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._int_gen = torch.Generator(device=self.device_cfg.device)
            self._int_gen = self._int_gen.manual_seed(getattr(sequencer, "seed", 123))
            self._generator_initial_state = self._int_gen.get_state().clone()
            self._index_buffer = None

    def reset(self):
        """Reset generator to initial state."""
        if self._sample_buffer is not None:
            self._int_gen.set_state(self._generator_initial_state)
        self.sequencer.reset()

    def fast_forward(self, steps: int):
        """Fast forward sequencer by steps.

        Args:
            steps: Number of steps to advance the sequencer
        """
        self.sequencer.fast_forward(steps)
        if self.fixed_samples:
            log_and_raise("fast forward will not work with fixed samples.")

    def _get_samples(self, num_samples: int):
        """Get raw uniform samples from sequencer or buffer.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Tensor of uniform samples in [0,1]^ndims
        """
        if self._sample_buffer is not None:
            # Sample from pre-generated buffer
            out_buffer = None
            if self._index_buffer is not None and self._index_buffer.shape[0] == num_samples:
                out_buffer = self._index_buffer
            samples, index = self.sample_by_random_index(
                self._sample_buffer, num_samples, self._int_gen, self.device_cfg.device, out_buffer
            )
            self._index_buffer = index
        else:
            # Generate samples directly from sequencer
            samples = torch.tensor(
                self.sequencer.random(num_samples),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            ).contiguous()
        return samples

    @profiler.record_function("generator/samples")
    def get_samples(self, num_samples: int, bounded: bool = False) -> torch.Tensor:
        """Get uniform samples, optionally bounded to specified range.

        Args:
            num_samples: Number of samples to generate
            bounded: Whether to apply bounds transformation

        Returns:
            Tensor of samples in [0,1]^ndims or [low_bounds, up_bounds]^ndims
        """
        samples = self._get_samples(num_samples)
        if bounded:
            samples = self.bound_samples(samples, self.range_b, self.low_bounds)
        return samples

    @profiler.record_function("generator/gaussian_samples")
    def get_gaussian_samples(self, num_samples: int, variance: float = 1.0) -> torch.Tensor:
        """Get Gaussian-distributed samples.

        Args:
            num_samples: Number of samples to generate
            variance: Variance of the Gaussian distribution

        Returns:
            Tensor of Gaussian-distributed samples
        """
        std_dev = np.sqrt(variance)
        uniform_samples = self.get_samples(num_samples, False)
        gaussian_samples = self.gaussian_transform(
            uniform_samples, self.proj_mat, self.i_mat, std_dev
        )
        return gaussian_samples

    @staticmethod
    @get_torch_jit_decorator()
    def bound_samples(samples: torch.Tensor, range_b: torch.Tensor, low_bounds: torch.Tensor):
        """Transform uniform [0,1] samples to bounded range.

        Args:
            samples: Uniform samples in [0,1]
            range_b: Range (up_bounds - low_bounds)
            low_bounds: Lower bounds

        Returns:
            Samples transformed to [low_bounds, up_bounds]
        """
        samples = samples * range_b + low_bounds
        return samples

    @staticmethod
    @get_torch_jit_decorator(dynamic=True)
    def gaussian_transform(
        uniform_samples: torch.Tensor, proj_mat: torch.Tensor, i_mat: torch.Tensor, std_dev: float
    ):
        """Transform uniform samples to Gaussian distribution.

        Args:
            uniform_samples: Uniform samples in the range [0,1]
            proj_mat: Projection matrix for scaling
            i_mat: Identity matrix for dimension scaling
            std_dev: Standard deviation of target Gaussian distribution

        Returns:
            Gaussian-distributed samples
        """
        # Scale input to avoid inf values from erfinv at boundaries
        # erfinv returns inf when value is -1 or +1, so we scale to avoid these
        changed_samples = 1.98 * uniform_samples - 0.99

        # Apply inverse error function to get Gaussian samples
        gaussian_samples = proj_mat * torch.erfinv(changed_samples)
        i_mat = i_mat * std_dev
        gaussian_samples = torch.matmul(gaussian_samples, i_mat)
        return gaussian_samples

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, only_valid_for_compile=True)
    def sample_by_random_index(
        sample_buffer: torch.Tensor,
        num_samples: int,
        int_generator: torch.Generator,
        device: torch.device,
        out_buffer: Optional[torch.Tensor],
    ):
        """Sample from buffer using random indices.

        Args:
            sample_buffer: Pre-generated sample buffer
            num_samples: Number of samples to extract
            int_generator: PyTorch random number generator for indices
            device: Target device for tensors
            out_buffer: Optional output buffer for indices

        Returns:
            Tuple of (samples, indices)
        """
        index = torch.randint(
            0,
            sample_buffer.shape[0],
            (num_samples,),
            generator=int_generator,
            device=device,
            out=out_buffer,
        )

        samples = sample_buffer[index]
        return samples, index

    @classmethod
    def create_halton_sample_buffer(
        cls,
        ndims: int,
        up_bounds: List[float],
        low_bounds: List[float],
        store_buffer: Optional[int] = 2000,
        seed: int = 123,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        """Create a Halton sample buffer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Seed for the random number generator
            store_buffer: Size of sample buffer for performance (None to disable)
        """
        from .sequencer_halton import HaltonSequencer

        sequencer = HaltonSequencer(ndims=ndims, seed=seed)
        return cls(
            sequencer=sequencer,
            ndims=ndims,
            store_buffer=store_buffer,
            device_cfg=device_cfg,
            up_bounds=up_bounds,
            low_bounds=low_bounds,
        )

    @classmethod
    def create_random_sample_buffer(
        cls,
        ndims: int,
        up_bounds: List[float],
        low_bounds: List[float],
        store_buffer: Optional[int] = 2000,
        seed: int = 123,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        """Create a random sample buffer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Seed for the random number generator
            store_buffer: Size of sample buffer for performance (None to disable)
        """
        from .sequencer_random import RandomSequencer

        sequencer = RandomSequencer(ndims=ndims, seed=seed)
        return cls(
            sequencer=sequencer,
            ndims=ndims,
            store_buffer=store_buffer,
            device_cfg=device_cfg,
            up_bounds=up_bounds,
            low_bounds=low_bounds,
        )

    @classmethod
    def create_roberts_sample_buffer(
        cls,
        ndims: int,
        up_bounds: List[float],
        low_bounds: List[float],
        store_buffer: Optional[int] = 2000,
        seed: int = 123,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        """Create a Roberts sample buffer.

        Args:
            ndims: Number of dimensions for sample generation
            seed: Seed for the random number generator
            store_buffer: Size of sample buffer for performance (None to disable)
        """
        from .sequencer_roberts import RobertsSequencer

        sequencer = RobertsSequencer(ndims=ndims, seed=seed)
        return cls(
            sequencer=sequencer,
            ndims=ndims,
            store_buffer=store_buffer,
            device_cfg=device_cfg,
            up_bounds=up_bounds,
            low_bounds=low_bounds,
        )
