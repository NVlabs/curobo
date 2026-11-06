# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid frontier and global sampling for the bounded bidirectional RRT planner."""

from __future__ import annotations

# Standard Library
from typing import Callable, Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.graph_planner.rrt.kernel_rrt_runtime import (
    create_rrt_frontier_kernel,
    sample_uniform_rrt,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.warp import get_warp_device_stream

RRT_SAMPLE_BUFFER_SIZE = 2000


class RRTSampler:
    """Own one hybrid proposal stream and its warmup sampling state."""

    def __init__(
        self,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        sample_rejection_ratio: int,
        seed: int,
        device_cfg: DeviceCfg,
        check_feasibility_fn: Callable[[torch.Tensor], torch.Tensor],
        distance_weight: torch.Tensor,
    ) -> None:
        """Create a bounded Halton stream over the robot action limits."""
        wp.init()
        self.action_dim = action_lows.numel()
        self.sample_rejection_ratio = sample_rejection_ratio
        self.check_feasibility_fn = check_feasibility_fn
        self.distance_weight = distance_weight
        self._proposal_output = torch.empty((0,), **device_cfg.as_torch_dict())
        self._seed = seed
        self._sample_offset = 0
        self._sample_output = torch.empty((0, self.action_dim), **device_cfg.as_torch_dict())
        self.action_sample_generator = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            device_cfg=device_cfg,
            up_bounds=action_highs,
            low_bounds=action_lows,
            seed=seed,
            store_buffer=RRT_SAMPLE_BUFFER_SIZE,
            cache_bounded_samples=True,
        )

    def generate_action_samples(self, n_samples: int, bounded: bool = True) -> torch.Tensor:
        """Draw global proposals for warmup and the hybrid sampler.

        Returned storage is reused by subsequent calls.

        Preserve zero-copy views during the first cached pass. Subsequent passes
        rotate the unit-cube design rather than revisiting a finite sample set.
        The global row counter makes the stream independent of request sizes.
        """
        if n_samples < 0:
            log_and_raise("n_samples must be non-negative")
        buffer = self.action_sample_generator
        end = self._sample_offset + n_samples
        if end <= buffer._sample_buffer.shape[0]:
            values = buffer._bounded_sample_buffer if bounded else buffer._sample_buffer
            output = values[self._sample_offset : end]
        else:
            if self._sample_output.shape[0] < n_samples:
                self._sample_output = torch.empty(
                    (max(n_samples, buffer._sample_buffer.shape[0]), self.action_dim),
                    device=buffer.low_bounds.device,
                    dtype=buffer.low_bounds.dtype,
                )
            output = self._sample_output[:n_samples]
            if n_samples:
                device, stream = get_warp_device_stream(output.device)
                wp.launch(
                    sample_uniform_rrt,
                    dim=tuple(output.shape),
                    inputs=[
                        wp.from_torch(buffer._sample_buffer),
                        wp.from_torch(buffer.low_bounds),
                        wp.from_torch(buffer.up_bounds),
                        wp.from_torch(output),
                        wp.uint64(self._sample_offset),
                        self._seed,
                        int(bounded),
                    ],
                    device=device,
                    stream=stream,
                )
        self._sample_offset = end
        return output

    def generate_proposals(
        self,
        batch_size: int,
        candidates: int,
        active: Optional[torch.Tensor] = None,
        passive: Optional[torch.Tensor] = None,
        scale: float = 0.7,
    ) -> torch.Tensor:
        """Return hybrid proposals with shape ``[batch, candidates, dof]``.

        The first growth step has no endpoints and uses the global stream alone.
        Later steps guide two thirds of proposals around the four closest opposing
        endpoint pairs. The remaining rows retain global exploration. Endpoints
        come from the previous growth step; sampling adds no collision rollouts.
        Returned storage is reused, and the cached Halton design stays unchanged.
        """
        if batch_size < 1 or candidates < 1:
            log_and_raise("RRT proposal batch and candidate counts must be positive")
        if (active is None) != (passive is None):
            log_and_raise("RRT frontier endpoints must be supplied together")
        shape = (batch_size, candidates, self.action_dim)
        if active is not None and (active.shape != shape or passive.shape != shape):
            log_and_raise("RRT frontier endpoints must match the proposal shape")
        offset = self._sample_offset
        uniform = self.generate_action_samples(batch_size * candidates).reshape(shape)
        if active is None:
            return uniform
        if self._proposal_output.shape != uniform.shape:
            self._proposal_output = torch.empty_like(uniform)
        buffer = self.action_sample_generator
        device, stream = get_warp_device_stream(uniform)
        wp.launch(
            create_rrt_frontier_kernel(self.action_dim, candidates),
            dim=batch_size * candidates,
            inputs=[
                wp.from_torch(uniform.reshape(-1, self.action_dim)),
                wp.from_torch(active.reshape(-1, self.action_dim)),
                wp.from_torch(passive.reshape(-1, self.action_dim)),
                wp.from_torch(buffer.low_bounds),
                wp.from_torch(buffer.up_bounds),
                wp.from_torch(self.distance_weight),
                wp.from_torch(self._proposal_output.reshape(-1, self.action_dim)),
                wp.uint64(offset),
                self._seed,
                2 * candidates // 3,
                scale,
            ],
            device=device,
            stream=stream,
        )
        return self._proposal_output

    def generate_feasible_action_samples(self, num_samples: int) -> torch.Tensor:
        """Generate and filter the warmup samples needed by the planner."""
        candidate_count = num_samples + int(num_samples * self.sample_rejection_ratio)
        candidates = self.generate_action_samples(candidate_count, bounded=True)
        feasible = candidates[self.check_feasibility_fn(candidates).reshape(-1)]
        return feasible[:num_samples]

    def reset_seed(self) -> None:
        """Reset only the RRT Halton stream."""
        self.action_sample_generator.reset()
        self._sample_offset = 0
