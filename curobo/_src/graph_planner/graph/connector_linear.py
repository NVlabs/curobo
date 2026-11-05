# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Validate straight-line connections in robot configuration space."""

from __future__ import annotations

# Standard Library
import math
from dataclasses import dataclass
from typing import Optional, Protocol

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class LinearConnectorConfig(Protocol):
    """Configuration surface required by :class:`LinearConnector`."""

    device_cfg: DeviceCfg
    cspace_similarity_threshold: float
    steer_buffer_size: int


@dataclass
class LinearConnectionResult:
    """Result of validating one batch of configuration-space line segments."""

    #: Last feasible configuration on each line, shape ``(batch, action_dim)``.
    endpoint: torch.Tensor

    #: True when every sampled configuration through the exact goal was feasible.
    reached: torch.Tensor

    #: True when the endpoint moved farther than the configured similarity threshold.
    progressed: torch.Tensor


class LinearConnector:
    """Responsible for generating linear connections between configurations.

    This class handles the creation of straight-line paths in configuration space,
    checking for collisions along these paths, and determining the furthest
    feasible point that can be reached.
    """

    def __init__(self, config: LinearConnectorConfig, device_cfg: Optional[DeviceCfg] = None):
        """Initialize the LinearConnector.

        Args:
            config: Configuration for the graph planner
            device_cfg: Tensor device and type arguments. If None, uses config.device_cfg
        """
        self.config = config
        self.device_cfg = config.device_cfg if device_cfg is None else device_cfg
        self.cspace_similarity_threshold = config.cspace_similarity_threshold

        # Will be set through dependency injection
        self.action_dim = None
        self.cspace_distance_weight = None
        self._check_feasibility_fn = None
        self._preallocated_steer_buffer = torch.arange(
            0,
            self.config.steer_buffer_size,
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self._preallocated_idx_buffer = torch.arange(
            0,
            self.config.steer_buffer_size,
            device=self.device_cfg.device,
            dtype=torch.int64,
        )

    def set_dependencies(
        self,
        action_dim: int,
        cspace_distance_weight: torch.Tensor,
        check_feasibility_fn,
        preallocated_idx_buffer: Optional[torch.Tensor] = None,
    ):
        """Set required dependencies from parent class.

        Args:
            action_dim: Dimensionality of the action space
            cspace_distance_weight: Weight vector for distance calculations in C-space
            check_feasibility_fn: Function to check feasibility of configurations
            preallocated_idx_buffer: Shared index buffer for linear steering reductions
        """
        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self._check_feasibility_fn = check_feasibility_fn
        if preallocated_idx_buffer is not None:
            self._preallocated_idx_buffer = preallocated_idx_buffer

    # @profiler.record_function("linear_connector/connect")
    def connect(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
        cspace_resolution: Optional[float] = None,
    ) -> LinearConnectionResult:
        """Validate linear configuration-space paths through their exact goals.

        Interpolation uses ``cspace_similarity_threshold`` unless an explicit
        resolution is supplied. The returned endpoint is the sample immediately
        before the first infeasible configuration, or the exact goal when the
        complete line is feasible.

        Args:
            start: Start configurations with shape ``(batch, action_dim)``.
            goal: Goal configurations with shape ``(batch, action_dim)``.
            env_query_idx: Optional int32 environment index for every edge.
            goal_query_idx: Optional int32 tool-pose goal row for every edge.
            cspace_resolution: Optional interpolation resolution. When omitted,
                preserve the connector's configured similarity threshold.

        Returns:
            Endpoint, complete-connection, and progress tensors for every edge.
        """
        self._validate_connection_inputs(
            start,
            goal,
            env_query_idx,
            goal_query_idx,
            cspace_resolution,
        )

        line_points = self._compute_steering_line_points(
            start,
            goal,
            cspace_resolution=cspace_resolution,
        )
        batch_size, point_count, _ = line_points.shape

        flat_points = line_points.reshape(batch_size * point_count, self.action_dim)
        expanded_env = (
            None if env_query_idx is None else env_query_idx.repeat_interleave(point_count)
        )
        expanded_goal = (
            None if goal_query_idx is None else goal_query_idx.repeat_interleave(point_count)
        )
        if expanded_env is None and expanded_goal is None:
            feasible = self._check_feasibility_fn(flat_points)
        else:
            route_kwargs = {}
            if expanded_env is not None:
                route_kwargs["env_query_idx"] = expanded_env
            if expanded_goal is not None:
                route_kwargs["goal_query_idx"] = expanded_goal
            feasible = self._check_feasibility_fn(flat_points, **route_kwargs)
        feasible = feasible.reshape(batch_size, point_count).to(dtype=torch.bool)

        endpoint, reached = self._find_last_feasible_point(line_points, feasible, point_count)
        weighted_progress = torch.max(
            torch.abs((endpoint - start) * self.cspace_distance_weight), dim=-1
        ).values
        progressed = weighted_progress > self.cspace_similarity_threshold
        return LinearConnectionResult(
            endpoint=endpoint,
            reached=reached,
            progressed=progressed,
        )

    def _validate_connection_inputs(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
        goal_query_idx: Optional[torch.Tensor],
        cspace_resolution: Optional[float],
    ) -> None:
        """Validate the configuration-only connection contract."""
        if self.action_dim is None or self.cspace_distance_weight is None:
            log_and_raise("LinearConnector dependencies must be set before connecting")
        if self._check_feasibility_fn is None:
            log_and_raise("LinearConnector feasibility dependency is not set")
        if start.ndim != 2 or start.shape[1] != self.action_dim:
            log_and_raise("start must have shape (batch_size, action_dim)")
        if goal.ndim != 2 or goal.shape[1] != self.action_dim:
            log_and_raise("goal must have shape (batch_size, action_dim)")
        if start.shape != goal.shape:
            log_and_raise("start and goal must have matching shapes")
        if start.device != goal.device or start.dtype != goal.dtype:
            log_and_raise("start and goal must share device and dtype")
        if env_query_idx is not None and env_query_idx.shape != (start.shape[0],):
            log_and_raise("env_query_idx must have one value per steering edge")
        if env_query_idx is not None:
            if env_query_idx.dtype != torch.int32:
                log_and_raise("env_query_idx must use torch.int32")
            if env_query_idx.device != start.device:
                log_and_raise("env_query_idx must share the configuration device")
        if goal_query_idx is not None and goal_query_idx.shape != (start.shape[0],):
            log_and_raise("goal_query_idx must have one value per steering edge")
        if goal_query_idx is not None:
            if goal_query_idx.dtype != torch.int32:
                log_and_raise("goal_query_idx must use torch.int32")
            if goal_query_idx.device != start.device:
                log_and_raise("goal_query_idx must share the configuration device")
        if cspace_resolution is not None and (
            not isinstance(cspace_resolution, (int, float))
            or isinstance(cspace_resolution, bool)
            or not math.isfinite(cspace_resolution)
            or cspace_resolution <= 0.0
        ):
            log_and_raise("cspace_resolution must be finite and positive")

    # @profiler.record_function("linear_connector/_compute_steering_line_points")
    def _compute_steering_line_points(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        cspace_resolution: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute points along configuration-space lines, including exact goals.

        Args:
            start: Start configurations with shape ``(batch, action_dim)``.
            goal: Goal configurations with shape ``(batch, action_dim)``.
            cspace_resolution: Optional interpolation resolution. When omitted,
                preserve the connector's configured similarity threshold.

        Returns:
            Interpolated points with shape ``(batch, points, action_dim)``.
        """
        direction = goal - start

        weighted_direction = direction * self.cspace_distance_weight
        max_component_change = torch.max(torch.abs(weighted_direction))
        resolution = (
            self.cspace_similarity_threshold if cspace_resolution is None else cspace_resolution
        )
        segment_count = max(1, int(torch.ceil(max_component_change / resolution).item()))
        point_count = segment_count + 1
        if point_count > self.config.steer_buffer_size:
            log_and_raise(
                "steer_buffer_size is too small for linear connection: "
                f"capacity={self.config.steer_buffer_size}, required={point_count}"
            )

        interpolation = self._preallocated_steer_buffer[:point_count] / segment_count
        return start.unsqueeze(1) + interpolation.view(1, point_count, 1) * direction.unsqueeze(1)

    # @profiler.record_function("linear_connector/_find_last_feasible_point")
    @get_torch_jit_decorator(only_valid_for_compile=True, dynamic=True, slow_to_compile=True)
    def _find_last_feasible_point(
        self,
        line_points: torch.Tensor,
        feasible: torch.Tensor,
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the last feasible point and whether the exact goal was reached.

        Args:
            line_points: Interpolated configurations with shape ``(batch, points, action_dim)``.
            feasible: Boolean feasibility with shape ``(batch, points)``.
            point_count: Number of points on every line.

        Returns:
            Last feasible configurations and complete-line success flags.
        """
        sample_idx = self._preallocated_idx_buffer[:point_count].view(1, point_count)
        first_infeasible_idx = (
            torch.where(
                ~feasible,
                sample_idx,
                point_count,
            )
            .min(dim=1)
            .values
        )

        idx = torch.where(
            first_infeasible_idx < point_count,
            first_infeasible_idx - 1,
            point_count - 1,
        )
        idx = torch.clamp(idx, min=0)

        batch_indices = torch.arange(line_points.shape[0], device=line_points.device)
        endpoint = line_points[batch_indices, idx]
        reached = first_infeasible_idx == point_count
        return endpoint, reached
