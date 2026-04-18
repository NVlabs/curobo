# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class LinearConnector:
    """Responsible for generating linear connections between configurations.

    This class handles the creation of straight-line paths in configuration space,
    checking for collisions along these paths, and determining the furthest
    feasible point that can be reached.
    """

    def __init__(self, config: PRMGraphPlannerCfg, device_cfg: Optional[DeviceCfg] = None):
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
            device=self.config.device_cfg.device,
            dtype=self.config.device_cfg.dtype,
        )
        self._node_idx_padding_buffer = torch.as_tensor(
            [0.0], device=self.config.device_cfg.device, dtype=self.config.device_cfg.dtype
        )

    def set_dependencies(
        self,
        action_dim: int,
        cspace_distance_weight: torch.Tensor,
        check_feasibility_fn,
    ):
        """Set required dependencies from parent class.

        Args:
            action_dim: Dimensionality of the action space
            cspace_distance_weight: Weight vector for distance calculations in C-space
            check_feasibility_fn: Function to check feasibility of configurations
        """
        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self._check_feasibility_fn = check_feasibility_fn

    # @profiler.record_function("linear_connector/steer_until_infeasible")
    def steer_until_infeasible(
        self,
        start_nodes: torch.Tensor,
        desired_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation from start nodes towards desired nodes until not feasible.

        The interpolation is done at a resolution of cspace_similarity_threshold. Returns
        the last feasible node along the path from start to desired node.

        Args:
            start_nodes: (B, DOF+1) Tensor of start configurations
            desired_nodes: (B, DOF+1) Tensor of goal configurations

        Returns:
            feasible_nodes: (B, DOF+1) The last feasible configuration along each path
        """
        if start_nodes.shape[0] != desired_nodes.shape[0]:
            log_and_raise("start_nodes and desired_nodes must have the same batch size")
        if start_nodes.shape[1] != self.action_dim + 1:
            log_and_raise("start_nodes must have the same number of dimensions as the action space")
        if desired_nodes.shape[1] != self.action_dim + 1:
            log_and_raise("desired_nodes must have the same number of dimensions as the action space")

        line_vec = self._compute_steering_line_points(start_nodes, desired_nodes)
        b, h, _ = line_vec.shape

        line_vec = line_vec.view(b * h, self.action_dim)
        mask = self._check_feasibility_fn(line_vec)
        line_vec = line_vec.view(b, h, self.action_dim)
        mask = mask.view(b, h).to(dtype=torch.int8)

        steered_feasible_nodes = self._find_last_feasible_point(line_vec, mask, h)
        return steered_feasible_nodes

    # @profiler.record_function("linear_connector/_compute_steering_line_points")
    def _compute_steering_line_points(
        self,
        start_nodes: torch.Tensor,
        desired_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute points along a line from start_nodes to desired_nodes.

        Args:
            start_nodes: (B, DOF+1) Tensor of start configurations
            desired_nodes: (B, DOF+1) Tensor of goal configurations

        Returns:
            line_vec: (B, H, DOF) Interpolated points along the line
        """
        # Extract action vectors and calculate difference vector
        start_actions = start_nodes[..., : self.action_dim]
        desireaction_dims = desired_nodes[..., : self.action_dim]

        # Calculate direction vector
        direction_vec = desireaction_dims - start_actions

        # Apply distance weights for accurate spacing calculation
        weighted_direction = direction_vec * self.cspace_distance_weight

        # Calculate required steps based on maximum component change
        max_component_change = torch.max(torch.abs(weighted_direction), dim=1)[0]
        num_steps = torch.ceil(max_component_change / self.config.cspace_similarity_threshold) + 1
        max_steps = torch.max(num_steps).to(dtype=torch.int64)

        # Create interpolation coefficients (0 to 1)
        steps = self._preallocated_steer_buffer[: max_steps + 1]
        interp_coeffs = steps / max_steps

        # Generate all points in a single vectorized operation
        line_vec = start_actions.unsqueeze(1) + interp_coeffs.unsqueeze(1).unsqueeze(
            0
        ) * direction_vec.unsqueeze(1)

        return line_vec

    # @profiler.record_function("linear_connector/_find_last_feasible_point")
    @get_torch_jit_decorator(only_valid_for_compile=True, dynamic=True, slow_to_compile=True)
    def _find_last_feasible_point(
        self,
        line_vec: torch.Tensor,
        mask: torch.Tensor,
        h: int,
    ) -> torch.Tensor:
        """Find the last feasible point along the interpolated line.

        Args:
            line_vec: (B, H, DOF) Interpolated points along the line
            mask: (B, H) Feasibility mask for each point
            h: Number of interpolation steps

        Returns:
            last_feasible_nodes: (B, DOF+1) The last feasible configuration along each path
        """
        # Pre-compute buffer values once
        step_weights = self._preallocated_steer_buffer[:h] / h + 1.0

        # Convert mask to proper format in fewer operations
        mask = torch.where(
            mask == 0.0,
            -step_weights,  # If mask is 0, use negative values
            step_weights,  # If mask is 1, use positive values
        )

        # Invert negative values in a single operation
        mask = torch.where(mask < 0.0, 1.0 / mask.abs(), mask)

        # Find indices of first collision points
        _, idx = torch.min(mask, dim=1)

        # Adjust indices to get last feasible point before collision
        idx = torch.where(idx > 0, idx - 1, h - 1)

        # Gather the feasible points efficiently
        batch_indices = torch.arange(line_vec.shape[0], device=line_vec.device)
        steered_feasible_points = line_vec[batch_indices, idx]

        # Create node index padding buffer with exact size
        batch_size = steered_feasible_points.shape[0]
        extra_data = self._node_idx_padding_buffer.expand(batch_size, 1)

        # Concatenate in a single operation
        steered_feasible_nodes = torch.cat((steered_feasible_points, extra_data), dim=1)

        return steered_feasible_nodes
