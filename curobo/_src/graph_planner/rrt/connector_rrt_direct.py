# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity CUDA-graph connector for RRT terminal edges."""

from __future__ import annotations

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.graph_planner.rrt.kernel_rrt_runtime import (
    create_rrt_runtime_kernels,
)
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


class RRTDirectConnector:
    """Validate fixed-width paired RRT edges without a host-sized steering line."""

    _BLOCK_DIM = 128

    def __init__(
        self,
        rollout: RobotRollout,
        distance_weight: torch.Tensor,
        edge_count: int,
        max_steps: int,
        cspace_resolution: float,
        similarity_threshold: float,
    ) -> None:
        """Allocate graph-stable storage for ``edge_count`` complete edges."""
        if edge_count < 2 or edge_count % 2 != 0:
            log_and_raise("RRT direct connector requires a positive even edge count")
        if max_steps < 2:
            log_and_raise("RRT direct connector requires at least two steering steps")
        if distance_weight.ndim != 1 or distance_weight.dtype != torch.float32:
            log_and_raise("RRT direct connector distance weights must be one-dimensional float32")

        self.rollout = rollout
        self.device = distance_weight.device
        self.action_dim = distance_weight.shape[0]
        self.edge_count = edge_count
        self.plan_count = edge_count // 2
        self.max_steps = max_steps
        self.cspace_resolution = cspace_resolution
        self.similarity_threshold = similarity_threshold
        self.distance_weight = distance_weight

        self.request_starts = torch.empty(
            (self.plan_count, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.request_goals = torch.empty_like(self.request_starts)
        self.request_env_query_idx = torch.zeros(
            self.plan_count, device=self.device, dtype=torch.int32
        )
        self.request_goal_query_idx = torch.zeros_like(self.request_env_query_idx)
        self._request_env_is_zero = True
        self._request_goal_is_zero = True
        self.starts = torch.empty(
            (edge_count, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.desired = torch.empty_like(self.starts)
        self.plan_active = torch.ones(self.plan_count, device=self.device, dtype=torch.bool)
        self.env_query_idx = torch.zeros(edge_count, device=self.device, dtype=torch.int32)
        self.goal_query_idx = torch.zeros_like(self.env_query_idx)
        self.rollout_env_query_idx = torch.empty(
            edge_count * max_steps,
            device=self.device,
            dtype=torch.int32,
        )
        self.rollout_goal_query_idx = torch.empty_like(self.rollout_env_query_idx)
        self.point_count = torch.empty(1, device=self.device, dtype=torch.int32)
        self.line_points = torch.empty(
            (edge_count * max_steps, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.line_valid = torch.empty(
            edge_count * max_steps,
            device=self.device,
            dtype=torch.bool,
        )
        self.endpoints = torch.empty_like(self.starts)
        self.reached = torch.empty(edge_count, device=self.device, dtype=torch.bool)
        self.progressed = torch.empty_like(self.reached)
        self.terminal_feasible = torch.empty_like(self.reached)

        kernels = create_rrt_runtime_kernels(
            self.action_dim,
            1,
            2,
            max_steps,
        )
        self._pack_kernel = kernels["pack_direct_pairs"]
        self._prepare_kernel = kernels["prepare_direct_edges"]
        self._interpolate_kernel = kernels["interpolate_direct_edges"]
        self._reduce_kernel = kernels["reduce_direct_edges"]
        self._views = {
            "request_starts": wp.from_torch(self.request_starts, dtype=wp.float32),
            "request_goals": wp.from_torch(self.request_goals, dtype=wp.float32),
            "request_env_query_idx": wp.from_torch(self.request_env_query_idx, dtype=wp.int32),
            "request_goal_query_idx": wp.from_torch(self.request_goal_query_idx, dtype=wp.int32),
            "starts": wp.from_torch(self.starts, dtype=wp.float32),
            "desired": wp.from_torch(self.desired, dtype=wp.float32),
            "distance_weight": wp.from_torch(self.distance_weight, dtype=wp.float32),
            "plan_active": wp.from_torch(self.plan_active, dtype=wp.bool),
            "point_count": wp.from_torch(self.point_count, dtype=wp.int32),
            "line_points": wp.from_torch(self.line_points, dtype=wp.float32),
            "line_valid": wp.from_torch(self.line_valid, dtype=wp.bool),
            "endpoints": wp.from_torch(self.endpoints, dtype=wp.float32),
            "reached": wp.from_torch(self.reached, dtype=wp.bool),
            "progressed": wp.from_torch(self.progressed, dtype=wp.bool),
            "terminal_feasible": wp.from_torch(self.terminal_feasible, dtype=wp.bool),
        }
        self.rollout.update_batch_size(edge_count * max_steps)

    def _connect(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capture interpolation, generic feasibility, and edge reduction."""
        _, stream = get_warp_device_stream(self.device)
        if stream is None:
            log_and_raise("RRT direct connector requires a CUDA Warp stream")
        wp.launch(
            self._pack_kernel,
            dim=self.edge_count,
            inputs=[
                self._views["request_starts"],
                self._views["request_goals"],
                self._views["request_env_query_idx"],
                self._views["request_goal_query_idx"],
                self.plan_count,
            ],
            outputs=[
                self._views["starts"],
                self._views["desired"],
                wp.from_torch(self.env_query_idx, dtype=wp.int32),
                wp.from_torch(self.goal_query_idx, dtype=wp.int32),
            ],
            block_dim=self._BLOCK_DIM,
            stream=stream,
        )
        wp.launch(
            self._prepare_kernel,
            dim=1,
            inputs=[
                self._views["starts"],
                self._views["desired"],
                self._views["distance_weight"],
                self.edge_count,
                self.cspace_resolution,
            ],
            outputs=[self._views["point_count"]],
            block_dim=self._BLOCK_DIM,
            stream=stream,
        )
        wp.launch(
            self._interpolate_kernel,
            dim=(self.edge_count, self.max_steps),
            inputs=[
                self._views["starts"],
                self._views["desired"],
                self._views["point_count"],
            ],
            outputs=[
                self._views["line_points"],
                self._views["line_valid"],
            ],
            block_dim=self._BLOCK_DIM,
            stream=stream,
        )

        self.rollout_env_query_idx.view(self.edge_count, self.max_steps).copy_(
            self.env_query_idx.view(-1, 1)
        )
        self.rollout_goal_query_idx.view(self.edge_count, self.max_steps).copy_(
            self.goal_query_idx.view(-1, 1)
        )
        if self.rollout.has_goal_pose_routes:
            self.rollout.route_goal_poses(self.rollout_goal_query_idx.view(-1, 1))
        metrics = self.rollout._compute_metrics_from_action_impl(
            self.line_points.reshape(-1, 1, self.action_dim),
            idxs_env_query=self.rollout_env_query_idx,
        )
        feasible = metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        ).reshape(-1)
        wp.launch(
            self._reduce_kernel,
            dim=self.edge_count,
            inputs=[
                self._views["starts"],
                self._views["line_points"],
                self._views["line_valid"],
                wp.from_torch(feasible, dtype=wp.bool),
                self._views["distance_weight"],
                self._views["point_count"],
                self._views["plan_active"],
                self.plan_count,
                self.similarity_threshold,
            ],
            outputs=[
                self._views["endpoints"],
                self._views["reached"],
                self._views["progressed"],
                self._views["terminal_feasible"],
            ],
            block_dim=self._BLOCK_DIM,
            stream=stream,
        )
        return self.endpoints, self.reached, self.progressed, self.terminal_feasible

    def stage_request(
        self,
        starts: torch.Tensor,
        desired: torch.Tensor,
        env_query_idx: torch.Tensor | None = None,
        goal_query_idx: torch.Tensor | None = None,
    ) -> None:
        """Copy one paired request into graph-stable connector inputs."""
        if starts.shape != self.request_starts.shape or starts.dtype != torch.float32:
            log_and_raise("RRT direct starts do not match the connector profile")
        if desired.shape != self.request_goals.shape or desired.dtype != torch.float32:
            log_and_raise("RRT direct goals do not match the connector profile")
        if starts.device != self.device or desired.device != self.device:
            log_and_raise("RRT direct edges must use the connector device")
        for name, indices in (
            ("env_query_idx", env_query_idx),
            ("goal_query_idx", goal_query_idx),
        ):
            if indices is not None and (
                indices.shape != (self.plan_count,)
                or indices.dtype != torch.int32
                or indices.device != self.device
            ):
                log_and_raise(f"{name} must be CUDA int32 with one value per plan")

        self.request_starts.copy_(starts)
        self.request_goals.copy_(desired)
        if env_query_idx is None:
            if not self._request_env_is_zero:
                self.request_env_query_idx.zero_()
            self._request_env_is_zero = True
        else:
            self.request_env_query_idx.copy_(env_query_idx)
            self._request_env_is_zero = False
        if goal_query_idx is None:
            if not self._request_goal_is_zero:
                self.request_goal_query_idx.zero_()
            self._request_goal_is_zero = True
        else:
            self.request_goal_query_idx.copy_(goal_query_idx)
            self._request_goal_is_zero = False

    def evaluate_staged(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the currently staged request inside an owning CUDA graph."""
        return self._connect()

    def reset_cuda_graph(self) -> None:
        """Release rollout graphs while retaining fixed connector buffers."""
        self.rollout.reset_cuda_graph()
