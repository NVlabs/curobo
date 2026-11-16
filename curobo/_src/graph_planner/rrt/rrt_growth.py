# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Execute one packed bidirectional-RRT growth step.

Each step performs one active extension followed by one bounded passive
extension. It is the one-step-connect variant selected for the canonical GPU
planner, not a host loop that repeatedly extends the passive tree until trapped.
The shared executor uses CUDA-graph replay when enabled and direct execution
otherwise.
"""

from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.graph_planner.rrt.kernel_rrt_runtime import (
    create_rrt_runtime_kernels,
)
from curobo._src.graph_planner.rrt.rrt_tree_kernel import (
    NearestRRTTreeQueryBuffers,
    PackedRRTTreeAppendBuffers,
    RRTTreeKernelRunner,
)
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.util.cuda_graph_util import create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream

RRT_GROWTH_PROFILE_PHASES = (
    "off",
    "step_total",
    "env_expand",
    "active_stage",
    "active_nearest",
    "active_steering",
    "active_feasibility",
    "active_reduce",
    "passive_extension",
    "passive_nearest",
    "passive_steering",
    "passive_feasibility",
    "passive_reduce",
    "append",
    "status",
)


class RRTGrowthStep:
    """Execute nearest, exact steering, generic rollout, and append for ``B`` queries."""

    def __init__(
        self,
        *,
        tree_kernel_runner: RRTTreeKernelRunner,
        active_nearest: NearestRRTTreeQueryBuffers,
        passive_nearest: NearestRRTTreeQueryBuffers,
        active_append: PackedRRTTreeAppendBuffers,
        passive_append: PackedRRTTreeAppendBuffers,
        start_counts: torch.Tensor,
        goal_counts: torch.Tensor,
        connection_indices: torch.Tensor,
        start_is_active: bool,
        query_active: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
        distance_weight: torch.Tensor,
        max_extension_distance: float,
        cspace_resolution: float,
        max_steps: int,
        similarity_threshold: float,
        rollout: RobotRollout,
        profile_cuda_event_phase: str = "off",
    ) -> None:
        """Retain fixed-address batched buffers and a dedicated rollout."""
        self.tree_kernel_runner = tree_kernel_runner
        self.active_nearest = active_nearest
        self.passive_nearest = passive_nearest
        self.active_append = active_append
        self.passive_append = passive_append
        self.start_counts = start_counts
        self.goal_counts = goal_counts
        self.connection_indices = connection_indices
        if not isinstance(start_is_active, bool):
            log_and_raise("start_is_active must be a boolean")
        self.start_is_active = start_is_active
        self.query_active = query_active
        self.env_query_idx = env_query_idx
        self.goal_query_idx = goal_query_idx
        self.rollout = rollout
        if profile_cuda_event_phase not in RRT_GROWTH_PROFILE_PHASES:
            log_and_raise(
                f"profile_cuda_event_phase must be one of {list(RRT_GROWTH_PROFILE_PHASES)}"
            )
        self.profile_cuda_event_phase = profile_cuda_event_phase
        self.device = active_nearest.device
        self.batch_size, self.sample_count, self.action_dim = active_nearest.desired.shape
        self.max_steps = max_steps
        expected_count_shape = (self.batch_size,)
        for name, count in (("start_counts", start_counts), ("goal_counts", goal_counts)):
            if (
                count.shape != expected_count_shape
                or count.dtype != torch.int32
                or count.device != self.device
            ):
                log_and_raise(f"{name} must be CUDA int32 with shape (batch,)")
        if (
            query_active.shape != expected_count_shape
            or query_active.dtype != torch.bool
            or query_active.device != self.device
        ):
            log_and_raise("query_active must be CUDA bool with shape (batch,)")
        if (
            connection_indices.shape != (self.batch_size, 2)
            or connection_indices.dtype != torch.int64
            or connection_indices.device != self.device
        ):
            log_and_raise("connection_indices must be CUDA int64 with shape (batch, 2)")
        if (
            env_query_idx.shape != expected_count_shape
            or env_query_idx.dtype != torch.int32
            or env_query_idx.device != self.device
        ):
            log_and_raise("env_query_idx must be CUDA int32 with shape (batch,)")
        if (
            goal_query_idx.shape != expected_count_shape
            or goal_query_idx.dtype != torch.int32
            or goal_query_idx.device != self.device
        ):
            log_and_raise("goal_query_idx must be CUDA int32 with shape (batch,)")
        if passive_nearest.desired.shape != active_nearest.desired.shape:
            log_and_raise("active and passive batched proposals must match")
        if active_append.candidates.shape != active_nearest.desired.shape:
            log_and_raise("active append candidates must match proposals")
        if passive_append.candidates.shape != passive_nearest.desired.shape:
            log_and_raise("passive append candidates must match proposals")
        for name, nearest, append in (
            ("active", active_nearest, active_append),
            ("passive", passive_nearest, passive_append),
        ):
            if (
                nearest.profile != tree_kernel_runner.profile
                or append.profile != tree_kernel_runner.profile
            ):
                log_and_raise(f"{name} bindings must use the RRT tree-kernel profile")
            if (
                nearest.tree_nodes.data_ptr() != append.tree_nodes.data_ptr()
                or nearest.tree_counts.data_ptr() != append.tree_counts.data_ptr()
            ):
                log_and_raise(f"{name} nearest and append bindings must share tree storage")
            if nearest.parent_indices.data_ptr() != append.candidate_parents.data_ptr():
                log_and_raise(f"{name} nearest parents must alias append parents")

        self._edge_count = self.batch_size * self.sample_count
        self._distance_weight = distance_weight
        self._max_extension_distance = max_extension_distance
        self._cspace_resolution = cspace_resolution
        self._similarity_threshold = similarity_threshold
        kernels = create_rrt_runtime_kernels(
            self.action_dim,
            self.sample_count,
            self.max_steps,
            2,
        )
        self._steer_edges_kernel = kernels["steer_edges"]
        self._evaluate_edges_kernel = kernels["evaluate_edges"]
        self._select_connection_kernel = kernels["select_connection"]
        self._stage_buffers = {}
        for stage, append in (
            ("active", self.active_append),
            ("passive", self.passive_append),
        ):
            self._stage_buffers[stage] = {
                "targets": torch.empty(
                    (self._edge_count, self.action_dim),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "line_points": torch.empty(
                    (self._edge_count * self.max_steps, self.action_dim),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "line_valid": torch.empty(
                    self._edge_count * self.max_steps,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "endpoints": append.candidates.reshape(self._edge_count, self.action_dim),
                "reached": torch.empty(
                    self._edge_count,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "progressed": append.candidate_valid.reshape(self._edge_count),
            }
        if self._stage_buffers["active"]["endpoints"].data_ptr() != (
            self.passive_nearest.desired.data_ptr()
        ):
            log_and_raise("active endpoints must alias passive nearest targets")
        self._status = torch.empty(
            (self.batch_size, 5),
            dtype=torch.int64,
            device=self.device,
        )
        self._status_host = torch.empty(
            (self.batch_size, 5),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._rollout_env_query_idx = torch.empty(
            self.batch_size * self.sample_count * self.max_steps,
            dtype=torch.int32,
            device=self.device,
        )
        self._rollout_goal_query_idx = torch.empty_like(self._rollout_env_query_idx)
        self.rollout.update_batch_size(self._rollout_env_query_idx.shape[0])

        self._profile_start_event: Optional[torch.cuda.Event] = None
        self._profile_end_event: Optional[torch.cuda.Event] = None
        self._last_cuda_event_profile: Optional[dict[str, float]] = None
        if self.profile_cuda_event_phase != "off":
            # ``external=True`` is required for elapsed-time queries on event
            # record nodes captured by Torch around Torch and Warp launches.
            self._profile_start_event = torch.cuda.Event(enable_timing=True, external=True)
            self._profile_end_event = torch.cuda.Event(enable_timing=True, external=True)
        self._executor = create_graph_executor(
            capture_fn=self._growth_step,
            device=self.device,
            use_cuda_graph=True,
            clone_inputs=False,
            clone_outputs=False,
        )

    @property
    def is_initialized(self) -> bool:
        """Return whether this orientation has an initialized executor."""
        return self._executor.is_initialized

    @property
    def last_cuda_event_profile(self) -> Optional[dict[str, float]]:
        """Return the selected CUDA event phase duration from the last execution."""
        if self._last_cuda_event_profile is None:
            return None
        return dict(self._last_cuda_event_profile)

    def _profile_begin(self, phase: str) -> None:
        if self.profile_cuda_event_phase == phase:
            if self._profile_start_event is None:
                log_and_raise("CUDA event profiler is missing its start event")
            self._profile_start_event.record()

    def _profile_end(self, phase: str) -> None:
        if self.profile_cuda_event_phase == phase:
            if self._profile_end_event is None:
                log_and_raise("CUDA event profiler is missing its end event")
            self._profile_end_event.record()

    def _rollout_edges(
        self,
        starts: torch.Tensor,
        desired: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return endpoint, reached, and progressed tensors for one steering chunk."""
        starts_flat = starts.reshape(self._edge_count, self.action_dim)
        desired_flat = desired.reshape(self._edge_count, self.action_dim)
        buffers = self._stage_buffers[stage]
        self._profile_begin(f"{stage}_steering")
        wp.launch_tiled(
            self._steer_edges_kernel,
            dim=(self._edge_count,),
            inputs=[
                wp.from_torch(starts_flat, dtype=wp.float32),
                wp.from_torch(desired_flat, dtype=wp.float32),
                wp.from_torch(self._distance_weight, dtype=wp.float32),
                wp.from_torch(buffers["targets"], dtype=wp.float32),
                wp.from_torch(buffers["line_points"], dtype=wp.float32),
                wp.from_torch(buffers["line_valid"], dtype=wp.bool),
                self._max_extension_distance,
                self._cspace_resolution,
            ],
            block_dim=self.tree_kernel_runner.profile.block_dim,
            stream=get_warp_device_stream(starts)[1],
        )
        self._profile_end(f"{stage}_steering")
        line_points = buffers["line_points"]
        line_valid = buffers["line_valid"]
        actions = line_points.reshape(-1, 1, self.action_dim)
        self._profile_begin(f"{stage}_feasibility")
        metrics = self.rollout._compute_metrics_from_action_impl(
            actions,
            idxs_env_query=self._rollout_env_query_idx,
        )
        feasible = metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        ).reshape(-1)
        self._profile_end(f"{stage}_feasibility")
        self._profile_begin(f"{stage}_reduce")
        # The generic rollout has already classified every state. Supply
        # address-valid views for the raw native-runtime inputs, which the
        # shared evaluator skips in precomputed-feasibility mode.
        unused_state_values = wp.from_torch(line_points.reshape(-1), dtype=wp.float32)
        unused_joint_values = wp.from_torch(self._distance_weight, dtype=wp.float32)
        wp.launch_tiled(
            self._evaluate_edges_kernel,
            dim=(self._edge_count,),
            inputs=[
                wp.from_torch(starts_flat, dtype=wp.float32),
                wp.from_torch(desired_flat, dtype=wp.float32),
                wp.from_torch(buffers["targets"], dtype=wp.float32),
                wp.from_torch(line_points, dtype=wp.float32),
                wp.from_torch(line_valid, dtype=wp.bool),
                wp.from_torch(feasible, dtype=wp.bool),
                unused_state_values,
                unused_state_values,
                unused_state_values,
                unused_joint_values,
                unused_joint_values,
                unused_joint_values,
                unused_joint_values,
                wp.from_torch(self.query_active, dtype=wp.bool),
                self.sample_count,
                1,
                self._similarity_threshold,
                0,
                1,
            ],
            outputs=[
                wp.from_torch(buffers["endpoints"], dtype=wp.float32),
                wp.from_torch(buffers["reached"], dtype=wp.bool),
                wp.from_torch(buffers["progressed"], dtype=wp.bool),
            ],
            block_dim=self.tree_kernel_runner.profile.block_dim,
            stream=get_warp_device_stream(starts)[1],
        )
        self._profile_end(f"{stage}_reduce")
        return (
            buffers["endpoints"].reshape(
                self.batch_size,
                self.sample_count,
                self.action_dim,
            ),
            buffers["reached"].reshape(self.batch_size, self.sample_count),
            buffers["progressed"].reshape(self.batch_size, self.sample_count),
        )

    def _rollout_stage(
        self,
        binding: NearestRRTTreeQueryBuffers,
        desired: torch.Tensor,
        stream: wp.Stream,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return parent, endpoint, reached, and progressed tensors for one tree side."""
        self._profile_begin(f"{stage}_nearest")
        starts, parents = self.tree_kernel_runner.batched_nearest_target(
            binding,
            stream=stream,
        )
        self._profile_end(f"{stage}_nearest")
        endpoints, reached, progressed = self._rollout_edges(starts, desired, stage)
        return parents, endpoints, reached, progressed

    def _passive_extension(
        self,
        desired: torch.Tensor,
        stream: wp.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one bounded passive extension toward the active endpoints."""
        self._profile_begin("passive_extension")
        if desired.data_ptr() != self.passive_nearest.desired.data_ptr():
            log_and_raise("passive extension target must use its aliased fixed buffer")
        passive = self._rollout_stage(
            self.passive_nearest,
            self.passive_nearest.desired,
            stream,
            "passive",
        )
        self._profile_end("passive_extension")
        return passive

    def _growth_step(
        self,
        samples: torch.Tensor,
        query_active: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Run one step against the executor's borrowed planner-owned inputs."""
        _, stream = get_warp_device_stream(samples)
        if stream is None:
            log_and_raise("RRT growth step requires a CUDA Warp stream")
        self._profile_begin("step_total")
        self._profile_begin("env_expand")
        self._rollout_env_query_idx.view(self.batch_size, -1).copy_(env_query_idx.view(-1, 1))
        self._rollout_goal_query_idx.view(self.batch_size, -1).copy_(goal_query_idx.view(-1, 1))
        if self.rollout.has_goal_pose_routes:
            self.rollout.route_goal_poses(self._rollout_goal_query_idx.view(-1, 1))
        self._profile_end("env_expand")
        self._profile_begin("active_stage")
        _, active_end, _, _ = self._rollout_stage(
            self.active_nearest,
            samples,
            stream,
            "active",
        )
        self._profile_end("active_stage")
        passive_parents, _, connected, _ = self._passive_extension(
            active_end,
            stream,
        )

        self._profile_begin("append")
        active_slots, active_written, _ = self.tree_kernel_runner.batched_packed_append(
            self.active_append,
            stream=stream,
        )
        passive_slots, passive_written, _ = self.tree_kernel_runner.batched_packed_append(
            self.passive_append,
            stream=stream,
        )
        self._profile_end("append")

        self._profile_begin("status")
        wp.launch(
            self._select_connection_kernel,
            dim=self.batch_size,
            inputs=[
                wp.from_torch(connected.reshape(-1), dtype=wp.bool),
                wp.from_torch(active_slots.reshape(-1), dtype=wp.int64),
                wp.from_torch(active_written.reshape(-1), dtype=wp.bool),
                wp.from_torch(passive_slots.reshape(-1), dtype=wp.int64),
                wp.from_torch(passive_written.reshape(-1), dtype=wp.bool),
                wp.from_torch(passive_parents.reshape(-1), dtype=wp.int64),
                wp.from_torch(self.start_counts, dtype=wp.int32),
                wp.from_torch(self.goal_counts, dtype=wp.int32),
                wp.from_torch(self.query_active, dtype=wp.bool),
                wp.from_torch(self.connection_indices.reshape(-1), dtype=wp.int64),
                int(self.start_is_active),
            ],
            outputs=[wp.from_torch(self._status, dtype=wp.int64)],
            block_dim=self.tree_kernel_runner.profile.block_dim,
            stream=stream,
        )
        self._profile_end("status")
        self._profile_end("step_total")
        return self._status

    def _snapshot(self) -> tuple[torch.Tensor, ...]:
        return (
            self.active_nearest.tree_nodes.clone(),
            self.active_append.tree_parents.clone(),
            self.active_nearest.tree_counts.clone(),
            self.passive_nearest.tree_nodes.clone(),
            self.passive_append.tree_parents.clone(),
            self.passive_nearest.tree_counts.clone(),
            self.connection_indices.clone(),
        )

    def _restore(self, snapshot: tuple[torch.Tensor, ...]) -> None:
        self.active_nearest.tree_nodes.copy_(snapshot[0])
        self.active_append.tree_parents.copy_(snapshot[1])
        self.active_nearest.tree_counts.copy_(snapshot[2])
        self.passive_nearest.tree_nodes.copy_(snapshot[3])
        self.passive_append.tree_parents.copy_(snapshot[4])
        self.passive_nearest.tree_counts.copy_(snapshot[5])
        self.connection_indices.copy_(snapshot[6])

    def _validate_inputs(
        self,
        samples: torch.Tensor,
        query_active: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> None:
        """Validate the fixed executor tensor ABI before initialization or execution."""
        expected_samples = self.active_nearest.desired
        if (
            samples.shape != expected_samples.shape
            or samples.dtype != expected_samples.dtype
            or samples.device != self.device
            or not samples.is_contiguous()
        ):
            log_and_raise(
                "samples must be contiguous and match the proposal shape, dtype, and device"
            )
        if (
            query_active.shape != self.query_active.shape
            or query_active.dtype != torch.bool
            or query_active.device != self.device
            or not query_active.is_contiguous()
        ):
            log_and_raise("query_active must be contiguous CUDA bool with shape (batch,)")
        if (
            env_query_idx.shape != self.env_query_idx.shape
            or env_query_idx.dtype != torch.int32
            or env_query_idx.device != self.device
            or not env_query_idx.is_contiguous()
        ):
            log_and_raise("env_query_idx must be contiguous CUDA int32 with shape (batch,)")
        if (
            goal_query_idx.shape != self.goal_query_idx.shape
            or goal_query_idx.dtype != torch.int32
            or goal_query_idx.device != self.device
            or not goal_query_idx.is_contiguous()
        ):
            log_and_raise("goal_query_idx must be contiguous CUDA int32 with shape (batch,)")

    def initialize(
        self,
        samples: torch.Tensor,
        query_active: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> None:
        """Initialize this orientation without changing logical tree storage."""
        if self.is_initialized:
            return
        self._validate_inputs(samples, query_active, env_query_idx, goal_query_idx)
        initial_active = query_active.clone()
        snapshot = self._snapshot()
        self.active_nearest.desired.copy_(samples)
        self.query_active.copy_(query_active)
        self.env_query_idx.copy_(env_query_idx)
        self.goal_query_idx.copy_(goal_query_idx)
        # Borrow these fixed planner-owned buffers instead of cloning them.
        # GraphExecutor owns Torch capture; _growth_step resolves its capture stream
        # through cuRobo's central Warp stream helper.
        self._executor.warmup(
            self.active_nearest.desired,
            self.query_active,
            self.env_query_idx,
            self.goal_query_idx,
        )
        self._restore(snapshot)
        self.query_active.copy_(initial_active)
        self._last_cuda_event_profile = None

    def execute(
        self,
        samples: torch.Tensor,
        query_active: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> list[list[int]]:
        """Execute one growth step and transfer one status matrix."""
        if not self.is_initialized:
            log_and_raise("RRT growth-step executor must be initialized before execution")
        self._validate_inputs(samples, query_active, env_query_idx, goal_query_idx)
        status = self._executor(
            samples,
            query_active,
            env_query_idx,
            goal_query_idx,
            clone_outputs=False,
        )
        if not isinstance(status, torch.Tensor):
            log_and_raise("RRT growth-step executor must return a status tensor")
        self._status_host.copy_(status, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        status_host = self._status_host.tolist()
        if self.profile_cuda_event_phase != "off":
            if self._profile_start_event is None or self._profile_end_event is None:
                log_and_raise("CUDA event profiler is missing captured events")
            # The blocking status transfer above has already completed the
            # growth step, so elapsed_time adds no synchronization here.
            self._last_cuda_event_profile = {
                self.profile_cuda_event_phase: (
                    self._profile_start_event.elapsed_time(self._profile_end_event) / 1000.0
                )
            }
        return status_host

    def reset(self) -> None:
        """Release executor state; the next request may initialize it again."""
        self._executor.reset()
        self._last_cuda_event_profile = None
