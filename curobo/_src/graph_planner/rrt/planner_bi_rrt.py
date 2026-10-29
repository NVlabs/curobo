# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU bounded bidirectional RRT planner for cuRobo.

The planner composes focused RRT resources for collision, sampling, and
robot-model services. PRM roadmap construction, NetworkX search, and PRM
shortcutting are not created when this planner is selected.

The canonical implementation plans paired queries as independent fixed-capacity
lanes. Each growth-step executor combines generated Warp bookkeeping, exact
cuRobo rollout, and packed appends; masks, optional completion-owner IDs, and
environment IDs are request data rather than separate algorithms.

Each step performs one bounded extension from the active tree followed by one
bounded extension from the passive tree.
"""

from __future__ import annotations

# Standard Library
import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

# Third Party
import torch
import warp as wp
from torch.nn import functional as F

# CuRobo
from curobo._src import runtime
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.graph_planner.rrt.connector_rrt_direct import RRTDirectConnector
from curobo._src.graph_planner.rrt.kernel_rrt_runtime import create_rrt_runtime_kernels
from curobo._src.graph_planner.rrt.resources_rrt_planner import RRTPlannerResources
from curobo._src.graph_planner.rrt.rrt_growth import (
    RRT_GROWTH_PROFILE_PHASES,
    RRTGrowthStep,
)
from curobo._src.graph_planner.rrt.rrt_path_shortcut import (
    RRTPathShortcutter,
    RRTPathShortcutterConfig,
)
from curobo._src.graph_planner.rrt.rrt_projection_sample import (
    RRTSampleProjectionConfig,
    RRTSampleProjectionResult,
    RRTSampleProjector,
)
from curobo._src.graph_planner.rrt.rrt_sampling import InformedRRTProposalGenerator
from curobo._src.graph_planner.rrt.rrt_tree_kernel import (
    RRTPathReconstructionBuffers,
    RRTTreeKernelConfig,
    RRTTreeKernelRunner,
)
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.warp import get_warp_device_stream


@dataclass
class BiRRTConfig:
    """Configuration for the canonical bounded bidirectional RRT planner."""

    samples_per_iteration: int = 24
    max_iterations: int = 12
    failure_extension_iterations: int = 20
    max_nodes_per_tree: int = 1024
    max_batched_lanes: int = 128
    max_extension_distance: Optional[float] = None
    cspace_resolution: float = 0.02
    fixed_steer_steps: int = 64
    goal_bias_count: int = 0

    # Sampling remains request data for the same growth-step engine.
    sampling_policy: str = "uniform"
    informed_sample_count: int = 8
    informed_initial_scale: float = 1.5
    informed_growth_factor: float = 1.25
    informed_growth_interval: int = 4
    informed_max_scale: float = 3.0
    informed_min_slack: Optional[float] = None

    profile_cuda_events: bool = False
    profile_cuda_event_phase: str = "step_total"
    measure_shortcut_time: bool = False

    # The only shortcut implementation is the packed one-shot GPU kernel.
    shortcut_path: bool = True
    shortcut_path_capacity: int = 16
    shortcut_state_capacity: Optional[int] = None

    # Projection remains absent from the default growth path. When configured,
    # it repairs a bounded set of invalid ambient proposals before ordinary growth.
    sample_projection: Optional[RRTSampleProjectionConfig] = None

    def __post_init__(self) -> None:
        """Resolve derived shapes and reject unsupported deployment profiles."""
        if not math.isfinite(self.cspace_resolution) or self.cspace_resolution <= 0.0:
            log_and_raise("cspace_resolution must be finite and positive")
        if (
            not isinstance(self.fixed_steer_steps, int)
            or isinstance(self.fixed_steer_steps, bool)
            or self.fixed_steer_steps < 3
        ):
            log_and_raise("fixed_steer_steps must be an integer of at least 3")
        derived_extension = (self.fixed_steer_steps - 2) * self.cspace_resolution
        if self.max_extension_distance is None:
            self.max_extension_distance = derived_extension
        elif not math.isfinite(self.max_extension_distance) or self.max_extension_distance <= 0.0:
            log_and_raise("max_extension_distance must be finite and positive")
        elif self.max_extension_distance > derived_extension:
            log_and_raise(
                "max_extension_distance exceeds the safe fixed steering-buffer "
                f"coverage ({derived_extension})"
            )

        if (
            not isinstance(self.samples_per_iteration, int)
            or isinstance(self.samples_per_iteration, bool)
            or self.samples_per_iteration <= 0
            or not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations <= 0
        ):
            log_and_raise("RRT samples_per_iteration and max_iterations must be positive")
        if (
            not isinstance(self.failure_extension_iterations, int)
            or isinstance(self.failure_extension_iterations, bool)
            or self.failure_extension_iterations < 0
        ):
            log_and_raise("failure_extension_iterations must be non-negative")
        if self.max_batched_lanes <= 0 or self.max_batched_lanes & (self.max_batched_lanes - 1):
            log_and_raise("max_batched_lanes must be a positive power of two")
        required_nodes = 2 + self.samples_per_iteration * (
            self.max_iterations + self.failure_extension_iterations
        )
        if self.max_nodes_per_tree < required_nodes:
            log_and_raise(
                "RRT requires max_nodes_per_tree >= two direct/root slots + "
                "samples_per_iteration * total growth-step budget "
                f"({required_nodes})"
            )

        valid_sampling_policies = {"uniform", "mixed_ellipsoid"}
        if self.sampling_policy not in valid_sampling_policies:
            log_and_raise(f"sampling_policy must be one of {sorted(valid_sampling_policies)}")
        if (
            not isinstance(self.informed_sample_count, int)
            or isinstance(self.informed_sample_count, bool)
            or self.informed_sample_count < 1
        ):
            log_and_raise("informed_sample_count must be a positive integer")
        if (
            not isinstance(self.goal_bias_count, int)
            or isinstance(self.goal_bias_count, bool)
            or not 0 <= self.goal_bias_count <= self.samples_per_iteration
        ):
            log_and_raise("goal_bias_count must be an integer within the proposal batch")
        if (
            self.sampling_policy == "mixed_ellipsoid"
            and self.goal_bias_count + self.informed_sample_count > self.samples_per_iteration
        ):
            log_and_raise(
                "goal_bias_count + informed_sample_count cannot exceed samples_per_iteration"
            )
        for field_name, field_value in (
            ("informed_initial_scale", self.informed_initial_scale),
            ("informed_growth_factor", self.informed_growth_factor),
            ("informed_max_scale", self.informed_max_scale),
        ):
            if not math.isfinite(field_value) or field_value < 1.0:
                log_and_raise(f"{field_name} must be finite and at least 1.0")
        if self.informed_max_scale < self.informed_initial_scale:
            log_and_raise("informed_max_scale cannot be smaller than informed_initial_scale")
        if (
            not isinstance(self.informed_growth_interval, int)
            or isinstance(self.informed_growth_interval, bool)
            or self.informed_growth_interval < 1
        ):
            log_and_raise("informed_growth_interval must be a positive integer")
        if self.informed_min_slack is None:
            self.informed_min_slack = 2.0 * self.max_extension_distance
        elif not math.isfinite(self.informed_min_slack) or self.informed_min_slack < 0.0:
            log_and_raise("informed_min_slack must be finite and non-negative")

        if (
            not isinstance(self.shortcut_path_capacity, int)
            or isinstance(self.shortcut_path_capacity, bool)
            or self.shortcut_path_capacity < 3
        ):
            log_and_raise("shortcut_path_capacity must be an integer of at least 3")
        if self.shortcut_state_capacity is not None and (
            not isinstance(self.shortcut_state_capacity, int)
            or isinstance(self.shortcut_state_capacity, bool)
            or self.shortcut_state_capacity < 2
        ):
            log_and_raise("shortcut_state_capacity must be an integer of at least 2")

        if isinstance(self.sample_projection, dict):
            self.sample_projection = RRTSampleProjectionConfig(**self.sample_projection)
        elif self.sample_projection is not None and not isinstance(
            self.sample_projection, RRTSampleProjectionConfig
        ):
            log_and_raise("sample_projection must be an RRTSampleProjectionConfig or None")
        if not isinstance(self.profile_cuda_events, bool):
            log_and_raise("profile_cuda_events must be a boolean")
        if not isinstance(self.measure_shortcut_time, bool):
            log_and_raise("measure_shortcut_time must be a boolean")
        if self.profile_cuda_event_phase not in RRT_GROWTH_PROFILE_PHASES:
            log_and_raise(
                f"profile_cuda_event_phase must be one of {list(RRT_GROWTH_PROFILE_PHASES)}"
            )


@dataclass
class _RRTTree:
    nodes: torch.Tensor
    parents: torch.Tensor
    counts: torch.Tensor


@dataclass
class _RRTWorkspace:
    batch_size: int
    start_tree: _RRTTree
    goal_tree: _RRTTree
    growth_steps: List[RRTGrowthStep]
    query_active: torch.Tensor
    env_query_idx: torch.Tensor
    goal_query_idx: torch.Tensor
    path_reconstruction: RRTPathReconstructionBuffers
    path_reconstruction_executor: GraphExecutor
    path_shortcutter: Optional[RRTPathShortcutter]
    finalization_rollout: Optional[RobotRollout]
    finalization_executors: dict[tuple[bool, bool], GraphExecutor]
    path_counts_host: torch.Tensor
    shortcut_status_host: torch.Tensor
    direct_status: torch.Tensor
    direct_status_host: torch.Tensor
    request_query_mask: torch.Tensor
    request_query_mask_all_active: bool
    preprocessing_executors: dict[tuple[int, int], GraphExecutor]
    kernel_views: dict[str, object]


@dataclass
class _RRTRequest:
    """Normalized fixed-width inputs for one canonical lane invocation."""

    x_start: torch.Tensor
    x_goal: torch.Tensor
    query_mask: torch.Tensor
    env_query_idx: torch.Tensor
    goal_query_idx: torch.Tensor
    batch_size: int
    group_count: int
    explicit_env: bool
    has_goal_constraint: bool
    route_goals: bool
    has_requested_lanes: bool
    query_mask_host: List[bool]
    group_ids_host: List[int]


@dataclass
class _RRTConnectionRecord:
    """One observed start/goal-tree connection in a flattened lane."""

    lane: int
    iterations: int
    start_count: int
    goal_count: int
    start_index: Optional[int] = None
    goal_index: Optional[int] = None
    is_direct: bool = False
    path: Optional[torch.Tensor] = None
    raw_waypoints: int = 0


@dataclass
class _RRTSearchState:
    """Mutable host control state shared by the search phases."""

    group_ids_host: List[int]
    group_valid_host: List[bool]
    paths: List[Optional[torch.Tensor]]
    query_debug: List[dict]
    active_host: List[bool]
    group_solved: List[bool]
    connection_records: List[_RRTConnectionRecord]


class BiRRTPlanner:
    """Bounded bidirectional RRT with one passive extension per GPU growth step."""

    def __init__(
        self,
        resources: RRTPlannerResources,
        config: Optional[BiRRTConfig] = None,
    ) -> None:
        """Initialize the planner with its cuRobo runtime resources."""
        self.resources = resources
        self.config = config or BiRRTConfig()
        if resources.device_cfg.dtype != torch.float32:
            log_and_raise("BiRRT currently requires float32 planner tensors")

        self._shortcut_state_capacity = (
            resources.config.feasibility_buffer_size
            if self.config.shortcut_state_capacity is None
            else self.config.shortcut_state_capacity
        )
        if self._shortcut_state_capacity > resources.config.feasibility_buffer_size:
            log_and_raise(
                "shortcut_state_capacity cannot exceed the planner-resource feasibility buffer; "
                "one-shot shortcutting requires one collision rollout"
            )

        self._sample_projector: Optional[RRTSampleProjector] = None
        self._sample_projection_config: Optional[RRTSampleProjectionConfig] = None
        self._constraint_goal_tool_poses: Optional[GoalToolPose] = None
        self._constraint_tool_pose_criteria: Optional[Dict[str, ToolPoseCriteria]] = None
        self._constraint_tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = (
            None
        )
        self._constraint_tool_relative_transform_criteria: Optional[
            Dict[str, ToolRelativeTransformCriteria]
        ] = None
        self._informed_proposal_generator = None
        self._informed_scale_schedule = None
        self._informed_latent_schedule: Optional[torch.Tensor] = None
        self._informed_proposal_executors: dict[int, GraphExecutor] = {}
        if self.config.sampling_policy == "mixed_ellipsoid":
            if self.config.informed_min_slack is None:
                log_and_raise("informed_min_slack was not resolved by config validation")
            proposal_generator = InformedRRTProposalGenerator(
                resources.cspace_distance_weight,
                resources.action_bound_lows,
                resources.action_bound_highs,
                self.config.samples_per_iteration,
                self.config.informed_sample_count,
                self.config.goal_bias_count,
                self.config.informed_min_slack,
            ).to(resources.device_cfg.device)
            self._informed_proposal_generator = torch.compile(
                proposal_generator,
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False},
            )
            primary_sample_buffer = resources.uniform_sampler.action_sample_generator
            informed_latent_sample_buffer = SampleBuffer.create_halton_sample_buffer(
                ndims=resources.action_dim,
                up_bounds=torch.ones_like(resources.action_bound_highs),
                low_bounds=torch.zeros_like(resources.action_bound_lows),
                store_buffer=primary_sample_buffer._store_buffer,
                seed=resources.config.sampler_seed + 1,
                device_cfg=resources.device_cfg,
            )
            iteration_budget = (
                self.config.max_iterations + self.config.failure_extension_iterations
            )
            latent_values = self.config.informed_sample_count * (resources.action_dim + 1)
            latent_rows = (latent_values + resources.action_dim - 1) // resources.action_dim
            latent_sample_count = iteration_budget * self.config.max_batched_lanes * latent_rows
            self._informed_latent_schedule = informed_latent_sample_buffer.get_samples(
                latent_sample_count,
                bounded=False,
            ).reshape(
                iteration_budget,
                self.config.max_batched_lanes,
                latent_rows,
                resources.action_dim,
            )
            self._informed_scale_schedule = torch.tensor(
                [
                    min(
                        self.config.informed_max_scale,
                        self.config.informed_initial_scale
                        * self.config.informed_growth_factor
                        ** (iteration // self.config.informed_growth_interval),
                    )
                    for iteration in range(iteration_budget)
                ],
                device=resources.device_cfg.device,
                dtype=resources.device_cfg.dtype,
            )

        self._tree_kernel_runner = RRTTreeKernelRunner(
            RRTTreeKernelConfig(
                action_dim=resources.action_dim,
                max_candidates=self.config.samples_per_iteration,
                max_path_nodes=2
                * (self.config.max_iterations + self.config.failure_extension_iterations + 2),
            )
        )
        self._workspaces: dict[int, _RRTWorkspace] = {}
        self._direct_connectors: dict[int, RRTDirectConnector] = {}
        self._default_query_masks: dict[int, torch.Tensor] = {}
        self._default_query_groups: dict[tuple[int, int], torch.Tensor] = {}
        maximum_direct_distance = torch.max(
            torch.abs(
                (resources.action_bound_highs - resources.action_bound_lows)
                * resources.cspace_distance_weight
            )
        ).item()
        self._direct_steer_steps = max(
            2,
            math.ceil(maximum_direct_distance / self.config.cspace_resolution) + 1,
        )
        if self._direct_steer_steps > resources.config.steer_buffer_size:
            log_and_raise("graph-planner steer buffer is too small for a complete RRT direct edge")
        runtime_kernels = create_rrt_runtime_kernels(
            resources.action_dim,
            self.config.samples_per_iteration,
            self.config.fixed_steer_steps,
            self._direct_steer_steps,
        )
        self._reset_workspace_kernel = runtime_kernels["reset_workspace"]
        self._seed_direct_kernel = runtime_kernels["seed_direct"]

    def _allocate_tree(self, batch_size: int) -> _RRTTree:
        """Allocate one side of a fixed-capacity batched packed tree."""
        device = self.resources.device_cfg.device
        dtype = self.resources.device_cfg.dtype
        capacity = self.config.max_nodes_per_tree
        return _RRTTree(
            nodes=torch.empty(
                (batch_size, capacity, self.resources.action_dim),
                device=device,
                dtype=dtype,
            ),
            parents=torch.empty(
                (batch_size, capacity),
                device=device,
                dtype=torch.int64,
            ),
            counts=torch.ones(batch_size, device=device, dtype=torch.int32),
        )

    def _apply_constraint_goal(self, rollout: RobotRollout, batch_size: int) -> None:
        """Route all installed task-space targets across a fixed rollout batch."""
        if (
            self._constraint_goal_tool_poses is None
            and self._constraint_tool_relative_transform_target is None
        ):
            return
        goal = GoalRegistry(
            link_goal_poses=self._constraint_goal_tool_poses,
            idxs_link_pose=(
                torch.zeros(
                    (batch_size, 1),
                    device=self.resources.device_cfg.device,
                    dtype=torch.int32,
                )
                if self._constraint_goal_tool_poses is not None
                else None
            ),
            tool_relative_transform_target=self._constraint_tool_relative_transform_target,
            idxs_tool_relative_transform_target=(
                torch.zeros(
                    (batch_size, 1),
                    device=self.resources.device_cfg.device,
                    dtype=torch.int32,
                )
                if self._constraint_tool_relative_transform_target is not None
                else None
            ),
            batch_size=batch_size,
        )
        rollout.update_params(goal)
        if self._constraint_goal_tool_poses is not None:
            rollout.alias_goal_tool_poses(self._constraint_goal_tool_poses)

    def _apply_constraints(self, rollout: RobotRollout, batch_size: int) -> None:
        """Apply every stored task-space constraint to one fixed-shape rollout."""
        criteria_updates = {}
        if self._constraint_tool_pose_criteria is not None:
            criteria_updates["tool_pose_criteria"] = self._constraint_tool_pose_criteria
        if self._constraint_tool_relative_transform_criteria is not None:
            criteria_updates["tool_relative_transform_criteria"] = (
                self._constraint_tool_relative_transform_criteria
            )
        if criteria_updates:
            rollout.update_params_cost_managers(**criteria_updates)
        self._apply_constraint_goal(rollout, batch_size)

    def _apply_constraints_to_all_rollouts(self) -> None:
        """Synchronize stored constraints across every current RRT rollout."""
        for rollout, batch_size in self._constraint_rollouts():
            self._apply_constraints(rollout, batch_size)

    def _apply_tool_pose_criteria_to_all_rollouts(self) -> None:
        """Synchronize only the fixed-shape tool-pose criteria buffers."""
        if self._constraint_tool_pose_criteria is None:
            return
        for rollout, _ in self._constraint_rollouts():
            rollout.update_params_cost_managers(
                tool_pose_criteria=self._constraint_tool_pose_criteria
            )

    def _constraint_rollouts(self) -> List[Tuple[RobotRollout, int]]:
        """Return every constraint rollout with its fixed evaluation batch size."""
        rollouts = [
            (
                self.resources.feasibility_rollout,
                self.resources.config.feasibility_buffer_size,
            )
        ]
        if self._sample_projector is not None:
            projection_batch = self._sample_projector.config.projection_slots * (
                len(self._sample_projector.config.line_search_scales) - 1
            )
            rollouts.append((self.resources.auxiliary_rollout, projection_batch))
        for connector in getattr(self, "_direct_connectors", {}).values():
            rollouts.append(
                (
                    connector.rollout,
                    connector.edge_count * connector.max_steps,
                )
            )
        for workspace in self._workspaces.values():
            for growth_step in workspace.growth_steps:
                rollouts.append(
                    (
                        growth_step.rollout,
                        growth_step.batch_size * growth_step.sample_count * growth_step.max_steps,
                    )
                )
            finalization_rollout = getattr(workspace, "finalization_rollout", None)
            if finalization_rollout is not None:
                rollouts.append(
                    (
                        finalization_rollout,
                        self._shortcut_state_capacity,
                    )
                )
        return rollouts

    def _create_rollout(self) -> RobotRollout:
        """Create one graph-owned rollout while sharing robot and scene storage."""
        rollout_config_template = self.resources.config.rollout_config
        scene_checker = self.resources.scene_collision_checker
        robot_config = rollout_config_template.transition_model_cfg.robot_config
        rollout_config = copy.deepcopy(
            rollout_config_template,
            {
                id(scene_checker): scene_checker,
                id(robot_config): robot_config,
            },
        )
        return RobotRollout(rollout_config, scene_checker, use_cuda_graph=False)

    def _get_direct_connector(self, edge_count: int) -> RRTDirectConnector:
        """Return a fixed-shape direct connector for one paired edge batch."""
        connector = self._direct_connectors.get(edge_count)
        if connector is not None:
            return connector
        rollout_batch = edge_count * self._direct_steer_steps
        connector = RRTDirectConnector(
            rollout=self._create_rollout(),
            distance_weight=self.resources.cspace_distance_weight,
            edge_count=edge_count,
            max_steps=self._direct_steer_steps,
            cspace_resolution=self.config.cspace_resolution,
            similarity_threshold=self.resources.config.cspace_similarity_threshold,
        )
        self._apply_constraints(connector.rollout, rollout_batch)
        self._direct_connectors[edge_count] = connector
        return connector

    def _get_batched_workspace(self, batch_size: int) -> _RRTWorkspace:
        """Create/cache the smallest supported paired-query batch bucket."""
        bucket = 1 << (batch_size - 1).bit_length()
        if bucket > self.config.max_batched_lanes:
            log_and_raise(
                "BiRRT planning received "
                f"{batch_size} lanes, maximum is {self.config.max_batched_lanes}"
            )
        if bucket in self._workspaces:
            return self._workspaces[bucket]
        if self.config.max_nodes_per_tree % self._tree_kernel_runner.profile.block_dim != 0:
            log_and_raise(
                "batched max_nodes_per_tree must be a multiple of the generated block_dim"
            )

        device = self.resources.device_cfg.device
        dtype = self.resources.device_cfg.dtype
        sample_count = self.config.samples_per_iteration
        action_dim = self.resources.action_dim
        start_tree = self._allocate_tree(bucket)
        goal_tree = self._allocate_tree(bucket)
        path_reconstruction = self._tree_kernel_runner.bind_batched_path_reconstruction(
            start_tree.nodes,
            start_tree.parents,
            goal_tree.nodes,
            goal_tree.parents,
            self.resources.cspace_distance_weight,
        )
        path_shortcutter = None
        finalization_rollout = None
        if self.config.shortcut_path:
            path_shortcutter = RRTPathShortcutter(
                RRTPathShortcutterConfig(
                    action_dim=action_dim,
                    path_capacity=min(
                        self.config.shortcut_path_capacity,
                        self._tree_kernel_runner.profile.max_path_nodes,
                    ),
                    raw_path_capacity=self._tree_kernel_runner.profile.max_path_nodes,
                    state_capacity=self._shortcut_state_capacity,
                    batch_capacity=bucket,
                    cspace_resolution=self.config.cspace_resolution,
                ),
                device,
                self.resources.cspace_distance_weight,
                raw_path=path_reconstruction.output_paths,
                path_count=path_reconstruction.path_counts,
                raw_path_length=path_reconstruction.path_lengths,
            )
            finalization_rollout = self._create_rollout()
            self._apply_constraints(finalization_rollout, self._shortcut_state_capacity)
        path_reconstruction_executor = create_graph_executor(
            capture_fn=lambda: self._tree_kernel_runner.batched_reconstruct_path(
                path_reconstruction
            ),
            device=device,
            use_cuda_graph=True,
            clone_inputs=False,
            clone_outputs=False,
        )
        trees = (start_tree, goal_tree)
        query_active = torch.zeros(bucket, device=device, dtype=torch.bool)
        env_query_idx = torch.zeros(bucket, device=device, dtype=torch.int32)
        goal_query_idx = torch.zeros(bucket, device=device, dtype=torch.int32)
        desired_buffers = [
            torch.empty((bucket, sample_count, action_dim), device=device, dtype=dtype)
            for _ in range(2)
        ]
        # Slot zero is the active extension output and the next passive
        # nearest-query target. Both bindings therefore share one buffer.
        append_nodes = [desired_buffers[1], torch.empty_like(desired_buffers[0])]
        append_valid = [
            torch.empty((bucket, sample_count), device=device, dtype=torch.bool) for _ in range(2)
        ]
        nearest_bindings = {}
        append_bindings = {}
        for tree_index, tree in enumerate(trees):
            for slot in range(2):
                nearest_bindings[(tree_index, slot)] = (
                    self._tree_kernel_runner.bind_batched_nearest_target(
                        tree.nodes,
                        tree.counts,
                        desired_buffers[slot],
                        self.resources.cspace_distance_weight,
                    )
                )
                append_bindings[(tree_index, slot)] = (
                    self._tree_kernel_runner.bind_batched_packed_append(
                        tree.nodes,
                        tree.parents,
                        tree.counts,
                        append_nodes[slot],
                        nearest_bindings[(tree_index, slot)].parent_indices,
                        append_valid[slot],
                    )
                )

        growth_steps = []
        for active_index, passive_index in ((0, 1), (1, 0)):
            rollout_batch = bucket * sample_count * self.config.fixed_steer_steps
            rollout = self._create_rollout()
            growth_steps.append(
                RRTGrowthStep(
                    tree_kernel_runner=self._tree_kernel_runner,
                    active_nearest=nearest_bindings[(active_index, 0)],
                    passive_nearest=nearest_bindings[(passive_index, 1)],
                    active_append=append_bindings[(active_index, 0)],
                    passive_append=append_bindings[(passive_index, 1)],
                    start_counts=start_tree.counts,
                    goal_counts=goal_tree.counts,
                    connection_indices=path_reconstruction.connection_indices,
                    start_is_active=active_index == 0,
                    query_active=query_active,
                    env_query_idx=env_query_idx,
                    goal_query_idx=goal_query_idx,
                    distance_weight=self.resources.cspace_distance_weight,
                    max_extension_distance=self.config.max_extension_distance,
                    cspace_resolution=self.config.cspace_resolution,
                    max_steps=self.config.fixed_steer_steps,
                    similarity_threshold=self.resources.config.cspace_similarity_threshold,
                    rollout=rollout,
                    profile_cuda_event_phase=(
                        self.config.profile_cuda_event_phase
                        if self.config.profile_cuda_events
                        else "off"
                    ),
                )
            )
            self._apply_constraints(rollout, rollout_batch)
        self._tree_kernel_runner.load_modules(device)
        direct_status = torch.empty(bucket, device=device, dtype=torch.int32)
        request_query_mask = torch.ones(bucket, device=device, dtype=torch.bool)
        workspace = _RRTWorkspace(
            batch_size=bucket,
            start_tree=start_tree,
            goal_tree=goal_tree,
            growth_steps=growth_steps,
            query_active=query_active,
            env_query_idx=env_query_idx,
            goal_query_idx=goal_query_idx,
            path_reconstruction=path_reconstruction,
            path_reconstruction_executor=path_reconstruction_executor,
            path_shortcutter=path_shortcutter,
            finalization_rollout=finalization_rollout,
            finalization_executors={},
            path_counts_host=torch.zeros(
                bucket,
                dtype=torch.int32,
                pin_memory=True,
            ),
            shortcut_status_host=torch.zeros(
                (bucket, 5),
                dtype=torch.int32,
                pin_memory=True,
            ),
            direct_status=direct_status,
            direct_status_host=torch.empty(bucket, dtype=torch.int32, pin_memory=True),
            request_query_mask=request_query_mask,
            request_query_mask_all_active=True,
            preprocessing_executors={},
            kernel_views={
                "start_nodes": wp.from_torch(start_tree.nodes, dtype=wp.float32),
                "start_parents": wp.from_torch(start_tree.parents, dtype=wp.int64),
                "start_counts": wp.from_torch(start_tree.counts, dtype=wp.int32),
                "goal_nodes": wp.from_torch(goal_tree.nodes, dtype=wp.float32),
                "goal_parents": wp.from_torch(goal_tree.parents, dtype=wp.int64),
                "goal_counts": wp.from_torch(goal_tree.counts, dtype=wp.int32),
                "query_active": wp.from_torch(query_active, dtype=wp.bool),
                "env_query_idx": wp.from_torch(env_query_idx, dtype=wp.int32),
                "goal_query_idx": wp.from_torch(goal_query_idx, dtype=wp.int32),
                "connection_indices": wp.from_torch(
                    path_reconstruction.connection_indices.reshape(-1), dtype=wp.int64
                ),
                "path_counts": wp.from_torch(path_reconstruction.path_counts, dtype=wp.int32),
                "path_lengths": wp.from_torch(path_reconstruction.path_lengths, dtype=wp.float32),
                "direct_status": wp.from_torch(direct_status, dtype=wp.int32),
                "request_query_mask": wp.from_torch(request_query_mask, dtype=wp.bool),
            },
        )
        self._workspaces[bucket] = workspace
        return workspace

    @staticmethod
    def _get_workspace_kernel_views(workspace: _RRTWorkspace) -> dict[str, object]:
        """Return fixed-address Warp views, lazily creating views for test workspaces."""
        views = getattr(workspace, "kernel_views", None)
        if views is not None:
            return views
        views = {
            "start_nodes": wp.from_torch(workspace.start_tree.nodes, dtype=wp.float32),
            "start_parents": wp.from_torch(workspace.start_tree.parents, dtype=wp.int64),
            "start_counts": wp.from_torch(workspace.start_tree.counts, dtype=wp.int32),
            "goal_nodes": wp.from_torch(workspace.goal_tree.nodes, dtype=wp.float32),
            "goal_parents": wp.from_torch(workspace.goal_tree.parents, dtype=wp.int64),
            "goal_counts": wp.from_torch(workspace.goal_tree.counts, dtype=wp.int32),
            "query_active": wp.from_torch(workspace.query_active, dtype=wp.bool),
            "env_query_idx": wp.from_torch(workspace.env_query_idx, dtype=wp.int32),
            "goal_query_idx": wp.from_torch(workspace.goal_query_idx, dtype=wp.int32),
            "connection_indices": wp.from_torch(
                workspace.path_reconstruction.connection_indices.reshape(-1), dtype=wp.int64
            ),
            "path_counts": wp.from_torch(
                workspace.path_reconstruction.path_counts, dtype=wp.int32
            ),
            "path_lengths": wp.from_torch(
                workspace.path_reconstruction.path_lengths, dtype=wp.float32
            ),
            "direct_status": wp.from_torch(workspace.direct_status, dtype=wp.int32),
            "request_query_mask": wp.from_torch(workspace.request_query_mask, dtype=wp.bool),
        }
        workspace.kernel_views = views
        return views

    def _get_finalization_executor(
        self,
        workspace: _RRTWorkspace,
        *,
        route_environment: bool,
        route_goal: bool,
    ) -> GraphExecutor:
        """Return one graph that reconstructs and shortcuts every workspace lane."""
        route_key = (route_environment, route_goal)
        executor = workspace.finalization_executors.get(route_key)
        if executor is not None:
            return executor
        shortcutter = workspace.path_shortcutter
        rollout = workspace.finalization_rollout
        if shortcutter is None or rollout is None:
            log_and_raise("RRT finalization requires shortcut buffers and rollout")

        def launch_finalization() -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            self._tree_kernel_runner.batched_reconstruct_path(workspace.path_reconstruction)
            shortcutter._launch_prepare()
            if route_goal:
                rollout.route_goal_poses(shortcutter.goal_query_idx.view(-1, 1))
            metrics = rollout.compute_metrics_from_action(
                shortcutter.packed_states.unsqueeze(1),
                idxs_env_query=(shortcutter.env_query_idx if route_environment else None),
            )
            feasible = metrics.costs_and_constraints.get_feasible(
                include_all_hybrid=False,
                sum_horizon=True,
            )
            shortcutter.feasible.copy_(feasible.reshape(-1))
            shortcutter._launch_select()
            return (
                shortcutter.output_path,
                shortcutter.output_count,
                shortcutter.path_length,
                shortcutter.success,
                shortcutter.status,
            )

        executor = create_graph_executor(
            capture_fn=launch_finalization,
            device=self.resources.device_cfg.device,
            use_cuda_graph=True,
            clone_inputs=False,
            clone_outputs=False,
        )
        workspace.finalization_executors[route_key] = executor
        return executor

    @staticmethod
    def _empty_projection_stats(enabled: bool) -> dict[str, float | int | bool]:
        """Create query-level counters for the optional proposal projector."""
        return {
            "enabled": enabled,
            "calls": 0,
            "eligible_candidates": 0,
            "initially_invalid": 0,
            "selected": 0,
            "repaired": 0,
            "substituted": 0,
            "displacement_sum": 0.0,
            "displacement_max": 0.0,
        }

    def _project_growth_samples(
        self,
        samples: torch.Tensor,
        active: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
        route_goals: bool,
    ) -> dict[str, float | int]:
        """Repair selected invalid active proposal targets before ordinary growth.

        Only samples that become exactly feasible replace their original
        targets. Valid, unselected, inactive, and unsuccessfully repaired
        proposals remain byte-for-byte unchanged. The growth step still
        recomputes nearest steering lines and validates both active and passive
        edges.
        """
        sample_count = self.config.samples_per_iteration
        flat_samples = samples.reshape(-1, self.resources.action_dim)
        eligible = (
            active.view(-1, 1).expand(active.shape[0], sample_count).reshape(-1).contiguous()
        )
        sample_env = env_query_idx.repeat_interleave(sample_count).contiguous()
        sample_goal = goal_query_idx.repeat_interleave(sample_count).contiguous()
        projection = self.project_samples(
            flat_samples,
            env_query_idx=sample_env,
            goal_query_idx=sample_goal if route_goals else None,
            eligible_mask=eligible,
        )
        selected = eligible & projection.violation_measured & ~projection.initially_feasible
        repaired = selected & projection.feasible
        finite = torch.isfinite(projection.samples).all(dim=-1)
        substituted = repaired & finite
        flat_samples.copy_(
            torch.where(
                substituted.view(-1, 1),
                projection.samples,
                flat_samples,
            )
        )

        counts = (
            torch.stack(
                (
                    torch.count_nonzero(eligible),
                    torch.count_nonzero(eligible & ~projection.initially_feasible),
                    torch.count_nonzero(selected),
                    torch.count_nonzero(repaired),
                    torch.count_nonzero(substituted),
                )
            )
            .cpu()
            .tolist()
        )
        substituted_displacement = torch.where(
            substituted,
            projection.displacement,
            torch.zeros_like(projection.displacement),
        )
        displacement_sum, displacement_max = (
            torch.stack((substituted_displacement.sum(), substituted_displacement.amax()))
            .cpu()
            .tolist()
        )
        return {
            "calls": 1,
            "eligible_candidates": int(counts[0]),
            "initially_invalid": int(counts[1]),
            "selected": int(counts[2]),
            "repaired": int(counts[3]),
            "substituted": int(counts[4]),
            "displacement_sum": float(displacement_sum),
            "displacement_max": float(displacement_max),
        }

    @staticmethod
    def _accumulate_projection_stats(
        total: dict[str, float | int | bool],
        update: dict[str, float | int],
    ) -> None:
        """Accumulate per-step projection counters into query telemetry."""
        for name in (
            "calls",
            "eligible_candidates",
            "initially_invalid",
            "selected",
            "repaired",
            "substituted",
            "displacement_sum",
        ):
            total[name] += update[name]
        total["displacement_max"] = max(
            float(total["displacement_max"]),
            float(update["displacement_max"]),
        )

    def _sample_batched(self, batch_size: int) -> torch.Tensor:
        """Draw one fixed-width proposal batch for every query lane."""
        return self.resources.uniform_sampler.generate_action_samples(
            batch_size * self.config.samples_per_iteration,
            bounded=True,
        ).reshape(batch_size, self.config.samples_per_iteration, self.resources.action_dim)

    def _sample_mixed_batched(
        self,
        batch_size: int,
        starts: torch.Tensor,
        goals: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Draw uniform fallbacks and filled-ball latents in one fixed RNG call."""
        if self._informed_proposal_generator is None:
            log_and_raise("mixed ellipsoid sampler was not initialized")
        action_dim = self.resources.action_dim
        informed_count = self.config.informed_sample_count
        latent_values = informed_count * (action_dim + 1)
        latent_rows = (latent_values + action_dim - 1) // action_dim
        rows_per_lane = self.config.samples_per_iteration + latent_rows
        primary_sample_buffer = self.resources.uniform_sampler.action_sample_generator
        uniform_raw = primary_sample_buffer.get_samples(
            batch_size * self.config.samples_per_iteration,
            bounded=False,
        ).reshape(batch_size, self.config.samples_per_iteration, action_dim)
        if self._informed_latent_schedule is None:
            log_and_raise("mixed ellipsoid latent schedule was not initialized")
        if batch_size > self._informed_latent_schedule.shape[1]:
            log_and_raise("mixed ellipsoid batch exceeds max_batched_lanes")
        latent_raw = self._informed_latent_schedule[iteration, :batch_size]
        raw = torch.cat((uniform_raw, latent_raw), dim=1).reshape(
            batch_size, rows_per_lane, action_dim
        )
        if self._informed_scale_schedule is None:
            log_and_raise("mixed ellipsoid scale schedule was not initialized")
        executor = self._informed_proposal_executors.get(batch_size)
        if executor is None:
            executor = create_graph_executor(
                capture_fn=self._informed_proposal_generator,
                device=raw.device,
                use_cuda_graph=True,
                clone_outputs=False,
            )
            self._informed_proposal_executors[batch_size] = executor
        samples, _ = executor(
            raw,
            starts,
            goals,
            self._informed_scale_schedule[iteration],
        )
        return samples

    def _generate_proposal_batch(
        self,
        batch_size: int,
        starts: torch.Tensor,
        goals: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Use the unchanged uniform path or the opt-in mixed focal sampler."""
        if self.config.sampling_policy == "uniform":
            return self._sample_batched(batch_size)
        return self._sample_mixed_batched(batch_size, starts, goals, iteration)

    def _interpolate_paths(
        self,
        paths: List[Optional[torch.Tensor]],
        interpolation_steps: int,
        interpolation_type: TrajInterpolationType,
    ) -> torch.Tensor:
        """Interpolate linear graph seeds on-device.

        The generic cubic/quintic fallback copies every path through
        CPU/NumPy/SciPy. RRT already owns GPU waypoint tensors, and
        MotionPlanner requests uniform linear interpolation, which is exactly
        the one-dimensional operation implemented by
        ``torch.nn.functional.interpolate`` with aligned end points.
        """
        if interpolation_type != TrajInterpolationType.LINEAR:
            success = torch.tensor(
                [path is not None for path in paths],
                device=self.resources.device_cfg.device,
                dtype=torch.bool,
            )
            return self.resources.get_interpolated_trajectory(
                paths, success, interpolation_steps, interpolation_type
            )

        interpolated = torch.zeros(
            (len(paths), interpolation_steps, self.resources.action_dim),
            device=self.resources.device_cfg.device,
            dtype=self.resources.device_cfg.dtype,
        )
        for batch_index, path in enumerate(paths):
            if path is None:
                continue
            interpolated[batch_index].copy_(
                F.interpolate(
                    path.transpose(0, 1).unsqueeze(0),
                    size=interpolation_steps,
                    mode="linear",
                    align_corners=True,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return interpolated

    @staticmethod
    def _clone_paths(paths: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        """Copy all successful variable-length paths with one device operation."""
        present = [(index, path) for index, path in enumerate(paths) if path is not None]
        if not present:
            return [None] * len(paths)
        packed = torch.cat([path for _, path in present], dim=0)
        stable: List[Optional[torch.Tensor]] = [None] * len(paths)
        offset = 0
        for index, path in present:
            next_offset = offset + path.shape[0]
            stable[index] = packed[offset:next_offset]
            offset = next_offset
        return stable

    def _environment_capacity(self) -> int:
        """Return environment IDs valid for both robot and scene storage."""
        robot_capacity = self.resources.kinematics.config.kinematics_config.num_envs
        scene = self.resources.scene_collision_checker
        if scene is None:
            return robot_capacity
        return min(robot_capacity, scene.num_envs)

    def _normalize_batched_request(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
        query_mask: Optional[torch.Tensor],
        query_group_id: Optional[torch.Tensor],
        group_count: Optional[int],
    ) -> _RRTRequest:
        """Validate public inputs and normalize optional lane metadata."""
        if x_start.ndim != 2 or x_goal.ndim != 2:
            log_and_raise("x_start and x_goal must be 2D tensors")
        if x_start.shape != x_goal.shape or x_start.shape[1] != self.resources.action_dim:
            log_and_raise("x_start and x_goal must have matching (batch, action_dim) shapes")
        batch_size = x_start.shape[0]
        if batch_size < 1:
            log_and_raise("batched planning requires at least one query")
        if batch_size > self.config.max_batched_lanes:
            log_and_raise(
                "BiRRT planning received "
                f"{batch_size} lanes, maximum is {self.config.max_batched_lanes}"
            )

        supplied_query_mask = query_mask is not None
        supplied_group_ids = query_group_id is not None
        if query_mask is None:
            query_mask = self._default_query_masks.get(batch_size)
            if query_mask is None:
                query_mask = torch.ones(batch_size, device=x_start.device, dtype=torch.bool)
                self._default_query_masks[batch_size] = query_mask
            query_mask_host = [True] * batch_size
        else:
            if (
                query_mask.shape != (batch_size,)
                or query_mask.dtype != torch.bool
                or query_mask.device != x_start.device
            ):
                log_and_raise("query_mask must be CUDA bool with shape (batch,)")
            query_mask_host = []

        if query_group_id is None:
            if group_count not in (None, 1, batch_size):
                log_and_raise("group_count without query_group_id must be one or equal batch size")
            resolved_group_count = batch_size if group_count is None else group_count
            group_key = (batch_size, resolved_group_count)
            query_group_id = self._default_query_groups.get(group_key)
            if query_group_id is None:
                query_group_id = (
                    torch.zeros(batch_size, device=x_start.device, dtype=torch.int32)
                    if resolved_group_count == 1
                    else torch.arange(batch_size, device=x_start.device, dtype=torch.int32)
                )
                self._default_query_groups[group_key] = query_group_id
            group_ids_host = (
                [0] * batch_size if resolved_group_count == 1 else list(range(batch_size))
            )
        else:
            if (
                query_group_id.shape != (batch_size,)
                or query_group_id.dtype != torch.int32
                or query_group_id.device != x_start.device
            ):
                log_and_raise("query_group_id must be CUDA int32 with shape (batch,)")
            resolved_group_count = -1 if group_count is None else group_count
            group_ids_host = []

        if supplied_query_mask and supplied_group_ids:
            request_metadata_host = (
                torch.stack((query_mask.to(torch.int32), query_group_id), dim=1).cpu().tolist()
            )
            query_mask_host = [bool(value[0]) for value in request_metadata_host]
            group_ids_host = [int(value[1]) for value in request_metadata_host]
        elif supplied_query_mask:
            query_mask_host = [bool(value) for value in query_mask.cpu().tolist()]
        elif supplied_group_ids:
            group_ids_host = [int(value) for value in query_group_id.cpu().tolist()]
        if resolved_group_count < 0:
            resolved_group_count = max(group_ids_host) + 1
        if resolved_group_count <= 0:
            log_and_raise("group_count must be positive")
        if min(group_ids_host) < 0 or max(group_ids_host) >= resolved_group_count:
            log_and_raise("query_group_id values must be in [0, group_count)")
        present_groups = [False] * resolved_group_count
        for group_index in group_ids_host:
            present_groups[group_index] = True
        if not all(present_groups):
            log_and_raise("every query group must contain at least one lane")

        constraint_goal_count = (
            0
            if self._constraint_goal_tool_poses is None
            else self._constraint_goal_tool_poses.batch_size
        )
        if constraint_goal_count not in (0, 1, resolved_group_count):
            log_and_raise(
                "RRT constraint goal batch must be one or match group_count: "
                f"{constraint_goal_count} not in (1, {resolved_group_count})"
            )
        route_goals = constraint_goal_count > 1
        # The growth-step ABI always receives a route tensor. Borrow existing
        # group metadata when goal routing is disabled to keep unconstrained
        # requests allocation-free.
        goal_query_idx = query_group_id

        explicit_env = env_query_idx is not None
        if env_query_idx is None:
            env_query_idx = torch.zeros(
                batch_size,
                device=x_start.device,
                dtype=torch.int32,
            )
        else:
            if (
                env_query_idx.shape != (batch_size,)
                or env_query_idx.dtype != torch.int32
                or env_query_idx.device != x_start.device
            ):
                log_and_raise("env_query_idx must be CUDA int32 with shape (batch,)")
            env_min, env_max = torch.aminmax(env_query_idx)
            env_bounds = torch.stack((env_min, env_max)).cpu().tolist()
            environment_capacity = self._environment_capacity()
            if env_bounds[0] < 0 or env_bounds[1] >= environment_capacity:
                log_and_raise(f"env_query_idx values must be in [0, {environment_capacity})")

        has_requested_lanes = any(query_mask_host)
        if not has_requested_lanes:
            # These tensors are never evaluated by the all-masked fast path,
            # but keeping raw masked values out of the normalized request makes
            # that isolation explicit and robust to later refactors.
            x_start = torch.zeros_like(x_start)
            x_goal = torch.zeros_like(x_goal)
            env_query_idx = torch.zeros_like(env_query_idx)
            if route_goals:
                goal_query_idx = torch.zeros_like(goal_query_idx)
        elif not all(query_mask_host):
            # Preprocessing kernels are vectorized across the full lane bucket.
            # Replace masked inputs with one requested start configuration so
            # NaN, Inf, or extreme masked values cannot affect global line-size
            # decisions, feasibility staging, or fixed-address tree roots.
            anchor_index = query_mask_host.index(True)
            anchor_start = x_start[anchor_index].view(1, -1)
            lane_mask = query_mask.view(-1, 1)
            x_start = torch.where(lane_mask, x_start, anchor_start)
            x_goal = torch.where(lane_mask, x_goal, anchor_start)
            env_query_idx = torch.where(
                query_mask,
                env_query_idx,
                env_query_idx[anchor_index],
            )
            if route_goals:
                goal_query_idx = torch.where(
                    query_mask,
                    goal_query_idx,
                    goal_query_idx[anchor_index],
                )

        return _RRTRequest(
            x_start=x_start,
            x_goal=x_goal,
            query_mask=query_mask,
            env_query_idx=env_query_idx,
            goal_query_idx=goal_query_idx,
            batch_size=batch_size,
            group_count=resolved_group_count,
            explicit_env=explicit_env,
            has_goal_constraint=constraint_goal_count > 0,
            route_goals=route_goals,
            has_requested_lanes=has_requested_lanes,
            query_mask_host=query_mask_host,
            group_ids_host=group_ids_host,
        )

    def _initialize_search_state(
        self,
        request: _RRTRequest,
    ) -> _RRTSearchState:
        """Construct control metadata before fused direct-edge validation."""
        query_valid_host = request.query_mask_host.copy()
        group_members: List[List[int]] = [[] for _ in range(request.group_count)]
        for lane_index, lane_group in enumerate(request.group_ids_host):
            group_members[lane_group].append(lane_index)
        group_valid_host = [
            any(query_valid_host[lane_index] for lane_index in members)
            for members in group_members
        ]
        query_debug = []
        for requested in request.query_mask_host:
            if not requested:
                query_debug.append({"skipped": "query masked"})
            else:
                query_debug.append({})
        return _RRTSearchState(
            group_ids_host=request.group_ids_host,
            group_valid_host=group_valid_host,
            paths=[None] * request.batch_size,
            query_debug=query_debug,
            active_host=query_valid_host.copy(),
            group_solved=[False] * request.group_count,
            connection_records=[],
        )

    def _launch_preprocessing(
        self,
        workspace: _RRTWorkspace,
        connector: RRTDirectConnector,
        request_count: int,
        goal_route_mode: int,
    ) -> torch.Tensor:
        """Reset, validate terminal edges, and seed both trees in one graph."""
        _, stream = get_warp_device_stream(connector.request_starts)
        if stream is None:
            log_and_raise("RRT preprocessing requires a CUDA Warp stream")
        views = self._get_workspace_kernel_views(workspace)
        wp.launch(
            self._reset_workspace_kernel,
            dim=workspace.batch_size,
            inputs=[
                connector._views["request_starts"],
                connector._views["request_goals"],
                connector._views["request_env_query_idx"],
                connector._views["request_goal_query_idx"],
                request_count,
                goal_route_mode,
            ],
            outputs=[
                views["start_nodes"],
                views["start_parents"],
                views["start_counts"],
                views["goal_nodes"],
                views["goal_parents"],
                views["goal_counts"],
                views["query_active"],
                views["env_query_idx"],
                views["goal_query_idx"],
                views["connection_indices"],
                views["path_counts"],
                views["path_lengths"],
            ],
            block_dim=self._tree_kernel_runner.profile.block_dim,
            stream=stream,
        )
        direct_endpoints, direct_reached, direct_progressed, terminal_feasible = (
            self._evaluate_staged_direct(connector)
        )
        wp.launch(
            self._seed_direct_kernel,
            dim=workspace.batch_size,
            inputs=[
                wp.from_torch(direct_endpoints, dtype=wp.float32),
                wp.from_torch(direct_reached, dtype=wp.bool),
                wp.from_torch(direct_progressed, dtype=wp.bool),
                wp.from_torch(terminal_feasible, dtype=wp.bool),
                views["request_query_mask"],
                request_count,
            ],
            outputs=[
                views["start_nodes"],
                views["start_parents"],
                views["start_counts"],
                views["goal_nodes"],
                views["goal_parents"],
                views["goal_counts"],
                views["query_active"],
                views["connection_indices"],
                views["direct_status"],
            ],
            block_dim=self._tree_kernel_runner.profile.block_dim,
            stream=stream,
        )
        return workspace.direct_status

    @staticmethod
    def _evaluate_staged_direct(
        connector: RRTDirectConnector,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate graph-stable paired terminal inputs."""
        return connector.evaluate_staged()

    def _preprocess_search(
        self,
        request: _RRTRequest,
        state: _RRTSearchState,
    ) -> _RRTWorkspace:
        """Run graph-captured request initialization and update host scheduling state."""
        workspace = self._get_batched_workspace(request.batch_size)
        connector = self._get_direct_connector(2 * request.batch_size)
        connector.stage_request(
            request.x_start,
            request.x_goal,
            request.env_query_idx if request.explicit_env else None,
            request.goal_query_idx if request.route_goals else None,
        )
        all_active = all(request.query_mask_host)
        if all_active != workspace.request_query_mask_all_active:
            if all_active:
                workspace.request_query_mask.fill_(True)
            else:
                workspace.request_query_mask[: request.batch_size].copy_(request.query_mask)
            workspace.request_query_mask_all_active = all_active
        elif not all_active:
            workspace.request_query_mask[: request.batch_size].copy_(request.query_mask)

        goal_route_mode = 2 if request.route_goals else int(request.has_goal_constraint)
        executor_key = (request.batch_size, goal_route_mode)
        executor = workspace.preprocessing_executors.get(executor_key)
        if executor is None:
            executor = create_graph_executor(
                capture_fn=lambda: self._launch_preprocessing(
                    workspace,
                    connector,
                    request.batch_size,
                    goal_route_mode,
                ),
                device=request.x_start.device,
                use_cuda_graph=True,
                clone_inputs=False,
                clone_outputs=False,
            )
            workspace.preprocessing_executors[executor_key] = executor
        direct_status_device = executor()
        if not isinstance(direct_status_device, torch.Tensor):
            log_and_raise("RRT preprocessing must return one status tensor")
        workspace.direct_status_host.copy_(direct_status_device, non_blocking=True)
        torch.cuda.current_stream(request.x_start.device).synchronize()
        direct_status = workspace.direct_status_host[: request.batch_size].tolist()
        state.active_host[:] = [status == 1 for status in direct_status]
        group_valid_host = [False] * request.group_count
        for batch_index, status in enumerate(direct_status):
            if status > 0:
                group_valid_host[state.group_ids_host[batch_index]] = True
            elif request.query_mask_host[batch_index]:
                state.query_debug[batch_index] = {"error": "start or goal is in collision"}
        state.group_valid_host[:] = group_valid_host
        for batch_index, status in enumerate(direct_status):
            if status == 2:
                group_index = state.group_ids_host[batch_index]
                state.connection_records.append(
                    _RRTConnectionRecord(
                        lane=batch_index,
                        iterations=0,
                        start_count=1,
                        goal_count=1,
                        is_direct=True,
                    )
                )
                state.group_solved[group_index] = True
                state.active_host[batch_index] = False
        return workspace

    def _run_growth_steps(
        self,
        request: _RRTRequest,
        state: _RRTSearchState,
        workspace: _RRTWorkspace,
        padded_env: torch.Tensor,
        padded_goal: torch.Tensor,
        profile_checkpoint: Callable[[str], None],
    ) -> Tuple[int, dict[str, float], dict[str, float | int | bool]]:
        """Execute complete growth steps until every valid completion owner succeeds."""
        iteration_budget = self.config.max_iterations + self.config.failure_extension_iterations
        growth_steps = 0
        captured_component_times: dict[str, float] = {}
        projection_stats = self._empty_projection_stats(self.config.sample_projection is not None)
        for iteration in range(iteration_budget):
            if not any(state.active_host):
                break
            if all(
                solved or not valid
                for solved, valid in zip(state.group_solved, state.group_valid_host)
            ):
                break
            start_is_active = iteration % 2 == 0
            profile_checkpoint("growth_pre_control")
            samples = self._generate_proposal_batch(
                workspace.batch_size,
                workspace.start_tree.nodes[:, 0],
                workspace.goal_tree.nodes[:, 0],
                iteration,
            )
            if self.config.goal_bias_count > 0:
                bias_target = (
                    workspace.goal_tree.nodes[:, 0]
                    if start_is_active
                    else workspace.start_tree.nodes[:, 0]
                )
                samples[:, : self.config.goal_bias_count].copy_(
                    bias_target.view(workspace.batch_size, 1, -1)
                )
            profile_checkpoint("growth_sample")
            active = workspace.query_active
            if self.config.sample_projection is not None:
                profile_checkpoint("growth_pre_project")
                self._accumulate_projection_stats(
                    projection_stats,
                    self._project_growth_samples(
                        samples,
                        active,
                        padded_env,
                        padded_goal,
                        request.route_goals,
                    ),
                )
                profile_checkpoint("growth_project")
            growth_step = workspace.growth_steps[0 if start_is_active else 1]
            growth_samples = samples
            if isinstance(growth_step, RRTGrowthStep):
                growth_samples = growth_step.active_nearest.desired
                growth_samples.copy_(samples)
            if not growth_step.is_initialized:
                growth_step.initialize(growth_samples, active, padded_env, padded_goal)
            status = growth_step.execute(growth_samples, active, padded_env, padded_goal)
            if self.config.profile_cuda_events:
                step_profile = growth_step.last_cuda_event_profile
                if step_profile is not None:
                    for phase, elapsed in step_profile.items():
                        key = f"captured_{phase}"
                        captured_component_times[key] = (
                            captured_component_times.get(key, 0.0) + elapsed
                        )
            profile_checkpoint("growth_execute_status")
            growth_steps += 1
            for batch_index in range(request.batch_size):
                if not state.active_host[batch_index]:
                    continue
                selected, selected_active, selected_passive, start_count, goal_count = status[
                    batch_index
                ]
                if selected >= self.config.samples_per_iteration:
                    continue
                if start_is_active:
                    start_index, goal_index = selected_active, selected_passive
                else:
                    start_index, goal_index = selected_passive, selected_active
                group_index = state.group_ids_host[batch_index]
                state.connection_records.append(
                    _RRTConnectionRecord(
                        lane=batch_index,
                        start_index=start_index,
                        goal_index=goal_index,
                        iterations=iteration + 1,
                        start_count=start_count,
                        goal_count=goal_count,
                    )
                )
                state.group_solved[group_index] = True
                state.active_host[batch_index] = False
            profile_checkpoint("growth_post_control")
        return growth_steps, captured_component_times, projection_stats

    def _reconstruct_paths(
        self,
        state: _RRTSearchState,
        workspace: _RRTWorkspace,
    ) -> None:
        """Reconstruct every connected lane into fixed GPU storage."""
        self._reconstruct_growth_paths(workspace, state.connection_records)

    def _reconstruct_growth_paths(
        self,
        workspace: _RRTWorkspace,
        records: List[_RRTConnectionRecord],
    ) -> None:
        """Populate fixed path storage with one graph-replayed GPU trace."""
        if not records:
            return
        seen_lanes: set[int] = set()
        for record in records:
            if not record.is_direct and (record.start_index is None or record.goal_index is None):
                log_and_raise("non-direct RRT connection is missing parent indices")
            if record.lane in seen_lanes:
                log_and_raise("RRT produced multiple non-direct connections for one lane")
            seen_lanes.add(record.lane)

        if (
            self.config.shortcut_path
            and any(not record.is_direct for record in records)
            and getattr(workspace, "finalization_rollout", None) is not None
        ):
            return
        output = workspace.path_reconstruction_executor()
        if not isinstance(output, tuple) or len(output) != 3:
            log_and_raise("RRT path reconstruction must return paths, counts, and lengths")

    def _shortcut_paths(
        self,
        request: _RRTRequest,
        state: _RRTSearchState,
        workspace: _RRTWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Finalize every reconstructed lane with one batched shortcut rollout."""
        reconstruction = workspace.path_reconstruction
        output_paths = reconstruction.output_paths
        output_counts = reconstruction.path_counts
        path_lengths = reconstruction.path_lengths
        success = output_counts >= 2
        shortcut_status = None
        run_shortcut = self.config.shortcut_path and any(
            not record.is_direct for record in state.connection_records
        )
        if run_shortcut:
            if workspace.path_shortcutter is None:
                log_and_raise("RRT shortcut workspace is not initialized")
            workspace.path_shortcutter.route_query_indices(
                workspace.env_query_idx if request.explicit_env else None,
                workspace.goal_query_idx if request.route_goals else None,
            )
            output = self._get_finalization_executor(
                workspace,
                route_environment=request.explicit_env,
                route_goal=request.route_goals,
            )()
            if not isinstance(output, tuple) or len(output) != 5:
                log_and_raise("RRT finalization graph returned invalid outputs")
            (
                output_paths,
                output_counts,
                path_lengths,
                success,
                shortcut_status,
            ) = output

        workspace.path_counts_host.copy_(reconstruction.path_counts, non_blocking=True)
        if shortcut_status is not None:
            workspace.shortcut_status_host.copy_(shortcut_status, non_blocking=True)
        torch.cuda.current_stream(reconstruction.output_paths.device).synchronize()

        shortcut_records = []
        for record in state.connection_records:
            batch_index = record.lane
            raw_waypoints = int(workspace.path_counts_host[batch_index])
            if raw_waypoints < 2:
                log_and_raise(
                    "GPU RRT path reconstruction exceeded capacity or found invalid parents"
                )
            selected_waypoints = raw_waypoints
            shortcut_stats = {}
            if shortcut_status is not None:
                (
                    selected_waypoints,
                    used_states,
                    tested_edges,
                    overflow_edges,
                    capacity_miss,
                ) = workspace.shortcut_status_host[batch_index].tolist()
                if raw_waypoints > 2:
                    shortcut_stats = {
                        "selected_waypoints": int(selected_waypoints),
                        "used_states": int(used_states),
                        "tested_edges": int(tested_edges),
                        "overflow_edges": int(overflow_edges),
                    }
                    if capacity_miss:
                        shortcut_stats["capacity_miss"] = int(capacity_miss)
                    shortcut_records.append(shortcut_stats)
            if selected_waypoints < 2:
                log_and_raise("GPU RRT shortcut returned an invalid waypoint count")
            path = output_paths[batch_index, :selected_waypoints]
            record.path = path
            record.raw_waypoints = raw_waypoints
            state.paths[batch_index] = path
            state.query_debug[batch_index] = {
                "iterations": record.iterations,
                "start_tree_nodes": record.start_count,
                "goal_tree_nodes": record.goal_count,
                "raw_waypoints": raw_waypoints,
                "waypoints": selected_waypoints,
                "bi_rrt_growth_step": not record.is_direct,
                "batch_bucket": workspace.batch_size,
            }
            if shortcut_stats:
                state.query_debug[batch_index]["shortcut"] = shortcut_stats
        shortcut_summary = {
            "calls": len(shortcut_records),
            "selected_waypoints": sum(
                int(record.get("selected_waypoints", 0)) for record in shortcut_records
            ),
            "used_states": sum(int(record.get("used_states", 0)) for record in shortcut_records),
            "tested_edges": sum(int(record.get("tested_edges", 0)) for record in shortcut_records),
            "overflow_edges": sum(
                int(record.get("overflow_edges", 0)) for record in shortcut_records
            ),
            "capacity_misses": sum(
                int(record.get("capacity_miss", 0)) for record in shortcut_records
            ),
        }
        return (
            success[: request.batch_size],
            path_lengths[: request.batch_size].clone(),
            shortcut_summary,
        )

    def find_path_batched(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        *,
        env_query_idx: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        query_group_id: Optional[torch.Tensor] = None,
        group_count: Optional[int] = None,
        interpolate_waypoints: bool = True,
        interpolation_steps: int = 100,
        interpolation_type: TrajInterpolationType = TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory: bool = True,
    ) -> GraphPlannerResult:
        """Plan paired queries through the canonical growth-step executors.

        Every returned field remains lane-shaped. ``query_group_id`` is only
        optional completion-owner metadata: execution stops after a complete
        vectorized stage has produced at least one successful lane for every
        valid owner. Without explicit IDs, ``group_count=1`` assigns every
        lane to one owner; otherwise lanes use identity owners.
        """
        request = self._normalize_batched_request(
            x_start,
            x_goal,
            env_query_idx,
            query_mask,
            query_group_id,
            group_count,
        )
        timer = CudaEventTimer().start()
        profile_cuda_events = self.config.profile_cuda_events

        def profile_checkpoint(name: str) -> None:
            if profile_cuda_events:
                timer.checkpoint(name)

        if not request.has_requested_lanes:
            profile_checkpoint("masked_request")
            iteration_budget = (
                self.config.max_iterations + self.config.failure_extension_iterations
            )
            result = GraphPlannerResult(
                success=torch.zeros(
                    request.batch_size,
                    device=request.x_start.device,
                    dtype=torch.bool,
                ),
                plan_waypoints=[None] * request.batch_size,
                joint_names=self.resources.joint_names,
                path_length=torch.full(
                    (request.batch_size,),
                    torch.inf,
                    device=request.x_start.device,
                    dtype=request.x_start.dtype,
                ),
                valid_query=False,
                debug_info={
                    "queries": [{"skipped": "query masked"} for _ in range(request.batch_size)],
                    "implementation": "bi_rrt_growth_step",
                    "batch_queries": request.batch_size,
                    "batch_bucket": 1 << (request.batch_size - 1).bit_length(),
                    "group_count": request.group_count,
                    "query_group_ids": request.group_ids_host,
                    "group_valid_mask": [False] * request.group_count,
                    "group_solved_mask": [False] * request.group_count,
                    "growth_steps": 0,
                    "step_budget": iteration_budget,
                    "budget_exhausted": False,
                    "used_failure_extension": False,
                    "failure_extension_steps": 0,
                    "projection": self._empty_projection_stats(
                        self.config.sample_projection is not None
                    ),
                    "cuda_event_profile_phase": (
                        self.config.profile_cuda_event_phase if profile_cuda_events else "off"
                    ),
                    "shortcut": {
                        "calls": 0,
                        "selected_waypoints": 0,
                        "used_states": 0,
                        "tested_edges": 0,
                        "overflow_edges": 0,
                        "capacity_misses": 0,
                    },
                },
            )
            result.solve_time = timer.stop(
                final_interval="finalize" if profile_cuda_events else None
            )
            if profile_cuda_events:
                result.component_times = timer.intervals
            return result

        state = self._initialize_search_state(request)
        profile_checkpoint("terminal_metadata")
        workspace = self._preprocess_search(request, state)
        profile_checkpoint("preprocess")

        growth_steps, captured_component_times, projection_stats = self._run_growth_steps(
            request,
            state,
            workspace,
            workspace.env_query_idx,
            workspace.goal_query_idx,
            profile_checkpoint,
        )
        self._reconstruct_paths(state, workspace)
        profile_checkpoint("reconstruct")
        shortcut_start_event = None
        shortcut_end_event = None
        if (
            self.config.measure_shortcut_time
            and not profile_cuda_events
            and runtime.cuda_event_timers
        ):
            shortcut_start_event = torch.cuda.Event(enable_timing=True)
            shortcut_end_event = torch.cuda.Event(enable_timing=True)
            shortcut_start_event.record()
        success, path_lengths, shortcut_summary = self._shortcut_paths(request, state, workspace)
        if shortcut_end_event is not None:
            shortcut_end_event.record()
        profile_checkpoint("shortcut")
        iteration_budget = self.config.max_iterations + self.config.failure_extension_iterations
        unresolved_valid_group = any(
            valid and not solved
            for valid, solved in zip(state.group_valid_host, state.group_solved)
        )
        budget_exhausted = growth_steps == iteration_budget and unresolved_valid_group
        result = GraphPlannerResult(
            success=success,
            plan_waypoints=state.paths,
            joint_names=self.resources.joint_names,
            path_length=path_lengths,
            valid_query=all(state.group_valid_host),
            debug_info={
                "queries": state.query_debug,
                "implementation": "bi_rrt_growth_step",
                "batch_queries": request.batch_size,
                "batch_bucket": workspace.batch_size,
                "group_count": request.group_count,
                "query_group_ids": state.group_ids_host,
                "group_valid_mask": state.group_valid_host,
                "group_solved_mask": state.group_solved,
                "growth_steps": growth_steps,
                "step_budget": iteration_budget,
                "budget_exhausted": budget_exhausted,
                "used_failure_extension": growth_steps > self.config.max_iterations,
                "failure_extension_steps": max(0, growth_steps - self.config.max_iterations),
                "projection": projection_stats,
                "cuda_event_profile_phase": (
                    self.config.profile_cuda_event_phase if profile_cuda_events else "off"
                ),
                "shortcut": shortcut_summary,
            },
        )
        if interpolate_waypoints and any(state.group_solved):
            result.interpolated_waypoints = self._interpolate_paths(
                state.paths,
                interpolation_steps,
                interpolation_type,
            )
            if validate_interpolated_trajectory:
                samples = result.interpolated_waypoints.reshape(
                    request.batch_size * interpolation_steps,
                    self.resources.action_dim,
                )
                sample_env = request.env_query_idx.repeat_interleave(interpolation_steps)
                sample_goal = request.goal_query_idx.repeat_interleave(interpolation_steps)
                valid = (
                    self.resources.check_samples_feasibility(
                        samples,
                        clone_output=False,
                        env_query_idx=sample_env if request.explicit_env else None,
                        goal_query_idx=sample_goal if request.route_goals else None,
                    )
                    .reshape(request.batch_size, interpolation_steps)
                    .all(dim=1)
                )
                search_success = result.success.clone()
                result.success &= valid
                invalidated = search_success & ~result.success
                if bool(invalidated.any().item()):
                    result.path_length[invalidated] = torch.inf
                    invalidated_host = invalidated.cpu().tolist()
                    for lane_index, failed_validation in enumerate(invalidated_host):
                        if failed_validation:
                            state.query_debug[lane_index]["interpolated_validation"] = "failed"
                    result.debug_info["interpolated_validation_failed_lanes"] = [
                        lane_index
                        for lane_index, failed_validation in enumerate(invalidated_host)
                        if failed_validation
                    ]

                final_group_solved = [False] * request.group_count
                for lane_success, group_index in zip(
                    result.success.cpu().tolist(), state.group_ids_host
                ):
                    if lane_success:
                        final_group_solved[group_index] = True
                state.group_solved[:] = final_group_solved
        profile_checkpoint("interpolate_validate")
        result.plan_waypoints = self._clone_paths(state.paths)
        result.solve_time = timer.stop(final_interval="finalize" if profile_cuda_events else None)
        if profile_cuda_events:
            component_times = timer.intervals
            component_times.update(captured_component_times)
            result.component_times = component_times
        elif shortcut_start_event is not None and shortcut_end_event is not None:
            # The outer timer stop synchronized a later event on the same
            # stream, so this elapsed-time query does not add a second sync.
            result.component_times = {
                "shortcut": shortcut_start_event.elapsed_time(shortcut_end_event) / 1000.0
            }
        return result

    def find_path(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        interpolate_waypoints: bool = True,
        interpolation_steps: int = 100,
        interpolation_type: TrajInterpolationType = TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory: bool = True,
        *,
        env_query_idx: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        query_group_id: Optional[torch.Tensor] = None,
        group_count: Optional[int] = None,
    ) -> GraphPlannerResult:
        """Plan paired starts and goals with optional completion-owner metadata."""
        return self.find_path_batched(
            x_start,
            x_goal,
            env_query_idx=env_query_idx,
            query_mask=query_mask,
            query_group_id=query_group_id,
            group_count=group_count,
            interpolate_waypoints=interpolate_waypoints,
            interpolation_steps=interpolation_steps,
            interpolation_type=interpolation_type,
            validate_interpolated_trajectory=validate_interpolated_trajectory,
        )

    def reset_buffer(self) -> None:
        """Retain the graph-planner lifecycle API; RRT trees are query-local."""

    def initialize_query_shape(self, batch_size: int = 1) -> None:
        """Allocate one query bucket before constraint-update graph capture."""
        if batch_size < 1:
            log_and_raise("batch_size must be positive")
        self._get_batched_workspace(batch_size)

    def reset_seed(self) -> None:
        """Reset the primary sample stream; informed latents are immutable."""
        self.resources.reset_seed()

    def _get_projector(
        self,
        config: Optional[RRTSampleProjectionConfig] = None,
    ) -> RRTSampleProjector:
        """Return the one lazily configured optional projection operator."""
        resolved_config = (
            config
            or self.config.sample_projection
            or RRTSampleProjectionConfig(
                projection_slots=4,
                use_cuda_graph=True,
            )
        )
        if self._sample_projector is None or self._sample_projection_config != resolved_config:
            if self._sample_projector is not None:
                self._sample_projector.reset_cuda_graph()
            self._sample_projector = RRTSampleProjector(
                self.resources,
                resolved_config,
            )
            self._sample_projection_config = resolved_config
            if (
                self._constraint_goal_tool_poses is not None
                or self._constraint_tool_relative_transform_target is not None
            ):
                projection_batch = resolved_config.projection_slots * (
                    len(resolved_config.line_search_scales) - 1
                )
                self._apply_constraints(
                    self.resources.auxiliary_rollout,
                    projection_batch,
                )
        return self._sample_projector

    def project_samples(
        self,
        samples: torch.Tensor,
        config: Optional[RRTSampleProjectionConfig] = None,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> RRTSampleProjectionResult:
        """Project samples into their routed collision-free inequality sets.

        Projected samples still require exact steering and generic-rollout edge
        validation before they may be appended to a tree. The default planner
        never calls this method during growth; ``sample_projection`` explicitly
        opts into the ambient-proposal projection stage.
        """
        return self._get_projector(config).project(
            samples,
            env_query_idx=env_query_idx,
            goal_query_idx=goal_query_idx,
            eligible_mask=eligible_mask,
        )

    def update_tool_pose_constraint(
        self,
        goal_tool_poses: GoalToolPose,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ) -> None:
        """Atomically update the tool-pose path constraint for projected RRT.

        Every RRT state is evaluated as an independent horizon-one rollout.
        Path fields from ``tool_pose_criteria`` are therefore copied to
        both terminal and path fields before routing them to projection,
        feasibility, and growth rollouts. The configuration-space goal remains
        responsible for enforcing the terminal pose.

        Args:
            goal_tool_poses: One pose goal per query group, with goalset size
                one. A singleton batch may be shared by every group.
            tool_pose_criteria: Criteria for every configured tool frame. The
                path projectors, scales, and tolerances define the path manifold.
        """
        self._validate_tool_pose_constraint_goal(goal_tool_poses)
        self._validate_tool_pose_constraint_criteria(tool_pose_criteria)

        if self._constraint_goal_tool_poses is None:
            # The first goal adds rollout buffers that must exist before graph capture.
            self.reset_cuda_graph()
            self._constraint_goal_tool_poses = goal_tool_poses.clone()
            self._constraint_tool_pose_criteria = {
                frame: tool_pose_criteria[frame].as_path_constraint()
                for frame in self.resources.kinematics.tool_frames
            }
            self._apply_constraints_to_all_rollouts()
            return

        self.update_tool_pose_constraint_goal(goal_tool_poses)
        self.update_tool_pose_constraint_criteria(tool_pose_criteria)

    def _validate_tool_pose_constraint_goal(self, goal_tool_poses: GoalToolPose) -> None:
        """Validate the fixed-shape RRT tool-pose target contract."""
        constraint_cfg = self.resources.config.rollout_config.constraint_cfg
        if constraint_cfg is None or constraint_cfg.tool_pose_cfg is None:
            log_and_raise("Projected RRT requires a strict tool-pose constraint rollout")
        if goal_tool_poses.batch_size < 1 or goal_tool_poses.num_goalset != 1:
            log_and_raise("RRT constraint goal requires batch_size>=1 and num_goalset=1")
        if goal_tool_poses.tool_frames != self.resources.kinematics.tool_frames:
            log_and_raise("RRT constraint goal tool frames must match graph-planner tool frames")
        if (
            self._constraint_goal_tool_poses is not None
            and goal_tool_poses.shape != self._constraint_goal_tool_poses.shape
        ):
            log_and_raise("RRT constraint goal shape cannot change after installation")

    def _validate_tool_pose_constraint_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ) -> None:
        """Validate one complete fixed-shape tool-pose criteria update."""
        if set(tool_pose_criteria) != set(self.resources.kinematics.tool_frames):
            log_and_raise("RRT tool-pose criteria must match graph-planner tool frames")
        if not all(isinstance(value, ToolPoseCriteria) for value in tool_pose_criteria.values()):
            log_and_raise("RRT tool-pose criteria values must be ToolPoseCriteria")

    def update_tool_pose_constraint_goal(self, goal_tool_poses: GoalToolPose) -> None:
        """Copy target poses into the persistent projected-RRT goal buffers.

        This hot path preserves every tensor address used by existing CUDA
        graphs and does not touch criteria, routing indices, or rollout state.
        """
        self._validate_tool_pose_constraint_goal(goal_tool_poses)
        if self._constraint_goal_tool_poses is None:
            log_and_raise("Install tool-pose criteria before updating only its goal")
        self._constraint_goal_tool_poses.copy_(goal_tool_poses)

    def update_tool_pose_constraint_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ) -> None:
        """Update arbitrary fixed-shape pose projectors on the existing RRT instance."""
        self._validate_tool_pose_constraint_criteria(tool_pose_criteria)
        if self._constraint_tool_pose_criteria is None:
            log_and_raise("Install a tool-pose constraint before updating only its criteria")
        for frame in self.resources.kinematics.tool_frames:
            self._constraint_tool_pose_criteria[frame].copy_path_constraint_(
                tool_pose_criteria[frame]
            )
        self._apply_tool_pose_criteria_to_all_rollouts()

    def disable_tool_pose_constraint(self) -> None:
        """Disable the installed tool-pose constraint without changing tensor shapes."""
        if self._constraint_tool_pose_criteria is None:
            return
        for criteria in self._constraint_tool_pose_criteria.values():
            criteria.terminal_position_projector.zero_()
            criteria.terminal_rotation_projector.zero_()
            criteria.path_position_projector.zero_()
            criteria.path_rotation_projector.zero_()
        self._apply_tool_pose_criteria_to_all_rollouts()

    def update_tool_relative_transform_constraint(
        self,
        target: ToolRelativeTransformTarget,
        criteria: Dict[str, ToolRelativeTransformCriteria],
    ) -> None:
        """Update one-reference relative tool constraints for projected RRT."""
        constraint_cfg = self.resources.config.rollout_config.constraint_cfg
        if constraint_cfg is None or constraint_cfg.tool_relative_transform_cfg is None:
            log_and_raise(
                "Projected RRT requires tool_relative_transform_cfg in its constraint rollout"
            )
        relative_cfg = constraint_cfg.tool_relative_transform_cfg
        tool_frames = self.resources.kinematics.tool_frames
        if target.num_target_groups != 1:
            log_and_raise("RRT relative-transform target requires one shared target group")
        if target.tool_frames != tool_frames:
            log_and_raise(
                "RRT relative-transform target frames must match graph-planner tool frames"
            )
        if target.reference_tool_frame != relative_cfg.reference_tool_frame:
            log_and_raise(
                "RRT relative-transform target reference must match the configured reference"
            )
        if set(criteria) != set(tool_frames):
            log_and_raise("RRT relative-transform criteria must match graph-planner tool frames")
        if not all(
            isinstance(value, ToolRelativeTransformCriteria) for value in criteria.values()
        ):
            log_and_raise(
                "RRT relative-transform criteria values must be ToolRelativeTransformCriteria"
            )
        if criteria[target.reference_tool_frame].enabled:
            log_and_raise("RRT relative-transform reference criterion must be disabled")

        if self._constraint_tool_relative_transform_target is None:
            self.reset_cuda_graph()
        self._constraint_tool_relative_transform_target = target.clone()
        self._constraint_tool_relative_transform_criteria = {
            frame: criteria[frame].clone() for frame in tool_frames
        }
        self._apply_constraints_to_all_rollouts()

    def disable_tool_relative_transform_constraint(self) -> None:
        """Disable installed relative constraints without changing captured shapes."""
        if self._constraint_tool_relative_transform_criteria is None:
            return
        for criteria in self._constraint_tool_relative_transform_criteria.values():
            criteria.position_projector.zero_()
            criteria.rotation_projector.zero_()
        self._apply_constraints_to_all_rollouts()

    def reset_cuda_graph(self) -> None:
        """Release executor and CUDA graph state owned by this planner and its resources."""
        if self._sample_projector is not None:
            self._sample_projector.reset_cuda_graph()
        for executor in self._informed_proposal_executors.values():
            executor.reset()
        self._informed_proposal_executors.clear()
        for workspace in self._workspaces.values():
            for executor in workspace.preprocessing_executors.values():
                executor.reset()
            workspace.preprocessing_executors.clear()
            workspace.path_reconstruction_executor.reset()
            if workspace.path_shortcutter is not None:
                workspace.path_shortcutter.reset_cuda_graph()
            for executor in workspace.finalization_executors.values():
                executor.reset()
            workspace.finalization_executors.clear()
            if workspace.finalization_rollout is not None:
                workspace.finalization_rollout.reset_cuda_graph()
            for growth_step in workspace.growth_steps:
                growth_step.reset()
                growth_step.rollout.reset_cuda_graph()
        for connector in getattr(self, "_direct_connectors", {}).values():
            connector.reset_cuda_graph()
        self.resources.reset_cuda_graph()

    def get_all_rollout_instances(self):
        """Expose resource-owned rollouts for lifecycle and world-update integration."""
        batched_rollouts = [
            growth_step.rollout
            for workspace in self._workspaces.values()
            for growth_step in workspace.growth_steps
        ]
        finalization_rollouts = [
            getattr(workspace, "finalization_rollout", None)
            for workspace in self._workspaces.values()
            if getattr(workspace, "finalization_rollout", None) is not None
        ]
        direct_rollouts = [
            connector.rollout for connector in getattr(self, "_direct_connectors", {}).values()
        ]
        return (
            self.resources.get_all_rollout_instances()
            + direct_rollouts
            + batched_rollouts
            + finalization_rollouts
        )

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        """Update one link in enabled dynamics models; skip geometric RRT transitions."""
        for rollout in self.get_all_rollout_instances():
            for model in (rollout.transition_model, rollout.metrics_transition_model):
                if model.compute_inverse_dynamics:
                    model.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self,
        link_properties: Dict[str, Dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        """Update links in enabled dynamics models; skip geometric RRT transitions."""
        for rollout in self.get_all_rollout_instances():
            for model in (rollout.transition_model, rollout.metrics_transition_model):
                if model.compute_inverse_dynamics:
                    model.update_links_inertial(link_properties)

    def _initialize_growth_steps(
        self,
        actions: torch.Tensor,
        lane_count: int,
    ) -> None:
        """Initialize both canonical growth-step orientations for one batch bucket."""
        if lane_count <= 0:
            return
        if lane_count > self.config.max_batched_lanes:
            return
        workspace = self._get_batched_workspace(lane_count)
        if all(growth_step.is_initialized for growth_step in workspace.growth_steps):
            return
        bucket = workspace.batch_size
        sample_count = self.config.samples_per_iteration
        if actions.shape[0] < sample_count + 2:
            log_and_raise("BiRRT growth-step warmup received too few feasible actions")
        for tree, root in (
            (workspace.start_tree, actions[0]),
            (workspace.goal_tree, actions[1]),
        ):
            tree.nodes[:, 0].copy_(root.view(1, -1).expand(bucket, -1))
            tree.parents[:, 0].fill_(-1)
            tree.counts.fill_(1)
        samples = (
            actions[2 : sample_count + 2]
            .view(1, sample_count, self.resources.action_dim)
            .expand(bucket, -1, -1)
            .contiguous()
        )
        active = torch.zeros(bucket, device=actions.device, dtype=torch.bool)
        active[:lane_count] = True
        env_query_idx = torch.zeros(bucket, device=actions.device, dtype=torch.int32)
        goal_query_idx = torch.zeros_like(env_query_idx)
        workspace.growth_steps[0].initialize(
            samples,
            active,
            env_query_idx,
            goal_query_idx,
        )
        workspace.growth_steps[1].initialize(
            samples.flip(1),
            active,
            env_query_idx,
            goal_query_idx,
        )

    def _generate_warmup_actions(self, action_count: int) -> torch.Tensor:
        """Return a fixed-size bounded batch even when rejection leaves too few samples."""
        actions = self.resources.uniform_sampler.generate_feasible_action_samples(action_count)
        missing_count = action_count - actions.shape[0]
        if missing_count <= 0:
            return actions[:action_count]
        fallback = self.resources.uniform_sampler.generate_action_samples(
            missing_count,
            bounded=True,
        )
        return torch.cat((actions, fallback), dim=0)

    def warmup(self, num_warmup_iterations: int = 10, max_batch_size: int = 1) -> None:
        """Warm collision, sampling, growth-step executors, and shortcut kernels."""
        if (
            not isinstance(max_batch_size, int)
            or isinstance(max_batch_size, bool)
            or not 1 <= max_batch_size <= self.config.max_batched_lanes
        ):
            log_and_raise(
                f"max_batch_size must be an integer in [1, {self.config.max_batched_lanes}]"
            )
        action_count = max(
            num_warmup_iterations * 2 * max_batch_size,
            self.config.samples_per_iteration + 2,
            3,
        )
        if self._sample_projector is not None and self._sample_projector.config.use_cuda_graph:
            action_count = max(
                action_count,
                self._sample_projector.config.projection_slots,
            )
        actions = self._generate_warmup_actions(action_count)
        maximum_lane_count = min(max_batch_size, self.config.max_batched_lanes)
        lane_counts = {max_batch_size}
        lane_bucket = 1
        while lane_bucket < maximum_lane_count:
            lane_bucket *= 2
            lane_counts.add(lane_bucket)

        if self.config.sampling_policy == "mixed_ellipsoid":
            for lane_count in sorted(lane_counts):
                self._sample_mixed_batched(
                    lane_count,
                    actions[0:1].expand(lane_count, -1),
                    actions[1:2].expand(lane_count, -1),
                    0,
                )
            if actions.is_cuda:
                torch.cuda.synchronize(actions.device)

        if (
            self._sample_projector is not None
            and self._sample_projector.config.use_cuda_graph
            and actions.is_cuda
        ):
            projection_actions = actions[: self._sample_projector.config.projection_slots]
            projection_env = (
                torch.zeros(
                    projection_actions.shape[0],
                    device=actions.device,
                    dtype=torch.int32,
                )
                if self.config.sample_projection is not None
                else None
            )
            self._sample_projector.warmup_cuda_graph(projection_actions, projection_env)
        for lane_count in sorted(lane_counts):
            self._initialize_growth_steps(actions, lane_count)

        for iteration in range(num_warmup_iterations):
            start = iteration * max_batch_size
            goal = (num_warmup_iterations + iteration) * max_batch_size
            self.find_path(
                actions[start : start + max_batch_size],
                actions[goal : goal + max_batch_size],
                interpolate_waypoints=False,
                validate_interpolated_trajectory=False,
            )
        self.reset_seed()

    @property
    def kinematics(self):
        """Return the exact kinematics instance used by collision checking."""
        return self.resources.kinematics

    @property
    def action_dim(self) -> int:
        """Return robot configuration-space dimensionality."""
        return self.resources.action_dim

    @property
    def joint_names(self) -> List[str]:
        """Return ordered robot joint names."""
        return self.resources.joint_names
