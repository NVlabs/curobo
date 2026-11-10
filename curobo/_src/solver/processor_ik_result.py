# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape-managed Warp result processing for inverse kinematics."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Sequence

# Third Party
import torch
import torch.autograd.profiler as profiler
import warp as wp

# CuRobo
from curobo import runtime as curobo_runtime
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, init_warp

IK_FAILURE_PENALTY = wp.constant(1.0e16)


@dataclass(frozen=True)
class _IKConvergenceLayout:
    """Indices into a stable IK convergence metric collection."""

    position_indices: tuple[int, ...]
    orientation_indices: tuple[int, ...]
    goalset_indices: tuple[int, ...]
    cspace_indices: tuple[int, ...]


@wp.kernel(enable_backward=False)
def _reduce_ik_result_metrics_kernel(
    position_values: wp.array(dtype=wp.float32),
    orientation_values: wp.array(dtype=wp.float32),
    cspace_values: wp.array(dtype=wp.float32),
    feasible: wp.array(dtype=wp.bool),
    problem_count: wp.int32,
    position_count: wp.int32,
    orientation_count: wp.int32,
    cspace_count: wp.int32,
    position_tolerance: wp.float32,
    orientation_tolerance: wp.float32,
    position_rank_weight: wp.float32,
    orientation_rank_weight: wp.float32,
    cspace_rank_weight: wp.float32,
    success_requires_convergence: wp.bool,
    metric_summary: wp.array(dtype=wp.float32),
    metric_flags: wp.array(dtype=wp.bool),
):
    """Reduce per-link errors and form one scalar ranking cost per candidate."""
    problem_index = wp.tid()

    position_offset = problem_index * position_count
    position_max = position_values[position_offset]
    position_sum = wp.float32(0.0)
    for value_index in range(position_count):
        value = position_values[position_offset + value_index]
        position_max = wp.max(position_max, value)
        position_sum += value

    orientation_offset = problem_index * orientation_count
    orientation_max = orientation_values[orientation_offset]
    orientation_sum = wp.float32(0.0)
    for value_index in range(orientation_count):
        value = orientation_values[orientation_offset + value_index]
        orientation_max = wp.max(orientation_max, value)
        orientation_sum += value

    cspace_sum = wp.float32(0.0)
    cspace_offset = problem_index * cspace_count
    for value_index in range(cspace_count):
        cspace_sum += cspace_values[cspace_offset + value_index]

    is_feasible = feasible[problem_index]
    is_converged = position_max < position_tolerance and orientation_max < orientation_tolerance
    is_success = is_feasible
    if success_requires_convergence:
        is_success = is_feasible and is_converged

    rank_cost = (
        position_rank_weight * position_sum
        + orientation_rank_weight * orientation_sum
        + cspace_rank_weight * cspace_sum
    )
    if not is_success:
        rank_cost += wp.float32(IK_FAILURE_PENALTY)

    metric_summary[problem_index] = position_max
    metric_summary[problem_count + problem_index] = orientation_max
    metric_summary[2 * problem_count + problem_index] = rank_cost
    metric_flags[problem_index] = is_success
    metric_flags[problem_count + problem_index] = is_feasible


@wp.kernel(enable_backward=False)
def _gather_ik_result_kernel(
    relative_indices: wp.array(dtype=wp.int64),
    optimized_position: wp.array(dtype=wp.float32),
    metric_summary: wp.array(dtype=wp.float32),
    metric_flags: wp.array(dtype=wp.bool),
    goalset_index: wp.array(dtype=wp.int32),
    tool_position: wp.array(dtype=wp.float32),
    tool_quaternion: wp.array(dtype=wp.float32),
    robot_spheres: wp.array(dtype=wp.float32),
    robot_com: wp.array(dtype=wp.float32),
    full_to_active: wp.array(dtype=wp.int32),
    locked_position: wp.array(dtype=wp.float32),
    problem_count: wp.int32,
    num_seeds: wp.int32,
    return_seeds: wp.int32,
    action_dim: wp.int32,
    num_links: wp.int32,
    num_spheres: wp.int32,
    num_com: wp.int32,
    full_dim: wp.int32,
    solution: wp.array(dtype=wp.float32),
    position_error: wp.array(dtype=wp.float32),
    rotation_error: wp.array(dtype=wp.float32),
    selected_flags: wp.array(dtype=wp.bool),
    selected_goalset_index: wp.array(dtype=wp.int32),
    selected_tool_position: wp.array(dtype=wp.float32),
    selected_tool_quaternion: wp.array(dtype=wp.float32),
    selected_robot_spheres: wp.array(dtype=wp.float32),
    selected_robot_com: wp.array(dtype=wp.float32),
    active_derivatives: wp.array(dtype=wp.float32),
    full_position: wp.array(dtype=wp.float32),
    full_derivatives: wp.array(dtype=wp.float32),
    absolute_indices: wp.array(dtype=wp.int64),
):
    """Gather all selected result fields and reconstruct full-DOF positions."""
    selected_index, lane = wp.tid()
    batch_index = selected_index / return_seeds
    relative_index = wp.int32(relative_indices[selected_index])
    source_index = batch_index * num_seeds + relative_index
    selected_count = problem_count / num_seeds * return_seeds

    if lane == 0:
        position_error[selected_index] = metric_summary[source_index]
        rotation_error[selected_index] = metric_summary[problem_count + source_index]
        selected_flags[selected_index] = metric_flags[source_index]
        selected_flags[selected_count + selected_index] = metric_flags[
            problem_count + source_index
        ]
        absolute_indices[selected_index] = wp.int64(source_index)

    if lane < action_dim:
        solution[selected_index * action_dim + lane] = optimized_position[
            source_index * action_dim + lane
        ]
        active_derivatives[selected_index * action_dim + lane] = wp.float32(0.0)
        active_derivatives[selected_count * action_dim + selected_index * action_dim + lane] = (
            wp.float32(0.0)
        )
        active_derivatives[
            2 * selected_count * action_dim + selected_index * action_dim + lane
        ] = wp.float32(0.0)

    if lane < num_links:
        selected_goalset_index[selected_index * num_links + lane] = goalset_index[
            source_index * num_links + lane
        ]

    tool_position_count = num_links * 3
    if lane < tool_position_count:
        selected_tool_position[selected_index * tool_position_count + lane] = tool_position[
            source_index * tool_position_count + lane
        ]

    tool_quaternion_count = num_links * 4
    if lane < tool_quaternion_count:
        selected_tool_quaternion[selected_index * tool_quaternion_count + lane] = tool_quaternion[
            source_index * tool_quaternion_count + lane
        ]

    sphere_value_count = num_spheres * 4
    if lane < sphere_value_count:
        selected_robot_spheres[selected_index * sphere_value_count + lane] = robot_spheres[
            source_index * sphere_value_count + lane
        ]

    com_value_count = num_com * 4
    if lane < com_value_count:
        selected_robot_com[selected_index * com_value_count + lane] = robot_com[
            source_index * com_value_count + lane
        ]

    if lane < full_dim:
        active_index = full_to_active[lane]
        value = locked_position[lane]
        if active_index >= 0:
            value = optimized_position[source_index * action_dim + active_index]
        full_position[selected_index * full_dim + lane] = value
        full_derivatives[selected_index * full_dim + lane] = wp.float32(0.0)
        full_derivatives[selected_count * full_dim + selected_index * full_dim + lane] = (
            wp.float32(0.0)
        )
        full_derivatives[2 * selected_count * full_dim + selected_index * full_dim + lane] = (
            wp.float32(0.0)
        )


@dataclass(frozen=True)
class IKResultProcessorConfig:
    """Static configuration for IK result reduction and assembly."""

    #: Device used by input and output tensors.
    device_cfg: DeviceCfg

    #: Number of optimized robot degrees of freedom.
    action_dim: int

    #: Ordered active joint names.
    joint_names: Sequence[str]

    #: Ordered active and locked joint names returned in the full state.
    full_joint_names: Sequence[str]

    #: Names and values of joints excluded from optimization.
    locked_joint_names: Sequence[str]
    locked_joint_position: Optional[torch.Tensor]

    #: Pose-error thresholds used to determine success.
    position_tolerance: float
    orientation_tolerance: float

    #: Scalar weights applied to position, orientation, and start-cspace
    #: convergence values when ranking successful seeds.
    position_rank_weight: float
    orientation_rank_weight: float
    cspace_rank_weight: float

    #: Whether feasibility alone or feasibility and convergence define success.
    success_requires_convergence: bool

    #: Capture result reduction, ranking, and gathering in a CUDA graph.
    use_cuda_graph: bool = True


class WarpIKResultProcessor:
    """Reduce, rank, and gather IK results with Warp, Torch top-k, and CUDA graphs.

    Shape-dependent tensors and their Warp views are created only by
    :meth:`update_batch_size`. Returned result tensors borrow these workspaces
    and remain valid until the next result is processed with the same object.
    """

    def __init__(self, config: IKResultProcessorConfig) -> None:
        """Create static full-joint mappings without allocating batch workspaces."""
        init_warp(device_cfg=config.device_cfg)
        self._config = config
        self._use_cuda_graph = config.use_cuda_graph and curobo_runtime.cuda_graphs
        self._batch_size = 0
        self._num_seeds = 0
        self._return_seeds = 0
        self._num_links = 0
        self._num_spheres = 0
        self._num_com = 0
        self._tool_frames: tuple[str, ...] = ()
        self._max_gather_width = 0

        active_lookup = {name: index for index, name in enumerate(config.joint_names)}
        locked_values = (
            []
            if config.locked_joint_position is None
            else config.locked_joint_position.detach().view(-1).cpu().tolist()
        )
        locked_lookup = {
            name: locked_values[index] for index, name in enumerate(config.locked_joint_names)
        }
        full_to_active = []
        full_locked_position = []
        for joint_name in config.full_joint_names:
            if joint_name in active_lookup:
                full_to_active.append(active_lookup[joint_name])
                full_locked_position.append(0.0)
            elif joint_name in locked_lookup:
                full_to_active.append(-1)
                full_locked_position.append(locked_lookup[joint_name])
            else:
                log_and_raise(f"No active or locked value found for joint {joint_name}")

        device = config.device_cfg.device
        self._full_to_active = torch.tensor(full_to_active, device=device, dtype=torch.int32)
        self._locked_position = torch.tensor(
            full_locked_position, device=device, dtype=torch.float32
        )
        self._dummy_float = torch.empty((1,), device=device, dtype=torch.float32)
        self._wp_full_to_active = wp.from_torch(
            self._full_to_active, requires_grad=False, return_ctype=True
        )
        self._wp_locked_position = wp.from_torch(
            self._locked_position, requires_grad=False, return_ctype=True
        )

        self._metric_summary: Optional[torch.Tensor] = None
        self._metric_flags: Optional[torch.Tensor] = None
        self._topk_values: Optional[torch.Tensor] = None
        self._relative_indices: Optional[torch.Tensor] = None
        self._absolute_indices: Optional[torch.Tensor] = None
        self._solution: Optional[torch.Tensor] = None
        self._position_error: Optional[torch.Tensor] = None
        self._rotation_error: Optional[torch.Tensor] = None
        self._selected_flags: Optional[torch.Tensor] = None
        self._goalset_index: Optional[torch.Tensor] = None
        self._tool_position: Optional[torch.Tensor] = None
        self._tool_quaternion: Optional[torch.Tensor] = None
        self._robot_spheres: Optional[torch.Tensor] = None
        self._robot_com: Optional[torch.Tensor] = None
        self._active_derivatives: Optional[torch.Tensor] = None
        self._full_position: Optional[torch.Tensor] = None
        self._full_derivatives: Optional[torch.Tensor] = None
        self._solution_state: Optional[RobotState] = None
        self._js_solution: Optional[JointState] = None
        self._optimized_position: Optional[torch.Tensor] = None
        self._metrics_result: Optional[RolloutMetrics] = None
        self._convergence_names: tuple[str, ...] = ()
        self._convergence_layout: Optional[_IKConvergenceLayout] = None
        self._position_count = 0
        self._orientation_count = 0
        self._cspace_count = 0
        self._result_graph: Optional[GraphExecutor] = None
        self._wp_inputs: dict[str, object] = {}
        self._wp_outputs: dict[str, object] = {}

    @property
    def is_shape_initialized(self) -> bool:
        """Return whether shape-dependent workspaces have been created."""
        return self._solution is not None

    @property
    def is_cuda_graph_initialized(self) -> bool:
        """Return whether the result-processing CUDA graph has been captured."""
        return self._result_graph is not None and self._result_graph.is_initialized

    @profiler.record_function("ik_result_processor/reset_shape")
    def reset_shape(self) -> None:
        """Invalidate every shape-dependent output tensor and Warp view."""
        self.reset_cuda_graph()
        self._batch_size = 0
        self._num_seeds = 0
        self._return_seeds = 0
        self._num_links = 0
        self._num_spheres = 0
        self._num_com = 0
        self._tool_frames = ()
        self._max_gather_width = 0
        self._metric_summary = None
        self._metric_flags = None
        self._topk_values = None
        self._relative_indices = None
        self._absolute_indices = None
        self._solution = None
        self._position_error = None
        self._rotation_error = None
        self._selected_flags = None
        self._goalset_index = None
        self._tool_position = None
        self._tool_quaternion = None
        self._robot_spheres = None
        self._robot_com = None
        self._active_derivatives = None
        self._full_position = None
        self._full_derivatives = None
        self._solution_state = None
        self._js_solution = None
        self._wp_outputs = {}

    @profiler.record_function("ik_result_processor/reset_cuda_graph")
    def reset_cuda_graph(self) -> None:
        """Invalidate the captured result graph and its bound metric tensors."""
        if self._result_graph is not None:
            self._result_graph.reset()
        self._result_graph = None
        self._optimized_position = None
        self._metrics_result = None
        self._position_count = 0
        self._orientation_count = 0
        self._cspace_count = 0
        self._wp_inputs = {}

    @profiler.record_function("ik_result_processor/update_batch_size")
    def update_batch_size(
        self,
        batch_size: int,
        num_seeds: int,
        return_seeds: int,
        tool_frames: Sequence[str],
        num_spheres: int,
        num_com: int,
    ) -> None:
        """Create all workspaces for one logical IK result shape."""
        num_links = len(tool_frames)
        tool_frame_names = tuple(tool_frames)
        shape = (
            batch_size,
            num_seeds,
            return_seeds,
            tool_frame_names,
            num_spheres,
            num_com,
        )
        current_shape = (
            self._batch_size,
            self._num_seeds,
            self._return_seeds,
            self._tool_frames,
            self._num_spheres,
            self._num_com,
        )
        if self.is_shape_initialized and shape == current_shape:
            return
        if batch_size < 1 or num_seeds < 1 or return_seeds < 1:
            log_and_raise("batch_size, num_seeds, and return_seeds must all be positive")
        if return_seeds > num_seeds:
            log_and_raise("return_seeds cannot exceed num_seeds")
        if num_links < 1 or num_spheres < 1 or num_com < 1:
            log_and_raise("IK kinematics outputs must contain links, spheres, and COM values")

        self.reset_shape()
        self._batch_size = batch_size
        self._num_seeds = num_seeds
        self._return_seeds = return_seeds
        self._num_links = num_links
        self._num_spheres = num_spheres
        self._num_com = num_com
        self._tool_frames = tool_frame_names
        selected_count = batch_size * return_seeds
        problem_count = batch_size * num_seeds
        action_dim = self._config.action_dim
        full_dim = len(self._config.full_joint_names)
        self._max_gather_width = max(
            action_dim,
            num_links * 4,
            num_spheres * 4,
            num_com * 4,
            full_dim,
        )

        device = self._config.device_cfg.device
        self._metric_summary = torch.empty((3, problem_count), device=device, dtype=torch.float32)
        self._metric_flags = torch.empty((2, problem_count), device=device, dtype=torch.bool)
        self._topk_values = torch.empty(
            (batch_size, return_seeds), device=device, dtype=torch.float32
        )
        self._relative_indices = torch.empty(
            (batch_size, return_seeds), device=device, dtype=torch.long
        )
        self._absolute_indices = torch.empty_like(self._relative_indices)
        self._solution = torch.empty(
            (batch_size, return_seeds, action_dim), device=device, dtype=torch.float32
        )
        self._position_error = torch.empty(
            (batch_size, return_seeds), device=device, dtype=torch.float32
        )
        self._rotation_error = torch.empty_like(self._position_error)
        self._selected_flags = torch.empty((2, selected_count), device=device, dtype=torch.bool)
        self._goalset_index = torch.empty(
            (batch_size, return_seeds, num_links), device=device, dtype=torch.int32
        )
        self._tool_position = torch.empty(
            (batch_size, return_seeds, num_links, 3),
            device=device,
            dtype=torch.float32,
        )
        self._tool_quaternion = torch.empty(
            (batch_size, return_seeds, num_links, 4),
            device=device,
            dtype=torch.float32,
        )
        self._robot_spheres = torch.empty(
            (batch_size, return_seeds, num_spheres, 4),
            device=device,
            dtype=torch.float32,
        )
        self._robot_com = torch.empty(
            (batch_size, return_seeds, num_com, 4), device=device, dtype=torch.float32
        )
        self._active_derivatives = torch.empty(
            (3, batch_size, return_seeds, action_dim),
            device=device,
            dtype=torch.float32,
        )
        self._full_position = torch.empty(
            (batch_size, return_seeds, full_dim), device=device, dtype=torch.float32
        )
        self._full_derivatives = torch.empty(
            (3, batch_size, return_seeds, full_dim),
            device=device,
            dtype=torch.float32,
        )

        active_joint_state = JointState(
            position=self._solution,
            velocity=self._active_derivatives[0],
            acceleration=self._active_derivatives[1],
            jerk=self._active_derivatives[2],
            joint_names=list(self._config.joint_names),
        )
        self._solution_state = RobotState(
            joint_state=active_joint_state,
            cuda_robot_model_state=KinematicsState(
                tool_poses=ToolPose(
                    tool_frames=list(tool_frames),
                    position=self._tool_position,
                    quaternion=self._tool_quaternion,
                ),
                robot_spheres=self._robot_spheres,
                robot_com=self._robot_com,
            ),
        )
        self._js_solution = JointState(
            position=self._full_position,
            velocity=self._full_derivatives[0],
            acceleration=self._full_derivatives[1],
            jerk=self._full_derivatives[2],
            joint_names=list(self._config.full_joint_names),
        )

        output_tensors = {
            "metric_summary": self._metric_summary,
            "metric_flags": self._metric_flags,
            "relative_indices": self._relative_indices,
            "absolute_indices": self._absolute_indices,
            "solution": self._solution,
            "position_error": self._position_error,
            "rotation_error": self._rotation_error,
            "selected_flags": self._selected_flags,
            "goalset_index": self._goalset_index,
            "tool_position": self._tool_position,
            "tool_quaternion": self._tool_quaternion,
            "robot_spheres": self._robot_spheres,
            "robot_com": self._robot_com,
            "active_derivatives": self._active_derivatives,
            "full_position": self._full_position,
            "full_derivatives": self._full_derivatives,
        }
        self._wp_outputs = {
            name: wp.from_torch(tensor.reshape(-1), requires_grad=False, return_ctype=True)
            for name, tensor in output_tensors.items()
        }

    def _get_convergence_layout(
        self,
        metrics_result: RolloutMetrics,
    ) -> Optional[_IKConvergenceLayout]:
        """Return cached field indices for a stable convergence collection."""
        names = tuple(metrics_result.convergence.names)
        if names != self._convergence_names:
            self._convergence_names = names
            self._convergence_layout = _IKConvergenceLayout(
                position_indices=tuple(
                    index for index, name in enumerate(names) if "position_tolerance" in name
                ),
                orientation_indices=tuple(
                    index for index, name in enumerate(names) if "orientation_tolerance" in name
                ),
                goalset_indices=tuple(
                    index for index, name in enumerate(names) if "goalset_index" in name
                ),
                cspace_indices=tuple(
                    index
                    for index, name in enumerate(names)
                    if "start_cspace_dist_tolerance" in name
                ),
            )
        layout = self._convergence_layout
        if layout is None or not layout.position_indices or not layout.orientation_indices:
            return None
        return layout

    @staticmethod
    def _get_convergence_values(
        metrics_result: RolloutMetrics,
        indices: tuple[int, ...],
        problem_count: int,
    ) -> torch.Tensor:
        """Return selected convergence fields flattened after the problem dimension."""
        values = metrics_result.convergence.values
        if len(indices) == 1:
            selected = values[indices[0]]
        else:
            selected = torch.cat([values[index] for index in indices], dim=-1)
        return selected.view(problem_count, -1)

    def _get_feasibility(
        self,
        metrics_result: RolloutMetrics,
        problem_count: int,
    ) -> torch.Tensor:
        """Reduce per-timestep feasibility to one flag per IK candidate."""
        hybrid_values = metrics_result.costs_and_constraints.hybrid_costs_constraints.values
        feasible = metrics_result.feasible
        if feasible is None or hybrid_values:
            feasible = metrics_result.costs_and_constraints.get_feasible(
                include_all_hybrid=False,
                sum_horizon=True,
            )
        if isinstance(feasible, bool):
            return torch.full(
                (problem_count,),
                feasible,
                device=self._config.device_cfg.device,
                dtype=torch.bool,
            )
        feasible_view = feasible.view(problem_count, -1)
        return (
            feasible_view[:, 0]
            if feasible_view.shape[-1] == 1
            else torch.all(feasible_view, dim=-1)
        )

    def _bind_inputs(
        self,
        optimized_position: torch.Tensor,
        metrics_result: RolloutMetrics,
    ) -> bool:
        """Bind stable metric tensors and validate the specialized kernel contract."""
        if (
            metrics_result.state is None
            or metrics_result.convergence is None
            or metrics_result.costs_and_constraints is None
        ):
            return False
        state = metrics_result.state.cuda_robot_model_state
        if (
            state is None
            or state.tool_poses is None
            or state.tool_poses.position is None
            or state.tool_poses.quaternion is None
            or state.robot_spheres is None
            or state.robot_com is None
        ):
            return False

        problem_count = optimized_position.shape[0]
        layout = self._get_convergence_layout(metrics_result)
        if layout is None or not layout.goalset_indices:
            return False
        values = metrics_result.convergence.values
        position_tensors = [values[index] for index in layout.position_indices]
        orientation_tensors = [values[index] for index in layout.orientation_indices]
        cspace_tensors = [values[index] for index in layout.cspace_indices]
        goalset_tensors = [values[index] for index in layout.goalset_indices]
        self._position_count = sum(tensor.numel() for tensor in position_tensors) // problem_count
        self._orientation_count = (
            sum(tensor.numel() for tensor in orientation_tensors) // problem_count
        )
        self._cspace_count = sum(tensor.numel() for tensor in cspace_tensors) // problem_count
        goalset_count = sum(tensor.numel() for tensor in goalset_tensors) // problem_count
        if goalset_count != self._num_links:
            return False

        float_tensors = [
            optimized_position,
            *position_tensors,
            *orientation_tensors,
            *cspace_tensors,
            state.tool_poses.position,
            state.tool_poses.quaternion,
            state.robot_spheres,
            state.robot_com,
        ]
        tensors = [*float_tensors, *goalset_tensors]
        feasible = metrics_result.feasible
        if isinstance(feasible, torch.Tensor):
            tensors.append(feasible)
            if feasible.dtype != torch.bool:
                return False
        if not optimized_position.is_cuda or optimized_position.dtype != torch.float32:
            return False
        if any(
            tensor.device != optimized_position.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            return False
        if any(tensor.dtype != torch.float32 for tensor in float_tensors):
            return False
        if any(tensor.dtype != torch.int32 for tensor in goalset_tensors):
            return False

        self._optimized_position = optimized_position
        self._metrics_result = metrics_result
        return True

    def _process_impl(self) -> torch.Tensor:
        """Launch metric reduction, top-k, and result gathering on bound tensors."""
        optimized_position = self._optimized_position
        metrics_result = self._metrics_result
        layout = self._convergence_layout
        if optimized_position is None or metrics_result is None or layout is None:
            log_and_raise("IK result processor inputs have not been bound")

        problem_count = optimized_position.shape[0]
        position_values = self._get_convergence_values(
            metrics_result, layout.position_indices, problem_count
        )
        orientation_values = self._get_convergence_values(
            metrics_result, layout.orientation_indices, problem_count
        )
        cspace_values = (
            self._get_convergence_values(metrics_result, layout.cspace_indices, problem_count)
            if layout.cspace_indices
            else None
        )
        goalset_index = self._get_convergence_values(
            metrics_result, layout.goalset_indices, problem_count
        )
        feasible = self._get_feasibility(metrics_result, problem_count)
        state = metrics_result.state.cuda_robot_model_state

        device, stream = get_warp_device_stream(optimized_position)
        input_tensors = {
            "optimized_position": optimized_position.reshape(-1),
            "position_values": position_values.reshape(-1),
            "orientation_values": orientation_values.reshape(-1),
            "cspace_values": (
                self._dummy_float if cspace_values is None else cspace_values.reshape(-1)
            ),
            "feasible": feasible.reshape(-1),
            "goalset_index": goalset_index.reshape(-1),
            "tool_position": state.tool_poses.position.reshape(-1),
            "tool_quaternion": state.tool_poses.quaternion.reshape(-1),
            "robot_spheres": state.robot_spheres.reshape(-1),
            "robot_com": state.robot_com.reshape(-1),
        }
        self._wp_inputs = {
            name: wp.from_torch(tensor, requires_grad=False, return_ctype=True)
            for name, tensor in input_tensors.items()
        }
        wp.launch(
            kernel=_reduce_ik_result_metrics_kernel,
            dim=problem_count,
            inputs=[
                self._wp_inputs["position_values"],
                self._wp_inputs["orientation_values"],
                self._wp_inputs["cspace_values"],
                self._wp_inputs["feasible"],
                problem_count,
                self._position_count,
                self._orientation_count,
                self._cspace_count,
                self._config.position_tolerance,
                self._config.orientation_tolerance,
                self._config.position_rank_weight,
                self._config.orientation_rank_weight,
                self._config.cspace_rank_weight,
                self._config.success_requires_convergence,
            ],
            outputs=[
                self._wp_outputs["metric_summary"],
                self._wp_outputs["metric_flags"],
            ],
            device=device,
            stream=stream,
        )

        torch.topk(
            self._metric_summary[2].view(self._batch_size, self._num_seeds),
            k=self._return_seeds,
            largest=False,
            dim=-1,
            out=(self._topk_values, self._relative_indices),
        )

        wp.launch(
            kernel=_gather_ik_result_kernel,
            dim=(self._batch_size * self._return_seeds, self._max_gather_width),
            inputs=[
                self._wp_outputs["relative_indices"],
                self._wp_inputs["optimized_position"],
                self._wp_outputs["metric_summary"],
                self._wp_outputs["metric_flags"],
                self._wp_inputs["goalset_index"],
                self._wp_inputs["tool_position"],
                self._wp_inputs["tool_quaternion"],
                self._wp_inputs["robot_spheres"],
                self._wp_inputs["robot_com"],
                self._wp_full_to_active,
                self._wp_locked_position,
                problem_count,
                self._num_seeds,
                self._return_seeds,
                self._config.action_dim,
                self._num_links,
                self._num_spheres,
                self._num_com,
                len(self._config.full_joint_names),
            ],
            outputs=[
                self._wp_outputs["solution"],
                self._wp_outputs["position_error"],
                self._wp_outputs["rotation_error"],
                self._wp_outputs["selected_flags"],
                self._wp_outputs["goalset_index"],
                self._wp_outputs["tool_position"],
                self._wp_outputs["tool_quaternion"],
                self._wp_outputs["robot_spheres"],
                self._wp_outputs["robot_com"],
                self._wp_outputs["active_derivatives"],
                self._wp_outputs["full_position"],
                self._wp_outputs["full_derivatives"],
                self._wp_outputs["absolute_indices"],
            ],
            device=device,
            stream=stream,
        )
        return self._solution

    @profiler.record_function("ik_result_processor/process")
    def process(
        self,
        opt_result: torch.Tensor,
        metrics_result: RolloutMetrics,
        num_seeds: int,
        return_seeds: int,
    ) -> Optional[IKSolverResult]:
        """Reduce candidate metrics and return selected states, or ``None`` if unsupported."""
        optimized_position = (
            metrics_result.actions
            if self._use_cuda_graph and metrics_result.actions is not None
            else opt_result
        )
        problem_count = optimized_position.shape[0]
        batch_size = problem_count // num_seeds
        state = metrics_result.state.cuda_robot_model_state
        num_spheres = state.robot_spheres.shape[-2]
        num_com = state.robot_com.numel() // (problem_count * 4)
        self.update_batch_size(
            batch_size,
            num_seeds,
            return_seeds,
            state.tool_poses.tool_frames,
            num_spheres,
            num_com,
        )

        if self._use_cuda_graph:
            if self._result_graph is not None and metrics_result is not self._metrics_result:
                self.reset_cuda_graph()
            if self._result_graph is None:
                if not self._bind_inputs(optimized_position, metrics_result):
                    return None
                self._result_graph = create_graph_executor(
                    capture_fn=self._process_impl,
                    device=self._config.device_cfg.device,
                    use_cuda_graph=True,
                    clone_outputs=False,
                )
            self._result_graph()
        else:
            if not self._bind_inputs(optimized_position, metrics_result):
                return None
            self._process_impl()

        return IKSolverResult(
            success=self._selected_flags[0].view(batch_size, return_seeds),
            feasible=self._selected_flags[1].view(batch_size, return_seeds),
            solution=self._solution,
            js_solution=self._js_solution,
            position_error=self._position_error,
            rotation_error=self._rotation_error,
            goalset_index=self._goalset_index,
            solve_time=0.0,
            debug_info={"seed_idx": self._absolute_indices},
            optimized_seeds=optimized_position.view(
                batch_size, num_seeds, self._config.action_dim
            ),
            solution_state=self._solution_state,
        )
