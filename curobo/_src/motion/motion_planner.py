# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-problem motion planner with retry logic and graph-planner seeding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Union

import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.collision_scene import create_scene_collision
from curobo._src.geom.types import SceneCfg
from curobo._src.motion.motion_planner_cfg import (
    MotionPlannerCfg,
    is_rrt_graph_planner_algorithm,
)
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.motion.planner_grasp import (
    GraspPhasePlanner,
    get_result_goal_configuration,
)
from curobo._src.motion.planner_grasp import axis_string_to_vector as _axis_string_to_vector
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import repeat_joint_state_seeds
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.pose import Pose
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise


def _make_warmup_goalset_poses(
    compute_kinematics: Callable[[JointState], KinematicsState],
    current_state: JointState,
    num_goalset: int,
    joint_index: int,
    joint_delta: float,
) -> Dict[str, Pose]:
    """Create reachable per-batch warmup poses with ``num_goalset`` candidates."""
    positions_per_frame: Dict[str, List[torch.Tensor]] = {}
    quaternions_per_frame: Dict[str, List[torch.Tensor]] = {}
    for goal_index in range(num_goalset):
        goal_state = current_state.clone()
        delta = joint_delta * (goal_index + 1) / num_goalset
        goal_state.position[..., joint_index] += delta
        kinematics_state = compute_kinematics(goal_state)
        for frame, pose in kinematics_state.tool_poses.to_dict().items():
            positions_per_frame.setdefault(frame, []).append(pose.position)
            quaternions_per_frame.setdefault(frame, []).append(pose.quaternion)

    return {
        frame: Pose(
            position=torch.stack(positions_per_frame[frame], dim=1).reshape(-1, 3),
            quaternion=torch.stack(quaternions_per_frame[frame], dim=1).reshape(-1, 4),
        )
        for frame in positions_per_frame
    }


def _linear_seed_trajectories(
    current_state: JointState,
    seed_config: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Interpolate one slot-aligned fallback trajectory per goal configuration."""
    batch_size, num_seeds, dof = seed_config.shape
    start = current_state.position.reshape(batch_size, 1, 1, dof)
    goal = seed_config.reshape(batch_size, num_seeds, 1, dof)
    interpolation = torch.linspace(
        0.0,
        1.0,
        horizon,
        device=seed_config.device,
        dtype=seed_config.dtype,
    ).reshape(1, 1, horizon, 1)
    return start + interpolation * (goal - start)


def _fit_linear_seed_trajectories(
    current_state: JointState,
    seed_config: torch.Tensor,
    action_horizon: int,
    seed_fitter=None,
) -> torch.Tensor:
    """Prepare linear seeds through the fitter's complete preparation graph."""
    return _prepare_seed_trajectories(
        current_state, seed_config, None, None, action_horizon, seed_fitter
    )


def _fit_linear_seed_trajectories_eager(
    current_state: JointState,
    seed_config: torch.Tensor,
    action_horizon: int,
    seed_fitter=None,
) -> torch.Tensor:
    """Create boundary-consistent B-spline seeds for straight joint paths."""
    if seed_fitter is None:
        return _linear_seed_trajectories(current_state, seed_config, action_horizon)

    batch_size, num_seeds, dof = seed_config.shape
    dense_target = _linear_seed_trajectories(
        current_state,
        seed_config,
        seed_fitter.config.rollout_horizon,
    )
    fitted = seed_fitter.fit(
        dense_target.reshape(batch_size * num_seeds, dense_target.shape[-2], dof),
        repeat_joint_state_seeds(current_state, num_seeds),
        seed_config.reshape(batch_size * num_seeds, dof),
    )
    return fitted.reshape(batch_size, num_seeds, action_horizon, dof)


def _should_evaluate_seed_before_trajopt(
    config: MotionPlannerCfg,
    used_graph_seed: bool,
) -> bool:
    """Return whether a valid fitted seed may bypass TrajOpt."""
    return not config.enable_trajopt or (
        not config.always_run_trajopt
        and not (used_graph_seed and config.run_trajopt_on_graph_seed)
    )


def _can_evaluate_seed(config: MotionPlannerCfg, seed_fitter, warming_up: bool) -> bool:
    """Return whether the seed evaluator supports the current planning call."""
    return (seed_fitter is not None or not config.enable_trajopt) and (
        not warming_up or not config.enable_trajopt
    )


def _merge_graph_seed_trajectories(
    current_state: JointState,
    seed_config: torch.Tensor,
    graph_waypoints: Optional[torch.Tensor],
    graph_success: torch.Tensor,
    action_horizon: int,
    graph_seed_fitter=None,
) -> torch.Tensor:
    """Prepare RRT paths and linear fallbacks in one CUDA graph replay."""
    return _prepare_seed_trajectories(
        current_state, seed_config, graph_waypoints, graph_success,
        action_horizon, graph_seed_fitter,
    )


def _prepare_seed_trajectories(
    current_state: JointState,
    seed_config: torch.Tensor,
    graph_waypoints: Optional[torch.Tensor],
    graph_success: Optional[torch.Tensor],
    action_horizon: int,
    seed_fitter=None,
) -> torch.Tensor:
    """Cache complete preparation graphs by tensor layout on the owning fitter."""
    from curobo._src.util.cuda_graph_util import GraphExecutor

    inputs = (
        current_state.position, current_state.velocity, current_state.acceleration,
        current_state.jerk, seed_config, graph_waypoints,
        graph_success if graph_waypoints is not None else None,
    )
    executors = getattr(seed_fitter, "_preparation_executors", None)
    if executors is None:
        return _merge_graph_seed_trajectories_eager(
            current_state, seed_config, graph_waypoints, graph_success,
            action_horizon, seed_fitter,
        )
    key = (action_horizon, tuple(
        (tuple(t.shape), t.dtype, t.device) if t is not None else None for t in inputs
    ))
    if key not in executors:
        def prepare(
            position: torch.Tensor,
            velocity: Optional[torch.Tensor],
            acceleration: Optional[torch.Tensor],
            jerk: Optional[torch.Tensor],
            goals: torch.Tensor,
            waypoints: Optional[torch.Tensor],
            success: Optional[torch.Tensor],
        ) -> torch.Tensor:
            state = JointState(position=position, velocity=velocity,
                               acceleration=acceleration, jerk=jerk)
            return _merge_graph_seed_trajectories_eager(
                state, goals, waypoints, success, action_horizon, seed_fitter
            )

        executors[key] = GraphExecutor(
            capture_fn=prepare, device=seed_config.device,
            use_cuda_graph=seed_fitter.config.use_cuda_graph, clone_outputs=False,
        )
    return executors[key](*inputs)


def _merge_graph_seed_trajectories_eager(
    current_state: JointState,
    seed_config: torch.Tensor,
    graph_waypoints: Optional[torch.Tensor],
    graph_success: torch.Tensor,
    action_horizon: int,
    graph_seed_fitter=None,
) -> torch.Tensor:
    """Fit graph paths and straight-line fallbacks in one fixed lane batch."""
    batch_size, num_seeds, dof = seed_config.shape
    if graph_waypoints is None:
        return _fit_linear_seed_trajectories_eager(
            current_state,
            seed_config,
            action_horizon,
            graph_seed_fitter,
        )

    lane_count = batch_size * num_seeds
    flat_goals = seed_config.reshape(lane_count, dof)
    if graph_seed_fitter is not None:
        linear_waypoints = _linear_seed_trajectories(
            current_state,
            seed_config,
            graph_waypoints.shape[-2],
        ).reshape(lane_count, graph_waypoints.shape[-2], dof)
        graph_success_mask = graph_success.reshape(lane_count, 1, 1)
        fit_target = torch.where(
            graph_success_mask,
            graph_waypoints.reshape(lane_count, graph_waypoints.shape[-2], dof),
            linear_waypoints,
        )
        graph_seed = graph_seed_fitter.fit(
            fit_target,
            repeat_joint_state_seeds(current_state, num_seeds),
            flat_goals,
        )
        return graph_seed.reshape(batch_size, num_seeds, action_horizon, dof)
    else:
        linear_seed = _linear_seed_trajectories(current_state, seed_config, action_horizon)
        graph_seed = graph_waypoints.reshape(lane_count, action_horizon, dof)

    graph_seed = graph_seed.reshape(batch_size, num_seeds, action_horizon, dof)
    graph_success = graph_success.reshape(batch_size, num_seeds, 1, 1)
    return torch.where(graph_success, graph_seed, linear_seed)


def _create_graph_seed_fitter(config: MotionPlannerCfg, trajopt_solver, num_lanes: int):
    """Create a fixed-shape fitter only for enabled B-spline TrajOpt planners."""
    if not config.fit_graph_seed_to_bspline:
        return None
    if trajopt_solver.transition_model.control_space not in ControlSpace.bspline_types():
        return None

    from curobo._src.trajectory.trajectory_seed_fitter import (
        BSplineTimeOptimalSeedFitter,
    )

    return BSplineTimeOptimalSeedFitter.from_transition_model(
        trajopt_solver.transition_model,
        num_lanes=num_lanes,
        use_cuda_graph=config.trajopt_solver_config.use_cuda_graph,
    )


@dataclass(frozen=True)
class GraphCallTelemetry:
    """Sanitized scalar telemetry emitted by one graph-planner invocation."""

    solve_time: float
    success: bool
    wall_time: Optional[float] = None
    growth_steps: Optional[int] = None
    budget_exhausted: Optional[bool] = None
    used_failure_extension: Optional[bool] = None
    component_times: Optional[Mapping[str, float]] = None
    shortcut_stats: Optional[Mapping[str, int]] = None
    projection_stats: Optional[Mapping[str, Union[bool, int, float]]] = None

    def __post_init__(self) -> None:
        """Freeze nested telemetry mappings as defensive copies."""
        if self.component_times is not None:
            object.__setattr__(
                self,
                "component_times",
                MappingProxyType(dict(self.component_times)),
            )
        if self.shortcut_stats is not None:
            object.__setattr__(
                self,
                "shortcut_stats",
                MappingProxyType(dict(self.shortcut_stats)),
            )
        if self.projection_stats is not None:
            object.__setattr__(
                self,
                "projection_stats",
                MappingProxyType(dict(self.projection_stats)),
            )

    @staticmethod
    def _component_times(value: object) -> Optional[Dict[str, float]]:
        if not isinstance(value, dict):
            return None
        sanitized = {
            component: float(seconds)
            for component, seconds in value.items()
            if isinstance(component, str)
            and isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) >= 0.0
        }
        return sanitized or None

    @staticmethod
    def _shortcut_stats(value: object) -> Optional[Dict[str, int]]:
        if not isinstance(value, dict):
            return None
        sanitized = {
            name: count
            for name, count in value.items()
            if isinstance(name, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        }
        return sanitized or None

    @staticmethod
    def _projection_stats(
        value: object,
    ) -> Optional[Dict[str, Union[bool, int, float]]]:
        """Copy finite nonnegative scalar projection diagnostics."""
        if not isinstance(value, dict):
            return None
        sanitized: Dict[str, Union[bool, int, float]] = {}
        for name, scalar in value.items():
            if not isinstance(name, str):
                continue
            if isinstance(scalar, bool):
                sanitized[name] = scalar
            elif isinstance(scalar, int) and scalar >= 0:
                sanitized[name] = scalar
            elif isinstance(scalar, float) and math.isfinite(scalar) and scalar >= 0.0:
                sanitized[name] = scalar
        return sanitized or None

    @classmethod
    def from_result(
        cls,
        result,
        *,
        wall_time: Optional[float] = None,
    ) -> "GraphCallTelemetry":
        """Create one immutable record without retaining planner output tensors."""
        result_debug = getattr(result, "debug_info", None)
        if not isinstance(result_debug, dict):
            result_debug = {}
        growth_steps = result_debug.get("growth_steps")
        if isinstance(growth_steps, bool) or not isinstance(growth_steps, int):
            growth_steps = None
        budget_exhausted = result_debug.get("budget_exhausted")
        if not isinstance(budget_exhausted, bool):
            budget_exhausted = None
        used_failure_extension = result_debug.get("used_failure_extension")
        if not isinstance(used_failure_extension, bool):
            used_failure_extension = None
        if wall_time is not None and (
            isinstance(wall_time, bool)
            or not isinstance(wall_time, (int, float))
            or not math.isfinite(float(wall_time))
            or wall_time < 0.0
        ):
            wall_time = None
        return cls(
            solve_time=float(result.solve_time),
            success=bool((torch.count_nonzero(result.success) > 0).item()),
            wall_time=None if wall_time is None else float(wall_time),
            growth_steps=growth_steps,
            budget_exhausted=budget_exhausted,
            used_failure_extension=used_failure_extension,
            component_times=cls._component_times(getattr(result, "component_times", None)),
            shortcut_stats=cls._shortcut_stats(result_debug.get("shortcut")),
            projection_stats=cls._projection_stats(result_debug.get("projection")),
        )


class MotionPlanner:
    """Single-problem motion planner with retry logic and graph-planner seeding.

    Solves one planning problem at a time (``batch_size=1``).  For batched
    planning see :class:`BatchMotionPlanner`.
    """

    def __init__(self, config: MotionPlannerCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = None
        self._destroyed = False
        self._warming_up = False
        self._last_graph_call_telemetry: List[GraphCallTelemetry] = []
        self._last_planning_attempts = 0
        self._grasp_phase_planner: GraspPhasePlanner
        self._initialize_components()

    @property
    def last_graph_time(self) -> float:
        """CUDA-synchronized graph-planner time accumulated by the last plan call."""
        return sum(record.solve_time for record in self._last_graph_call_telemetry)

    @property
    def last_graph_calls(self) -> int:
        """Number of graph-planner invocations made by the last plan call."""
        return len(self._last_graph_call_telemetry)

    @property
    def last_graph_success(self) -> bool:
        """Whether any graph-planner invocation in the last plan call found a path."""
        return any(record.success for record in self._last_graph_call_telemetry)

    @property
    def last_graph_call_telemetry(self) -> List[GraphCallTelemetry]:
        """Return one immutable telemetry record per graph-planner invocation."""
        return list(self._last_graph_call_telemetry)

    @property
    def last_planning_attempts(self) -> int:
        """Number of outer planning attempts executed by the last plan call."""
        return self._last_planning_attempts

    def _reset_graph_telemetry(self) -> None:
        self._last_graph_call_telemetry = []
        self._last_graph_seed_success = None

    def _initialize_components(self):
        if self.config.scene_collision_cfg is not None:
            self.scene_collision_checker = create_scene_collision(self.config.scene_collision_cfg)

        self.ik_solver = IKSolver(self.config.ik_solver_config, self.scene_collision_checker)
        self.trajopt_solver = TrajOptSolver(
            self.config.trajopt_solver_config, self.scene_collision_checker
        )
        self.graph_planner = None
        self.constrained_rrt_planner = None
        self.constrained_rrt_seed_fitter = None
        self.graph_seed_fitter = _create_graph_seed_fitter(
            self.config,
            self.trajopt_solver,
            self.trajopt_solver.config.num_seeds,
        )
        if self.config.rrt_graph_planner_config is not None:
            if is_rrt_graph_planner_algorithm(self.config.graph_planner_algorithm):
                from curobo._src.graph_planner.rrt import (
                    BiRRTPlanner,
                    RRTPlannerResources,
                )

                rrt_config = self.config.rrt_graph_planner_config
                required_lanes = self.config.trajopt_solver_config.num_seeds
                if required_lanes > rrt_config.planner_config.max_batched_lanes:
                    log_and_raise(
                        "RRT trajopt seeds exceed max_batched_lanes: "
                        f"{required_lanes} > {rrt_config.planner_config.max_batched_lanes}"
                    )
                self.graph_planner = BiRRTPlanner(
                    RRTPlannerResources(
                        rrt_config.resources_config,
                        self.scene_collision_checker,
                    ),
                    rrt_config.planner_config,
                )
        elif self.config.prm_graph_planner_config is not None:
            from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner

            self.graph_planner = PRMGraphPlanner(
                self.config.prm_graph_planner_config,
                self.scene_collision_checker,
            )
        if self.config.constrained_rrt_planner_config is not None:
            from curobo._src.graph_planner.rrt import (
                BiRRTPlanner,
                RRTPlannerResources,
            )

            constrained_config = self.config.constrained_rrt_planner_config
            self.constrained_rrt_planner = BiRRTPlanner(
                RRTPlannerResources(
                    constrained_config.resources_config,
                    self.scene_collision_checker,
                ),
                constrained_config.planner_config,
            )
            self.constrained_rrt_planner.initialize_query_shape(1)
            self.constrained_rrt_seed_fitter = _create_graph_seed_fitter(
                self.config,
                self.trajopt_solver,
                num_lanes=1,
            )
        self._grasp_phase_planner = GraspPhasePlanner(
            config=self.config,
            batch_size=1,
            kinematics=self.kinematics,
            ik_solver=self.ik_solver,
            trajopt_solver=self.trajopt_solver,
            constrained_rrt_planner=self.constrained_rrt_planner,
            constrained_rrt_seed_fitter=self.constrained_rrt_seed_fitter,
            plan_pose=lambda *args, **kwargs: self.plan_pose(*args, **kwargs),
            update_tool_pose_criteria=lambda criteria: self.update_tool_pose_criteria(criteria),
            enable_link_collision=lambda links: self.enable_link_collision(links),
            disable_link_collision=lambda links: self.disable_link_collision(links),
            record_graph_result=lambda result: self._last_graph_call_telemetry.append(
                GraphCallTelemetry.from_result(result)
            ),
        )

    # -- Lifecycle --

    def destroy(self):
        """Release all CUDA graph resources.

        Call before dropping the last reference to avoid ``cudaMallocAsync``
        warnings about captured allocations freed outside graph replay.

        Does not synchronize, the ``gc.collect()`` + ``synchronize()`` fence
        in :meth:`GraphExecutor._initialize_cuda_graph` ensures stale
        ``cuGraphExecDestroy`` calls complete before any new capture begins.
        """
        if self._destroyed:
            return
        self._destroyed = True
        grasp_phase_planner = getattr(self, "_grasp_phase_planner", None)
        if grasp_phase_planner is not None:
            grasp_phase_planner.destroy()
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()
        if self.graph_planner is not None and hasattr(self.graph_planner, "reset_cuda_graph"):
            self.graph_planner.reset_cuda_graph()
        if self.constrained_rrt_planner is not None:
            self.constrained_rrt_planner.reset_cuda_graph()
        graph_seed_fitter = getattr(self, "graph_seed_fitter", None)
        if graph_seed_fitter is not None:
            graph_seed_fitter.reset_cuda_graph()
        constrained_rrt_seed_fitter = getattr(self, "constrained_rrt_seed_fitter", None)
        if constrained_rrt_seed_fitter is not None:
            constrained_rrt_seed_fitter.reset_cuda_graph()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def attachment_manager(self) -> AttachmentManager:
        """Attachment manager for attaching/detaching obstacles to robot links."""
        return self.trajopt_solver.core.attachment_manager

    # -- Properties --

    @property
    def joint_names(self) -> List[str]:
        return self.ik_solver.joint_names

    @property
    def action_dim(self) -> int:
        return self.trajopt_solver.action_dim

    @property
    def tool_frames(self) -> List[str]:
        return self.trajopt_solver.tool_frames

    @property
    def default_joint_state(self) -> JointState:
        return self.trajopt_solver.default_joint_state

    @property
    def kinematics(self) -> Kinematics:
        return self.trajopt_solver.kinematics

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        return self.trajopt_solver.compute_kinematics(state)

    # -- Warmup --

    def warmup(
        self,
        enable_graph: bool = True,
        warmup_joint_index: int = 0,
        warmup_joint_delta: float = 0.2,
        num_warmup_iterations: int = 10,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> bool:
        self._warming_up = True
        num_goalset = self.config.ik_solver_config.max_goalset
        og_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False

        for _ in range(num_warmup_iterations):
            current_state = self.default_joint_state.clone().unsqueeze(0)

            if num_goalset > 1:
                goalset_poses = self._make_warmup_goalset(
                    current_state,
                    num_goalset,
                    warmup_joint_index,
                    warmup_joint_delta,
                )
                goalset_goal = GoalToolPose.from_poses(
                    goalset_poses,
                    num_goalset=num_goalset,
                )
                result = self.plan_pose(
                    goalset_goal,
                    current_state,
                    max_attempts=1,
                    tool_relative_transform_target=tool_relative_transform_target,
                )
                if result is None:
                    goal_tool_poses = GoalToolPose.from_poses(
                        goalset_poses,
                        num_goalset=num_goalset,
                    )
                    seed_config = current_state.position.view(1, 1, -1).repeat(
                        1, self.trajopt_solver.config.num_seeds, 1
                    )
                    self.trajopt_solver.solve_pose(
                        goal_tool_poses,
                        current_state,
                        seed_config=seed_config,
                        use_implicit_goal=True,
                        tool_relative_transform_target=tool_relative_transform_target,
                    )
            else:
                goal_state = current_state.clone()
                goal_state.position[..., warmup_joint_index] += warmup_joint_delta
                goal_tool_poses = self.compute_kinematics(goal_state).tool_poses.as_goal()
                self.plan_pose(
                    goal_tool_poses,
                    current_state,
                    max_attempts=1,
                    tool_relative_transform_target=tool_relative_transform_target,
                )

            self.reset_seed()

        if enable_graph and self.graph_planner is not None:
            self.graph_planner.warmup(
                num_warmup_iterations=num_warmup_iterations,
                max_batch_size=self.trajopt_solver.config.num_seeds,
            )
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if enable_graph and constrained_rrt_planner is not None:
            constrained_warmup_state = self.default_joint_state.clone().unsqueeze(0)
            constrained_warmup_goal = self.compute_kinematics(
                constrained_warmup_state
            ).tool_poses.as_goal()
            self._grasp_phase_planner.install_rrt_constraint(constrained_warmup_goal)
            constrained_rrt_planner.warmup(
                num_warmup_iterations=num_warmup_iterations,
                max_batch_size=1,
            )
            self._grasp_phase_planner.activate_rrt_constraint(
                constrained_warmup_goal,
                axis="z",
                in_tool_frame=True,
            )
            self.ik_solver.solve_goalset(
                constrained_warmup_goal,
                current_state=constrained_warmup_state,
                return_seeds=1,
            )
        graph_seed_fitter = getattr(self, "graph_seed_fitter", None)
        if graph_seed_fitter is not None:
            graph_seed_fitter.warmup()
        constrained_rrt_seed_fitter = getattr(self, "constrained_rrt_seed_fitter", None)
        if constrained_rrt_seed_fitter is not None:
            constrained_rrt_seed_fitter.warmup()

        self.ik_solver.config.exit_early = og_exit_early
        # Warmup is not a user planning request; do not expose its synthetic
        # graph calls as last-request telemetry.
        self._reset_graph_telemetry()
        self._warming_up = False
        return True

    def _make_warmup_goalset(
        self,
        current_state: JointState,
        num_goalset: int,
        joint_index: int,
        joint_delta: float,
    ) -> Dict[str, Pose]:
        return _make_warmup_goalset_poses(
            self.compute_kinematics,
            current_state,
            num_goalset,
            joint_index,
            joint_delta,
        )

    # -- Planning --

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 20,
        enable_graph_attempt: int = 1,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
        finetune_attempts: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a trajectory to reach target tool poses.

        Goalset is auto-detected from ``goal_tool_poses.num_goalset``.  When
        ``num_goalset > 1`` the planner uses a simpler IK+TrajOpt loop without
        graph seeding. ``finetune_attempts`` controls the number of additional
        time-optimal TrajOpt passes after the initial solve.
        The default configuration uses 50 initial iterations and one 25-iteration pass.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")

        if goal_tool_poses.num_goalset > 1:
            return self._plan_pose_goalset(
                goal_tool_poses,
                current_state,
                use_implicit_goal,
                max_attempts,
                tool_relative_transform_target=tool_relative_transform_target,
                finetune_attempts=finetune_attempts,
            )

        return self._plan_pose_single(
            goal_tool_poses,
            current_state,
            use_implicit_goal,
            max_attempts,
            enable_graph_attempt,
            tool_relative_transform_target=tool_relative_transform_target,
            finetune_attempts=finetune_attempts,
        )

    def _plan_pose_single(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool,
        max_attempts: int,
        enable_graph_attempt: int,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
        finetune_attempts: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan one goal, reusing direct-attempt IK once for graph fallback."""
        self._reset_graph_telemetry()
        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds
        self._last_planning_attempts = 0
        reusable_ik_result: Optional[IKSolverResult] = None

        for current_attempt in range(max_attempts):
            self._last_planning_attempts = current_attempt + 1
            current_state = og_current_state.clone()
            reuse_ik_result = (
                current_attempt == enable_graph_attempt
                and self.graph_planner is not None
                and reusable_ik_result is not None
            )
            if reuse_ik_result:
                ik_result = reusable_ik_result
                reusable_ik_result = None
            else:
                reusable_ik_result = None
                ik_result = self.ik_solver.solve_pose(
                    goal_tool_poses,
                    return_seeds=num_seeds,
                    current_state=current_state,
                    tool_relative_transform_target=tool_relative_transform_target,
                )
                total_time += ik_result.total_time
                solve_time += ik_result.solve_time

            success_count = torch.count_nonzero(ik_result.success)
            if success_count == 0:
                continue

            if current_attempt + 1 == enable_graph_attempt and self.graph_planner is not None:
                reusable_ik_result = ik_result

            seed_config = ik_result.solution
            ik_success = ik_result.success
            if success_count < num_seeds:
                good_solution = seed_config[ik_success][0].view(1, 1, -1)
                seed_config = torch.where(
                    ik_success.unsqueeze(-1),
                    seed_config,
                    good_solution,
                )

            seed_fitter = getattr(self, "graph_seed_fitter", None) if use_implicit_goal else None
            seed_traj = _fit_linear_seed_trajectories(
                current_state,
                seed_config,
                self.trajopt_solver.action_horizon,
                seed_fitter,
            )
            trajectory_source = "linear_fit" if seed_fitter is not None else "linear"
            used_graph_seed = False
            finetune_dt_scale = 0.6
            time_optimal_iters = 25
            finetune_iters = 25
            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                graph_seed = self._get_graph_seed_trajectories(
                    current_state,
                    seed_config,
                    goal_valid=ik_success,
                    use_bspline_fitter=use_implicit_goal,
                )
                if graph_seed is None:
                    continue
                seed_traj = graph_seed
                total_time += 0.0  # graph time already in graph_seed call
                trajectory_source = "graph_fit" if seed_fitter is not None else "graph"
                used_graph_seed = True
                # Preserve the existing IK+TrajOpt total-time contract. Graph time is
                # exposed separately through ``last_graph_time``.

            if _should_evaluate_seed_before_trajopt(
                self.config, used_graph_seed
            ) and _can_evaluate_seed(self.config, seed_fitter, self._warming_up):
                seed_result = self.trajopt_solver.evaluate_pose_seed(
                    goal_tool_poses,
                    current_state,
                    seed_traj,
                    use_implicit_goal=use_implicit_goal,
                    trajectory_source=trajectory_source,
                    seed_valid_mask=(self._last_graph_seed_success if used_graph_seed else None),
                    tool_relative_transform_target=tool_relative_transform_target,
                )
                total_time += seed_result.total_time
                if torch.count_nonzero(seed_result.success) > 0:
                    seed_result.total_time = total_time
                    seed_result.solve_time = solve_time
                    return seed_result
                if not self.config.enable_trajopt:
                    trajopt_result = seed_result
                    continue

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses,
                current_state,
                seed_config=seed_config,
                seed_traj=seed_traj,
                use_implicit_goal=use_implicit_goal,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
                time_optimal_iters=time_optimal_iters,
                finetune_iters=finetune_iters,
                tool_relative_transform_target=tool_relative_transform_target,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            trajopt_result.debug_info["trajectory_source"] = "trajopt"
            trajopt_result.trajectory_source = "trajopt"
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result

    def _plan_pose_goalset(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 10,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
        finetune_attempts: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Goalset planning: IK + TrajOpt, no graph seeding."""
        self._reset_graph_telemetry()
        self._last_planning_attempts = 0
        for current_attempt in range(max_attempts):
            self._last_planning_attempts = current_attempt + 1
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=self.trajopt_solver.config.num_seeds,
                current_state=current_state,
                tool_relative_transform_target=tool_relative_transform_target,
            )
            if torch.count_nonzero(ik_result.success) == 0:
                return None

            seed_fitter = getattr(self, "graph_seed_fitter", None) if use_implicit_goal else None
            seed_traj = _fit_linear_seed_trajectories(
                current_state,
                ik_result.solution,
                self.trajopt_solver.action_horizon,
                seed_fitter,
            )
            if _should_evaluate_seed_before_trajopt(self.config, False) and _can_evaluate_seed(
                self.config, seed_fitter, self._warming_up
            ):
                seed_result = self.trajopt_solver.evaluate_pose_seed(
                    goal_tool_poses,
                    current_state,
                    seed_traj,
                    use_implicit_goal=use_implicit_goal,
                    trajectory_source="linear_fit",
                    tool_relative_transform_target=tool_relative_transform_target,
                )
                if torch.count_nonzero(seed_result.success) > 0:
                    return seed_result
                if not self.config.enable_trajopt:
                    trajopt_result = seed_result
                    continue

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses,
                current_state,
                seed_config=ik_result.solution,
                seed_traj=seed_traj,
                use_implicit_goal=use_implicit_goal,
                finetune_attempts=finetune_attempts,
                tool_relative_transform_target=tool_relative_transform_target,
            )
            trajopt_result.debug_info["trajectory_source"] = "trajopt"
            trajopt_result.trajectory_source = "trajopt"
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        return trajopt_result

    def plan_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        max_attempts: int = 20,
        enable_graph_attempt: int = 1,
        finetune_attempts: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a collision-free trajectory to a joint configuration.

        Args:
            goal_state: Target joint configuration.
            current_state: Initial joint state.
            max_attempts: Maximum planning attempts.
            enable_graph_attempt: Attempt at which to start graph seeding.
            finetune_attempts: Additional 25-iteration TrajOpt refinement passes.

        Returns:
            TrajOptSolverResult, or None if planning failed.
        """
        self._reset_graph_telemetry()
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")
        if goal_state.ndim > 2:
            log_and_raise(f"goal_state must be a 2D tensor, got shape: {goal_state.shape}")

        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds
        self._last_planning_attempts = 0

        for current_attempt in range(max_attempts):
            self._last_planning_attempts = current_attempt + 1
            current_state = og_current_state.clone()
            goal_configs = goal_state.position.view(1, 1, -1).repeat(1, num_seeds, 1)
            seed_fitter = getattr(self, "graph_seed_fitter", None)
            seed_traj = _fit_linear_seed_trajectories(
                current_state,
                goal_configs,
                self.trajopt_solver.action_horizon,
                seed_fitter,
            )
            trajectory_source = "linear_fit" if seed_fitter is not None else "linear"
            used_graph_seed = False
            finetune_dt_scale = 0.6

            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                graph_seed = self._get_graph_seed_trajectories(
                    current_state,
                    goal_configs,
                )
                if graph_seed is None:
                    continue
                seed_traj = graph_seed
                trajectory_source = "graph_fit" if seed_fitter is not None else "graph"
                used_graph_seed = True
                finetune_dt_scale = 0.6

            if _should_evaluate_seed_before_trajopt(
                self.config, used_graph_seed
            ) and _can_evaluate_seed(self.config, seed_fitter, self._warming_up):
                seed_result = self.trajopt_solver.evaluate_cspace_seed(
                    goal_state,
                    current_state,
                    seed_traj,
                    trajectory_source=trajectory_source,
                    seed_valid_mask=(self._last_graph_seed_success if used_graph_seed else None),
                )
                total_time += seed_result.total_time
                if torch.count_nonzero(seed_result.success) > 0:
                    seed_result.total_time = total_time
                    seed_result.solve_time = solve_time
                    return seed_result
                if not self.config.enable_trajopt:
                    trajopt_result = seed_result
                    continue

            trajopt_result = self.trajopt_solver.solve_cspace(
                goal_state,
                current_state,
                seed_traj=seed_traj,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            trajopt_result.debug_info["trajectory_source"] = "trajopt"
            trajopt_result.trajectory_source = "trajopt"
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result

    def _get_graph_seed_trajectories(
        self,
        current_state: JointState,
        seed_config: torch.Tensor,
        goal_valid: Optional[torch.Tensor] = None,
        use_bspline_fitter: bool = True,
    ) -> Optional[torch.Tensor]:
        """Query graph planner for trajectory seeds.

        Args:
            current_state: Start state ``(1, dof)``.
            seed_config: Goal configs ``(1, num_seeds, dof)``.
            goal_valid: Optional IK-success mask ``(1, num_seeds)``. Failed IK
                rows are never exposed as graph roots.
            use_bspline_fitter: Apply the implicit-goal B-spline fitter when
                configured. Disable this for non-implicit-goal TrajOpt calls.

        Returns:
            One slot-aligned seed per goal. Successful graph paths are
            reparameterized by joint-space arc length and fitted to B-spline
            knots when configured; lanes without a graph path fit a straight
            joint-space line to the same B-spline representation.
        """
        dof = self.trajopt_solver.action_dim
        num_seeds = seed_config.shape[1] if seed_config.ndim == 3 else seed_config.shape[0]
        seed_config = seed_config.reshape(1, num_seeds, dof)
        graph_seed_fitter = (
            getattr(self, "graph_seed_fitter", None) if use_bspline_fitter else None
        )
        graph_horizon = (
            self.trajopt_solver.horizon
            if graph_seed_fitter is not None
            else self.trajopt_solver.action_horizon
        )

        graph_kwargs = {
            "interpolate_waypoints": True,
            "interpolation_steps": graph_horizon,
            "interpolation_type": TrajInterpolationType.LINEAR,
            "validate_interpolated_trajectory": False,
        }

        def record_result(result) -> bool:
            record = GraphCallTelemetry.from_result(result)
            self._last_graph_call_telemetry.append(record)
            return record.success

        flat_valid = (
            torch.ones(num_seeds, device=seed_config.device, dtype=torch.bool)
            if goal_valid is None
            else goal_valid.reshape(num_seeds).to(dtype=torch.bool)
        )
        graph_starts = current_state.position.view(1, dof).expand(num_seeds, -1)
        graph_goals = seed_config.reshape(num_seeds, dof)
        if is_rrt_graph_planner_algorithm(self.config.graph_planner_algorithm):
            graph_result = self.graph_planner.find_path(
                graph_starts,
                graph_goals,
                query_mask=flat_valid,
                group_count=1,
                **graph_kwargs,
            )
        else:
            graph_result = self.graph_planner.find_path(
                graph_starts.clone(),
                graph_goals.clone(),
                **graph_kwargs,
            )

        record_result(graph_result)
        graph_success = graph_result.success.reshape(num_seeds) & flat_valid
        self._last_graph_seed_success = graph_success.view(1, num_seeds)
        return _merge_graph_seed_trajectories(
            current_state,
            seed_config,
            graph_result.interpolated_waypoints,
            graph_success,
            self.trajopt_solver.action_horizon,
            graph_seed_fitter,
        )

    # -- Grasp planning --

    def plan_grasp(
        self,
        grasp_poses: GoalToolPose,
        current_state: JointState,
        grasp_approach_axis: str = "z",
        grasp_approach_offset: float = -0.15,
        grasp_approach_in_tool_frame: bool = True,
        grasp_lift_axis: str = "z",
        grasp_lift_offset: float = -0.15,
        grasp_lift_in_tool_frame: bool = True,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_lift: bool = True,
        disable_collision_links: Optional[List[str]] = None,
        use_trajopt_for_goalset_selection: bool = False,
        use_rrt_for_constrained_phases: bool = False,
        use_double_pass_trajopt_for_constrained_phases: bool = False,
    ) -> GraspPlanResult:
        """Plan a grasp motion: goalset -> approach -> grasp -> lift.

        Args:
            grasp_poses: Candidate grasp poses with ``num_goalset`` entries per link.
                Construct via ``GoalToolPose`` with the desired goalset size.
            current_state: Initial joint state with shape ``[1, dof]``.
            grasp_approach_axis: Approach direction, one of ``"x"``, ``"y"``, or ``"z"``.
            grasp_approach_offset: Signed approach offset in meters.
            grasp_approach_in_tool_frame: Interpret the approach direction in the
                selected grasp frame instead of the world frame.
            grasp_lift_axis: Lift direction, one of ``"x"``, ``"y"``, or ``"z"``.
            grasp_lift_offset: Signed lift offset in meters.
            grasp_lift_in_tool_frame: Interpret the lift direction in the selected
                grasp frame instead of the world frame.
            plan_approach_to_grasp: Plan the constrained insertion phase.
            plan_grasp_to_lift: Plan the constrained lift phase after insertion.
            disable_collision_links: Links allowed to contact during goalset selection,
                insertion, and lift. Defaults to the robot's grasp-contact links.
            use_trajopt_for_goalset_selection: Run a full TrajOpt call to select
                the grasp goal. By default, use the IK solver's highest-ranked
                successful configuration. IK ranking includes distance from
                ``current_state``.
            use_rrt_for_constrained_phases: Attempt projected RRT for insertion
                and lift, with single-pass constrained TrajOpt as the failure fallback.
                By default, both phases use single-pass constrained TrajOpt.
            use_double_pass_trajopt_for_constrained_phases: Add one time-optimal
                TrajOpt refinement pass to insertion and lift. This cannot be
                combined with ``use_rrt_for_constrained_phases``.

        Returns:
            Per-phase success, trajectories, and selected goalset metadata.
        """
        if use_rrt_for_constrained_phases and use_double_pass_trajopt_for_constrained_phases:
            log_and_raise(
                "use_rrt_for_constrained_phases and "
                "use_double_pass_trajopt_for_constrained_phases are mutually exclusive"
            )

        if disable_collision_links is None:
            disable_collision_links = (
                self.kinematics.config.kinematics_config.grasp_contact_link_names
            )
        if disable_collision_links is None:
            disable_collision_links = []

        device = current_state.device
        result = GraspPlanResult()
        result.success = torch.tensor([False], device=device)
        result.approach_success = torch.tensor([False], device=device)
        result.grasp_success = torch.tensor([False], device=device)
        result.lift_success = torch.tensor([False], device=device)

        # Step 1: Select one grasp pose from the goalset.
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=False)
        if use_trajopt_for_goalset_selection:
            goalset_result = self.plan_pose(grasp_poses, current_state)
        else:
            goalset_result = self.ik_solver.solve_goalset(
                grasp_poses,
                return_seeds=1,
                current_state=current_state,
            )
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)

        if goalset_result is None:
            result.status = "Goalset planning returned None."
            return result

        result.success = goalset_result.success.clone()
        result.success[:] = False
        result.goalset_result = goalset_result

        if not goalset_result.success.any():
            result.status = "No grasp in goal set was reachable."
            return result

        result.goalset_index = goalset_result.goalset_index.clone()
        grasp_goal_configuration = get_result_goal_configuration(
            goalset_result,
            self.kinematics,
        )
        goal_index = int(goalset_result.goalset_index.view(-1)[0].item())

        grasp_poses_dict = {}
        for frame in grasp_poses.tool_frames:
            li = grasp_poses.tool_frames.index(frame)
            pos = grasp_poses.position[:, 0, li, goal_index, :]
            quat = grasp_poses.quaternion[:, 0, li, goal_index, :]
            grasp_poses_dict[frame] = Pose(position=pos, quaternion=quat)

        # Step 2: Plan to approach pose (offset from grasp)
        approach_offset = Pose.from_list(
            [grasp_approach_offset * v for v in _axis_string_to_vector(grasp_approach_axis)]
            + [1, 0, 0, 0]
        )

        approach_poses_dict = {}
        for frame, goal_pose in grasp_poses_dict.items():
            if grasp_approach_in_tool_frame:
                approach_poses_dict[frame] = goal_pose.multiply(approach_offset)
            else:
                approach_poses_dict[frame] = approach_offset.multiply(goal_pose)

        approach_tool_pose = GoalToolPose.from_poses(
            approach_poses_dict,
            ordered_tool_frames=grasp_poses.tool_frames,
            num_goalset=1,
        )
        approach_result = self.plan_pose(approach_tool_pose, current_state)
        result.approach_result = approach_result

        if approach_result is None or not approach_result.success.any():
            result.status = "Planning to approach pose failed."
            result.approach_success[:] = False
            result.success[:] = False
            return result

        result.approach_success = approach_result.success.clone()
        # Preserve this phase before the shared TrajOpt result buffers are reused.
        result.approach_trajectory = approach_result.js_solution.clone()
        result.approach_trajectory_dt = result.approach_trajectory.dt
        result.approach_interpolated_trajectory = approach_result.interpolated_trajectory.clone()
        result.approach_interpolated_last_tstep = approach_result.interpolated_last_tstep.clone()
        result.status = "Planning to approach pose succeeded."

        if not plan_approach_to_grasp:
            result.success[:] = True
            return result

        # Step 3: Linear motion from approach to grasp
        approach_end = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
        approach_end = self.kinematics.get_active_js(approach_end)
        approach_end.dt = None

        grasp_tool_pose = GoalToolPose.from_poses(
            grasp_poses_dict,
            ordered_tool_frames=grasp_poses.tool_frames,
            num_goalset=1,
        )
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=False)
        grasp_result = self._grasp_phase_planner.plan_linear_phase(
            grasp_tool_pose,
            approach_end,
            grasp_approach_axis,
            grasp_approach_in_tool_frame,
            use_rrt=use_rrt_for_constrained_phases,
            finetune_attempts=int(use_double_pass_trajopt_for_constrained_phases),
            goal_configuration=grasp_goal_configuration,
            goal_valid=goalset_result.success.any(dim=-1),
        )

        if grasp_result is None or not grasp_result.success.any():
            self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)
            result.status = "Planning to grasp pose failed."
            result.grasp_success[:] = False
            result.success[:] = False
            return result

        # Preserve this phase before lift planning reuses the same solver buffers.
        result.grasp_trajectory = grasp_result.js_solution.clone()
        result.grasp_trajectory_dt = result.grasp_trajectory.dt
        result.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory.clone()
        result.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep.clone()
        result.success = grasp_result.success.clone()
        result.grasp_success = grasp_result.success.clone()

        if not plan_grasp_to_lift:
            self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)
            return result

        # Step 4: Lift motion
        lift_start = get_joint_state_at_horizon_index(grasp_result.js_solution, -1).squeeze(0)
        lift_start = self.kinematics.get_active_js(lift_start)
        lift_start.dt = None

        lift_offset = Pose.from_list(
            [grasp_lift_offset * v for v in _axis_string_to_vector(grasp_lift_axis)] + [1, 0, 0, 0]
        )

        lift_poses_dict = {}
        for frame, goal_pose in grasp_poses_dict.items():
            if grasp_lift_in_tool_frame:
                lift_poses_dict[frame] = goal_pose.multiply(lift_offset)
            else:
                lift_poses_dict[frame] = lift_offset.multiply(goal_pose)

        lift_tool_pose = GoalToolPose.from_poses(
            lift_poses_dict,
            ordered_tool_frames=grasp_poses.tool_frames,
            num_goalset=1,
        )
        lift_result = self._grasp_phase_planner.plan_linear_phase(
            lift_tool_pose,
            lift_start,
            grasp_lift_axis,
            grasp_lift_in_tool_frame,
            use_rrt=use_rrt_for_constrained_phases,
            finetune_attempts=int(use_double_pass_trajopt_for_constrained_phases),
        )
        self._grasp_phase_planner.activate_collision(disable_collision_links, enable=True)

        if lift_result is None or not lift_result.success.any():
            result.status = "Planning to lift pose failed."
            result.success[:] = False
            result.lift_success[:] = False
            return result

        result.lift_trajectory = lift_result.js_solution
        result.lift_trajectory_dt = lift_result.js_solution.dt
        result.lift_interpolated_trajectory = lift_result.interpolated_trajectory
        result.lift_interpolated_last_tstep = lift_result.interpolated_last_tstep
        result.success = lift_result.success.clone()
        result.lift_success = lift_result.success.clone()
        result.status = "Planning to lift pose succeeded."
        return result

    # -- Scene / collision management --

    def enable_link_collision(self, enable_collision_links: List[str]):
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

    def update_world(self, scene_cfg: SceneCfg):
        """Update fixed world storage and clear graph-planner nodes only."""
        self.scene_collision_checker.load_collision_model(scene_cfg)
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if constrained_rrt_planner is not None:
            constrained_rrt_planner.reset_buffer()

    def clear_scene_cache(self):
        """Clear fixed world storage and graph-planner nodes, retaining captures."""
        self.scene_collision_checker.clear_cache()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
        constrained_rrt_planner = getattr(self, "constrained_rrt_planner", None)
        if constrained_rrt_planner is not None:
            constrained_rrt_planner.reset_buffer()

    def reset_seed(self):
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner.reset_seed()
        if self.constrained_rrt_planner is not None:
            self.constrained_rrt_planner.reset_seed()

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        """Update one link's inertial properties in IK, RRT, and TrajOpt."""
        self.ik_solver.update_link_inertial(link_name, mass, com, inertia)
        if self.graph_planner is not None and is_rrt_graph_planner_algorithm(
            self.config.graph_planner_algorithm
        ):
            self.graph_planner.update_link_inertial(link_name, mass, com, inertia)
        if self.constrained_rrt_planner is not None:
            self.constrained_rrt_planner.update_link_inertial(link_name, mass, com, inertia)
        self.trajopt_solver.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self,
        link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        """Update multiple links' inertial properties in IK, RRT, and TrajOpt."""
        self.ik_solver.update_links_inertial(link_properties)
        if self.graph_planner is not None and is_rrt_graph_planner_algorithm(
            self.config.graph_planner_algorithm
        ):
            self.graph_planner.update_links_inertial(link_properties)
        if self.constrained_rrt_planner is not None:
            self.constrained_rrt_planner.update_links_inertial(link_properties)
        self.trajopt_solver.update_links_inertial(link_properties)

    def update_tool_pose_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ) -> None:
        grasp_phase_planner = getattr(self, "_grasp_phase_planner", None)
        if grasp_phase_planner is not None:
            grasp_phase_planner.invalidate_tool_pose_criteria()
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.trajopt_solver.update_tool_pose_criteria(tool_pose_criteria)

    def update_tool_relative_transform_criteria(
        self,
        criteria: Dict[str, ToolRelativeTransformCriteria],
    ) -> None:
        """Update fixed-reference transform criteria in IK and TrajOpt rollouts."""
        self.ik_solver.update_tool_relative_transform_criteria(criteria)
        self.trajopt_solver.update_tool_relative_transform_criteria(criteria)

    def update_tool_velocity_limit_criteria(
        self,
        criteria: Dict[str, ToolVelocityLimitCriteria],
    ) -> None:
        """Update tool-speed limits in time-parameterized TrajOpt rollouts."""
        self.trajopt_solver.update_tool_velocity_limit_criteria(criteria)
