# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot cost manager that aggregates and evaluates all cost terms for a rollout.

Constructs cost objects (tool-pose, collision, self-collision, cspace, cspace-dist)
from a RobotCostManagerCfg, computes per-step costs given a RobotState, and exposes
methods to enable/disable individual cost terms at runtime for goal switching.
"""
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.geom.collision import SceneCollision
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import CostCollection
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)
from curobo._src.util.logging import log_and_raise, log_info

if TYPE_CHECKING:
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg


class RobotCostManager:
    """Registers and evaluates cost components for robot motion planning.

    Maintains a registry of named :class:`BaseCost` instances (collision,
    c-space distance, tool pose, etc.) with per-cost CUDA streams, and
    evaluates them in parallel to produce the aggregate cost used by the
    optimizer.
    """

    def __init__(self, device_cfg: DeviceCfg = DeviceCfg()):
        self.costs: Dict[str, BaseCost] = {}
        self.device_cfg = device_cfg
        self.config: Optional[RobotCostManagerCfg] = None
        self._initialized = False
        self._batch_size: Optional[int] = None
        self._horizon: Optional[int] = None
        self._costs_streams: Dict[str, Any] = {}
        self._costs_events: Dict[str, Any] = {}

    # -- Registry --

    def register_cost(self, name: str, component: BaseCost) -> None:
        """Add a named cost component and allocate its CUDA stream pair."""
        if name in self.costs:
            log_and_raise(f"Component {name} already registered")
        self.costs[name] = component
        self._costs_streams[name], self._costs_events[name] = create_cuda_stream_pair(
            self.device_cfg.device
        )

    def get_cost(self, name: str) -> Optional[BaseCost]:
        return self.costs.get(name)

    def has_cost(self, name: str) -> bool:
        return name in self.costs

    def enable_cost_component(self, name: str) -> None:
        """Enable a previously registered cost component by name."""
        if name not in self.costs:
            log_and_raise(f"Cost component {name} not found")
        self.costs[name].enable_cost()

    def disable_cost_component(self, name: str) -> None:
        """Disable a registered cost component so it is skipped during evaluation."""
        if name not in self.costs:
            log_and_raise(f"Cost component {name} not found")
        self.costs[name].disable_cost()

    def get_enabled_costs(self) -> List[str]:
        return [n for n, c in self.costs.items() if c.enabled]

    def get_cost_component_names(self) -> List[str]:
        return list(self.costs.keys())

    def get_cost_components(self) -> Dict[str, BaseCost]:
        return self.costs

    # -- Batch lifecycle --

    def setup_batch_tensors(self, batch_size: int, horizon: int) -> None:
        reinitialize = (
            self._batch_size is None
            or self._horizon is None
            or self._batch_size != batch_size
            or self._horizon != horizon
        )
        if not reinitialize:
            return
        for component in self.costs.values():
            component.setup_batch_tensors(batch_size, horizon)
        self._batch_size = batch_size
        self._horizon = horizon

    def reset(self, reset_problem_ids: Optional[torch.Tensor] = None, **kwargs) -> None:
        for component in self.costs.values():
            component.reset(reset_problem_ids=reset_problem_ids, **kwargs)

    def update_dt(self, dt: float) -> None:
        for component in self.costs.values():
            component.update_dt(dt)

    # -- Initialize from config --

    def initialize_from_config(
        self,
        config: RobotCostManagerCfg,
        transition_model: RobotStateTransition,
        scene_collision_checker: Optional[SceneCollision] = None,
        **kwargs,
    ) -> None:
        """Create and register all cost components from a configuration.

        Inspects each optional sub-config (self-collision, scene-collision,
        c-space bounds, tool-pose tracking, start/target c-space distance)
        and, when present, builds the corresponding :class:`BaseCost`,
        wires it to the ``transition_model`` and ``scene_collision_checker``,
        and registers it under a canonical name.  Costs whose robot model
        has zero spheres are registered but disabled.

        Args:
            config: Manager configuration carrying per-cost sub-configs.
            transition_model: Provides the robot model, interpolation
                steps, and kinematics needed by individual cost terms.
            scene_collision_checker: Shared collision checker instance.
                Required for scene-collision costs; ignored when ``None``.
        """
        self.config = config
        robot_model = transition_model.robot_model

        # Self-collision
        self_collision_kin_config = robot_model.get_self_collision_config()
        if config.self_collision_cfg is not None and self_collision_kin_config is not None:
            config.self_collision_cfg.self_collision_kin_config = self_collision_kin_config
            if transition_model.interpolation_steps > 1:
                config.self_collision_cfg.weight = (
                    config.self_collision_cfg.weight / transition_model.interpolation_steps
                )
            self_collision_cost = SelfCollisionCost(config.self_collision_cfg)
            if robot_model.total_spheres == 0:
                self_collision_cost.disable_cost()
            self.register_cost("self_collision", self_collision_cost)

        # Scene collision
        if config.scene_collision_cfg is not None and scene_collision_checker is not None:
            config.scene_collision_cfg.scene_collision_checker = scene_collision_checker
            config.scene_collision_cfg.update_num_spheres(robot_model.total_spheres)
            scene_collision_cost = SceneCollisionCost(config.scene_collision_cfg)
            if robot_model.total_spheres == 0:
                scene_collision_cost.disable_cost()
            self.register_cost("scene_collision", scene_collision_cost)

        # Cspace bounds/limits
        if config.cspace_cfg is not None:
            config.cspace_cfg.initialize_from_transition_model(transition_model)
            self.register_cost("cspace", config.cspace_cfg.class_type(config.cspace_cfg))

        # Tool pose tracking
        if config.tool_pose_cfg is not None:
            config.tool_pose_cfg.set_tool_frames(transition_model.robot_model.tool_frames)
            self.register_cost("tool_pose", ToolPoseCost(config.tool_pose_cfg))

        # Start cspace distance
        if config.start_cspace_dist_cfg is not None:
            config.start_cspace_dist_cfg.initialize_from_transition_model(transition_model)
            self.register_cost("start_cspace_dist", CSpaceDistCost(config.start_cspace_dist_cfg))

        # Target cspace distance
        if config.target_cspace_dist_cfg is not None:
            config.target_cspace_dist_cfg.initialize_from_transition_model(transition_model)
            self.register_cost("target_cspace_dist", CSpaceDistCost(config.target_cspace_dist_cfg))

        self._initialized = True
        log_info(f"Initialized {len(self.costs)} costs for robot rollout")

    # -- Compute costs --

    def compute_costs(
        self,
        state: RobotState,
        cost_collection: Optional[CostCollection] = None,
        goal: Optional[GoalRegistry] = None,
        **kwargs,
    ) -> CostCollection:
        """Evaluate all enabled cost components on a robot state trajectory.

        Each cost (tool-pose, c-space, self-collision, scene-collision) is
        evaluated in its own CUDA stream for overlap, then all streams are
        synchronized before the collection is returned.  Batch tensors are
        re-allocated if the state shape has changed since the last call.

        Args:
            state: Robot state containing ``joint_state``
                ``(batch, horizon, n_dof)``, ``tool_poses``,
                ``robot_spheres``, and ``cuda_robot_model_state``.
            cost_collection: Existing collection to append to.  A new
                :class:`CostCollection` is created when ``None``.
            goal: Goal registry supplying target poses, joint states,
                and environment indices for the cost terms.

        Returns:
            :class:`CostCollection` with one entry per evaluated cost.
        """
        batch_size = state.joint_state.shape[0]
        horizon = state.joint_state.shape[1]
        if self._batch_size != batch_size or self._horizon != horizon:
            self.setup_batch_tensors(batch_size, horizon)
        if cost_collection is None:
            cost_collection = CostCollection()

        # Tool pose
        if self.has_cost("tool_pose") and goal is not None and goal.link_goal_poses is not None:
            tool_pose_cost = self.get_cost("tool_pose")
            if tool_pose_cost.enabled:
                with self._stream_context("tool_pose"):
                    cost_value, _, _, _ = tool_pose_cost.forward(
                        state.tool_poses,
                        goal.link_goal_poses,
                        goal.idxs_link_pose,
                    )
                    cost_collection.add(cost_value, "tool_pose")

        # Cspace bounds/limits
        if self.has_cost("cspace"):
            cspace_cost = self.get_cost("cspace")
            if cspace_cost.enabled:
                with self._stream_context("cspace"):
                    cost_value = cspace_cost.forward(
                        state.joint_state,
                        joint_torque=state.joint_torque,
                        target_joint_state=goal.goal_js if goal is not None else None,
                        idxs_target_joint_state=goal.idxs_goal_js if goal is not None else None,
                        current_joint_state=goal.current_js if goal is not None else None,
                        idxs_current_joint_state=(
                            goal.idxs_current_js if goal is not None else None
                        ),
                        current_state_dt=(
                            goal.current_state_dt if goal is not None else None
                        ),
                    )
                    cost_collection.add(cost_value, "cspace")

        # Self-collision
        if self.has_cost("self_collision"):
            self_collision_cost = self.get_cost("self_collision")
            if self_collision_cost.enabled:
                with self._stream_context("self_collision"):
                    cost_value = self_collision_cost.forward(state.robot_spheres)
                    cost_collection.add(cost_value, "self_collision")

        # Scene collision
        if self.has_cost("scene_collision"):
            scene_collision_cost = self.get_cost("scene_collision")
            if scene_collision_cost.enabled:
                with self._stream_context("scene_collision"):
                    idxs_env = (
                        goal.idxs_env.view(-1)
                        if goal is not None and goal.idxs_env is not None
                        else None
                    )
                    cost_value = scene_collision_cost.forward(
                        state.cuda_robot_model_state,
                        idxs_env,
                        trajectory_dt=state.joint_state.dt,
                    )
                    cost_collection.add(cost_value, "scene_collision")

        synchronize_cuda_streams(self._costs_events, self.device_cfg.device)
        return cost_collection

    # -- Compute convergence --

    def compute_convergence(
        self,
        state: RobotState,
        goal: Optional[GoalRegistry] = None,
        **kwargs,
    ) -> CostCollection:
        """Compute convergence tolerances for solver termination checks.

        Evaluates start/target c-space distance and tool-pose
        position/orientation errors when the corresponding costs are
        registered and enabled.  These scalar tolerance values let the
        solver decide whether a trajectory has converged.

        Args:
            state: Robot state trajectory with ``joint_state``
                ``(batch, horizon, n_dof)`` and ``tool_poses``.
            goal: Goal registry providing target joint states, current
                joint states, and link goal poses for distance
                computation.

        Returns:
            :class:`CostCollection` keyed by tolerance names such as
            ``"start_cspace_dist_tolerance"``,
            ``"target_cspace_dist_tolerance"``,
            ``"tool_pose_position_tolerance"``,
            ``"tool_pose_orientation_tolerance"``, and
            ``"tool_pose_goalset_index"``.
        """
        convergence = CostCollection()
        batch_size = state.joint_state.shape[0]
        horizon = state.joint_state.shape[1]
        if self._batch_size != batch_size or self._horizon != horizon:
            self.setup_batch_tensors(batch_size, horizon)

        if goal is None:
            return convergence

        if self.has_cost("start_cspace_dist") and goal.current_js is not None:
            cost = self.get_cost("start_cspace_dist")
            if cost.enabled:
                value = cost.forward(
                    state.joint_state.position,
                    goal.current_js.position,
                    goal.idxs_current_js.view(-1),
                )
                convergence.add(value, "start_cspace_dist_tolerance")

        if self.has_cost("target_cspace_dist") and goal.goal_js is not None:
            cost = self.get_cost("target_cspace_dist")
            if cost.enabled:
                value = cost.forward(
                    state.joint_state.position,
                    goal.goal_js.position,
                    goal.idxs_goal_js.view(-1),
                )
                convergence.add(value, "target_cspace_dist_tolerance")

        if self.has_cost("tool_pose") and goal.link_goal_poses is not None:
            tool_pose_cost = self.get_cost("tool_pose")
            if tool_pose_cost.enabled:
                _, position_error, rotation_error, goalset_idx = tool_pose_cost.forward(
                    state.tool_poses,
                    goal.link_goal_poses,
                    goal.idxs_link_pose,
                )
                convergence.add(position_error, "tool_pose_position_tolerance")
                convergence.add(rotation_error, "tool_pose_orientation_tolerance")
                convergence.add(goalset_idx, "tool_pose_goalset_index")

        return convergence

    # -- Update params --

    def update_params(self, **kwargs) -> None:
        if not self._initialized:
            return
        if "tool_pose_criteria" in kwargs:
            tool_pose_criteria = kwargs["tool_pose_criteria"]
            tool_pose_cost = self.get_cost("tool_pose")
            if not isinstance(tool_pose_criteria, dict):
                log_and_raise(
                    f"tool_pose_criteria must be a dict, got {type(tool_pose_criteria)}"
                )
                return
            if tool_pose_cost is not None:
                tool_pose_cost.update_tool_pose_criteria(tool_pose_criteria)
        if "dt" in kwargs:
            self.update_dt(kwargs["dt"])

    # -- CUDA stream helper --

    def _stream_context(self, cost_name: str):
        return cuda_stream_context(
            cost_name, self._costs_streams, self._costs_events, self.device_cfg.device
        )
