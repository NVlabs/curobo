# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared linear waypoint planning for single and batched grasps."""

from __future__ import annotations

# Standard Library
import math
from typing import Callable, Dict, List, Optional, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_pose_reference import validate_pose_tolerance
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise


def axis_string_to_vector(axis: str) -> List[float]:
    """Convert an axis name to an xyz unit vector."""
    if axis == "x":
        return [1.0, 0.0, 0.0]
    if axis == "y":
        return [0.0, 1.0, 0.0]
    if axis == "z":
        return [0.0, 0.0, 1.0]
    log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")


class GraspPhasePlanner:
    """Plan linear waypoint phases and cache grasp-contact collision updates."""

    def __init__(
        self,
        config: MotionPlannerCfg,
        kinematics: Kinematics,
        plan_pose_waypoints: Callable[..., Optional[TrajOptSolverResult]],
        max_attempts: int,
        enable_link_collision: Callable[[List[str]], None],
        disable_link_collision: Callable[[List[str]], None],
    ) -> None:
        """Share waypoint planning with single and batched motion planners."""
        self.config = config
        self.device_cfg: DeviceCfg = config.device_cfg
        self.kinematics = kinematics
        self._plan_pose_waypoints = plan_pose_waypoints
        self._max_attempts = max_attempts
        self._enable_link_collision = enable_link_collision
        self._disable_link_collision = disable_link_collision
        self._update_executors: Dict[tuple, GraphExecutor] = {}

    def destroy(self) -> None:
        """Release CUDA graphs owned by cached collision updates."""
        for executor in self._update_executors.values():
            executor.reset()
        self._update_executors.clear()

    def _create_update_executor(
        self,
        key: tuple,
        update: Callable[[], None],
    ) -> GraphExecutor:
        """Capture and cache one fixed-address grasp-state update."""
        executor = GraphExecutor(
            capture_fn=update,
            device=self.device_cfg.device,
            use_cuda_graph=self.config.trajopt_solver_config.use_cuda_graph,
            clone_outputs=False,
            clone_inputs=False,
        )
        executor.warmup()
        self._update_executors[key] = executor
        return executor

    def activate_collision(self, links: List[str], *, enable: bool) -> None:
        """Activate one cached grasp-contact collision state."""
        if not links:
            return
        stable_links = tuple(links)
        key = ("collision", stable_links, enable)
        executor = self._update_executors.get(key)
        if executor is None:
            if self.device_cfg.device.type == "cuda":
                kinematics_config = self.kinematics.config.kinematics_config
                sphere_indices = torch.cat(
                    [
                        kinematics_config.get_sphere_index_from_link_name(link)
                        for link in stable_links
                    ]
                )
                target_radii = kinematics_config.link_spheres[..., 3]
                if enable:
                    source_radii = kinematics_config.reference_link_spheres[..., 3].index_select(
                        1,
                        sphere_indices,
                    )
                else:
                    source_radii = torch.full(
                        (target_radii.shape[0], sphere_indices.shape[0]),
                        -100.0,
                        device=target_radii.device,
                        dtype=target_radii.dtype,
                    )

                def update() -> None:
                    target_radii.index_copy_(1, sphere_indices, source_radii)

            elif enable:

                def update() -> None:
                    self._enable_link_collision(list(stable_links))

            else:

                def update() -> None:
                    self._disable_link_collision(list(stable_links))

            executor = self._create_update_executor(key, update)
        executor.replay(clone_outputs=False)

    @staticmethod
    def validate_linear_path_tolerance(tolerance: Tuple[float, float]) -> None:
        """Reject invalid position/orientation bounds before changing planner state."""
        validate_pose_tolerance(tolerance, context="path")

    def plan_linear_phase(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        finetune_attempts: int,
        path_tolerance: Tuple[float, float] = (0.02, math.radians(10.0)),
    ) -> Optional[TrajOptSolverResult]:
        """Track the interpolated start-to-end pose reference with the grasp iteration budget."""
        self.validate_linear_path_tolerance(path_tolerance)
        config = self.config.trajopt_solver_config
        return self._plan_pose_waypoints(
            goal_tool_poses,
            current_state,
            path_tolerance=path_tolerance,
            final_tolerance=(config.position_tolerance, config.orientation_tolerance),
            max_attempts=self._max_attempts,
            finetune_attempts=finetune_attempts,
            outer_iterations=1,
            track_orientation=True,
        )
