# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Differentiable robot-scene collision checking for use in PyTorch pipelines."""

from __future__ import annotations

# Standard Library
from typing import Optional, Tuple

# Third Party
import torch

from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.types import SceneCfg

# CuRobo
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float16_tensors, check_float32_tensors
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class RobotSceneCollision(RobotSceneCollisionCfg):
    """Differentiable robot-scene collision checker.

    This class provides differentiable collision checking between a robot and
    the environment, suitable for use in optimization-based motion planning.

    Example:
        ```python
        from curobo._src.collision import RobotSceneCollision, RobotSceneCollisionCfg

        cfg = RobotSceneCollisionCfg.load_from_config(
            robot_config="franka.yml",
            scene_model="scene.yml",
        )
        checker = RobotSceneCollision(cfg)

        # Check collision at joint configurations
        joint_positions = torch.rand((10, 7))
        scene_dist, self_dist = checker.get_scene_self_collision_distance_from_joints(
            joint_positions
        )
        ```
    """

    def __init__(self, config: RobotSceneCollisionCfg) -> None:
        """Initialize the collision checker.

        Args:
            config: Configuration for the collision checker.
        """
        RobotSceneCollisionCfg.__init__(self, **vars(config))
        self._batch_pose_idx = None
        self._camera_projection_rays = None
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        self.cspace_cost.setup_batch_tensors(batch_size, horizon)
        if self.self_collision_cost is not None:
            self.self_collision_cost.setup_batch_tensors(batch_size, horizon)
        if self.collision_cost is not None:
            self.collision_cost.setup_batch_tensors(batch_size, horizon)
        if self.collision_constraint is not None:
            self.collision_constraint.setup_batch_tensors(batch_size, horizon)
    def get_kinematics(self, joint_position: torch.Tensor) -> KinematicsState:
        """Compute forward kinematics for joint positions.

        Args:
            joint_position: Joint positions of shape [batch, horizon, dof].

        Returns:
            Robot state including link poses and collision spheres.

        Raises:
            ValueError: If joint_position is not 2D tensor.
        """
        batch, horizon, dof = joint_position.shape

        state = self.kinematics.compute_kinematics(
            JointState.from_position(joint_position, joint_names=self.kinematics.joint_names)
        )
        return state

    def update_world(self, scene_cfg: SceneCfg) -> None:
        """Update the scene collision model.

        Args:
            scene_cfg: New scene configuration to load.
        """
        self.scene_model.load_collision_model(scene_cfg)

    def clear_scene_cache(self) -> None:
        """Clear the scene collision cache."""
        self.scene_model.clear_cache()

    def get_collision_distance(
        self, x_sph: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get scene collision distance for robot spheres.

        Args:
            x_sph: Robot collision spheres [batch, horizon, num_spheres, 4].
            env_query_idx: Optional environment indices for batched environments.

        Returns:
            Collision distance tensor.
        """
        d = self.collision_cost.forward(x_sph, idxs_env_query=env_query_idx)
        return d

    def get_collision_constraint(
        self, x_sph: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get scene collision constraint value for robot spheres.

        Args:
            x_sph: Robot collision spheres [batch, horizon, num_spheres, 4].
            env_query_idx: Optional environment indices for batched environments.

        Returns:
            Collision constraint tensor.
        """
        d = self.collision_constraint.forward(x_sph, idxs_env_query=env_query_idx)
        return d

    def get_self_collision_distance(self, x_sph: torch.Tensor) -> torch.Tensor:
        """Get self-collision distance for robot spheres.

        Args:
            x_sph: Robot collision spheres [batch, horizon, num_spheres, 4].

        Returns:
            Self-collision distance tensor.
        """
        return self.get_self_collision(x_sph)

    def get_self_collision(self, x_sph: torch.Tensor) -> torch.Tensor:
        """Get self-collision cost for robot spheres.

        Args:
            x_sph: Robot collision spheres [batch, horizon, num_spheres, 4].

        Returns:
            Self-collision cost tensor.
        """
        d = self.self_collision_cost.forward(x_sph)
        return d

    def get_collision_vector(
        self, x_sph: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get collision distance and gradient vector.

        Note: This function is not differentiable.

        Args:
            x_sph: Robot collision spheres.
            env_query_idx: Optional environment indices.

        Returns:
            Tuple of (collision distance, gradient vector).
        """
        x_sph.requires_grad = True
        d = self.collision_cost.forward(x_sph, idxs_env_query=env_query_idx)
        vec = self.collision_cost.get_gradient_buffer()
        d = d.detach()
        x_sph.requires_grad = False
        x_sph.grad = None
        return d, vec

    def get_scene_self_collision_distance_from_joints(
        self,
        q: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get scene and self-collision distances from joint positions.

        Args:
            q: Joint positions [batch, dof].
            env_query_idx: Optional environment indices.

        Returns:
            Tuple of (scene collision distance, self-collision distance).
        """
        state = self.get_kinematics(q)
        d_world = self.get_collision_distance(
            state.robot_spheres, env_query_idx=env_query_idx
        )
        d_self = self.get_self_collision_distance(state.robot_spheres)
        return d_world, d_self

    def get_scene_self_collision_distance_from_joint_trajectory(
        self,
        q: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get collision distances for a trajectory of joint positions.

        Args:
            q: Joint trajectory [batch, horizon, dof].
            env_query_idx: Optional environment indices.

        Returns:
            Tuple of (scene collision distance, self-collision distance).
        """
        b, h, dof = q.shape
        state = self.get_kinematics(q.view(b * h, dof))
        x_sph = state.robot_spheres.view(b, h, -1, 4)
        d_world = self.get_collision_distance(x_sph, env_query_idx=env_query_idx)
        d_self = self.get_self_collision_distance(x_sph)
        return d_world, d_self

    def get_bound(
        self, q: torch.Tensor, q_tau: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get joint bound violation cost.

        Args:
            q: Joint positions. [batch, horizon, dof]
            q_tau: Optional joint torques. [batch, horizon, dof]

        Returns:
            Bound violation cost. [batch, horizon]
        """
        if q_tau is None:
            q_tau = q.clone() * 0.0
        batch, horizon, dof = q.shape
        self.cspace_cost.setup_batch_tensors(batch, horizon)
        d = self.cspace_cost.forward(JointState(position=q), joint_torque=q_tau)
        return d

    def sample(
        self,
        n: int,
        mask_valid: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample collision-free joint configurations.

        Note: This does not support batched environments, use sample_trajectory instead.

        Args:
            n: Number of samples to generate.
            mask_valid: Whether to filter for valid configurations.
            env_query_idx: Optional environment indices.

        Returns:
            Sampled joint configurations [n, dof].
        """
        n_samples = n
        if mask_valid:
            n_samples = n * self.rejection_ratio
        q = self.sampler.get_samples(n_samples, bounded=True)
        if mask_valid:
            q_mask = self.validate(q, env_query_idx)
            q = q[q_mask][:n]
        return q

    def validate(
        self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Validate joint configurations for collisions.

        Note: This does not support batched environments, use validate_trajectory instead.

        Args:
            q: Joint positions [batch, dof].
            env_query_idx: Optional environment indices.

        Returns:
            Boolean mask of valid configurations.
        """
        b, h, dof = q.shape
        self.setup_batch_tensors(b, h)
        kin_state = self.get_kinematics(q)
        spheres = kin_state.robot_spheres.view(b, 1, -1, 4)
        d_bound = self.get_bound(q.view(b, 1, dof))

        sum_list = [d_bound.sum(dim=-1,keepdim=True)]
        if self.self_collision_cost is not None:
            d_self = self.get_self_collision(spheres)
            sum_list.append(d_self.view(b,h, 1))
        if self.collision_constraint is not None:
            d_world = self.get_collision_constraint(spheres, env_query_idx)
            sum_list.append(d_world.view(b,h, 1))
        d_mask = torch.sum(torch.stack(sum_list, dim=-1), dim=-1) == 0.0
        return d_mask.view(b, h)

    def sample_trajectory(
        self,
        batch: int,
        horizon: int,
        mask_valid: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample collision-free trajectories.

        Args:
            batch: Number of trajectories to sample.
            horizon: Length of each trajectory.
            mask_valid: Whether to filter for valid configurations.
            env_query_idx: Optional environment indices.

        Returns:
            Sampled trajectories [batch, horizon, dof].
        """
        n_samples = batch * horizon
        if mask_valid:
            n_samples = batch * horizon * self.rejection_ratio
        q = self.sampler.get_samples(n_samples, bounded=True)
        q = q.reshape(batch, horizon * self.rejection_ratio, -1)
        if mask_valid:
            q_mask = self.validate_trajectory(q, env_query_idx)
            q = [q[i][q_mask[i, :], :][:horizon, :].unsqueeze(0) for i in range(batch)]
            q = torch.cat(q)
        return q

    def validate_trajectory(
        self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Validate trajectory for collisions.

        Args:
            q: Joint trajectory [batch, horizon, dof].
            env_query_idx: Optional environment indices [batch, 1].

        Returns:
            Boolean mask of valid configurations.
        """
        b, h, dof = q.shape
        q = q.view(b * h, dof)
        kin_state = self.get_kinematics(q)
        spheres = kin_state.robot_spheres.view(b, h, -1, 4)
        d_self = self.get_self_collision(spheres)
        d_world = self.get_collision_constraint(spheres, env_query_idx)
        d_bound = self.get_bound(q.view(b, h, dof))
        d_mask = _mask(d_self, d_world, d_bound)
        return d_mask

    def pose_distance(
        self, x_des: Pose, x_current: Pose, resize: bool = False
    ) -> torch.Tensor:
        """Calculate pose distance.

        Args:
            x_des: Desired pose.
            x_current: Current pose.
            resize: Whether to resize output.

        Returns:
            Pose distance tensor.
        """
        unsqueeze = False
        if len(x_current.position.shape) == 2:
            x_current = x_current.unsqueeze(1)
            unsqueeze = True
        if (
            self._batch_pose_idx is None
            or self._batch_pose_idx.shape[0] != x_current.position.shape[0]
        ):
            self._batch_pose_idx = torch.arange(
                0,
                x_current.position.shape[0],
                1,
                device=self.device_cfg.device,
                dtype=torch.int32,
            )
        distance = self.pose_cost.forward_pose(x_des, x_current, self._batch_pose_idx)
        if unsqueeze and resize:
            distance = distance.squeeze(1)
        return distance

    def get_point_robot_distance(
        self, points: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance from robot to points (e.g., pointcloud).

        Args:
            points: Point cloud [n_points, 3] or [batch, n_points, 3].
            q: Joint positions [1, dof].

        Returns:
            Distance tensor [n_points] or [batch, n_points]. Positive values
            indicate collision.

        Note:
            This currently does not support batched robot configurations.
        """
        if len(q.shape) == 1:
            log_and_raise("q should be of shape [b, dof]")
        kin_state = self.get_kinematics(q)

        robot_spheres = kin_state.robot_spheres.view(
            kin_state.robot_spheres.shape[0], -1, 4
        )
        if robot_spheres.shape[0] != 1:
            if robot_spheres.shape[0] != points.shape[0]:
                log_and_raise(
                    "robot_spheres batch must be 1 or match points batch: "
                    f"got {robot_spheres.shape[0]} vs {points.shape[0]}"
                )
        if robot_spheres.dtype == torch.float16:
            check_float16_tensors(robot_spheres.device, robot_spheres=robot_spheres)
        else:
            check_float32_tensors(robot_spheres.device, robot_spheres=robot_spheres)

        squeeze_shape = False
        n = 1
        if len(points.shape) == 2:
            squeeze_shape = True
            n = points.shape[0]
            points = points.unsqueeze(0)

        pt_distance = _point_robot_distance(robot_spheres, points)
        if squeeze_shape:
            pt_distance = pt_distance.view(n)
        return pt_distance

    def get_active_js(self, full_js: JointState) -> JointState:
        """Get active joint state from full joint state.

        Args:
            full_js: Full joint state with all joints.

        Returns:
            Joint state with only active joints.
        """
        active_jnames = self.kinematics.joint_names
        out_js = full_js.reorder(active_jnames)
        return out_js

    @property
    def tool_frames(self):
        """Get robot link names."""
        return self.kinematics.tool_frames



@get_torch_jit_decorator()
def _mask(d1: torch.Tensor, d2: torch.Tensor, d3: torch.Tensor) -> torch.Tensor:
    """Sum costs and return validity mask."""
    d_total = d1 + d2 + d3
    d_mask = d_total == 0.0
    return d_mask


@get_torch_jit_decorator()
def _point_robot_distance(
    robot_spheres: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Compute distance between robot spheres and points.

    Caller must ensure shapes and dtypes are validated and normalized:
    ``robot_spheres`` has shape ``[batch_robot, n_robot_spheres, 4]`` and
    ``points`` has shape ``[batch_points, n_points, 3]``. ``batch_robot`` must
    be 1 or equal to ``batch_points``.

    Args:
        robot_spheres: Robot spheres [batch_robot, n_robot_spheres, 4].
        points: Point cloud [batch_points, n_points, 3].

    Returns:
        Distance tensor [batch_points, n_points].
    """
    robot_spheres = robot_spheres.unsqueeze(-3)
    robot_radius = robot_spheres[..., 3]
    points = points.unsqueeze(-2)
    sph_distance = -1 * (
        torch.linalg.norm(points - robot_spheres[..., :3], dim=-1) - robot_radius
    )
    return torch.max(sph_distance, dim=-1)[0]

