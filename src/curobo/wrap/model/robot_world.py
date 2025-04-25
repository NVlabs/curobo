#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""This module has differentiable layers built from CuRobo's core features for use in Pytorch."""

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Third Party
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.bound_cost import BoundCost, BoundCostConfig, BoundCostType
from curobo.rollout.cost.pose_cost import PoseCost, PoseCostConfig, PoseErrorType
from curobo.rollout.cost.primitive_collision_cost import (
    PrimitiveCollisionCost,
    PrimitiveCollisionCostConfig,
)
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error
from curobo.util.sample_lib import HaltonGenerator
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.warp import init_warp
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml


@dataclass
class RobotWorldConfig:
    kinematics: CudaRobotModel
    sampler: HaltonGenerator
    bound_scale: torch.Tensor
    bound_cost: BoundCost
    pose_cost: PoseCost
    self_collision_cost: Optional[SelfCollisionCost] = None
    collision_cost: Optional[PrimitiveCollisionCost] = None
    collision_constraint: Optional[PrimitiveCollisionCost] = None
    world_model: Optional[WorldCollision] = None
    rejection_ratio: int = 10
    tensor_args: TensorDeviceType = TensorDeviceType()
    contact_distance: float = 0.0

    @staticmethod
    def load_from_config(
        robot_config: Union[RobotConfig, str] = "franka.yml",
        world_model: Union[None, str, Dict, WorldConfig, List[WorldConfig], List[str]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        n_envs: int = 1,
        n_meshes: int = 50,
        n_cuboids: int = 50,
        collision_activation_distance: float = 0.2,
        self_collision_activation_distance: float = 0.0,
        max_collision_distance: float = 1.0,
        collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH,
        world_collision_checker: Optional[WorldCollision] = None,
        pose_weight: List[float] = [1, 1, 1, 1],
    ):
        init_warp(tensor_args=tensor_args)
        world_collision_cost = self_collision_cost = world_collision_constraint = None
        if isinstance(robot_config, str):
            robot_config = load_yaml(join_path(get_robot_configs_path(), robot_config))["robot_cfg"]
        if isinstance(robot_config, Dict):
            if "robot_cfg" in robot_config:
                robot_config = robot_config["robot_cfg"]
            robot_config = RobotConfig.from_dict(robot_config, tensor_args)
        kinematics = CudaRobotModel(robot_config.kinematics)

        if isinstance(world_model, str):
            world_model = load_yaml(join_path(get_world_configs_path(), world_model))
        if isinstance(world_model, List):
            if isinstance(world_model[0], str):
                world_model = [
                    load_yaml(join_path(get_world_configs_path(), x)) for x in world_model
                ]
        if world_collision_checker is None and world_model is not None:
            world_cfg = WorldCollisionConfig.load_from_dict(
                {
                    "checker_type": collision_checker_type,
                    "cache": {"mesh": n_meshes, "obb": n_cuboids},
                    "max_distance": max_collision_distance,
                    "n_envs": n_envs,
                },
                world_model,
                tensor_args,
            )
            world_collision_checker = create_collision_checker(world_cfg)

        if world_collision_checker is not None:
            collision_cost_config = PrimitiveCollisionCostConfig(
                tensor_args.to_device([1.0]),
                tensor_args,
                return_loss=True,
                world_coll_checker=world_collision_checker,
                activation_distance=collision_activation_distance,
            )
            world_collision_cost = PrimitiveCollisionCost(collision_cost_config)
            collision_constraint_config = PrimitiveCollisionCostConfig(
                tensor_args.to_device([1.0]),
                tensor_args,
                return_loss=True,
                world_coll_checker=world_collision_checker,
                activation_distance=0.0,
            )
            world_collision_constraint = PrimitiveCollisionCost(collision_constraint_config)

        self_collision_config = SelfCollisionCostConfig(
            tensor_args.to_device([1.0]),
            tensor_args,
            return_loss=True,
            self_collision_kin_config=kinematics.get_self_collision_config(),
            distance_threshold=self_collision_activation_distance,
        )

        self_collision_cost = SelfCollisionCost(self_collision_config)
        bound_config = BoundCostConfig(
            tensor_args.to_device([1.0]),
            tensor_args,
            return_loss=True,
            cost_type=BoundCostType.POSITION,
            activation_distance=[0.0],
        )
        bound_config.set_bounds(kinematics.get_joint_limits(), teleport_mode=True)

        bound_cost = BoundCost(bound_config)
        sample_gen = HaltonGenerator(
            kinematics.get_dof(),
            tensor_args,
            up_bounds=kinematics.get_joint_limits().position[1],
            low_bounds=kinematics.get_joint_limits().position[0],
        )
        pose_cost_config = PoseCostConfig(
            tensor_args.to_device(pose_weight),
            tensor_args=tensor_args,
            terminal=False,
            return_loss=True,
            cost_type=PoseErrorType.BATCH_GOAL,
        )
        pose_cost = PoseCost(pose_cost_config)
        bound_scale = (
            kinematics.get_joint_limits().position[1] - kinematics.get_joint_limits().position[0]
        ).unsqueeze(0) / 2.0
        dist_threshold = 0.0
        if collision_activation_distance > 0.0:
            dist_threshold = (
                (0.5 / collision_activation_distance)
                * collision_activation_distance
                * collision_activation_distance
            )

        return RobotWorldConfig(
            kinematics,
            sample_gen,
            bound_scale,
            bound_cost,
            pose_cost,
            self_collision_cost,
            world_collision_cost,
            world_collision_constraint,
            world_collision_checker,
            tensor_args=tensor_args,
            contact_distance=dist_threshold,
        )


class RobotWorld(RobotWorldConfig):
    def __init__(self, config: RobotWorldConfig) -> None:
        RobotWorldConfig.__init__(self, **vars(config))
        self._batch_pose_idx = None
        self._camera_projection_rays = None

    def get_kinematics(self, q: torch.Tensor) -> CudaRobotModelState:
        if len(q.shape) == 1:
            log_error("q should be of shape [b, dof]")
        state = self.kinematics.get_state(q)
        return state

    def update_world(self, world_config: WorldConfig):
        self.world_model.load_collision_model(world_config)

    def clear_world_cache(self):
        self.world_model.clear_cache()

    def get_collision_distance(
        self, x_sph: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get collision distance

        Args:
            x_sph (torch.Tensor): batch, horizon, n_spheres, 4

        Returns:
            torch.Tensor: _description_
        """
        d = self.collision_cost.forward(x_sph, env_query_idx=env_query_idx)
        return d

    def get_collision_constraint(
        self, x_sph: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get collision distance

        Args:
            x_sph (torch.Tensor): batch, horizon, n_spheres, 4

        Returns:
            torch.Tensor: _description_
        """
        d = self.collision_constraint.forward(x_sph, env_query_idx=env_query_idx)
        return d

    def get_self_collision_distance(self, x_sph: torch.Tensor) -> torch.Tensor:
        return self.get_self_collision(x_sph)

    def get_self_collision(self, x_sph: torch.Tensor) -> torch.Tensor:
        d = self.self_collision_cost.forward(x_sph)
        return d

    def get_collision_vector(
        self, x_sph: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ):
        """
        NOTE: This function is not differentiable
        """
        x_sph.requires_grad = True
        d = self.collision_cost.forward(x_sph, env_query_idx=env_query_idx)
        vec = self.collision_cost.get_gradient_buffer()
        d = d.detach()
        x_sph.requires_grad = False
        x_sph.grad = None
        return d, vec

    def get_world_self_collision_distance_from_joints(
        self,
        q: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q : batch, dof
        """
        state = self.get_kinematics(q)
        d_world = self.get_collision_distance(
            state.link_spheres_tensor.unsqueeze(1), env_query_idx=env_query_idx
        ).squeeze(1)
        d_self = self.get_self_collision_distance(state.link_spheres_tensor.unsqueeze(1)).squeeze(1)
        return d_world, d_self

    def get_world_self_collision_distance_from_joint_trajectory(
        self,
        q: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q : batch, horizon, dof
        """
        b, h, dof = q.shape
        state = self.get_kinematics(q.view(b * h, dof))
        x_sph = state.link_spheres_tensor.view(b, h, -1, 4)
        d_world = self.get_collision_distance(x_sph, env_query_idx=env_query_idx)
        d_self = self.get_self_collision_distance(x_sph)
        return d_world, d_self

    def get_bound(self, q: torch.Tensor) -> torch.Tensor:
        d = self.bound_cost.forward(JointState(position=q))
        return d

    def sample(self, n: int, mask_valid: bool = True, env_query_idx: Optional[torch.Tensor] = None):
        """
        This does not support batched environments, use sample_trajectory instead.
        """
        n_samples = n
        if mask_valid:
            n_samples = n * self.rejection_ratio
        q = self.sampler.get_samples(n_samples, bounded=True)
        if mask_valid:
            q_mask = self.validate(q, env_query_idx)
            q = q[q_mask][:n]
        return q

    def validate(self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None):
        """
        This does not support batched environments, use validate_trajectory instead
        """
        # run collision, self collision, bounds
        b, dof = q.shape
        kin_state = self.get_kinematics(q)
        spheres = kin_state.link_spheres_tensor.view(b, 1, -1, 4)
        d_self = self.get_self_collision(spheres)
        d_world = self.get_collision_constraint(spheres, env_query_idx)
        d_bound = self.get_bound(q.view(b, 1, dof))
        d_mask = sum_mask(d_self, d_world, d_bound)
        return d_mask

    def sample_trajectory(
        self,
        batch: int,
        horizon: int,
        mask_valid: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
    ):
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

    def validate_trajectory(self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None):
        """
        q: batch , horizon, dof
        env_query_idx: batch, 1
        """
        # run collision, self collision, bounds
        b, h, dof = q.shape
        q = q.view(b * h, dof)
        kin_state = self.get_kinematics(q)
        spheres = kin_state.link_spheres_tensor.view(b, h, -1, 4)
        d_self = self.get_self_collision(spheres)
        d_world = self.get_collision_constraint(spheres, env_query_idx)
        d_bound = self.get_bound(q.view(b, h, dof))
        d_mask = mask(d_self, d_world, d_bound)
        return d_mask

    def pose_distance(self, x_des: Pose, x_current: Pose, resize: bool = False):
        unsqueeze = False
        if len(x_current.position.shape) == 2:
            x_current = x_current.unsqueeze(1)
            unsqueeze = True
        # calculate pose loss:
        if (
            self._batch_pose_idx is None
            or self._batch_pose_idx.shape[0] != x_current.position.shape[0]
        ):
            self._batch_pose_idx = torch.arange(
                0, x_current.position.shape[0], 1, device=self.tensor_args.device, dtype=torch.int32
            )
        distance = self.pose_cost.forward_pose(x_des, x_current, self._batch_pose_idx)
        if unsqueeze and resize:
            distance = distance.squeeze(1)
        return distance

    def get_point_robot_distance(self, points: torch.Tensor, q: torch.Tensor):
        """Compute distance from the robot at q joint configuration to points (e.g., pointcloud)

        Args:
            points: [b,n,3]
            q: [1, dof]

        Returns:
            distance: [b,1] Positive is in collision with robot
        NOTE: This currently does not support batched robot but can be done easily.
        """
        if len(q.shape) == 1:
            log_error("q should be of shape [b, dof]")
        kin_state = self.get_kinematics(q)
        pt_distance = point_robot_distance(kin_state.link_spheres_tensor, points)
        return pt_distance

    def get_active_js(self, full_js: JointState):
        active_jnames = self.kinematics.joint_names
        out_js = full_js.get_ordered_joint_state(active_jnames)
        return out_js


@get_torch_jit_decorator()
def sum_mask(d1, d2, d3):
    d_total = d1 + d2 + d3
    d_mask = d_total == 0.0
    return d_mask.view(-1)


@get_torch_jit_decorator()
def mask(d1, d2, d3):
    d_total = d1 + d2 + d3
    d_mask = d_total == 0.0
    return d_mask


@get_torch_jit_decorator()
def point_robot_distance(link_spheres_tensor, points):
    """Compute distance between robot and points

    Args:
        link_spheres_tensor: [batch_robot, n_robot_spheres, 4]
        points: [batch_points, n_points, 3]

    Returns:
        distance: [batch_points, n_points]
    """
    if link_spheres_tensor.shape[0] != 1:
        assert link_spheres_tensor.shape[0] == points.shape[0]
    squeeze_shape = False
    n = 1
    if len(points.shape) == 2:
        squeeze_shape = True
        n, _ = points.shape
        points = points.unsqueeze(0)

    robot_spheres = link_spheres_tensor.view(link_spheres_tensor.shape[0], -1, 4).contiguous()
    robot_spheres = robot_spheres.unsqueeze(-3)

    robot_radius = robot_spheres[..., 3]
    points = points.unsqueeze(-2)
    sph_distance = -1 * (
        torch.linalg.norm(points - robot_spheres[..., :3], dim=-1) - robot_radius
    )  # b, n_spheres
    pt_distance = torch.max(sph_distance, dim=-1)[0]

    if squeeze_shape:
        pt_distance = pt_distance.view(n)
    return pt_distance
