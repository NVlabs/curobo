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
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.tensor import T_DOF
from curobo.util.tensor_util import copy_if_not_none


class JointType(Enum):
    FIXED = -1
    X_PRISM = 0
    Y_PRISM = 1
    Z_PRISM = 2
    X_ROT = 3
    Y_ROT = 4
    Z_ROT = 5
    X_PRISM_NEG = 6
    Y_PRISM_NEG = 7
    Z_PRISM_NEG = 8
    X_ROT_NEG = 9
    Y_ROT_NEG = 10
    Z_ROT_NEG = 11


@dataclass
class JointLimits:
    joint_names: List[str]
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    effort: Optional[torch.Tensor] = None
    tensor_args: TensorDeviceType = TensorDeviceType()

    @staticmethod
    def from_data_dict(data: Dict, tensor_args: TensorDeviceType = TensorDeviceType()):
        p = tensor_args.to_device(data["position"])
        v = tensor_args.to_device(data["velocity"])
        a = tensor_args.to_device(data["acceleration"])
        j = tensor_args.to_device(data["jerk"])
        e = None
        if "effort" in data and data["effort"] is not None:
            e = tensor_args.to_device(data["effort"])

        return JointLimits(data["joint_names"], p, v, a, j, e)

    def clone(self) -> JointLimits:
        return JointLimits(
            self.joint_names.copy(),
            self.position.clone(),
            self.velocity.clone(),
            self.acceleration.clone(),
            self.jerk.clone(),
            self.effort.clone() if self.effort is not None else None,
            self.tensor_args,
        )


@dataclass
class CSpaceConfig:
    joint_names: List[str]
    retract_config: Optional[T_DOF] = None
    cspace_distance_weight: Optional[T_DOF] = None
    null_space_weight: Optional[T_DOF] = None  # List[float]
    tensor_args: TensorDeviceType = TensorDeviceType()
    max_acceleration: Union[float, List[float]] = 10.0
    max_jerk: Union[float, List[float]] = 500.0
    velocity_scale: Union[float, List[float]] = 1.0
    acceleration_scale: Union[float, List[float]] = 1.0
    jerk_scale: Union[float, List[float]] = 1.0
    position_limit_clip: Union[float, List[float]] = 0.01

    def __post_init__(self):
        if self.retract_config is not None:
            self.retract_config = self.tensor_args.to_device(self.retract_config)
        if self.cspace_distance_weight is not None:
            self.cspace_distance_weight = self.tensor_args.to_device(self.cspace_distance_weight)
        if self.null_space_weight is not None:
            self.null_space_weight = self.tensor_args.to_device(self.null_space_weight)
        if isinstance(self.max_acceleration, float):
            self.max_acceleration = self.tensor_args.to_device(
                [self.max_acceleration for _ in self.joint_names]
            )

        if isinstance(self.velocity_scale, float) or len(self.velocity_scale) == 1:
            self.velocity_scale = self.tensor_args.to_device(
                [self.velocity_scale for _ in self.joint_names]
            ).view(-1)

        if isinstance(self.acceleration_scale, float) or len(self.acceleration_scale) == 1:
            self.acceleration_scale = self.tensor_args.to_device(
                [self.acceleration_scale for _ in self.joint_names]
            ).view(-1)

        if isinstance(self.jerk_scale, float) or len(self.jerk_scale) == 1:
            self.jerk_scale = self.tensor_args.to_device(
                [self.jerk_scale for _ in self.joint_names]
            ).view(-1)

        if isinstance(self.max_acceleration, List):
            self.max_acceleration = self.tensor_args.to_device(self.max_acceleration)
        if isinstance(self.max_jerk, float):
            self.max_jerk = [self.max_jerk for _ in self.joint_names]
        if isinstance(self.max_jerk, List):
            self.max_jerk = self.tensor_args.to_device(self.max_jerk)
        if isinstance(self.velocity_scale, List):
            self.velocity_scale = self.tensor_args.to_device(self.velocity_scale)

        if isinstance(self.acceleration_scale, List):
            self.acceleration_scale = self.tensor_args.to_device(self.acceleration_scale)
        if isinstance(self.jerk_scale, List):
            self.jerk_scale = self.tensor_args.to_device(self.jerk_scale)

    def inplace_reindex(self, joint_names: List[str]):
        new_index = [self.joint_names.index(j) for j in joint_names]
        if self.retract_config is not None:
            self.retract_config = self.retract_config[new_index].clone()
        if self.cspace_distance_weight is not None:
            self.cspace_distance_weight = self.cspace_distance_weight[new_index].clone()
        if self.null_space_weight is not None:
            self.null_space_weight = self.null_space_weight[new_index].clone()
        self.max_acceleration = self.max_acceleration[new_index].clone()
        self.max_jerk = self.max_jerk[new_index].clone()
        self.velocity_scale = self.velocity_scale[new_index].clone()
        self.acceleration_scale = self.acceleration_scale[new_index].clone()
        self.jerk_scale = self.jerk_scale[new_index].clone()
        joint_names = [self.joint_names[n] for n in new_index]
        self.joint_names = joint_names

    def clone(self) -> CSpaceConfig:
        return CSpaceConfig(
            joint_names=self.joint_names.copy(),
            retract_config=copy_if_not_none(self.retract_config),
            null_space_weight=copy_if_not_none(self.null_space_weight),
            cspace_distance_weight=copy_if_not_none(self.cspace_distance_weight),
            tensor_args=self.tensor_args,
            max_jerk=self.max_jerk.clone(),
            max_acceleration=self.max_acceleration.clone(),
            velocity_scale=self.velocity_scale.clone(),
            acceleration_scale=self.acceleration_scale.clone(),
            jerk_scale=self.jerk_scale.clone(),
        )

    def scale_joint_limits(self, joint_limits: JointLimits):
        if self.velocity_scale is not None:
            joint_limits.velocity = joint_limits.velocity * self.velocity_scale
        if self.acceleration_scale is not None:
            joint_limits.acceleration = joint_limits.acceleration * self.acceleration_scale
        if self.jerk_scale is not None:
            joint_limits.jerk = joint_limits.jerk * self.jerk_scale

        return joint_limits

    @staticmethod
    def load_from_joint_limits(
        joint_position_upper: torch.Tensor,
        joint_position_lower: torch.Tensor,
        joint_names: List[str],
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        retract_config = ((joint_position_upper + joint_position_lower) / 2).flatten()
        n_dof = retract_config.shape[-1]
        null_space_weight = torch.ones(n_dof, **vars(tensor_args))
        cspace_distance_weight = torch.ones(n_dof, **vars(tensor_args))
        return CSpaceConfig(
            joint_names,
            retract_config,
            cspace_distance_weight,
            null_space_weight,
            tensor_args=tensor_args,
        )


@dataclass
class KinematicsTensorConfig:
    fixed_transforms: torch.Tensor
    link_map: torch.Tensor
    joint_map: torch.Tensor
    joint_map_type: torch.Tensor
    store_link_map: torch.Tensor
    link_chain_map: torch.Tensor
    link_names: List[str]
    joint_limits: JointLimits
    non_fixed_joint_names: List[str]
    n_dof: int
    mesh_link_names: Optional[List[str]] = None
    joint_names: Optional[List[str]] = None
    lock_jointstate: Optional[JointState] = None
    link_spheres: Optional[torch.Tensor] = None
    link_sphere_idx_map: Optional[torch.Tensor] = None
    link_name_to_idx_map: Optional[Dict[str, int]] = None
    total_spheres: int = 0
    debug: Optional[Any] = None
    ee_idx: int = 0
    cspace: Optional[CSpaceConfig] = None
    base_link: str = "base_link"
    ee_link: str = "ee_link"

    def __post_init__(self):
        if self.cspace is None and self.joint_limits is not None:
            self.load_cspace_cfg_from_kinematics()
        if self.joint_limits is not None and self.cspace is not None:
            self.joint_limits = self.cspace.scale_joint_limits(self.joint_limits)

    def load_cspace_cfg_from_kinematics(self):
        retract_config = (
            (self.joint_limits.position[1] + self.joint_limits.position[0]) / 2
        ).flatten()
        null_space_weight = torch.ones(self.n_dof, **vars(self.tensor_args))
        cspace_distance_weight = torch.ones(self.n_dof, **vars(self.tensor_args))
        joint_names = self.joint_names
        self.cspace = CSpaceConfig(
            joint_names,
            retract_config,
            cspace_distance_weight,
            null_space_weight,
            tensor_args=self.tensor_args,
            max_acceleration=self.joint_limits.acceleration[1],
            max_jerk=self.joint_limits.max_jerk[1],
        )

    def get_sphere_index_from_link_name(self, link_name: str) -> torch.Tensor:
        link_idx = self.link_name_to_idx_map[link_name]
        # get sphere index for this link:
        link_spheres_idx = torch.nonzero(self.link_sphere_idx_map == link_idx).view(-1)
        return link_spheres_idx

    def update_link_spheres(
        self, link_name: str, sphere_position_radius: torch.Tensor, start_sph_idx: int = 0
    ):
        """Update sphere parameters

        #NOTE: This currently does not update self collision distances.
        Args:
            link_name: _description_
            sphere_position_radius: _description_
            start_sph_idx: _description_. Defaults to 0.
        """
        # get sphere indices from link name:
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)[
            start_sph_idx : start_sph_idx + sphere_position_radius.shape[0]
        ]
        # update sphere data:
        self.link_spheres[link_sphere_index, :] = sphere_position_radius

    def get_link_spheres(
        self,
        link_name: str,
    ):
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        return self.link_spheres[link_sphere_index, :]

    def attach_object(
        self,
        sphere_radius: Optional[float] = None,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> bool:
        """_summary_

        Args:
            sphere_radius: _description_. Defaults to None.
            sphere_tensor: _description_. Defaults to None.
            link_name: _description_. Defaults to "attached_object".

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _description_
        """
        if link_name not in self.link_name_to_idx_map.keys():
            raise ValueError(link_name + " not found in spheres")
        curr_spheres = self.get_link_spheres(link_name)

        if sphere_radius is not None:
            curr_spheres[:, 3] = sphere_radius
        if sphere_tensor is not None:
            if sphere_tensor.shape != curr_spheres.shape and sphere_tensor.shape[0] != 1:
                raise ValueError("sphere_tensor shape does not match current spheres")
            curr_spheres[:, :] = sphere_tensor
        self.update_link_spheres(link_name, curr_spheres)
        return True

    def detach_object(self, link_name: str = "attached_object") -> bool:
        """Detach object from robot end-effector

        Args:
            link_name: _description_. Defaults to "attached_object".

        Raises:
            ValueError: _description_

        Returns:
            _description_
        """
        if link_name not in self.link_name_to_idx_map.keys():
            raise ValueError(link_name + " not found in spheres")
        curr_spheres = self.get_link_spheres(link_name)
        curr_spheres[:, 3] = -100.0
        curr_spheres[:, :3] = 0.0
        self.update_link_spheres(link_name, curr_spheres)

        return True

    def get_number_of_spheres(self, link_name: str) -> int:
        """Get number of spheres for a link

        Args:
            link_name: name of link
        """
        return self.get_link_spheres(link_name).shape[0]


@dataclass
class SelfCollisionKinematicsConfig:
    """Dataclass that stores self collision attributes to pass to cuda kernel."""

    offset: Optional[torch.Tensor] = None
    distance_threshold: Optional[torch.Tensor] = None
    thread_location: Optional[torch.Tensor] = None
    thread_max: Optional[int] = None
    collision_matrix: Optional[torch.Tensor] = None
    experimental_kernel: bool = True
    checks_per_thread: int = 32


@dataclass(frozen=True)
class CudaRobotModelState:
    """Dataclass that stores kinematics information."""

    #: End-effector position stored as x,y,z in meters [b, 3]. End-effector is defined by
    #: :py:attr:`curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGeneratorConfig.ee_link`.
    ee_position: torch.Tensor

    #: End-effector orientaiton stored as quaternion qw, qx, qy, qz [b,4]. End-effector is defined
    # by :py:attr:`CudaRobotModelConfig.ee_link`.
    ee_quaternion: torch.Tensor

    #: Linear Jacobian. Currently not supported.
    lin_jacobian: Optional[torch.Tensor] = None

    #: Angular Jacobian. Currently not supported.
    ang_jacobian: Optional[torch.Tensor] = None

    #: Position of links specified by link_names  (:py:attr:`CudaRobotModelConfig.link_names`).
    links_position: Optional[torch.Tensor] = None

    #: Quaternions of links specified by link names (:py:attr:`CudaRobotModelConfig.link_names`).
    links_quaternion: Optional[torch.Tensor] = None

    #: Position of spheres specified by collision spheres (:py:attr:`CudaRobotModelConfig.collision_spheres`)
    #: in x, y, z, r format [b,n,4].
    link_spheres_tensor: Optional[torch.Tensor] = None

    link_names: Optional[str] = None

    @property
    def ee_pose(self):
        return Pose(self.ee_position, self.ee_quaternion)

    def get_link_spheres(self):
        return self.link_spheres_tensor

    @property
    def link_pose(self):
        link_poses = None
        if self.link_names is not None:
            link_poses = {}
            link_pos = self.links_position.contiguous()
            link_quat = self.links_quaternion.contiguous()
            for i, v in enumerate(self.link_names):
                link_poses[v] = Pose(link_pos[..., i, :], link_quat[..., i, :])
        return link_poses
