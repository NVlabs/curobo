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
"""Common structures used for Kinematics are defined in this module."""

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
from curobo.util.logger import log_error
from curobo.util.tensor_util import clone_if_not_none, copy_if_not_none


class JointType(Enum):
    """Type of Joint. Arbitrary axis of change is not supported."""

    #: Fixed joint.
    FIXED = -1

    #: Prismatic joint along x-axis.
    X_PRISM = 0

    #: Prismatic joint along y-axis.
    Y_PRISM = 1

    #: Prismatic joint along z-axis.
    Z_PRISM = 2

    #: Revolute joint along x-axis.
    X_ROT = 3

    #: Revolute joint along y-axis.
    Y_ROT = 4

    #: Revolute joint along z-axis.
    Z_ROT = 5

    #: Prismatic joint along negative x-axis.
    X_PRISM_NEG = 6

    #: Prismatic joint along negative y-axis.
    Y_PRISM_NEG = 7

    #: Prismatic joint along negative z-axis.
    Z_PRISM_NEG = 8

    #: Revolute joint along negative x-axis.
    X_ROT_NEG = 9

    #: Revolute joint along negative y-axis.
    Y_ROT_NEG = 10

    #: Revolute joint along negative z-axis.
    Z_ROT_NEG = 11


@dataclass
class JointLimits:
    """Joint limits for a robot."""

    #: Names of the joints. All tensors are indexed by joint names.
    joint_names: List[str]

    #: Position limits for each joint. Shape [n_joints, 2] with columns having [min, max] values.
    position: torch.Tensor

    #: Velocity limits for each joint. Shape [n_joints, 2] with columns having [min, max] values.
    velocity: torch.Tensor

    #: Acceleration limits for each joint. Shape [n_joints, 2] with columns having [min, max]
    #: values.
    acceleration: torch.Tensor

    #: Jerk limits for each joint. Shape [n_joints, 2] with columns having [min, max] values.
    jerk: torch.Tensor

    #: Effort limits for each joint. This is not used.
    effort: Optional[torch.Tensor] = None

    #: Device and floating point precision for tensors.
    tensor_args: TensorDeviceType = TensorDeviceType()

    @staticmethod
    def from_data_dict(
        data: Dict, tensor_args: TensorDeviceType = TensorDeviceType()
    ) -> JointLimits:
        """Create JointLimits from a dictionary.

        Args:
            data: Dictionary containing joint limits. E.g., {"position": [0, 1], ...}.
            tensor_args: Device and floating point precision for tensors.

        Returns:
            JointLimits: Joint limits instance.
        """
        p = tensor_args.to_device(data["position"])
        v = tensor_args.to_device(data["velocity"])
        a = tensor_args.to_device(data["acceleration"])
        j = tensor_args.to_device(data["jerk"])
        e = None
        if "effort" in data and data["effort"] is not None:
            e = tensor_args.to_device(data["effort"])

        return JointLimits(data["joint_names"], p, v, a, j, e)

    def clone(self) -> JointLimits:
        """Clone joint limits."""
        return JointLimits(
            self.joint_names.copy(),
            self.position.clone(),
            self.velocity.clone(),
            self.acceleration.clone(),
            self.jerk.clone(),
            self.effort.clone() if self.effort is not None else None,
            self.tensor_args,
        )

    def copy_(self, new_jl: JointLimits) -> JointLimits:
        """Copy joint limits from another instance. This maintains reference and copies the data.

        Args:
            new_jl: JointLimits instance to copy from.

        Returns:
            JointLimits: Data copied joint limits.
        """
        self.joint_names = new_jl.joint_names.copy()
        self.position.copy_(new_jl.position)
        self.velocity.copy_(new_jl.velocity)
        self.acceleration.copy_(new_jl.acceleration)
        self.effort = copy_if_not_none(new_jl.effort, self.effort)
        return self


@dataclass
class CSpaceConfig:
    """Configuration space parameters of the robot."""

    #: Names of the joints.
    joint_names: List[str]

    #: Retract configuration for the robot. This is the configuration used to bias graph search
    #: and also regularize inverse kinematics. This configuration is also used to initialize
    #: the robot during warmup phase of an optimizer. Set this to a collision-free configuration
    #: for good performance. When this configuration is in collision, it's not used to bias
    #: graph search.
    retract_config: Optional[T_DOF] = None

    #: Weight for each joint in configuration space. Used to measure distance between nodes in
    #: graph search-based planning.
    cspace_distance_weight: Optional[T_DOF] = None

    #: Weight for each joint, used in regularization cost term for inverse kinematics.
    null_space_weight: Optional[T_DOF] = None

    #: Device and floating point precision for tensors.
    tensor_args: TensorDeviceType = TensorDeviceType()

    #: Maximum acceleration for each joint. Accepts a scalar or a list of values for each joint.
    max_acceleration: Union[float, List[float]] = 10.0

    #: Maximum jerk for each joint. Accepts a scalar or a list of values for each joint.
    max_jerk: Union[float, List[float]] = 500.0

    #: Velocity scale for each joint. Accepts a scalar or a list of values for each joint.
    #: This is used to scale the velocity limits for each joint.
    velocity_scale: Union[float, List[float]] = 1.0

    #: Acceleration scale for each joint. Accepts a scalar or a list of values for each joint.
    #: This is used to scale the acceleration limits for each joint.
    acceleration_scale: Union[float, List[float]] = 1.0

    #: Jerk scale for each joint. Accepts a scalar or a list of values for each joint.
    #: This is used to scale the jerk limits for each joint.
    jerk_scale: Union[float, List[float]] = 1.0

    #: Position limit clip value. This is used to clip the position limits for each joint.
    #: Accepts a scalar or a list of values for each joint. This is useful to truncate limits
    #: to account for any safety margins imposed by real robot controllers.
    position_limit_clip: Union[float, List[float]] = 0.0

    def __post_init__(self):
        """Post initialization checks and data transfer to device tensors."""
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
        if isinstance(self.position_limit_clip, List):
            self.position_limit_clip = self.tensor_args.to_device(self.position_limit_clip)
        # check shapes:
        if self.retract_config is not None:
            dof = self.retract_config.shape
            if self.cspace_distance_weight is not None and self.cspace_distance_weight.shape != dof:
                log_error("cspace_distance_weight shape does not match retract_config")
            if self.null_space_weight is not None and self.null_space_weight.shape != dof:
                log_error("null_space_weight shape does not match retract_config")

    def inplace_reindex(self, joint_names: List[str]):
        """Change order of joints in configuration space tensors to match given order of names.

        Args:
            joint_names: New order of joint names.

        """
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

    def copy_(self, new_config: CSpaceConfig) -> CSpaceConfig:
        """Copy parameters from another instance.

        This maintains reference and copies the data. Assumes that the new instance has the same
        number of joints as the current instance and also same shape of tensors.

        Args:
            new_config: New parameters to copy into current instance.

        Returns:
            CSpaceConfig: Same instance of cspace configuration which has updated parameters.
        """
        self.joint_names = new_config.joint_names.copy()
        self.retract_config = copy_if_not_none(new_config.retract_config, self.retract_config)
        self.null_space_weight = copy_if_not_none(
            new_config.null_space_weight, self.null_space_weight
        )
        self.cspace_distance_weight = copy_if_not_none(
            new_config.cspace_distance_weight, self.cspace_distance_weight
        )
        self.tensor_args = self.tensor_args
        self.max_jerk = copy_if_not_none(new_config.max_jerk, self.max_jerk)
        self.max_acceleration = copy_if_not_none(new_config.max_acceleration, self.max_acceleration)
        self.velocity_scale = copy_if_not_none(new_config.velocity_scale, self.velocity_scale)
        self.acceleration_scale = copy_if_not_none(
            new_config.acceleration_scale, self.acceleration_scale
        )
        self.jerk_scale = copy_if_not_none(new_config.jerk_scale, self.jerk_scale)
        return self

    def clone(self) -> CSpaceConfig:
        """Clone configuration space parameters."""

        return CSpaceConfig(
            joint_names=self.joint_names.copy(),
            retract_config=clone_if_not_none(self.retract_config),
            null_space_weight=clone_if_not_none(self.null_space_weight),
            cspace_distance_weight=clone_if_not_none(self.cspace_distance_weight),
            tensor_args=self.tensor_args,
            max_jerk=self.max_jerk.clone(),
            max_acceleration=self.max_acceleration.clone(),
            velocity_scale=self.velocity_scale.clone(),
            acceleration_scale=self.acceleration_scale.clone(),
            jerk_scale=self.jerk_scale.clone(),
            position_limit_clip=(
                self.position_limit_clip.clone()
                if isinstance(self.position_limit_clip, torch.Tensor)
                else self.position_limit_clip
            ),
        )

    def scale_joint_limits(self, joint_limits: JointLimits) -> JointLimits:
        """Scale joint limits by the given scale factors.

        Args:
            joint_limits: Joint limits to scale.

        Returns:
            JointLimits: Scaled joint limits.
        """
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
    ) -> CSpaceConfig:
        """Load CSpace configuration from joint limits.

        Args:
            joint_position_upper: Upper position limits for each joint.
            joint_position_lower: Lower position limits for each joint.
            joint_names: Names of the joints. This should match the order of joints in the upper
                and lower limits.
            tensor_args: Device and floating point precision for tensors.

        Returns:
            CSpaceConfig: CSpace configuration with retract configuration set to the middle of the
                joint limits and all weights set to 1.
        """
        retract_config = ((joint_position_upper + joint_position_lower) / 2).flatten()
        n_dof = retract_config.shape[-1]
        null_space_weight = torch.ones(n_dof, **(tensor_args.as_torch_dict()))
        cspace_distance_weight = torch.ones(n_dof, **(tensor_args.as_torch_dict()))
        return CSpaceConfig(
            joint_names,
            retract_config,
            cspace_distance_weight,
            null_space_weight,
            tensor_args=tensor_args,
        )


@dataclass
class KinematicsTensorConfig:
    """Stores robot's kinematics parameters as Tensors to use in Kinematics computations.

    Use :meth:`curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGenerator` to generate this
    configuration from a urdf or usd.

    """

    #: Static Homogenous Transform from parent link to child link for all links [n_links,4,4].
    fixed_transforms: torch.Tensor

    #: Index of fixed_transform given link index [n_links].
    link_map: torch.Tensor

    #: Joint index given link index [n_links].
    joint_map: torch.Tensor

    #: Type of joint given link index [n_links].
    joint_map_type: torch.Tensor

    #: Joint offset to store scalars for mimic joints and negative axis joints.
    joint_offset_map: torch.Tensor

    #: Index of link to write out pose [n_store_links].
    store_link_map: torch.Tensor

    #: Mapping between each link to every other link, this is used to check
    #: if a link is part of a serial chain formed by another link [n_links, n_links].
    link_chain_map: torch.Tensor

    #: Name of links to compute pose during kinematics computation [n_store_links].
    link_names: List[str]

    #: Joint limits
    joint_limits: JointLimits

    #: Name of joints which are not fixed.
    non_fixed_joint_names: List[str]

    #: Number of joints that are active. Each joint is only actuated along 1 dimension.
    n_dof: int

    #: Name of links which have a mesh. Currently only used for debugging and rendering.
    mesh_link_names: Optional[List[str]] = None

    #: Name of all actuated joints.
    joint_names: Optional[List[str]] = None

    #: Name of joints to lock to a fixed value along with the locked value
    lock_jointstate: Optional[JointState] = None

    #: Joints that mimic other joints. This will be populated by :class:~`CudaRobotGenerator`
    #  when parsing the kinematics of the robot.
    mimic_joints: Optional[dict] = None

    #: Sphere representation of the robot's geometry. This is used for collision detection.
    link_spheres: Optional[torch.Tensor] = None

    #: Mapping of link index to sphere index. This is used to get spheres for a link.
    link_sphere_idx_map: Optional[torch.Tensor] = None

    #: Mapping of link name to link index. This is used to get link index from link name.
    link_name_to_idx_map: Optional[Dict[str, int]] = None

    #: Total number of spheres that represent the robot's geometry.
    total_spheres: int = 0

    #: Additional debug parameters.
    debug: Optional[Any] = None

    #: Index of end-effector in stored link poses.
    ee_idx: int = 0

    #: Cspace parameters for the robot.
    cspace: Optional[CSpaceConfig] = None

    #: Name of base link. This is the root link from which all kinematic parameters were computed.
    base_link: str = "base_link"

    #: Name of end-effector link for which the Cartesian pose will be computed.
    ee_link: str = "ee_link"

    #: A copy of link spheres that is used as reference, in case the link_spheres get modified at
    #: runtime.
    reference_link_spheres: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Post initialization checks and data transfer to device tensors."""
        if self.cspace is None and self.joint_limits is not None:
            self.load_cspace_cfg_from_kinematics()
        if self.joint_limits is not None and self.cspace is not None:
            self.joint_limits = self.cspace.scale_joint_limits(self.joint_limits)
        if self.link_spheres is not None and self.reference_link_spheres is None:
            self.reference_link_spheres = self.link_spheres.clone()

    def copy_(self, new_config: KinematicsTensorConfig) -> KinematicsTensorConfig:
        """Copy parameters from another instance into current instance.

        This maintains reference and copies the data. Assumes that the new instance has the same
        number of joints as the current instance and also same shape of tensors.

        Args:
            new_config: New parameters to copy into current instance.

        Returns:
            KinematicsTensorConfig: Same instance of kinematics configuration which has updated
                parameters.
        """
        self.fixed_transforms.copy_(new_config.fixed_transforms)
        self.link_map.copy_(new_config.link_map)
        self.joint_map.copy_(new_config.joint_map)
        self.joint_map_type.copy_(new_config.joint_map_type)
        self.store_link_map.copy_(new_config.store_link_map)
        self.link_chain_map.copy_(new_config.link_chain_map)
        self.joint_limits.copy_(new_config.joint_limits)
        self.joint_offset_map.copy_(new_config.joint_offset_map)
        if new_config.link_spheres is not None and self.link_spheres is not None:
            self.link_spheres.copy_(new_config.link_spheres)
        if new_config.link_sphere_idx_map is not None and self.link_sphere_idx_map is not None:
            self.link_sphere_idx_map.copy_(new_config.link_sphere_idx_map)
        if new_config.link_name_to_idx_map is not None and self.link_name_to_idx_map is not None:
            self.link_name_to_idx_map = new_config.link_name_to_idx_map.copy()
        if (
            new_config.reference_link_spheres is not None
            and self.reference_link_spheres is not None
        ):
            self.reference_link_spheres.copy_(new_config.reference_link_spheres)
        self.base_link = new_config.base_link
        self.ee_idx = new_config.ee_idx
        self.ee_link = new_config.ee_link
        self.debug = new_config.debug
        self.cspace.copy_(new_config.cspace)
        self.n_dof = new_config.n_dof
        self.non_fixed_joint_names = new_config.non_fixed_joint_names
        self.joint_names = new_config.joint_names
        self.link_names = new_config.link_names
        self.mesh_link_names = new_config.mesh_link_names
        self.total_spheres = new_config.total_spheres
        if self.lock_jointstate is not None and new_config.lock_jointstate is not None:
            self.lock_jointstate.copy_(new_config.lock_jointstate)
        self.mimic_joints = new_config.mimic_joints

        return self

    def load_cspace_cfg_from_kinematics(self):
        """Load CSpace configuration from joint limits.

        This sets the retract configuration to the middle of the joint limits and all weights to 1.
        """
        retract_config = (
            (self.joint_limits.position[1] + self.joint_limits.position[0]) / 2
        ).flatten()
        null_space_weight = torch.ones(self.n_dof, **(self.tensor_args.as_torch_dict()))
        cspace_distance_weight = torch.ones(self.n_dof, **(self.tensor_args.as_torch_dict()))
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
        """Get indices of spheres for a link.

        Args:
            link_name: Name of the link.

        Returns:
            torch.Tensor: Indices of spheres for the link.
        """
        link_idx = self.link_name_to_idx_map[link_name]
        link_spheres_idx = torch.nonzero(self.link_sphere_idx_map == link_idx).view(-1)
        return link_spheres_idx

    def update_link_spheres(
        self, link_name: str, sphere_position_radius: torch.Tensor, start_sph_idx: int = 0
    ):
        """Update sphere parameters of a specific link given by name.

        Args:
            link_name: Name of the link.
            sphere_position_radius: Tensor of shape [n_spheres, 4] with columns [x, y, z, r].
            start_sph_idx: If providing a subset of spheres, this is the starting index.
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
    ) -> torch.Tensor:
        """Get spheres of a link.

        Args:
            link_name: Name of link.

        Returns:
            torch.Tensor: Spheres of the link with shape [n_spheres, 4] with columns [x, y, z, r].
        """
        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        return self.link_spheres[link_sphere_index, :]

    def get_reference_link_spheres(
        self,
        link_name: str,
    ) -> torch.Tensor:
        """Get link spheres from the original robot configuration data before any modifications.

        Args:
            link_name: Name of link.

        Returns:
            torch.Tensor: Spheres of the link with shape [n_spheres, 4] with columns [x, y, z, r].
        """

        link_sphere_index = self.get_sphere_index_from_link_name(link_name)
        return self.reference_link_spheres[link_sphere_index, :]

    def attach_object(
        self,
        sphere_radius: Optional[float] = None,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> bool:
        """Attach object approximated by spheres to a link of the robot.

        This function updates the sphere parameters of the link to represent the attached object.

        Args:
            sphere_radius: Radius to change for existing spheres. If changing position of spheres
                as well, then set this to None.
            sphere_tensor: New sphere tensor to replace existing spheres. Shape [n_spheres, 4] with
                columns [x, y, z, r]. If changing only radius, set this to None and use
                sphere_radius.
            link_name: Name of the link to attach object to. Defaults to "attached_object".

        Returns:
            bool: True if successful.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_error(link_name + " not found in spheres")
        curr_spheres = self.get_link_spheres(link_name)

        if sphere_radius is not None:
            curr_spheres[:, 3] = sphere_radius
        if sphere_tensor is not None:
            if sphere_tensor.shape != curr_spheres.shape and sphere_tensor.shape[0] != 1:
                log_error("sphere_tensor shape does not match current spheres")
            curr_spheres[:, :] = sphere_tensor
        self.update_link_spheres(link_name, curr_spheres)
        return True

    def detach_object(self, link_name: str = "attached_object") -> bool:
        """Detach object spheres from a link by setting all spheres to zero with negative radius.

        Args:
            link_name: Name of the link to detach object from.

        Returns:
            bool: True if successful.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_error(link_name + " not found in spheres")
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

    def disable_link_spheres(self, link_name: str):
        """Disable spheres of a link by setting all spheres to zero with negative radius.

        Args:
            link_name: Name of the link to disable spheres.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_error(link_name + " not found in spheres")
        curr_spheres = self.get_link_spheres(link_name)
        curr_spheres[:, 3] = -100.0
        self.update_link_spheres(link_name, curr_spheres)

    def enable_link_spheres(self, link_name: str):
        """Enable spheres of a link by resetting to values from initial robot configuration data.

        Args:
            link_name: Name of the link to enable spheres.
        """
        if link_name not in self.link_name_to_idx_map.keys():
            log_error(link_name + " not found in spheres")
        curr_spheres = self.get_reference_link_spheres(link_name)
        self.update_link_spheres(link_name, curr_spheres)


@dataclass
class SelfCollisionKinematicsConfig:
    """Dataclass that stores self collision attributes to pass to cuda kernel."""

    #: Offset radii for each sphere. This is used to inflate the spheres for self collision
    #: detection.
    offset: Optional[torch.Tensor] = None

    #: Sphere index to use for a given thread.
    thread_location: Optional[torch.Tensor] = None

    #: Maximum number of threads to launch for computing self collision between spheres.
    thread_max: Optional[int] = None

    #: Distance threshold for self collision detection. This is currently not used.
    distance_threshold: Optional[torch.Tensor] = None

    #: Two kernel implementations are available. Set this to True to use the experimental kernel
    #: which is faster. Set this to False to use the collision matrix based kernel which is slower.
    experimental_kernel: bool = True

    #: Collision matrix containing information about which pair of spheres need to be checked for
    #: collision. This is only used when experimental_kernel is set to False.
    collision_matrix: Optional[torch.Tensor] = None

    #: Number of collision checks to perform per thread. Each thread loads a sphere and is allowed
    #: to check upto checks_per_thread other spheres for collision. Note that all checks have to
    #: be performed within 1024 threads as shared memory is used. So,
    # checks_per_thread * n_spheres <= 1024.
    checks_per_thread: int = 32
