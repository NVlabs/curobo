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

"""
This module builds a kinematic representation of a robot on the GPU and provides
differentiable mapping from it's joint configuration to Cartesian pose of it's links
(forward kinematics). This module also computes the position of the spheres of the robot as part
of the forward kinematics function.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.cuda_robot_model.cuda_robot_generator import (
    CudaRobotGenerator,
    CudaRobotGeneratorConfig,
)
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser
from curobo.cuda_robot_model.types import (
    CSpaceConfig,
    JointLimits,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)
from curobo.cuda_robot_model.util import load_robot_yaml
from curobo.curobolib.kinematics import get_cuda_kinematics
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Mesh, Obstacle, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util_file import is_file_xrdf


@dataclass
class CudaRobotModelConfig:
    """
    Configuration for robot kinematics on GPU.

    Helper functions are provided to load this configuration from an URDF file or from
    a cuRobo robot configuration file (:ref:`tut_robot_configuration`). To create from a XRDF, use
    :ref:`curobo.util.xrdf_utils.convert_xrdf_to_curobo`.
    """

    #: Device and floating point precision to use for kinematics.
    tensor_args: TensorDeviceType

    #: Names of links to compute poses with forward kinematics.
    link_names: List[str]

    #: Tensors representing kinematics of the robot. This can be created using
    #: :class:`~curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGenerator`.
    kinematics_config: KinematicsTensorConfig

    #: Collision pairs to ignore when computing self collision between spheres across all
    #: robot links. This also contains distance threshold between spheres pairs and which thread
    #: indices for calculating the distances. More details on computing these parameters is in
    #: :func:`~curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGenerator._build_collision_model`.
    self_collision_config: Optional[SelfCollisionKinematicsConfig] = None

    #: Parser to load kinematics from URDF or USD files. This is used to load kinematics
    #: representation of the robot. This is created using
    #: :class:`~curobo.cuda_robot_model.kinematics_parser.KinematicsParser`.
    #: USD is an experimental feature and might not work for all robots.
    kinematics_parser: Optional[KinematicsParser] = None

    #: Output jacobian during forward kinematics. This is not implemented. The forward kinematics
    #: function does use Jacobian during backward pass. What's not supported is
    compute_jacobian: bool = False

    #: Store transformation matrix of every link during forward kinematics call in global memory.
    #: This helps speed up backward pass as we don't need to recompute the transformation matrices.
    #: However, this increases memory usage and also slightly slows down forward kinematics.
    #: Enabling this is recommended for getting the best performance.
    use_global_cumul: bool = True

    #: Generator config used to create this robot kinematics model.
    generator_config: Optional[CudaRobotGeneratorConfig] = None

    def get_joint_limits(self) -> JointLimits:
        """Get limits of actuated joints of the robot.

        Returns:
            JointLimits: Joint limits of the robot's actuated joints.
        """
        return self.kinematics_config.joint_limits

    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        ee_link: str,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load a cuda robot model from only urdf. This does not support collision queries.

        Args:
            urdf_path : Path of urdf file.
            base_link : Name of base link.
            ee_link : Name of end-effector link.
            tensor_args : Device to load robot model. Defaults to TensorDeviceType().

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        config = CudaRobotGeneratorConfig(base_link, ee_link, tensor_args, urdf_path=urdf_path)
        return CudaRobotModelConfig.from_config(config)

    @staticmethod
    def from_basic_usd(
        usd_path: str,
        usd_robot_root: str,
        base_link: str,
        ee_link: str,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load a cuda robot model from only urdf. This does not support collision queries.

        Args:
            urdf_path : Path of urdf file.
            base_link : Name of base link.
            ee_link : Name of end-effector link.
            tensor_args : Device to load robot model. Defaults to TensorDeviceType().

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        config = CudaRobotGeneratorConfig(
            tensor_args,
            base_link,
            ee_link,
            usd_path=usd_path,
            usd_robot_root=usd_robot_root,
            use_usd_kinematics=True,
        )
        return CudaRobotModelConfig.from_config(config)

    @staticmethod
    def from_content_path(
        content_path: ContentPath,
        ee_link: Optional[str] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load robot from Contentpath containing paths to robot description files.

        Args:
            content_path: Path to robot configuration files.
            ee_link: End-effector link name. If None, it is read from the file.
            tensor_args: Device to load robot model, defaults to cuda:0.

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """

        config_file = load_robot_yaml(content_path)
        if "robot_cfg" in config_file:
            config_file = config_file["robot_cfg"]
        if "kinematics" in config_file:
            config_file = config_file["kinematics"]
        if ee_link is not None:
            config_file["ee_link"] = ee_link

        return CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(**config_file, tensor_args=tensor_args)
        )

    @staticmethod
    def from_robot_yaml_file(
        file_path: Union[str, Dict],
        ee_link: Optional[str] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        urdf_path: Optional[str] = None,
    ) -> CudaRobotModelConfig:
        """Load robot from a yaml file that is in cuRobo's format (:ref:`tut_robot_configuration`).

        Args:
            file_path: Path to robot configuration file (yml or xrdf).
            ee_link: End-effector link name. If None, it is read from the file.
            tensor_args: Device to load robot model, defaults to cuda:0.
            urdf_path: Path to urdf file. This is required when loading a xrdf file.

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        if isinstance(file_path, dict):
            content_path = ContentPath(robot_urdf_file=urdf_path, robot_config_file=file_path)
        else:
            if is_file_xrdf(file_path):
                content_path = ContentPath(robot_urdf_file=urdf_path, robot_xrdf_file=file_path)
            else:
                content_path = ContentPath(robot_urdf_file=urdf_path, robot_config_file=file_path)

        return CudaRobotModelConfig.from_content_path(content_path, ee_link, tensor_args)

    @staticmethod
    def from_data_dict(
        data_dict: Dict[str, Any],
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load robot from a dictionary containing data for :class:`~curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGeneratorConfig`.

        :tut_robot_configuration discusses the data required to load a robot.

        Args:
            data_dict: Input dictionary containing robot configuration.
            tensor_args: Device to load robot model, defaults to cuda:0.

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        if "robot_cfg" in data_dict:
            data_dict = data_dict["robot_cfg"]
        if "kinematics" in data_dict:
            data_dict = data_dict["kinematics"]
        return CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(**data_dict, tensor_args=tensor_args)
        )

    @staticmethod
    def from_config(config: CudaRobotGeneratorConfig) -> CudaRobotModelConfig:
        """Create a robot model configuration from a generator configuration.

        Args:
            config: Input robot generator configuration.

        Returns:
            CudaRobotModelConfig: robot model configuration.
        """
        # create a config generator and load all values
        generator = CudaRobotGenerator(config)
        return CudaRobotModelConfig(
            tensor_args=generator.tensor_args,
            link_names=generator.link_names,
            kinematics_config=generator.kinematics_config,
            self_collision_config=generator.self_collision_config,
            kinematics_parser=generator.kinematics_parser,
            use_global_cumul=generator.use_global_cumul,
            compute_jacobian=generator.compute_jacobian,
            generator_config=config,
        )

    @property
    def cspace(self) -> CSpaceConfig:
        """Get cspace parameters of the robot."""
        return self.kinematics_config.cspace

    @property
    def dof(self) -> int:
        """Get the number of actuated joints (degrees of freedom) of the robot"""
        return self.kinematics_config.n_dof


@dataclass
class CudaRobotModelState:
    """Kinematic state of robot."""

    #: End-effector position stored as x,y,z in meters [b, 3]. End-effector is defined by
    #: :attr:`CudaRobotModel.ee_link`.
    ee_position: torch.Tensor

    #: End-effector orientaiton stored as quaternion qw, qx, qy, qz [b,4]. End-effector is defined
    #: by :attr:`CudaRobotModel.ee_link`.
    ee_quaternion: torch.Tensor

    #: Linear Jacobian. Currently not supported.
    lin_jacobian: Optional[torch.Tensor] = None

    #: Angular Jacobian. Currently not supported.
    ang_jacobian: Optional[torch.Tensor] = None

    #: Position of links specified by link_names  (:attr:`CudaRobotModel.link_names`).
    links_position: Optional[torch.Tensor] = None

    #: Quaternions of links specified by link names (:attr:`CudaRobotModel.link_names`).
    links_quaternion: Optional[torch.Tensor] = None

    #: Position of spheres specified by collision spheres (:attr:`CudaRobotModel.robot_spheres`)
    #: in x, y, z, r format [b,n,4].
    link_spheres_tensor: Optional[torch.Tensor] = None

    #: Names of links that each index in :attr:`links_position` and :attr:`links_quaternion`
    #: corresponds to.
    link_names: Optional[str] = None

    @property
    def ee_pose(self) -> Pose:
        """Get end-effector pose as a Pose object."""
        return Pose(self.ee_position, self.ee_quaternion)

    def get_link_spheres(self) -> torch.Tensor:
        """Get spheres representing robot geometry as a tensor with [batch,4],  [x,y,z,radius]."""
        return self.link_spheres_tensor

    @property
    def link_pose(self) -> Union[None, Dict[str, Pose]]:
        """Deprecated, use link_poses."""
        return self.link_poses

    @property
    def link_poses(self) -> Union[None, Dict[str, Pose]]:
        """Get link poses as a dictionary of link name to Pose object."""
        link_poses = None
        if self.link_names is not None:
            link_poses = {}
            link_pos = self.links_position.contiguous()
            link_quat = self.links_quaternion.contiguous()
            for i, v in enumerate(self.link_names):
                link_poses[v] = Pose(link_pos[..., i, :], link_quat[..., i, :])
        return link_poses


class CudaRobotModel(CudaRobotModelConfig):
    """
    CUDA Accelerated Robot Model

    Load basic kinematics from an URDF with :func:`~CudaRobotModelConfig.from_basic_urdf`.
    Check :ref:`tut_robot_configuration` for details on how to also create a geometric
    representation of the robot.
    Currently dof is created only for links that we need to compute kinematics. E.g., for robots
    with many serial chains, add all links of the robot to get the correct dof. This is not an
    issue if you are loading collision spheres as that will cover the full geometry of the robot.
    """

    def __init__(self, config: CudaRobotModelConfig):
        """Initialize kinematics instance with a robot model configuration.

        Args:
            config: Input robot model configuration.
        """
        super().__init__(**vars(config))
        self._batch_size = 0
        self.update_batch_size(1, reset_buffers=True)

    def update_batch_size(
        self, batch_size: int, force_update: bool = False, reset_buffers: bool = False
    ):
        """Update batch size of the robot model.

        Args:
            batch_size: Batch size to update the robot model.
            force_update: Detach gradients of tensors. This is not supported.
            reset_buffers: Recreate the tensors even if the batch size is same.
        """
        if batch_size == 0:
            log_error("batch size is zero")
        if force_update and self._batch_size == batch_size and self.compute_jacobian:
            log_error("Outputting jacobian is not supported")
            self.lin_jac = self.lin_jac.detach()  # .requires_grad_(True)
            self.ang_jac = self.ang_jac.detach()  # .requires_grad_(True)
        elif self._batch_size != batch_size or reset_buffers:
            self._batch_size = batch_size
            self._link_pos_seq = torch.zeros(
                (self._batch_size, len(self.link_names), 3),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._link_quat_seq = torch.zeros(
                (self._batch_size, len(self.link_names), 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )

            self._batch_robot_spheres = torch.zeros(
                (self._batch_size, self.kinematics_config.total_spheres, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.collision_geometry_dtype,
            )
            self._grad_out_q = torch.zeros(
                (self._batch_size, self.get_dof()),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._global_cumul_mat = torch.zeros(
                (self._batch_size, self.kinematics_config.link_map.shape[0], 4, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            if self.compute_jacobian:
                log_error("Outputting jacobian is not supported")
                self.lin_jac = torch.zeros(
                    [batch_size, 3, self.kinematics_config.n_dofs],
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                self.ang_jac = torch.zeros(
                    [batch_size, 3, self.kinematics_config.n_dofs],
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )

    @profiler.record_function("cuda_robot_model/forward_kinematics")
    def forward(
        self, q, link_name=None, calculate_jacobian=False
    ) -> Tuple[Tensor, Tensor, None, None, Tensor, Tensor, Tensor]:
        """Compute forward kinematics of the robot.

        Use :func:`~get_state` to get a structured output.

        Args:
            q: Joint configuration of the robot. Shape should be [batch_size, dof].
            link_name: Name of link to return pose of. If None, returns end-effector pose.
            calculate_jacobian: Calculate jacobian of the robot. Not supported.

        Returns:
            Tuple[Tensor, Tensor, None, None, Tensor, Tensor, Tensor]: End-effector position,
            end-effector quaternion (wxyz), linear jacobian(None), angular jacobian(None),
            link positions, link quaternion (wxyz), link spheres.
        """
        if len(q.shape) > 2:
            log_error("q shape should be [batch_size, dof]")
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        batch_size = q.shape[0]
        self.update_batch_size(batch_size, force_update=q.requires_grad)

        # do fused forward:
        link_pos_seq, link_quat_seq, link_spheres_tensor = self._cuda_forward(q)

        if len(self.link_names) == 1:
            ee_pos = link_pos_seq.squeeze(1)
            ee_quat = link_quat_seq.squeeze(1)
        else:
            link_idx = self.kinematics_config.ee_idx
            if link_name is not None:
                link_idx = self.link_names.index(link_name)
            ee_pos = link_pos_seq.contiguous()[..., link_idx, :]
            ee_quat = link_quat_seq.contiguous()[..., link_idx, :]
        lin_jac = ang_jac = None

        # compute jacobians?
        if calculate_jacobian:
            log_error("Outputting jacobian is not supported")
        return (
            ee_pos,
            ee_quat,
            lin_jac,
            ang_jac,
            link_pos_seq,
            link_quat_seq,
            link_spheres_tensor,
        )

    def get_state(
        self, q: torch.Tensor, link_name: str = None, calculate_jacobian: bool = False
    ) -> CudaRobotModelState:
        """Get kinematic state of the robot by computing forward kinematics.

        Args:
            q: Joint configuration of the robot. Shape should be [batch_size, dof].
            link_name: Name of link to return pose of. If None, returns end-effector pose.
            calculate_jacobian: Calculate jacobian of the robot. Not supported.

        Returns:
            CudaRobotModelState: Kinematic state of the robot.
        """
        out = self.forward(q, link_name, calculate_jacobian)
        state = CudaRobotModelState(
            out[0],
            out[1],
            None,
            None,
            out[4],
            out[5],
            out[6],
            self.link_names,
        )
        return state

    def compute_kinematics(
        self, js: JointState, link_name: Optional[str] = None, calculate_jacobian: bool = False
    ) -> CudaRobotModelState:
        """Compute forward kinematics of the robot.

        Args:
            js: Joint state of robot.
            link_name: Name of link to return pose of. If None, returns end-effector pose.
            calculate_jacobian: Calculate jacobian of the robot. Not supported.


        Returns:
            CudaRobotModelState: Kinematic state of the robot.

        """
        if js.joint_names is not None:
            if js.joint_names != self.kinematics_config.joint_names:
                log_error("Joint names do not match, reoder joints before forward kinematics")

        return self.get_state(js.position, link_name, calculate_jacobian)

    def compute_kinematics_from_joint_state(
        self, js: JointState, link_name: Optional[str] = None, calculate_jacobian: bool = False
    ) -> CudaRobotModelState:
        """Compute forward kinematics of the robot.

        Args:
            js: Joint state of robot.
            link_name: Name of link to return pose of. If None, returns end-effector pose.
            calculate_jacobian: Calculate jacobian of the robot. Not supported.


        Returns:
            CudaRobotModelState: Kinematic state of the robot.

        """
        if js.joint_names is not None:
            if js.joint_names != self.kinematics_config.joint_names:
                log_error("Joint names do not match, reoder joints before forward kinematics")

        return self.get_state(js.position, link_name, calculate_jacobian)

    def compute_kinematics_from_joint_position(
        self,
        joint_position: torch.Tensor,
        link_name: Optional[str] = None,
        calculate_jacobian: bool = False,
    ) -> CudaRobotModelState:
        """Compute forward kinematics of the robot.

        Args:
            joint_position: Joint position of robot. Assumed to only contain active joints in the
                order specified in :attr:`CudaRobotModel.joint_names`.
            link_name: Name of link to return pose of. If None, returns end-effector pose.
            calculate_jacobian: Calculate jacobian of the robot. Not supported.


        Returns:
            CudaRobotModelState: Kinematic state of the robot.

        """

        return self.get_state(joint_position, link_name, calculate_jacobian)

    def get_robot_link_meshes(self) -> List[Mesh]:
        """Get meshes of all links of the robot.

        Returns:
            List[Mesh]: List of all link meshes.
        """
        m_list = [self.get_link_mesh(l) for l in self.kinematics_config.mesh_link_names]

        return m_list

    def get_robot_as_mesh(self, q: torch.Tensor) -> List[Mesh]:
        """Transform robot links to Cartesian poses using forward kinematics and return as meshes.

        Args:
            q: Joint configuration of the robot, shape should be [1, dof].

        Returns:
            List[Mesh]: List of all link meshes.
        """
        # get all link meshes:
        m_list = self.get_robot_link_meshes()
        pose = self.get_link_poses(q, self.kinematics_config.mesh_link_names)
        for li, l in enumerate(self.kinematics_config.mesh_link_names):
            m_list[li].pose = (
                pose.get_index(0, li).multiply(Pose.from_list(m_list[li].pose)).tolist()
            )

        return m_list

    def get_robot_as_spheres(self, q: torch.Tensor, filter_valid: bool = True) -> List[Sphere]:
        """Get robot spheres using forward kinematics on given joint configuration q.

        Args:
            q: Joint configuration of the robot, shape should be [1, dof].
            filter_valid: Filter out spheres with radius <= 0.

        Returns:
            List[Sphere]: List of all robot spheres.
        """
        state = self.get_state(q)

        # state has sphere position and radius

        sph_all = state.get_link_spheres().cpu().numpy()

        sph_traj = []
        for j in range(sph_all.shape[0]):
            sph = sph_all[j, :, :]
            if filter_valid:
                sph_list = [
                    Sphere(
                        name="robot_curobo_sphere_" + str(i),
                        pose=[sph[i, 0], sph[i, 1], sph[i, 2], 1, 0, 0, 0],
                        radius=sph[i, 3],
                    )
                    for i in range(sph.shape[0])
                    if (sph[i, 3] > 0.0)
                ]
            else:
                sph_list = [
                    Sphere(
                        name="robot_curobo_sphere_" + str(i),
                        pose=[sph[i, 0], sph[i, 1], sph[i, 2], 1, 0, 0, 0],
                        radius=sph[i, 3],
                    )
                    for i in range(sph.shape[0])
                ]
            sph_traj.append(sph_list)
        return sph_traj

    def get_link_poses(self, q: torch.Tensor, link_names: List[str]) -> Pose:
        """Get Pose of links at given joint configuration q using forward kinematics.

        Note that only the links specified in :class:`~CudaRobotModelConfig.link_names` are returned.

        Args:
            q: Joint configuration of the robot, shape should be [batch_size, dof].
            link_names: Names of links to get pose of. This should be a subset of
                :class:`~CudaRobotModelConfig.link_names`.

        Returns:
            Pose: Poses of links at given joint configuration.
        """
        state = self.get_state(q)
        position = torch.zeros(
            (q.shape[0], len(link_names), 3),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        quaternion = torch.zeros(
            (q.shape[0], len(link_names), 4),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        for li, l in enumerate(link_names):
            i = self.link_names.index(l)
            position[:, li, :] = state.links_position[:, i, :]
            quaternion[:, li, :] = state.links_quaternion[:, i, :]
        return Pose(position=position, quaternion=quaternion)

    def _cuda_forward(self, q: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute forward kinematics on GPU. Use :func:`~get_state` or :func:`~forward` instead.

        Args:
            q: Joint configuration of the robot, shape should be [batch_size, dof].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Link positions, link quaternions, link
        """
        link_pos, link_quat, robot_spheres = get_cuda_kinematics(
            self._link_pos_seq,
            self._link_quat_seq,
            self._batch_robot_spheres,
            self._global_cumul_mat,
            q,
            self.kinematics_config.fixed_transforms,
            self.kinematics_config.link_spheres,
            self.kinematics_config.link_map,  # tells which link is attached to which link i
            self.kinematics_config.joint_map,  # tells which joint is attached to a link i
            self.kinematics_config.joint_map_type,  # joint type
            self.kinematics_config.store_link_map,
            self.kinematics_config.link_sphere_idx_map,  # sphere idx map
            self.kinematics_config.link_chain_map,
            self.kinematics_config.joint_offset_map,
            self._grad_out_q,
            self.use_global_cumul,
        )
        return link_pos, link_quat, robot_spheres

    @property
    def all_articulated_joint_names(self) -> List[str]:
        """Names of all articulated joints of the robot."""
        return self.kinematics_config.non_fixed_joint_names

    def get_self_collision_config(self) -> SelfCollisionKinematicsConfig:
        """Get self collision configuration parameters of the robot."""
        return self.self_collision_config

    def get_link_mesh(self, link_name: str) -> Mesh:
        """Get mesh of a link of the robot."""
        mesh = self.kinematics_parser.get_link_mesh(link_name)
        return mesh

    def get_link_transform(self, link_name: str) -> Pose:
        """Get pose offset of a link from it's parent joint.

        Args:
            link_name: Name of link to get pose of.

        Returns:
            Pose: Pose of the link.
        """
        mat = self.kinematics_config.fixed_transforms[
            self.kinematics_config.link_name_to_idx_map[link_name]
        ]
        pose = Pose(position=mat[:3, 3], rotation=mat[:3, :3])
        return pose

    def get_all_link_transforms(self) -> Pose:
        """Get offset pose of all links with respect to their parent joint."""
        pose = Pose(
            self.kinematics_config.fixed_transforms[:, :3, 3],
            rotation=self.kinematics_config.fixed_transforms[:, :3, :3],
        )
        return pose

    def get_dof(self) -> int:
        """Get degrees of freedom of the robot."""
        return self.kinematics_config.n_dof

    @property
    def dof(self) -> int:
        """Degrees of freedom of the robot."""
        return self.kinematics_config.n_dof

    @property
    def joint_names(self) -> List[str]:
        """Names of actuated joints."""
        return self.kinematics_config.joint_names

    @property
    def total_spheres(self) -> int:
        """Number of spheres used to approximate robot geometry."""
        return self.kinematics_config.total_spheres

    @property
    def lock_jointstate(self) -> JointState:
        """State of joints that are locked in the kinematic representation."""
        return self.kinematics_config.lock_jointstate

    def get_full_js(self, js: JointState) -> JointState:
        """Get state of all joints, including locked joints.

        This function will not provide state of mimic joints. If you need mimic joints, use
        :func:`~get_mimic_js`.

        Args:
            js: State containing articulated joints.

        Returns:
            JointState: State of all joints.
        """
        all_joint_names = self.all_articulated_joint_names
        lock_joint_state = self.lock_jointstate

        new_js = js.get_augmented_joint_state(all_joint_names, lock_joint_state)
        return new_js

    def get_mimic_js(self, js: JointState) -> JointState:
        """Get state of mimic joints from active joints.

        Current implementation uses a for loop over joints to calculate the state. This can be
        optimized by using a custom CUDA kernel or a matrix multiplication.

        Args:
            js: State containing articulated joints.

        Returns:
            JointState: State of active, locked, and mimic joints.
        """
        if self.kinematics_config.mimic_joints is None:
            return None
        extra_joints = {"position": [], "joint_names": []}
        # for every joint in mimic_joints, get active joint name
        for j in self.kinematics_config.mimic_joints:
            active_q = js.position[..., js.joint_names.index(j)]
            for k in self.kinematics_config.mimic_joints[j]:
                extra_joints["joint_names"].append(k["joint_name"])
                extra_joints["position"].append(
                    k["joint_offset"][0] * active_q + k["joint_offset"][1]
                )
        extra_js = JointState.from_position(
            position=torch.stack(extra_joints["position"]), joint_names=extra_joints["joint_names"]
        )
        new_js = js.get_augmented_joint_state(js.joint_names + extra_js.joint_names, extra_js)
        return new_js

    def update_kinematics_config(self, new_kin_config: KinematicsTensorConfig):
        """Update kinematics representation of the robot.

        A kinematics representation can be updated with new parameters. Some parameters that could
        require updating are state of locked joints, when a robot grasps an object. Another instance
        is when using different planners for different parts of the robot, example updating the
        state of robot base or another arm. Updations should result in the same tensor dimensions,
        if not then the instance of this class requires reinitialization.

        Args:
            new_kin_config: New kinematics representation of the robot.
        """

        self.kinematics_config.copy_(new_kin_config)

    def attach_external_objects_to_robot(
        self,
        joint_state: JointState,
        external_objects: List[Obstacle],
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> bool:
        """Attach external objects to a robot's link. See :ref:`attach_object_note` for details.

        Args:
            joint_state: Joint state of the robot.
            external_objects: List of external objects to attach to the robot.
            surface_sphere_radius: Radius (in meters) to use for points sampled on surface of the
                object. A smaller radius will allow for generating motions very close to obstacles.
            link_name: Name of the link (frame) to attach the objects to. The assumption is that
                this link does not have any geometry and all spheres of this link represent
                attached objects.
            sphere_fit_type: Sphere fit algorithm to use. See :ref:`attach_object_note` for more
                details. The default method :attr:`SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE`
                voxelizes the volume of the objects and adds spheres representing the voxels, then
                samples points on the surface of the object, adds :attr:`surface_sphere_radius` to
                these points. This should be used for most cases.
            voxelize_method: Method to use for voxelization, passed to
                :py:func:`trimesh.voxel.creation.voxelize`.
            world_objects_pose_offset: Offset to apply to the object poses before attaching to the
                robot. This is useful when attaching an object that's in contact with the world.
                The offset is applied in the world frame before attaching to the robot.
        """
        log_info("Attach objects to robot")
        if len(external_objects) == 0:
            log_error("no object in external_objects")
        kin_state = self.compute_kinematics(joint_state)
        ee_pose = kin_state.ee_pose  # w_T_ee
        if world_objects_pose_offset is not None:
            # add offset from ee:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
            # new ee_pose:
            # w_T_ee = offset_T_w * w_T_ee
            # ee_T_w
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.kinematics_config.get_number_of_spheres(link_name)
        object_names = [x.name for x in external_objects]
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            log_warn(
                "No spheres found, max_spheres: "
                + str(max_spheres)
                + " n_objects: "
                + str(len(object_names))
            )
            return False
        for i, x in enumerate(object_names):
            obs = external_objects[i]
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]

        log_info("MG: Computed spheres for attach objects to robot")

        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.kinematics_config.attach_object(sphere_tensor=sphere_tensor, link_name=link_name)

        return True

    def get_active_js(self, full_js: JointState):
        """Get joint state of active joints of the robot.

        Args:
            full_js: Joint state of all joints.

        Returns:
            JointState: Joint state of active joints.
        """
        active_jnames = self.joint_names
        out_js = full_js.get_ordered_joint_state(active_jnames)
        return out_js

    @property
    def ee_link(self) -> str:
        """End-effector link of the robot. Changing requires reinitializing this class."""
        return self.kinematics_config.ee_link

    @property
    def base_link(self) -> str:
        """Base link of the robot. Changing requires reinitializing this class."""
        return self.kinematics_config.base_link

    @property
    def robot_spheres(self):
        """Spheres representing robot geometry."""
        return self.kinematics_config.link_spheres

    @property
    def retract_config(self) -> torch.Tensor:
        """Retract configuration of the robot. Use :func:`~joint_names` to get joint names."""
        return self.kinematics_config.cspace.retract_config
