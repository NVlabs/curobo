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
from typing import Any, Dict, List, Optional

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
    CudaRobotModelState,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)
from curobo.curobolib.kinematics import get_cuda_kinematics
from curobo.geom.types import Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_error
from curobo.util_file import get_robot_path, join_path, load_yaml


@dataclass
class CudaRobotModelConfig:
    tensor_args: TensorDeviceType
    link_names: List[str]
    kinematics_config: KinematicsTensorConfig
    self_collision_config: Optional[SelfCollisionKinematicsConfig] = None
    kinematics_parser: Optional[KinematicsParser] = None
    compute_jacobian: bool = False
    use_global_cumul: bool = False
    generator_config: Optional[CudaRobotGeneratorConfig] = None

    def get_joint_limits(self):
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
    def from_robot_yaml_file(
        file_path: str,
        ee_link: Optional[str] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        config_file = load_yaml(join_path(get_robot_path(), file_path))["robot_cfg"]["kinematics"]
        if ee_link is not None:
            config_file["ee_link"] = ee_link
        return CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(**config_file, tensor_args=tensor_args)
        )

    @staticmethod
    def from_data_dict(
        data_dict: Dict[str, Any],
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(**data_dict, tensor_args=tensor_args)
        )

    @staticmethod
    def from_config(config: CudaRobotGeneratorConfig):
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
    def cspace(self):
        return self.kinematics_config.cspace


class CudaRobotModel(CudaRobotModelConfig):
    """
    CUDA Accelerated Robot Model

    NOTE: Currently dof is created only for links that we need to compute kinematics.
    E.g., for robots with many serial chains, add all links of the robot to get the correct dof.
    This is not an issue if you are loading collision spheres as that will cover the full geometry
    of the robot.
    """

    def __init__(self, config: CudaRobotModelConfig):
        super().__init__(**vars(config))
        self._batch_size = 0
        self.update_batch_size(1, reset_buffers=True)

    def update_batch_size(self, batch_size, force_update=False, reset_buffers=False):
        if batch_size == 0:
            log_error("batch size is zero")
        if force_update and self._batch_size == batch_size and self.compute_jacobian:
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
                dtype=self.tensor_args.dtype,
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
    def forward(self, q, link_name=None, calculate_jacobian=False):
        # pos, rot = self.compute_forward_kinematics(q, qd, link_name)
        if len(q.shape) > 2:
            raise ValueError("q shape should be [batch_size, dof]")
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
            raise NotImplementedError
        return (
            ee_pos,
            ee_quat,
            lin_jac,
            ang_jac,
            link_pos_seq,
            link_quat_seq,
            link_spheres_tensor,
        )

    def get_state(self, q, link_name=None, calculate_jacobian=False) -> CudaRobotModelState:
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

    def get_robot_link_meshes(self):
        m_list = [self.get_link_mesh(l) for l in self.kinematics_config.mesh_link_names]

        return m_list

    def get_robot_as_mesh(self, q: torch.Tensor):
        # get all link meshes:
        m_list = self.get_robot_link_meshes()
        pose = self.get_link_poses(q, self.kinematics_config.mesh_link_names)
        for li, l in enumerate(self.kinematics_config.mesh_link_names):
            m_list[li].pose = (
                Pose.from_list(m_list[li].pose).multiply(pose.get_index(0, li)).tolist()
            )

        return m_list

    def get_robot_as_spheres(self, q: torch.Tensor, filter_valid: bool = True):
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

    def _cuda_forward(self, q):
        link_pos, link_quat, robot_spheres = get_cuda_kinematics(
            # self._link_mat_seq,  # data will be stored here
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
            self._grad_out_q,
            self.use_global_cumul,
        )
        # if(robot_spheres.shape[0]<10):
        #    print(robot_spheres)
        return link_pos, link_quat, robot_spheres

    @property
    def all_articulated_joint_names(self):
        return self.kinematics_config.non_fixed_joint_names

    def get_self_collision_config(self) -> SelfCollisionKinematicsConfig:
        return self.self_collision_config

    def get_link_mesh(self, link_name: str) -> Mesh:
        mesh = self.kinematics_parser.get_link_mesh(link_name)
        return mesh

    def get_link_transform(self, link_name: str) -> Pose:
        mat = self._kinematics_config.fixed_transforms[self._name_to_idx_map[link_name]]
        pose = Pose(position=mat[:3, 3], rotation=mat[:3, :3])
        return pose

    def get_all_link_transforms(self) -> Pose:
        pose = Pose(
            self.kinematics_config.fixed_transforms[:, :3, 3],
            rotation=self.kinematics_config.fixed_transforms[:, :3, :3],
        )
        return pose

    def get_dof(self) -> int:
        return self.kinematics_config.n_dof

    @property
    def joint_names(self) -> List[str]:
        return self.kinematics_config.joint_names

    @property
    def total_spheres(self) -> int:
        return self.kinematics_config.total_spheres

    @property
    def lock_jointstate(self):
        return self.kinematics_config.lock_jointstate

    @property
    def ee_link(self):
        return self.kinematics_config.ee_link

    @property
    def base_link(self):
        return self.kinematics_config.base_link

    def update_kinematics_config(self, new_kin_config: KinematicsTensorConfig):
        self.kinematics_config.copy(new_kin_config)
