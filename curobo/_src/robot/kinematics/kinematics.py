# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""This module builds a kinematic representation of a robot on the GPU and provides
differentiable mapping from it's joint configuration to Cartesian pose of it's links
(forward kinematics). This module also computes the position of the spheres of the robot as part
of the forward kinematics function.
"""

from __future__ import annotations

# Standard Library
from typing import List, Optional, Union

import torch
import torch.autograd.profiler as profiler

# Third Party
import trimesh

from curobo._src.curobolib.cuda_ops.kinematics import KinematicsFusedFunction
from curobo._src.geom.types import Mesh, Sphere
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.kinematics_params import KinematicsParams

# CuRobo
from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import augment_joint_state
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise


class Kinematics:
    """CUDA Accelerated Robot Model

    Load basic kinematics from an URDF with :func:`~KinematicsCfg.from_basic_urdf`.
    Check :ref:`tut_robot_configuration` for details on how to also create a geometric
    representation of the robot.
    Currently dof is created only for links that we need to compute kinematics. E.g., for robots
    with many serial chains, add all links of the robot to get the correct dof. This is not an
    issue if you are loading collision spheres as that will cover the full geometry of the robot.
    """

    def __init__(
        self,
        config: KinematicsCfg,
        compute_jacobian: bool = False,
        compute_spheres: bool = True,
        compute_com: bool = False,
    ):
        """Initialize kinematics instance with a robot model configuration.

        Args:
            config: Input robot model configuration.
        """
        self.config = config
        self.device_cfg = config.device_cfg
        self.compute_jacobian = compute_jacobian
        self.compute_spheres = compute_spheres
        self.compute_com = compute_com
        self._batch = 0
        self._horizon = 0
        self.update_batch_size(1, 1, reset_buffers=True)

    @property
    def tool_frames(self) -> List[str]:
        return self.config.tool_frames

    @profiler.record_function("cuda_robot_model/update_batch_size")
    def update_batch_size(
        self, batch: int, horizon: int, force_update: bool = False, reset_buffers: bool = False
    ):
        """Update batch size of the robot model.

        Args:
            batch: Batch dimension of [batch, horizon, dof].
            horizon: Trajectory horizon dimension.
            force_update: Detach gradients of tensors. This is not supported.
            reset_buffers: Recreate the tensors even if the batch size is same.
        """
        if batch == 0 or horizon == 0:
            log_and_raise("batch and horizon must be > 0")
        if self._batch != batch or self._horizon != horizon or reset_buffers:
            self._batch = batch
            self._horizon = horizon
            self._buffers = KinematicsFusedFunction.create_buffers(
                batch,
                horizon,
                self.config.kinematics_config,
                self.device_cfg,
            )
            self._buffers["idxs_env"] = torch.zeros(
                (batch,), dtype=torch.int32, device=self.device_cfg.device
            )

    @profiler.record_function("cuda_robot_model/forward_kinematics")
    def _forward(
        self,
        joint_position: torch.Tensor,
        idxs_env: Optional[torch.Tensor] = None,
    ) -> KinematicsState:
        """Low-level forward kinematics (class-internal only).

        External callers must use :meth:`compute_kinematics`.

        Args:
            joint_position: Joint configuration of the robot.
                Shape must be [batch, horizon, dof].
            idxs_env: Environment query index. Shape [num_envs], compact.
                Maps each environment to a sphere configuration. The kernel uses
                ``env_query_idx[batch_index / horizon]`` to look up the config.
                When None, uses env 0 for all elements.

        Returns:
            KinematicsState: Kinematic state of the robot.
        """
        if joint_position.ndim != 3:
            log_and_raise(
                f"joint_position must be [batch, horizon, dof], got shape {joint_position.shape}"
            )
        if joint_position.shape[-1] != self.dof:
            log_and_raise(f"q should have dof = {self.dof}, got {joint_position.shape[-1]}")

        batch, horizon, dof = joint_position.shape

        self.update_batch_size(batch, horizon, force_update=joint_position.requires_grad)

        if idxs_env is None:
            idxs_env = self._buffers["idxs_env"]

        link_position, link_quaternion, robot_spheres, robot_com, link_jacobian = (
            KinematicsFusedFunction.apply(
                joint_position,
                self._buffers["batch_link_position"],
                self._buffers["batch_link_quaternion"],
                self._buffers["batch_robot_spheres"],
                self._buffers["batch_com"],
                self._buffers["batch_jacobian"],
                self._buffers["batch_cumul_mat"],
                self.config.kinematics_config,
                self._buffers["grad_out_q"],
                self._buffers["grad_out_q_jacobian"],
                self._buffers["grad_in_link_pos"],
                self._buffers["grad_in_link_quat"],
                self._buffers["grad_in_robot_spheres"],
                self._buffers["grad_in_com"],
                self.compute_jacobian,
                self.compute_spheres,
                self.compute_com,
                idxs_env,
                horizon,
            )
        )
        link_poses = ToolPose(
            tool_frames=self.tool_frames,
            position=link_position,
            quaternion=link_quaternion,
        )
        return KinematicsState(
            robot_spheres=robot_spheres,
            tool_poses=link_poses,
            tool_jacobians=link_jacobian,
            robot_com=robot_com,
            robot_collision_geometry=self.config.kinematics_config.get_robot_collision_geometry(),
        )

    def compute_kinematics(
        self,
        joint_state: JointState,
        idxs_env: Optional[torch.Tensor] = None,
    ) -> KinematicsState:
        """Compute forward kinematics of the robot.

        This is the **single public API** for forward kinematics. All external
        callers must use this method.

        Args:
            joint_state: Joint state of robot. ``joint_names`` must be set for validation.
            idxs_env: Environment query index. Shape [num_envs], compact.

        Returns:
            KinematicsState: Kinematic state of the robot.
        """
        if joint_state.joint_names is not None:
            if joint_state.joint_names != self.config.kinematics_config.joint_names:
                log_and_raise("Joint names do not match, reorder joints before forward kinematics")

        q = joint_state.position
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif q.ndim == 2:
            q = q.unsqueeze(1)
        return self._forward(q, idxs_env=idxs_env)

    def get_robot_link_meshes(self) -> List[Mesh]:
        """Get meshes of all links of the robot.

        Returns:
            List[Mesh]: List of all link meshes.
        """
        m_list = [self.get_link_mesh(l) for l in self.config.kinematics_config.mesh_link_names]

        return m_list

    def get_robot_as_mesh(self, joint_position: torch.Tensor) -> List[Mesh]:
        """Transform robot links to Cartesian poses using forward kinematics and return as meshes.

        Args:
            joint_position: Joint configuration of the robot, shape should be [1, dof].

        Returns:
            List[Mesh]: List of all link meshes.
        """
        if joint_position.ndim == 2:
            joint_position = joint_position.unsqueeze(1)
        # get all link meshes:
        m_list = self.get_robot_link_meshes()
        pose = self.get_link_poses(joint_position, self.config.kinematics_config.mesh_link_names)
        for li, l in enumerate(self.config.kinematics_config.mesh_link_names):
            m_list[li].pose = (
                pose.get_index(0, li).multiply(Pose.from_list(m_list[li].pose)).tolist()
            )

        return m_list

    def get_robot_as_spheres(
        self, q: torch.Tensor, filter_valid: bool = True
    ) -> Union[List[Sphere], List[List[Sphere]]]:
        """Get robot spheres using forward kinematics on given joint configuration q.

        Args:
            q: Joint configuration of the robot, shape should be [1, dof].
            filter_valid: Filter out spheres with radius <= 0.

        Returns:
            List[Sphere]: List of all robot spheres.
        """
        if q.ndim == 1:
            log_and_raise("q should be [batch_size, dof]")
        if q.shape[-1] != self.dof:
            log_and_raise(f"q should have dof = {self.dof}, got {q.shape[-1]}")
        if q.ndim == 2:
            q = q.unsqueeze(1)
        state = self._forward(q)

        sph_all = state.get_link_spheres().squeeze(1).cpu().numpy()

        sph_traj = []
        for j in range(sph_all.shape[0]):
            sph = sph_all[j, :, :]
            if filter_valid:
                sph_list = [
                    Sphere(
                        name="curobo/robot_sphere_" + str(i),
                        pose=[sph[i, 0], sph[i, 1], sph[i, 2], 1, 0, 0, 0],
                        radius=sph[i, 3],
                    )
                    for i in range(sph.shape[0])
                    if (sph[i, 3] > 0.0)
                ]
            else:
                sph_list = [
                    Sphere(
                        name="curobo/robot_sphere_" + str(i),
                        pose=[sph[i, 0], sph[i, 1], sph[i, 2], 1, 0, 0, 0],
                        radius=sph[i, 3],
                    )
                    for i in range(sph.shape[0])
                ]
            sph_traj.append(sph_list)
        return sph_traj

    def get_link_poses(self, joint_position: torch.Tensor, query_link_names: List[str]) -> Pose:
        """Get Pose of links at given joint configuration using forward kinematics.

        Note that only the links specified in :class:`~KinematicsCfg.tool_frames` are returned.

        Args:
            joint_position: Joint configuration of the robot, shape should be [batch_size, dof].
            query_link_names: Names of links to get pose of. This should be a subset of
                :class:`~KinematicsCfg.tool_frames`.

        Returns:
            Pose: Poses of links at given joint configuration.
        """
        if joint_position.ndim == 2:
            joint_position = joint_position.unsqueeze(1)
        state = self._forward(joint_position)
        position = torch.zeros(
            (joint_position.shape[0], len(query_link_names), 3),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        quaternion = torch.zeros(
            (joint_position.shape[0], len(query_link_names), 4),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        for li, l in enumerate(query_link_names):
            i = self.tool_frames.index(l)
            pose = state.tool_poses.get_link_pose(l)
            position[:, li, :] = pose.position
            quaternion[:, li, :] = pose.quaternion
        return Pose(position=position, quaternion=quaternion)

    @property
    def all_articulated_joint_names(self) -> List[str]:
        """Names of all articulated joints of the robot."""
        return self.config.kinematics_config.non_fixed_joint_names

    def get_self_collision_config(self) -> SelfCollisionKinematicsCfg:
        """Get self collision configuration parameters of the robot."""
        return self.config.self_collision_config

    def get_link_mesh(self, link_name: str) -> Mesh:
        """Get mesh of a link of the robot."""
        #mesh = self.config.kinematics_parser.get_link_mesh(link_name)
        mesh_list = self.config.kinematics_parser.get_link_geometry(link_name)
        if len(mesh_list) == 0:
            return None
        if len(mesh_list) == 1:
            mesh = mesh_list[0]
            # Ensure mesh has the link name set (parser may return None for name)
            if mesh.name is None:
                mesh.name = link_name
        else:
            # load each mesh as a trimesh object:
            trimesh_list = []
            for mesh in mesh_list:
                trimesh_list.append(mesh.get_trimesh_mesh(transform_with_pose=True))
            trimesh_mesh = trimesh.util.concatenate(trimesh_list)
            mesh = Mesh(name=link_name, vertices=trimesh_mesh.vertices,
             faces=trimesh_mesh.faces, scale=1.0,
            pose=[0, 0, 0, 1, 0, 0, 0])


        return mesh

    def get_link_transform(self, link_name: str) -> Pose:
        """Get pose offset of a link from it's parent joint.

        Args:
            link_name: Name of link to get pose of.

        Returns:
            Pose: Pose of the link.
        """
        mat = self.config.kinematics_config.fixed_transforms[
            self.config.kinematics_config.link_name_to_idx_map[link_name]
        ]
        pose = Pose(position=mat[:3, 3].contiguous(), rotation=mat[:3, :3].contiguous())
        return pose

    def get_all_link_transforms(self) -> Pose:
        """Get offset pose of all links with respect to their parent joint."""
        pose = Pose(
            self.config.kinematics_config.fixed_transforms[:, :3, 3].contiguous(),
            rotation=self.config.kinematics_config.fixed_transforms[:, :3, :3].contiguous(),
        )
        return pose

    def get_dof(self) -> int:
        """Get degrees of freedom of the robot."""
        return self.config.kinematics_config.num_dof

    @property
    def dof(self) -> int:
        """Degrees of freedom of the robot."""
        return self.config.kinematics_config.num_dof

    @property
    def joint_names(self) -> List[str]:
        """Names of actuated joints."""
        return self.config.kinematics_config.joint_names

    @property
    def total_spheres(self) -> int:
        """Number of spheres used to approximate robot geometry."""
        return self.config.kinematics_config.total_spheres

    @property
    def lock_jointstate(self) -> JointState:
        """State of joints that are locked in the kinematic representation."""
        return self.config.kinematics_config.lock_jointstate

    def get_full_js(self, joint_state: JointState) -> JointState:
        """Get state of all joints, including locked joints.

        This function will not provide state of mimic joints. If you need mimic joints, use
        :func:`~get_mimic_js`.

        Args:
            joint_state: State containing articulated joints.

        Returns:
            JointState: State of all joints.
        """
        all_joint_names = self.all_articulated_joint_names
        lock_joint_state = self.lock_jointstate

        new_js = augment_joint_state(joint_state, all_joint_names, lock_joint_state)
        return new_js

    def get_mimic_js(self, joint_state: JointState) -> JointState:
        """Get state of mimic joints from active joints.

        Current implementation uses a for loop over joints to calculate the state. This can be
        optimized by using a custom CUDA kernel or a matrix multiplication.

        Args:
            joint_state: State containing articulated joints.

        Returns:
            JointState: State of active, locked, and mimic joints.
        """
        if self.config.kinematics_config.mimic_joints is None:
            return None
        extra_joints = {"position": [], "joint_names": []}
        # for every joint in mimic_joints, get active joint name
        for j in self.config.kinematics_config.mimic_joints:
            active_q = joint_state.position[..., joint_state.joint_names.index(j)]
            for k in self.config.kinematics_config.mimic_joints[j]:
                extra_joints["joint_names"].append(k["joint_name"])
                extra_joints["position"].append(
                    k["joint_offset"][0] * active_q + k["joint_offset"][1]
                )
        # Check if any mimic joints were actually found
        if len(extra_joints["position"]) == 0:
            return None
        extra_js = JointState.from_position(
            position=torch.stack(extra_joints["position"]).view(-1),
            joint_names=extra_joints["joint_names"],
        )
        new_js = augment_joint_state(joint_state, joint_state.joint_names + extra_js.joint_names, extra_js)
        return new_js

    def update_kinematics_config(self, new_kin_config: KinematicsParams):
        """Update kinematics representation of the robot.

        A kinematics representation can be updated with new parameters. Some parameters that could
        require updating are state of locked joints, when a robot grasps an object. Another instance
        is when using different planners for different parts of the robot, example updating the
        state of robot base or another arm. Updations should result in the same tensor dimensions,
        if not then the instance of this class requires reinitialization.

        Args:
            new_kin_config: New kinematics representation of the robot.
        """
        self.config.kinematics_config.copy_(new_kin_config)

    def get_active_js(self, full_js: JointState):
        """Get joint state of active joints of the robot.

        Args:
            full_js: Joint state of all joints.

        Returns:
            JointState: Joint state of active joints.
        """
        active_jnames = self.joint_names
        out_js = full_js.reorder(active_jnames)
        return out_js

    @property
    def base_link(self) -> str:
        """Base link of the robot. Changing requires reinitializing this class."""
        return self.config.kinematics_config.base_link

    @property
    def robot_spheres(self):
        """Spheres representing robot geometry (config 0). Shape [num_spheres, 4]."""
        return self.config.kinematics_config.link_spheres[0]

    @property
    def default_joint_position(self) -> torch.Tensor:
        """Default joint position of the robot. Use :func:`~joint_names` to get joint names."""
        return self.config.kinematics_config.cspace.default_joint_position

    @property
    def default_joint_state(self) -> JointState:
        """Default joint state of the robot. Use :func:`~joint_names` to get joint names."""
        return JointState.from_position(self.default_joint_position, joint_names=self.joint_names)

    def get_joint_limits(self) -> JointLimits:
        """Get joint limits of the robot."""
        return self.config.get_joint_limits()

    @property
    def kinematics_config(self) -> KinematicsParams:
        """Kinematics configuration of the robot."""
        return self.config.kinematics_config
