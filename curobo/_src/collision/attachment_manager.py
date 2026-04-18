# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Manages attaching/detaching obstacles to robot links for collision-aware planning."""

from __future__ import annotations

from typing import List, Optional

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitResult
from curobo._src.geom.types import Obstacle
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_info


class AttachmentManager:
    """Manages attaching/detaching obstacles to robot links.

    Operates on :class:`KinematicsParams` (sphere geometry per env) and optionally
    :class:`SceneCollision` (world obstacle enable/disable). Same obstacles across all
    envs; attach/detach always applies to all envs.

    Each env can have a different robot pose and obstacle world pose. Sphere fitting
    is done once. Per-env obstacle-to-link offsets are computed via batched FK, producing
    per-env link-local spheres written to ``link_spheres[i]``. The FK kernel then
    transforms each env's link-local spheres to world frame during optimization.
    """

    def __init__(
        self,
        kinematics: Kinematics,
        scene_collision: Optional[SceneCollision] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        """Initialize attachment manager.

        Args:
            kinematics: Robot kinematics model used for FK.
            scene_collision: Scene collision checker for obstacle enable/disable.
            device_cfg: Device and dtype configuration.
        """
        self._kinematics = kinematics
        self._scene_collision = scene_collision
        self._device_cfg = device_cfg
        self._last_fit_result: Optional[SphereFitResult] = None
        self._attached_link_name: Optional[str] = None
        self._disabled_obstacle_names: List[str] = []
        self._disabled_num_envs: int = 0

    @property
    def kinematics_params(self) -> KinematicsParams:
        """Access the kinematics parameters that hold link_spheres."""
        return self._kinematics.config.kinematics_config

    def fit_spheres(
        self,
        obstacles: List[Obstacle],
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
    ) -> torch.Tensor:
        """Fit spheres to obstacle geometry. Expensive; call once, reuse across envs.

        Converts obstacles to a combined trimesh, then delegates to
        :func:`fit_spheres_to_mesh`.

        Args:
            obstacles: Obstacle primitives to approximate with spheres.
            num_spheres: Number of spheres to fit. None for automatic estimation.
            surface_radius: Radius for surface-sampled spheres.
            sphere_fit_type: Fitting algorithm.

        Returns:
            torch.Tensor: Sphere positions and radii in obstacle frame [num_spheres, 4].
        """
        combined_mesh = self._obstacles_to_trimesh(obstacles)
        result = fit_spheres_to_mesh(
            combined_mesh,
            num_spheres=num_spheres,
            surface_radius=surface_radius,
            fit_type=sphere_fit_type,
            device_cfg=self._device_cfg,
        )
        self._last_fit_result = result
        sphere_tensor = torch.cat(
            [result.centers, result.radii.unsqueeze(-1)], dim=-1
        )
        log_info(
            f"AttachmentManager.fit_spheres: fitted {result.num_spheres} spheres"
            f" in {result.fit_time_s:.3f}s"
        )
        return sphere_tensor

    def update(
        self,
        sphere_tensor: torch.Tensor,
        joint_states: JointState,
        link_name: str = "attached_object",
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> None:
        """Compute per-env obstacle-to-link offsets and write link-local spheres.

        Runs batched FK on ``joint_states`` to get per-env link poses, computes
        per-env obstacle-to-link transforms, and writes the result to
        ``link_spheres[i, attached_indices, :]`` for each env.

        Args:
            sphere_tensor: Spheres in obstacle frame [num_spheres, 4] (x, y, z, r).
            joint_states: Per-env grasp configurations [num_envs, dof].
            link_name: Robot link to which obstacles are attached.
            world_objects_pose_offset: Obstacle world pose(s) [num_envs] or [1] (broadcast).
                When None, obstacles are assumed at the link origin (identity offset).
        """
        q = joint_states.position
        if q.dim() == 1:
            q = q.unsqueeze(0)
        num_envs = q.shape[0]

        kparams = self.kinematics_params
        link_sphere_idx = kparams.get_sphere_index_from_link_name(link_name)
        n_link_spheres = link_sphere_idx.shape[0]
        n_fit_spheres = sphere_tensor.shape[0]

        if n_fit_spheres > n_link_spheres:
            log_and_raise(
                f"Fitted {n_fit_spheres} spheres but link '{link_name}' only has"
                f" {n_link_spheres} sphere slots. Reduce num_spheres or increase"
                f" the link's sphere allocation."
            )

        centers = sphere_tensor[:, :3].contiguous()
        radii = sphere_tensor[:, 3]

        if world_objects_pose_offset is not None:
            ee_link = self._kinematics.tool_frames[0]
            joint_state = JointState.from_position(
                q, joint_names=self._kinematics.joint_names,
            )
            fk_result = self._kinematics.compute_kinematics(joint_state)
            if fk_result.tool_poses is None:
                log_and_raise("FK result has no tool_poses; cannot resolve EE for attachment offset.")
            ee_poses = fk_result.tool_poses.get_link_pose(ee_link)
            obj_to_link_poses = ee_poses.inverse().multiply(world_objects_pose_offset)
        else:
            obj_to_link_poses = None

        padding = None
        if n_fit_spheres < n_link_spheres:
            padding = torch.zeros(
                n_link_spheres - n_fit_spheres, 4,
                device=sphere_tensor.device, dtype=sphere_tensor.dtype,
            )
            padding[:, 3] = -100.0

        for i in range(num_envs):
            if obj_to_link_poses is not None:
                env_pose = Pose(
                    position=obj_to_link_poses.position[i: i + 1],
                    quaternion=obj_to_link_poses.quaternion[i: i + 1],
                )
                transformed = env_pose.transform_points(centers).squeeze(0)
            else:
                transformed = centers

            env_spheres = torch.cat([transformed, radii.unsqueeze(-1)], dim=-1)

            if padding is not None:
                env_spheres = torch.cat([env_spheres, padding], dim=0)

            kparams.link_spheres[i, link_sphere_idx, :] = env_spheres

        self._attached_link_name = link_name

    def attach(
        self,
        joint_states: JointState,
        obstacles: List[Obstacle],
        link_name: str = "attached_object",
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
        world_objects_pose_offset: Optional[Pose] = None,
        disable_obstacle_names: Optional[List[str]] = None,
    ) -> None:
        """Fit spheres, update all envs, and optionally disable world obstacles.

        Convenience method combining :meth:`fit_spheres` + :meth:`update` +
        world obstacle disable.

        Args:
            joint_states: Per-env grasp configurations [num_envs, dof].
            obstacles: Obstacle primitives to attach.
            link_name: Robot link to attach to.
            num_spheres: Number of spheres to fit. None for automatic.
            surface_radius: Radius for surface-sampled spheres.
            sphere_fit_type: Fitting algorithm.
            world_objects_pose_offset: Obstacle world pose(s) [num_envs] or [1] (broadcast).
            disable_obstacle_names: World obstacle names to disable across all envs.
        """
        sphere_tensor = self.fit_spheres(
            obstacles,
            num_spheres=num_spheres,
            surface_radius=surface_radius,
            sphere_fit_type=sphere_fit_type,
        )
        self.update(sphere_tensor, joint_states, link_name, world_objects_pose_offset)

        if disable_obstacle_names and self._scene_collision is not None:
            num_envs = self._get_num_envs(joint_states)
            for name in disable_obstacle_names:
                for env_idx in range(num_envs):
                    self._scene_collision.enable_obstacle(
                        name, enable=False, env_idx=env_idx,
                    )
            self._disabled_obstacle_names = list(disable_obstacle_names)
            self._disabled_num_envs = num_envs

    def attach_from_scene(
        self,
        joint_states: JointState,
        obstacle_names: List[str],
        link_name: str = "attached_object",
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> None:
        """Attach obstacles that already exist in the scene, looked up by name.

        Resolves obstacle geometry from :attr:`SceneCollision.scene_model`, fits
        spheres, updates all envs, and auto-disables the named obstacles in the
        world collision checker.

        Args:
            joint_states: Per-env grasp configurations [num_envs, dof].
            obstacle_names: Names of obstacles in the scene to attach.
            link_name: Robot link to attach to.
            num_spheres: Number of spheres to fit. None for automatic.
            surface_radius: Radius for surface-sampled spheres.
            sphere_fit_type: Fitting algorithm.
            world_objects_pose_offset: Obstacle world pose(s) [num_envs] or [1] (broadcast).
        """
        if self._scene_collision is None:
            log_and_raise(
                "attach_from_scene requires scene_collision to be set."
            )
        scene_model = self._scene_collision.scene_model
        if scene_model is None:
            log_and_raise(
                "attach_from_scene requires scene_collision.scene_model to be set."
            )

        if isinstance(scene_model, list):
            scene_model = scene_model[0]

        obstacles = []
        for name in obstacle_names:
            obs = scene_model.get_obstacle(name)
            if obs is None:
                log_and_raise(
                    f"Obstacle '{name}' not found in scene_collision.scene_model."
                )
            obstacles.append(obs)

        self.attach(
            joint_states,
            obstacles,
            link_name=link_name,
            num_spheres=num_spheres,
            surface_radius=surface_radius,
            sphere_fit_type=sphere_fit_type,
            world_objects_pose_offset=world_objects_pose_offset,
            disable_obstacle_names=obstacle_names,
        )

    def detach(
        self,
        link_name: Optional[str] = None,
        enable_obstacle_names: Optional[List[str]] = None,
    ) -> None:
        """Detach obstacles from all envs.

        Resets link spheres to reference values and re-enables world obstacles.

        Args:
            link_name: Link to detach from. None uses the last attached link.
            enable_obstacle_names: World obstacle names to re-enable. None re-enables
                obstacles disabled during the last :meth:`attach` call.
        """
        if link_name is None:
            link_name = self._attached_link_name
        if link_name is None:
            return

        self.kinematics_params.reset_link_spheres(link_name)

        names_to_enable = enable_obstacle_names or self._disabled_obstacle_names
        if names_to_enable and self._scene_collision is not None:
            for name in names_to_enable:
                for env_idx in range(self._disabled_num_envs):
                    self._scene_collision.enable_obstacle(
                        name, enable=True, env_idx=env_idx,
                    )

        self._attached_link_name = None
        self._disabled_obstacle_names = []
        self._disabled_num_envs = 0

    @staticmethod
    def _obstacles_to_trimesh(obstacles: List[Obstacle]):
        """Convert a list of obstacles into a single combined trimesh.

        Args:
            obstacles: Obstacle primitives to combine.

        Returns:
            trimesh.Trimesh: Combined mesh in obstacle frame.
        """
        import trimesh

        meshes = []
        for obs in obstacles:
            mesh = obs.get_trimesh_mesh(transform_with_pose=True)
            meshes.append(mesh)
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)

    @staticmethod
    def _get_num_envs(joint_states: JointState) -> int:
        """Get number of environments from joint states."""
        q = joint_states.position
        if q.dim() == 1:
            return 1
        return q.shape[0]
