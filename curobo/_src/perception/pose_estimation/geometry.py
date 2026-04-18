# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Geometry models for pose estimation."""

import numpy as np
import torch
import trimesh

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


class RigidObjectGeometry:
    """Rigid object geometry (0 DOF).

    Represents static objects like boxes, tables, furniture.
    Surface points and normals are sampled from a fixed mesh.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        """Initialize rigid object geometry.

        Args:
            mesh: Trimesh object.
            device_cfg: Device/dtype configuration.
        """
        self.mesh = mesh
        self._tensor_args = device_cfg

    def update(self, joint_angles: torch.Tensor):
        """Update configuration (no-op for rigid objects).

        Args:
            joint_angles: Ignored for rigid objects.
        """
        pass  # No-op for rigid objects

    def sample_surface_points(
        self,
        n_points: int,
    ):
        """Sample points and normals from static mesh.

        Args:
            n_points: Number of points to sample.

        Returns:
            (points, normals): Tuple of [n_points, 3] tensors.
        """
        # Sample points and get face indices
        # These points might not be exactly n_points:
        points, face_indices = trimesh.sample.sample_surface_even(self.mesh, n_points)

        if len(points) != n_points:
            extra_points, extra_face_indices = trimesh.sample.sample_surface(self.mesh, n_points - len(points))
            points = np.concatenate([points, extra_points], axis=0)
            face_indices = np.concatenate([face_indices, extra_face_indices], axis=0)
        points_tensor = torch.tensor(
            points,
            device=self._tensor_args.device,
            dtype=self._tensor_args.dtype,
        )

        # Get face normals at sampled points
        face_normals = self.mesh.face_normals[face_indices]
        normals_tensor = torch.tensor(
            face_normals,
            device=self._tensor_args.device,
            dtype=self._tensor_args.dtype,
        )

        return points_tensor, normals_tensor

    def get_dof(self) -> int:
        """Rigid object has 0 degrees of freedom."""
        return 0

    @property
    def device_cfg(self):
        return self._tensor_args


class ArticulatedRobotGeometry:
    """Articulated robot geometry (n DOF).

    Efficiently represents robot surface at any configuration using:
    1. Cache surface points and normals for each link (in link frames)
    2. At runtime, transform cached data using FK

    This is 300-400× faster than merging meshes and is differentiable.
    """

    def __init__(
        self,
        robot_model: Kinematics,
        device_cfg: DeviceCfg = DeviceCfg(),
        points_per_cubic_meter: float = 150000.0,
        min_points_per_link: int = 50,
        max_points_per_link: int = 200,
    ):
        """Initialize articulated robot geometry with cached surface points.

        Args:
            robot_model: CuRobo robot model.
            device_cfg: Device/dtype configuration.
            points_per_cubic_meter: Target sampling density (points/m³).
            min_points_per_link: Minimum points per link.
            max_points_per_link: Maximum points per link.
        """
        self.robot_model = robot_model
        self._tensor_args = device_cfg
        self._n_dof = len(robot_model.joint_names)

        # Initialize cached surface points and normals
        self._initialize_cached_points(
            points_per_cubic_meter, min_points_per_link, max_points_per_link
        )

    def _initialize_cached_points(
        self,
        points_per_cubic_meter: float,
        min_points_per_link: int,
        max_points_per_link: int,
    ):
        """Cache surface points and normals for each link in its local frame.

        Points are distributed based on target density (points per cubic meter).
        Each link gets min_points_per_link to max_points_per_link.
        """
        # Get link meshes in their local frames
        link_meshes = self.robot_model.get_robot_link_meshes()
        mesh_link_names = self.robot_model.config.kinematics_config.mesh_link_names

        print(f"\nInitializing cached surface points for {len(link_meshes)} links...")
        print(f"  Target density: {points_per_cubic_meter:.0f} points/m³")
        print(f"  Point bounds: {min_points_per_link} to {max_points_per_link} per link")

        # Compute volumes and points for each link
        link_volumes = []
        points_per_link = []

        for i, (mesh, link_name) in enumerate(zip(link_meshes, mesh_link_names)):
            trimesh_mesh = mesh.get_trimesh_mesh()
            volume = abs(trimesh_mesh.volume)  # abs in case of inverted normals
            link_volumes.append(volume)

            # Calculate points based on density
            n_points = int(volume * points_per_cubic_meter)
            n_points = min(max_points_per_link, max(min_points_per_link, n_points))
            points_per_link.append(n_points)

        total_volume = sum(link_volumes)
        print(f"  Total volume: {total_volume:.6f} m³")
        print("  Point distribution:")

        # Sample and cache points and normals for each link
        self.cached_link_points = []
        self.cached_link_normals = []
        self.cached_link_names = mesh_link_names

        actual_total = 0
        for i, (mesh, link_name, volume, n_points) in enumerate(
            zip(link_meshes, mesh_link_names, link_volumes, points_per_link)
        ):
            # Sample in mesh frame
            trimesh_mesh = mesh.get_trimesh_mesh()
            points, face_indices = trimesh.sample.sample_surface_even(trimesh_mesh, n_points)
            points_tensor = torch.tensor(
                points,
                device=self._tensor_args.device,
                dtype=self._tensor_args.dtype,
            )

            # Get normals at sampled points (from face normals)
            face_normals = trimesh_mesh.face_normals[face_indices]
            normals_tensor = torch.tensor(
                face_normals,
                device=self._tensor_args.device,
                dtype=self._tensor_args.dtype,
            )

            # Apply mesh's local offset to get points and normals in link frame
            mesh_local_pose = Pose.from_list(mesh.pose, device_cfg=self._tensor_args)
            points_link_frame = mesh_local_pose.transform_points(points_tensor)

            # Transform normals (rotation only, no translation)
            R_local = mesh_local_pose.get_rotation()
            if R_local.ndim == 3:
                R_local = R_local.squeeze(0)
            normals_link_frame = (R_local @ normals_tensor.T).T

            self.cached_link_points.append(points_link_frame)
            self.cached_link_normals.append(normals_link_frame)
            actual_total += len(points_link_frame)

            actual_density = n_points / volume if volume > 0 else 0
            print(
                f"    {i}: {link_name:20s} - {n_points:3d} points "
                f"({volume * 1000:6.2f} cm³, {actual_density:7.0f} pts/m³)"
            )

        actual_density = actual_total / total_volume
        print(f"  Total cached: {actual_total} points")
        print(f"  Actual density: {actual_density:.0f} points/m³\n")

    def update(self, joint_angles: torch.Tensor):
        """Update robot configuration.

        Args:
            joint_angles: Joint angles [num_dof] or [1, num_dof].
        """
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)
        self._current_config = joint_angles

    def sample_surface_points(
        self,
        n_points: int,
    ):
        """Sample points and normals from robot mesh at current configuration.

        Uses cached link data + FK for efficiency and differentiability.
        Call update() first to set joint configuration.

        Args:
            n_points: Number of points to sample.

        Returns:
            (points, normals): Tuple of [n_points, 3] tensors.
        """
        if not hasattr(self, "_current_config") or self._current_config is None:
            raise ValueError("Must call update() with joint angles before sampling")

        # Get kinematic state
        state = self.robot_model.compute_kinematics(
            JointState.from_position(self._current_config, joint_names=self.robot_model.joint_names)
        )
        if state.tool_poses is None:
            raise ValueError("KinematicsState.tool_poses is None")

        # Transform each link's cached points and normals using FK
        all_points = []
        all_normals = []

        for i, link_points in enumerate(self.cached_link_points):
            link_name = self.cached_link_names[i]
            link_pose = state.tool_poses[link_name]

            # Transform points from link frame to base frame
            transformed_points = link_pose.transform_points(link_points)
            all_points.append(transformed_points)

            # Transform normals (rotation only)
            link_normals = self.cached_link_normals[i]
            R = link_pose.get_rotation()
            if R.ndim == 3:
                R = R.squeeze(0)
            transformed_normals = (R @ link_normals.T).T
            all_normals.append(transformed_normals)

        # Concatenate all links
        robot_points = torch.cat(all_points, dim=0)
        robot_normals = torch.cat(all_normals, dim=0)

        # Downsample to requested count if needed
        if len(robot_points) > n_points:
            indices = torch.randperm(len(robot_points), device=robot_points.device)[:n_points]
            robot_points = robot_points[indices]
            robot_normals = robot_normals[indices]

        return robot_points, robot_normals

    def get_dof(self) -> int:
        """Return number of robot joints."""
        return self._n_dof

    @property
    def device_cfg(self):
        return self._tensor_args
