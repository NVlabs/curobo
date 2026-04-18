# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unified mesh representation for pose estimation.

RobotMesh provides a single interface for both rigid objects and articulated robots,
supporting both ICP-based and SDF-based pose detection methods.

Key features:
- Unified API for rigid and articulated meshes
- In-place vertex updates with BVH refit (CUDA graph compatible)
- Surface point sampling for ICP
- wp.Mesh for SDF queries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import trimesh
import warp as wp

from curobo._src.state.state_joint import JointState

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.types.pose import Pose


@dataclass
class SurfaceSampleCache:
    """Pre-computed indices for surface point sampling.

    Stores face indices and barycentric coordinates for consistent sampling.

    Attributes:
        face_indices: [N] indices into faces array for each sample.
        bary_coords: [N, 3] barycentric coordinates for each sample.
    """

    face_indices: torch.Tensor
    bary_coords: torch.Tensor


class RobotMesh:
    """Unified mesh representation for rigid objects and articulated robots.

    Provides mesh data for both ICP-based and SDF-based pose detection:
    - `mesh` property: wp.Mesh for SDF queries
    - `sample_surface_points()`: returns points + normals for ICP

    For articulated robots, call `update(joint_angles)` to transform
    vertices and refit BVH. The mesh reference remains constant (CUDA graph safe).

    Example:
        # Rigid object
        robot_mesh = RobotMesh.from_trimesh(trimesh.load("object.stl"))
        points, normals = robot_mesh.sample_surface_points(1000)

        # Articulated robot
        robot_mesh = RobotMesh.from_kinematics(kinematics)
        robot_mesh.update(joint_angles)
        points, normals = robot_mesh.sample_surface_points(1000)

        # Use with SDF detector
        sdf = wp.mesh_query_point(robot_mesh.mesh.id, query_point, max_dist)
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        device: str = "cuda:0",
        kinematics: Optional[Kinematics] = None,
        link_vertex_ranges: Optional[List[Tuple[int, int]]] = None,
        link_names: Optional[List[str]] = None,
        link_vertices_local: Optional[List[torch.Tensor]] = None,
    ):
        """Initialize RobotMesh.

        Use factory methods `from_trimesh()` or `from_kinematics()` instead of
        calling this constructor directly.

        Args:
            vertices: [V, 3] mesh vertices in world frame.
            faces: [F, 3] face indices.
            device: CUDA device string.
            kinematics: Kinematics instance for articulated robots.
            link_vertex_ranges: List of (start, end) indices for each link's vertices.
            link_names: List of link names (for FK lookup).
            link_vertices_local: List of vertices in each link's local frame.
        """
        self.device = device
        self._kinematics = kinematics
        self._link_vertex_ranges = link_vertex_ranges
        self._link_names = link_names
        self._link_vertices_local = link_vertices_local
        self._is_articulated = kinematics is not None

        # Store vertices on device
        self._vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
        self._faces = faces.to(device=device, dtype=torch.int32).contiguous()

        # Create Warp mesh
        self._vertices_wp = wp.from_torch(self._vertices, dtype=wp.vec3)
        self._mesh = wp.Mesh(
            points=self._vertices_wp,
            indices=wp.from_torch(self._faces.view(-1), dtype=wp.int32),
        )

        # Pre-compute surface samples (indices + barycentric coords)
        self._sample_cache: Optional[SurfaceSampleCache] = None

        # Current joint configuration (for articulated robots)
        if self._is_articulated:
            n_joints = len(kinematics.joint_names)
            self._current_joints = torch.zeros(n_joints, device=device, dtype=torch.float32)
        else:
            self._current_joints = None

    @classmethod
    def from_trimesh(
        cls,
        mesh: trimesh.Trimesh,
        device: str = "cuda:0",
    ) -> RobotMesh:
        """Create RobotMesh from a trimesh object (rigid object).

        Args:
            mesh: Trimesh object.
            device: CUDA device string.

        Returns:
            RobotMesh instance for rigid object.
        """
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int32)

        return cls(
            vertices=vertices,
            faces=faces,
            device=device,
            kinematics=None,
            link_vertex_ranges=None,
            link_names=None,
            link_vertices_local=None,
        )

    @classmethod
    def from_kinematics(
        cls,
        kinematics: Kinematics,
        device: str = "cuda:0",
        initial_joint_angles: Optional[torch.Tensor] = None,
    ) -> RobotMesh:
        """Create RobotMesh from a Kinematics instance (articulated robot).

        Combines all link meshes into a single mesh. Call `update(joint_angles)`
        to transform vertices based on FK.

        Args:
            kinematics: CuRobo Kinematics instance.
            device: CUDA device string.
            initial_joint_angles: Initial joint configuration. If None, uses zeros.

        Returns:
            RobotMesh instance for articulated robot.
        """
        # Get all link meshes
        link_meshes = kinematics.get_robot_link_meshes()
        mesh_link_names = kinematics.config.kinematics_config.mesh_link_names

        # Combine all link meshes
        all_vertices = []
        all_faces = []
        link_vertex_ranges = []
        link_vertices_local = []
        vertex_offset = 0

        for mesh, link_name in zip(link_meshes, mesh_link_names):
            tm = mesh.get_trimesh_mesh()

            # Get vertices in link frame (apply mesh's local offset)
            local_pose = Pose.from_list(mesh.pose)
            verts_local = local_pose.transform_points(
                torch.tensor(tm.vertices, dtype=torch.float32, device=device)
            )

            # Store for FK transformation
            link_vertices_local.append(verts_local)

            # Track vertex range for this link
            start_idx = vertex_offset
            end_idx = vertex_offset + len(verts_local)
            link_vertex_ranges.append((start_idx, end_idx))

            # Add to combined mesh
            all_vertices.append(verts_local)

            # Offset face indices
            faces_offset = torch.tensor(tm.faces, dtype=torch.int32, device=device) + vertex_offset
            all_faces.append(faces_offset)

            vertex_offset = end_idx

        # Concatenate all links
        combined_vertices = torch.cat(all_vertices, dim=0)
        combined_faces = torch.cat(all_faces, dim=0)

        # Create instance
        robot_mesh = cls(
            vertices=combined_vertices,
            faces=combined_faces,
            device=device,
            kinematics=kinematics,
            link_vertex_ranges=link_vertex_ranges,
            link_names=mesh_link_names,
            link_vertices_local=link_vertices_local,
        )

        # Apply initial joint configuration
        if initial_joint_angles is not None:
            robot_mesh.update(initial_joint_angles)
        else:
            # Initialize at zero config
            robot_mesh.update(torch.zeros(len(kinematics.joint_names), device=device))

        return robot_mesh

    @property
    def mesh(self) -> wp.Mesh:
        """Get Warp mesh for SDF queries.

        The mesh reference is constant (CUDA graph safe), but vertices
        are updated in-place when `update()` is called.

        Returns:
            wp.Mesh instance.
        """
        return self._mesh

    @property
    def mesh_id(self) -> wp.uint64:
        """Get Warp mesh ID for kernel use.

        Returns:
            Mesh ID as wp.uint64.
        """
        return self._mesh.id

    @property
    def vertices(self) -> torch.Tensor:
        """Get current vertices as torch tensor.

        Returns:
            [V, 3] vertex positions in world frame.
        """
        return self._vertices

    @property
    def faces(self) -> torch.Tensor:
        """Get face indices.

        Returns:
            [F, 3] face vertex indices.
        """
        return self._faces

    @property
    def n_vertices(self) -> int:
        """Number of vertices in mesh."""
        return len(self._vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces in mesh."""
        return len(self._faces)

    @property
    def is_articulated(self) -> bool:
        """True if this is an articulated robot mesh."""
        return self._is_articulated

    @property
    def current_joint_angles(self) -> Optional[torch.Tensor]:
        """Current joint configuration (None for rigid objects)."""
        return self._current_joints

    def update(self, joint_angles: torch.Tensor) -> None:
        """Update mesh vertices based on joint configuration.

        Transforms all link vertices using FK and refits BVH.
        No-op for rigid objects.

        Args:
            joint_angles: [N_joints] tensor of joint angles.
        """
        if not self._is_articulated:
            return  # No-op for rigid objects

        # Ensure correct shape and device
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)
        joint_angles = joint_angles.to(device=self.device, dtype=torch.float32)

        # Compute FK to get link poses
        state = self._kinematics.compute_kinematics(
            JointState.from_position(joint_angles, joint_names=self._kinematics.joint_names)
        )
        if state.tool_poses is None:
            raise ValueError("KinematicsState.tool_poses is None")

        # Transform each link's vertices
        for i, link_name in enumerate(self._link_names):
            link_pose = state.tool_poses[link_name]
            start_idx, end_idx = self._link_vertex_ranges[i]

            # Transform from link frame to world frame
            local_verts = self._link_vertices_local[i]
            world_verts = link_pose.transform_points(local_verts)

            # Update in-place
            self._vertices[start_idx:end_idx] = world_verts

        # Sync Warp array and refit BVH
        wp.copy(self._vertices_wp, wp.from_torch(self._vertices, dtype=wp.vec3))
        self._mesh.refit()

        # Store current configuration
        self._current_joints = joint_angles.squeeze(0).clone()

    def sample_surface_points(
        self,
        n_points: int,
        resample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points and normals from mesh surface.

        Uses pre-cached sample indices for consistency. Set `resample=True`
        to generate new random samples.

        Args:
            n_points: Number of points to sample.
            resample: If True, regenerate sample indices. If False, reuse cached.

        Returns:
            points: [n_points, 3] surface points in world frame.
            normals: [n_points, 3] surface normals (unit vectors).
        """
        # Check if we need to regenerate cache
        if (
            self._sample_cache is None
            or len(self._sample_cache.face_indices) != n_points
            or resample
        ):
            self._sample_cache = self._generate_sample_cache(n_points)

        # Get face vertices using cached indices
        face_idx = self._sample_cache.face_indices
        bary = self._sample_cache.bary_coords

        # Get 3 vertices per face
        v0 = self._vertices[self._faces[face_idx, 0]]  # [n_points, 3]
        v1 = self._vertices[self._faces[face_idx, 1]]
        v2 = self._vertices[self._faces[face_idx, 2]]

        # Barycentric interpolation for point positions
        points = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2

        # Compute normals from cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = torch.cross(edge1, edge2, dim=1)
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)

        return points, normals

    def _generate_sample_cache(self, n_points: int) -> SurfaceSampleCache:
        """Generate cached sample indices and barycentric coordinates.

        Samples faces weighted by area for uniform distribution.

        Args:
            n_points: Number of samples to generate.

        Returns:
            SurfaceSampleCache with face indices and barycentric coords.
        """
        # Compute face areas for weighted sampling
        v0 = self._vertices[self._faces[:, 0]]
        v1 = self._vertices[self._faces[:, 1]]
        v2 = self._vertices[self._faces[:, 2]]

        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        areas = 0.5 * torch.norm(cross, dim=1)

        # Sample faces weighted by area
        probs = areas / (areas.sum() + 1e-8)
        face_indices = torch.multinomial(probs, n_points, replacement=True)

        # Generate random barycentric coordinates
        # Use sqrt for uniform distribution on triangle
        r1 = torch.sqrt(torch.rand(n_points, device=self.device))
        r2 = torch.rand(n_points, device=self.device)

        bary = torch.zeros(n_points, 3, device=self.device)
        bary[:, 0] = 1 - r1
        bary[:, 1] = r1 * (1 - r2)
        bary[:, 2] = r1 * r2

        return SurfaceSampleCache(face_indices=face_indices, bary_coords=bary)

    def get_dof(self) -> int:
        """Get degrees of freedom.

        Returns:
            0 for rigid objects, n_joints for articulated robots.
        """
        if self._kinematics is None:
            return 0
        return len(self._kinematics.joint_names)

    def get_trimesh(self) -> trimesh.Trimesh:
        """Get current mesh as trimesh object.

        Useful for visualization and debugging.

        Returns:
            trimesh.Trimesh with current vertex positions.
        """
        return trimesh.Trimesh(
            vertices=self._vertices.cpu().numpy(),
            faces=self._faces.cpu().numpy(),
        )

