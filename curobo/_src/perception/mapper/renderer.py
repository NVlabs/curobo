# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-Sparse TSDF Renderer - Render depth/normal/color images via raycasting.

This module provides a renderer for block-sparse TSDF storage that can
render synthetic depth, normal, and color images from arbitrary camera poses.

Features:
- Depth rendering with sub-voxel precision
- Surface normal visualization
- RGB color rendering from integrated color data
- Shaded visualization with Lambertian lighting

Usage:
    from curobo._src.perception.mapper import (
        BlockSparseESDFIntegrator,
        BlockSparseTSDFRenderer,
    )

    integrator = BlockSparseESDFIntegrator(config)
    renderer = BlockSparseTSDFRenderer(integrator)

    # Render depth and normals
    depth, normals, valid = renderer.render(intrinsics, pose, (480, 640))

    # Render with color
    depth, normals, colors, valid = renderer.render_color(intrinsics, pose, (480, 640))
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.wp_raycast import (
    raycast_block_sparse_accelerated_color_kernel,
    raycast_block_sparse_accelerated_kernel,
    raycast_block_sparse_color_kernel,
    raycast_block_sparse_kernel,
)
from curobo._src.types.pose import Pose


@dataclass
class BlockSparseTSDFRendererCfg:
    """Configuration for BlockSparseTSDFRenderer.

    Attributes:
        depth_minimum_distance: Minimum ray distance in meters.
        depth_maximum_distance: Maximum ray distance in meters.
        minimum_tsdf_weight: Minimum TSDF weight for valid voxel.
            Since weight = 1/depth² (clamped to [0.001, 2.0]), this can be interpreted
            as the number of observations at 1m depth. E.g., 0.5 = half an observation
            at 1m, or one observation at ~1.4m, or two observations at 2m.
        use_block_acceleration: If True, use block-level skipping to accelerate
            raymarching. This skips unallocated blocks entirely, providing 5-50x
            speedup depending on scene sparsity. Default True.
    """

    depth_minimum_distance: float = 0.2
    depth_maximum_distance: float = 15.0
    minimum_tsdf_weight: float = 0.2
    use_block_acceleration: bool = True


class BlockSparseTSDFRenderer:
    """Render depth, normal, and color images from block-sparse TSDF.

    Uses sphere tracing through the TSDF to render synthetic images
    from arbitrary camera poses. Supports rendering:
    - Depth images (in meters)
    - Surface normals (world frame)
    - RGB colors (from integrated TSDF color)

    Example:
        renderer = BlockSparseTSDFRenderer(integrator)
        depth, normals, valid = renderer.render(intrinsics, pose, (480, 640))
        depth_color = renderer.render_depth_colormap(intrinsics, pose, (480, 640))
        rgb = renderer.render_color_only(intrinsics, pose, (480, 640))
    """

    def __init__(
        self,
        integrator,  # BlockSparseESDFIntegrator or BlockSparseTSDFIntegrator
        use_block_acceleration: bool = True,
    ):
        """Initialize BlockSparseTSDFRenderer.

        Args:
            integrator: Integrator instance with block-sparse TSDF.
            use_block_acceleration: If True, use block-level skipping for faster
                raymarching. This skips unallocated blocks entirely, providing
                significant speedup for sparse scenes. Default True.
        """
        self.integrator = integrator
        self.config = BlockSparseTSDFRendererCfg(
            depth_minimum_distance=integrator.config.depth_minimum_distance,
            depth_maximum_distance=integrator.config.depth_maximum_distance,
            minimum_tsdf_weight=integrator.config.minimum_tsdf_weight,
            use_block_acceleration=use_block_acceleration,
        )
        self.device = integrator.device

        # Pre-allocated buffers
        self._buffer_size = 0
        self._hit_points = None
        self._hit_normals = None
        self._hit_colors = None
        self._hit_depths = None
        self._hit_mask = None

        # Camera parameter buffers
        self.cam_position = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.cam_quaternion = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.intrinsics_matrix = torch.zeros((3, 3), dtype=torch.float32, device=self.device)

    def _ensure_buffers(self, n_pixels: int):
        """Ensure output buffers are large enough."""
        if self._buffer_size >= n_pixels:
            return

        self._hit_points = torch.zeros((n_pixels, 3), dtype=torch.float32, device=self.device)
        self._hit_normals = torch.zeros((n_pixels, 3), dtype=torch.float32, device=self.device)
        self._hit_colors = torch.zeros((n_pixels, 3), dtype=torch.uint8, device=self.device)
        self._hit_depths = torch.zeros(n_pixels, dtype=torch.float32, device=self.device)
        self._hit_mask = torch.zeros(n_pixels, dtype=torch.uint8, device=self.device)
        self._buffer_size = n_pixels

    def _extract_pose(self, pose: Pose):
        """Extract position and quaternion from Pose.

        Args:
            pose: Camera-to-world transform as Pose.
        """
        position = pose.position.squeeze().to(self.device, dtype=torch.float32)
        quaternion = pose.quaternion.squeeze().to(self.device, dtype=torch.float32)
        self.cam_position.copy_(position)
        self.cam_quaternion.copy_(quaternion)

    def _extract_intrinsics(self, intrinsics: torch.Tensor):
        """Extract intrinsics matrix."""
        intrinsics = intrinsics.to(self.device, dtype=torch.float32)
        if intrinsics.dim() == 2:
            self.intrinsics_matrix.copy_(intrinsics[:3, :3])
        else:
            # [fx, fy, cx, cy] format
            fx, fy, cx, cy = intrinsics[:4].tolist()
            self.intrinsics_matrix.zero_()
            self.intrinsics_matrix[0, 0] = fx
            self.intrinsics_matrix[1, 1] = fy
            self.intrinsics_matrix[0, 2] = cx
            self.intrinsics_matrix[1, 2] = cy

    def render(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render depth and normal images from block-sparse TSDF.

        Args:
            intrinsics: [3, 3] or [4] camera intrinsics (fx, fy, cx, cy).
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output image dimensions.

        Returns:
            depth_image: [H, W] rendered depth in meters (0 where no hit).
            normal_image: [H, W, 3] surface normals in world frame.
            valid_mask: [H, W] boolean mask of valid pixels.
        """
        H, W = image_shape
        n_pixels = H * W

        self._ensure_buffers(n_pixels)
        self._extract_pose(pose)
        self._extract_intrinsics(intrinsics)

        # Get block-sparse TSDF data
        tsdf = self.integrator._tsdf
        warp_data = tsdf.get_warp_data()

        # Select kernel based on acceleration setting
        kernel = (
            raycast_block_sparse_accelerated_kernel
            if self.config.use_block_acceleration
            else raycast_block_sparse_kernel
        )

        # Launch raycast kernel with struct-based API
        wp.launch(
            kernel=kernel,
            dim=n_pixels,
            inputs=[
                wp.from_torch(self.intrinsics_matrix, dtype=wp.float32),
                wp.from_torch(self.cam_position, dtype=wp.float32),
                wp.from_torch(self.cam_quaternion, dtype=wp.float32),
                warp_data,
                self.config.depth_minimum_distance,
                self.config.depth_maximum_distance,
                self.config.minimum_tsdf_weight,
                wp.from_torch(self._hit_points[:n_pixels], dtype=wp.float32),
                wp.from_torch(self._hit_normals[:n_pixels], dtype=wp.float32),
                wp.from_torch(self._hit_depths[:n_pixels], dtype=wp.float32),
                wp.from_torch(self._hit_mask[:n_pixels], dtype=wp.uint8),
                H,
                W,
            ],
        )

        # Kernel clears hit_* to zero on miss, so output buffers are
        # authoritative and no additional masking is required here.
        depth_image = self._hit_depths[:n_pixels].view(H, W)
        normal_image = self._hit_normals[:n_pixels].view(H, W, 3)
        valid_mask = self._hit_mask[:n_pixels].view(H, W).bool()

        return depth_image, normal_image, valid_mask

    def render_color(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render depth, normals, and color from block-sparse TSDF.

        Args:
            intrinsics: [3, 3] or [4] camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.

        Returns:
            depth_image: [H, W] rendered depth in meters.
            normal_image: [H, W, 3] surface normals.
            color_image: [H, W, 3] uint8 RGB colors.
            valid_mask: [H, W] boolean mask.
        """
        H, W = image_shape
        n_pixels = H * W

        self._ensure_buffers(n_pixels)
        self._extract_pose(pose)
        self._extract_intrinsics(intrinsics)

        # Get block-sparse TSDF data
        tsdf = self.integrator._tsdf
        warp_data = tsdf.get_warp_data()

        # Select kernel based on acceleration setting
        kernel = (
            raycast_block_sparse_accelerated_color_kernel
            if self.config.use_block_acceleration
            else raycast_block_sparse_color_kernel
        )

        # Launch color raycast kernel with struct-based API
        wp.launch(
            kernel=kernel,
            dim=n_pixels,
            inputs=[
                wp.from_torch(self.intrinsics_matrix, dtype=wp.float32),
                wp.from_torch(self.cam_position, dtype=wp.float32),
                wp.from_torch(self.cam_quaternion, dtype=wp.float32),
                warp_data,
                self.config.depth_minimum_distance,
                self.config.depth_maximum_distance,
                self.config.minimum_tsdf_weight,
                wp.from_torch(self._hit_points[:n_pixels], dtype=wp.float32),
                wp.from_torch(self._hit_normals[:n_pixels], dtype=wp.float32),
                wp.from_torch(self._hit_colors[:n_pixels], dtype=wp.uint8),
                wp.from_torch(self._hit_depths[:n_pixels], dtype=wp.float32),
                wp.from_torch(self._hit_mask[:n_pixels], dtype=wp.uint8),
                H,
                W,
            ],
        )

        # Kernel clears hit_* to zero on miss, so output buffers are
        # authoritative and no additional masking is required here.
        depth_image = self._hit_depths[:n_pixels].view(H, W)
        normal_image = self._hit_normals[:n_pixels].view(H, W, 3)
        color_image = self._hit_colors[:n_pixels].view(H, W, 3)
        valid_mask = self._hit_mask[:n_pixels].view(H, W).bool()

        return depth_image, normal_image, color_image, valid_mask

    def render_depth(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render only depth image (convenience method).

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.

        Returns:
            depth_image: [H, W] rendered depth in meters.
        """
        depth, _, _ = self.render(intrinsics, pose, image_shape)
        return depth

    def render_normals(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render only normal image (convenience method).

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.

        Returns:
            normal_image: [H, W, 3] surface normals.
        """
        _, normals, _ = self.render(intrinsics, pose, image_shape)
        return normals

    def render_color_only(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render only color image (convenience method).

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.

        Returns:
            color_image: [H, W, 3] uint8 RGB colors.
        """
        _, _, color, _ = self.render_color(intrinsics, pose, image_shape)
        # Kernel already writes 0 for invalid pixels (see render_color).
        return color

    def render_depth_colormap(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render depth as colormap for visualization.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.

        Returns:
            color_image: [H, W, 3] uint8 RGB colormap.
        """
        depth, _, valid = self.render(intrinsics, pose, image_shape)
        return depth_to_colormap(depth, self.config.depth_minimum_distance, self.config.depth_maximum_distance, valid)

    def render_normal_colormap(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Render normals as colormap for visualization.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.

        Returns:
            color_image: [H, W, 3] uint8 RGB colormap.
        """
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        return normals_to_colormap(normals, valid)

    def render_shaded(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
        light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        ambient: float = 0.3,
        use_color: bool = True,
    ) -> torch.Tensor:
        """Render Lambertian-shaded image.

        Args:
            intrinsics: Camera intrinsics.
            pose: Camera-to-world transform as Pose.
            image_shape: (H, W) output dimensions.
            light_direction: Light direction in camera frame.
            ambient: Ambient lighting factor (0-1).
            use_color: If True, use TSDF color. If False, use gray.

        Returns:
            shaded_image: [H, W, 3] uint8 RGB shaded image.
        """
        if use_color:
            _, normals, colors, valid = self.render_color(intrinsics, pose, image_shape)
            base_color = colors.float() / 255.0
        else:
            _, normals, valid = self.render(intrinsics, pose, image_shape)
            base_color = torch.ones(*normals.shape[:-1], 3, device=self.device)

        # Transform light to world frame and normalize
        R = pose.get_rotation_matrix().squeeze().to(self.device, dtype=torch.float32)
        light_cam = torch.tensor(light_direction, device=self.device, dtype=torch.float32)
        light_world = R @ light_cam
        light_world = light_world / light_world.norm()

        # Lambertian shading: max(0, n · l)
        n_dot_l = (normals * light_world.view(1, 1, 3)).sum(dim=-1)
        shading = torch.clamp(n_dot_l, 0, 1)

        # Apply ambient + diffuse
        intensity = ambient + (1.0 - ambient) * shading
        intensity = intensity.unsqueeze(-1)  # [H, W, 1]

        # Apply to base color
        shaded = base_color * intensity
        shaded = torch.clamp(shaded * 255, 0, 255).to(torch.uint8)

        # Set invalid pixels to black
        shaded[~valid] = 0

        return shaded


# =============================================================================
# Visualization Utilities
# =============================================================================


def depth_to_colormap(
    depth: torch.Tensor,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 5.0,
    valid_mask: Optional[torch.Tensor] = None,
    invalid_color: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Convert depth image to RGB colormap (turbo colormap).

    Args:
        depth: [H, W] depth image in meters.
        depth_minimum_distance: Minimum depth for colormap.
        depth_maximum_distance: Maximum depth for colormap.
        valid_mask: [H, W] boolean mask of valid pixels.
        invalid_color: RGB color for invalid pixels.

    Returns:
        color_image: [H, W, 3] uint8 RGB image.
    """
    if valid_mask is None:
        valid_mask = depth > 0

    # Normalize depth to [0, 1]
    depth_normalized = (depth - depth_minimum_distance) / (depth_maximum_distance - depth_minimum_distance)
    depth_normalized = torch.clamp(depth_normalized, 0, 1)

    t = depth_normalized

    # Turbo colormap approximation
    r = torch.clamp(
        0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (
            132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
        0, 1
    )
    g = torch.clamp(
        0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (
            -14.18503333 + t * (4.27729857 + t * 2.82956604)))),
        0, 1
    )
    b = torch.clamp(
        0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (
            110.36276771 + t * (-89.90310912 + t * 27.34824973)))),
        0, 1
    )

    color = torch.stack([r, g, b], dim=-1)
    color = (color * 255).to(torch.uint8)

    color[~valid_mask] = torch.tensor(invalid_color, dtype=torch.uint8, device=depth.device)

    return color


def normals_to_colormap(
    normals: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert normal image to RGB colormap.

    Maps normal directions to colors: X->R, Y->G, Z->B.

    Args:
        normals: [H, W, 3] normal vectors.
        valid_mask: [H, W] boolean mask.

    Returns:
        color_image: [H, W, 3] uint8 RGB image.
    """
    # Map [-1, 1] to [0, 255]
    color = ((normals + 1.0) * 0.5 * 255).to(torch.uint8)
    color[~valid_mask] = 128  # Gray for invalid
    return color

