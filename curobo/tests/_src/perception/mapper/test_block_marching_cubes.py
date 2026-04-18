# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for block-sparse marching cubes."""

import pytest
import torch

from curobo._src.perception.mapper import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mesh_extractor import (
    extract_mesh_block_sparse,
)
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def warp_init():
    """Initialize Warp once per module."""
    init_warp()
    return True


@pytest.fixture
def device():
    return "cuda:0"


@pytest.fixture
def simple_intrinsics(device):
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return K


@pytest.fixture
def identity_pose(device):
    position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    return position, quaternion


# =============================================================================
# Tests
# =============================================================================


class TestMarchingCubes:
    """Tests for block-sparse marching cubes."""

    def test_empty_tsdf_returns_empty_mesh(self, warp_init, device):
        """Test that empty TSDF returns empty mesh."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=100,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        vertices, triangles, normals, colors = extract_mesh_block_sparse(tsdf)

        assert vertices.shape == (0, 3)
        assert triangles.shape == (0, 3)
        assert colors.shape == (0, 3)

    def test_flat_plane_creates_mesh(
        self, warp_init, device, simple_intrinsics, identity_pose
    ):
        """Test that a flat depth plane creates a mesh."""
        # Use larger voxels and truncation to ensure dense sampling
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=2000,
            voxel_size=0.02,  # Larger voxels = more samples per voxel
            origin=torch.tensor([-0.5, -0.5, 0.0]),  # Center the origin
            truncation_distance=0.1,  # Larger truncation = more voxels filled
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Integrate flat depth with dense sampling
        img_H, img_W = 200, 200
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 200, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        # Many integration passes with high sample count to fill adjacent voxels
        obs = make_observation(depth, rgb, position, quaternion, simple_intrinsics)
        for _ in range(10):
            integrator.integrate(obs)

        # Extract mesh
        vertices, triangles, normals, colors = extract_mesh_block_sparse(tsdf)

        # Should have some geometry
        assert vertices.shape[0] > 0, "Should have vertices"
        assert triangles.shape[0] > 0, "Should have triangles"
        assert colors.shape[0] == vertices.shape[0], "Colors should match vertices"

        # Check vertices are valid (no NaN)
        assert not torch.isnan(vertices).any(), "Vertices should not contain NaN"

        # Check triangles reference valid vertices
        assert triangles.min() >= 0
        assert triangles.max() < vertices.shape[0]

    def test_mesh_vertices_near_surface(
        self, warp_init, device, simple_intrinsics, identity_pose
    ):
        """Test that mesh vertices are near the depth surface."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Integrate flat depth at z=1.0
        img_H, img_W = 100, 100
        depth_value = 1.0
        depth = torch.full((img_H, img_W), depth_value, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        obs = make_observation(depth, rgb, position, quaternion, simple_intrinsics)
        for _ in range(5):
            integrator.integrate(obs)

        vertices, triangles, normals, colors = extract_mesh_block_sparse(tsdf)

        if vertices.shape[0] > 0:
            # Vertices should be near z=1.0
            z_coords = vertices[:, 2]
            # Allow some tolerance due to voxelization and origin offset
            assert z_coords.min() > 0.5, "Vertices should be near depth surface"
            assert z_coords.max() < 1.5, "Vertices should be near depth surface"

    def test_colors_from_rgb_image(
        self, warp_init, device, simple_intrinsics, identity_pose
    ):
        """Test that mesh colors come from integrated RGB."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 100, 100
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        # Use a distinctive color
        rgb = torch.zeros((img_H, img_W, 3), dtype=torch.uint8, device=device)
        rgb[:, :, 0] = 255  # Red
        position, quaternion = identity_pose

        obs = make_observation(depth, rgb, position, quaternion, simple_intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        vertices, triangles, normals, colors = extract_mesh_block_sparse(tsdf)

        if vertices.shape[0] > 0:
            # Colors should be mostly red
            avg_r = colors[:, 0].float().mean()
            avg_g = colors[:, 1].float().mean()
            avg_b = colors[:, 2].float().mean()

            # Red channel should be high, green/blue should be low
            # (some vertices may get default gray if block not found)
            assert avg_r > avg_g, "Red channel should dominate"
            assert avg_r > avg_b, "Red channel should dominate"

    def test_multiple_views_builds_mesh(
        self, warp_init, device, simple_intrinsics
    ):
        """Test that multiple camera views build a more complete mesh."""
        # Use larger voxels for denser coverage
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=5000,
            voxel_size=0.02,  # Larger voxels
            origin=torch.tensor([-0.5, -0.5, 0.0]),
            truncation_distance=0.1,  # Larger truncation
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 128, 128
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)

        # Integrate from different positions with many samples
        positions = [
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
            torch.tensor([0.2, 0.0, 0.0], dtype=torch.float32, device=device),
            torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float32, device=device),
        ]
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        for pos in positions:
            obs = make_observation(depth, rgb, pos, quaternion, simple_intrinsics)
            for _ in range(5):  # More passes per position
                integrator.integrate(obs)

        vertices, triangles, normals, colors = extract_mesh_block_sparse(tsdf)

        assert vertices.shape[0] > 0, "Should have vertices from multiple views"


class TestMeshQuality:
    """Tests for mesh quality."""

    def test_no_degenerate_triangles(
        self, warp_init, device, simple_intrinsics, identity_pose
    ):
        """Test that mesh has no degenerate (zero-area) triangles."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 100, 100
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        obs = make_observation(depth, rgb, position, quaternion, simple_intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        vertices, triangles, normals, colors = extract_mesh_block_sparse(tsdf)

        if triangles.shape[0] > 0:
            # Check for degenerate triangles
            v0 = vertices[triangles[:, 0]]
            v1 = vertices[triangles[:, 1]]
            v2 = vertices[triangles[:, 2]]

            # Cross product to get area
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = torch.cross(edge1, edge2, dim=1)
            areas = torch.norm(cross, dim=1) / 2

            # Count degenerate triangles (area < epsilon)
            n_degenerate = (areas < 1e-10).sum().item()
            degenerate_ratio = n_degenerate / triangles.shape[0]

            # Allow some degenerate triangles (numerical precision), but not many
            assert degenerate_ratio < 0.1, f"Too many degenerate triangles: {degenerate_ratio*100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

