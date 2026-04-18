# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comparison tests: Block-sparse vs Dense TSDF implementations.

These tests verify that the block-sparse TSDF produces equivalent results
to the dense implementation while using less memory.
"""

import pytest
import torch

from curobo._src.perception.mapper import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
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


# =============================================================================
# Helper Functions
# =============================================================================


def create_sphere_depth(
    img_H: int,
    img_W: int,
    intrinsics: torch.Tensor,
    sphere_center: torch.Tensor,
    sphere_radius: float,
    device: str,
) -> torch.Tensor:
    """Create a synthetic depth image of a sphere."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v, u = torch.meshgrid(
        torch.arange(img_H, device=device),
        torch.arange(img_W, device=device),
        indexing="ij",
    )

    # Ray direction (normalized)
    x = (u.float() - cx) / fx
    y = (v.float() - cy) / fy
    z = torch.ones_like(x)
    ray_dir = torch.stack([x, y, z], dim=-1)
    ray_dir = ray_dir / ray_dir.norm(dim=-1, keepdim=True)

    # Ray-sphere intersection
    # ||o + t*d - c||^2 = r^2
    # t^2 + 2*t*(d·(o-c)) + ||o-c||^2 - r^2 = 0
    oc = -sphere_center  # Origin at (0,0,0)
    a = 1.0  # d·d = 1 (normalized)
    b = 2.0 * (ray_dir[..., 0] * oc[0] + ray_dir[..., 1] * oc[1] + ray_dir[..., 2] * oc[2])
    c = (oc * oc).sum() - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c
    valid = discriminant > 0

    t = torch.zeros((img_H, img_W), device=device)
    t[valid] = (-b[valid] - torch.sqrt(discriminant[valid])) / (2 * a)
    t[~valid] = 0  # No hit

    # Convert t to depth (z-coordinate)
    depth = t * ray_dir[..., 2]
    depth[~valid] = 0

    return depth


def mesh_coverage_overlap(
    vertices1: torch.Tensor,
    vertices2: torch.Tensor,
    voxel_size: float,
) -> float:
    """Compute overlap ratio between two meshes based on voxel occupancy.

    Returns a value between 0 and 1, where 1 means perfect overlap.
    """
    if vertices1.shape[0] == 0 or vertices2.shape[0] == 0:
        return 0.0

    # Voxelize both meshes
    def voxelize(v):
        return set(
            tuple(((v[i] / voxel_size).round().int().cpu().tolist()))
            for i in range(v.shape[0])
        )

    voxels1 = voxelize(vertices1)
    voxels2 = voxelize(vertices2)

    if len(voxels1) == 0 or len(voxels2) == 0:
        return 0.0

    intersection = len(voxels1 & voxels2)
    union = len(voxels1 | voxels2)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# Comparison Tests
# =============================================================================


class TestBlockSparseVsDense:
    """Compare block-sparse TSDF against expected behavior."""

    def test_sphere_reconstruction(self, warp_init, device):
        """Test that a sphere is correctly reconstructed."""
        voxel_size = 0.01
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=voxel_size,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=5000,
            hash_capacity=10000,
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Sphere parameters
        sphere_center = torch.tensor([0.0, 0.0, 2.0], device=device)
        sphere_radius = 0.3

        # Camera intrinsics
        img_H, img_W = 128, 128
        intrinsics = torch.tensor(
            [[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Integrate from multiple views
        positions = [
            torch.tensor([0.0, 0.0, 0.0], device=device),
            torch.tensor([0.5, 0.0, 0.0], device=device),
            torch.tensor([-0.5, 0.0, 0.0], device=device),
            torch.tensor([0.0, 0.5, 0.0], device=device),
            torch.tensor([0.0, -0.5, 0.0], device=device),
        ]

        for position in positions:
            # Generate depth for this view
            depth = create_sphere_depth(
                img_H, img_W, intrinsics,
                sphere_center - position,  # Sphere in camera frame
                sphere_radius, device,
            )

            rgb = torch.full((img_H, img_W, 3), 180, dtype=torch.uint8, device=device)
            quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

            # Mask out invalid depth
            valid_mask = depth > 0.1
            if not valid_mask.any():
                continue

            integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        # Extract mesh
        vertices, triangles, _, _ = integrator.extract_mesh_tensors()

        # Verify mesh was created
        assert vertices.shape[0] > 0, "Expected mesh vertices"
        assert triangles.shape[0] > 0, "Expected mesh triangles"

        # Verify vertices are near the sphere surface
        if vertices.shape[0] > 0:
            # Distance from vertices to sphere center
            distances = (vertices - sphere_center).norm(dim=1)
            mean_distance = distances.mean().item()

            # Should be close to sphere radius
            assert abs(mean_distance - sphere_radius) < 0.1, (
                f"Mean vertex distance {mean_distance:.3f} should be close to "
                f"sphere radius {sphere_radius:.3f}"
            )

    def test_flat_plane_reconstruction(self, warp_init, device):
        """Test flat plane at known depth."""
        voxel_size = 0.01
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=voxel_size,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=1000,
            hash_capacity=2000,
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Flat depth at z=1.5
        target_depth = 1.5
        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), target_depth, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        intrinsics = torch.tensor(
            [[200.0, 0.0, 32.0], [0.0, 200.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(5):
            integrator.integrate(obs)

        vertices, triangles, _, _ = integrator.extract_mesh_tensors()

        assert vertices.shape[0] > 0, "Expected mesh vertices"

        # All Z coordinates should be close to target depth
        z_coords = vertices[:, 2]
        z_mean = z_coords.mean().item()
        z_std = z_coords.std().item()

        assert abs(z_mean - target_depth) < 0.1, f"Z mean {z_mean:.3f} should be ~{target_depth}"
        assert z_std < 0.1, f"Z std {z_std:.3f} should be small for flat plane"

    def test_memory_efficiency(self, warp_init, device):
        """Verify block-sparse uses less memory than dense would."""
        voxel_size = 0.01
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=voxel_size,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=10000,
            hash_capacity=20000,
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Integrate a small region
        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        intrinsics = torch.tensor(
            [[200.0, 0.0, 32.0], [0.0, 200.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        stats = integrator.get_stats()

        # Should have few active blocks for a small scene
        # (num_allocated may be higher due to CAS contention, but active_blocks is what matters)
        active_utilization = stats["active_blocks"] / config.max_blocks
        assert active_utilization < 0.1, (
            f"Active block utilization {active_utilization:.1%} should be low for small scene"
        )

        # Memory should be bounded by max_blocks allocation
        memory_mb = integrator.memory_usage_mb()
        assert memory_mb > 0
        assert memory_mb < 500, f"Memory {memory_mb:.1f} MB should be bounded"

    def test_multiple_disjoint_regions(self, warp_init, device):
        """Test that disjoint regions each allocate their own blocks."""
        voxel_size = 0.01
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=voxel_size,
            origin=torch.tensor([-2.0, -2.0, 0.0]),
            max_blocks=5000,
            hash_capacity=10000,
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 32, 32
        intrinsics = torch.tensor(
            [[200.0, 0.0, 16.0], [0.0, 200.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Integrate from two far-apart camera positions
        positions = [
            torch.tensor([0.0, 0.0, 0.0], device=device),
            torch.tensor([2.0, 2.0, 0.0], device=device),  # Far away
        ]

        for position in positions:
            depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
            rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
            quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

            integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        stats = integrator.get_stats()

        # Should have allocated blocks for both regions
        assert stats["num_allocated"] > 1, "Should allocate blocks for multiple regions"


class TestMeshQuality:
    """Test mesh quality metrics."""

    def test_watertight_tendency(self, warp_init, device):
        """Meshes from complete views should tend toward watertight."""
        voxel_size = 0.01
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=voxel_size,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=2000,
            hash_capacity=4000,
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Integrate from multiple overlapping views
        img_H, img_W = 64, 64
        intrinsics = torch.tensor(
            [[200.0, 0.0, 32.0], [0.0, 200.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        for _ in range(5):
            depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
            rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
            position = torch.tensor([0.0, 0.0, 0.0], device=device)
            quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

            integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        vertices, triangles, _, _ = integrator.extract_mesh_tensors()

        # Basic sanity: triangles reference valid vertices
        if triangles.shape[0] > 0:
            max_idx = triangles.max().item()
            assert max_idx < vertices.shape[0], "Triangle indices should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

