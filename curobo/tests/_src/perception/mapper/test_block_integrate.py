# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for block-sparse TSDF integration."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
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
    """Get CUDA device."""
    return "cuda:0"


@pytest.fixture
def simple_intrinsics(device):
    """Simple camera intrinsics for testing."""
    # 640x480 camera with 500px focal length
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return K


@pytest.fixture
def identity_pose(device):
    """Identity camera pose (looking down +Z)."""
    position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    # wxyz quaternion for identity rotation
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    return position, quaternion


# =============================================================================
# Tests
# =============================================================================


class TestIntegration:
    """Tests for depth integration."""

    def test_integrate_flat_depth(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test integrating a flat depth plane."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.01,  # 1cm voxels
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,  # 5cm truncation
            device=device,
            image_height=48,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Create a flat depth image at 1.0m
        img_H, img_W = 48, 64  # Small image for fast tests
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)

        position, quaternion = identity_pose

        # Initial state
        assert tsdf.data.num_allocated.item() == 0

        # Integrate
        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        # Check that blocks were allocated
        num_allocated = tsdf.data.num_allocated.item()
        assert num_allocated > 0, "Should have allocated blocks"

        # Check stats
        stats = tsdf.get_stats()
        assert stats["num_allocated"] == num_allocated

    def test_integrate_multiple_frames(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test integrating multiple depth frames."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=device,
            image_height=48,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 48, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        # Integrate 3 frames
        obs = make_observation(depth, rgb, position, quaternion, simple_intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        # Same blocks should be updated, not new ones allocated
        num_allocated = tsdf.data.num_allocated.item()
        assert num_allocated > 0, "Should have allocated blocks"

    def test_block_rgb_matches_constant_input_color(
        self, warp_init, device, simple_intrinsics, identity_pose
    ):
        """Per-block RGB averages should preserve a uniform input color."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=device,
            image_height=48,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 48, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.empty((img_H, img_W, 3), dtype=torch.uint8, device=device)
        rgb[..., 0] = 64
        rgb[..., 1] = 128
        rgb[..., 2] = 192

        position, quaternion = identity_pose
        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        n = tsdf.data.num_allocated.item()
        assert n > 0

        block_rgb = tsdf.data.block_rgb[:n].float()
        active = block_rgb[:, 3] > 0
        assert active.any(), "Expected at least one block with RGB observations"

        normalized = block_rgb[active, :3] / block_rgb[active, 3:4]
        expected = torch.tensor(
            [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0],
            dtype=torch.float32,
            device=device,
        ).view(1, 3)
        torch.testing.assert_close(
            normalized,
            expected.expand_as(normalized),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_integrate_sphere_depth(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test integrating a spherical depth pattern."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=5000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=device,
            image_height=100,
            image_width=100,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Create spherical depth pattern
        img_H, img_W = 100, 100
        center_depth = 1.5
        sphere_radius = 0.3

        # Generate depth for a sphere at center_depth with radius sphere_radius
        u = torch.arange(img_W, device=device, dtype=torch.float32)
        v = torch.arange(img_H, device=device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        # Pixel coordinates relative to center
        cx, cy = img_W / 2, img_H / 2
        fx, fy = 500.0, 500.0  # From simple_intrinsics

        # Normalized coordinates
        x_norm = (uu - cx) / fx
        y_norm = (vv - cy) / fy

        # Distance from center in normalized space
        r_sq = x_norm**2 + y_norm**2

        # Depth for sphere: z = center_depth - sqrt(radius² - (x²+y²)*z²)
        # Simplified: depth varies with r
        depth = center_depth - sphere_radius * (1 - r_sq / (r_sq + 1))
        depth = depth.T  # (H, W)

        rgb = torch.randint(0, 256, (img_H, img_W, 3), dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        stats = tsdf.get_stats()
        assert stats["num_allocated"] > 0

    def test_no_allocation_for_out_of_range(
        self, warp_init, device, simple_intrinsics, identity_pose
    ):
        """Test that no blocks are allocated for depth outside range."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=100,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            depth_maximum_distance=5.0,
            device=device,
            image_height=48,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 48, 64
        # Depth outside valid range (too far)
        depth = torch.full((img_H, img_W), 100.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        assert tsdf.data.num_allocated.item() == 0, "No blocks should be allocated"

    def test_moved_camera(self, warp_init, device, simple_intrinsics):
        """Test integration with camera at different positions."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=2000,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=device,
            image_height=48,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 48, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)

        # Camera looking down +Z from position (0.5, 0, 0)
        position1 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        integrator.integrate(
            make_observation(depth, rgb, position1, quaternion1, simple_intrinsics)
        )

        blocks_after_first = tsdf.data.num_allocated.item()

        # Camera from different position
        position2 = torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion2 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        integrator.integrate(
            make_observation(depth, rgb, position2, quaternion2, simple_intrinsics)
        )

        blocks_after_second = tsdf.data.num_allocated.item()

        # Should have more blocks from second view (different blocks visible)
        assert blocks_after_first > 0
        # Blocks should be accumulated (some may overlap)


class TestBlockAllocation:
    """Tests specifically for block allocation behavior."""

    def test_block_reuse(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test that same blocks are reused across frames."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=100,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        # First frame
        obs = make_observation(depth, rgb, position, quaternion, simple_intrinsics)
        integrator.integrate(obs)

        blocks_first = tsdf.data.num_allocated.item()

        # Second frame with same view
        integrator.integrate(obs)

        blocks_second = tsdf.data.num_allocated.item()

        # Same blocks should be used
        assert blocks_first == blocks_second, "Same blocks should be reused"

    def test_allocation_failure_counter(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test that allocation failures are counted when pool is exhausted."""
        # Very small pool that will get exhausted
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=5,  # Very limited
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.1,  # Larger truncation = more blocks needed
            device=device,
            image_height=200,
            image_width=200,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Large image that will need many blocks
        img_H, img_W = 200, 200
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        stats = tsdf.get_stats()
        # Pool should be exhausted
        assert stats["num_allocated"] == 5
        # Should have recorded failures. The allocation kernels increment
        # ``allocation_failures`` each time a thread hits the pool cap;
        # with a 200x200 image at voxel_size=0.01 this is guaranteed to
        # far exceed max_blocks=5.
        assert stats["allocation_failures"] > 0, (
            f"Expected allocation_failures > 0 after pool exhaustion, "
            f"got {stats['allocation_failures']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
