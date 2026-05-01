# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for BlockSparseTSDFIntegrator."""

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
    return "cuda:0"


# =============================================================================
# Tests
# =============================================================================


class TestBlockSparseTSDFIntegrator:
    """Tests for the high-level integrator interface."""

    def test_initialization(self, warp_init, device):
        """Test integrator initialization."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=100,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        assert integrator.tsdf is not None
        assert integrator.memory_usage_mb() > 0

    def test_visible_capacity_defaults_to_max_blocks(self, warp_init, device):
        """Omitted per-frame visible capacity preserves prior max_blocks behavior."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=123,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        assert config.max_visible_blocks_per_integration == 123
        assert integrator._integrator.max_visible_blocks_per_integration == 123
        assert integrator._integrator.pool_indices.shape == (123,)

    @pytest.mark.parametrize("capacity", [0, -1, 101])
    def test_visible_capacity_validation(self, warp_init, device, capacity):
        """Visible capacity must be positive and no larger than max_blocks."""
        with pytest.raises(ValueError, match="max_visible_blocks_per_integration"):
            BlockSparseTSDFIntegratorCfg(
                voxel_size=0.01,
                origin=torch.tensor([0.0, 0.0, 0.0]),
                grid_shape=(512, 512, 512),
                max_blocks=100,
                max_visible_blocks_per_integration=capacity,
                device=device,
                image_height=32,
                image_width=32,
            )

    def test_support_capacity_validation(self, warp_init, device):
        """Support capacity must be a positive construction-time value."""
        with pytest.raises(ValueError, match="max_support_pixels_per_block_camera"):
            BlockSparseTSDFIntegratorCfg(
                voxel_size=0.01,
                origin=torch.tensor([0.0, 0.0, 0.0]),
                grid_shape=(512, 512, 512),
                max_blocks=100,
                max_support_pixels_per_block_camera=0,
                device=device,
                image_height=32,
                image_width=32,
            )

    def test_visible_capacity_overflow_raises_loudly(self, warp_init, device):
        """Frames that discover more than C visible blocks must not truncate."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            max_visible_blocks_per_integration=1,
            feature_dim=3,
            feature_grid_height=1,
            feature_grid_width=1,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        integrator.tsdf.data.block_data.fill_(7.0)
        integrator.tsdf.data.block_rgb.fill_(5.0)
        integrator.tsdf.data.block_features.fill_(3.0)
        integrator.tsdf.data.block_feature_weight.fill_(2.0)

        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        with pytest.raises(
            ValueError,
            match=(
                "num_visible_blocks=.* exceeds "
                "max_visible_blocks_per_integration=1"
            ),
        ):
            integrator.integrate(
                make_observation(depth, rgb, position, quaternion, intrinsics)
            )

        n_allocated = int(integrator.tsdf.data.num_allocated.item())
        assert n_allocated > 0
        assert torch.count_nonzero(integrator.tsdf.data.block_data[:n_allocated]).item() == 0
        assert torch.count_nonzero(integrator.tsdf.data.block_rgb[:n_allocated]).item() == 0
        assert (
            torch.count_nonzero(integrator.tsdf.data.block_features[:n_allocated]).item()
            == 0
        )
        assert (
            torch.count_nonzero(
                integrator.tsdf.data.block_feature_weight[:n_allocated]
            ).item()
            == 0
        )

    def test_integrate(self, warp_init, device):
        """Test depth integration via integrate method."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Create test data
        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Integrate
        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        stats = integrator.get_stats()
        assert stats["num_allocated"] > 0
        assert stats["frame_count"] == 1

    def test_extract_mesh(self, warp_init, device):
        """Test mesh extraction."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Integrate multiple frames
        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 200, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        # Extract mesh
        mesh = integrator.extract_mesh()

        assert mesh.vertices is not None
        assert mesh.faces is not None

    def test_extract_mesh_tensors(self, warp_init, device):
        """Test raw tensor mesh extraction."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        vertices, triangles, normals, colors = integrator.extract_mesh_tensors()

        assert vertices.dtype == torch.float32
        assert triangles.dtype == torch.int32
        assert normals.dtype == torch.float32
        assert colors.dtype == torch.uint8

    def test_reset(self, warp_init, device):
        """Test integrator reset."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=100,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 16.0], [0.0, 500.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        assert integrator.get_stats()["num_allocated"] > 0

        integrator.reset()

        assert integrator.get_stats()["num_allocated"] == 0
        assert integrator.get_stats()["frame_count"] == 0

    def test_decay_recycle_functions(self, warp_init, device):
        """Test decay and recycle launch paths."""
        from curobo._src.perception.mapper.kernel.wp_decay import (
            decay_and_recycle,
            launch_recycle,
        )

        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=100,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 0.5, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 16.0], [0.0, 500.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Integrate a few frames
        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        initial_blocks = integrator.get_stats()["num_allocated"]
        assert initial_blocks > 0

        # Test decay_and_recycle
        recycled = decay_and_recycle(integrator._tsdf, 0.5)
        assert recycled >= 0

        # Test recycle-only launch path.
        launch_recycle(integrator._tsdf)

        # Heavily decay to trigger recycling
        for _ in range(20):
            decay_and_recycle(integrator._tsdf, 0.5)

        # Test decay_and_recycle (with aggressive factor)
        recycled = decay_and_recycle(integrator._tsdf, 0.1)
        assert recycled >= 0  # May or may not recycle depending on thresholds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
