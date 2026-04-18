# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for BlockSparseTSDFIntegrator."""

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
# Tests
# =============================================================================


class TestBlockSparseTSDFIntegrator:
    """Tests for the high-level integrator interface."""

    def test_initialization(self, warp_init, device):
        """Test integrator initialization."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=100,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        assert integrator.tsdf is not None
        assert integrator.memory_usage_mb() > 0

    def test_integrate(self, warp_init, device):
        """Test depth integration via integrate method."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=500,
            device=device,
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
            max_blocks=500,
            device=device,
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
            max_blocks=500,
            device=device,
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
            max_blocks=100,
            device=device,
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

    def test_decay_graph_safe_functions(self, warp_init, device):
        """Test CUDA graph safe decay functions."""
        from curobo._src.perception.mapper.kernel.wp_decay import (
            decay_and_recycle,
            recycle_graph_safe,
        )

        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            max_blocks=100,
            device=device,
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

        # Test recycle_graph_safe
        recycle_graph_safe(integrator._tsdf)

        # Heavily decay to trigger recycling
        for _ in range(20):
            decay_and_recycle(integrator._tsdf, 0.5)

        # Test decay_and_recycle (with aggressive factor)
        recycled = decay_and_recycle(integrator._tsdf, 0.1)
        assert recycled >= 0  # May or may not recycle depending on thresholds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

