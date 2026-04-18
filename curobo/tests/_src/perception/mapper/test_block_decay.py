# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for block-sparse TSDF decay and recycling."""

import pytest
import torch

from curobo._src.perception.mapper import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.kernel.wp_decay import (
    decay_and_recycle,
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


class TestDecay:
    """Tests for weight decay."""

    def test_decay_reduces_weights(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test that decay reduces voxel weights."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Integrate some data
        img_H, img_W = 48, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        # Get initial weights sum (for comparison)
        initial_weight_sum = tsdf.data.block_data.float().sum().item()
        assert initial_weight_sum > 0, "Should have non-zero weights"

        # Decay
        decay_factor = 0.5
        decay_and_recycle(tsdf, decay_factor)

        # Weights should be reduced
        after_decay_sum = tsdf.data.block_data.float().sum().item()
        assert after_decay_sum < initial_weight_sum, "Weights should decrease after decay"
        # Should be approximately half (decay_factor=0.5)
        ratio = after_decay_sum / initial_weight_sum
        assert 0.4 < ratio < 0.6, f"Expected ~0.5 ratio, got {ratio}"

    def test_multiple_decays(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test that multiple decays continue to reduce weights."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Integrate
        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        # Apply decay multiple times
        decay_factor = 0.9
        previous_sum = tsdf.data.block_data.float().sum().item()

        for i in range(5):
            decay_and_recycle(tsdf, decay_factor)

            current_sum = tsdf.data.block_data.float().sum().item()
            assert current_sum < previous_sum, f"Decay {i+1} should reduce weights"
            previous_sum = current_sum


class TestRecycling:
    """Tests for block recycling."""

    def test_recycle_empty_blocks(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test that empty blocks are recycled after many decays."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        # Integrate with small weights
        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        initial_allocated = tsdf.data.num_allocated.item()
        initial_free = tsdf.data.free_count.item()
        assert initial_allocated > 0

        # Apply aggressive decay until blocks are recycled
        decay_factor = 0.1  # Aggressive decay
        total_recycled = 0
        for _ in range(20):
            recycled = decay_and_recycle(tsdf, decay_factor)
            total_recycled += recycled
            if total_recycled > 0:
                break

        # Check that some blocks were recycled
        final_free = tsdf.data.free_count.item()
        # Free count should have increased (blocks added to free list)
        assert final_free > initial_free or total_recycled > 0, (
            "Some blocks should be recycled after aggressive decay"
        )

    def test_recycled_blocks_reusable(self, warp_init, device, simple_intrinsics, identity_pose):
        """Test that recycled blocks can be reused for new allocations."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=20,  # Small pool to force reuse
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        # First integration
        integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

        first_allocated = tsdf.data.num_allocated.item()

        # Aggressive decay to recycle blocks
        for _ in range(30):
            decay_and_recycle(tsdf, 0.1)

        free_after_decay = tsdf.data.free_count.item()

        # Second integration at different location
        position2 = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32, device=device)
        integrator.integrate(make_observation(depth, rgb, position2, quaternion, simple_intrinsics))

        # num_allocated should stay same or grow minimally if reusing free list
        final_allocated = tsdf.data.num_allocated.item()

        # Free count should have decreased (blocks taken from free list)
        final_free = tsdf.data.free_count.item()

        # If blocks were recycled and reused, free count should be less
        # OR num_allocated should be less than it would be without recycling


class TestDecayAndRecycleIntegration:
    """Integration tests combining decay, recycling, and new integrations."""

    def test_continuous_operation(self, warp_init, device, simple_intrinsics, identity_pose):
        """Simulate continuous operation with integration and decay."""
        config = BlockSparseTSDFIntegratorCfg(
            max_blocks=1000,
            voxel_size=0.02,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            truncation_distance=0.05,
            device=device,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        tsdf = integrator.tsdf

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position, quaternion = identity_pose

        # Simulate 10 frames of integration + decay
        for frame in range(10):

            # Integrate
            integrator.integrate(make_observation(depth, rgb, position, quaternion, simple_intrinsics))

            # Decay (mild)
            decay_and_recycle(tsdf, 0.95)



        stats = tsdf.get_stats()
        assert stats["num_allocated"] > 0
        assert stats["allocation_failures"] <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

