# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for block-sparse hash table operations."""

import pytest
import torch
import warp as wp

from curobo._src.perception.mapper import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
)
from curobo._src.perception.mapper.kernel.wp_coord import (
    linear_to_local_coords,
    local_to_linear_index,
    world_to_block_coords,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    clear_new_blocks_kernel,
    find_or_insert_block,
    hash_lookup,
    pack_block_key,
    spatial_hash,
    unpack_block_key,
)
from curobo._src.util.warp import init_warp

# =============================================================================
# Test Kernels
# =============================================================================


@wp.kernel
def _kernel_pack_unpack(
    coords_in: wp.array2d(dtype=wp.int32),
    coords_out: wp.array2d(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
):
    """Test pack/unpack roundtrip."""
    tid = wp.tid()
    bx = coords_in[tid, 0]
    by = coords_in[tid, 1]
    bz = coords_in[tid, 2]

    key = pack_block_key(bx, by, bz)
    keys[tid] = key

    unpacked = unpack_block_key(key)
    coords_out[tid, 0] = unpacked[0]
    coords_out[tid, 1] = unpacked[1]
    coords_out[tid, 2] = unpacked[2]


@wp.kernel
def _kernel_spatial_hash(
    coords: wp.array2d(dtype=wp.int32),
    capacity: wp.int32,
    slots: wp.array(dtype=wp.int32),
):
    """Test spatial hash function."""
    tid = wp.tid()
    bx = coords[tid, 0]
    by = coords[tid, 1]
    bz = coords[tid, 2]
    slots[tid] = spatial_hash(bx, by, bz, capacity)


@wp.kernel
def _kernel_hash_lookup(
    hash_table: wp.array(dtype=wp.int64),
    query_coords: wp.array2d(dtype=wp.int32),
    capacity: wp.int32,
    results: wp.array(dtype=wp.int32),
):
    """Test hash lookup."""
    tid = wp.tid()
    bx = query_coords[tid, 0]
    by = query_coords[tid, 1]
    bz = query_coords[tid, 2]
    results[tid] = hash_lookup(hash_table, bx, by, bz, capacity)


@wp.kernel
def _kernel_find_or_insert(
    hash_table: wp.array(dtype=wp.int64),
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    num_allocated: wp.array(dtype=wp.int32),
    allocation_failures: wp.array(dtype=wp.int32),
    new_blocks: wp.array(dtype=wp.int32),
    new_block_count: wp.array(dtype=wp.int32),
    query_coords: wp.array2d(dtype=wp.int32),
    capacity: wp.int32,
    max_blocks: wp.int32,
    results: wp.array(dtype=wp.int32),
):
    """Test find-or-insert."""
    tid = wp.tid()
    bx = query_coords[tid, 0]
    by = query_coords[tid, 1]
    bz = query_coords[tid, 2]
    results[tid] = find_or_insert_block(
        hash_table,
        block_coords,
        block_to_hash_slot,
        free_list,
        free_count,
        num_allocated,
        allocation_failures,
        new_blocks,
        new_block_count,
        bx,
        by,
        bz,
        capacity,
        max_blocks,
    )


@wp.kernel
def _kernel_world_to_block(
    world_positions: wp.array2d(dtype=wp.float32),
    origin: wp.vec3,
    voxel_size: float,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    block_coords: wp.array2d(dtype=wp.int32),
):
    """Test world to block coordinate conversion."""
    tid = wp.tid()
    pos = wp.vec3(world_positions[tid, 0], world_positions[tid, 1], world_positions[tid, 2])
    coords = world_to_block_coords(pos, origin, voxel_size, block_size, grid_W, grid_H, grid_D)
    block_coords[tid, 0] = coords[0]
    block_coords[tid, 1] = coords[1]
    block_coords[tid, 2] = coords[2]


@wp.kernel
def _kernel_local_index(
    local_coords: wp.array2d(dtype=wp.int32),
    linear_indices: wp.array(dtype=wp.int32),
    coords_out: wp.array2d(dtype=wp.int32),
):
    """Test local coords <-> linear index conversion."""
    tid = wp.tid()
    lx = local_coords[tid, 0]
    ly = local_coords[tid, 1]
    lz = local_coords[tid, 2]

    linear = local_to_linear_index(lx, ly, lz)
    linear_indices[tid] = linear

    recovered = linear_to_local_coords(linear, 8)  # block_size = 8
    coords_out[tid, 0] = recovered[0]
    coords_out[tid, 1] = recovered[1]
    coords_out[tid, 2] = recovered[2]


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
def block_sparse_tsdf(device):
    """Create a BlockSparseTSDF instance for testing."""
    config = BlockSparseTSDFCfg(
        max_blocks=1000,
        hash_capacity=2000,
        voxel_size=0.002,
        origin=torch.tensor([0.0, 0.0, 0.0]),
        device=device,
    )
    return BlockSparseTSDF(config)


# =============================================================================
# Tests
# =============================================================================


class TestBlockKeyEncoding:
    """Tests for pack_block_key and unpack_block_key."""

    def test_roundtrip_positive_coords(self, warp_init, device):
        """Test pack/unpack roundtrip with positive coordinates."""
        coords = torch.tensor(
            [[0, 0, 0], [1, 2, 3], [100, 200, 300], [1000, 2000, 3000]],
            dtype=torch.int32,
            device=device,
        )
        n = coords.shape[0]
        coords_out = torch.zeros_like(coords)
        keys = torch.zeros(n, dtype=torch.int64, device=device)

        wp.launch(
            _kernel_pack_unpack,
            dim=n,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                wp.from_torch(coords_out, dtype=wp.int32),
                wp.from_torch(keys, dtype=wp.int64),
            ],
        )


        assert torch.equal(coords, coords_out), "Pack/unpack roundtrip failed for positive coords"

    def test_roundtrip_negative_coords(self, warp_init, device):
        """Test pack/unpack roundtrip with negative coordinates."""
        # Updated to use ±2048 range (12-bit signed in packed 64-bit format)
        coords = torch.tensor(
            [[-1, -1, -1], [-100, -200, -300], [-2000, -1500, -1000]],
            dtype=torch.int32,
            device=device,
        )
        n = coords.shape[0]
        coords_out = torch.zeros_like(coords)
        keys = torch.zeros(n, dtype=torch.int64, device=device)

        wp.launch(
            _kernel_pack_unpack,
            dim=n,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                wp.from_torch(coords_out, dtype=wp.int32),
                wp.from_torch(keys, dtype=wp.int64),
            ],
        )


        assert torch.equal(coords, coords_out), "Pack/unpack roundtrip failed for negative coords"

    def test_roundtrip_mixed_coords(self, warp_init, device):
        """Test pack/unpack roundtrip with mixed coordinates."""
        # Updated to use ±2048 range (12-bit signed in packed 64-bit format)
        coords = torch.tensor(
            [[-1, 0, 1], [100, -200, 300], [-2000, 2000, 0]],
            dtype=torch.int32,
            device=device,
        )
        n = coords.shape[0]
        coords_out = torch.zeros_like(coords)
        keys = torch.zeros(n, dtype=torch.int64, device=device)

        wp.launch(
            _kernel_pack_unpack,
            dim=n,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                wp.from_torch(coords_out, dtype=wp.int32),
                wp.from_torch(keys, dtype=wp.int64),
            ],
        )


        assert torch.equal(coords, coords_out), "Pack/unpack roundtrip failed for mixed coords"

    def test_unique_keys(self, warp_init, device):
        """Test that different coordinates produce different keys."""
        coords = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=torch.int32,
            device=device,
        )
        n = coords.shape[0]
        coords_out = torch.zeros_like(coords)
        keys = torch.zeros(n, dtype=torch.int64, device=device)

        wp.launch(
            _kernel_pack_unpack,
            dim=n,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                wp.from_torch(coords_out, dtype=wp.int32),
                wp.from_torch(keys, dtype=wp.int64),
            ],
        )


        unique_keys = torch.unique(keys)
        assert len(unique_keys) == n, "Different coords should produce different keys"


class TestSpatialHash:
    """Tests for spatial hash function."""

    def test_hash_in_range(self, warp_init, device):
        """Test that hash values are within capacity."""
        capacity = 1000
        coords = torch.tensor(
            [[0, 0, 0], [1, 2, 3], [-100, 200, -300], [999, 999, 999]],
            dtype=torch.int32,
            device=device,
        )
        n = coords.shape[0]
        slots = torch.zeros(n, dtype=torch.int32, device=device)

        wp.launch(
            _kernel_spatial_hash,
            dim=n,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                capacity,
                wp.from_torch(slots, dtype=wp.int32),
            ],
        )


        assert (slots >= 0).all(), "Hash values should be non-negative"
        assert (slots < capacity).all(), "Hash values should be less than capacity"

    def test_hash_deterministic(self, warp_init, device):
        """Test that hash is deterministic."""
        capacity = 1000
        coords = torch.tensor([[123, 456, 789]], dtype=torch.int32, device=device)
        slots1 = torch.zeros(1, dtype=torch.int32, device=device)
        slots2 = torch.zeros(1, dtype=torch.int32, device=device)

        wp.launch(
            _kernel_spatial_hash,
            dim=1,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                capacity,
                wp.from_torch(slots1, dtype=wp.int32),
            ],
        )
        wp.launch(
            _kernel_spatial_hash,
            dim=1,
            inputs=[
                wp.from_torch(coords, dtype=wp.int32),
                capacity,
                wp.from_torch(slots2, dtype=wp.int32),
            ],
        )


        assert slots1[0] == slots2[0], "Hash should be deterministic"


class TestBlockSparseTSDF:
    """Tests for BlockSparseTSDF class."""

    def test_initialization(self, block_sparse_tsdf):
        """Test that BlockSparseTSDF initializes correctly."""
        tsdf = block_sparse_tsdf
        data = tsdf.data

        # Check hash table initialization
        assert (data.hash_table == -1).all(), "Hash table should be -1 (empty)"

        # Check free list initialization (starts empty - for recycling)
        assert data.free_count.item() == 0, "Free list should start empty"

        # Check counters
        assert data.num_allocated.item() == 0, "No blocks allocated initially"
        assert data.allocation_failures.item() == 0

    def test_reset(self, block_sparse_tsdf):
        """Test reset functionality."""
        tsdf = block_sparse_tsdf

        # Modify some values
        tsdf.data.hash_table[0] = 12345
        tsdf.data.num_allocated.fill_(50)

        # Reset
        tsdf.reset()

        # Verify reset state
        assert (tsdf.data.hash_table == -1).all()
        assert tsdf.data.num_allocated.item() == 0
        assert tsdf.data.free_count.item() == 0  # Free list starts empty

    def test_memory_usage(self, block_sparse_tsdf):
        """Test memory usage calculation."""
        tsdf = block_sparse_tsdf
        mb = tsdf.memory_usage_mb()

        # Should be reasonable for 1000 blocks
        # ~3.5 MB for 1000 blocks (much less than 100K blocks)
        assert 1.0 < mb < 10.0, f"Memory usage {mb} MB seems wrong for 1000 blocks"

    def test_get_stats(self, block_sparse_tsdf):
        """Test statistics reporting."""
        tsdf = block_sparse_tsdf
        stats = tsdf.get_stats()

        assert "num_allocated" in stats
        assert "free_count" in stats
        assert "active_blocks" in stats
        assert "pool_usage_pct" in stats
        assert stats["num_allocated"] == 0
        assert stats["free_count"] == 0  # Free list starts empty

    def test_warp_data_caching(self, block_sparse_tsdf):
        """Test that Warp struct is cached."""
        tsdf = block_sparse_tsdf

        warp1 = tsdf.get_warp_data()
        warp2 = tsdf.get_warp_data()

        assert warp1 is warp2, "Warp struct should be cached"

        tsdf.invalidate_cache()
        warp3 = tsdf.get_warp_data()

        assert warp3 is not warp1, "After invalidate, should get new struct"


class TestFindOrInsert:
    """Tests for find_or_insert_block function."""

    def test_insert_single_block(self, warp_init, device):
        """Test inserting a single block."""
        config = BlockSparseTSDFCfg(
            max_blocks=100,
            hash_capacity=200,
            device=device,
        )
        tsdf = BlockSparseTSDF(config)

        query = torch.tensor([[5, 10, 15]], dtype=torch.int32, device=device)
        results = torch.full((1,), -999, dtype=torch.int32, device=device)

        wp.launch(
            _kernel_find_or_insert,
            dim=1,
            inputs=[
                wp.from_torch(tsdf.data.hash_table, dtype=wp.int64),
                wp.from_torch(tsdf.data.block_coords, dtype=wp.int32),
                wp.from_torch(tsdf.data.block_to_hash_slot, dtype=wp.int32),
                wp.from_torch(tsdf.data.free_list, dtype=wp.int32),
                wp.from_torch(tsdf.data.free_count, dtype=wp.int32),
                wp.from_torch(tsdf.data.num_allocated, dtype=wp.int32),
                wp.from_torch(tsdf.data.allocation_failures, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_blocks, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_block_count, dtype=wp.int32),
                wp.from_torch(query, dtype=wp.int32),
                config.hash_capacity,
                config.max_blocks,
                wp.from_torch(results, dtype=wp.int32),
            ],
        )


        # Should have allocated block 0
        assert results[0].item() == 0, f"Expected pool_idx 0, got {results[0].item()}"
        assert tsdf.data.num_allocated.item() == 1
        assert tsdf.data.new_block_count.item() == 1

    def test_insert_same_block_twice(self, warp_init, device):
        """Test that inserting same block twice returns same pool_idx."""
        config = BlockSparseTSDFCfg(
            max_blocks=100,
            hash_capacity=200,
            device=device,
        )
        tsdf = BlockSparseTSDF(config)

        # Insert same coords twice (sequential)
        query = torch.tensor([[5, 10, 15]], dtype=torch.int32, device=device)
        results1 = torch.full((1,), -999, dtype=torch.int32, device=device)
        results2 = torch.full((1,), -999, dtype=torch.int32, device=device)

        for results in [results1, results2]:
            wp.launch(
                _kernel_find_or_insert,
                dim=1,
                inputs=[
                    wp.from_torch(tsdf.data.hash_table, dtype=wp.int64),
                    wp.from_torch(tsdf.data.block_coords, dtype=wp.int32),
                    wp.from_torch(tsdf.data.block_to_hash_slot, dtype=wp.int32),
                    wp.from_torch(tsdf.data.free_list, dtype=wp.int32),
                    wp.from_torch(tsdf.data.free_count, dtype=wp.int32),
                    wp.from_torch(tsdf.data.num_allocated, dtype=wp.int32),
                    wp.from_torch(tsdf.data.allocation_failures, dtype=wp.int32),
                    wp.from_torch(tsdf.data.new_blocks, dtype=wp.int32),
                    wp.from_torch(tsdf.data.new_block_count, dtype=wp.int32),
                    wp.from_torch(query, dtype=wp.int32),
                    config.hash_capacity,
                    config.max_blocks,
                    wp.from_torch(results, dtype=wp.int32),
                ],
            )


        # Should return same pool_idx
        assert results1[0] == results2[0], "Same block should return same pool_idx"
        # Should only allocate once
        assert tsdf.data.num_allocated.item() == 1

    def test_insert_different_blocks(self, warp_init, device):
        """Test inserting different blocks."""
        config = BlockSparseTSDFCfg(
            max_blocks=100,
            hash_capacity=200,
            device=device,
        )
        tsdf = BlockSparseTSDF(config)

        query = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.int32,
            device=device,
        )
        n = query.shape[0]
        results = torch.full((n,), -999, dtype=torch.int32, device=device)

        wp.launch(
            _kernel_find_or_insert,
            dim=n,
            inputs=[
                wp.from_torch(tsdf.data.hash_table, dtype=wp.int64),
                wp.from_torch(tsdf.data.block_coords, dtype=wp.int32),
                wp.from_torch(tsdf.data.block_to_hash_slot, dtype=wp.int32),
                wp.from_torch(tsdf.data.free_list, dtype=wp.int32),
                wp.from_torch(tsdf.data.free_count, dtype=wp.int32),
                wp.from_torch(tsdf.data.num_allocated, dtype=wp.int32),
                wp.from_torch(tsdf.data.allocation_failures, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_blocks, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_block_count, dtype=wp.int32),
                wp.from_torch(query, dtype=wp.int32),
                config.hash_capacity,
                config.max_blocks,
                wp.from_torch(results, dtype=wp.int32),
            ],
        )


        # All should succeed (no -1)
        assert (results >= 0).all(), "All blocks should be allocated"
        # All should have unique pool indices
        assert len(torch.unique(results)) == n, "Each block should have unique pool_idx"
        # Should allocate n blocks
        assert tsdf.data.num_allocated.item() == n


class TestHashLookup:
    """Tests for hash_lookup function."""

    def test_lookup_nonexistent(self, warp_init, device):
        """Test looking up non-existent block returns -1."""
        config = BlockSparseTSDFCfg(
            max_blocks=100,
            hash_capacity=200,
            device=device,
        )
        tsdf = BlockSparseTSDF(config)

        query = torch.tensor([[5, 10, 15]], dtype=torch.int32, device=device)
        results = torch.full((1,), -999, dtype=torch.int32, device=device)

        wp.launch(
            _kernel_hash_lookup,
            dim=1,
            inputs=[
                wp.from_torch(tsdf.data.hash_table, dtype=wp.int64),
                wp.from_torch(query, dtype=wp.int32),
                config.hash_capacity,
                wp.from_torch(results, dtype=wp.int32),
            ],
        )


        assert results[0].item() == -1, "Non-existent block should return -1"

    def test_lookup_after_insert(self, warp_init, device):
        """Test lookup after insertion."""
        config = BlockSparseTSDFCfg(
            max_blocks=100,
            hash_capacity=200,
            device=device,
        )
        tsdf = BlockSparseTSDF(config)

        query = torch.tensor([[5, 10, 15]], dtype=torch.int32, device=device)
        insert_results = torch.full((1,), -999, dtype=torch.int32, device=device)
        lookup_results = torch.full((1,), -999, dtype=torch.int32, device=device)

        # Insert
        wp.launch(
            _kernel_find_or_insert,
            dim=1,
            inputs=[
                wp.from_torch(tsdf.data.hash_table, dtype=wp.int64),
                wp.from_torch(tsdf.data.block_coords, dtype=wp.int32),
                wp.from_torch(tsdf.data.block_to_hash_slot, dtype=wp.int32),
                wp.from_torch(tsdf.data.free_list, dtype=wp.int32),
                wp.from_torch(tsdf.data.free_count, dtype=wp.int32),
                wp.from_torch(tsdf.data.num_allocated, dtype=wp.int32),
                wp.from_torch(tsdf.data.allocation_failures, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_blocks, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_block_count, dtype=wp.int32),
                wp.from_torch(query, dtype=wp.int32),
                config.hash_capacity,
                config.max_blocks,
                wp.from_torch(insert_results, dtype=wp.int32),
            ],
        )


        # Lookup
        wp.launch(
            _kernel_hash_lookup,
            dim=1,
            inputs=[
                wp.from_torch(tsdf.data.hash_table, dtype=wp.int64),
                wp.from_torch(query, dtype=wp.int32),
                config.hash_capacity,
                wp.from_torch(lookup_results, dtype=wp.int32),
            ],
        )


        assert insert_results[0] == lookup_results[0], "Lookup should return same pool_idx as insert"


class TestCoordinateConversion:
    """Tests for coordinate conversion functions."""

    def test_world_to_block_coords(self, warp_init, device):
        """Test world to block coordinate conversion."""
        voxel_size = 0.002  # 2mm
        block_size = 8
        origin = wp.vec3(0.0, 0.0, 0.0)
        # Grid dimensions (large enough to cover test positions)
        grid_W = 1024
        grid_H = 1024
        grid_D = 1024

        # World positions that should map to specific blocks
        # Block size in world units = 8 * 0.002 = 0.016m
        # With center-origin convention, (0,0,0) world maps to center of grid
        world_pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Center of grid
                [0.015, 0.015, 0.015],  # Slightly offset
                [0.016, 0.0, 0.0],  # One block in +X
                [-0.016, 0.0, 0.0],  # One block in -X
            ],
            dtype=torch.float32,
            device=device,
        )
        n = world_pos.shape[0]
        block_coords = torch.zeros((n, 3), dtype=torch.int32, device=device)

        wp.launch(
            _kernel_world_to_block,
            dim=n,
            inputs=[
                wp.from_torch(world_pos, dtype=wp.float32),
                origin,
                voxel_size,
                block_size,
                grid_W,
                grid_H,
                grid_D,
                wp.from_torch(block_coords, dtype=wp.int32),
            ],
        )

        # With center-origin: grid center is at block (grid_W/block_size/2, ...)
        # For grid 1024 with block_size 8: center block is 64
        center_block = grid_W // block_size // 2
        expected = torch.tensor(
            [
                [center_block, center_block, center_block],  # Center
                [center_block, center_block, center_block],  # Still same block
                [center_block + 1, center_block, center_block],  # +1 in X
                [center_block - 1, center_block, center_block],  # -1 in X
            ],
            dtype=torch.int32,
            device=device,
        )
        assert torch.equal(block_coords, expected), f"Got {block_coords}, expected {expected}"

    def test_local_index_roundtrip(self, warp_init, device):
        """Test local coords <-> linear index roundtrip."""
        # All valid local coords for 8x8x8 block
        coords = []
        for lz in range(8):
            for ly in range(8):
                for lx in range(8):
                    coords.append([lx, ly, lz])
        local_coords = torch.tensor(coords, dtype=torch.int32, device=device)
        n = local_coords.shape[0]

        linear_indices = torch.zeros(n, dtype=torch.int32, device=device)
        coords_out = torch.zeros_like(local_coords)

        wp.launch(
            _kernel_local_index,
            dim=n,
            inputs=[
                wp.from_torch(local_coords, dtype=wp.int32),
                wp.from_torch(linear_indices, dtype=wp.int32),
                wp.from_torch(coords_out, dtype=wp.int32),
            ],
        )


        # All indices should be unique and in [0, 512)
        assert (linear_indices >= 0).all()
        assert (linear_indices < 512).all()
        assert len(torch.unique(linear_indices)) == 512

        # Roundtrip should work
        assert torch.equal(local_coords, coords_out)


class TestClearNewBlocks:
    """Tests for clear_new_blocks_kernel."""

    def test_clear_blocks(self, warp_init, device):
        """Test that new blocks are cleared."""
        config = BlockSparseTSDFCfg(
            max_blocks=100,
            hash_capacity=200,
            device=device,
        )
        tsdf = BlockSparseTSDF(config)

        # Simulate having allocated 2 new blocks
        tsdf.data.new_blocks[0] = 0
        tsdf.data.new_blocks[1] = 5
        tsdf.data.new_block_count.fill_(2)

        # Put garbage in those blocks
        # block_data is (max_blocks, 512, 2) - 3D layout
        tsdf.data.block_data[0, :, :].fill_(1.0)  # Block 0
        tsdf.data.block_data[5, :, :].fill_(2.0)  # Block 5
        # block_rgb is (max_blocks, 4) - per-block weighted sums [R×w, G×w, B×w, W]
        tsdf.data.block_rgb[0, :].fill_(100.0)
        tsdf.data.block_rgb[5, :].fill_(128.0)

        # Run clear kernel
        wp.launch(
            clear_new_blocks_kernel,
            dim=config.max_blocks * 512,
            inputs=[
                wp.from_torch(tsdf.data.block_data, dtype=wp.float16),
                wp.from_torch(tsdf.data.block_rgb, dtype=wp.float32),
                wp.from_torch(tsdf.data.new_blocks, dtype=wp.int32),
                wp.from_torch(tsdf.data.new_block_count, dtype=wp.int32),
                config.max_blocks,
            ],
        )


        # Block 0 should be cleared
        assert (tsdf.data.block_data[0, :, :] == 0).all()
        assert (tsdf.data.block_rgb[0, :] == 0).all()

        # Block 5 should be cleared
        assert (tsdf.data.block_data[5, :, :] == 0).all()
        assert (tsdf.data.block_rgb[5, :] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
