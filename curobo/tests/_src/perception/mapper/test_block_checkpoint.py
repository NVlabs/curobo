# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for mapper block checkpoint helpers."""

import pytest
import torch
import warp as wp

from curobo._src.perception.mapper.checkpoint_blocks import (
    BLOCK_CHECKPOINT_SCHEMA_VERSION,
    build_block_metadata,
    load_block_checkpoint,
    prepare_blocks_for_import,
    rebuild_import_hash_state,
    save_block_checkpoint,
    validate_import_block_coords_for_grid,
    validate_import_block_coords_unique,
    validate_block_metadata,
    validate_block_payload,
)
from curobo._src.perception.mapper.constants import (
    PY_HASH_EMPTY,
    PY_HASH_TOMBSTONE,
    PY_VALUE_MASK,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    make_block_sparse_kernels,
)
from curobo._src.util.warp import init_warp


cuda_lookup_kernels = make_block_sparse_kernels(block_size=2)
cuda_hash_lookup = cuda_lookup_kernels.hash_lookup


@wp.kernel
def lookup_imported_blocks_kernel(
    hash_table: wp.array(dtype=wp.int64),
    coords: wp.array2d(dtype=wp.int32),
    capacity: wp.int32,
    results: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    results[tid] = cuda_hash_lookup(
        hash_table,
        coords[tid, 0],
        coords[tid, 1],
        coords[tid, 2],
        capacity,
    )


def make_metadata(
    *,
    block_size: int = 2,
    has_dynamic: bool = True,
    has_static: bool = False,
    feature_dim: int = 0,
):
    return {
        "voxel_size": 0.01,
        "block_size": block_size,
        "truncation_distance": 0.04,
        "grid_center": [0.0, 0.0, 0.0],
        "grid_shape": [16, 16, 16],
        "has_dynamic": has_dynamic,
        "has_static": has_static,
        "feature_dim": feature_dim,
    }


def make_dynamic_blocks(
    *,
    n_blocks: int = 2,
    block_size: int = 2,
    feature_dim: int = 0,
):
    block_voxels = block_size**3
    coords = torch.arange(n_blocks * 3, dtype=torch.int32).reshape(n_blocks, 3)
    block_data = torch.zeros((n_blocks, block_voxels, 2), dtype=torch.float16)
    block_data[..., 0] = 0.5
    block_data[..., 1] = 1.0
    block_rgb = torch.zeros((n_blocks, 4), dtype=torch.float16)
    block_rgb[:, :3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    block_rgb[:, 3] = 1.0
    blocks = {
        "active_block_coords": coords,
        "block_data": block_data,
        "block_rgb": block_rgb,
    }
    if feature_dim > 0:
        blocks["block_features"] = torch.ones(
            (n_blocks, feature_dim),
            dtype=torch.float16,
        )
        blocks["block_feature_weight"] = torch.ones(n_blocks, dtype=torch.float16)
    return blocks


def test_save_and_load_block_checkpoint_roundtrip(tmp_path):
    metadata = make_metadata()
    blocks = make_dynamic_blocks()
    path = tmp_path / "blocks.pt"

    save_block_checkpoint(path, metadata, blocks)
    loaded = load_block_checkpoint(path)

    assert loaded["format"] == "curobo.mapper_blocks"
    assert loaded["schema_version"] == BLOCK_CHECKPOINT_SCHEMA_VERSION
    assert loaded["block_metadata"] == metadata
    assert torch.equal(
        loaded["blocks"]["active_block_coords"],
        blocks["active_block_coords"].cpu(),
    )
    assert torch.equal(loaded["blocks"]["block_data"], blocks["block_data"].cpu())


def test_validate_block_payload_rejects_wrong_block_data_dtype():
    metadata = make_metadata()
    blocks = make_dynamic_blocks()
    blocks["block_data"] = blocks["block_data"].float()

    with pytest.raises(ValueError, match="block_data.*dtype"):
        validate_block_payload(blocks, metadata)


def test_validate_block_metadata_rejects_extra_fields():
    metadata = make_metadata()
    metadata["unexpected"] = True

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_block_metadata(metadata)


def test_validate_block_payload_rejects_extra_fields():
    metadata = make_metadata()
    blocks = make_dynamic_blocks()
    blocks["unexpected"] = torch.zeros(1)

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_block_payload(blocks, metadata)


def test_prepare_blocks_for_import_applies_constant_weight():
    metadata = make_metadata(feature_dim=2)
    blocks = make_dynamic_blocks(n_blocks=1, feature_dim=2)
    blocks["block_data"].zero_()
    blocks["block_data"][0, 0, 0] = 1.0
    blocks["block_data"][0, 0, 1] = 2.0
    blocks["block_rgb"][0] = torch.tensor([2.0, 4.0, 6.0, 2.0], dtype=torch.float16)
    blocks["block_features"][0] = torch.tensor([2.0, 6.0], dtype=torch.float16)
    blocks["block_feature_weight"][0] = 2.0

    prepared = prepare_blocks_for_import(
        blocks,
        metadata,
        import_weight=4.0,
        minimum_tsdf_weight=0.1,
        block_empty_threshold=0.01,
    )

    assert prepared["block_data"][0, 0, 0].item() == pytest.approx(2.0)
    assert prepared["block_data"][0, 0, 1].item() == pytest.approx(4.0)
    assert prepared["block_data"][0, 1, 0].item() == pytest.approx(0.0)
    assert prepared["block_data"][0, 1, 1].item() == pytest.approx(0.0)
    assert prepared["block_rgb"][0, 0].item() == pytest.approx(4.0)
    assert prepared["block_rgb"][0, 1].item() == pytest.approx(8.0)
    assert prepared["block_rgb"][0, 2].item() == pytest.approx(12.0)
    assert prepared["block_rgb"][0, 3].item() == pytest.approx(4.0)
    assert prepared["block_features"][0, 0].item() == pytest.approx(4.0)
    assert prepared["block_features"][0, 1].item() == pytest.approx(12.0)
    assert prepared["block_feature_weight"][0].item() == pytest.approx(4.0)


def test_prepare_blocks_for_import_rejects_low_constant_weight():
    metadata = make_metadata()
    blocks = make_dynamic_blocks()

    with pytest.raises(ValueError, match="minimum_tsdf_weight"):
        prepare_blocks_for_import(
            blocks,
            metadata,
            import_weight=0.05,
            minimum_tsdf_weight=0.1,
            block_empty_threshold=0.01,
        )


def test_rebuild_import_hash_state_sets_pool_slots():
    coords = torch.tensor([[-1, 0, 1], [0, 0, 0], [2, -2, 3]], dtype=torch.int64)

    hash_table, block_to_hash_slot = rebuild_import_hash_state(
        coords,
        hash_capacity=32,
        max_blocks=8,
    )

    for pool_idx in range(coords.shape[0]):
        slot = int(block_to_hash_slot[pool_idx].item())
        assert slot >= 0
        entry = int(hash_table[slot].item())
        assert entry != PY_HASH_EMPTY
        assert (entry & PY_VALUE_MASK) == pool_idx
    assert torch.all(block_to_hash_slot[coords.shape[0] :] == PY_HASH_EMPTY)


def test_validate_import_block_coords_unique_rejects_duplicates():
    coords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.int64)

    with pytest.raises(ValueError, match="duplicate"):
        validate_import_block_coords_unique(coords)


def test_validate_import_block_coords_for_grid_rejects_outside_target_grid():
    coords = torch.tensor([[1, 0, 0]], dtype=torch.int64)

    with pytest.raises(ValueError, match="outside target grid"):
        validate_import_block_coords_for_grid(
            coords,
            grid_shape=(8, 8, 8),
            block_size=4,
        )


def make_storage(
    max_blocks: int = 4,
    hash_capacity: int = 8,
    device: str = "cpu",
    feature_dim: int = 0,
    enable_static: bool = False,
) -> BlockSparseTSDF:
    cfg = BlockSparseTSDFCfg(
        max_blocks=max_blocks,
        hash_capacity=hash_capacity,
        voxel_size=0.01,
        origin=torch.zeros(3),
        truncation_distance=0.04,
        device=device,
        grid_shape=(8, 8, 8),
        block_size=2,
        feature_dim=feature_dim,
        feature_grid_height=1 if feature_dim > 0 else None,
        feature_grid_width=1 if feature_dim > 0 else None,
        enable_static=enable_static,
    )
    return BlockSparseTSDF(cfg)


def make_static_storage(max_blocks: int = 4, hash_capacity: int = 8) -> BlockSparseTSDF:
    cfg = BlockSparseTSDFCfg(
        max_blocks=max_blocks,
        hash_capacity=hash_capacity,
        voxel_size=0.01,
        origin=torch.zeros(3),
        truncation_distance=0.04,
        device="cpu",
        grid_shape=(8, 8, 8),
        block_size=2,
        enable_dynamic=False,
        enable_static=True,
    )
    return BlockSparseTSDF(cfg)


def test_block_sparse_tsdf_export_import_compacts_active_blocks():
    source = make_storage(max_blocks=4, hash_capacity=8)
    source.data.num_allocated.fill_(3)
    source.data.free_count.fill_(1)
    source.data.free_list[0] = 1
    source.data.block_to_hash_slot[:3] = torch.tensor([0, PY_HASH_EMPTY, 1], dtype=torch.int32)
    source.data.block_coords[:9] = torch.tensor(
        [-1, 0, 0, 0, 0, 0, 0, 1, 0],
        dtype=torch.int32,
    )
    source.data.block_data[0, :, 1] = 1.0
    source.data.block_data[2, :, 1] = 3.0
    source.data.block_rgb[0] = torch.tensor([1.0, 2.0, 3.0, 1.0], dtype=torch.float16)
    source.data.block_rgb[2] = torch.tensor([3.0, 6.0, 9.0, 3.0], dtype=torch.float16)

    blocks = source.export_blocks()
    target = make_storage(max_blocks=6, hash_capacity=16)
    target.import_blocks(blocks)

    assert int(target.data.num_allocated.item()) == 2
    assert int(target.data.free_count.item()) == 0
    assert target.data.block_coords[:6].tolist() == [-1, 0, 0, 0, 1, 0]
    assert target.data.block_sums[:2].tolist() == pytest.approx([8.0, 24.0])
    assert target.data.block_rgb[:2, 3].tolist() == pytest.approx([1.0, 3.0])
    assert torch.all(target.data.block_to_hash_slot[:2] >= 0)
    assert torch.all(target.data.block_to_hash_slot[2:] == PY_HASH_EMPTY)


def test_block_sparse_tsdf_import_recomputes_static_block_sums():
    source = make_static_storage(max_blocks=4, hash_capacity=8)
    source.data.num_allocated.fill_(1)
    source.data.block_to_hash_slot[0] = 0
    source.data.block_coords[:3] = torch.tensor([-1, 0, 0], dtype=torch.int32)
    source.data.static_block_data[0, 0] = -0.01
    source.data.static_block_data[0, 3] = 0.02

    blocks = source.export_blocks()
    target = make_static_storage(max_blocks=6, hash_capacity=16)
    target.import_blocks(blocks)

    assert int(target.data.num_allocated.item()) == 1
    assert target.data.static_block_sums[0].item() == 2
    assert target.data.static_block_data[0, 0].item() == pytest.approx(-0.01, abs=1e-5)
    assert target.data.static_block_data[0, 3].item() == pytest.approx(0.02, abs=1e-5)


def test_block_sparse_tsdf_import_rejects_tombstone_target():
    source = make_storage(max_blocks=4, hash_capacity=8)
    blocks = source.export_blocks()
    target = make_storage(max_blocks=4, hash_capacity=8)
    target.data.hash_table[0] = PY_HASH_TOMBSTONE

    with pytest.raises(ValueError, match="empty target"):
        target.import_blocks(blocks)


def test_cuda_checkpoint_roundtrip_preserves_block_data(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    init_warp()
    source = make_storage(
        max_blocks=4,
        hash_capacity=8,
        device="cuda:0",
        feature_dim=3,
        enable_static=True,
    )
    source.data.num_allocated.fill_(3)
    source.data.free_count.fill_(1)
    source.data.free_list[0] = 1
    source.data.block_to_hash_slot[:3] = torch.tensor(
        [0, PY_HASH_EMPTY, 1],
        dtype=torch.int32,
        device="cuda:0",
    )
    source.data.block_coords[:9] = torch.tensor(
        [-1, 0, 0, 0, 0, 0, 0, 1, 0],
        dtype=torch.int32,
        device="cuda:0",
    )

    active_pool_idx = torch.tensor([0, 2], dtype=torch.long, device="cuda:0")
    expected_coords = source.data.block_coords.view(source.config.max_blocks, 3)[
        active_pool_idx
    ].detach().clone()

    source.data.block_data.zero_()
    source.data.block_data[0, 0, :] = torch.tensor(
        [0.2, 2.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    source.data.block_data[0, 5, :] = torch.tensor(
        [-0.3, 3.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    source.data.block_data[2, 1, :] = torch.tensor(
        [0.4, 4.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    source.data.block_data[2, 7, :] = torch.tensor(
        [-0.5, 5.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    expected_block_data = source.data.block_data[active_pool_idx].detach().clone()

    source.data.block_rgb[0] = torch.tensor(
        [1.0, 2.0, 3.0, 2.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    source.data.block_rgb[2] = torch.tensor(
        [3.0, 6.0, 9.0, 3.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    expected_rgb = source.data.block_rgb[active_pool_idx].detach().clone()

    source.data.block_features[0] = torch.tensor(
        [2.0, 4.0, 6.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    source.data.block_features[2] = torch.tensor(
        [3.0, 6.0, 9.0],
        dtype=torch.float16,
        device="cuda:0",
    )
    source.data.block_feature_weight[0] = 2.0
    source.data.block_feature_weight[2] = 3.0
    expected_features = source.data.block_features[active_pool_idx].detach().clone()
    expected_feature_weight = source.data.block_feature_weight[active_pool_idx].detach().clone()

    source.data.static_block_data[0, 0] = -0.01
    source.data.static_block_data[0, 2] = 0.02
    source.data.static_block_data[2, 1] = -0.03
    expected_static = source.data.static_block_data[active_pool_idx].detach().clone()

    path = tmp_path / "cuda_blocks.pt"
    save_block_checkpoint(
        path,
        block_metadata=build_block_metadata(source),
        blocks=source.export_blocks(),
    )
    checkpoint = load_block_checkpoint(path)

    target = make_storage(
        max_blocks=8,
        hash_capacity=16,
        device="cuda:0",
        feature_dim=3,
        enable_static=True,
    )
    target.import_blocks(checkpoint["blocks"])

    assert int(target.data.num_allocated.item()) == 2
    assert int(target.data.free_count.item()) == 0
    assert torch.equal(
        target.data.block_coords[:6].view(2, 3).detach().cpu(),
        expected_coords.cpu(),
    )
    assert torch.equal(target.data.block_data[:2].detach().cpu(), expected_block_data.cpu())
    assert torch.equal(target.data.block_rgb[:2].detach().cpu(), expected_rgb.cpu())
    assert torch.equal(target.data.block_features[:2].detach().cpu(), expected_features.cpu())
    assert torch.equal(
        target.data.block_feature_weight[:2].detach().cpu(),
        expected_feature_weight.cpu(),
    )
    assert torch.equal(target.data.static_block_data[:2].detach().cpu(), expected_static.cpu())
    assert target.data.block_sums[:2].detach().cpu().tolist() == pytest.approx([5.0, 9.0])
    assert target.data.static_block_sums[:2].detach().cpu().tolist() == [2, 1]

    rgb_avg = target.data.block_rgb[:2, :3].float() / target.data.block_rgb[
        :2,
        3:4,
    ].float()
    torch.testing.assert_close(
        rgb_avg.detach().cpu(),
        torch.tensor([[0.5, 1.0, 1.5], [1.0, 2.0, 3.0]]),
    )
    feature_avg = target.data.block_features[:2].float() / target.data.block_feature_weight[
        :2
    ].float().unsqueeze(-1)
    torch.testing.assert_close(
        feature_avg.detach().cpu(),
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    )

    lookup_results = torch.full((2,), -1, dtype=torch.int32, device="cuda:0")
    wp.launch(
        lookup_imported_blocks_kernel,
        dim=2,
        inputs=[
            wp.from_torch(target.data.hash_table, dtype=wp.int64),
            wp.from_torch(target.data.block_coords[:6].view(2, 3), dtype=wp.int32),
            target.config.hash_capacity,
        ],
        outputs=[wp.from_torch(lookup_results, dtype=wp.int32)],
        device="cuda:0",
    )
    wp.synchronize_device("cuda:0")
    assert lookup_results.detach().cpu().tolist() == [0, 1]
