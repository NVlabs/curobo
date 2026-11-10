# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Surface-point extraction contracts, including partially observed neighborhoods."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo.perception import SemanticFilter


def make_plane(
    *, partial: bool = False, blocks_xy: int = 2, capacity: int = 64, voxel_size: float = 0.01
) -> BlockSparseTSDFIntegrator:
    """Import a metric plane crossing block boundaries, with two instance owners."""
    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=capacity,
            block_size=8,
            voxel_size=voxel_size,
            grid_shape=(1024, 1024, 1024),
            origin=torch.zeros(3),
            truncation_distance=2 * voxel_size,
            minimum_tsdf_weight=0.1,
            device="cuda:0",
            image_height=1,
            image_width=1,
            feature_dim=2,
            feature_grid_height=1,
            feature_grid_width=1,
            feature_block_grid_size=8,
            instance_grid_size=8,
        )
    )
    coords = torch.tensor(
        [
            (x, y, z)
            for z in (-1, 0)
            for y in range(-blocks_xy // 2, blocks_xy // 2)
            for x in range(-blocks_xy // 2, blocks_xy // 2)
        ],
        dtype=torch.int32,
    )
    local = torch.arange(512, dtype=torch.int32)
    z = (coords[:, 2:3].float() * 8 + (local // 64).float() + 0.5) * voxel_size
    sdf = -z  # z=0 lies between two different sparse blocks.
    data = torch.zeros((len(coords), 512, 2), dtype=torch.float16)
    data[..., 0] = sdf.clamp(-2 * voxel_size, 2 * voxel_size).half()
    data[..., 1] = 1
    if partial:
        known = ((local % 8 == 3) & (local // 8 % 8 == 3))[None]
        data *= known[..., None]
    features = torch.zeros((len(coords), 512, 2), dtype=torch.float16)
    features[coords[:, 0] < 0, :, 0] = 1
    features[coords[:, 0] >= 0, :, 1] = 1
    instances = torch.where(coords[:, 0:1] < 0, 1, 2).expand(-1, 512).short()
    integrator.import_blocks(
        {
            "active_block_coords": coords,
            "block_data": data,
            "block_grid_rgb": torch.ones((len(coords), 1, 4), dtype=torch.float16),
            "block_features": features,
            "block_feature_weight": torch.ones((len(coords), 512), dtype=torch.float16),
            "block_instance_state": instances,
        }
    )
    return integrator


@pytest.fixture(autouse=True)
def require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.parametrize("partial", [False, True])
def test_plane_surface_and_source_indices(partial: bool) -> None:
    mapper = make_plane(partial=partial)
    points = mapper.extract_occupied_points(sample_budget=2000)
    count = points.count.item()
    assert count == 2000
    assert torch.all(torch.abs(points.positions[:count, 2]) < 1e-6)
    assert torch.isfinite(points.positions[:count]).all()
    assert torch.unique(points.positions[:count], dim=0).shape[0] > 1900
    assert points.source_voxels.shape == (2000, 2)
    assert points.source_voxels[:count, 0].min() >= 0
    assert points.source_voxels[:count, 1].max() < 512


def test_instance_semantic_and_voxel_selection() -> None:
    mapper = make_plane()
    left = mapper.extract_occupied_points(sample_budget=300, instance_slot=1)
    assert left.count.item() == 300
    assert (left.positions[:, 0] < 0).all()
    assert (left.instance_state_uint16().int() & 255 == 1).all()
    torch.testing.assert_close(left.features()[:, 0], torch.ones(300, device="cuda"))
    assert (left.colors_uint8() == 255).all()
    target = torch.tensor([0.0, 1.0], device="cuda")
    right = mapper.extract_occupied_points(
        sample_budget=300, semantic_filter=SemanticFilter(target=target, require_argmax=True)
    )
    assert right.count.item() == 300
    assert (right.positions[:, 0] >= 0).all()
    mask = torch.zeros((64, 512), dtype=torch.uint8, device="cuda")
    mask[right.source_voxels[0, 0], :] = 1
    subset = mapper.extract_occupied_points(sample_budget=300, voxel_selection=mask)
    assert subset.count.item() == 300
    assert torch.unique(subset.source_voxels[:, 0]).numel() == 1
    empty = mapper.extract_occupied_points(sample_budget=300, instance_slot=99)
    assert empty.count.item() == 0


def test_cuda_graph_and_no_unknown_boundary_crossings() -> None:
    mapper = make_plane()
    points = mapper.extract_occupied_points(sample_budget=1000)
    expected = points.positions.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mapper.extract_occupied_points(sample_budget=1000)
    graph.replay()
    assert captured.count.item() == 1000
    torch.testing.assert_close(captured.positions, expected, rtol=0, atol=0)
    # Keep only positive SDF observations. Unknown neighbors must not create faces.
    data = mapper.tsdf.data.block_data
    data[..., 1].masked_fill_(data[..., 0] <= 0, 0)
    graph.replay()
    assert captured.count.item() == 0


def test_invalid_point_query() -> None:
    mapper = make_plane()
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            mapper.extract_occupied_points(sample_budget=count)
    with pytest.raises(ValueError):
        mapper.extract_occupied_points(instance_slot=256)
    with pytest.raises(ValueError):
        mapper.extract_occupied_points(voxel_selection=torch.zeros((1, 1), device="cuda"))


def test_semantics_refresh_inside_graph() -> None:
    mapper = make_plane()
    query = SemanticFilter(torch.tensor([1.0, 0.0], device="cuda"), require_argmax=True)
    mapper.extract_occupied_points(sample_budget=300, semantic_filter=query)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = mapper.extract_occupied_points(sample_budget=300, semantic_filter=query)
    graph.replay()
    assert result.count.item() == 300
    mapper.tsdf.data.block_features[..., 0] = 0
    mapper.tsdf.data.block_features[..., 1] = 1
    graph.replay()
    assert result.count.item() == 0


def test_diagonal_surface_pruning_and_seed() -> None:
    """Squares cover all three edge directions and prune off-surface proposals."""
    mapper = make_plane()
    data = mapper.tsdf.data
    blocks = int(data.num_allocated.item())
    coords = data.block_coords.reshape(-1, 3)[:blocks].float()
    local = torch.arange(512, device="cuda")
    xyz = torch.stack((local % 8, local // 8 % 8, local // 64), dim=-1).float()
    centers = (coords[:, None] * 8 + xyz[None] + 0.5) * 0.01
    sdf = centers.sum(dim=-1) / (3.0**0.5)
    data.block_data[:blocks, :, 0] = sdf.clamp(-0.02, 0.02).half()
    result = mapper.extract_occupied_points(sample_budget=3500, seed=3)
    count = result.count.item()
    assert 2000 < count < 3500
    accepted = result.positions[:count].clone()
    # Avoid truncated map boundaries when comparing to the analytic full plane.
    interior = accepted.abs().amax(dim=-1) < 0.07
    residual = accepted[interior].sum(dim=-1).abs() / (3.0**0.5)
    assert (residual < 0.00501).all()
    changed = mapper.extract_occupied_points(sample_budget=3500, seed=4)
    assert not torch.equal(accepted[:100], changed.positions[:100])
