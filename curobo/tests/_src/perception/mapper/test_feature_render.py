# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deferred rendering of fused TSDF features."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.types.pose import Pose
from curobo.perception import RenderedFeatures, SemanticFilter
from curobo.tests._src.perception.mapper.conftest import make_observation
from curobo.tests._src.perception.mapper.test_semantic_filter import (
    _make_two_surface_integrator,
)


@pytest.fixture
def cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for feature-render kernel tests")
    return "cuda:0"


def _make_pose(device: str, count: int = 1) -> Pose:
    position = torch.zeros((count, 3), dtype=torch.float32, device=device)
    quaternion = torch.zeros((count, 4), dtype=torch.float32, device=device)
    quaternion[:, 0] = 1.0
    return Pose(position=position, quaternion=quaternion)


def _make_intrinsics(device: str) -> torch.Tensor:
    return torch.tensor(
        [[80.0, 0.0, 8.0], [0.0, 80.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _make_integrator(
    device: str,
    *,
    feature_dim: int,
    feature_block_grid_size: int,
) -> BlockSparseTSDFIntegrator:
    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=128,
            voxel_size=0.02,
            origin=torch.zeros(3),
            grid_shape=(128, 128, 128),
            truncation_distance=0.04,
            device=device,
            image_height=16,
            image_width=16,
            feature_dim=feature_dim,
            feature_grid_height=1,
            feature_grid_width=1,
            feature_block_grid_size=feature_block_grid_size,
        )
    )
    depth = torch.ones((16, 16), dtype=torch.float32, device=device)
    rgb = torch.zeros((16, 16, 3), dtype=torch.uint8, device=device)
    observation = make_observation(
        depth,
        rgb,
        torch.zeros(3, dtype=torch.float32, device=device),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        _make_intrinsics(device),
    )
    observation.feature_grid = torch.ones(
        (1, 1, 1, feature_dim),
        dtype=torch.float16,
        device=device,
    )
    integrator.integrate(observation)

    data = integrator.tsdf.data
    active_count = int(data.num_allocated.item())
    node_count = feature_block_grid_size**3
    values = torch.arange(
        active_count * node_count * feature_dim,
        dtype=torch.float32,
        device=device,
    ).view(active_count, node_count, feature_dim)
    values = torch.remainder(values, 97.0) / 97.0 - 0.5
    weights = torch.full(
        (active_count, node_count),
        2.0,
        dtype=torch.float16,
        device=device,
    )
    data.block_features[:active_count].copy_((values * weights.unsqueeze(-1)).half())
    data.block_feature_weight[:active_count].copy_(weights)
    integrator._mark_semantic_data_changed()
    return integrator


def test_feature_render_patch_validation() -> None:
    renderer = BlockSparseTSDFRenderer.__new__(BlockSparseTSDFRenderer)

    assert renderer._normalize_image_patch((8, 10), None) == (0, 0, 8, 10)
    assert renderer._normalize_image_patch((8, 10), (2, 3, 4, 5)) == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="exceeds"):
        renderer._normalize_image_patch((8, 10), (2, 3, 7, 5))
    with pytest.raises(ValueError, match="positive"):
        renderer._normalize_image_patch((8, 10), (2, 3, 0, 5))
    with pytest.raises(TypeError, match="integers"):
        renderer._normalize_image_patch((8, 10), (2, 3, 4.0, 5))


@pytest.mark.parametrize(
    ("feature_dim", "feature_block_grid_size"),
    [(3, 1), (129, 2), (5, 8)],
)
def test_feature_render_gather_matches_storage_for_any_layout(
    cuda_device: str,
    feature_dim: int,
    feature_block_grid_size: int,
) -> None:
    integrator = _make_integrator(
        cuda_device,
        feature_dim=feature_dim,
        feature_block_grid_size=feature_block_grid_size,
    )
    renderer = BlockSparseTSDFRenderer(integrator)
    rendered = renderer.render_features(
        _make_intrinsics(cuda_device),
        _make_pose(cuda_device),
        (16, 16),
    )

    assert isinstance(rendered, RenderedFeatures)
    assert rendered.depth.shape == (16, 16)
    assert rendered.normals.shape == (16, 16, 3)
    assert rendered.valid.shape == (16, 16)
    assert rendered.feature_valid.shape == (16, 16)
    assert rendered.feature_block_idx.dtype == torch.int32
    assert rendered.feature_node_idx.dtype == torch.int16
    assert rendered.feature_valid.any()
    assert torch.all(rendered.feature_block_idx[~rendered.valid] == -1)
    assert torch.all(rendered.feature_node_idx[~rendered.valid] == -1)

    dense = rendered.features()
    valid = rendered.feature_valid
    block_idx = rendered.feature_block_idx[valid].long()
    node_idx = rendered.feature_node_idx[valid].long()
    data = integrator.tsdf.data
    expected = data.block_features[block_idx, node_idx].float()
    expected /= data.block_feature_weight[block_idx, node_idx].float().unsqueeze(-1)
    torch.testing.assert_close(dense[valid], expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.count_nonzero(dense[~valid]) == 0

    channel_start = max(0, feature_dim - 2)
    patch = (4, 5, 6, 7)
    gathered_patch = rendered.features(
        image_patch=patch,
        channels=slice(channel_start, feature_dim),
        dtype=torch.float16,
    )
    torch.testing.assert_close(
        gathered_patch,
        dense[4:10, 5:12, channel_start:feature_dim].half(),
    )
    assert rendered.features(channels=slice(2, 2)).shape == (16, 16, 0)


def test_feature_render_absolute_patch_and_batch_parity(cuda_device: str) -> None:
    integrator = _make_integrator(
        cuda_device,
        feature_dim=7,
        feature_block_grid_size=4,
    )
    renderer = BlockSparseTSDFRenderer(integrator)
    intrinsics = _make_intrinsics(cuda_device)
    pose = _make_pose(cuda_device)
    full = renderer.render_features(intrinsics, pose, (16, 16))
    full_depth = full.depth.clone()
    full_normals = full.normals.clone()
    full_valid = full.valid.clone()
    full_feature_valid = full.feature_valid.clone()
    full_block_idx = full.feature_block_idx.clone()
    full_node_idx = full.feature_node_idx.clone()
    full_features = full.features(channels=slice(1, 6)).clone()

    image_patch = (3, 5, 8, 7)
    patch = renderer.render_features(
        intrinsics,
        pose,
        (16, 16),
        image_patch=image_patch,
    )
    assert patch.image_origin == (3, 5)
    assert patch.full_image_shape == (16, 16)
    torch.testing.assert_close(patch.depth, full_depth[3:11, 5:12])
    torch.testing.assert_close(patch.normals, full_normals[3:11, 5:12])
    torch.testing.assert_close(patch.valid, full_valid[3:11, 5:12])
    torch.testing.assert_close(patch.feature_valid, full_feature_valid[3:11, 5:12])
    torch.testing.assert_close(patch.feature_block_idx, full_block_idx[3:11, 5:12])
    torch.testing.assert_close(patch.feature_node_idx, full_node_idx[3:11, 5:12])
    torch.testing.assert_close(
        patch.features(channels=slice(1, 6)),
        full_features[3:11, 5:12],
    )
    with pytest.raises(ValueError, match="outside the raycast region"):
        patch.features(image_patch=(0, 0, 2, 2))

    batched = renderer.render_features(
        torch.stack((intrinsics, intrinsics)),
        _make_pose(cuda_device, count=2),
        (16, 16),
        image_patch=image_patch,
    )
    assert batched.depth.shape == (2, 8, 7)
    assert batched.feature_block_idx.shape == (2, 8, 7)
    dense_batch = batched.features(channels=slice(0, 3))
    assert dense_batch.shape == (2, 8, 7, 3)
    torch.testing.assert_close(dense_batch[0], dense_batch[1])


def test_feature_render_distinguishes_unknown_features_and_stale_results(
    cuda_device: str,
) -> None:
    integrator = _make_integrator(
        cuda_device,
        feature_dim=3,
        feature_block_grid_size=2,
    )
    renderer = BlockSparseTSDFRenderer(integrator)
    intrinsics = _make_intrinsics(cuda_device)
    pose = _make_pose(cuda_device)
    first = renderer.render_features(intrinsics, pose, (16, 16))
    assert first.features().numel() > 0

    second = renderer.render_features(intrinsics, pose, (16, 16))
    with pytest.raises(RuntimeError, match="later feature render"):
        first.features()

    integrator.tsdf.data.block_feature_weight.zero_()
    integrator._mark_semantic_data_changed()
    with pytest.raises(RuntimeError, match="mapper changed"):
        second.features()

    unknown = renderer.render_features(intrinsics, pose, (16, 16))
    assert unknown.valid.any()
    assert not unknown.feature_valid.any()
    assert torch.count_nonzero(unknown.features()) == 0


def test_semantic_feature_render_returns_rear_surface_feature(cuda_device: str) -> None:
    integrator = _make_two_surface_integrator(cuda_device)
    renderer = BlockSparseTSDFRenderer(integrator)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=cuda_device,
    )
    rendered = renderer.render_features(
        intrinsics,
        _make_pose(cuda_device),
        (1, 1),
        semantic_filter=SemanticFilter(
            target=torch.tensor([1.0, 0.0, 0.0]),
            mode="exclude",
            require_argmax=True,
        ),
    )

    assert rendered.valid.item()
    assert rendered.feature_valid.item()
    assert rendered.depth.item() == pytest.approx(0.905, abs=0.03)
    torch.testing.assert_close(
        rendered.features()[0, 0],
        torch.tensor([0.0, 1.0, 0.0], device=cuda_device),
    )
