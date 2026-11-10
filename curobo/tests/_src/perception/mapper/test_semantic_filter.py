# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for semantic filtering of block-sparse TSDF outputs."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.types.pose import Pose
from curobo.perception import SemanticFilter
from curobo.tests._src.perception.mapper.conftest import make_observation


@pytest.fixture
def cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for semantic-filter kernel tests")
    return "cuda:0"


def _make_feature_integrator(
    device: str,
    *,
    feature_dim: int = 3,
    feature_block_grid_size: int = 2,
) -> BlockSparseTSDFIntegrator:
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=64,
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


def _make_feature_esdf_integrator(
    device: str,
    *,
    seeding_method: str,
    use_cuda_graph: bool = False,
) -> BlockSparseESDFIntegrator:
    return BlockSparseESDFIntegrator(
        BlockSparseESDFIntegratorCfg(
            max_blocks=64,
            voxel_size=0.02,
            esdf_voxel_size=0.04,
            esdf_grid_shape=(32, 32, 32),
            origin=torch.zeros(3),
            grid_shape=(128, 128, 128),
            truncation_distance=0.04,
            device=device,
            image_height=16,
            image_width=16,
            feature_dim=3,
            feature_grid_height=1,
            feature_grid_width=1,
            feature_block_grid_size=2,
            seeding_method=seeding_method,
            edt_solver="jfa",
            use_cuda_graph=use_cuda_graph,
        )
    )


def _allocate_test_blocks(integrator: BlockSparseTSDFIntegrator, device: str) -> int:
    depth = torch.ones((16, 16), dtype=torch.float32, device=device)
    rgb = torch.zeros((16, 16, 3), dtype=torch.uint8, device=device)
    position = torch.zeros(3, dtype=torch.float32, device=device)
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    intrinsics = torch.tensor(
        [[80.0, 0.0, 8.0], [0.0, 80.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))
    return int(integrator.tsdf.data.num_allocated.item())


def _make_two_surface_integrator(device: str) -> BlockSparseTSDFIntegrator:
    block_size = 8
    voxel_size = 0.02
    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=64,
            voxel_size=voxel_size,
            origin=torch.zeros(3),
            grid_shape=(128, 128, 128),
            truncation_distance=0.04,
            device=device,
            image_height=1,
            image_width=1,
            feature_dim=3,
            feature_grid_height=1,
            feature_grid_width=1,
            feature_block_grid_size=block_size,
        )
    )
    block_coords = torch.tensor(
        [
            [block_x, block_y, block_z]
            for block_z in range(8)
            for block_y in (-1, 0)
            for block_x in (-1, 0)
        ],
        dtype=torch.int32,
    )
    block_count = block_coords.shape[0]
    block_voxels = block_size**3
    local_z = torch.arange(block_voxels, dtype=torch.float32) // (block_size**2)
    world_z = (
        block_coords[:, 2:3].float() * block_size + local_z[None] + 0.5
    ) * voxel_size
    front_sdf = torch.maximum(0.405 - world_z, world_z - 0.585)
    rear_sdf = torch.maximum(0.905 - world_z, world_z - 1.085)
    sdf = torch.minimum(front_sdf, rear_sdf).clamp(-0.04, 0.04)
    block_data = torch.empty(
        (block_count, block_voxels, 2),
        dtype=torch.float16,
    )
    block_data[..., 0] = sdf.to(torch.float16)
    block_data[..., 1] = 1.0

    block_features = torch.zeros(
        (block_count, block_voxels, 3),
        dtype=torch.float16,
    )
    semantic_class = torch.full((block_count,), 2, dtype=torch.long)
    semantic_class[(block_coords[:, 2] == 2) | (block_coords[:, 2] == 3)] = 0
    semantic_class[(block_coords[:, 2] == 5) | (block_coords[:, 2] == 6)] = 1
    block_features[
        torch.arange(block_count),
        :,
        semantic_class,
    ] = 1.0

    block_grid_rgb = torch.zeros(
        (block_count, 1, 4),
        dtype=torch.float16,
    )
    block_grid_rgb[..., 3] = 1.0
    block_grid_rgb[:, 0, 2] = 64.0
    front_blocks = (block_coords[:, 2] == 2) | (block_coords[:, 2] == 3)
    rear_blocks = (block_coords[:, 2] == 5) | (block_coords[:, 2] == 6)
    block_grid_rgb[front_blocks, 0, :3] = torch.tensor(
        [255.0, 0.0, 0.0], dtype=torch.float16
    )
    block_grid_rgb[rear_blocks, 0, :3] = torch.tensor(
        [0.0, 255.0, 0.0], dtype=torch.float16
    )
    integrator.import_blocks(
        {
            "active_block_coords": block_coords,
            "block_data": block_data,
            "block_grid_rgb": block_grid_rgb,
            "block_features": block_features,
            "block_feature_weight": torch.ones(
                (block_count, block_voxels),
                dtype=torch.float16,
            ),
        }
    )
    return integrator


def test_semantic_filter_requires_matching_criterion() -> None:
    with pytest.raises(ValueError, match="matching criterion"):
        SemanticFilter(target=torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("mode", ["keep", "remove", ""])
def test_semantic_filter_rejects_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        SemanticFilter(
            target=torch.tensor([1.0, 0.0]),
            mode=mode,
            require_argmax=True,
        )


@pytest.mark.parametrize("threshold", [-1.01, 1.01, float("inf"), float("nan")])
def test_semantic_filter_rejects_invalid_cosine_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="min_similarity"):
        SemanticFilter(
            target=torch.tensor([1.0, 0.0]),
            min_similarity=threshold,
        )


def test_semantic_filter_rejects_invalid_target() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        SemanticFilter(target=[1.0, 0.0], require_argmax=True)
    with pytest.raises(ValueError, match="shape"):
        SemanticFilter(target=torch.ones((1, 2)), require_argmax=True)
    with pytest.raises(ValueError, match="non-empty"):
        SemanticFilter(target=torch.empty(0), require_argmax=True)
    with pytest.raises(ValueError, match="finite"):
        SemanticFilter(target=torch.tensor([1.0, float("nan")]), require_argmax=True)


def test_semantic_filter_cosine_requires_nonzero_target() -> None:
    with pytest.raises(ValueError, match="nonzero norm"):
        SemanticFilter(target=torch.zeros(3), min_similarity=0.5)


def test_semantic_filter_accepts_cosine_argmax_and_combined() -> None:
    target = torch.tensor([0.0, 1.0, 0.0])

    cosine = SemanticFilter(target=target, min_similarity=0.8)
    argmax = SemanticFilter(target=target, require_argmax=True)
    combined = SemanticFilter(
        target=target,
        mode="exclude",
        min_similarity=0.8,
        require_argmax=True,
    )

    assert cosine.mode == "only"
    assert cosine.min_similarity == 0.8
    assert not cosine.require_argmax
    assert argmax.min_similarity is None
    assert argmax.require_argmax
    assert combined.mode == "exclude"


def test_prepared_semantic_filter_matches_reference_and_reuses_storage(
    cuda_device: str,
) -> None:
    integrator = _make_feature_integrator(cuda_device)
    num_allocated = _allocate_test_blocks(integrator, cuda_device)
    assert num_allocated > 0

    data = integrator.tsdf.data
    data.block_features[:num_allocated].zero_()
    data.block_feature_weight[:num_allocated].zero_()
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float16,
        device=cuda_device,
    )
    data.block_features[:num_allocated] = vectors.unsqueeze(0)
    data.block_feature_weight[:num_allocated] = 1.0
    data.block_feature_weight[:num_allocated, 7] = 0.0
    integrator._mark_semantic_data_changed()

    target = torch.tensor([1.0, 0.0, 0.0])
    only_filter = SemanticFilter(
        target=target,
        min_similarity=0.95,
        require_argmax=True,
    )
    prepared = integrator._prepare_semantic_filter(only_filter)
    node_match = integrator._semantic_filter_workspace.node_match
    block_has_keep = integrator._semantic_filter_workspace.block_has_keep
    assert node_match is not None
    assert block_has_keep is not None
    expected = torch.tensor(
        [1, 1, 0, 0, 0, 0, 1, 0],
        dtype=torch.uint8,
        device=cuda_device,
    )
    torch.testing.assert_close(
        node_match[:num_allocated],
        expected.expand(num_allocated, -1),
    )
    torch.testing.assert_close(
        block_has_keep[:num_allocated],
        torch.ones(num_allocated, dtype=torch.uint8, device=cuda_device),
    )
    node_pointer = node_match.data_ptr()
    block_pointer = block_has_keep.data_ptr()

    exclude_filter = SemanticFilter(
        target=target,
        mode="exclude",
        min_similarity=0.95,
        require_argmax=True,
    )
    exclude_prepared = integrator._prepare_semantic_filter(exclude_filter)

    assert prepared.node_match.ptr == exclude_prepared.node_match.ptr
    assert node_match.data_ptr() == node_pointer
    assert block_has_keep.data_ptr() == block_pointer
    torch.testing.assert_close(
        block_has_keep[:num_allocated],
        torch.ones(num_allocated, dtype=torch.uint8, device=cuda_device),
    )
    torch.testing.assert_close(
        block_has_keep[num_allocated:],
        torch.zeros(64 - num_allocated, dtype=torch.uint8, device=cuda_device),
    )

    data.block_features[:num_allocated, 0].zero_()
    data.block_features[:num_allocated, 0, 1] = 1.0
    integrator._mark_semantic_data_changed()
    integrator._prepare_semantic_filter(only_filter)

    assert node_match.data_ptr() == node_pointer
    assert block_has_keep.data_ptr() == block_pointer
    torch.testing.assert_close(
        node_match[:num_allocated, 0],
        torch.zeros(num_allocated, dtype=torch.uint8, device=cuda_device),
    )


def test_semantic_filter_channel_cooperative_match_matches_torch_reference(
    cuda_device: str,
) -> None:
    feature_dim = 129
    integrator = _make_feature_integrator(cuda_device, feature_dim=feature_dim)
    num_allocated = _allocate_test_blocks(integrator, cuda_device)
    data = integrator.tsdf.data
    node_count = data.feature_block_grid_size**3
    values = torch.arange(
        num_allocated * node_count * feature_dim,
        dtype=torch.float32,
        device=cuda_device,
    ).reshape(num_allocated, node_count, feature_dim)
    features = torch.sin(values * 0.017).to(torch.float16)
    target = torch.cos(
        torch.arange(feature_dim, dtype=torch.float32) * 0.031
    )
    data.block_features[:num_allocated] = features
    data.block_feature_weight[:num_allocated] = 1.0
    data.block_feature_weight[0, -1] = 0.0
    integrator._mark_semantic_data_changed()

    features_fp32 = features.float()
    target_cuda = target.to(cuda_device)
    similarity = torch.sum(features_fp32 * target_cuda, dim=-1) / (
        torch.linalg.vector_norm(features_fp32, dim=-1)
        * torch.linalg.vector_norm(target_cuda)
    )
    observed = data.block_feature_weight[:num_allocated] > 1e-6
    cosine_match = observed & (similarity >= 0.1)
    argmax_match = observed & (
        torch.argmax(features_fp32, dim=-1) == torch.argmax(target_cuda)
    )

    for semantic_filter, expected in (
        (SemanticFilter(target=target, min_similarity=0.1), cosine_match),
        (SemanticFilter(target=target, require_argmax=True), argmax_match),
        (
            SemanticFilter(
                target=target,
                min_similarity=0.1,
                require_argmax=True,
            ),
            cosine_match & argmax_match,
        ),
    ):
        integrator._prepare_semantic_filter(semantic_filter)
        node_match = integrator._semantic_filter_workspace.node_match
        assert node_match is not None
        torch.testing.assert_close(
            node_match[:num_allocated].bool(),
            expected,
        )


def test_semantic_filter_requires_configured_feature_dimension(cuda_device: str) -> None:
    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=8,
            voxel_size=0.02,
            origin=torch.zeros(3),
            grid_shape=(32, 32, 32),
            device=cuda_device,
            image_height=4,
            image_width=4,
        )
    )
    semantic_filter = SemanticFilter(
        target=torch.tensor([1.0, 0.0]),
        require_argmax=True,
    )
    with pytest.raises(ValueError, match="feature_dim > 0"):
        integrator._prepare_semantic_filter(semantic_filter)


def test_semantic_filter_requires_matching_mapper_dimension(cuda_device: str) -> None:
    integrator = _make_feature_integrator(cuda_device)
    semantic_filter = SemanticFilter(
        target=torch.tensor([1.0, 0.0]),
        require_argmax=True,
    )
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        integrator._prepare_semantic_filter(semantic_filter)


def _voxel_keys(centers: torch.Tensor, pool_indices: torch.Tensor) -> set[tuple]:
    centers_cpu = centers.detach().cpu().tolist()
    pools_cpu = pool_indices.detach().cpu().tolist()
    return {
        (int(pool_idx), *(round(float(value), 6) for value in center))
        for center, pool_idx in zip(centers_cpu, pools_cpu, strict=True)
    }


def _point_multiset(points: torch.Tensor) -> list[tuple[float, ...]]:
    return sorted(
        tuple(round(float(value), 6) for value in point)
        for point in points.detach().cpu().tolist()
    )


@pytest.mark.parametrize("feature_block_grid_size", [1, 2, 4])
def test_semantic_filter_applies_before_voxel_generation_for_any_feature_grid(
    cuda_device: str,
    feature_block_grid_size: int,
) -> None:
    integrator = _make_feature_integrator(
        cuda_device,
        feature_block_grid_size=feature_block_grid_size,
    )
    num_allocated = _allocate_test_blocks(integrator, cuda_device)
    assert num_allocated > 0

    data = integrator.tsdf.data
    node_count = feature_block_grid_size**3
    node_features = torch.zeros(
        (node_count, 3),
        dtype=torch.float16,
        device=cuda_device,
    )
    node_features[0::2, 0] = 1.0
    node_features[1::2, 1] = 1.0
    data.block_features[:num_allocated] = node_features.unsqueeze(0)
    data.block_feature_weight[:num_allocated] = 1.0
    integrator._mark_semantic_data_changed()

    unfiltered = integrator.extract_occupied_voxels(
        surface_only=True,
        sdf_threshold=integrator.config.truncation_distance,
    )
    assert len(unfiltered) > 0
    node_indices = unfiltered.block_data._feature_node_indices_at_centers(
        unfiltered.centers,
        unfiltered.block_idx_per_voxel,
    )
    expected_only = node_indices.remainder(2) == 0

    target = torch.tensor([1.0, 0.0, 0.0])
    only = integrator.extract_occupied_voxels(
        surface_only=True,
        sdf_threshold=integrator.config.truncation_distance,
        semantic_filter=SemanticFilter(target=target, require_argmax=True),
    )
    exclude = integrator.extract_occupied_voxels(
        surface_only=True,
        sdf_threshold=integrator.config.truncation_distance,
        semantic_filter=SemanticFilter(
            target=target,
            mode="exclude",
            require_argmax=True,
        ),
    )

    assert _voxel_keys(
        only.centers,
        only.block_idx_per_voxel,
    ) == _voxel_keys(
        unfiltered.centers[expected_only],
        unfiltered.block_idx_per_voxel[expected_only],
    )
    assert _voxel_keys(
        exclude.centers,
        exclude.block_idx_per_voxel,
    ) == _voxel_keys(
        unfiltered.centers[~expected_only],
        unfiltered.block_idx_per_voxel[~expected_only],
    )
    assert len(only) + len(exclude) == len(unfiltered)


def test_semantic_filter_applies_before_mesh_generation(cuda_device: str) -> None:
    integrator = _make_feature_integrator(cuda_device)
    num_allocated = _allocate_test_blocks(integrator, cuda_device)
    data = integrator.tsdf.data
    data.block_features[:num_allocated].zero_()
    data.block_features[:num_allocated, :, 0] = 1.0
    data.block_feature_weight[:num_allocated] = 1.0
    integrator._mark_semantic_data_changed()

    unfiltered_vertices, _, _, _ = integrator.extract_mesh_tensors()
    assert unfiltered_vertices.shape[0] > 0
    target = torch.tensor([1.0, 0.0, 0.0])
    only_vertices, _, _, _ = integrator.extract_mesh_tensors(
        semantic_filter=SemanticFilter(target=target, require_argmax=True)
    )
    excluded_vertices, excluded_faces, excluded_normals, excluded_colors = (
        integrator.extract_mesh_tensors(
            semantic_filter=SemanticFilter(
                target=target,
                mode="exclude",
                require_argmax=True,
            )
        )
    )

    assert _point_multiset(only_vertices) == _point_multiset(unfiltered_vertices)
    assert excluded_vertices.shape == (0, 3)
    assert excluded_faces.shape == (0, 3)
    assert excluded_normals.shape == (0, 3)
    assert excluded_colors.shape == (0, 3)


def test_semantic_filter_applies_during_depth_and_color_rendering(
    cuda_device: str,
) -> None:
    integrator = _make_feature_integrator(cuda_device)
    num_allocated = _allocate_test_blocks(integrator, cuda_device)
    data = integrator.tsdf.data
    data.block_features[:num_allocated].zero_()
    data.block_features[:num_allocated, :, 0] = 1.0
    data.block_feature_weight[:num_allocated] = 1.0
    integrator._mark_semantic_data_changed()

    renderer = BlockSparseTSDFRenderer(integrator)
    intrinsics = torch.tensor(
        [[80.0, 0.0, 8.0], [0.0, 80.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=cuda_device,
    )
    pose = Pose(
        position=torch.zeros((1, 3), dtype=torch.float32, device=cuda_device),
        quaternion=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=cuda_device,
        ),
    )
    target = torch.tensor([1.0, 0.0, 0.0])
    exclude_filter = SemanticFilter(
        target=target,
        mode="exclude",
        require_argmax=True,
    )

    depth, normals, valid = renderer.render(intrinsics, pose, (16, 16))
    depth = depth.clone()
    normals = normals.clone()
    valid = valid.clone()
    only_depth, only_normals, only_valid = renderer.render(
        intrinsics,
        pose,
        (16, 16),
        semantic_filter=SemanticFilter(target=target, require_argmax=True),
    )
    only_depth = only_depth.clone()
    only_normals = only_normals.clone()
    only_valid = only_valid.clone()
    excluded_depth, excluded_normals, excluded_colors, excluded_valid = (
        renderer.render_color(
            intrinsics,
            pose,
            (16, 16),
            semantic_filter=exclude_filter,
        )
    )
    excluded_depth = excluded_depth.clone()
    excluded_normals = excluded_normals.clone()
    excluded_colors = excluded_colors.clone()
    excluded_valid = excluded_valid.clone()
    batched_pose = Pose(
        position=pose.position.repeat(2, 1),
        quaternion=pose.quaternion.repeat(2, 1),
    )
    batched_depth, batched_normals, batched_valid = renderer.render(
        torch.stack((intrinsics, intrinsics)),
        batched_pose,
        (16, 16),
        semantic_filter=SemanticFilter(target=target, require_argmax=True),
    )

    assert valid.any()
    torch.testing.assert_close(only_depth, depth)
    torch.testing.assert_close(only_normals, normals)
    torch.testing.assert_close(only_valid, valid)
    torch.testing.assert_close(batched_depth, torch.stack((only_depth, only_depth)))
    torch.testing.assert_close(
        batched_normals,
        torch.stack((only_normals, only_normals)),
    )
    torch.testing.assert_close(batched_valid, torch.stack((only_valid, only_valid)))
    assert not excluded_valid.any()
    assert torch.count_nonzero(excluded_depth) == 0
    assert torch.count_nonzero(excluded_normals) == 0
    assert torch.count_nonzero(excluded_colors) == 0
    assert torch.count_nonzero(
        renderer.render_depth(intrinsics, pose, (16, 16), exclude_filter)
    ) == 0
    assert torch.count_nonzero(
        renderer.render_normals(intrinsics, pose, (16, 16), exclude_filter)
    ) == 0
    assert torch.count_nonzero(
        renderer.render_color_only(intrinsics, pose, (16, 16), exclude_filter)
    ) == 0
    assert torch.count_nonzero(
        renderer.render_depth_colormap(intrinsics, pose, (16, 16), exclude_filter)
    ) == 0
    assert torch.all(
        renderer.render_normal_colormap(intrinsics, pose, (16, 16), exclude_filter)
        == 128
    )
    assert torch.count_nonzero(
        renderer.render_shaded(
            intrinsics,
            pose,
            (16, 16),
            semantic_filter=exclude_filter,
        )
    ) == 0


def test_semantic_filter_render_exclusion_reveals_retained_rear_surface(
    cuda_device: str,
) -> None:
    integrator = _make_two_surface_integrator(cuda_device)
    renderer = BlockSparseTSDFRenderer(integrator)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=cuda_device,
    )
    pose = Pose(
        position=torch.zeros((1, 3), dtype=torch.float32, device=cuda_device),
        quaternion=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=cuda_device,
        ),
    )
    front_target = torch.tensor([1.0, 0.0, 0.0])

    front_depth, _, front_color, front_valid = renderer.render_color(
        intrinsics,
        pose,
        (1, 1),
    )
    front_depth = front_depth.clone()
    front_color = front_color.clone()
    only_depth, _, only_color, only_valid = renderer.render_color(
        intrinsics,
        pose,
        (1, 1),
        semantic_filter=SemanticFilter(
            target=front_target,
            require_argmax=True,
        ),
    )
    only_depth = only_depth.clone()
    only_color = only_color.clone()
    rear_depth, _, rear_color, rear_valid = renderer.render_color(
        intrinsics,
        pose,
        (1, 1),
        semantic_filter=SemanticFilter(
            target=front_target,
            mode="exclude",
            require_argmax=True,
        ),
    )

    assert front_valid.item()
    assert only_valid.item()
    assert rear_valid.item()
    assert front_depth.item() == pytest.approx(0.405, abs=0.03)
    assert only_depth.item() == pytest.approx(front_depth.item(), abs=0.01)
    assert rear_depth.item() == pytest.approx(0.905, abs=0.03)
    assert rear_depth.item() > front_depth.item() + 0.4
    assert front_color[0, 0, 0] > front_color[0, 0, 1]
    assert only_color[0, 0, 0] > only_color[0, 0, 1]
    assert rear_color[0, 0, 1] > rear_color[0, 0, 0]


@pytest.mark.parametrize("seeding_method", ["gather", "scatter"])
def test_semantic_filter_applies_before_esdf_generation(
    cuda_device: str,
    seeding_method: str,
) -> None:
    integrator = _make_feature_esdf_integrator(
        cuda_device,
        seeding_method=seeding_method,
    )
    num_allocated = _allocate_test_blocks(integrator._tsdf_integrator, cuda_device)
    data = integrator.tsdf.data
    data.block_features[:num_allocated].zero_()
    data.block_features[:num_allocated, :, 0] = 1.0
    data.block_feature_weight[:num_allocated] = 1.0
    integrator._tsdf_integrator._mark_semantic_data_changed()

    esdf_origin = torch.tensor([0.0, 0.0, 1.0], device=cuda_device)
    unfiltered = integrator.compute_esdf(esdf_origin=esdf_origin).clone()
    assert torch.count_nonzero(integrator._site_index != -1) > 0
    assert torch.any(unfiltered < 10_000.0)
    target = torch.tensor([1.0, 0.0, 0.0])
    only = integrator.compute_esdf(
        esdf_origin=esdf_origin,
        semantic_filter=SemanticFilter(target=target, require_argmax=True)
    ).clone()
    excluded = integrator.compute_esdf(
        esdf_origin=esdf_origin,
        semantic_filter=SemanticFilter(
            target=target,
            mode="exclude",
            require_argmax=True,
        )
    ).clone()

    torch.testing.assert_close(only, unfiltered, atol=0.03, rtol=0.03)
    assert torch.all(integrator._site_index == -1)
    assert torch.all(excluded > 0)


def test_semantic_filter_esdf_gather_reuses_cuda_graph_with_refreshed_mask(
    cuda_device: str,
) -> None:
    integrator = _make_feature_esdf_integrator(
        cuda_device,
        seeding_method="gather",
        use_cuda_graph=True,
    )
    num_allocated = _allocate_test_blocks(integrator._tsdf_integrator, cuda_device)
    data = integrator.tsdf.data
    data.block_features[:num_allocated].zero_()
    data.block_features[:num_allocated, :, 0] = 1.0
    data.block_feature_weight[:num_allocated] = 1.0
    integrator._tsdf_integrator._mark_semantic_data_changed()
    esdf_origin = torch.tensor([0.0, 0.0, 1.0], device=cuda_device)

    retained = integrator.compute_esdf(
        esdf_origin=esdf_origin,
        semantic_filter=SemanticFilter(
            target=torch.tensor([1.0, 0.0, 0.0]),
            require_argmax=True,
        ),
    ).clone()
    graph = integrator._compute_esdf_semantic_graphs["only"]
    if not graph._use_cuda_graph:
        pytest.skip("CUDA graphs are disabled by the runtime configuration")
    assert graph.is_initialized
    assert torch.any(retained < 10_000.0)

    rejected = integrator.compute_esdf(
        esdf_origin=esdf_origin,
        semantic_filter=SemanticFilter(
            target=torch.tensor([0.0, 1.0, 0.0]),
            require_argmax=True,
        ),
    ).clone()

    assert integrator._compute_esdf_semantic_graphs["only"] is graph
    assert torch.all(integrator._site_index == -1)
    assert torch.all(rejected > 0)
