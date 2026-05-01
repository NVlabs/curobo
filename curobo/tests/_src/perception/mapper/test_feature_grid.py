# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for mapper feature-grid integration input."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation


def test_camera_observation_feature_grid_copy_clone_to_cpu():
    feature_grid = torch.arange(2 * 3 * 4 * 5, dtype=torch.float16).reshape(2, 3, 4, 5)
    obs = CameraObservation(feature_grid=feature_grid)

    cloned = obs.clone()
    assert cloned.feature_grid is not obs.feature_grid
    torch.testing.assert_close(cloned.feature_grid, feature_grid)

    dst = CameraObservation(feature_grid=torch.empty_like(feature_grid))
    dst.copy_(obs)
    torch.testing.assert_close(dst.feature_grid, feature_grid)

    moved = obs.to(torch.device("cpu"))
    assert moved is obs
    assert obs.feature_grid.device.type == "cpu"


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for block-sparse mapper integration tests")
    init_warp()
    return "cuda:0"


def _make_feature_observation(
    device: str,
    feature_grid: torch.Tensor,
    image_height: int = 16,
    image_width: int = 16,
) -> CameraObservation:
    depth = torch.full((image_height, image_width), 1.0, dtype=torch.float32, device=device)
    rgb = torch.zeros((image_height, image_width, 3), dtype=torch.uint8, device=device)
    position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    intrinsics = torch.tensor(
        [
            [100.0, 0.0, image_width / 2.0],
            [0.0, 100.0, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    obs = make_observation(depth, rgb, position, quaternion, intrinsics)
    obs.feature_grid = feature_grid
    return obs


def _make_integrator(
    device: str,
    feature_dim: int,
    feature_channels_per_thread: int = 4,
    feature_grid_shape: tuple[int, int] | None = None,
) -> BlockSparseTSDFIntegrator:
    feature_grid_kwargs = {}
    if feature_dim > 0:
        if feature_grid_shape is None:
            feature_grid_shape = (2, 2)
        feature_grid_kwargs = {
            "feature_grid_height": feature_grid_shape[0],
            "feature_grid_width": feature_grid_shape[1],
        }
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=device,
            image_height=16,
            image_width=16,
            feature_dim=feature_dim,
            feature_channels_per_thread=feature_channels_per_thread,
            max_support_pixels_per_block_camera=8,
            **feature_grid_kwargs,
        )
    )


def test_feature_grid_shape_required_when_features_enabled(cuda_device):
    with pytest.raises(ValueError, match="feature_dim > 0 requires"):
        BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=cuda_device,
            image_height=16,
            image_width=16,
            feature_dim=3,
        )


def test_feature_grid_shape_rejected_when_features_disabled(cuda_device):
    with pytest.raises(ValueError, match="require feature_dim > 0"):
        BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.05,
            device=cuda_device,
            image_height=16,
            image_width=16,
            feature_dim=0,
            feature_grid_height=2,
            feature_grid_width=2,
        )


def test_feature_grid_with_disabled_features_raises(cuda_device):
    integrator = _make_integrator(cuda_device, feature_dim=0)
    feature_grid = torch.zeros((1, 2, 2, 3), dtype=torch.float16, device=cuda_device)
    obs = _make_feature_observation(cuda_device, feature_grid)

    with pytest.raises(ValueError, match="feature_grid was provided but feature_dim == 0"):
        integrator.integrate(obs)


def test_feature_grid_requires_fp16(cuda_device):
    integrator = _make_integrator(cuda_device, feature_dim=3)
    feature_grid = torch.zeros((1, 2, 2, 3), dtype=torch.float32, device=cuda_device)
    obs = _make_feature_observation(cuda_device, feature_grid)

    with pytest.raises(ValueError, match="feature_grid dtype must be torch.float16"):
        integrator.integrate(obs)


def test_feature_grid_requires_channel_stride_one(cuda_device):
    integrator = _make_integrator(cuda_device, feature_dim=3)
    base = torch.zeros((1, 3, 2, 2), dtype=torch.float16, device=cuda_device)
    feature_grid = base.permute(0, 2, 3, 1)
    assert feature_grid.shape == (1, 2, 2, 3)
    assert feature_grid.stride(-1) != 1
    obs = _make_feature_observation(cuda_device, feature_grid)

    with pytest.raises(ValueError, match="stride 1 on the channel dim"):
        integrator.integrate(obs)


def test_feature_grid_device_must_match_depth(cuda_device):
    integrator = _make_integrator(cuda_device, feature_dim=3)
    feature_grid = torch.zeros((1, 2, 2, 3), dtype=torch.float16, device="cpu")
    obs = _make_feature_observation(cuda_device, feature_grid)

    with pytest.raises(ValueError, match="feature_grid device .* does not match"):
        integrator.integrate(obs)


def test_feature_grid_smoke_integrates_patch_grid(cuda_device):
    integrator = _make_integrator(cuda_device, feature_dim=2)
    feature_grid = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [-1.0, 1.0]]]],
        dtype=torch.float16,
        device=cuda_device,
    )
    obs = _make_feature_observation(cuda_device, feature_grid)

    integrator.integrate(obs)

    n_allocated = int(integrator.tsdf.data.num_allocated.item())
    assert n_allocated > 0
    weights = integrator.tsdf.data.block_feature_weight[:n_allocated].float()
    assert (weights > 0).any()
    features = integrator.tsdf.data.block_features[:n_allocated].float()
    assert torch.isfinite(features).all()


def test_feature_grid_grouping_keeps_trailing_channels(cuda_device):
    integrator = _make_integrator(
        cuda_device,
        feature_dim=7,
        feature_channels_per_thread=5,
        feature_grid_shape=(4, 4),
    )
    channel_values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        device=cuda_device,
    )
    feature_grid = channel_values.to(torch.float16).view(1, 1, 1, 7).expand(1, 4, 4, 7)
    obs = _make_feature_observation(cuda_device, feature_grid.contiguous())

    integrator.integrate(obs)

    n_allocated = int(integrator.tsdf.data.num_allocated.item())
    assert n_allocated > 0
    weights = integrator.tsdf.data.block_feature_weight[:n_allocated].float()
    active = weights > 0
    assert active.any()
    features = integrator.tsdf.data.block_features[:n_allocated].float()
    normalized = features[active] / weights[active].unsqueeze(-1)
    expected = channel_values.expand_as(normalized)
    torch.testing.assert_close(normalized, expected, atol=0.05, rtol=0.05)
