# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared test utilities for block-sparse perception tests."""

import pytest
import torch

from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    make_block_sparse_kernels,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="session", autouse=True)
def _prewarm_block_sparse_kernels():
    """Build the default-BS kernel bundle once per session.

    Warp still owns compiled-kernel reuse; the Python bundle itself is
    intentionally not cached.
    """
    if not torch.cuda.is_available():
        return

    init_warp()
    make_block_sparse_kernels(block_size=8)


def make_observation(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    intrinsics: torch.Tensor,
) -> CameraObservation:
    """Build a batched CameraObservation from raw tensors.

    Tensors without a leading camera dimension are unsqueezed automatically
    so the returned observation always has shape ``(num_cameras, ...)``.

    Args:
        depth: Depth image ``(H, W)`` or ``(num_cameras, H, W)`` [m].
        rgb: RGB image ``(H, W, 3)`` or ``(num_cameras, H, W, 3)`` uint8.
        position: Camera position ``(3,)`` or ``(num_cameras, 3)`` [m].
        quaternion: Camera quaternion wxyz ``(4,)`` or ``(num_cameras, 4)``.
        intrinsics: Camera intrinsic matrix ``(3, 3)`` or ``(num_cameras, 3, 3)``.

    Returns:
        CameraObservation with leading camera dimension.
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if position.ndim == 1:
        position = position.unsqueeze(0)
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    return CameraObservation(
        depth_image=depth,
        rgb_image=rgb,
        pose=Pose(position=position, quaternion=quaternion),
        intrinsics=intrinsics,
    )
