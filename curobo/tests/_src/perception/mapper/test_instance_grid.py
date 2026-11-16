# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for categorical instance-grid integration."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.storage import BlockDataView
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


@pytest.fixture
def cuda_device() -> str:
    """Return the CUDA device used by mapper integration tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for instance-grid integration tests")
    init_warp()
    return "cuda:0"


def _make_integrator(
    device: str,
    *,
    num_cameras: int = 1,
    instance_grid_size: int = 4,
) -> BlockSparseTSDFIntegrator:
    """Build a small mapper with categorical instance storage enabled."""
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=500,
            voxel_size=0.01,
            origin=torch.zeros(3, dtype=torch.float32),
            grid_shape=(256, 256, 256),
            truncation_distance=0.05,
            device=device,
            num_cameras=num_cameras,
            image_height=16,
            image_width=16,
            instance_grid_size=instance_grid_size,
            max_support_pixels_per_block_camera=8,
        )
    )


def _make_observation(
    device: str,
    instance_slots: torch.Tensor,
) -> CameraObservation:
    """Build a planar RGB-D observation carrying compact instance slots."""
    num_cameras = int(instance_slots.shape[0])
    image_height = int(instance_slots.shape[1])
    image_width = int(instance_slots.shape[2])
    depth = torch.full(
        (num_cameras, image_height, image_width),
        1.0,
        dtype=torch.float32,
        device=device,
    )
    rgb = torch.zeros(
        (num_cameras, image_height, image_width, 3),
        dtype=torch.uint8,
        device=device,
    )
    intrinsics = (
        torch.tensor(
            [
                [100.0, 0.0, image_width / 2.0],
                [0.0, 100.0, image_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        .expand(num_cameras, -1, -1)
        .contiguous()
    )
    positions = torch.zeros((num_cameras, 3), dtype=torch.float32, device=device)
    quaternions = torch.zeros((num_cameras, 4), dtype=torch.float32, device=device)
    quaternions[:, 0] = 1.0
    return CameraObservation(
        depth_image=depth,
        rgb_image=rgb,
        pose=Pose(position=positions, quaternion=quaternions),
        intrinsics=intrinsics,
        instance_image=instance_slots,
    )


def _active_instance_state(integrator: BlockSparseTSDFIntegrator) -> torch.Tensor:
    """Return packed instance state for the allocated block prefix as int32."""
    data = integrator.tsdf.data
    num_allocated = int(data.num_allocated.item())
    return data.block_instance_state[:num_allocated].to(dtype=torch.int32) & 0xFFFF


def test_camera_observation_instance_image_copy_clone_and_to() -> None:
    """CameraObservation preserves instance images in standard tensor operations."""
    instance_image = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
    observation = CameraObservation(instance_image=instance_image)

    cloned = observation.clone()
    assert cloned.instance_image is not observation.instance_image
    torch.testing.assert_close(cloned.instance_image, instance_image)

    destination = CameraObservation(instance_image=torch.empty_like(instance_image))
    destination.copy_(observation)
    torch.testing.assert_close(destination.instance_image, instance_image)

    moved = observation.to(torch.device("cpu"))
    assert moved is observation
    assert observation.instance_image.device.type == "cpu"


def test_block_data_view_samples_containing_instance_cells() -> None:
    """Fine TSDF voxel centers map categorically into the containing coarse cell."""
    packed = torch.tensor(
        [[(10 << 8) | slot for slot in range(1, 9)]],
        dtype=torch.int32,
    ).to(torch.int16)
    view = BlockDataView(
        rgb_grid=torch.zeros((1, 1, 4), dtype=torch.float16),
        coords=torch.zeros(3, dtype=torch.int32),
        num_allocated=1,
        origin=torch.zeros(3, dtype=torch.float32),
        voxel_size=1.0,
        block_size=4,
        grid_shape=(4, 4, 4),
        color_grid_size=1,
        feature_block_grid_size=1,
        features=torch.zeros((1, 1, 1), dtype=torch.float16),
        feature_weight=torch.zeros((1, 1), dtype=torch.float16),
        feature_dim=0,
        instance_state=packed,
        instance_grid_size=2,
    )
    centers = torch.tensor(
        [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
        dtype=torch.float32,
    )
    block_indices = torch.zeros(2, dtype=torch.int32)

    sampled = view.sample_instance_state_at_centers(centers, block_indices).to(dtype=torch.int32)

    torch.testing.assert_close(sampled & 0xFF, torch.tensor([1, 8], dtype=torch.int32))
    torch.testing.assert_close(sampled >> 8, torch.tensor([10, 10], dtype=torch.int32))


def test_instance_grid_size_validation(cuda_device: str) -> None:
    """Instance storage is optional and cannot exceed the TSDF block resolution."""
    _make_integrator(cuda_device, instance_grid_size=0)
    with pytest.raises(ValueError, match="instance_grid_size must be <= block_size"):
        _make_integrator(cuda_device, instance_grid_size=9)


def test_instance_image_rejected_when_storage_disabled(cuda_device: str) -> None:
    """Input instance slots require an enabled instance block grid."""
    integrator = _make_integrator(cuda_device, instance_grid_size=0)
    instance_slots = torch.full((1, 16, 16), 7, dtype=torch.uint8, device=cuda_device)

    with pytest.raises(ValueError, match="instance_image was provided"):
        integrator.integrate(_make_observation(cuda_device, instance_slots))


def test_instance_image_requires_uint8(cuda_device: str) -> None:
    """The cuRobo mapper consumes compact uint8 slots, not detector track IDs."""
    integrator = _make_integrator(cuda_device)
    instance_slots = torch.full((1, 16, 16), 7, dtype=torch.int32, device=cuda_device)

    with pytest.raises(ValueError, match="instance_image dtype must be torch.uint8"):
        integrator.integrate(_make_observation(cuda_device, instance_slots))


def test_single_view_instance_confidence_accumulates(cuda_device: str) -> None:
    """Repeated matching observations keep ownership and increase confidence."""
    integrator = _make_integrator(cuda_device)
    instance_slots = torch.full((1, 16, 16), 7, dtype=torch.uint8, device=cuda_device)
    observation = _make_observation(cuda_device, instance_slots)

    integrator.integrate(observation)
    first = _active_instance_state(integrator)
    first_owned = (first & 0xFF) == 7
    assert bool(first_owned.any().item())
    first_confidence = first[first_owned] >> 8

    integrator.integrate(observation)
    second = _active_instance_state(integrator)
    second_owned = (second & 0xFF) == 7
    second_confidence = second[second_owned] >> 8

    assert int(second_confidence.min().item()) > int(first_confidence.min().item())


def test_two_view_equal_conflict_abstains_independent_of_order(cuda_device: str) -> None:
    """Equal conflicting multiview votes do not initialize ownership."""
    forward = _make_integrator(cuda_device, num_cameras=2)
    reverse = _make_integrator(cuda_device, num_cameras=2)
    slots_forward = torch.stack(
        (
            torch.full((16, 16), 3, dtype=torch.uint8, device=cuda_device),
            torch.full((16, 16), 7, dtype=torch.uint8, device=cuda_device),
        )
    )

    forward.integrate(_make_observation(cuda_device, slots_forward))
    reverse.integrate(_make_observation(cuda_device, slots_forward.flip(0).contiguous()))

    assert not bool((_active_instance_state(forward) != 0).any().item())
    assert not bool((_active_instance_state(reverse) != 0).any().item())


def test_two_agreeing_views_add_more_confidence(cuda_device: str) -> None:
    """Two agreeing cameras reinforce ownership more than one camera."""
    single = _make_integrator(cuda_device, num_cameras=1)
    multi = _make_integrator(cuda_device, num_cameras=2)
    one_view = torch.full((1, 16, 16), 5, dtype=torch.uint8, device=cuda_device)
    two_views = torch.full((2, 16, 16), 5, dtype=torch.uint8, device=cuda_device)

    single.integrate(_make_observation(cuda_device, one_view))
    multi.integrate(_make_observation(cuda_device, two_views))

    single_confidence = _active_instance_state(single) >> 8
    multi_confidence = _active_instance_state(multi) >> 8
    assert int(multi_confidence.max().item()) > int(single_confidence.max().item())


def test_contradictory_instance_switches_only_after_hysteresis(cuda_device: str) -> None:
    """One contradictory frame weakens ownership; sustained evidence replaces it."""
    integrator = _make_integrator(cuda_device)
    old_slots = torch.full((1, 16, 16), 3, dtype=torch.uint8, device=cuda_device)
    new_slots = torch.full((1, 16, 16), 7, dtype=torch.uint8, device=cuda_device)
    old_observation = _make_observation(cuda_device, old_slots)
    new_observation = _make_observation(cuda_device, new_slots)

    for _ in range(8):
        integrator.integrate(old_observation)
    before = _active_instance_state(integrator)
    assert bool(((before & 0xFF) == 3).any().item())

    integrator.integrate(new_observation)
    after_one = _active_instance_state(integrator)
    assert bool(((after_one & 0xFF) == 3).any().item())

    for _ in range(8):
        integrator.integrate(new_observation)
    after_many = _active_instance_state(integrator)
    assert bool(((after_many & 0xFF) == 7).any().item())


def test_background_decays_slowly_and_reset_clears(cuda_device: str) -> None:
    """A detector miss does not erase ownership immediately; reset always does."""
    integrator = _make_integrator(cuda_device)
    object_slots = torch.full((1, 16, 16), 9, dtype=torch.uint8, device=cuda_device)
    background_slots = torch.zeros((1, 16, 16), dtype=torch.uint8, device=cuda_device)

    integrator.integrate(_make_observation(cuda_device, object_slots))
    integrator.integrate(_make_observation(cuda_device, background_slots))
    after_background = _active_instance_state(integrator)
    assert bool(((after_background & 0xFF) == 9).any().item())

    integrator.reset()
    assert not bool((integrator.tsdf.data.block_instance_state != 0).any().item())


def test_clear_instance_slot_preserves_other_slots(cuda_device: str) -> None:
    """Retiring a registry slot clears only cells owned by that slot."""
    integrator = _make_integrator(cuda_device)
    state = integrator.tsdf.data.block_instance_state
    state.zero_()
    integrator.tsdf.data.num_allocated.fill_(1)
    state[0, 0] = (20 << 8) | 3
    state[0, 1] = (20 << 8) | 7

    integrator.clear_instance_slot(3)
    unpacked = state[0, :2].to(dtype=torch.int32)

    assert int(unpacked[0].item()) == 0
    assert int((unpacked[1] & 0xFF).item()) == 7
