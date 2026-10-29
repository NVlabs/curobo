# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Goal-registry lifecycle tests for relative tool-transform targets."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget


def _make_target(offset: float = 0.0) -> ToolRelativeTransformTarget:
    position = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]],
        ],
        dtype=torch.float32,
    )
    position = position + offset
    quaternion = torch.zeros((2, 2, 4), dtype=torch.float32)
    quaternion[..., 0] = 1.0
    return ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=position,
        quaternion=quaternion,
    )


def test_relative_target_creates_default_routing_indices() -> None:
    target = _make_target()

    registry = GoalRegistry(tool_relative_transform_target=target)

    assert registry.batch_size == 2
    torch.testing.assert_close(
        registry.idxs_tool_relative_transform_target,
        torch.tensor([[0], [1]], dtype=torch.int32),
    )


def test_shared_relative_target_preserves_explicit_problem_batch_size() -> None:
    target = _make_target()
    shared = ToolRelativeTransformTarget(
        reference_tool_frame=target.reference_tool_frame,
        tool_frames=target.tool_frames,
        position=target.position[:1],
        quaternion=target.quaternion[:1],
    )

    registry = GoalRegistry(
        batch_size=4,
        tool_relative_transform_target=shared,
    )

    assert registry.batch_size == 4
    torch.testing.assert_close(
        registry.idxs_tool_relative_transform_target,
        torch.zeros((4, 1), dtype=torch.int32),
    )


def test_relative_target_default_routing_covers_every_seed() -> None:
    target = _make_target()

    registry = GoalRegistry(
        batch_size=2,
        num_seeds=3,
        tool_relative_transform_target=target,
    )

    torch.testing.assert_close(
        registry.idxs_tool_relative_transform_target,
        torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.int32),
    )


def test_relative_target_rejects_route_with_wrong_seed_row_count() -> None:
    with pytest.raises(ValueError, match=r"batch_size \* num_seeds"):
        GoalRegistry(
            batch_size=2,
            num_seeds=3,
            tool_relative_transform_target=_make_target(),
            idxs_tool_relative_transform_target=torch.tensor(
                [[0], [1]],
                dtype=torch.int32,
            ),
        )


def test_relative_target_repeat_seeds_preserves_target_and_repeats_indices() -> None:
    target = _make_target()
    registry = GoalRegistry(tool_relative_transform_target=target)

    repeated = registry.repeat_seeds(3)

    assert repeated.tool_relative_transform_target is target
    assert repeated.batch_size == 2
    assert repeated.num_seeds == 3
    torch.testing.assert_close(
        repeated.idxs_tool_relative_transform_target,
        torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.int32),
    )


def test_relative_target_clone_is_independent() -> None:
    registry = GoalRegistry(tool_relative_transform_target=_make_target())

    cloned = registry.clone()

    assert cloned.tool_relative_transform_target is not registry.tool_relative_transform_target
    assert (
        cloned.tool_relative_transform_target.position.data_ptr()
        != registry.tool_relative_transform_target.position.data_ptr()
    )
    cloned.tool_relative_transform_target.position.add_(1.0)
    assert not torch.equal(
        cloned.tool_relative_transform_target.position,
        registry.tool_relative_transform_target.position,
    )


def test_relative_target_apply_kernel_routes_target_groups() -> None:
    registry = GoalRegistry(tool_relative_transform_target=_make_target())
    kernel = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )

    transformed = registry.apply_kernel(kernel)

    assert transformed.tool_relative_transform_target is registry.tool_relative_transform_target
    assert transformed.batch_size == 3
    torch.testing.assert_close(
        transformed.idxs_tool_relative_transform_target,
        torch.tensor([[0], [1], [0]], dtype=torch.int32),
    )


def test_relative_target_copy_updates_existing_storage_and_indices() -> None:
    target = GoalRegistry(tool_relative_transform_target=_make_target())
    source = GoalRegistry(
        tool_relative_transform_target=_make_target(offset=0.5),
        idxs_tool_relative_transform_target=torch.tensor([[1], [0]], dtype=torch.int32),
    )
    position_ptr = target.tool_relative_transform_target.position.data_ptr()
    index_ptr = target.idxs_tool_relative_transform_target.data_ptr()

    target.copy_(source, allow_clone=False)

    assert target.tool_relative_transform_target.position.data_ptr() == position_ptr
    assert target.idxs_tool_relative_transform_target.data_ptr() == index_ptr
    torch.testing.assert_close(
        target.tool_relative_transform_target.position,
        source.tool_relative_transform_target.position,
    )
    torch.testing.assert_close(
        target.idxs_tool_relative_transform_target,
        source.idxs_tool_relative_transform_target,
    )


def test_relative_target_copy_rejects_presence_change_without_clone() -> None:
    target = GoalRegistry(tool_relative_transform_target=_make_target())

    try:
        target.copy_(GoalRegistry(batch_size=2), allow_clone=False)
    except ValueError as error:
        assert "allow_clone=False" in str(error)
    else:
        raise AssertionError("Expected fixed-shape copy to reject target removal")


def test_relative_target_copy_clears_target_and_index_when_clone_allowed() -> None:
    target = GoalRegistry(tool_relative_transform_target=_make_target())

    target.copy_(GoalRegistry(batch_size=2), allow_clone=True)

    assert target.tool_relative_transform_target is None
    assert target.idxs_tool_relative_transform_target is None


def test_relative_only_registry_reports_index_size() -> None:
    registry = GoalRegistry(tool_relative_transform_target=_make_target())

    assert registry.get_index_size() == 2


def test_create_idx_without_relative_target_does_not_create_relative_route() -> None:
    registry = GoalRegistry.create_idx(
        pose_batch_size=2,
        multi_env=False,
        num_seeds=3,
        device_cfg=DeviceCfg(device=torch.device("cpu")),
    )

    assert registry.tool_relative_transform_target is None
    assert registry.idxs_tool_relative_transform_target is None
