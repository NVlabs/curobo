# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for relative tool-transform target query data."""

# Third Party
import pytest
import torch

# CuRobo
from curobo.types import ToolPose, ToolRelativeTransformTarget


def _make_target(
    num_target_groups: int = 2,
    tool_frames: list[str] | None = None,
    requires_grad: bool = False,
) -> ToolRelativeTransformTarget:
    if tool_frames is None:
        tool_frames = ["reference", "left_tool", "right_tool"]
    position = torch.arange(
        num_target_groups * len(tool_frames) * 3,
        dtype=torch.float32,
    ).reshape(num_target_groups, len(tool_frames), 3)
    quaternion = torch.zeros(num_target_groups, len(tool_frames), 4)
    quaternion[..., 0] = 2.0
    position.requires_grad_(requires_grad)
    quaternion.requires_grad_(requires_grad)
    return ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=tool_frames,
        position=position,
        quaternion=quaternion,
    )


def _make_tool_pose() -> ToolPose:
    tool_frames = ["reference", "tool"]
    position = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 1.0, 3.0]],
            ],
            [
                [[10.0, 0.0, 0.0], [10.0, 2.0, 0.0]],
                [[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
            ],
        ]
    )
    quaternion = torch.zeros(2, 2, 2, 4)
    quaternion[..., 0] = 2.0
    return ToolPose(
        tool_frames=tool_frames,
        position=position,
        quaternion=quaternion,
    )


def test_target_shape_properties_and_quaternion_values() -> None:
    target = _make_target()

    assert target.num_target_groups == 2
    assert target.num_tool_frames == 3
    assert target.shape == (2, 3, 3)
    assert target.ndim == 3
    assert target.device == torch.device("cpu")
    assert target.dtype == torch.float32
    assert torch.all(target.quaternion[..., 0] == 2.0)


def test_target_factory_preserves_batch_and_selects_horizon() -> None:
    tool_pose = _make_tool_pose()

    first = ToolRelativeTransformTarget.from_tool_pose(tool_pose, "reference")
    second = ToolRelativeTransformTarget.from_tool_pose(
        tool_pose,
        "reference",
        horizon_index=1,
    )

    assert first.num_target_groups == 2
    assert first.tool_frames == ["reference", "tool"]
    assert first.tool_frames is not tool_pose.tool_frames
    assert torch.allclose(
        first.position,
        torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        ),
    )
    assert torch.allclose(
        second.position,
        torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ]
        ),
    )
    assert torch.allclose(first.quaternion[..., 0], torch.ones(2, 2))
    assert torch.allclose(first.quaternion[..., 1:], torch.zeros(2, 2, 3))

    tool_pose.tool_frames[1] = "renamed"
    assert first.tool_frames == ["reference", "tool"]


def test_target_factory_uses_reference_frame_for_position_and_rotation() -> None:
    root_half = 2.0**-0.5
    reference_quaternion = torch.tensor([root_half, 0.0, 0.0, root_half])
    position = torch.tensor([[[[1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [2.0, 2.0, 0.0]]]])
    quaternion = torch.stack(
        (
            2.0 * reference_quaternion,
            torch.tensor([3.0, 0.0, 0.0, 0.0]),
            -4.0 * reference_quaternion,
        )
    ).reshape(1, 1, 3, 4)
    tool_pose = ToolPose(
        tool_frames=["reference", "world_aligned", "reference_aligned"],
        position=position,
        quaternion=quaternion,
    )

    target = ToolRelativeTransformTarget.from_tool_pose(tool_pose, "reference")

    assert torch.allclose(
        target.position,
        torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        target.quaternion,
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [root_half, 0.0, 0.0, -root_half],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        atol=1.0e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(target.quaternion, dim=-1),
        torch.ones(1, 3),
    )


def test_target_factory_normalizes_small_nonzero_float64_quaternions() -> None:
    tool_pose = ToolPose(
        tool_frames=["reference", "tool"],
        position=torch.zeros(1, 1, 2, 3, dtype=torch.float64),
        quaternion=torch.tensor(
            [[[[1.0e-10, 0.0, 0.0, 0.0], [0.0, 1.0e-10, 0.0, 0.0]]]],
            dtype=torch.float64,
        ),
    )

    target = ToolRelativeTransformTarget.from_tool_pose(tool_pose, "reference")

    assert torch.allclose(
        target.quaternion,
        torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            dtype=torch.float64,
        ),
    )


def test_target_factory_rejects_missing_reference_frame() -> None:
    with pytest.raises(ValueError, match="reference_tool_frame"):
        ToolRelativeTransformTarget.from_tool_pose(_make_tool_pose(), "missing")


def test_target_factory_rejects_non_tool_pose_input() -> None:
    with pytest.raises(ValueError, match="requires a ToolPose"):
        ToolRelativeTransformTarget.from_tool_pose(  # type: ignore[arg-type]
            object(),
            "reference",
        )


@pytest.mark.parametrize(
    ("horizon_index", "error_match"),
    [
        (-1, "out of range"),
        (2, "out of range"),
        (0.5, "must be an integer"),
        (True, "must be an integer"),
    ],
)
def test_target_factory_rejects_invalid_horizon_index(
    horizon_index: object,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        ToolRelativeTransformTarget.from_tool_pose(
            _make_tool_pose(),
            "reference",
            horizon_index=horizon_index,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("position", "quaternion", "error_match"),
    [
        (
            torch.zeros(1, 1, 2, 2),
            torch.ones(1, 1, 2, 4),
            "position last dimension must be 3",
        ),
        (
            torch.zeros(1, 2, 2, 3),
            torch.ones(1, 1, 2, 4),
            "leading dimensions must match",
        ),
        (
            torch.zeros(1, 1, 2, 3, dtype=torch.float32),
            torch.ones(1, 1, 2, 4, dtype=torch.float64),
            "same dtype",
        ),
        (
            torch.zeros(1, 1, 2, 3, dtype=torch.int64),
            torch.ones(1, 1, 2, 4, dtype=torch.int64),
            "floating dtype",
        ),
    ],
)
def test_target_factory_rejects_incompatible_pose_tensors(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    error_match: str,
) -> None:
    tool_pose = ToolPose(
        tool_frames=["reference", "tool"],
        position=position,
        quaternion=quaternion,
    )

    with pytest.raises(ValueError, match=error_match):
        ToolRelativeTransformTarget.from_tool_pose(tool_pose, "reference")


@pytest.mark.parametrize(("field", "value"), [("position", float("nan")), ("quaternion", 0.0)])
def test_target_factory_rejects_invalid_pose_values(field: str, value: float) -> None:
    tool_pose = _make_tool_pose()
    if field == "position":
        tool_pose.position[0, 0, 0, 0] = value
        error_match = "position must contain only finite"
    else:
        tool_pose.quaternion[0, 0, 1].fill_(value)
        error_match = "quaternion norm must be nonzero"

    with pytest.raises(ValueError, match=error_match):
        ToolRelativeTransformTarget.from_tool_pose(tool_pose, "reference")


@pytest.mark.parametrize(
    ("position_shape", "quaternion_shape", "error_match"),
    [
        ((2, 3), (2, 3, 4), "position must be 3D"),
        ((2, 3, 3), (2, 4), "quaternion must be 3D"),
        ((2, 3, 2), (2, 3, 4), "last dimension must be 3"),
        ((2, 3, 3), (2, 3, 5), "last dimension must be 4"),
        ((2, 3, 3), (1, 3, 4), "leading dimensions must match"),
        ((2, 3, 3), (2, 2, 4), "leading dimensions must match"),
    ],
)
def test_target_rejects_invalid_tensor_shapes(
    position_shape: tuple[int, ...],
    quaternion_shape: tuple[int, ...],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference", "left_tool", "right_tool"],
            position=torch.zeros(position_shape),
            quaternion=torch.zeros(quaternion_shape),
        )


def test_target_rejects_tool_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match=r"len\(tool_frames\)"):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference", "left_tool"],
            position=torch.zeros(2, 3, 3),
            quaternion=torch.zeros(2, 3, 4),
        )


def test_target_rejects_missing_reference() -> None:
    with pytest.raises(ValueError, match="reference_tool_frame"):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["left_tool", "right_tool"],
            position=torch.zeros(2, 2, 3),
            quaternion=torch.zeros(2, 2, 4),
        )


def test_target_rejects_duplicate_tool_frames() -> None:
    with pytest.raises(ValueError, match="unique"):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference", "left_tool", "left_tool"],
            position=torch.zeros(2, 3, 3),
            quaternion=torch.zeros(2, 3, 4),
        )


def test_target_rejects_empty_target_groups() -> None:
    with pytest.raises(ValueError, match="at least one target group"):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference"],
            position=torch.zeros(0, 1, 3),
            quaternion=torch.zeros(0, 1, 4),
        )


def test_target_rejects_mixed_dtype() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference"],
            position=torch.zeros(1, 1, 3, dtype=torch.float32),
            quaternion=torch.zeros(1, 1, 4, dtype=torch.float64),
        )


def test_target_rejects_nonfloating_dtype() -> None:
    with pytest.raises(ValueError, match="floating dtype"):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference"],
            position=torch.zeros(1, 1, 3, dtype=torch.int32),
            quaternion=torch.zeros(1, 1, 4, dtype=torch.int32),
        )


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("position", float("nan"), "position must contain only finite"),
        ("quaternion", float("inf"), "quaternion must contain only finite"),
        ("quaternion", 0.0, "quaternion norm must be nonzero"),
        ("quaternion", 1.0e-12, "quaternion norm must be nonzero"),
    ],
)
def test_target_rejects_invalid_transform_values(
    field: str,
    value: float,
    error_match: str,
) -> None:
    position = torch.zeros(1, 1, 3)
    quaternion = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    if field == "position":
        position[0, 0, 0] = value
    else:
        quaternion.fill_(value)

    with pytest.raises(ValueError, match=error_match):
        ToolRelativeTransformTarget(
            reference_tool_frame="reference",
            tool_frames=["reference"],
            position=position,
            quaternion=quaternion,
        )


def test_target_clone_is_independent() -> None:
    target = _make_target()
    cloned = target.clone()

    assert cloned.reference_tool_frame == target.reference_tool_frame
    assert cloned.tool_frames == target.tool_frames
    assert cloned.tool_frames is not target.tool_frames
    assert torch.equal(cloned.position, target.position)
    assert torch.equal(cloned.quaternion, target.quaternion)

    cloned.position[0, 0, 0] = -1.0
    cloned.quaternion[0, 0, 0] = -1.0
    cloned.tool_frames[1] = "renamed"
    assert target.position[0, 0, 0] != -1.0
    assert target.quaternion[0, 0, 0] != -1.0
    assert target.tool_frames[1] == "left_tool"


def test_target_detach_and_requires_grad() -> None:
    target = _make_target(requires_grad=True)
    detached = target.detach()

    assert not detached.position.requires_grad
    assert not detached.quaternion.requires_grad
    assert detached.position.data_ptr() == target.position.data_ptr()
    assert detached.quaternion.data_ptr() == target.quaternion.data_ptr()

    detached.requires_grad_(True)
    assert detached.position.requires_grad
    assert detached.quaternion.requires_grad


def test_target_contiguous() -> None:
    tool_frames = ["reference", "left_tool", "right_tool"]
    position = torch.zeros(2, 3, 6)[..., ::2]
    quaternion = torch.zeros(2, 3, 8)[..., ::2]
    quaternion[..., 0] = 1.0
    target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=tool_frames,
        position=position,
        quaternion=quaternion,
    )
    assert not target.position.is_contiguous()
    assert not target.quaternion.is_contiguous()

    contiguous = target.contiguous()

    assert contiguous.position.is_contiguous()
    assert contiguous.quaternion.is_contiguous()
    assert torch.equal(contiguous.position, position)
    assert torch.equal(contiguous.quaternion, quaternion)


def test_target_copy_updates_storage_in_place() -> None:
    target = _make_target()
    source = _make_target()
    source.position.add_(100.0)
    source.quaternion.add_(3.0)
    position_ptr = target.position.data_ptr()
    quaternion_ptr = target.quaternion.data_ptr()

    target.copy_(source)

    assert target.position.data_ptr() == position_ptr
    assert target.quaternion.data_ptr() == quaternion_ptr
    assert torch.equal(target.position, source.position)
    assert torch.equal(target.quaternion, source.quaternion)


@pytest.mark.parametrize("mismatch", ["reference", "frames", "shape"])
def test_target_copy_rejects_incompatible_metadata_or_shape(mismatch: str) -> None:
    target = _make_target()
    source = _make_target()
    if mismatch == "reference":
        source.reference_tool_frame = "left_tool"
        error_match = "reference_tool_frame"
    elif mismatch == "frames":
        source.tool_frames = ["reference", "right_tool", "left_tool"]
        error_match = "tool_frames"
    else:
        source = _make_target(num_target_groups=1)
        error_match = "shape"

    with pytest.raises(ValueError, match=error_match):
        target.copy_(source)
