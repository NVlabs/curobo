# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for projector-backed tool-pose criteria."""

# Standard Library
from dataclasses import fields

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import (
    PoseErrorFrame,
    StackedToolPoseCriteria,
    ToolPoseCriteria,
)
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg() -> DeviceCfg:
    return DeviceCfg(device=torch.device("cpu"))


def _identity(device_cfg: DeviceCfg) -> torch.Tensor:
    return torch.eye(3, **device_cfg.as_torch_dict())


def _zero(device_cfg: DeviceCfg) -> torch.Tensor:
    return torch.zeros((3, 3), **device_cfg.as_torch_dict())


def test_default_criteria_uses_only_canonical_storage(device_cfg: DeviceCfg):
    criteria = ToolPoseCriteria(device_cfg=device_cfg)

    torch.testing.assert_close(criteria.terminal_position_projector, _identity(device_cfg))
    torch.testing.assert_close(criteria.terminal_rotation_projector, _identity(device_cfg))
    torch.testing.assert_close(criteria.path_position_projector, _zero(device_cfg))
    torch.testing.assert_close(criteria.path_rotation_projector, _zero(device_cfg))
    torch.testing.assert_close(criteria.terminal_position_axis_scale, torch.ones(3))
    torch.testing.assert_close(criteria.terminal_rotation_axis_scale, torch.ones(3))
    torch.testing.assert_close(criteria.path_position_axis_scale, torch.ones(3))
    torch.testing.assert_close(criteria.path_rotation_axis_scale, torch.ones(3))
    torch.testing.assert_close(criteria.terminal_tolerance, torch.zeros(2))
    torch.testing.assert_close(criteria.path_tolerance, torch.zeros(2))
    assert criteria.error_frame.item() == PoseErrorFrame.WORLD

    stored_names = {field.name for field in fields(criteria)}
    assert "terminal_pose_axes_weight_factor" not in stored_names
    assert "non_terminal_pose_axes_weight_factor" not in stored_names
    assert "terminal_pose_convergence_tolerance" not in stored_names
    assert "non_terminal_pose_convergence_tolerance" not in stored_names
    assert "project_distance_to_goal" not in stored_names


def test_legacy_arguments_convert_once_to_projectors_and_scales(device_cfg: DeviceCfg):
    criteria = ToolPoseCriteria(
        terminal_pose_axes_weight_factor=[2.0, 0.0, 0.5, 0.0, 3.0, 0.0],
        non_terminal_pose_axes_weight_factor=[0.0, 4.0, 0.0, 5.0, 0.0, 6.0],
        terminal_pose_convergence_tolerance=[0.01, 0.02],
        non_terminal_pose_convergence_tolerance=[0.03, 0.04],
        project_distance_to_goal=True,
        device_cfg=device_cfg,
    )

    torch.testing.assert_close(
        criteria.terminal_position_projector, torch.diag(torch.tensor([1.0, 0.0, 1.0]))
    )
    torch.testing.assert_close(
        criteria.terminal_position_axis_scale, torch.tensor([2.0, 1.0, 0.5])
    )
    torch.testing.assert_close(
        criteria.terminal_rotation_projector, torch.diag(torch.tensor([0.0, 1.0, 0.0]))
    )
    torch.testing.assert_close(
        criteria.terminal_rotation_axis_scale, torch.tensor([1.0, 3.0, 1.0])
    )
    torch.testing.assert_close(
        criteria.path_position_projector, torch.diag(torch.tensor([0.0, 1.0, 0.0]))
    )
    torch.testing.assert_close(criteria.path_position_axis_scale, torch.tensor([1.0, 4.0, 1.0]))
    torch.testing.assert_close(
        criteria.path_rotation_projector, torch.diag(torch.tensor([1.0, 0.0, 1.0]))
    )
    torch.testing.assert_close(criteria.path_rotation_axis_scale, torch.tensor([5.0, 1.0, 6.0]))
    torch.testing.assert_close(criteria.terminal_tolerance, torch.tensor([0.01, 0.02]))
    torch.testing.assert_close(criteria.path_tolerance, torch.tensor([0.03, 0.04]))
    assert criteria.error_frame.item() == PoseErrorFrame.GOAL


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "terminal_position_projector": torch.eye(3),
            "terminal_pose_axes_weight_factor": [1.0] * 6,
        },
        {
            "terminal_tolerance": [0.0, 0.0],
            "terminal_pose_convergence_tolerance": [0.0, 0.0],
        },
        {"error_frame": PoseErrorFrame.WORLD, "project_distance_to_goal": False},
    ],
)
def test_legacy_and_canonical_arguments_conflict(device_cfg: DeviceCfg, kwargs):
    with pytest.raises(ValueError, match="cannot be combined"):
        ToolPoseCriteria(device_cfg=device_cfg, **kwargs)


@pytest.mark.parametrize(
    "factor",
    [
        [1.0, 1.0, 1.0, 1.0, -1.0, 1.0],
        [1.0, 1.0, float("nan"), 1.0, 1.0, 1.0],
        [1.0, 1.0, float("inf"), 1.0, 1.0, 1.0],
    ],
)
def test_invalid_legacy_factor_is_rejected(device_cfg: DeviceCfg, factor):
    with pytest.raises(ValueError):
        ToolPoseCriteria(terminal_pose_axes_weight_factor=factor, device_cfg=device_cfg)


def test_canonical_validation(device_cfg: DeviceCfg):
    with pytest.raises(ValueError, match="symmetric"):
        ToolPoseCriteria(
            terminal_position_projector=[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="idempotent"):
        ToolPoseCriteria(terminal_position_projector=2.0 * torch.eye(3), device_cfg=device_cfg)
    with pytest.raises(ValueError, match="strictly positive"):
        ToolPoseCriteria(terminal_position_axis_scale=[1.0, 0.0, 1.0], device_cfg=device_cfg)
    with pytest.raises(ValueError, match="nonnegative"):
        ToolPoseCriteria(terminal_tolerance=[-0.1, 0.0], device_cfg=device_cfg)
    with pytest.raises(ValueError, match="error_frame"):
        ToolPoseCriteria(error_frame=7, device_cfg=device_cfg)


def test_factories_create_projectors_and_positive_scales(device_cfg: DeviceCfg):
    position = ToolPoseCriteria.track_position(xyz=[2.0, 0.0, 3.0], device_cfg=device_cfg)
    torch.testing.assert_close(
        position.terminal_position_projector, torch.diag(torch.tensor([1.0, 0.0, 1.0]))
    )
    torch.testing.assert_close(position.terminal_rotation_projector, _zero(device_cfg))
    assert torch.all(position.terminal_position_axis_scale > 0.0)

    orientation = ToolPoseCriteria.track_orientation(
        rpy=[0.0, 2.0, 1.0], non_terminal_scale=0.5, device_cfg=device_cfg
    )
    torch.testing.assert_close(orientation.terminal_position_projector, _zero(device_cfg))
    torch.testing.assert_close(orientation.path_rotation_axis_scale, torch.tensor([1.0, 1.0, 0.5]))

    pose = ToolPoseCriteria.track_pose(
        xyz=[1.0, 2.0, 3.0], rpy=[4.0, 5.0, 6.0], device_cfg=device_cfg
    )
    alias = ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 2.0, 3.0], rpy=[4.0, 5.0, 6.0], device_cfg=device_cfg
    )
    for name in (
        "terminal_position_projector",
        "terminal_rotation_projector",
        "path_position_projector",
        "path_rotation_projector",
        "terminal_position_axis_scale",
        "terminal_rotation_axis_scale",
        "path_position_axis_scale",
        "path_rotation_axis_scale",
    ):
        torch.testing.assert_close(getattr(alias, name), getattr(pose, name))

    disabled = ToolPoseCriteria.disabled(device_cfg=device_cfg)
    torch.testing.assert_close(disabled.terminal_position_projector, _zero(device_cfg))
    torch.testing.assert_close(disabled.path_rotation_projector, _zero(device_cfg))
    assert torch.all(disabled.terminal_position_axis_scale > 0.0)


def test_linear_motion_accepts_arbitrary_direction_and_axis_compatibility(device_cfg: DeviceCfg):
    direction = torch.tensor([1.0, 2.0, 2.0])
    unit_direction = direction / torch.linalg.vector_norm(direction)
    expected = torch.eye(3) - torch.outer(unit_direction, unit_direction)
    rotation_axis = torch.tensor([-2.0, 1.0, 2.0])
    unit_rotation_axis = rotation_axis / torch.linalg.vector_norm(rotation_axis)
    expected_rotation = torch.eye(3) - torch.outer(unit_rotation_axis, unit_rotation_axis)

    criteria = ToolPoseCriteria.linear_motion(
        direction=direction,
        rotation_axis=rotation_axis,
        error_frame=PoseErrorFrame.WORLD,
        device_cfg=device_cfg,
    )
    torch.testing.assert_close(criteria.path_position_projector, expected)
    torch.testing.assert_close(criteria.path_rotation_projector, expected_rotation)
    torch.testing.assert_close(criteria.terminal_position_projector, _identity(device_cfg))
    assert criteria.error_frame.item() == PoseErrorFrame.WORLD

    fixed_orientation = ToolPoseCriteria.linear_motion(
        direction=direction,
        device_cfg=device_cfg,
    )
    torch.testing.assert_close(fixed_orientation.path_rotation_projector, _identity(device_cfg))

    legacy = ToolPoseCriteria.linear_motion(
        axis="z", project_distance_to_goal=True, device_cfg=device_cfg
    )
    torch.testing.assert_close(
        legacy.path_position_projector, torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    )
    assert legacy.error_frame.item() == PoseErrorFrame.GOAL

    with pytest.raises(ValueError, match="nonzero"):
        ToolPoseCriteria.linear_motion(direction=[0.0, 0.0, 0.0], device_cfg=device_cfg)
    with pytest.raises(ValueError, match="nonzero"):
        ToolPoseCriteria.linear_motion(
            direction=[0.0, 0.0, 1.0],
            rotation_axis=[0.0, 0.0, 0.0],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="cannot both"):
        ToolPoseCriteria.linear_motion(axis="z", direction=[0.0, 0.0, 1.0], device_cfg=device_cfg)


def test_clone_copy_and_path_constraint_preserve_canonical_fields(device_cfg: DeviceCfg):
    source = ToolPoseCriteria.linear_motion(
        direction=[1.0, 1.0, 0.0],
        error_frame=PoseErrorFrame.GOAL,
        non_terminal_scale=0.25,
        device_cfg=device_cfg,
    )
    cloned = source.clone()
    assert (
        cloned.terminal_position_projector.data_ptr()
        != source.terminal_position_projector.data_ptr()
    )

    target = ToolPoseCriteria(device_cfg=device_cfg)
    pointers = {name: getattr(target, name).data_ptr() for name in target.tensor_field_names()}
    target.copy_(source)
    for name, pointer in pointers.items():
        assert getattr(target, name).data_ptr() == pointer
        torch.testing.assert_close(getattr(target, name), getattr(source, name))

    path = source.as_path_constraint()
    torch.testing.assert_close(path.terminal_position_projector, source.path_position_projector)
    torch.testing.assert_close(path.terminal_rotation_projector, source.path_rotation_projector)
    torch.testing.assert_close(path.terminal_position_axis_scale, source.path_position_axis_scale)
    torch.testing.assert_close(path.terminal_rotation_axis_scale, source.path_rotation_axis_scale)
    torch.testing.assert_close(path.terminal_tolerance, source.path_tolerance)


def test_stacked_criteria_shapes_clone_and_in_place_update(device_cfg: DeviceCfg):
    criteria = {
        "a": ToolPoseCriteria(device_cfg=device_cfg),
        "b": ToolPoseCriteria.track_position(device_cfg=device_cfg),
    }
    stacked = StackedToolPoseCriteria.from_tool_pose_criteria(criteria)

    assert stacked.terminal_position_projector.shape == (2, 3, 3)
    assert stacked.path_rotation_projector.shape == (2, 3, 3)
    assert stacked.terminal_position_axis_scale.shape == (2, 3)
    assert stacked.path_rotation_axis_scale.shape == (2, 3)
    assert stacked.terminal_tolerance.shape == (2, 2)
    assert stacked.error_frame.shape == (2,)

    pointers = {name: getattr(stacked, name).data_ptr() for name in stacked.tensor_field_names()}
    update = ToolPoseCriteria.linear_motion(
        direction=[0.0, 1.0, 1.0], error_frame=PoseErrorFrame.GOAL, device_cfg=device_cfg
    )
    stacked.update_tool_pose_criteria({"b": update})
    for name, pointer in pointers.items():
        assert getattr(stacked, name).data_ptr() == pointer
    torch.testing.assert_close(stacked.path_position_projector[1], update.path_position_projector)
    assert stacked.error_frame[1].item() == PoseErrorFrame.GOAL

    cloned = stacked.clone()
    assert cloned.tool_frames == stacked.tool_frames
    assert cloned.tool_frames is not stacked.tool_frames
    assert cloned.path_position_projector.data_ptr() != stacked.path_position_projector.data_ptr()

    with pytest.raises(ValueError, match="not found"):
        stacked.update_tool_pose_criteria({"missing": update})
