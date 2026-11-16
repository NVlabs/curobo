# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for projected end-effector velocity-limit criteria."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.criteria_tool_velocity_limit import (
    StackedToolVelocityLimitCriteria,
    ToolVelocityLimitCriteria,
)
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg() -> DeviceCfg:
    return DeviceCfg(device=torch.device("cpu"))


def test_default_criteria_is_disabled(device_cfg: DeviceCfg) -> None:
    criteria = ToolVelocityLimitCriteria(device_cfg=device_cfg)

    torch.testing.assert_close(criteria.linear_projector, torch.zeros(3, 3))
    torch.testing.assert_close(criteria.angular_projector, torch.zeros(3, 3))
    torch.testing.assert_close(criteria.linear_axis_scale, torch.ones(3))
    torch.testing.assert_close(criteria.angular_axis_scale, torch.ones(3))
    assert criteria.max_linear_speed.item() == 0.0
    assert criteria.max_angular_speed.item() == 0.0


def test_limit_factory_and_validation(device_cfg: DeviceCfg) -> None:
    linear_projector = torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    criteria = ToolVelocityLimitCriteria.limit(
        max_linear_speed=0.4,
        max_angular_speed=1.2,
        linear_projector=linear_projector,
        angular_axis_scale=[1.0, 2.0, 3.0],
        device_cfg=device_cfg,
    )

    torch.testing.assert_close(criteria.linear_projector, linear_projector)
    torch.testing.assert_close(criteria.angular_projector, torch.eye(3))
    torch.testing.assert_close(criteria.angular_axis_scale, torch.tensor([1.0, 2.0, 3.0]))
    assert criteria.max_linear_speed.item() == pytest.approx(0.4)
    assert criteria.max_angular_speed.item() == pytest.approx(1.2)

    with pytest.raises(ValueError, match="nonnegative"):
        ToolVelocityLimitCriteria(max_linear_speed=-0.1, device_cfg=device_cfg)
    with pytest.raises(ValueError, match="finite"):
        ToolVelocityLimitCriteria(max_angular_speed=float("inf"), device_cfg=device_cfg)
    with pytest.raises(ValueError, match="symmetric"):
        ToolVelocityLimitCriteria(
            linear_projector=[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        ToolVelocityLimitCriteria(linear_axis_scale=[1.0, 0.0, 1.0], device_cfg=device_cfg)


def test_clone_copy_and_stacked_update_preserve_addresses(device_cfg: DeviceCfg) -> None:
    first = ToolVelocityLimitCriteria.limit(
        max_linear_speed=0.5,
        max_angular_speed=1.0,
        device_cfg=device_cfg,
    )
    second = ToolVelocityLimitCriteria.disabled(device_cfg=device_cfg)
    stacked = StackedToolVelocityLimitCriteria.from_tool_velocity_limit_criteria(
        {"first": first, "second": second}
    )

    assert stacked.linear_projector.shape == (2, 3, 3)
    assert stacked.angular_axis_scale.shape == (2, 3)
    assert stacked.max_linear_speed.shape == (2,)
    pointers = {name: getattr(stacked, name).data_ptr() for name in stacked.tensor_field_names()}

    update = ToolVelocityLimitCriteria.limit(
        max_linear_speed=0.2,
        max_angular_speed=0.3,
        linear_axis_scale=[2.0, 1.0, 1.0],
        device_cfg=device_cfg,
    )
    stacked.update_tool_velocity_limit_criteria({"second": update})
    for name, pointer in pointers.items():
        assert getattr(stacked, name).data_ptr() == pointer
    torch.testing.assert_close(stacked.linear_axis_scale[1], update.linear_axis_scale)
    torch.testing.assert_close(stacked.max_angular_speed[1], update.max_angular_speed)

    cloned = stacked.clone()
    assert cloned.tool_frames == stacked.tool_frames
    assert cloned.linear_projector.data_ptr() != stacked.linear_projector.data_ptr()

    target = ToolVelocityLimitCriteria(device_cfg=device_cfg)
    target_pointers = {
        name: getattr(target, name).data_ptr() for name in target.tensor_field_names()
    }
    target.copy_(first)
    for name, pointer in target_pointers.items():
        assert getattr(target, name).data_ptr() == pointer
        torch.testing.assert_close(getattr(target, name), getattr(first, name))
