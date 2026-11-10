# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for relative tool-transform criteria and configuration ownership."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_relative_transform import ToolRelativeTransformCost
from curobo._src.cost.cost_tool_relative_transform_cfg import ToolRelativeTransformCostCfg
from curobo._src.cost.criteria_tool_relative_transform import (
    StackedToolRelativeTransformCriteria,
    ToolRelativeTransformCriteria,
)
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg() -> DeviceCfg:
    """Return a CPU float32 device configuration."""
    return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


def test_criteria_validate_projectors_scales_and_tolerance(device_cfg: DeviceCfg) -> None:
    """Reject invalid canonical residual policy tensors."""
    with pytest.raises(ValueError, match="symmetric"):
        ToolRelativeTransformCriteria(
            position_projector=[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="idempotent"):
        ToolRelativeTransformCriteria(
            rotation_projector=torch.eye(3) * 0.5,
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        ToolRelativeTransformCriteria(
            position_axis_scale=[1.0, 0.0, 1.0],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ToolRelativeTransformCriteria(tolerance=[0.0, -0.1], device_cfg=device_cfg)


def test_criteria_factories_and_copy_are_canonical(device_cfg: DeviceCfg) -> None:
    """Factories use zero projectors for disabled residual components."""
    rigid = ToolRelativeTransformCriteria.rigid(device_cfg=device_cfg)
    position = ToolRelativeTransformCriteria.track_position(device_cfg=device_cfg)
    orientation = ToolRelativeTransformCriteria.track_orientation(device_cfg=device_cfg)
    disabled = ToolRelativeTransformCriteria.disabled(device_cfg=device_cfg)

    torch.testing.assert_close(rigid.position_projector, torch.eye(3))
    torch.testing.assert_close(rigid.rotation_projector, torch.eye(3))
    torch.testing.assert_close(position.rotation_projector, torch.zeros((3, 3)))
    torch.testing.assert_close(orientation.position_projector, torch.zeros((3, 3)))
    assert disabled.enabled is False

    cloned = rigid.clone()
    cloned.position_projector.zero_()
    assert torch.any(rigid.position_projector != 0.0)
    disabled.copy_(orientation)
    torch.testing.assert_close(disabled.rotation_projector, torch.eye(3))


def test_stacked_criteria_update_refreshes_derived_enable_bit(device_cfg: DeviceCfg) -> None:
    """In-place updates preserve buffers and refresh the derived enable cache."""
    stacked = StackedToolRelativeTransformCriteria.from_criteria(
        {
            "reference": ToolRelativeTransformCriteria.disabled(device_cfg=device_cfg),
            "tool": ToolRelativeTransformCriteria.track_position(device_cfg=device_cfg),
        }
    )
    projector_address = stacked.position_projector.data_ptr()
    assert stacked.relation_enabled.tolist() == [0, 1]

    stacked.update_criteria(
        {"tool": ToolRelativeTransformCriteria.disabled(device_cfg=device_cfg)}
    )

    assert stacked.position_projector.data_ptr() == projector_address
    assert stacked.relation_enabled.tolist() == [0, 0]


def test_config_owns_fixed_reference_and_disables_its_slot(device_cfg: DeviceCfg) -> None:
    """The fixed reference must exist and cannot carry its own relation."""
    config = ToolRelativeTransformCostCfg(
        weight=[2.0, 3.0],
        reference_tool_frame="reference",
        tool_frames=["reference", "tool", "free"],
        tool_relative_transform_criteria={
            "tool": ToolRelativeTransformCriteria.rigid(device_cfg=device_cfg),
        },
        device_cfg=device_cfg,
    )

    assert config.class_type is ToolRelativeTransformCost
    assert config.reference_link_index == 0
    assert config.num_links == 3
    assert not config.tool_relative_transform_criteria["reference"].enabled
    assert not config.tool_relative_transform_criteria["free"].enabled

    with pytest.raises(ValueError, match="reference_tool_frame"):
        ToolRelativeTransformCostCfg(
            weight=[1.0, 1.0],
            reference_tool_frame="missing",
            tool_frames=["reference", "tool"],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="reference slot must be disabled"):
        ToolRelativeTransformCostCfg(
            weight=[1.0, 1.0],
            reference_tool_frame="reference",
            tool_frames=["reference", "tool"],
            tool_relative_transform_criteria={
                "reference": ToolRelativeTransformCriteria.rigid(device_cfg=device_cfg),
            },
            device_cfg=device_cfg,
        )


def test_config_rejects_unsupported_kernel_contracts(device_cfg: DeviceCfg) -> None:
    """The fused analytic-gradient kernel is float32 and unit-upstream only."""
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        ToolRelativeTransformCostCfg(
            weight=[1.0],
            reference_tool_frame="reference",
            tool_frames=["reference", "tool"],
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="use_grad_input=False"):
        ToolRelativeTransformCostCfg(
            weight=[1.0, 1.0],
            reference_tool_frame="reference",
            tool_frames=["reference", "tool"],
            use_grad_input=True,
            device_cfg=device_cfg,
        )
    with pytest.raises(ValueError, match="float32"):
        ToolRelativeTransformCostCfg(
            weight=[1.0, 1.0],
            reference_tool_frame="reference",
            tool_frames=["reference", "tool"],
            device_cfg=DeviceCfg(device=torch.device("cpu"), dtype=torch.float64),
        )


def test_config_clone_deep_copies_owned_criteria(device_cfg: DeviceCfg) -> None:
    """Cloning preserves configuration while separating mutable tensors."""
    config = ToolRelativeTransformCostCfg(
        weight=[2.0, 3.0],
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        tool_relative_transform_criteria={
            "tool": ToolRelativeTransformCriteria.rigid(device_cfg=device_cfg),
        },
        device_cfg=device_cfg,
    )

    cloned = config.clone()

    assert cloned.reference_tool_frame == config.reference_tool_frame
    assert cloned.tool_frames == config.tool_frames
    assert cloned.tool_frames is not config.tool_frames
    assert cloned.weight.data_ptr() != config.weight.data_ptr()
    assert (
        cloned.tool_relative_transform_criteria["tool"].position_projector.data_ptr()
        != config.tool_relative_transform_criteria["tool"].position_projector.data_ptr()
    )
