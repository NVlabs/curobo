# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for tool-pose cost configuration ownership."""

# Standard Library
from dataclasses import fields

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame, ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg() -> DeviceCfg:
    return DeviceCfg(device=torch.device("cpu"))


def test_default_init_without_tool_frames(device_cfg: DeviceCfg) -> None:
    cfg = ToolPoseCostCfg(weight=[1.0, 2.0], device_cfg=device_cfg)

    assert cfg.class_type is ToolPoseCost
    assert cfg.tool_frames is None
    assert cfg.num_links == 0
    assert cfg.tool_pose_criteria == {}
    assert isinstance(cfg.default_tool_pose_criteria, ToolPoseCriteria)

    stored_names = {field.name for field in fields(cfg)}
    assert "_terminal_pose_axes_weight_factor" not in stored_names
    assert "_non_terminal_pose_axes_weight_factor" not in stored_names
    assert "_terminal_pose_convergence_tolerance" not in stored_names
    assert "_non_terminal_pose_convergence_tolerance" not in stored_names
    assert "_project_distance_to_goal" not in stored_names
    assert "use_lie_group" not in stored_names
    assert "rotation_method" not in stored_names


def test_set_tool_frames_preserves_explicit_criteria(device_cfg: DeviceCfg) -> None:
    explicit = ToolPoseCriteria.track_position(device_cfg=device_cfg)
    default = ToolPoseCriteria.disabled(device_cfg=device_cfg)
    cfg = ToolPoseCostCfg(
        weight=[1.0, 1.0],
        tool_frames=["explicit", "new"],
        default_tool_pose_criteria=default,
        tool_pose_criteria={"explicit": explicit},
        device_cfg=device_cfg,
    )

    assert cfg.tool_pose_criteria["explicit"] is explicit
    assert cfg.tool_pose_criteria["new"] is not default
    torch.testing.assert_close(
        cfg.tool_pose_criteria["new"].terminal_position_projector,
        default.terminal_position_projector,
    )

    existing_new = cfg.tool_pose_criteria["new"]
    supplied_frames = ["new", "later"]
    cfg.set_tool_frames(supplied_frames)
    supplied_frames[0] = "mutated"

    assert cfg.tool_frames == ["new", "later"]
    assert cfg.tool_pose_criteria["new"] is existing_new
    assert cfg.tool_pose_criteria["later"] is not default
    assert "explicit" not in cfg.tool_pose_criteria


def test_set_tool_frames_after_deferred_configuration_preserves_explicit_criteria(
    device_cfg: DeviceCfg,
) -> None:
    explicit = ToolPoseCriteria.track_orientation(device_cfg=device_cfg)
    cfg = ToolPoseCostCfg(
        weight=[1.0, 1.0],
        tool_pose_criteria={"tool": explicit},
        device_cfg=device_cfg,
    )

    assert cfg.tool_frames is None
    assert cfg.tool_pose_criteria["tool"] is explicit

    cfg.set_tool_frames(["tool", "other"])

    assert cfg.tool_pose_criteria["tool"] is explicit
    assert set(cfg.tool_pose_criteria) == {"tool", "other"}


def test_raw_default_criteria_uses_config_device_and_dtype() -> None:
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    cfg = ToolPoseCostCfg(
        weight=[1.0, 1.0],
        default_tool_pose_criteria={
            "terminal_tolerance": [0.1, 0.2],
            "device_cfg": DeviceCfg(device=torch.device("cpu"), dtype=torch.float64),
        },
        device_cfg=device_cfg,
    )

    assert isinstance(cfg.default_tool_pose_criteria, ToolPoseCriteria)
    assert cfg.default_tool_pose_criteria.device_cfg == device_cfg
    assert cfg.default_tool_pose_criteria.terminal_tolerance.dtype == torch.float32
    torch.testing.assert_close(
        cfg.default_tool_pose_criteria.terminal_tolerance,
        torch.tensor([0.1, 0.2], dtype=torch.float32),
    )


def test_non_float32_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="supports only torch.float32"):
        ToolPoseCostCfg(
            weight=[1.0, 1.0],
            device_cfg=DeviceCfg(device=torch.device("cpu"), dtype=torch.float64),
        )


def test_invalid_default_criteria_type_is_rejected(device_cfg: DeviceCfg) -> None:
    with pytest.raises(ValueError, match="default_tool_pose_criteria"):
        ToolPoseCostCfg(
            weight=[1.0, 1.0],
            default_tool_pose_criteria=[1.0, 2.0],
            device_cfg=device_cfg,
        )


def test_criteria_device_cfg_must_match_cost_config(device_cfg: DeviceCfg) -> None:
    mismatched_device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float64)
    with pytest.raises(ValueError, match="must use the ToolPoseCostCfg device_cfg"):
        ToolPoseCostCfg(
            weight=[1.0, 1.0],
            default_tool_pose_criteria=ToolPoseCriteria(device_cfg=mismatched_device_cfg),
            device_cfg=device_cfg,
        )

    with pytest.raises(ValueError, match=r"tool_pose_criteria\['tool'\]"):
        ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_pose_criteria={"tool": ToolPoseCriteria(device_cfg=mismatched_device_cfg)},
            device_cfg=device_cfg,
        )


def test_legacy_private_inputs_convert_to_the_single_default(device_cfg: DeviceCfg) -> None:
    cfg = ToolPoseCostCfg(
        weight=[1.0, 1.0],
        _terminal_pose_axes_weight_factor=[2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        _non_terminal_pose_axes_weight_factor=[0.0, 5.0, 0.0, 6.0, 0.0, 7.0],
        _terminal_pose_convergence_tolerance=[0.01, 0.02],
        _non_terminal_pose_convergence_tolerance=[0.03, 0.04],
        _project_distance_to_goal=True,
        use_lie_group=True,
        device_cfg=device_cfg,
    )
    criteria = cfg.default_tool_pose_criteria

    torch.testing.assert_close(
        criteria.terminal_position_projector,
        torch.diag(torch.tensor([1.0, 0.0, 1.0])),
    )
    torch.testing.assert_close(
        criteria.terminal_position_axis_scale, torch.tensor([2.0, 1.0, 3.0])
    )
    torch.testing.assert_close(criteria.path_tolerance, torch.tensor([0.03, 0.04]))
    assert criteria.error_frame.item() == PoseErrorFrame.GOAL
    assert "use_lie_group" not in cfg.__dict__


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"_terminal_pose_axes_weight_factor": [1.0] * 6},
        {"_non_terminal_pose_axes_weight_factor": [1.0] * 6},
        {"_terminal_pose_convergence_tolerance": [0.0, 0.0]},
        {"_non_terminal_pose_convergence_tolerance": [0.0, 0.0]},
        {"_project_distance_to_goal": False},
    ],
)
def test_explicit_default_conflicts_with_legacy_private_inputs(
    device_cfg: DeviceCfg,
    legacy_kwargs: dict,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        ToolPoseCostCfg(
            weight=[1.0, 1.0],
            default_tool_pose_criteria=ToolPoseCriteria(device_cfg=device_cfg),
            device_cfg=device_cfg,
            **legacy_kwargs,
        )


def test_use_lie_group_is_accepted_but_not_stored(device_cfg: DeviceCfg) -> None:
    disabled = ToolPoseCostCfg(weight=[1.0, 1.0], use_lie_group=False, device_cfg=device_cfg)
    enabled = ToolPoseCostCfg(weight=[1.0, 1.0], use_lie_group=True, device_cfg=device_cfg)

    assert "use_lie_group" not in disabled.__dict__
    assert "use_lie_group" not in enabled.__dict__
    torch.testing.assert_close(
        disabled.default_tool_pose_criteria.terminal_rotation_projector,
        enabled.default_tool_pose_criteria.terminal_rotation_projector,
    )


def test_clone_deep_copies_all_owned_state_and_inherited_fields(device_cfg: DeviceCfg) -> None:
    default = ToolPoseCriteria.disabled(device_cfg=device_cfg)
    explicit = ToolPoseCriteria.track_position(device_cfg=device_cfg)
    cfg = ToolPoseCostCfg(
        weight=[3.0, 4.0],
        class_type=ToolPoseCost,
        device_cfg=device_cfg,
        convert_to_binary=True,
        use_grad_input=True,
        tool_frames=["tool"],
        default_tool_pose_criteria=default,
        tool_pose_criteria={"tool": explicit},
    )

    cloned = cfg.clone()

    assert cloned is not cfg
    assert cloned.class_type is cfg.class_type
    assert cloned.device_cfg == cfg.device_cfg
    assert cloned.convert_to_binary is True
    assert cloned.use_grad_input is True
    assert cloned.tool_frames == cfg.tool_frames
    assert cloned.tool_frames is not cfg.tool_frames
    assert cloned.weight.data_ptr() != cfg.weight.data_ptr()
    assert cloned.default_tool_pose_criteria is not cfg.default_tool_pose_criteria
    assert cloned.tool_pose_criteria["tool"] is not cfg.tool_pose_criteria["tool"]

    cloned.weight.zero_()
    cloned.default_tool_pose_criteria.terminal_position_projector.fill_(1.0)
    cloned.tool_pose_criteria["tool"].terminal_position_projector.zero_()
    assert torch.any(cfg.weight != 0.0)
    assert not torch.equal(
        cloned.default_tool_pose_criteria.terminal_position_projector,
        cfg.default_tool_pose_criteria.terminal_position_projector,
    )
    assert not torch.equal(
        cloned.tool_pose_criteria["tool"].terminal_position_projector,
        cfg.tool_pose_criteria["tool"].terminal_position_projector,
    )


def test_clone_handles_none_tool_frames(device_cfg: DeviceCfg) -> None:
    explicit = ToolPoseCriteria.track_position(device_cfg=device_cfg)
    cfg = ToolPoseCostCfg(
        weight=[1.0, 1.0],
        tool_frames=None,
        tool_pose_criteria={"future": explicit},
        device_cfg=device_cfg,
    )

    cloned = cfg.clone()

    assert cloned.tool_frames is None
    assert cloned.num_links == 0
    assert cloned.tool_pose_criteria["future"] is not explicit
    torch.testing.assert_close(
        cloned.tool_pose_criteria["future"].terminal_position_projector,
        explicit.terminal_position_projector,
    )
