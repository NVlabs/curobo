# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for end-effector velocity-limit cost configuration."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_velocity_limit import ToolVelocityLimitCost
from curobo._src.cost.cost_tool_velocity_limit_cfg import ToolVelocityLimitCostCfg
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg() -> DeviceCfg:
    return DeviceCfg(device=torch.device("cpu"))


def test_config_owns_one_criteria_source(device_cfg: DeviceCfg) -> None:
    explicit = ToolVelocityLimitCriteria.limit(
        max_linear_speed=0.4,
        max_angular_speed=0.8,
        device_cfg=device_cfg,
    )
    default = ToolVelocityLimitCriteria.disabled(device_cfg=device_cfg)
    cfg = ToolVelocityLimitCostCfg(
        weight=[2.0, 3.0],
        tool_frames=["explicit", "disabled"],
        default_tool_velocity_limit_criteria=default,
        tool_velocity_limit_criteria={"explicit": explicit},
        device_cfg=device_cfg,
        use_grad_input=True,
    )

    assert cfg.class_type is ToolVelocityLimitCost
    assert cfg.num_links == 2
    assert cfg.tool_velocity_limit_criteria["explicit"] is explicit
    assert cfg.tool_velocity_limit_criteria["disabled"] is not default
    cfg.set_tool_frames(["disabled", "new"])
    assert list(cfg.tool_velocity_limit_criteria) == ["disabled", "new"]

    cloned = cfg.clone()
    assert cloned.tool_frames == cfg.tool_frames
    assert cloned.tool_frames is not cfg.tool_frames
    assert cloned.weight.data_ptr() != cfg.weight.data_ptr()
    assert cloned.default_tool_velocity_limit_criteria is not default
    assert cloned.use_grad_input is True


def test_raw_default_criteria_uses_config_device(device_cfg: DeviceCfg) -> None:
    cfg = ToolVelocityLimitCostCfg(
        weight=[1.0, 1.0],
        default_tool_velocity_limit_criteria={
            "max_linear_speed": 0.25,
            "max_angular_speed": 0.75,
        },
        device_cfg=device_cfg,
    )
    assert cfg.default_tool_velocity_limit_criteria.device_cfg == device_cfg
    assert cfg.default_tool_velocity_limit_criteria.max_linear_speed.item() == pytest.approx(0.25)


def test_raw_per_frame_criteria_uses_config_device(device_cfg: DeviceCfg) -> None:
    cfg = ToolVelocityLimitCostCfg(
        weight=[1.0, 1.0],
        tool_frames=["tool"],
        tool_velocity_limit_criteria={
            "tool": {
                "max_linear_speed": 0.3,
                "max_angular_speed": 0.7,
                "linear_projector": torch.eye(3),
                "angular_projector": torch.eye(3),
            }
        },
        device_cfg=device_cfg,
    )

    criteria = cfg.tool_velocity_limit_criteria["tool"]
    assert isinstance(criteria, ToolVelocityLimitCriteria)
    assert criteria.device_cfg == device_cfg
    assert criteria.max_angular_speed.item() == pytest.approx(0.7)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"weight": [1.0]}, "shape"),
        ({"weight": [1.0, -1.0]}, "nonnegative"),
        ({"weight": [1.0, 1.0], "tool_frames": ["a", "a"]}, "unique"),
    ],
)
def test_invalid_config_is_rejected(device_cfg: DeviceCfg, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ToolVelocityLimitCostCfg(device_cfg=device_cfg, **kwargs)


def test_criteria_device_must_match_config(device_cfg: DeviceCfg) -> None:
    different = DeviceCfg(device=torch.device("cpu"), dtype=torch.float64)
    with pytest.raises(ValueError, match="device_cfg"):
        ToolVelocityLimitCostCfg(
            weight=[1.0, 1.0],
            default_tool_velocity_limit_criteria=ToolVelocityLimitCriteria(device_cfg=different),
            device_cfg=device_cfg,
        )
