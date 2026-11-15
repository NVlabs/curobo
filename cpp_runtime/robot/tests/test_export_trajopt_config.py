# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native dense-buffer defaults against the shared Python configuration."""

from pathlib import Path

import pytest
import torch

from cpp_runtime.robot.tools.export_robot_aot import build_trajopt_config
from curobo.types import ContentPath


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("num_knots", [16, 1024])
def test_default_dense_capacity_matches_python(num_knots: int) -> None:
    """Use 1024 samples for small splines and grow to cover larger layouts."""
    robot = Path(ContentPath(robot_config_file="franka.yml").get_robot_configuration_path())
    config, _, _ = build_trajopt_config(robot, max_seeds=4, num_knots=num_knots)

    assert config.interpolation_buffer_size == max(1024, config.minimum_interpolation_buffer_size)
    if num_knots == 16:
        assert config.interpolation_buffer_size == 1024
    else:
        assert config.interpolation_buffer_size > 1024


def test_explicit_dense_capacity_is_preserved() -> None:
    """Allow deployments to reserve a larger buffer explicitly."""
    robot = Path(ContentPath(robot_config_file="franka.yml").get_robot_configuration_path())
    config, _, _ = build_trajopt_config(robot, max_seeds=4, interpolation_buffer_size=2048)

    assert config.interpolation_buffer_size == 2048


def test_dense_capacity_below_layout_minimum_is_rejected() -> None:
    """Reject undersized native artifacts rather than truncating trajectories."""
    robot = Path(ContentPath(robot_config_file="franka.yml").get_robot_configuration_path())
    with pytest.raises(ValueError, match="interpolation buffer needs at least"):
        build_trajopt_config(robot, max_seeds=4, interpolation_buffer_size=2)
