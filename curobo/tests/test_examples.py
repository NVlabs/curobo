# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""End-to-end tests for getting_started examples.

Each example supports a ``--test`` flag that runs with assertions.
This file invokes them via subprocess so each gets a clean process.
"""

import subprocess
import sys

import pytest
import torch

from curobo.examples.getting_started.motion_planning import (
    _combine_grasp_phases,
    _prepare_interpolated_trajectory,
)
from curobo.types import JointState

EXAMPLES = [
    "curobo.examples.getting_started.forward_kinematics",
    "curobo.examples.getting_started.inverse_kinematics",
    "curobo.examples.getting_started.motion_planning",
    "curobo.examples.getting_started.reactive_control",
    "curobo.examples.getting_started.build_robot_model",
]


def test_prepare_interpolated_trajectory_normalizes_and_trims() -> None:
    """Select one seed and trim the interpolation backing buffer."""
    position = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    trajectory = JointState.from_position(position)
    trajectory.dt = torch.tensor([[0.1]], dtype=torch.float32)

    prepared = _prepare_interpolated_trajectory(
        trajectory,
        torch.tensor([[3]], dtype=torch.int32),
    )

    assert prepared.position.shape == (1, 3, 2)
    torch.testing.assert_close(prepared.position, position[:, 0, :3])
    assert prepared.dt is not None
    torch.testing.assert_close(prepared.dt, trajectory.dt)


def test_combine_grasp_phases_uses_each_dt() -> None:
    """Build continuous timestamps without repeating phase boundary states."""
    phases = []
    for label, horizon, dt in (
        ("Approach", 3, 0.1),
        ("Grasp", 4, 0.2),
        ("Lift", 2, 0.05),
    ):
        trajectory = JointState.from_position(torch.zeros(1, horizon, 2))
        trajectory.dt = torch.tensor([dt], dtype=torch.float32)
        phases.append((label, trajectory))

    positions, timestamps, boundaries, labels = _combine_grasp_phases(phases)

    assert positions.shape == (7, 2)
    torch.testing.assert_close(
        timestamps,
        torch.tensor([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85]),
    )
    assert boundaries == [0, 3, 6]
    assert labels == ["Approach", "Grasp", "Lift"]


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.rsplit(".", 1)[-1] for e in EXAMPLES])
def test_example(example):
    result = subprocess.run(
        [sys.executable, "-m", example, "--test"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{example} --test failed (exit {result.returncode})\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )
