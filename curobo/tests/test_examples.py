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

EXAMPLES = [
    "curobo.examples.getting_started.forward_kinematics",
    "curobo.examples.getting_started.inverse_kinematics",
    "curobo.examples.getting_started.motion_planning",
    "curobo.examples.getting_started.reactive_control",
    "curobo.examples.getting_started.build_robot_model",
]


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
