# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Lifecycle tests for motion planners: destruction and interpreter shutdown."""

# Standard Library
import os
import subprocess
import sys
import textwrap

# Third Party
import pytest

# cuRobo
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner


@pytest.mark.parametrize("planner_class", [MotionPlanner, BatchMotionPlanner])
def test_destroy_is_noop_when_init_did_not_run(planner_class):
    planner = object.__new__(planner_class)
    planner.destroy()
    planner.__del__()


_SHUTDOWN_SCRIPT = textwrap.dedent(
    """
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.batch_motion_planner import BatchMotionPlanner

    planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml"))
    batch_planner = BatchMotionPlanner(
        MotionPlannerCfg.create(robot="franka.yml", max_batch_size=2)
    )
    print("planners alive at exit")
    """
)


def test_planners_alive_at_interpreter_exit_do_not_raise():
    env = dict(os.environ, CUROBO_TORCH_COMPILE_DISABLE="1")
    result = subprocess.run(
        [sys.executable, "-c", _SHUTDOWN_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "planners alive at exit" in result.stdout
    assert "Exception ignored" not in result.stderr, result.stderr
