# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Lifecycle tests for motion planners: destruction and interpreter shutdown."""

# Standard Library
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Third Party
import pytest

# cuRobo
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner


_COMPONENTS = (
    ("ik_solver", "destroy"),
    ("trajopt_solver", "destroy"),
    ("graph_planner", "reset_cuda_graph"),
    ("graph_seed_fitter", "reset_cuda_graph"),
    ("_grasp_phase_planner", "destroy"),
)


@pytest.fixture(params=[MotionPlanner, BatchMotionPlanner], ids=["single", "batch"])
def planner_class(request: pytest.FixtureRequest) -> type[MotionPlanner] | type[BatchMotionPlanner]:
    """Exercise both planner lifecycle implementations."""
    return request.param


def test_destroy_is_noop_when_init_did_not_run(
    planner_class: type[MotionPlanner] | type[BatchMotionPlanner],
) -> None:
    """An object whose constructor never ran owns no resources to release."""
    planner = object.__new__(planner_class)
    planner.destroy()
    planner.__del__()


@pytest.mark.parametrize("num_components", range(len(_COMPONENTS)))
def test_destroy_after_partial_initialization(
    planner_class: type[MotionPlanner] | type[BatchMotionPlanner],
    num_components: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction failures release only the components that were created, once."""
    components = {}

    def initialize_until_failure(planner: MotionPlanner | BatchMotionPlanner) -> None:
        for name, cleanup in _COMPONENTS[:num_components]:
            components[name] = Mock(spec=[cleanup])
            setattr(planner, name, components[name])
        raise RuntimeError("injected initialization failure")

    monkeypatch.setattr(planner_class, "_initialize_components", initialize_until_failure)
    planner = object.__new__(planner_class)
    try:
        with pytest.raises(RuntimeError, match="injected initialization failure"):
            planner.__init__(SimpleNamespace(device_cfg=None))
        planner.destroy()
        planner.destroy()
        planner.__del__()
        for name, cleanup in _COMPONENTS[:num_components]:
            getattr(components[name], cleanup).assert_called_once_with()
        assert planner._destroyed
    finally:
        planner._destroyed = True


def test_destroy_releases_all_components_once(
    planner_class: type[MotionPlanner] | type[BatchMotionPlanner],
) -> None:
    """Explicit cleanup and normal finalization release each owned component once."""
    planner = object.__new__(planner_class)
    planner._destroyed = False
    components = {}
    for name, cleanup in _COMPONENTS:
        components[name] = Mock(spec=[cleanup])
        setattr(planner, name, components[name])
    planner.destroy()
    planner.destroy()
    planner.__del__()
    for name, cleanup in _COMPONENTS:
        getattr(components[name], cleanup).assert_called_once_with()


@pytest.mark.parametrize("finalizing", [False, True])
def test_del_only_skips_cleanup_during_finalization(
    planner_class: type[MotionPlanner] | type[BatchMotionPlanner], finalizing: bool
) -> None:
    """Finalizers clean up normally and avoid touching resources during shutdown."""
    destroy = Mock()
    with patch.object(sys, "is_finalizing", return_value=finalizing):
        planner_class.__del__(SimpleNamespace(destroy=destroy))
    assert destroy.call_count == (0 if finalizing else 1)


_SHUTDOWN_SCRIPT = textwrap.dedent(
    """
    import sys
    from typing import Callable
    from curobo.motion_planner import MotionPlanner
    from curobo.batch_motion_planner import BatchMotionPlanner

    class Component:
        def destroy(self, finalizing: Callable[[], bool] = sys.is_finalizing) -> None:
            if finalizing():
                raise RuntimeError("cleanup called during interpreter finalization")

    planners = []
    for planner_class in (MotionPlanner, BatchMotionPlanner):
        planner = object.__new__(planner_class)
        planner._destroyed = False
        planner.ik_solver = Component()
        planner.trajopt_solver = Component()
        planner.graph_planner = None
        planner._shutdown_cycle = planner
        planners.append(planner)
    print("planners alive at exit")
    """
)


def test_planners_alive_at_interpreter_exit_do_not_raise() -> None:
    """Real interpreter shutdown skips cleanup, including for cyclic references."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", _SHUTDOWN_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "planners alive at exit" in result.stdout
    assert "Exception ignored" not in result.stderr, result.stderr
