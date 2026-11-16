# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for graph-backed rollout updates in SolverCore."""

from __future__ import annotations

# Standard Library
from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock

# Third Party
import pytest
import torch

# CuRobo
import curobo._src.runtime as curobo_runtime
import curobo._src.solver.solver_core as solver_core_module
from curobo._src.solver.solver_core import SolverCore


class _FakeRollout:
    """Record goal propagation calls."""

    def __init__(self) -> None:
        self.update_calls = 0

    def update_params(self, goal: object) -> None:
        self.update_calls += 1


class _FakeOptimizer:
    """Record optimizer goal propagation calls."""

    def __init__(self) -> None:
        self.update_calls = 0

    def update_rollout_params(self, goal: object) -> None:
        self.update_calls += 1


class _TensorRollout:
    """Copy a source tensor into stable rollout storage."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.value = torch.zeros_like(tensor)

    def update_params(self, goal: object) -> None:
        self.value.copy_(goal.value)


class _FakeGraphExecutor:
    """Execute a captured callback eagerly while recording graph lifecycle calls."""

    def __init__(
        self,
        capture_fn: Callable[..., None],
        capture_fn_kwargs: dict[str, object],
    ) -> None:
        self._capture_fn = capture_fn
        self._capture_fn_kwargs = capture_fn_kwargs
        self.call_count = 0
        self.reset_count = 0

    def __call__(self) -> None:
        self.call_count += 1
        self._capture_fn(**self._capture_fn_kwargs)

    def reset(self) -> None:
        self.reset_count += 1


def _make_core() -> SolverCore:
    core = SolverCore.__new__(SolverCore)
    core.config = SimpleNamespace(use_cuda_graph=True)
    core.device_cfg = SimpleNamespace(device=torch.device("cuda:0"))
    core.metrics_rollout = _FakeRollout()
    core.auxiliary_rollout = _FakeRollout()
    core.optimizer = _FakeOptimizer()
    core.optimizer_rollouts = []
    core.additional_metrics_rollouts = {}
    core._goal_buffer = object()
    core._rollout_update_executors = {}
    return core


@pytest.mark.parametrize("batched", [False, True])
def test_inertial_updates_include_all_transition_models(batched: bool) -> None:
    """Refresh optimization and metrics dynamics, including interpolated metrics."""
    core = _make_core()
    rollouts = [
        SimpleNamespace(transition_model=Mock(), metrics_transition_model=Mock())
        for _ in range(5)
    ]
    core.metrics_rollout, core.auxiliary_rollout = rollouts[:2]
    core.optimizer_rollouts = rollouts[2:4]
    core.additional_metrics_rollouts = {"interpolated": rollouts[4]}
    com = torch.tensor([0.01, 0.02, 0.1], dtype=torch.float32)
    inertia = torch.tensor([0.01, 0.02, 0.03, 0.0, 0.0, 0.0], dtype=torch.float32)
    properties = {"attached_object": {"mass": 1.0, "com": com, "inertia": inertia}}

    if batched:
        core.update_links_inertial(properties)
    else:
        core.update_link_inertial("attached_object", 1.0, com, inertia)

    for rollout in rollouts:
        for model in (rollout.transition_model, rollout.metrics_transition_model):
            if batched:
                model.update_links_inertial.assert_called_once_with(properties)
            else:
                model.update_link_inertial.assert_called_once_with(
                    "attached_object", 1.0, com, inertia
                )


def test_rollout_update_graphs_are_gated_by_runtime_flag(monkeypatch) -> None:
    """Use eager rollout updates when runtime checks are enabled."""
    core = _make_core()
    monkeypatch.setattr(curobo_runtime, "skip_checks", False)

    def fail_factory(**kwargs: object) -> None:
        raise AssertionError("rollout update graph should not be created")

    monkeypatch.setattr(
        solver_core_module,
        "create_graph_executor",
        fail_factory,
        raising=False,
    )

    core.update_rollout_params(
        core._goal_buffer,
        include_auxiliary_rollout=False,
        include_optimizer_rollouts=False,
    )

    assert core.metrics_rollout.update_calls == 1
    assert core.auxiliary_rollout.update_calls == 0
    assert core.optimizer.update_calls == 0
    assert core._rollout_update_executors == {}


def test_rollout_update_graphs_cache_independent_variants(monkeypatch) -> None:
    """Delegate global graph policy while caching each rollout-selection variant."""
    core = _make_core()
    executors = []
    monkeypatch.setattr(curobo_runtime, "skip_checks", True)
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)

    def create_fake_executor(
        *, capture_fn: Callable[..., None], **kwargs: object
    ) -> _FakeGraphExecutor:
        executor = _FakeGraphExecutor(capture_fn, {"graph_key": kwargs["graph_key"]})
        executors.append(executor)
        return executor

    monkeypatch.setattr(
        solver_core_module,
        "create_graph_executor",
        create_fake_executor,
        raising=False,
    )

    core.update_rollout_params(
        core._goal_buffer,
        include_auxiliary_rollout=False,
        include_optimizer_rollouts=False,
    )
    core.update_rollout_params(
        core._goal_buffer,
        include_auxiliary_rollout=False,
        include_optimizer_rollouts=True,
        include_metrics_rollout=False,
    )
    core.update_rollout_params(core._goal_buffer)
    core.update_rollout_params(core._goal_buffer)

    assert len(executors) == 3
    assert set(core._rollout_update_executors) == {
        (True, False, False),
        (False, False, True),
        (True, True, True),
    }
    assert executors[-1].call_count == 2
    assert core.metrics_rollout.update_calls == 3
    assert core.auxiliary_rollout.update_calls == 2
    assert core.optimizer.update_calls == 3

    core._reset_rollout_update_cuda_graphs()
    assert all(executor.reset_count == 1 for executor in executors)


def test_rollout_update_graph_requires_owned_goal_buffer(monkeypatch) -> None:
    """Keep checks active when updating from a non-owned goal registry."""
    core = _make_core()
    monkeypatch.setattr(curobo_runtime, "skip_checks", True)

    def fail_factory(**kwargs: object) -> None:
        raise AssertionError("foreign goal buffer must use the eager path")

    monkeypatch.setattr(
        solver_core_module,
        "create_graph_executor",
        fail_factory,
        raising=False,
    )

    core.update_rollout_params(
        object(),
        include_auxiliary_rollout=False,
        include_optimizer_rollouts=False,
    )

    assert core.metrics_rollout.update_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rollout_update_graph_replays_latest_goal_values(monkeypatch) -> None:
    """Replay captured copies from the stable goal buffer after its values change."""
    source = torch.tensor([1.0, 2.0], device="cuda:0", dtype=torch.float32)
    goal = SimpleNamespace(value=source)
    core = _make_core()
    core._goal_buffer = goal
    core.metrics_rollout = _TensorRollout(source)
    monkeypatch.setattr(curobo_runtime, "skip_checks", True)
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", True)

    core.update_rollout_params(
        goal,
        include_auxiliary_rollout=False,
        include_optimizer_rollouts=False,
    )
    source.add_(3.0)
    core.update_rollout_params(
        goal,
        include_auxiliary_rollout=False,
        include_optimizer_rollouts=False,
    )

    torch.testing.assert_close(core.metrics_rollout.value, source)
    assert core._rollout_update_executors[(True, False, False)].is_initialized
    core._reset_rollout_update_cuda_graphs()
