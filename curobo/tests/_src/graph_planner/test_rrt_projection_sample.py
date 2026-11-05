# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optional collision-inequality sample projection."""

from __future__ import annotations

# Standard Library
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.config_graph_planner_resources import GraphPlannerResourcesCfg
from curobo._src.graph_planner.rrt import (
    BiRRTPlanner,
    RRTPlannerResources,
)
from curobo._src.graph_planner.rrt.rrt_projection_sample import (
    RRTSampleProjectionConfig,
    RRTSampleProjector,
    _strict_constraint_residuals,
)


class _EnvironmentAwareRollout:
    """Small differentiable inequality rollout used to audit environment IDs."""

    def __init__(self) -> None:
        self.batch_size = None
        self.env_calls = []

    @staticmethod
    def _thresholds(actions, env_query_idx):
        if env_query_idx is None:
            return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        return env_query_idx.to(dtype=actions.dtype)

    def compute_metrics_from_action(self, actions, **kwargs):
        self.batch_size = actions.shape[0]
        env_query_idx = kwargs.get("idxs_env_query")
        self.env_calls.append(None if env_query_idx is None else env_query_idx.detach().clone())
        threshold = self._thresholds(actions, env_query_idx)
        inequality = actions[:, 0, 0] - threshold
        values = [inequality]
        return SimpleNamespace(
            actions=actions,
            costs_and_constraints=SimpleNamespace(
                constraints=SimpleNamespace(values=values),
                hybrid_costs_constraints=SimpleNamespace(values=[]),
            ),
        )

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size


class _EnvironmentAwareResources:
    action_dim = 2
    environment_capacity = 2

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.action_bound_lows = torch.tensor([-10.0, -10.0], device=device)
        self.action_bound_highs = torch.tensor([10.0, 10.0], device=device)
        self.auxiliary_rollout = _EnvironmentAwareRollout()
        self.feasibility_env_calls = []

    def check_samples_feasibility(self, samples, env_query_idx=None):
        self.feasibility_env_calls.append(
            None if env_query_idx is None else env_query_idx.detach().clone()
        )
        threshold = _EnvironmentAwareRollout._thresholds(samples, env_query_idx)
        return samples[:, 0] <= threshold


class _ConstantViolationRollout:
    """Constraint rollout with a positive value and identically zero gradient."""

    def __init__(self) -> None:
        self.batch_size = None

    def compute_metrics_from_action(self, actions, **kwargs):
        self.batch_size = actions.shape[0]
        constant_violation = actions[:, 0, 0] * 0.0 + 1.0
        return SimpleNamespace(
            actions=actions,
            costs_and_constraints=SimpleNamespace(
                constraints=SimpleNamespace(values=[constant_violation]),
                hybrid_costs_constraints=SimpleNamespace(values=[]),
            ),
        )

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size


class _ConstantViolationResources:
    action_dim = 2

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.action_bound_lows = torch.tensor([-10.0, -10.0], device=device)
        self.action_bound_highs = torch.tensor([10.0, 10.0], device=device)
        self.auxiliary_rollout = _ConstantViolationRollout()

    def check_samples_feasibility(self, samples, env_query_idx=None):
        return torch.zeros(samples.shape[0], device=samples.device, dtype=torch.bool)


@pytest.fixture(scope="module")
def resources() -> RRTPlannerResources:
    """Create real CUDA collision resources for integration and capture tests."""
    return RRTPlannerResources(
        GraphPlannerResourcesCfg.create(robot="franka.yml", scene_model="collision_test.yml")
    )


def test_projection_config_is_frozen_and_lbfgs_only():
    config = RRTSampleProjectionConfig()

    assert config.iterations == 5
    assert config.projection_slots == 4
    assert config.history == 3
    assert config.line_search_scales == (0.0, 0.3, 1.0)
    assert config.step_scale == 0.05
    assert config.initial_step_scale == 0.1
    assert not hasattr(config, "method")
    assert not hasattr(config, "initial_damping")
    assert not hasattr(config, "max_displacement")
    with pytest.raises(FrozenInstanceError):
        config.iterations = 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"iterations": 0}, "iterations"),
        ({"iterations": True}, "iterations"),
        ({"iterations": 1.5}, "iterations"),
        ({"projection_slots": 0}, "projection_slots"),
        ({"projection_slots": True}, "projection_slots"),
        ({"history": 0}, "history"),
        ({"history": True}, "history"),
        ({"line_search_scales": []}, "line_search_scales"),
        ({"line_search_scales": ()}, "line_search_scales"),
        ({"line_search_scales": (-0.1, 0.0)}, "line_search_scales"),
        ({"line_search_scales": (float("nan"),)}, "line_search_scales"),
        ({"line_search_scales": (0.0, 0.5, 0.25)}, "line_search_scales"),
        ({"line_search_scales": (0.0, 0.25, 0.25)}, "line_search_scales"),
        ({"line_search_scales": (0.1, 0.5)}, "start at 0.0"),
        ({"line_search_scales": (0.0,)}, "positive probe"),
        ({"step_scale": 0.0}, "step_scale"),
        ({"initial_step_scale": float("nan")}, "initial_step_scale"),
        ({"lbfgs_epsilon": 0.0}, "lbfgs_epsilon"),
        ({"use_cuda_graph": 1}, "use_cuda_graph"),
    ),
)
def test_projection_config_rejects_invalid_profiles(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RRTSampleProjectionConfig(**kwargs)


@pytest.mark.parametrize(
    "env_query_idx",
    (
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0], [1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([-1, 0], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
    ),
)
def test_projection_rejects_invalid_environment_ids(env_query_idx):
    resources = _EnvironmentAwareResources()
    samples = torch.zeros((2, resources.action_dim))

    with pytest.raises(ValueError, match="env_query_idx"):
        RRTSampleProjector(resources).project(samples, env_query_idx=env_query_idx)


@pytest.mark.parametrize(
    "eligible_mask",
    (
        torch.tensor([True, False, True]),
        torch.tensor([[True], [False]]),
        torch.tensor([0, 1], dtype=torch.int64),
    ),
)
def test_projection_rejects_invalid_eligible_masks(eligible_mask):
    resources = _EnvironmentAwareResources()
    samples = torch.zeros((2, resources.action_dim))

    with pytest.raises(ValueError, match="eligible_mask"):
        RRTSampleProjector(resources).project(
            samples,
            eligible_mask=eligible_mask,
        )


def test_projection_rejects_eligible_mask_on_another_device(resources: RRTPlannerResources):
    samples = resources.uniform_sampler.generate_action_samples(2, bounded=True)
    eligible_mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="eligible_mask"):
        RRTSampleProjector(resources).project(
            samples,
            eligible_mask=eligible_mask,
        )


def test_projection_routes_selected_samples_to_their_environments():
    device = torch.device("cuda")
    resources = _EnvironmentAwareResources(device)
    samples = torch.tensor(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 0.0],
        ],
        device=device,
    )
    env_query_idx = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    projector = RRTSampleProjector(
        resources,
        RRTSampleProjectionConfig(
            iterations=5,
            projection_slots=2,
        ),
    )

    result = projector.project(samples, env_query_idx=env_query_idx)

    assert len(resources.feasibility_env_calls) == 2
    torch.testing.assert_close(resources.feasibility_env_calls[0], env_query_idx)
    selected_invalid = result.violation_measured & ~result.initially_feasible
    assert int(selected_invalid.sum().item()) == 2
    assert int((~result.violation_measured).sum().item()) == 2
    selected_env = env_query_idx[selected_invalid]
    torch.testing.assert_close(
        resources.feasibility_env_calls[1].sort().values,
        selected_env.sort().values,
    )
    expanded_selected_env = selected_env.repeat_interleave(
        len(projector.config.line_search_scales) - 1
    )
    assert resources.auxiliary_rollout.env_calls
    for recorded_env in resources.auxiliary_rollout.env_calls:
        torch.testing.assert_close(
            recorded_env.sort().values,
            expanded_selected_env.sort().values,
        )

    unmeasured = ~result.violation_measured
    torch.testing.assert_close(result.samples[unmeasured], samples[unmeasured])
    torch.testing.assert_close(
        result.initial_violation[unmeasured],
        torch.zeros_like(result.initial_violation[unmeasured]),
    )
    torch.testing.assert_close(
        result.final_violation[unmeasured], torch.zeros_like(result.final_violation[unmeasured])
    )
    assert (
        result.final_violation[result.violation_measured].sum()
        < result.initial_violation[result.violation_measured].sum()
    )
    torch.testing.assert_close(
        result.displacement,
        torch.linalg.vector_norm(result.samples - samples, dim=-1),
    )


def test_projection_selects_only_eligible_invalid_samples():
    device = torch.device("cuda")
    resources = _EnvironmentAwareResources(device)
    samples = torch.tensor(
        [
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ],
        device=device,
    )
    env_query_idx = torch.zeros(4, dtype=torch.int32, device=device)
    eligible_mask = torch.tensor([False, True, False, True], device=device)
    projector = RRTSampleProjector(
        resources,
        RRTSampleProjectionConfig(projection_slots=4),
    )

    result = projector.project(
        samples,
        env_query_idx=env_query_idx,
        eligible_mask=eligible_mask,
    )

    torch.testing.assert_close(result.violation_measured, eligible_mask)
    torch.testing.assert_close(result.samples[~eligible_mask], samples[~eligible_mask])
    torch.testing.assert_close(
        result.feasible[~eligible_mask], result.initially_feasible[~eligible_mask]
    )
    assert len(resources.feasibility_env_calls) == 2
    assert resources.feasibility_env_calls[0].shape == (4,)
    assert resources.feasibility_env_calls[1].shape == (2,)


def test_projection_omitted_eligible_mask_matches_all_eligible():
    device = torch.device("cuda")
    samples = torch.tensor([[2.0, -1.0], [3.0, 4.0]], device=device)
    implicit_resources = _ConstantViolationResources(device)
    explicit_resources = _ConstantViolationResources(device)

    implicit = RRTSampleProjector(implicit_resources).project(samples)
    explicit = RRTSampleProjector(explicit_resources).project(
        samples,
        eligible_mask=torch.ones(samples.shape[0], dtype=torch.bool, device=device),
    )

    for name, expected in implicit.__dict__.items():
        torch.testing.assert_close(explicit.__dict__[name], expected)


def test_projection_returns_optimized_sample_when_exact_recheck_fails(monkeypatch):
    resources = _ConstantViolationResources()
    projector = RRTSampleProjector(
        resources,
        RRTSampleProjectionConfig(projection_slots=1),
    )
    samples = torch.tensor([[2.0, -1.0], [3.0, 4.0]])
    optimized = samples[0:1] - 0.5

    monkeypatch.setattr(projector, "_ensure_optimizer", lambda *_: None)
    monkeypatch.setattr(
        projector,
        "_project_optimization_tensors",
        lambda *_: (
            optimized,
            torch.ones(1),
            torch.full((1,), 0.5),
        ),
    )

    result = projector.project(samples)

    selected = result.violation_measured
    assert int(selected.sum().item()) == 1
    assert not torch.any(result.feasible[selected])
    torch.testing.assert_close(result.samples[selected], optimized)


def test_projection_merit_excludes_regular_and_hybrid_costs():
    actions = torch.tensor([[[1.0, 2.0]]])
    metrics = SimpleNamespace(
        actions=actions,
        costs_and_constraints=SimpleNamespace(
            costs=SimpleNamespace(values=[actions[..., 0] * 100.0]),
            constraints=SimpleNamespace(values=[actions[..., 0] * 2.0]),
            hybrid_costs_constraints=SimpleNamespace(values=[actions[..., 1] * 1000.0]),
        ),
    )

    residuals = _strict_constraint_residuals(metrics)

    torch.testing.assert_close(residuals, torch.tensor([[2.0]]))


def test_projection_with_no_constraint_gradient_is_a_noop():
    device = torch.device("cuda")
    resources = _ConstantViolationResources(device)
    samples = torch.tensor([[2.0, -1.0], [3.0, 4.0]], device=device)

    result = RRTSampleProjector(resources).project(samples)

    torch.testing.assert_close(result.samples, samples)
    torch.testing.assert_close(result.initial_violation, result.final_violation)
    assert torch.all(result.initial_violation == 1.0)
    assert torch.all(result.displacement == 0.0)
    assert not torch.any(result.feasible)
    assert not hasattr(result, "final_damping")


def test_projection_starts_fresh_and_enforces_joint_bounds():
    device = torch.device("cuda")
    resources = _EnvironmentAwareResources(device)
    samples = torch.tensor([[20.0, -20.0], [3.0, 4.0]], device=device)
    projector = RRTSampleProjector(resources)

    first = projector.project(samples)
    second = projector.project(samples)

    torch.testing.assert_close(second.samples, first.samples)
    torch.testing.assert_close(second.initial_violation, first.initial_violation)
    torch.testing.assert_close(second.final_violation, first.final_violation)
    assert torch.all(first.samples >= resources.action_bound_lows)
    assert torch.all(first.samples <= resources.action_bound_highs)
    torch.testing.assert_close(
        first.displacement,
        torch.linalg.vector_norm(first.samples - samples, dim=-1),
    )


def test_cuda_graph_mode_rejects_cpu_samples_before_feasibility_work():
    resources = _EnvironmentAwareResources()
    projector = RRTSampleProjector(
        resources,
        RRTSampleProjectionConfig(use_cuda_graph=True),
    )

    with pytest.raises(ValueError, match="CUDA samples"):
        projector.project(torch.zeros((2, resources.action_dim)))
    assert resources.feasibility_env_calls == []


def test_sample_projection_repairs_a_bounded_number_of_samples(resources: RRTPlannerResources):
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_action_samples(128, bounded=True)
    result = RRTSampleProjector(resources).project(samples)
    displacement = torch.linalg.vector_norm(result.samples - samples, dim=-1)
    measured = result.violation_measured

    assert torch.count_nonzero(~result.initially_feasible) > 4
    torch.testing.assert_close(
        result.samples[result.initially_feasible], samples[result.initially_feasible]
    )
    assert int(torch.count_nonzero(displacement > 1.0e-7).item()) <= 4
    assert not torch.any(result.initially_feasible & ~result.feasible)
    assert result.repaired_count > 0
    assert result.final_violation[measured].sum() < result.initial_violation[measured].sum()
    assert torch.all(result.final_violation[measured] <= result.initial_violation[measured])
    torch.testing.assert_close(result.displacement, displacement)
    assert torch.any(result.displacement > 0.0)
    assert torch.all(torch.isfinite(result.samples))
    assert int(measured.sum().item()) == 4
    assert torch.all(~result.initially_feasible[measured])


def test_cuda_graph_warmup_primes_one_borrowed_output_replay(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Consume first-replay graph setup outside latency-sensitive projection calls."""
    projector = RRTSampleProjector(
        resources,
        RRTSampleProjectionConfig(projection_slots=4, use_cuda_graph=True),
    )
    samples = torch.zeros(
        (8, resources.action_dim),
        device=resources.device_cfg.device,
        dtype=resources.device_cfg.dtype,
    )
    env_query_idx = torch.zeros(8, device=samples.device, dtype=torch.int32)
    events = []

    class FakeExecutor:
        def warmup(self, *inputs):
            events.append(("warmup", inputs))
            return self

        def replay(self, **kwargs):
            events.append(("replay", kwargs))

    executor = FakeExecutor()

    def fake_get_executor(projection_samples, projection_env, projection_goal):
        torch.testing.assert_close(projection_samples, samples[:4], rtol=0.0, atol=0.0)
        assert torch.equal(projection_env, env_query_idx[:4])
        assert projection_goal is None
        return executor

    monkeypatch.setattr(projector, "_get_cuda_graph_executor", fake_get_executor)

    projector.warmup_cuda_graph(samples, env_query_idx=env_query_idx)

    assert len(events) == 2
    assert events[0][0] == "warmup"
    assert len(events[0][1]) == 3
    torch.testing.assert_close(events[0][1][0], samples[:4], rtol=0.0, atol=0.0)
    assert torch.equal(events[0][1][1], env_query_idx[:4])
    assert torch.equal(events[0][1][2], torch.zeros_like(env_query_idx[:4]))
    assert events[1] == ("replay", {"clone_outputs": False})


def test_cuda_graph_sample_projection_matches_eager(resources: RRTPlannerResources):
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_action_samples(64, bounded=True)
    env_query_idx = torch.zeros(samples.shape[0], device=samples.device, dtype=torch.int32)
    eager = RRTSampleProjector(resources, RRTSampleProjectionConfig(iterations=5)).project(
        samples,
        env_query_idx=env_query_idx,
    )
    graph_projector = RRTSampleProjector(
        resources,
        RRTSampleProjectionConfig(iterations=5, use_cuda_graph=True),
    )
    graph_projector.warmup_cuda_graph(samples, env_query_idx=env_query_idx)
    captured = graph_projector.project(samples, env_query_idx=env_query_idx)

    for field_name, expected in eager.__dict__.items():
        torch.testing.assert_close(
            captured.__dict__[field_name],
            expected,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    assert graph_projector._cuda_graph_executor is not None
    graph_projector.reset_cuda_graph()


def test_projection_remains_a_lazy_optional_planner_call(resources: RRTPlannerResources):
    planner = BiRRTPlanner(resources)
    assert planner._sample_projector is None
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_action_samples(64, bounded=True)

    result = planner.project_samples(
        samples,
        RRTSampleProjectionConfig(
            iterations=5,
            projection_slots=4,
            use_cuda_graph=True,
        ),
    )

    assert planner._sample_projector is not None
    assert planner._sample_projector._cuda_graph_executor is not None
    assert result.samples.shape == samples.shape
    assert int(result.violation_measured.sum().item()) <= 4
