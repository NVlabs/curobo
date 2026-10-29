# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Torch trajectory-optimization result processor."""

from __future__ import annotations

# Standard Library
from dataclasses import replace

# Third Party
import pytest
import torch
from torch.profiler import ProfilerActivity, profile

# CuRobo
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics
from curobo._src.solver.processor_trajopt_result import (
    TorchTrajOptResultProcessor,
    TrajOptResultCandidate,
    TrajOptResultProcessorConfig,
)
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def processor() -> TorchTrajOptResultProcessor:
    """Create a CPU processor for shape-lifecycle tests."""
    device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    config = TrajOptResultProcessorConfig(
        device_cfg=device_cfg,
        action_horizon=32,
        horizon=32,
        action_dim=7,
        interpolation_steps=4,
        goalset_index_dim=1,
        goalset_index_dtype=torch.int32,
        joint_names=[f"joint_{index}" for index in range(7)],
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        minimum_trajectory_dt=0.01,
        maximum_trajectory_dt=0.15,
    )
    return TorchTrajOptResultProcessor(config, lambda solution: solution)


def test_update_batch_size_is_only_shape_initialization(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Shape-dependent workspaces are absent until explicitly initialized."""
    assert not processor.is_shape_initialized

    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=1)

    assert processor.is_shape_initialized
    assert processor.batch_size == 2
    assert processor.num_seeds == 3
    assert processor.best_dt.shape == (2, 3)


def test_update_batch_size_is_idempotent(processor: TorchTrajOptResultProcessor) -> None:
    """Updating with an unchanged shape reuses the existing workspace."""
    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=1)
    data_ptr = processor.best_dt.data_ptr()

    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=1)

    assert processor.best_dt.data_ptr() == data_ptr


def test_reset_shape_requires_explicit_reinitialization(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Resetting invalidates workspaces until update_batch_size is called again."""
    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=1)
    processor.reset_shape()

    assert not processor.is_shape_initialized
    with pytest.raises(ValueError, match="update_batch_size"):
        processor.reset(torch.ones((2, 3), dtype=torch.float32))


def test_reset_rejects_shape_mismatch(processor: TorchTrajOptResultProcessor) -> None:
    """Per-solve reset cannot silently resize processor workspaces."""
    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=1)

    with pytest.raises(ValueError, match="initial_dt.shape"):
        processor.reset(torch.ones((1, 3), dtype=torch.float32))


def test_shape_change_reallocates_workspace(processor: TorchTrajOptResultProcessor) -> None:
    """A shape change is applied only through update_batch_size."""
    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=1)
    old_best_dt = processor.best_dt

    processor.update_batch_size(batch_size=1, num_seeds=4, return_seeds=1)

    assert processor.best_dt is not old_best_dt
    assert processor.best_dt.shape == (1, 4)


def _make_candidate(dt: torch.Tensor, action_offset: float = 0.0) -> TrajOptResultCandidate:
    """Create a deterministic two-seed candidate for processor behavior tests."""
    num_problems = 2
    horizon = 32
    action_dim = 7
    actions = torch.zeros(
        (num_problems, horizon, action_dim),
        device=dt.device,
        dtype=torch.float32,
    )
    actions[1] = 1.0
    actions += action_offset
    joint_state = JointState(
        position=actions.clone(),
        velocity=torch.zeros_like(actions),
        acceleration=torch.zeros_like(actions),
        jerk=torch.zeros_like(actions),
        dt=dt,
    )
    convergence = CostCollection()
    convergence.add(
        torch.tensor([0.002, 0.003], device=dt.device, dtype=torch.float32)
        .view(2, 1, 1)
        .expand(-1, horizon, -1),
        "position_tolerance",
    )
    convergence.add(
        torch.tensor([0.01, 0.02], device=dt.device, dtype=torch.float32)
        .view(2, 1, 1)
        .expand(-1, horizon, -1),
        "orientation_tolerance",
    )
    convergence.add(
        torch.zeros((num_problems, horizon, 1), device=dt.device, dtype=torch.int32),
        "goalset_index",
    )
    constraints = CostCollection()
    constraints.add(
        torch.zeros((num_problems, horizon, 1), device=dt.device, dtype=torch.float32),
        "joint_constraint",
    )
    metrics = RolloutMetrics(
        actions=actions,
        costs_and_constraints=CostsAndConstraints(constraints=constraints),
        state=RobotState(joint_state=joint_state),
        feasible=torch.ones((num_problems, horizon), device=dt.device, dtype=torch.bool),
        convergence=convergence,
    )
    return TrajOptResultCandidate(
        metrics=metrics,
        interpolated_metrics=None,
        debug_info={},
    )


def _make_interpolation_output(
    processor: TorchTrajOptResultProcessor,
) -> tuple[JointState, torch.Tensor]:
    """Create dense all-seed output for the external interpolation boundary."""
    interpolation_input = processor.get_interpolation_input()
    dense_shape = (processor.batch_size, processor.num_seeds, 16, 7)
    seed_value = interpolation_input.knot[:, 0, 0].view(
        processor.batch_size, processor.num_seeds, 1, 1
    )
    position = seed_value.expand(dense_shape).clone()
    trajectory = JointState(
        position=position,
        velocity=torch.zeros_like(position),
        acceleration=torch.zeros_like(position),
        jerk=torch.zeros_like(position),
        dt=interpolation_input.dt.view(processor.batch_size, processor.num_seeds),
    )
    last_tstep = torch.full(
        (processor.batch_size, processor.num_seeds),
        15,
        device=position.device,
        dtype=torch.long,
    )
    return trajectory, last_tstep


def _finalize(
    processor: TorchTrajOptResultProcessor,
    return_seeds: int,
) -> TrajOptSolverResult:
    """Finalize using a transient interpolation result produced outside the processor."""
    interpolated_trajectory, interpolated_last_tstep = _make_interpolation_output(processor)
    return processor.finalize(
        return_seeds,
        interpolated_trajectory,
        interpolated_last_tstep,
    )


def test_update_and_finalize_create_trajopt_result(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """The Torch processor ranks candidates and returns the existing result contract."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=1)
    processor.reset(initial_dt)

    update = processor.update(_make_candidate(initial_dt.view(-1)))
    result = _finalize(processor, return_seeds=1)

    assert update.improved
    assert update.any_success
    assert isinstance(result, TrajOptSolverResult)
    assert result.success.shape == (1, 1)
    assert result.num_seeds == 1
    assert torch.all(result.solution == 1.0)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_timeopt_ranks_all_seeds_and_rechecks_failure(
    processor: TorchTrajOptResultProcessor, use_cuda_graph: bool
) -> None:
    """TimeOpt duration selects seeds and replay cannot reuse an earlier success."""
    device = torch.device("cuda:0" if use_cuda_graph else "cpu")
    processor = TorchTrajOptResultProcessor(
        replace(
            processor._config,
            device_cfg=DeviceCfg(device=device),
            use_cuda_graph=use_cuda_graph,
        ),
        lambda state: state,
    )
    dt = torch.tensor([[0.1, 0.05]], device=device)
    processor.update_batch_size(1, 2, 1)
    processor.reset(dt)
    candidate = _make_candidate(dt.flatten())
    # Original timing fails; the independent post-timeopt validation may repair it.
    candidate.metrics.feasible.zero_()
    processor.update(candidate)
    dense, counts = _make_interpolation_output(processor)
    success = torch.ones((1, 2), device=device, dtype=torch.bool)
    duration = torch.tensor([[0.8, 1.2]], device=device)
    result = processor.finalize(1, dense, counts, success, duration)
    assert result.success.item()
    assert result.seed_rank.item() == 0
    assert torch.all(result.solution == 0)
    torch.testing.assert_close(result.motion_time(), duration[:, :1])
    saved = result.clone()
    success[:, 0] = False
    result = processor.finalize(1, dense, counts, success, duration)
    assert result.success.item()
    assert result.seed_rank.item() == 1
    assert torch.all(result.solution == 1)
    torch.testing.assert_close(result.motion_time(), duration[:, 1:])
    result.copy_successful_solutions(saved)
    torch.testing.assert_close(result.motion_time(), saved.motion_time())
    success.zero_()
    result = processor.finalize(1, dense, counts, success, duration)
    assert not result.success.item()
    assert saved.success.item()
    torch.testing.assert_close(saved.motion_time(), duration[:, :1])
    processor.reset_cuda_graph()


def test_solution_fields_are_materialized_only_during_finalize(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Accumulation stores actions once and finalization creates result aliases."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=1)
    processor.reset(initial_dt)

    update = processor.update(_make_candidate(initial_dt.view(-1)))
    snapshot = processor._snapshot

    assert not hasattr(snapshot, "metrics")
    assert not hasattr(snapshot, "interpolated_metrics")
    assert not hasattr(snapshot, "interpolated_trajectory")
    assert not hasattr(snapshot, "interpolated_last_tstep")
    assert snapshot.js_solution.knot is None
    assert snapshot.js_solution.knot_dt is None
    torch.testing.assert_close(
        update.next_seed,
        snapshot.actions.view(2, 32, 7),
    )

    result = _finalize(processor, return_seeds=1)

    assert result.solution is not None
    assert result.optimized_seeds is result.solution
    assert result.js_solution.knot is result.solution
    torch.testing.assert_close(
        result.js_solution.knot_dt,
        result.js_solution.dt * 4,
    )


def test_finalize_reuses_one_flat_seed_index(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Top-seed selection does not rebuild offsets or use dimension-specific gathers."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=1)
    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1)))

    with profile(activities=[ProfilerActivity.CPU], acc_events=True) as finalize_profile:
        _finalize(processor, return_seeds=1)

    operator_names = {event.key for event in finalize_profile.key_averages()}
    assert "aten::arange" not in operator_names
    assert "aten::gather" not in operator_names
    assert "aten::index_select" in operator_names


def test_flat_seed_index_selects_multiple_batches(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """The contiguous index workspace supports a top-k subset in every batch."""
    processor.update_batch_size(batch_size=2, num_seeds=3, return_seeds=2)
    selected_seeds = torch.tensor([[2, 0], [1, 2]])
    flat_indices = processor._flat_seed_indices[:4]
    torch.add(
        selected_seeds,
        processor._batch_seed_offsets,
        out=flat_indices.view(2, 2),
    )
    source = torch.arange(12, dtype=torch.float32).view(2, 3, 2)

    selected = processor._select_tensor(source, flat_indices, return_seeds=2)

    expected = torch.stack(
        (
            torch.stack((source[0, 2], source[0, 0])),
            torch.stack((source[1, 1], source[1, 2])),
        )
    )
    torch.testing.assert_close(selected, expected)


def test_snapshot_workspaces_are_reused_across_solves(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Candidate updates overwrite shape-managed storage instead of cloning it."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    action_data_ptr = processor._snapshot.actions.data_ptr()
    state_data_ptr = processor._snapshot.js_solution.position.data_ptr()

    for offset in (0.0, 10.0):
        processor.reset(initial_dt)
        processor.update(_make_candidate(initial_dt.view(-1), action_offset=offset))
        assert processor._snapshot.actions.data_ptr() == action_data_ptr
        assert processor._snapshot.js_solution.position.data_ptr() == state_data_ptr


def test_interpolation_input_is_a_borrowed_all_seed_view(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """External interpolation receives the accumulated state without a trajectory clone."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1)))

    interpolation_input = processor.get_interpolation_input()

    assert (
        interpolation_input.position.data_ptr()
        == processor._snapshot.js_solution.position.data_ptr()
    )
    assert interpolation_input.knot.data_ptr() == processor._snapshot.actions.data_ptr()
    assert interpolation_input.shape == (2, 32, 7)
    torch.testing.assert_close(interpolation_input.dt, initial_dt.view(-1))
    torch.testing.assert_close(interpolation_input.knot_dt, initial_dt.view(-1) * 4)


def test_update_accumulates_only_successful_faster_candidates(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Finetune updates retain the best candidate independently for every seed."""
    initial_dt = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1)))

    mixed_dt = torch.tensor([[0.08, 0.12]], dtype=torch.float32)
    update = processor.update(_make_candidate(mixed_dt.view(-1), action_offset=10.0))

    assert update.improved
    torch.testing.assert_close(update.best_dt, torch.tensor([[0.08, 0.1]]))
    assert torch.all(update.next_seed[0] == 10.0)
    assert torch.all(update.next_seed[1] == 1.0)

    slower_dt = torch.tensor([[0.09, 0.11]], dtype=torch.float32)
    slower_update = processor.update(_make_candidate(slower_dt.view(-1), action_offset=20.0))

    assert not slower_update.improved
    torch.testing.assert_close(slower_update.best_dt, torch.tensor([[0.08, 0.1]]))


def test_update_reduces_knot_metrics_feasibility(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """A seed must be feasible at every knot trajectory timestep."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    candidate = _make_candidate(initial_dt.view(-1))
    candidate.metrics.feasible[0, 4] = False
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)

    update = processor.update(candidate)

    assert update.any_success
    torch.testing.assert_close(
        processor._snapshot.success,
        torch.tensor([[False, True]]),
    )


def test_update_reduces_interpolated_metrics_feasibility(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """A seed must also be feasible at every dense trajectory timestep."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    candidate = _make_candidate(initial_dt.view(-1))
    dense_feasible = torch.ones((2, 16), dtype=torch.bool)
    dense_feasible[1, 9] = False
    candidate.interpolated_metrics = RolloutMetrics(feasible=dense_feasible)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)

    update = processor.update(candidate)

    assert update.any_success
    torch.testing.assert_close(
        processor._snapshot.success,
        torch.tensor([[True, False]]),
    )


def test_update_extracts_multiple_terminal_convergence_terms(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Cached extraction preserves generic multi-term Cartesian convergence."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    candidate = _make_candidate(initial_dt.view(-1))
    horizon = candidate.metrics.actions.shape[1]
    candidate.metrics.convergence.add(
        torch.tensor([0.004, 0.001], dtype=torch.float32)
        .view(2, 1, 1)
        .expand(-1, horizon, -1),
        "secondary_position_tolerance",
    )
    candidate.metrics.convergence.add(
        torch.tensor([0.03, 0.01], dtype=torch.float32)
        .view(2, 1, 1)
        .expand(-1, horizon, -1),
        "secondary_orientation_tolerance",
    )
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)

    processor.update(candidate)

    torch.testing.assert_close(
        processor._snapshot.position_error,
        torch.tensor([[0.004, 0.003]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        processor._snapshot.rotation_error,
        torch.tensor([[0.03, 0.02]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        processor._snapshot.seed_cost,
        torch.tensor([[0.046, 0.034]], dtype=torch.float32),
    )


def test_finalize_all_seeds_preserves_accumulated_trajectories(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Returning every seed owns the accumulated knot and dense trajectories."""
    initial_dt = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1)))

    mixed_dt = torch.tensor([[0.08, 0.12]], dtype=torch.float32)
    processor.update(_make_candidate(mixed_dt.view(-1), action_offset=10.0))
    snapshot = processor._snapshot
    result = _finalize(processor, return_seeds=2)

    assert result is not snapshot
    assert snapshot.js_solution.knot is None
    assert snapshot.js_solution.knot_dt is None
    assert result.metrics is None
    assert result.interpolated_metrics is None
    assert result.optimized_seeds is result.solution
    assert result.js_solution.knot is result.solution
    assert torch.all(result.solution[0, 0] == 10.0)
    assert torch.all(result.solution[0, 1] == 1.0)
    assert torch.all(result.interpolated_trajectory.position[0, 0] == 10.0)
    assert torch.all(result.interpolated_trajectory.position[0, 1] == 1.0)
    torch.testing.assert_close(result.js_solution.dt, torch.tensor([[0.08, 0.1]]))


def test_finalized_result_is_owned_across_next_solve(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """Reusing snapshot storage does not overwrite an already returned result."""
    initial_dt = torch.tensor([[0.1, 0.05]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1)))
    first_result = _finalize(processor, return_seeds=2)

    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1), action_offset=10.0))

    assert torch.all(first_result.solution[0, 0] == 0.0)
    assert torch.all(first_result.solution[0, 1] == 1.0)
    assert torch.all(first_result.interpolated_trajectory.position[0, 0] == 0.0)
    assert torch.all(first_result.interpolated_trajectory.position[0, 1] == 1.0)


def test_update_replaces_snapshot_when_every_seed_improves(
    processor: TorchTrajOptResultProcessor,
) -> None:
    """A fully improved candidate replaces the snapshot without changing its contract."""
    initial_dt = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=2)
    processor.reset(initial_dt)
    processor.update(_make_candidate(initial_dt.view(-1)))

    faster_dt = torch.tensor([[0.08, 0.09]], dtype=torch.float32)
    update = processor.update(_make_candidate(faster_dt.view(-1), action_offset=10.0))
    result = _finalize(processor, return_seeds=2)

    assert update.improved
    assert torch.all(update.next_seed[0] == 10.0)
    assert torch.all(update.next_seed[1] == 11.0)
    assert result.metrics is None
    torch.testing.assert_close(result.js_solution.dt, faster_dt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replays_updates_and_borrowed_finalization() -> None:
    """Replay fixed-shape update and finalization graphs against updated buffers."""
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    config = TrajOptResultProcessorConfig(
        device_cfg=device_cfg,
        action_horizon=32,
        horizon=32,
        action_dim=7,
        interpolation_steps=4,
        goalset_index_dim=1,
        goalset_index_dtype=torch.int32,
        joint_names=[f"joint_{index}" for index in range(7)],
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        minimum_trajectory_dt=0.01,
        maximum_trajectory_dt=0.15,
        use_cuda_graph=True,
    )
    processor = TorchTrajOptResultProcessor(config, lambda solution: solution)
    initial_dt = torch.tensor([[0.1, 0.1]], device=device_cfg.device)
    candidate = _make_candidate(initial_dt.view(-1))
    processor.update_batch_size(batch_size=1, num_seeds=2, return_seeds=1)
    processor.reset(initial_dt)

    processor.update(candidate)
    candidate.metrics.actions.add_(10.0)
    candidate.metrics.state.joint_state.position.add_(10.0)
    candidate.metrics.state.joint_state.dt.copy_(
        torch.tensor([0.08, 0.12], device=device_cfg.device)
    )
    processor.update(candidate)
    interpolated_trajectory, interpolated_last_tstep = _make_interpolation_output(processor)
    first_result = processor.finalize(
        1,
        interpolated_trajectory,
        interpolated_last_tstep,
    )
    first_solution = first_result.solution.clone()
    solution_data_ptr = first_result.solution.data_ptr()
    assert processor.is_cuda_graph_initialized

    processor.reset(initial_dt)
    candidate.metrics.actions.add_(10.0)
    candidate.metrics.state.joint_state.position.add_(10.0)
    candidate.metrics.state.joint_state.dt.copy_(initial_dt.view(-1))
    processor.update(candidate)
    interpolated_trajectory.position.add_(10.0)
    second_result = processor.finalize(
        1,
        interpolated_trajectory,
        interpolated_last_tstep,
    )

    assert second_result is first_result
    assert second_result.solution.data_ptr() == solution_data_ptr
    assert not torch.equal(second_result.solution, first_solution)
    processor.reset_cuda_graph()
