# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fixed-budget GPU RRT shortcut pass."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.rrt.rrt_path_shortcut import (
    RRTPathShortcutter,
    RRTPathShortcutterConfig,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def shortcut() -> RRTPathShortcutter:
    """Create one small compiled profile shared by the behavioral tests."""
    return RRTPathShortcutter(
        RRTPathShortcutterConfig(
            action_dim=2,
            path_capacity=5,
            state_capacity=128,
            cspace_resolution=0.02,
        ),
        torch.device("cuda"),
        torch.ones(2, device="cuda", dtype=torch.float32),
    )


def test_collision_free_path_collapses_with_one_rollout_and_routes_environment(shortcut):
    """Test every complete edge once and select the direct start-goal edge."""
    path = torch.tensor(
        [[-0.1, 0.0], [-0.1, 0.05], [0.0, 0.1], [0.1, 0.05], [0.1, 0.0]],
        device="cuda",
    )
    calls = []

    def all_feasible(samples, clone_output=True, env_query_idx=None):
        calls.append((samples.shape, clone_output, env_query_idx.clone()))
        return torch.ones(samples.shape[0], device=samples.device, dtype=torch.bool)

    output, stats = shortcut.shortcut(
        path,
        all_feasible,
        env_query_idx=torch.tensor(3, device="cuda", dtype=torch.int32),
    )

    assert len(calls) == 1
    assert calls[0][0] == (shortcut.profile.state_capacity, 2)
    assert calls[0][1] is False
    assert torch.all(calls[0][2] == 3)
    assert output.shape == (2, 2)
    torch.testing.assert_close(output, path[[0, -1]], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        shortcut.path_length[0],
        torch.linalg.vector_norm(path[-1] - path[0]),
    )
    assert stats["selected_waypoints"] == 2
    assert stats["tested_edges"] == shortcut.profile.edge_capacity
    assert stats["overflow_edges"] == 0


def test_blocked_direct_edge_selects_collision_free_detour(shortcut):
    """Run DAG selection over selective visibility rather than greedy rounds."""
    path = torch.tensor(
        [[-0.1, 0.0], [-0.1, 0.1], [0.1, 0.1], [0.1, 0.0]],
        device="cuda",
    )

    def outside_obstacle(samples, clone_output=True):
        blocked = (torch.abs(samples[:, 0]) < 0.05) & (samples[:, 1] < 0.025)
        return ~blocked

    output, stats = shortcut.shortcut(path, outside_obstacle)

    assert output.shape[0] == 3
    torch.testing.assert_close(output[0], path[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[-1], path[-1], rtol=0.0, atol=0.0)
    assert torch.any(output[1, 1] > 0.05)
    assert stats["selected_waypoints"] == 3

    # A later all-free call must not reuse stale edge visibility.
    direct, _ = shortcut.shortcut(
        path,
        lambda samples, clone_output=True: torch.ones(
            samples.shape[0], device=samples.device, dtype=torch.bool
        ),
    )
    assert direct.shape[0] == 2


def test_visibility_dag_matches_cpu_lexicographic_reference(shortcut):
    """Match minimum hops, weighted length, and stable predecessor ordering."""
    path = torch.tensor(
        [[0.0, 0.0], [0.03, 0.01], [0.07, 0.04], [0.12, 0.02], [0.18, 0.06]],
        device="cuda",
    )
    shortcut._prepare_states(path)
    torch.cuda.synchronize()
    sources = shortcut.pair_sources.cpu().tolist()
    destinations = shortcut.pair_destinations.cpu().tolist()
    offsets = shortcut.edge_offsets[0].cpu().tolist()
    steps = shortcut.edge_steps[0].cpu().tolist()
    tested = shortcut.edge_tested[0].cpu().tolist()
    generator = torch.Generator().manual_seed(29)

    for _ in range(10):
        visible = torch.rand(len(sources), generator=generator) > 0.45
        shortcut.feasible.fill_(True)
        for edge, is_visible in enumerate(visible.tolist()):
            if tested[edge] and not is_visible:
                assert steps[edge] > 0
                shortcut.feasible[offsets[edge]] = False

        _, stats = shortcut._select()
        actual = shortcut.output_indices[0, : stats["selected_waypoints"]].cpu().tolist()

        edge_visible = {
            (source, destination): bool(visible[edge]) and tested[edge]
            for edge, (source, destination) in enumerate(zip(sources, destinations))
        }
        hops = [0] + [path.shape[0] + 1] * (path.shape[0] - 1)
        costs = [0.0] + [float("inf")] * (path.shape[0] - 1)
        parents = [-1] * path.shape[0]
        for destination in range(1, path.shape[0]):
            candidates = []
            for source in range(destination):
                if destination != source + 1 and not edge_visible.get(
                    (source, destination), False
                ):
                    continue
                edge_cost = float(torch.linalg.vector_norm(path[destination] - path[source]))
                candidates.append((hops[source] + 1, costs[source] + edge_cost, source))
            hops[destination], costs[destination], parents[destination] = min(candidates)
        expected = []
        cursor = path.shape[0] - 1
        while cursor >= 0:
            expected.append(cursor)
            cursor = parents[cursor]
        assert actual == expected[::-1]


def test_complete_edge_overflow_preserves_raw_adjacent_path():
    """Never partially validate an edge when its interior grid cannot fit."""
    workspace = RRTPathShortcutter(
        RRTPathShortcutterConfig(
            action_dim=1,
            path_capacity=5,
            state_capacity=2,
            cspace_resolution=0.02,
        ),
        torch.device("cuda:0"),
        torch.ones(1, device="cuda", dtype=torch.float32),
    )
    path = torch.arange(5, device="cuda", dtype=torch.float32).unsqueeze(-1) * 0.1
    output, stats = workspace.shortcut(
        path,
        lambda samples, clone_output=True: torch.ones(
            samples.shape[0], device=samples.device, dtype=torch.bool
        ),
    )

    torch.testing.assert_close(output, path, rtol=0.0, atol=0.0)
    assert stats["tested_edges"] == 0
    assert stats["overflow_edges"] == workspace.profile.edge_capacity


def test_generated_profile_supports_non_franka_dof():
    """Compile constants from the deployment robot rather than assuming seven DoF."""
    action_dim = 9
    workspace = RRTPathShortcutter(
        RRTPathShortcutterConfig(
            action_dim=action_dim,
            path_capacity=3,
            state_capacity=32,
            cspace_resolution=0.02,
        ),
        torch.device("cuda"),
        torch.linspace(0.8, 1.2, action_dim, device="cuda"),
    )
    path = torch.zeros((3, action_dim), device="cuda")
    path[:, 0] = torch.tensor([0.0, 0.1, 0.2], device="cuda")
    output, _ = workspace.shortcut(
        path,
        lambda samples, clone_output=True: torch.ones(
            samples.shape[0], device=samples.device, dtype=torch.bool
        ),
    )
    torch.testing.assert_close(output, path[[0, -1]], rtol=0.0, atol=0.0)


def test_shortcut_captures_fixed_prepare_and_select_stages(
    shortcut: RRTPathShortcutter,
) -> None:
    """Replay fixed-address shortcut stages and clear both graphs on reset."""
    shortcut.reset_cuda_graph()
    assert not shortcut._prepare_executor.is_initialized
    assert not shortcut._select_executor.is_initialized
    path = torch.tensor(
        [[-0.1, 0.0], [-0.1, 0.1], [0.1, 0.1], [0.1, 0.0]],
        device="cuda",
    )

    direct, _ = shortcut.shortcut(
        path,
        lambda samples, clone_output=True: torch.ones(
            samples.shape[0], device=samples.device, dtype=torch.bool
        ),
    )
    assert shortcut._prepare_executor.is_initialized
    assert shortcut._select_executor.is_initialized
    torch.testing.assert_close(direct, path[[0, -1]], rtol=0.0, atol=0.0)

    def outside_obstacle(
        samples: torch.Tensor,
        clone_output: bool = True,
    ) -> torch.Tensor:
        blocked = (torch.abs(samples[:, 0]) < 0.05) & (samples[:, 1] < 0.025)
        return ~blocked

    detour, _ = shortcut.shortcut(path, outside_obstacle)
    assert detour.shape[0] == 3
    shortcut.reset_cuda_graph()
    assert not shortcut._prepare_executor.is_initialized
    assert not shortcut._select_executor.is_initialized


def test_batched_shortcut_uses_one_rollout_and_keeps_lanes_independent() -> None:
    """Shortcut fixed lanes together while preserving invalid and oversized paths."""
    workspace = RRTPathShortcutter(
        RRTPathShortcutterConfig(
            action_dim=2,
            path_capacity=4,
            raw_path_capacity=6,
            state_capacity=128,
            batch_capacity=3,
            cspace_resolution=0.02,
        ),
        torch.device("cuda"),
        torch.ones(2, device="cuda", dtype=torch.float32),
    )
    paths = torch.zeros((3, 6, 2), device="cuda", dtype=torch.float32)
    paths[0, :4] = torch.tensor([[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, 0.0]], device="cuda")
    paths[1, :6, 0] = torch.arange(6, device="cuda", dtype=torch.float32) * 0.02
    counts = torch.tensor([4, 6, 0], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0.3, 0.1, torch.inf], device="cuda", dtype=torch.float32)
    calls = []

    def all_feasible(samples: torch.Tensor, clone_output: bool = True) -> torch.Tensor:
        calls.append((samples.shape, clone_output))
        return torch.ones(samples.shape[0], device=samples.device, dtype=torch.bool)

    output, output_count, path_length, success, status = workspace.shortcut_batched(
        paths,
        counts,
        lengths,
        all_feasible,
    )
    torch.cuda.synchronize()

    assert calls == [((128, 2), False)]
    assert output_count.tolist() == [2, 6, 0]
    assert success.tolist() == [True, True, False]
    torch.testing.assert_close(output[0, :2], paths[0, [0, 3]], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[1, :6], paths[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(path_length[1], lengths[1], rtol=0.0, atol=0.0)
    assert status[:, 4].tolist() == [0, 1, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_dim": 0},
        {"path_capacity": 2},
        {"state_capacity": 1},
        {"cspace_resolution": 0.0},
    ],
)
def test_shortcut_profile_rejects_invalid_constants(kwargs):
    """Reject malformed values before runtime kernel generation."""
    values = {"action_dim": 7}
    values.update(kwargs)
    with pytest.raises(ValueError):
        RRTPathShortcutterConfig(**values)
