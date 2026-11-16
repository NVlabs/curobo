# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the canonical batched RRT tree kernels."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.rrt.rrt_tree_kernel import RRTTreeKernelConfig, RRTTreeKernelRunner
from curobo._src.util.warp import get_warp_device_stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _nearest_reference(
    nodes: torch.Tensor,
    count: int,
    desired: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the batched-kernel nearest-start contract in Torch."""
    distances = torch.sum(
        ((desired.unsqueeze(1) - nodes[:count].unsqueeze(0)) * weights) ** 2,
        dim=-1,
    )
    parents = torch.argmin(distances, dim=-1)
    starts = nodes[parents]
    return starts, parents


@pytest.mark.parametrize("action_dim", [6, 7, 65])
def test_batched_nearest_matches_independent_trees_with_partial_counts(action_dim):
    """Each batch lane scans only its runtime count, including partial tiles."""
    torch.manual_seed(100 + action_dim)
    device = torch.device("cuda")
    batch_size = 4
    capacity = 128
    candidate_count = 5
    counts = torch.tensor([1, 37, 65, 127], dtype=torch.int32, device=device)
    nodes = torch.randn((batch_size, capacity, action_dim), device=device)
    desired = torch.randn((batch_size, candidate_count, action_dim), device=device)
    weights = torch.linspace(0.7, 1.3, action_dim, device=device)
    core = RRTTreeKernelRunner(
        RRTTreeKernelConfig(
            action_dim=action_dim,
            max_candidates=candidate_count,
            block_dim=64,
        )
    )
    binding = core.bind_batched_nearest_target(nodes, counts, desired, weights)

    starts, parents = core.batched_nearest_target(binding)
    torch.cuda.synchronize(device)

    for batch_index, count in enumerate(counts.cpu().tolist()):
        expected_starts, expected_parents = _nearest_reference(
            nodes[batch_index],
            count,
            desired[batch_index],
            weights,
        )
        torch.testing.assert_close(starts[batch_index], expected_starts, rtol=0.0, atol=0.0)
        assert torch.equal(parents[batch_index], expected_parents)


def test_batched_nearest_capture_reuses_runtime_counts_and_buffers():
    """One captured launch accepts new logical counts and query values."""
    torch.manual_seed(211)
    device = torch.device("cuda")
    action_dim = 7
    batch_size = 2
    capacity = 128
    candidate_count = 24
    nodes = torch.randn((batch_size, capacity, action_dim), device=device)
    counts = torch.tensor([1, 65], dtype=torch.int32, device=device)
    desired = torch.randn((batch_size, candidate_count, action_dim), device=device)
    weights = torch.linspace(0.8, 1.2, action_dim, device=device)
    core = RRTTreeKernelRunner(
        RRTTreeKernelConfig(action_dim=action_dim, max_candidates=candidate_count)
    )
    binding = core.bind_batched_nearest_target(nodes, counts, desired, weights)
    core.load_modules(device)
    core.batched_nearest_target(binding)
    torch.cuda.synchronize(device)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _, warp_stream = get_warp_device_stream(desired)
        assert warp_stream is not None
        core.batched_nearest_target(binding, stream=warp_stream)
    torch.cuda.current_stream(device).wait_stream(capture_stream)

    counts.copy_(torch.tensor([127, 37], dtype=torch.int32, device=device))
    desired.normal_()
    graph.replay()
    torch.cuda.synchronize(device)

    for batch_index, count in enumerate(counts.cpu().tolist()):
        expected_starts, expected_parents = _nearest_reference(
            nodes[batch_index],
            count,
            desired[batch_index],
            weights,
        )
        torch.testing.assert_close(
            binding.starts[batch_index], expected_starts, rtol=0.0, atol=0.0
        )
        assert torch.equal(binding.parent_indices[batch_index], expected_parents)


def test_batched_packed_append_preserves_order_and_capacity():
    """Wide candidate waves append independently in stable frontier order."""
    torch.manual_seed(307)
    device = torch.device("cuda")
    batch_size = 3
    capacity = 128
    candidate_count = 96
    action_dim = 9
    counts = torch.tensor([1, 65, 120], dtype=torch.int32, device=device)
    profile = RRTTreeKernelConfig(
        action_dim=action_dim,
        max_candidates=candidate_count,
    )
    assert profile.block_dim == 128
    core = RRTTreeKernelRunner(profile)
    nodes = torch.randn((batch_size, capacity, action_dim), device=device)
    parents = torch.full(
        (batch_size, capacity),
        -1,
        dtype=torch.int64,
        device=device,
    )
    candidates = torch.randn((batch_size, candidate_count, action_dim), device=device)
    candidate_parents = torch.arange(candidate_count, device=device).expand(batch_size, -1).clone()
    candidate_parents %= counts.to(torch.int64).unsqueeze(-1)
    candidate_valid = torch.arange(candidate_count, device=device).remainder(3) != 0
    candidate_valid = candidate_valid.expand(batch_size, -1).clone()
    # Invalid parent indices are rejected before capacity packing.
    candidate_parents[0, 1] = -2
    candidate_parents[1, 2] = counts[1]
    original_nodes = nodes.clone()
    original_counts = counts.clone()
    binding = core.bind_batched_packed_append(
        nodes,
        parents,
        counts,
        candidates,
        candidate_parents,
        candidate_valid,
    )

    slots, written, updated_counts = core.batched_packed_append(binding)
    torch.cuda.synchronize(device)

    for batch_index, base in enumerate(original_counts.cpu().tolist()):
        valid = candidate_valid[batch_index] & (
            (candidate_parents[batch_index] >= 0) & (candidate_parents[batch_index] < base)
        )
        valid_indices = torch.where(valid)[0][: capacity - base]
        end = base + valid_indices.numel()
        assert int(updated_counts[batch_index].item()) == end
        assert int(written[batch_index].sum().item()) == valid_indices.numel()
        assert torch.equal(
            slots[batch_index, written[batch_index]],
            torch.arange(base, end, dtype=torch.int64, device=device),
        )
        torch.testing.assert_close(
            nodes[batch_index, base:end],
            candidates[batch_index, valid_indices],
        )
        assert torch.equal(
            parents[batch_index, base:end],
            candidate_parents[batch_index, valid_indices],
        )
        torch.testing.assert_close(
            nodes[batch_index, :base],
            original_nodes[batch_index, :base],
        )


def test_batched_packed_append_clamps_counts_and_rejects_invalid_parents():
    """Malformed counts stay in bounds and appends cannot introduce new roots."""
    device = torch.device("cuda")
    action_dim = 6
    capacity = 64
    candidate_count = 4
    core = RRTTreeKernelRunner(
        RRTTreeKernelConfig(action_dim=action_dim, max_candidates=candidate_count)
    )
    nodes = torch.zeros((2, capacity, action_dim), device=device)
    parents = torch.full((2, capacity), -1, dtype=torch.int64, device=device)
    counts = torch.tensor([-3, capacity + 7], dtype=torch.int32, device=device)
    candidates = torch.randn((2, candidate_count, action_dim), device=device)
    candidate_parents = torch.tensor(
        [[-1, 0, -2, capacity], [0, 0, 0, 0]],
        dtype=torch.int64,
        device=device,
    )
    candidate_valid = torch.ones((2, candidate_count), dtype=torch.bool, device=device)
    binding = core.bind_batched_packed_append(
        nodes,
        parents,
        counts,
        candidates,
        candidate_parents,
        candidate_valid,
    )

    _, written, updated_counts = core.batched_packed_append(binding)
    torch.cuda.synchronize(device)

    assert written.tolist() == [[False] * candidate_count, [False] * candidate_count]
    assert updated_counts.tolist() == [0, capacity]
    assert torch.count_nonzero(nodes) == 0


def test_batched_path_reconstruction_traces_both_trees_and_deduplicates_connection() -> None:
    """Trace parent chains on GPU and preserve the start-to-goal waypoint order."""
    device = torch.device("cuda")
    profile = RRTTreeKernelConfig(action_dim=2, max_candidates=4, max_path_nodes=68)
    core = RRTTreeKernelRunner(profile)
    start_nodes = torch.zeros((3, 64, 2), device=device)
    goal_nodes = torch.zeros_like(start_nodes)
    start_parents = torch.full((3, 64), -1, device=device, dtype=torch.int64)
    goal_parents = torch.full_like(start_parents, -1)

    start_nodes[0, :3] = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], device=device)
    start_parents[0, 1:3] = torch.tensor([0, 1], device=device)
    goal_nodes[0, :2] = torch.tensor([[0.3, 0.0], [0.2, 0.0]], device=device)
    goal_parents[0, 1] = 0

    start_nodes[1, :2] = torch.tensor([[1.0, 0.0], [1.1, 0.0]], device=device)
    start_parents[1, 1] = 0
    goal_nodes[1, :3] = torch.tensor([[1.4, 0.0], [1.3, 0.0], [1.2, 0.0]], device=device)
    goal_parents[1, 1:3] = torch.tensor([0, 1], device=device)

    binding = core.bind_batched_path_reconstruction(
        start_nodes,
        start_parents,
        goal_nodes,
        goal_parents,
        torch.ones(2, device=device, dtype=torch.float32),
    )
    binding.connection_indices.copy_(
        torch.tensor([[2, 1], [1, 2], [-1, -1]], device=device, dtype=torch.int64)
    )

    paths, counts, lengths = core.batched_reconstruct_path(binding)
    torch.cuda.synchronize(device)

    assert counts.tolist() == [4, 5, 0]
    torch.testing.assert_close(lengths[:2], torch.tensor([0.3, 0.4], device=device))
    assert torch.isinf(lengths[2])
    torch.testing.assert_close(
        paths[0, :4],
        torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]], device=device),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        paths[1, :5],
        torch.tensor(
            [[1.0, 0.0], [1.1, 0.0], [1.2, 0.0], [1.3, 0.0], [1.4, 0.0]],
            device=device,
        ),
        rtol=0.0,
        atol=0.0,
    )
