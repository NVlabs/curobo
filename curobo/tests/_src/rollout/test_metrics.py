# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for rollout metrics.

Tests the metrics.py module which provides data structures for storing and
managing costs, constraints, and rollout results in optimization.
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.metrics import (
    CostCollection,
    CostCollectionSum,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.state.state_joint import JointState


class TestCostCollectionInitialization:
    """Test CostCollection initialization."""

    def test_empty_initialization(self):
        """Test creating an empty cost collection."""
        collection = CostCollection()

        assert collection.is_empty()
        assert len(collection.values) == 0
        assert len(collection.names) == 0
        assert len(collection.weights) == 0
        assert len(collection.sq_weights) == 0

    def test_initialization_with_fields(self, cuda_device_cfg):
        """Test creating cost collection with initial data."""
        values = [
            torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        ]
        names = ["cost1"]
        weights = [torch.ones(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)]
        sq_weights = [torch.ones(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)]

        collection = CostCollection(
            values=values, names=names, weights=weights, sq_weights=sq_weights
        )

        assert not collection.is_empty()
        assert len(collection.values) == 1
        assert len(collection.names) == 1
        assert collection.names[0] == "cost1"


class TestCostCollectionAdd:
    """Test adding costs to a collection."""

    def test_add_simple_cost(self, cuda_device_cfg):
        """Test adding a cost without weights."""
        collection = CostCollection()
        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection.add(cost, "test_cost")

        assert len(collection.values) == 1
        assert len(collection.names) == 1
        assert collection.names[0] == "test_cost"
        assert torch.allclose(collection.values[0], cost)

    def test_add_cost_with_weight(self, cuda_device_cfg):
        """Test adding a cost with weight."""
        collection = CostCollection()
        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        weight = torch.tensor([2.5], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection.add(cost, "weighted_cost", weight=weight)

        assert len(collection.weights) == 1
        assert torch.allclose(collection.weights[0], weight)

    def test_add_cost_with_sq_weight(self, cuda_device_cfg):
        """Test adding a cost with squared weight."""
        collection = CostCollection()
        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        sq_weight = torch.tensor([4.0], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection.add(cost, "sq_weighted_cost", sq_weight=sq_weight)

        assert len(collection.sq_weights) == 1
        assert torch.allclose(collection.sq_weights[0], sq_weight)

    def test_add_multiple_costs(self, cuda_device_cfg):
        """Test adding multiple costs."""
        collection = CostCollection()

        for i in range(3):
            cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
            collection.add(cost, f"cost_{i}")

        assert len(collection.values) == 3
        assert len(collection.names) == 3
        assert collection.names == ["cost_0", "cost_1", "cost_2"]


class TestCostCollectionSum:
    """Test cost summation."""

    def test_get_sum_empty_collection(self):
        """Test summing an empty collection."""
        collection = CostCollection()
        result = collection.get_sum()

        # Empty collection returns zeros
        assert result.numel() == 1
        assert torch.allclose(result, torch.zeros_like(result))

    def test_get_sum_single_cost(self, cuda_device_cfg):
        """Test summing a single cost."""
        collection = CostCollection()
        cost = torch.ones(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        collection.add(cost, "cost1")

        result_sum_horizon = collection.get_sum(sum_horizon=True)
        result_no_sum = collection.get_sum(sum_horizon=False)

        # With sum_horizon=True, should sum over horizon and last dimension
        assert result_sum_horizon.shape[0] == 2  # batch dimension preserved
        # With sum_horizon=False, horizon dimension preserved, last dim summed
        assert result_no_sum.shape[0] == 2  # batch dimension preserved
        assert result_no_sum.shape[1] == 5  # horizon dimension preserved

    def test_get_sum_multiple_costs(self, cuda_device_cfg):
        """Test summing multiple costs."""
        collection = CostCollection()

        cost1 = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cost2 = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 2

        collection.add(cost1, "cost1")
        collection.add(cost2, "cost2")

        result = collection.get_sum(sum_horizon=True)

        # Should sum both costs and horizon
        # cat_sum concatenates [batch, horizon, 1] + [batch, horizon, 1] -> [batch, horizon, 2]
        # Then sums over (1, 2) dims -> [batch, 1]
        assert result.shape == (2, 1)
        # Verify values: (1 + 2) summed over 5 horizon steps = 15 per batch
        expected = torch.tensor([[15.0], [15.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        assert torch.allclose(result, expected)


class TestCostCollectionClone:
    """Test cloning cost collections."""

    def test_clone_empty_collection(self):
        """Test cloning an empty collection."""
        collection = CostCollection()
        cloned = collection.clone()

        assert cloned.is_empty()
        assert len(cloned.values) == 0

    def test_clone_with_costs(self, cuda_device_cfg):
        """Test cloning a collection with costs."""
        collection = CostCollection()
        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        weight = torch.ones(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection.add(cost, "cost1", weight=weight)

        cloned = collection.clone()

        assert len(cloned.values) == 1
        assert len(cloned.names) == 1
        assert cloned.names[0] == "cost1"
        assert torch.allclose(cloned.values[0], collection.values[0])

        # Verify they are separate tensors
        collection.values[0].fill_(999.0)
        assert not torch.allclose(cloned.values[0], collection.values[0])


class TestCostCollectionMerge:
    """Test merging cost collections."""

    def test_merge_empty_collections(self):
        """Test merging two empty collections."""
        collection1 = CostCollection()
        collection2 = CostCollection()

        collection1.merge(collection2)

        assert collection1.is_empty()

    def test_merge_into_empty(self, cuda_device_cfg):
        """Test merging into an empty collection."""
        collection1 = CostCollection()
        collection2 = CostCollection()

        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        collection2.add(cost, "cost2")

        collection1.merge(collection2)

        assert len(collection1.values) == 1
        assert collection1.names[0] == "cost2"

    def test_merge_two_nonempty_collections(self, cuda_device_cfg):
        """Test merging two non-empty collections."""
        collection1 = CostCollection()
        collection2 = CostCollection()

        cost1 = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cost2 = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection1.add(cost1, "cost1")
        collection2.add(cost2, "cost2")

        collection1.merge(collection2)

        assert len(collection1.values) == 2
        assert collection1.names == ["cost1", "cost2"]


class TestCostCollectionCopy:
    """Test copying operations."""

    def test_copy_at_batch_seed_indices(self, cuda_device_cfg):
        """Test copying at specific batch and seed indices."""
        collection1 = CostCollection()
        collection2 = CostCollection()

        cost1 = torch.zeros(4, 8, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cost2 = torch.ones(4, 8, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection1.add(cost1, "cost")
        collection2.add(cost2, "cost")

        batch_idx = torch.tensor([0, 2], device=cuda_device_cfg.device)
        seed_idx = torch.tensor([1, 3], device=cuda_device_cfg.device)

        collection1.copy_at_batch_seed_indices(collection2, batch_idx, seed_idx)

        # Check that specified indices were copied
        assert torch.allclose(
            collection1.values[0][0, 1],
            torch.tensor([[1.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )
        assert torch.allclose(
            collection1.values[0][2, 3],
            torch.tensor([[1.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )

        # Check that other indices remain unchanged
        assert torch.allclose(
            collection1.values[0][1, 1],
            torch.tensor([[0.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )

    def test_copy_only_index(self, cuda_device_cfg):
        """Test copying at a specific index."""
        collection1 = CostCollection()
        collection2 = CostCollection()

        cost1 = torch.zeros(4, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cost2 = torch.ones(4, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        collection1.add(cost1, "cost")
        collection2.add(cost2, "cost")

        collection1.copy_only_index(collection2, 2)

        # Check that index 2 was copied
        assert torch.allclose(collection1.values[0][2], torch.ones(5, 1, device=cuda_device_cfg.device))

        # Check that other indices remain unchanged
        assert torch.allclose(collection1.values[0][0], torch.zeros(5, 1, device=cuda_device_cfg.device))
        assert torch.allclose(collection1.values[0][1], torch.zeros(5, 1, device=cuda_device_cfg.device))


class TestCostsAndConstraintsInitialization:
    """Test CostsAndConstraints initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        cc = CostsAndConstraints()

        assert cc.costs.is_empty()
        assert cc.constraints.is_empty()
        assert cc.hybrid_costs_constraints.is_empty()

    def test_initialization_with_collections(self, cuda_device_cfg):
        """Test initialization with existing collections."""
        costs = CostCollection()
        constraints = CostCollection()

        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        costs.add(cost, "cost1")

        cc = CostsAndConstraints(costs=costs, constraints=constraints)

        assert len(cc.costs.values) == 1
        assert cc.constraints.is_empty()


class TestCostsAndConstraintsSums:
    """Test sum operations."""

    def test_get_sum_cost_empty(self):
        """Test getting sum of empty costs."""
        cc = CostsAndConstraints()

        result = cc.get_sum_cost()

        assert result is None

    def test_get_sum_cost_single(self, cuda_device_cfg):
        """Test getting sum of single cost."""
        cc = CostsAndConstraints()
        cost = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 3.0

        cc.costs.add(cost, "cost1")

        result = cc.get_sum_cost(sum_horizon=True)

        # Should sum over horizon: 3.0 * 5 = 15.0 per batch
        # Result shape is [batch, 1]
        expected = torch.tensor([[15.0], [15.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        assert torch.allclose(result, expected)

    def test_get_sum_constraint_empty(self):
        """Test getting sum of empty constraints."""
        cc = CostsAndConstraints()

        result = cc.get_sum_constraint()

        assert result is None

    def test_get_sum_cost_and_constraint(self, cuda_device_cfg):
        """Test getting combined sum of costs and constraints."""
        cc = CostsAndConstraints()

        cost = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 2.0
        constraint = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 3.0

        cc.costs.add(cost, "cost1")
        cc.constraints.add(constraint, "constraint1")

        result = cc.get_sum_cost_and_constraint(sum_horizon=True)

        # Should sum both: (2.0 + 3.0) * 5 = 25.0 per batch
        # Result shape is [batch, 1]
        expected = torch.tensor([[25.0], [25.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        assert torch.allclose(result, expected)

    def test_get_sum_with_hybrid(self, cuda_device_cfg):
        """Test sum with hybrid costs/constraints."""
        cc = CostsAndConstraints()

        cost = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        hybrid = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 2.0

        cc.costs.add(cost, "cost1")
        cc.hybrid_costs_constraints.add(hybrid, "hybrid1")

        result = cc.get_sum_cost(sum_horizon=True, include_all_hybrid=True)

        # Should sum both: (1.0 + 2.0) * 5 = 15.0 per batch
        expected = torch.tensor([[15.0], [15.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        assert torch.allclose(result, expected)

    def test_get_sum_exclude_hybrid(self, cuda_device_cfg):
        """Test sum excluding hybrid costs."""
        cc = CostsAndConstraints()

        cost = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        hybrid = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 10.0

        cc.costs.add(cost, "cost1")
        cc.hybrid_costs_constraints.add(hybrid, "hybrid1")

        result = cc.get_sum_cost(sum_horizon=True, include_all_hybrid=False)

        # Should only sum cost1: 1.0 * 5 = 5.0 per batch
        expected = torch.tensor([[5.0], [5.0]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        assert torch.allclose(result, expected)


class TestCostsAndConstraintsFeasibility:
    """Test feasibility checking."""

    def test_get_feasible_no_constraints(self):
        """Test feasibility with no constraints."""
        cc = CostsAndConstraints()

        result = cc.get_feasible()

        assert result is True

    def test_get_feasible_all_satisfied(self, cuda_device_cfg):
        """Test feasibility when all constraints satisfied."""
        cc = CostsAndConstraints()

        # Negative constraint values mean satisfied
        constraint = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * -1.0

        cc.constraints.add(constraint, "constraint1")

        result = cc.get_feasible(sum_horizon=True)

        assert torch.all(result == True)

    def test_get_feasible_violated(self, cuda_device_cfg):
        """Test feasibility when constraints violated."""
        cc = CostsAndConstraints()

        # Positive constraint values mean violated
        constraint = torch.ones(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 1.0

        cc.constraints.add(constraint, "constraint1")

        result = cc.get_feasible(sum_horizon=True)

        assert torch.all(result == False)


class TestCostsAndConstraintsClone:
    """Test cloning operations."""

    def test_clone_empty(self):
        """Test cloning empty CostsAndConstraints."""
        cc = CostsAndConstraints()
        cloned = cc.clone()

        assert cloned.costs.is_empty()
        assert cloned.constraints.is_empty()

    def test_clone_with_data(self, cuda_device_cfg):
        """Test cloning with data."""
        cc = CostsAndConstraints()

        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cc.costs.add(cost, "cost1")

        cloned = cc.clone()

        assert len(cloned.costs.values) == 1
        assert torch.allclose(cloned.costs.values[0], cost)

        # Verify independence
        cc.costs.values[0].fill_(999.0)
        assert not torch.allclose(cloned.costs.values[0], cc.costs.values[0])


class TestRolloutResultInitialization:
    """Test RolloutResult initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        result = RolloutResult()

        assert result.actions is None
        assert result.costs_and_constraints is None
        assert result.state is None
        assert result.debug is None

    def test_initialization_with_data(self, cuda_device_cfg):
        """Test initialization with data."""
        actions = torch.randn(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cc = CostsAndConstraints()
        state = JointState.from_position(actions)

        result = RolloutResult(actions=actions, costs_and_constraints=cc, state=state)

        assert result.actions is not None
        assert result.costs_and_constraints is not None
        assert result.state is not None


class TestRolloutResultLen:
    """Test length operations."""

    def test_len_with_actions(self, cuda_device_cfg):
        """Test length with actions."""
        actions = torch.randn(4, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        result = RolloutResult(actions=actions)

        assert len(result) == 4  # Batch size


class TestRolloutResultIndexing:
    """Test indexing operations."""

    def test_getitem_single_index(self, cuda_device_cfg):
        """Test getting a single item."""
        actions = torch.randn(4, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        result = RolloutResult(actions=actions)

        indexed = result[0]

        assert isinstance(indexed, RolloutResult)
        assert indexed.actions.shape == (5, 3)  # Single batch element

    def test_getitem_slice(self, cuda_device_cfg):
        """Test getting a slice."""
        actions = torch.randn(4, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        result = RolloutResult(actions=actions)

        sliced = result[1:3]

        assert isinstance(sliced, RolloutResult)
        assert sliced.actions.shape == (2, 5, 3)


class TestRolloutResultClone:
    """Test cloning operations."""

    def test_clone_empty(self):
        """Test cloning empty result."""
        result = RolloutResult()
        cloned = result.clone()

        assert cloned.actions is None
        assert cloned.costs_and_constraints is None

    def test_clone_with_data(self, cuda_device_cfg):
        """Test cloning with data."""
        actions = torch.randn(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cc = CostsAndConstraints()
        cost = torch.randn(2, 5, 1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cc.costs.add(cost, "cost1")

        result = RolloutResult(actions=actions, costs_and_constraints=cc)
        cloned = result.clone()

        assert torch.allclose(cloned.actions, actions)
        assert len(cloned.costs_and_constraints.costs.values) == 1

        # Verify independence
        result.actions.fill_(999.0)
        assert not torch.allclose(cloned.actions, result.actions)


class TestRolloutMetricsInitialization:
    """Test RolloutMetrics initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        metrics = RolloutMetrics()

        assert metrics.feasible is None
        assert metrics.convergence.is_empty()

    def test_initialization_with_data(self, cuda_device_cfg):
        """Test initialization with data."""
        actions = torch.randn(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        feasible = torch.tensor([True, False], device=cuda_device_cfg.device)

        metrics = RolloutMetrics(actions=actions, feasible=feasible)

        assert metrics.actions is not None
        assert metrics.feasible is not None


class TestRolloutMetricsClone:
    """Test RolloutMetrics cloning."""

    def test_clone_empty(self):
        """Test cloning empty metrics."""
        metrics = RolloutMetrics()
        cloned = metrics.clone()

        assert cloned.feasible is None
        assert cloned.convergence is None or cloned.convergence.is_empty()

    def test_clone_with_feasible_tensor(self, cuda_device_cfg):
        """Test cloning with feasible tensor."""
        feasible = torch.tensor([True, False], device=cuda_device_cfg.device)
        metrics = RolloutMetrics(feasible=feasible)

        cloned = metrics.clone()

        assert torch.allclose(cloned.feasible.to(torch.float32), feasible.to(torch.float32))

        # Verify independence
        metrics.feasible[0] = False
        assert cloned.feasible[0] == True

    def test_clone_with_feasible_bool(self):
        """Test cloning with feasible bool."""
        metrics = RolloutMetrics(feasible=True)
        cloned = metrics.clone()

        assert cloned.feasible is True


class TestRolloutMetricsCopy:
    """Test RolloutMetrics copy operations."""

    def test_copy_at_batch_seed_indices(self, cuda_device_cfg):
        """Test copying at specific batch and seed indices."""
        actions1 = torch.zeros(4, 8, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        actions2 = torch.ones(4, 8, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        metrics1 = RolloutMetrics(actions=actions1)
        metrics2 = RolloutMetrics(actions=actions2)

        batch_idx = torch.tensor([0, 2], device=cuda_device_cfg.device)
        seed_idx = torch.tensor([1, 3], device=cuda_device_cfg.device)

        metrics1.copy_at_batch_seed_indices(metrics2, batch_idx, seed_idx)

        # Check that specified indices were copied
        assert torch.allclose(
            metrics1.actions[0, 1],
            torch.ones(3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )
        assert torch.allclose(
            metrics1.actions[2, 3],
            torch.ones(3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )

        # Check that other indices remain unchanged
        assert torch.allclose(
            metrics1.actions[1, 1],
            torch.zeros(3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )

    def test_copy_only_index(self, cuda_device_cfg):
        """Test copying at a specific index."""
        actions1 = torch.zeros(4, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        actions2 = torch.ones(4, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        metrics1 = RolloutMetrics(actions=actions1)
        metrics2 = RolloutMetrics(actions=actions2)

        metrics1.copy_only_index(metrics2, 2)

        # Check that index 2 was copied
        assert torch.allclose(
            metrics1.actions[2],
            torch.ones(5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )

        # Check that other indices remain unchanged
        assert torch.allclose(
            metrics1.actions[0],
            torch.zeros(5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )
        assert torch.allclose(
            metrics1.actions[1],
            torch.zeros(5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        )


class TestCostCollectionSumAutograd:
    """Test CostCollectionSum autograd function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autograd test")
    def test_forward_pass(self, cuda_device_cfg):
        """Test forward pass of CostCollectionSum."""
        # Create cost tensors
        cost1 = torch.ones(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype, requires_grad=True)
        cost2 = torch.ones(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype, requires_grad=True) * 2.0

        # Create gradient output tensors
        grad1 = torch.ones_like(cost1)
        grad2 = torch.ones_like(cost2)

        # Apply the custom function
        result = CostCollectionSum.apply(cost1, cost2, grad1, grad2)

        # Result should sum costs over specified dimensions
        assert result.shape[0] == 2  # Batch dimension preserved

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autograd test")
    def test_backward_pass(self, cuda_device_cfg):
        """Test backward pass of CostCollectionSum."""
        # Create cost tensors with requires_grad (use only leaf tensors)
        cost1 = torch.ones(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype, requires_grad=True)
        cost2 = torch.randn(2, 5, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype, requires_grad=True)

        # Create gradient output tensors
        grad1 = torch.ones_like(cost1)
        grad2 = torch.ones_like(cost2)

        # Apply the custom function
        result = CostCollectionSum.apply(cost1, cost2, grad1, grad2)

        # Compute gradients
        result.sum().backward()

        # Gradients should be computed (note: the custom backward uses the grad_out values we provided)
        assert cost1.grad is not None
        assert cost2.grad is not None
        # The custom backward passes through the grad_out_values we provided
        assert torch.allclose(cost1.grad, grad1)
        assert torch.allclose(cost2.grad, grad2)

