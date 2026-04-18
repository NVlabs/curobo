# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

# Third Party
import torch
import torch.autograd.profiler as profiler
from torch.autograd import Function

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.types.tensor import (
    T_BHDOF_float,
    T_BHValue_bool,
    T_BHValue_float,
    T_BValue_bool,
    T_BValue_float,
)
from curobo._src.util.helpers import list_idx_if_not_none
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import cat_sum


class CostCollectionSum(Function):
    @staticmethod
    def forward(ctx, *values: torch.Tensor):
        # first half values are the values to sum, second half are the grad_out_values
        sum_values = values[: len(values) // 2]
        grad_out_values = values[len(values) // 2 :]

        sum_horizon = True
        if sum_horizon:
            sum_dim = (1, 2)
        else:
            sum_dim = 2

        sum_tensor = cat_sum(sum_values, sum_dim=sum_dim)
        ctx.grad_out_tuple = grad_out_values

        return sum_tensor

    @staticmethod
    def backward(ctx, grad_out_sum: torch.Tensor):
        grad_out_values = ctx.grad_out_tuple

        # add two nones to the grad_out_values which is a tuple
        grad_out_values_list = list(grad_out_values)
        grad_out_values_list += [None for _ in range(len(grad_out_values_list))]

        return tuple(grad_out_values_list)


@dataclass
class CostCollection:
    """Collection of related costs or constraints in an optimization problem"""

    values: List[torch.Tensor] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    weights: List[torch.Tensor] = field(default_factory=list)
    sq_weights: List[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        value: torch.Tensor,
        name: str,
        weight: Optional[torch.Tensor] = None,
        sq_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a cost with its name and optional weights"""
        self.values.append(value)
        self.names.append(name)

        if weight is not None:
            self.weights.append(weight)

        if sq_weight is not None:
            self.sq_weights.append(sq_weight)

    def get_sum(self, sum_horizon: bool = True) -> torch.Tensor:
        """Calculate the sum of all costs in this collection"""
        if not self.values:
            # Return appropriate zero tensor based on expected shape
            return torch.zeros(1, device=torch.device("cuda:0"))

        # Sum the values (implementation would depend on tensor shapes)
        # This is a placeholder - actual implementation would handle batching, etc.
        result = torch.stack(self.values).sum(dim=0)

        if sum_horizon:
            result = result.sum(dim=1)

        return result

    def is_empty(self) -> bool:
        """Check if the collection is empty"""
        return len(self.values) == 0

    def clone(self) -> CostCollection:
        """Create a deep copy of this collection"""
        new_collection = CostCollection()

        # Clone all values and their associated metadata
        for i, value in enumerate(self.values):
            new_collection.add(
                value.clone(),
                self.names[i],
                self.weights[i].clone() if i < len(self.weights) else None,
                self.sq_weights[i].clone() if i < len(self.sq_weights) else None,
            )

        return new_collection

    def merge(self, other: CostCollection) -> None:
        """Merge another CostCollection into this one"""
        if not isinstance(other, CostCollection):
            log_and_raise("Cannot merge non-CostCollection object")

        # Merge values
        self.values.extend(other.values)
        self.names.extend(other.names)
        self.weights.extend(other.weights)
        self.sq_weights.extend(other.sq_weights)

    def copy_at_batch_seed_indices(
        self, other: CostCollection, batch_idx: torch.Tensor, seed_idx: torch.Tensor
    ):
        """Copy cost collection at specific batch and seed indices"""
        for i in range(len(self.values)):
            self.values[i][batch_idx, seed_idx] = other.values[i][batch_idx, seed_idx]
        for i in range(len(self.weights)):
            self.weights[i][batch_idx, seed_idx] = other.weights[i][batch_idx, seed_idx]
        for i in range(len(self.sq_weights)):
            self.sq_weights[i][batch_idx] = other.sq_weights[i][batch_idx]
        return self

    def copy_only_index(self, other: CostCollection, index: int):
        """Copy cost collection at specific indices"""
        for i in range(len(self.values)):
            self.values[i][index] = other.values[i][index]
        for i in range(len(self.weights)):
            self.weights[i][index] = other.weights[i][index]
        # for i in range(len(self.sq_weights)):
        #    self.sq_weights[i][index] = other.sq_weights[i][index]
        return self


@dataclass
class CostsAndConstraints:
    costs: CostCollection = field(default_factory=CostCollection)
    constraints: CostCollection = field(default_factory=CostCollection)
    hybrid_costs_constraints: CostCollection = field(default_factory=CostCollection)

    _grad_out_values: List[torch.Tensor] = field(default_factory=list)

    def copy_at_batch_seed_indices(
        self, other: CostsAndConstraints, batch_idx: torch.Tensor, seed_idx: torch.Tensor
    ):
        """Copy costs and constraints at specific batch and seed indices"""
        self.costs.copy_at_batch_seed_indices(other.costs, batch_idx, seed_idx)
        self.constraints.copy_at_batch_seed_indices(other.constraints, batch_idx, seed_idx)
        self.hybrid_costs_constraints.copy_at_batch_seed_indices(
            other.hybrid_costs_constraints, batch_idx, seed_idx
        )
        return self

    def get_sum_cost(
        self,
        sum_horizon: bool = False,
        include_all_hybrid: bool = True,
        include_from_hybrid: List[str] = [],
    ) -> Optional[T_BHDOF_float]:
        # Start with regular costs
        cost_values = self.costs.values.copy()

        # Add hybrid costs if requested
        if include_all_hybrid:
            cost_values.extend(self.hybrid_costs_constraints.values)
        elif include_from_hybrid:
            for name in include_from_hybrid:
                if name in self.hybrid_costs_constraints.names:
                    idx = self.hybrid_costs_constraints.names.index(name)
                    cost_values.append(self.hybrid_costs_constraints.values[idx])

        if not cost_values:
            return None

        if sum_horizon:
            # cost values is a list of tensors with shape [batch_size, horizon, -1]
            # after stacking, it will be [batch_size, horizon, costs]
            sum_cost = cat_sum(cost_values, sum_dim=(1, 2))
        else:
            # cost values is a list of tensors with shape [batch_size, horizon, -1]
            # after stacking, it will be [batch_size, horizon, costs]
            sum_cost = cat_sum(cost_values, sum_dim=(2))

        return sum_cost

    def get_sum_constraint(
        self,
        sum_horizon: bool = False,
        include_all_hybrid: bool = True,
        include_from_hybrid: List[str] = [],
    ) -> Optional[T_BHValue_float]:
        # Start with regular constraints
        constraint_values = self.constraints.values.copy()

        # Add hybrid constraints if requested
        if include_all_hybrid:
            constraint_values.extend(self.hybrid_costs_constraints.values)
        elif include_from_hybrid:
            for name in include_from_hybrid:
                if name in self.hybrid_costs_constraints.names:
                    idx = self.hybrid_costs_constraints.names.index(name)
                    constraint_values.append(self.hybrid_costs_constraints.values[idx])

        if not constraint_values:
            return None

        if sum_horizon:
            # constraint values is a list of tensors with shape [batch_size, horizon, -1]
            # after stacking, it will be [batch_size, horizon, costs]
            sum_constraint = cat_sum(constraint_values, sum_dim=(1, 2))
        else:
            # constraint values is a list of tensors with shape [batch_size, horizon, costs]
            # after stacking, it will be [batch_size, horizon, costs]
            sum_constraint = cat_sum(constraint_values, sum_dim=(2))
        return sum_constraint

    def get_sum_cost_and_constraint(
        self, sum_horizon: bool = False, include_all_hybrid: bool = True
    ) -> Union[T_BHValue_float, T_BValue_float]:
        all_values = self.costs.values.copy()

        # Check if constraints exist before accessing
        if self.constraints is not None:
            all_values.extend(self.constraints.values)

        if include_all_hybrid:
            all_values.extend(self.hybrid_costs_constraints.values)

        if False:
            if (
                self._grad_out_values is None
                or len(self._grad_out_values) == 0
                or len(self._grad_out_values) != len(all_values)
            ):
                self._grad_out_values = [torch.ones_like(value) for value in all_values]

            sum_tensor = CostCollectionSum.apply(*all_values, *self._grad_out_values)

        if True:
            if sum_horizon:
                # all_values is a list of tensors with shape [batch_size, horizon, costs]
                # after stacking, it will be [batch_size, horizon, costs]
                sum_tensor = cat_sum(all_values, sum_dim=(1, 2))
            else:
                # all_values is a list of tensors with shape [batch_size, horizon, -1]
                # after stacking, it will be [batch_size, horizon, costs]
                sum_tensor = cat_sum(all_values, sum_dim=(2))

        return sum_tensor

    def get_list_costs_and_constraints(self) -> List[T_BHValue_float]:
        all_values = self.costs.values.copy()
        all_values.extend(self.constraints.values)
        all_values.extend(self.hybrid_costs_constraints.values)
        return all_values

    @profiler.record_function("CostsAndConstraints/_get_feasible")
    def get_feasible(
        self,
        sum_horizon: bool = False,
        include_all_hybrid: bool = True,
        include_from_hybrid: List[str] = [],
    ) -> Union[Union[T_BHValue_bool, T_BValue_bool], bool]:
        sum_constraint = self.get_sum_constraint(
            sum_horizon, include_all_hybrid, include_from_hybrid
        )
        if sum_constraint is None:
            return True
        return sum_constraint <= 0.0

    def clone(self) -> CostsAndConstraints:
        """Create a deep copy of this container"""
        return CostsAndConstraints(
            costs=self.costs.clone(),
            constraints=self.constraints.clone(),
            hybrid_costs_constraints=self.hybrid_costs_constraints.clone(),
        )

    def get_constraint_weights(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        constraint_weights = []
        constraint_sq_weights = []

        for i in range(len(self.constraints.weights)):
            if self.constraints.weights[i] is not None:
                constraint_weights.append(self.constraints.weights[i])
                constraint_sq_weights.append(self.constraints.sq_weights[i])
        for i in range(len(self.hybrid_costs_constraints.weights)):
            if self.hybrid_costs_constraints.weights[i] is not None:
                constraint_weights.append(self.hybrid_costs_constraints.weights[i])
                constraint_sq_weights.append(self.hybrid_costs_constraints.sq_weights[i])
        return constraint_weights, constraint_sq_weights

    def copy_only_index(self, other: CostsAndConstraints, index: int):
        """Copy costs and constraints at specific indices"""
        self.costs.copy_only_index(other.costs, index)
        self.constraints.copy_only_index(other.constraints, index)
        self.hybrid_costs_constraints.copy_only_index(other.hybrid_costs_constraints, index)
        return self


@dataclass
class RolloutResult(Sequence):
    """Evaluation result that stores the action sequence, state, costs and constraints.

    This class encapsulates the results of a rollout evaluation, storing the action sequence,
    resulting state, costs, constraints, and optional debug information. It is used by optimization
    algorithms (e.g., :class:`curobo.opt.opt_base.OptBase`) to minimize costs while satisfying
    constraints during trajectory optimization.

    The class provides a structured way to pass rollout results between components in the
    optimization pipeline, making it easier to analyze and debug the optimization process.
    """

    #: Shape: [batch_size, horizon, degrees_of_freedom]
    actions: Optional[T_BHDOF_float] = None

    #: Contains lists of cost terms, constraint terms, and hybrid cost/constraint terms
    #: computed during the rollout.
    costs_and_constraints: Optional[CostsAndConstraints] = None

    #: The resulting robot state after applying the action sequence.
    state: Optional[JointState] = None

    #: Additional information useful for debugging or analysis.
    debug: Optional[Any] = None

    def __getitem__(self, idx):
        d_list = [self.actions, self.costs_and_constraints, self.state]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return RolloutResult(idx_vals[0], idx_vals[1], idx_vals[2], self.debug)

    def __len__(self):
        if self.actions is not None:
            return self.actions.shape[0]
        else:
            return -1

    def clone(self):
        return RolloutResult(
            actions=self.actions.clone() if self.actions is not None else None,
            costs_and_constraints=(
                self.costs_and_constraints.clone()
                if self.costs_and_constraints is not None
                else None
            ),
            state=self.state.clone() if self.state is not None else None,
            debug=self.debug.clone() if self.debug is not None else None,
        )


@dataclass
class RolloutMetrics(RolloutResult):
    feasible: Optional[Union[Union[T_BHValue_bool, T_BValue_bool], bool]] = None
    convergence: CostCollection = field(default_factory=CostCollection)

    def __getitem__(self, idx):
        d_list = [self.costs_and_constraints, self.feasible, self.convergence, self.state]
        idx_vals = list_idx_if_not_none(d_list, idx)
        log_and_raise("not implemented")
        return RolloutMetrics(idx_vals[0], idx_vals[1], idx_vals[2])

    def __len__(self):
        if self.cost is not None:
            return self.cost.shape[0]
        else:
            return -1

    def clone(self, **kwargs):
        """Create a deep copy of this RolloutMetrics instance.

        Args:
            clone_state: If True, would clone the state (not implemented yet)

        Returns:
            A new RolloutMetrics instance with cloned data

        Raises:
            NotImplementedError: If clone_state is True
        """
        # Use super().clone() to efficiently handle base class attributes
        base = super().clone()

        # Construct directly with all attributes in one statement
        return RolloutMetrics(
            actions=base.actions,
            costs_and_constraints=base.costs_and_constraints,
            state=base.state,
            debug=base.debug,
            feasible=(
                self.feasible.clone() if isinstance(self.feasible, torch.Tensor) else self.feasible
            ),
            convergence=self.convergence.clone() if self.convergence is not None else None,
        )

    def copy_at_batch_seed_indices(
        self, other: RolloutMetrics, batch_idx: torch.Tensor, seed_idx: torch.Tensor
    ):
        """Copy rollout metrics at specific batch and seed indices"""
        if self.actions is not None:
            self.actions[batch_idx, seed_idx] = other.actions[batch_idx, seed_idx]
        if self.state is not None:
            self.state.copy_at_batch_seed_indices(other.state, batch_idx, seed_idx)
        if self.feasible is not None:
            self.feasible[batch_idx, seed_idx] = other.feasible[batch_idx, seed_idx]
        if self.convergence is not None:
            self.convergence.copy_at_batch_seed_indices(other.convergence, batch_idx, seed_idx)
        if self.costs_and_constraints is not None:
            self.costs_and_constraints.copy_at_batch_seed_indices(
                other.costs_and_constraints, batch_idx, seed_idx
            )

        return self

    def copy_only_index(self, other: RolloutMetrics, index: int):
        """Copy rollout metrics at specific indices"""
        if self.actions is not None:
            self.actions[index] = other.actions[index]
        if self.state is not None:
            self.state.copy_only_index(other.state, index)
        if self.feasible is not None:
            self.feasible[index] = other.feasible[index]
        if self.convergence is not None:
            self.convergence.copy_only_index(other.convergence, index)
        if self.costs_and_constraints is not None:
            self.costs_and_constraints.copy_only_index(other.costs_and_constraints, index)
        return self

    def get_only_batch_seed_indices(self, batch_idx: torch.Tensor, seed_idx: torch.Tensor):
        return None
