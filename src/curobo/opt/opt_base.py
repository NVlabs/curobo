#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Base module for Optimization."""
from __future__ import annotations

# Standard Library
import time
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.rollout.rollout_base import Goal, RolloutBase
from curobo.types.base import TensorDeviceType
from curobo.util.logger import log_info
from curobo.util.torch_utils import is_cuda_graph_available


@dataclass
class OptimizerConfig:
    """Configuration for an :meth:`Optimizer`."""

    #: Number of optimization variables per timestep.
    d_action: int

    #: Lower bound for optimization variables.
    action_lows: List[float]

    #: Higher bound for optimization variables
    action_highs: List[float]

    #:
    action_horizon: int

    #: Number of timesteps in optimization state, total variables = d_action * horizon
    horizon: int

    #: Number of iterations to run optimization
    n_iters: int

    #: Number of iterations to run optimization during the first instance. Setting to None will
    #: use n_iters. This parameter is useful in MPC like settings where we need to run many
    #: iterations during initialization (cold start) and then run only few iterations (warm start).
    cold_start_n_iters: Union[int, None]

    #: Rollout function to use for computing cost, given optimization variables.
    rollout_fn: RolloutBase

    #: Tensor device to use for optimization.
    tensor_args: TensorDeviceType

    #: Capture optimization iteration in a cuda graph and replay graph instead of eager execution.
    #: Enabling this can make optimization 10x faster. But changing control flow, tensor
    #: shapes, or problem type is not allowed.
    use_cuda_graph: bool

    #: Record debugging data such as optimization variables, and cost at every iteration. Enabling
    #: this will disable cuda graph.
    store_debug: bool

    #: Use this to record additional attributes from rollouts.
    debug_info: Any

    #: Number of parallel problems to optimize.
    n_problems: int

    #: Number of particles to use per problem. Common optimization solvers use many particles to
    #: optimize a single problem. E.g., MPPI rolls out many parallel samples and computes a weighted
    #: mean. In cuRobo, Quasi-Newton solvers use particles to run many line search magnitudes.
    #: Total optimization batch size = n_problems * num_particles.
    num_particles: Union[int, None]

    #: Synchronize device before computing solver time.
    sync_cuda_time: bool

    #: Matmul with a Sparse tensor is used to create particles for each problem index to save memory
    #: and compute. Some versions of pytorch do not support coo sparse, specifically during
    #: torch profile runs. Set this to False to use a standard tensor.
    use_coo_sparse: bool

    def __post_init__(self):
        object.__setattr__(self, "action_highs", self.tensor_args.to_device(self.action_highs))
        object.__setattr__(self, "action_lows", self.tensor_args.to_device(self.action_lows))
        if self.use_cuda_graph:
            self.use_cuda_graph = is_cuda_graph_available()
        if self.num_particles is None:
            self.num_particles = 1
        if self.cold_start_n_iters is None:
            self.cold_start_n_iters = self.n_iters

    @staticmethod
    def create_data_dict(
        data_dict: Dict,
        rollout_fn: RolloutBase,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        child_dict: Optional[Dict] = None,
    ):
        """Helper function to create dictionary from optimizer parameters and rollout class.

        Args:
            data_dict: optimizer parameters dictionary.
            rollout_fn: rollout function.
            tensor_args: tensor cuda device.
            child_dict: new dictionary where parameters will be stored.

        Returns:
            Dictionary with parameters to create a :meth:`OptimizerConfig`
        """
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict["d_action"] = rollout_fn.d_action
        child_dict["action_lows"] = rollout_fn.action_bound_lows
        child_dict["action_highs"] = rollout_fn.action_bound_highs
        child_dict["rollout_fn"] = rollout_fn
        child_dict["tensor_args"] = tensor_args
        child_dict["horizon"] = rollout_fn.horizon
        child_dict["action_horizon"] = rollout_fn.action_horizon

        if "num_particles" not in child_dict:
            child_dict["num_particles"] = 1
        return child_dict


class Optimizer(OptimizerConfig):
    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        """Base optimization solver class

        Args:
            config: Initialized with parameters from a dataclass.
        """
        if config is not None:
            super().__init__(**vars(config))
        self.opt_dt = 0.0
        self.COLD_START = True
        self.update_nproblems(self.n_problems)
        self._batch_goal = None
        self._rollout_list = None
        self.debug = []
        self.debug_cost = []
        self.cu_opt_graph = None

    def optimize(self, opt_tensor: torch.Tensor, shift_steps=0, n_iters=None) -> torch.Tensor:
        """Find a solution through optimization given the initial values for variables.

        Args:
            opt_tensor: Initial value of optimization variables.
                        Shape: [n_problems, action_horizon, d_action]
            shift_steps: Shift variables along action_horizon. Useful in mpc warm-start setting.
            n_iters: Override number of iterations to run optimization.

        Returns:
            Optimized values returned as a tensor of shape [n_problems, action_horizon, d_action].
        """
        if self.COLD_START:
            n_iters = self.cold_start_n_iters
            self.COLD_START = False
        st_time = time.time()
        out = self._optimize(opt_tensor, shift_steps, n_iters)
        if self.sync_cuda_time:
            torch.cuda.synchronize(device=self.tensor_args.device)
        self.opt_dt = time.time() - st_time
        return out

    def update_params(self, goal: Goal):
        """Update parameters in the :meth:`curobo.rollout.rollout_base.RolloutBase` instance.

        Args:
            goal: parameters to update rollout instance.
        """
        with profiler.record_function("OptBase/batch_goal"):
            if self._batch_goal is not None:
                self._batch_goal.copy_(goal, update_idx_buffers=True)  # why True?
            else:
                self._batch_goal = goal.repeat_seeds(self.num_particles)
        self.rollout_fn.update_params(self._batch_goal)

    def reset(self):
        """Reset optimizer."""
        self.rollout_fn.reset()
        self.debug = []
        self.debug_cost = []

    def update_nproblems(self, n_problems: int):
        """Update the number of problems that need to be optimized.

        Args:
            n_problems: number of problems.
        """
        assert n_problems > 0
        self._update_problem_kernel(n_problems, self.num_particles)
        self.n_problems = n_problems

    def get_nproblem_tensor(self, x):
        """This function takes an input tensor of shape (n_problem,....) and converts it into
        (n_particles,...).
        """

        # create a tensor
        nx_problem = self.problem_kernel_ @ x

        return nx_problem

    def reset_seed(self):
        """Reset seeds."""
        return True

    def reset_cuda_graph(self):
        """Reset CUDA Graphs. This does not work, workaround is to create a new instance."""
        if self.use_cuda_graph:
            self.cu_opt_init = False
        else:
            log_info("Cuda Graph was not enabled")
        self._batch_goal = None
        if self.cu_opt_graph is not None:
            self.cu_opt_graph.reset()
        self.rollout_fn.reset_cuda_graph()

    def reset_shape(self):
        """Reset any flags in rollout class. Useful to reinitialize tensors for a new shape."""
        self.rollout_fn.reset_shape()
        self._batch_goal = None

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        """Get all instances of Rollout class in the optimizer."""
        if self._rollout_list is None:
            self._rollout_list = [self.rollout_fn]
        return self._rollout_list

    @abstractmethod
    def _optimize(self, opt_tensor: torch.Tensor, shift_steps=0, n_iters=None) -> torch.Tensor:
        """Implement this function in a derived class containing the solver.

        Args:
            opt_tensor: Initial value of optimization variables.
                        Shape: [n_problems, action_horizon, d_action]
            shift_steps: Shift variables along action_horizon. Useful in mpc warm-start setting.
            n_iters: Override number of iterations to run optimization.

        Returns:
            Optimized variables in tensor shape [action_horizon, d_action].
        """
        return opt_tensor

    @abstractmethod
    def _shift(self, shift_steps=0):
        """Shift the variables in the solver to hotstart the next timestep.

        Args:
            shift_steps: Number of timesteps to shift.
        """
        return

    def _update_problem_kernel(self, n_problems: int, num_particles: int):
        """Update matrix used to map problem index to number of particles.

        Args:
            n_problems: Number of optimization problems.
            num_particles: Number of particles per problem.
        """
        log_info(
            "Updating problem kernel [n_problems: "
            + str(n_problems)
            + " , num_particles: "
            + str(num_particles)
            + " ]"
        )

        self.problem_col = torch.arange(
            0, n_problems, step=1, dtype=torch.long, device=self.tensor_args.device
        )
        self.n_select_ = torch.tensor(
            [x * n_problems + x for x in range(n_problems)],
            device=self.tensor_args.device,
            dtype=torch.long,
        )

        # create sparse tensor:
        sparse_indices = []
        for i in range(n_problems):
            sparse_indices.extend([[i * num_particles + x, i] for x in range(num_particles)])

        sparse_values = torch.ones(len(sparse_indices))
        sparse_indices = torch.tensor(sparse_indices)
        if self.use_coo_sparse:
            self.problem_kernel_ = torch.sparse_coo_tensor(
                sparse_indices.t(),
                sparse_values,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        else:
            self.problem_kernel_ = torch.sparse_coo_tensor(
                sparse_indices.t(),
                sparse_values,
                device="cpu",
                dtype=self.tensor_args.dtype,
            )
            self.problem_kernel_ = self.problem_kernel_.to_dense().to(
                device=self.tensor_args.device
            )
        self._problem_seeds = self.num_particles
