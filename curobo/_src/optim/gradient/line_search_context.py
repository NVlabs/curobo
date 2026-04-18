# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Callable, List, Union

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class LineSearchContext:
    """Context needed for line search algorithms."""

    # Tensor arguments for device/dtype
    device_cfg: DeviceCfg

    # Line search parameters
    line_search_scale: Union[List[float], torch.Tensor]

    line_search_c_1: float
    line_search_c_2: float

    # Dimension information
    num_problems: int
    opt_dim: int
    action_horizon: int
    action_dim: int
    step_scale: float
    fix_terminal_action: bool
    action_horizon_step_max: torch.Tensor

    # CUDA-specific settings
    use_cuda_kernel_line_search: bool

    # Method to compute cost and gradient
    compute_costs_and_gradients: Callable

    #: Number of iterations since last best cost update. Used to check for convergence.
    convergence_iteration: int

    #: Threshold delta difference in cost to store as new best. delta = cost_new - cost_best
    cost_delta_threshold: float

    #: Threshold relative difference in cost to store as new best. rel = (cost_best - cost_new)/ cost_best
    cost_relative_threshold: float

    def __post_init__(self):
        if isinstance(self.line_search_scale, List):
            self.line_search_scale = self._create_box_line_search(self.line_search_scale)
        if self.fix_terminal_action:
            log_and_raise("fix_terminal_action is not supported")
        self.update_num_problems(self.num_problems)

    @property
    def n_linesearch(self):
        return self.line_search_scale.shape[1]

    def update_num_problems(self, num_problems):
        n_line_search = self.n_linesearch
        self.num_problems = num_problems
        self.c_idx = torch.arange(
            0,
            num_problems * n_line_search,
            step=(n_line_search),
            device=self.device_cfg.device,
            dtype=torch.int32,
        )

    def _create_box_line_search(self, line_search_scale: List[float]) -> torch.Tensor:
        """Create a box line search.

        Args:
            line_search_scale (List[float]): should have n values

        Returns:
            torch.Tensor: (1, n, 1, 1)
        """
        gpu_line_search_scale = torch.as_tensor(
            line_search_scale, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        gpu_line_search_scale = gpu_line_search_scale.view(1, len(line_search_scale), 1, 1)
        return gpu_line_search_scale
