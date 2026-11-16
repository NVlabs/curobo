# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Callable, List, Union

import torch

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class LineSearchContext:
    """Context needed for line search algorithms."""

    # Tensor arguments for device/dtype
    device_cfg: DeviceCfg

    # Line search parameters. Scales follow the sorted, strictly positive probe contract.
    line_search_scale: Union[List[float], torch.Tensor]

    line_search_c_1: float
    line_search_c_2: float

    # Dimension information
    num_problems: int
    opt_dim: int
    action_horizon: int
    action_dim: int

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

    #: Retry failed parallel probe fans at one tenth the displacement.
    line_search_backtracking: bool = False
    #: Persistent per-problem backtracking multiplier. Shape: (num_problems,).
    beta: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.line_search_scale, list):
            self.line_search_scale = self._create_box_line_search(self.line_search_scale)
        else:
            self.line_search_scale = self.line_search_scale.to(
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            if self.line_search_scale.ndim == 4 and (
                self.line_search_scale.shape[0] != 1 or self.line_search_scale.shape[2:] != (1, 1)
            ):
                raise ValueError(
                    "line_search_scale must be a list, a 1D tensor, or a tensor with "
                    "shape (1, n_linesearch, 1, 1)"
                )
            if self.line_search_scale.ndim not in (1, 4):
                raise ValueError(
                    "line_search_scale must be a list, a 1D tensor, or a tensor with "
                    "shape (1, n_linesearch, 1, 1)"
                )
            self.line_search_scale = self.line_search_scale.reshape(-1).contiguous()
        if torch.any(self.line_search_scale <= 0.0):
            raise ValueError("line_search_scale must contain only positive values")
        self.update_num_problems(self.num_problems)

    @property
    def n_linesearch(self):
        return self.line_search_scale.shape[0]

    def get_line_search_scale(self) -> torch.Tensor:
        """Return the shared ascending positive probe fan with shape ``(n_linesearch,)``."""
        return self.line_search_scale

    def update_num_problems(self, num_problems: int) -> None:
        self.num_problems = num_problems
        if not hasattr(self, "beta") or self.beta.shape != (num_problems,):
            self.beta = torch.ones(
                num_problems, device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )

    def _create_box_line_search(self, line_search_scale: List[float]) -> torch.Tensor:
        """Create a box line search.

        Args:
            line_search_scale (List[float]): should have n values

        Returns:
            torch.Tensor: Shared scale vector with shape ``(n,)``.
        """
        gpu_line_search_scale = torch.as_tensor(
            line_search_scale, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        return gpu_line_search_scale
