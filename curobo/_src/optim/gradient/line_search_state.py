# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from curobo._src.util.logging import log_and_raise


@dataclass
class LineSearchState:
    """Represents the state of optimization at a point (action, cost, gradient triplet) during line search."""

    #: The action. Shape: (num_problems, action_horizon, action_dim)
    action: torch.Tensor

    #: The cost at the action. Shape: (num_problems), data type is float32.
    cost: torch.Tensor

    #: The gradient at the action. Shape: (num_problems, action_horizon, action_dim)
    gradient: torch.Tensor

    #: The index of the state. Shape: (num_problems, n_line_search).
    #: data type is torch.int32
    idxs: torch.Tensor

    def __post_init__(self):
        """Validate tensor shapes."""
        if len(self.action.shape) != 3:
            log_and_raise(
                "Action tensor must have shape (num_problems, action_horizon, action_dim). "
                + f"Got {self.action.shape}"
            )
        if len(self.gradient.shape) != 3:
            log_and_raise(
                "Gradient tensor must have shape (num_problems, action_horizon, action_dim). "
                + f"Got {self.gradient.shape}"
            )
        if len(self.cost.shape) != 1:
            log_and_raise("Cost tensor must have shape (num_problems). " + f"Got {self.cost.shape}")

        # Ensure consistent batch size
        num_problems = self.cost.shape[0]
        if self.action.shape[0] != num_problems:
            log_and_raise(
                "Action batch size " + f"{self.action.shape[0]} != cost batch size {num_problems}"
            )
        if self.gradient.shape[0] != num_problems:
            log_and_raise(
                f"Gradient batch size {self.gradient.shape[0]} != cost batch size {num_problems}"
            )
        if self.idxs.shape[0] != num_problems:
            log_and_raise(f"Idxs batch size {self.idxs.shape[0]} != cost batch size {num_problems}")
