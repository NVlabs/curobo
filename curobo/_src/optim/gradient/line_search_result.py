# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module contains the LineSearchResult class, which is used to store the result of a line search."""

from dataclasses import dataclass

import torch

from curobo._src.optim.gradient.line_search_state import LineSearchState


@dataclass
class LineSearchResult:
    """This class is used to store the result of a line search."""

    #: The state selected by line search criteria
    selected_state: LineSearchState

    #: The state to use for next iteration (may include noise/fallback)
    exploration_state: LineSearchState

    #: Local exploration probe index, or -1 when the baseline is retained.
    line_search_index: torch.Tensor

    #: Per-problem line-search outcome code.
    line_search_status: torch.Tensor

    #: Whether the exploration transition may update quasi-Newton history.
    curvature_update_valid: torch.Tensor
