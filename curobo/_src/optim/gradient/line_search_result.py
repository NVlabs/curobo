# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module contains the LineSearchResult class, which is used to store the result of a line search."""

from dataclasses import dataclass

from curobo._src.optim.gradient.line_search_state import LineSearchState


@dataclass
class LineSearchResult:
    """This class is used to store the result of a line search."""

    #: The state selected by line search criteria
    selected_state: LineSearchState

    #: The state to use for next iteration (may include noise/fallback)
    exploration_state: LineSearchState
