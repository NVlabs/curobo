# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass


@dataclass
class FilterCoeff:
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0

