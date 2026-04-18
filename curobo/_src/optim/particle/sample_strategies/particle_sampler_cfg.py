# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for particle sampling strategies."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class ParticleSamplerCfg:
    """Configuration for particle sampling strategies.

    This configuration supports various sampling strategies including:
    - Halton sequences for low-discrepancy sampling
    - Random sampling
    - Knot-based B-spline interpolation
    - STOMP trajectory optimization
    """

    device_cfg: DeviceCfg
    fixed_samples: bool = True
    sample_ratio: Dict[str, float] = field(
        default_factory=lambda: (
            {"halton": 1.0, "halton-knot": 0.0, "random": 0.0, "random-knot": 0.0, "stomp": 0.0}
        )
    )
    seed: int = 0
    filter_coeffs: Optional[List[float]] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    n_knots: int = 3
    scale_tril: Optional[float] = None
    covariance_matrix: Optional[torch.tensor] = None
    sample_method: str = "halton"
    degree: int = 3  # bspline
    stencil_type: str = "3point"
