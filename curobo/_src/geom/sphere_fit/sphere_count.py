# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Volume-based heuristic for automatic sphere count estimation."""

# Standard Library
from __future__ import annotations

# Third Party
import numpy as np
import trimesh

#: Internal baseline: 1 sphere per 15 cm³ of bounding-box volume.
_BASE_DENSITY = 1.0 / 15.0

_MIN_SPHERES = 3
_MAX_SPHERES = 100


def estimate_sphere_count(
    mesh: trimesh.Trimesh,
    sphere_density: float = 1.0,
) -> int:
    """Estimate the number of spheres needed to approximate a mesh.

    Uses bounding-box volume scaled by a density multiplier, clamped to
    ``[3, 100 * sphere_density]``.  Both the raw count and the per-link cap
    scale with *sphere_density*, so ``density=0.5`` halves the maximum and
    ``density=2.0`` doubles it.

    Args:
        mesh: Input triangle mesh.
        sphere_density: Dimensionless density multiplier.  ``1.0`` (default)
            gives a balanced sphere count; ``2.0`` doubles it; ``0.5`` halves
            it.  Practical range is ``0.1`` to ``10.0``.

    Returns:
        Estimated sphere count.
    """
    bbox_vol_cm3 = float(np.prod((mesh.bounds[1] - mesh.bounds[0]) * 100))
    n = int(sphere_density * _BASE_DENSITY * bbox_vol_cm3)
    max_spheres = max(int(_MAX_SPHERES * sphere_density), _MIN_SPHERES)
    return max(min(max_spheres, n), _MIN_SPHERES)
