# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Types and enums for the sphere fitting module."""

# Standard Library
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Third Party
import numpy as np
import torch

if TYPE_CHECKING:
    import trimesh


class SphereFitType(Enum):
    """Supported sphere fit types.

    See :ref:`attach_object_note` for more details.
    """

    #: Samples the surface of the mesh evenly, fixed radius. Fast fallback.
    SURFACE = "surface"
    #: Bbox grid + SDF filtering. Interior voxels with inscribed radii.
    VOXEL = "voxel"
    #: Voxel-grid seeding + Adam optimization.
    MORPHIT = "morphit"


@dataclass
class SphereFitMetrics:
    """Quality metrics for a sphere fit to a single mesh/link.

    All distance fields are in metres.  Fractions are in ``[0, 1]``.
    """

    num_spheres: int = 0
    """Number of spheres fitted to this link."""
    coverage: float = 0.0
    """Fraction of interior sample points covered by at least one sphere."""
    protrusion: float = 0.0
    """Fraction of sphere-surface sample points outside the mesh."""
    protrusion_dist_mean: float = 0.0
    """Mean distance (m) of protruding sphere-surface points to the mesh."""
    protrusion_dist_p95: float = 0.0
    """95th-percentile distance (m) of protruding points to the mesh."""
    surface_gap_mean: float = 0.0
    """Mean gap (m) from mesh surface samples to nearest sphere surface."""
    surface_gap_p95: float = 0.0
    """95th-percentile gap (m) from mesh surface samples to nearest sphere surface."""
    max_uncovered_gap: float = 0.0
    """Maximum gap (m) from mesh surface samples to nearest sphere surface."""
    volume_ratio: float = 0.0
    """Total sphere volume divided by mesh volume."""


@dataclass
class SphereFitResult:
    """Result of a sphere fitting operation.

    Always contains the fitted sphere geometry (``centers``, ``radii``).
    Quality metrics are populated in :attr:`metrics` only when
    ``compute_metrics=True`` is passed to :func:`fit_spheres_to_mesh`.
    """

    centers: torch.Tensor
    """Sphere centre positions, shape ``(N, 3)``, float32, CUDA."""
    radii: torch.Tensor
    """Sphere radii, shape ``(N,)``, float32, CUDA."""
    num_spheres: int = 0
    """Number of spheres."""

    metrics: Optional[SphereFitMetrics] = None
    """Quality metrics.  ``None`` until :func:`fit_spheres_to_mesh` is called
    with ``compute_metrics=True``."""

    fit_time_s: Optional[float] = None
    """Wall-clock time (seconds) for the fitting call."""

    used_mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False)
    """The mesh that was actually used for fitting.  May differ from the input
    when hollow-mesh detection replaces it with its convex hull."""

    history: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    """Per-iteration snapshots of ``(centers, radii)`` recorded during
    optimization.  Empty for non-iterative methods (SURFACE, VOXEL)."""

    debug_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Diagnostic metadata from the fitting pipeline.  Typical keys:

    - ``used_convex_hull`` (*bool*): convex hull was substituted for the input.
    - ``auto_n_spheres`` (*bool*): sphere count was determined automatically.
    - ``requested_n_spheres`` (*int | None*): caller's original ``num_spheres``.
    - ``resolved_n_spheres`` (*int*): sphere count actually used.
    - ``fallback_used`` (*bool*): primary method failed, distance-field fallback ran.
    - ``fit_type`` (*str*): the ``SphereFitType`` value used.
    """
