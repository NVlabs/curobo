# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Banding Algorithm for 3D Euclidean Distance Transform.

Exact EDT via three separable phases (Z-flood, Y-Maurer, X-Maurer
with transpose), requiring only 5 kernel launches (3 unique kernels).

Grid layout: (nx, ny, nz) - X slowest, Z fastest.
"""

from typing import Tuple

import torch

from curobo._src.curobolib.backends import pba as pba_cu
from curobo._src.perception.mapper.esdf.kernel.wp_jfa import validate_grid_size
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import profile_class_methods


@profile_class_methods
class ParallelBandingEDT:
    """Exact 3D EDT using the Parallel Banding Algorithm.

    Computes an exact Voronoi diagram in 5 fixed kernel launches,
    replacing the iterative JFA approach.

    Args:
        grid_shape: Grid dimensions (nx, ny, nz) - X slowest, Z fastest.
        voxel_size: Size of each voxel [m].
        device: Computation device (must be CUDA).
        m3: Color-axis thread-block Y dimension (tuning, default 2).
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        device: torch.device,
        m3: int = 2,
    ):
        if device.type != "cuda":
            raise ValueError(f"ParallelBandingEDT requires CUDA device, got {device}")

        self.grid_shape = grid_shape
        self.voxel_size = float(voxel_size)
        self.device = device
        self.m3 = m3
        self.n_voxels = validate_grid_size(grid_shape, "ParallelBandingEDT")

        # Scratch buffer (same size as site_index)
        self._buffer = torch.empty(
            self.n_voxels, dtype=torch.int32, device=device,
        )

        mem_gb = self.n_voxels * 8 / (1024**3)
        log_info(
            f"ParallelBandingEDT: {grid_shape}, 5 kernel launches (exact), "
            f"{mem_gb:.2f} GB"
        )

    def propagate(self, site_index: torch.Tensor) -> None:
        """Run PBA+ propagation on site_index in-place.

        Launches 5 CUDA kernels:
            FloodZ -> MaurerAxis -> ColorAxis -> MaurerAxis -> ColorAxis

        Args:
            site_index: Packed site indices (nx, ny, nz) or (nx*ny*nz,),
                dtype int32. Modified in-place.
        """
        nx, ny, nz = self.grid_shape
        pba_cu.launch_pba3d(
            site_index.view(-1),
            self._buffer,
            nx, ny, nz,
            m3=self.m3,
        )
