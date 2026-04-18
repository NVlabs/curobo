# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Jump Flooding Algorithm for 3D Euclidean Distance Transform.

Approximate EDT using iterative neighbor propagation with logarithmic
convergence. Uses a 1+JFA+2 pass schedule for accuracy.

Grid layout: (nx, ny, nz) - X slowest, Z fastest.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import warp as wp

from curobo._src.perception.mapper.esdf.kernel.wp_jfa import (
    compute_num_passes,
    jfa_propagate_kernel_18,
    jfa_propagate_kernel_26,
    validate_grid_size,
)
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import profile_class_methods
from curobo._src.util.warp import get_warp_device_stream


@profile_class_methods
class JumpFloodingEDT:
    """3D EDT using the Jump Flooding Algorithm.

    Propagates nearest-site information through a voxel grid using
    geometrically decreasing step sizes. Uses single-buffer chaotic
    relaxation by default for memory efficiency.

    Args:
        grid_shape: Grid dimensions (nx, ny, nz) - X slowest, Z fastest.
        voxel_size: Size of each voxel [m].
        device: Computation device (must be CUDA).
        single_buffer: If True, use chaotic relaxation (halves memory).
        max_distance: Maximum propagation distance [m]. If None, exact.
        neighbors: Connectivity: 18 (default, less bandwidth) or 26.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        device: torch.device,
        single_buffer: bool = True,
        max_distance: Optional[float] = None,
        neighbors: int = 18,
    ):
        if device.type != "cuda":
            raise ValueError(f"JumpFloodingEDT requires CUDA device, got {device}")
        if neighbors not in (18, 26):
            raise ValueError(f"neighbors must be 18 or 26, got {neighbors}")

        self.grid_shape = grid_shape
        self.voxel_size = float(voxel_size)
        self.device = device
        self.single_buffer = single_buffer
        self.neighbors = neighbors
        self._kernel = (
            jfa_propagate_kernel_18 if neighbors == 18 else jfa_propagate_kernel_26
        )
        self.n_voxels = validate_grid_size(grid_shape, "JumpFloodingEDT")

        # Number of JFA passes (log2 of longest axis)
        if max_distance is None:
            self.num_passes = compute_num_passes(grid_shape)
        else:
            max_voxels = int(np.ceil(max_distance / voxel_size))
            self.num_passes = max(1, int(np.ceil(np.log2(max_voxels + 1))))

        # Scratch buffer for double-buffer mode
        self._buffer_b = (
            None
            if single_buffer
            else torch.full((self.n_voxels,), -1, dtype=torch.int32, device=device)
        )

        total_passes = 1 + self.num_passes + 2
        mem_gb = self.n_voxels * (4 if single_buffer else 8) / (1024**3)
        mode = "single-buffer" if single_buffer else "double-buffer"
        log_info(
            f"JumpFloodingEDT ({mode}): {grid_shape}, "
            f"1+{self.num_passes}+2={total_passes} passes, {mem_gb:.2f} GB"
        )

    def propagate(self, site_index: torch.Tensor) -> None:
        """Run 1+JFA+2 propagation on site_index in-place.

        Launch sequence:
          1. Prefix:  step=1 (local seed sharing)
          2. Main JFA: large-to-small steps
          3. Suffix:  step=2, step=1 (error correction)

        Args:
            site_index: Site indices (nx, ny, nz) or (nx*ny*nz,), int32.
                Modified in-place.
        """
        nx, ny, nz = self.grid_shape
        device, stream = get_warp_device_stream(site_index)
        site_flat = site_index.view(-1)
        kernel = self._kernel

        if self.single_buffer:
            self._propagate_single_buffer(site_flat, kernel, nx, ny, nz, device, stream)
        else:
            self._propagate_double_buffer(site_flat, kernel, nx, ny, nz, device, stream)

    def _propagate_single_buffer(self, site_flat, kernel, nx, ny, nz, device, stream):
        """Chaotic relaxation: read and write the same buffer."""
        site_wp = wp.from_torch(site_flat, dtype=wp.int32)

        for step in self._step_sequence():
            wp.launch(
                kernel,
                dim=self.n_voxels,
                inputs=[site_wp, site_wp, step, nx, ny, nz],
                device=device,
                stream=stream,
                block_dim=256,
            )

    def _propagate_double_buffer(self, site_flat, kernel, nx, ny, nz, device, stream):
        """Ping-pong between two buffers."""
        buf_a = wp.from_torch(site_flat, dtype=wp.int32)
        buf_b = wp.from_torch(self._buffer_b, dtype=wp.int32)

        steps = self._step_sequence()
        for i, step in enumerate(steps):
            src, dst = (buf_a, buf_b) if i % 2 == 0 else (buf_b, buf_a)
            wp.launch(
                kernel,
                dim=self.n_voxels,
                inputs=[src, dst, step, nx, ny, nz],
                device=device,
                stream=stream,
                block_dim=256,
            )

        # Copy result back if it ended in buf_b
        if len(steps) % 2 == 1:
            site_flat.copy_(
                torch.as_tensor(buf_b.numpy(), device=site_flat.device)
            )

    def _step_sequence(self):
        """Build the 1+JFA+2 step sequence."""
        steps = [1]  # prefix
        for p in range(self.num_passes):
            steps.append(1 << (self.num_passes - 1 - p))
        steps.extend([2, 1])  # suffix
        return steps
