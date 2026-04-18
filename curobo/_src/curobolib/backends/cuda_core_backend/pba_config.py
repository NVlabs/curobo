# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for PBA+ 3D EDT kernel compilation."""

# Standard Library
from pathlib import Path
from typing import List

from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


def _cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


class PBAKernelCfg(CudaCoreKernelCfg):
    """Configuration for PBA+ 3D kernel compilation."""

    def __init__(self):
        super().__init__("parallel_banding")

    def get_kernel_files(self) -> List[str]:
        """Get kernel source files.

        Returns:
            List of kernel filenames.
        """
        return ["pba3d_kernel.cuh"]

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation."""
        base_dirs = self.get_base_include_dirs()
        return base_dirs + [self.kernel_dir]


class PBALaunchCfg:
    """Launch configuration calculator for PBA+ 3D kernels.

    All methods are static; dimensions are in PBA convention
    (sx, sy, sz) which maps to CuRobo (nz, ny, nx).
    """

    @staticmethod
    def flood_z(sx: int, sy: int) -> LaunchConfig:
        """Phase 1: bidirectional flood along Z.

        Grid:  (cdiv(sx, 32), cdiv(sy, 4))
        Block: (32, 4)
        """
        return LaunchConfig(
            grid=(_cdiv(sx, 32), _cdiv(sy, 4)),
            block=(32, 4),
        )

    @staticmethod
    def maurer_axis(sx: int, sz: int) -> LaunchConfig:
        """Phase 2a/3a: Maurer stack along Y.

        Grid:  (cdiv(sx, 32), cdiv(sz, 4))
        Block: (32, 4)
        """
        return LaunchConfig(
            grid=(_cdiv(sx, 32), _cdiv(sz, 4)),
            block=(32, 4),
        )

    @staticmethod
    def color_axis(sx: int, sz: int, m3: int = 2) -> LaunchConfig:
        """Phase 2b/3b: color axis with X<->Y transpose.

        Grid:  (cdiv(sx, 32), sz)
        Block: (32, m3)
        """
        return LaunchConfig(
            grid=(_cdiv(sx, 32), sz),
            block=(32, m3),
        )
