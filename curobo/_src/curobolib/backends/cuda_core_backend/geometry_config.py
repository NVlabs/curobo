# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for geometry/collision kernel compilation."""

# Standard Library
from pathlib import Path
from typing import List

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class GeometryKernelCfg(CudaCoreKernelCfg):
    """Configuration for geometry/collision kernel compilation"""

    def __init__(self):
        super().__init__("geometry")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("self_collision")

        Returns:
            List of kernel filenames
        """
        kernel_files = {
            "self_collision": ["self_collision/self_collision_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation"""
        # Get base include dirs and add geometry-specific ones
        base_dirs = self.get_base_include_dirs()
        geometry_dirs = [
            self.kernel_dir,  # kernels/geometry/
            self.kernel_dir / "common",
            self.kernel_dir / "self_collision",
        ]
        return base_dirs + geometry_dirs
