# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Base configuration for CUDA kernel compilation using cuda.core.
Provides common functionality shared across all kernel modules.
"""

# Standard Library
from pathlib import Path
from typing import List

# CuRobo
from curobo._src.runtime import debug_cuda_compile as cuda_debug_compile


class CudaCoreKernelCfg:
    """Base class for cuda.core kernel compilation configuration"""

    def __init__(self, kernel_subdir: str):
        """Initialize kernel configuration.

        Args:
            kernel_subdir: Subdirectory name under kernels/ (e.g., "kinematics", "optimization")
        """
        self._kernel_dir = Path(__file__).parent.parent.parent / "kernels" / kernel_subdir

    @property
    def kernel_dir(self) -> Path:
        """Get the kernel directory"""
        return self._kernel_dir

    def get_compile_flags(self, debug: bool = False) -> List[str]:
        """Get compilation flags for kernel compilation.

        Args:
            debug: If True, use debug compilation flags

        Returns:
            List of compilation flags
        """
        if debug or cuda_debug_compile:
            # Debug compilation flags
            return [
                "-G",  # Generate debug information
                "-g",  # Host debug information
                "--generate-line-info",
                "--device-debug",
            ]
        else:
            # Release compilation flags (matching setup.py)
            return [
                "-O3",
                "--ftz=true",
                "--fmad=true",
                "--prec-div=false",
                "--prec-sqrt=false",
                "--generate-line-info",
            ]

    def get_base_include_dirs(self) -> List[Path]:
        """Get common include directories for all kernels.

        Returns:
            List of common include directory paths
        """
        return [
            self._kernel_dir.parent,  # kernels/
            self._kernel_dir.parent / "common",
            self._kernel_dir.parent / "third_party",
        ]
