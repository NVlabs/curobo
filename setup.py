# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""curobo package setuptools.

This file handles optional CUDA extension compilation via pybind11.
All package metadata is now in pyproject.toml following PEP 621.
"""

# Standard Library
import os
import sys
from pathlib import Path

# Third Party
import setuptools

source_dir = Path(__file__).parent.resolve()

# Check if user wants to use pybind compiled extensions. This is deprecated.
# Default: False (use cuda.core runtime compilation)
# Enable with: CUROBO_USE_PYBIND=1 pip install -e . --no-build-isolation
USE_PYBIND = os.environ.get("CUROBO_USE_PYBIND", "0") == "1"
ext_modules = []
cmdclass = {}

if USE_PYBIND:
    # Only import torch if compiling
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        extra_cuda_args = {
            "nvcc": [
                "--threads=8",
                "-O3",
                "--ftz=true",
                "--fmad=true",
                "--prec-div=false",
                "--prec-sqrt=false",
                "--generate-line-info",
            ]
        }

        if sys.platform == "win32":
            extra_cuda_args["nvcc"].append("--allow-unsupported-compiler")

        # Create a list of modules to be compiled
        ext_modules = [
            CUDAExtension(
                "curobo._src.curobolib.backends.pybind.kinematics",
                [
                    "curobo/_src/curobolib/backends/pybind/kinematics_bindings.cpp",
                    "curobo/_src/curobolib/backends/pybind/kinematics_forward_kernel_launch.cu",
                    "curobo/_src/curobolib/backends/pybind/kinematics_backward_kernel_launch.cu",
                    "curobo/_src/curobolib/backends/pybind/kinematics_backward_jacobian_kernel_launch.cu",
                ],
                include_dirs=[str(source_dir / "curobo/_src/curobolib/kernels")],
                extra_compile_args=extra_cuda_args,
            ),
            CUDAExtension(
                "curobo._src.curobolib.backends.pybind.optimization",
                [
                    "curobo/_src/curobolib/backends/pybind/optimization_bindings.cpp",
                    "curobo/_src/curobolib/backends/pybind/line_search_kernel_launch.cu",
                    "curobo/_src/curobolib/backends/pybind/lbfgs_step_kernel_launch.cu",
                ],
                include_dirs=[str(source_dir / "curobo/_src/curobolib/kernels")],
                extra_compile_args=extra_cuda_args,
            ),
            CUDAExtension(
                "curobo._src.curobolib.backends.pybind.trajectory",
                [
                    "curobo/_src/curobolib/backends/pybind/trajectory_bindings.cpp",
                    "curobo/_src/curobolib/backends/pybind/trajectory_kernel_launch.cu",
                ],
                include_dirs=[str(source_dir / "curobo/_src/curobolib/kernels")],
                extra_compile_args=extra_cuda_args,
            ),
            CUDAExtension(
                "curobo._src.curobolib.backends.pybind.geometry",
                [
                    "curobo/_src/curobolib/backends/pybind/geometry_bindings.cpp",
                    "curobo/_src/curobolib/backends/pybind/sphere_obb_kernel_launch.cu",
                    "curobo/_src/curobolib/backends/pybind/self_collision_kernel_launch.cu",
                    "curobo/_src/curobolib/backends/pybind/sphere_voxel_kernel_launch.cu",
                ],
                include_dirs=[str(source_dir / "curobo/_src/curobolib/kernels")],
                extra_compile_args=extra_cuda_args,
            ),
        ]

        cmdclass = {"build_ext": BuildExtension}

    except ImportError as e:
        print(f"Warning: Could not import torch for compilation: {e}")
        print("Installing without compiled extensions - will use cuda.core backend")


# All package metadata is now in pyproject.toml (PEP 621)
# This setup.py only handles optional CUDA extension compilation
setuptools.setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
