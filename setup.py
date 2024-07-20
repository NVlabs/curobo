#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""curobo package setuptools."""

# NOTE: This file is still needed to allow the package to be
# installed in editable mode.
#
# References:
# * https://setuptools.pypa.io/en/latest/setuptools.html#setup-cfg-only-projects

# Standard Library
import sys

# Third Party
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_cuda_args = {
    "nvcc": [
        "--threads=8",
        "-O3",
        "--ftz=true",
        "--fmad=true",
        "--prec-div=false",
        "--prec-sqrt=false",
    ]
}

if sys.platform == "win32":
    extra_cuda_args["nvcc"].append("--allow-unsupported-compiler")

# create a list of modules to be compiled:
ext_modules = [
    CUDAExtension(
        "curobo.curobolib.lbfgs_step_cu",
        [
            "src/curobo/curobolib/cpp/lbfgs_step_cuda.cpp",
            "src/curobo/curobolib/cpp/lbfgs_step_kernel.cu",
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        "curobo.curobolib.kinematics_fused_cu",
        [
            "src/curobo/curobolib/cpp/kinematics_fused_cuda.cpp",
            "src/curobo/curobolib/cpp/kinematics_fused_kernel.cu",
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        "curobo.curobolib.line_search_cu",
        [
            "src/curobo/curobolib/cpp/line_search_cuda.cpp",
            "src/curobo/curobolib/cpp/line_search_kernel.cu",
            "src/curobo/curobolib/cpp/update_best_kernel.cu",
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        "curobo.curobolib.tensor_step_cu",
        [
            "src/curobo/curobolib/cpp/tensor_step_cuda.cpp",
            "src/curobo/curobolib/cpp/tensor_step_kernel.cu",
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        "curobo.curobolib.geom_cu",
        [
            "src/curobo/curobolib/cpp/geom_cuda.cpp",
            "src/curobo/curobolib/cpp/sphere_obb_kernel.cu",
            "src/curobo/curobolib/cpp/pose_distance_kernel.cu",
            "src/curobo/curobolib/cpp/self_collision_kernel.cu",
        ],
        extra_compile_args=extra_cuda_args,
    ),
]

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    package_data={"": ["*.so"]},
    include_package_data=True,
)
