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


def _ext_modules():
    """Lazily construct extension modules only when actually building.

    Returns empty list during metadata-only builds (e.g., uv lock)
    to avoid requiring torch at metadata generation time.
    """
    try:
        from torch.utils.cpp_extension import CUDAExtension  # lazy import
    except Exception:
        # During metadata-only builds (uv lock / pip prepare_metadata), Torch isn't available.
        # Return no extensions so metadata can be generated without compiling.
        return []

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

    return [
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


def _cmdclass():
    """Lazily construct cmdclass only when torch is available.

    Returns empty dict during metadata-only builds to avoid requiring torch.
    """
    try:
        from torch.utils.cpp_extension import BuildExtension  # lazy import
    except Exception:
        # No torch during metadata build: don't register a cmdclass
        return {}
    return {"build_ext": BuildExtension}

setuptools.setup(
    ext_modules=_ext_modules(),
    cmdclass=_cmdclass(),
    package_data={"": ["*.so"]},
    include_package_data=True,
)
