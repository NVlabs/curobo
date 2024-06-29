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


# Third Party
import warp as wp
from packaging import version

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.util.logger import log_info


def init_warp(quiet=True, tensor_args: TensorDeviceType = TensorDeviceType()):
    wp.config.quiet = quiet
    # wp.config.print_launches = True
    # wp.config.verbose = True
    # wp.config.mode = "debug"
    # wp.config.enable_backward = True
    wp.init()

    # wp.force_load(wp.device_from_torch(tensor_args.device))
    return True


def warp_support_sdf_struct(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.0.0"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.0.0, using older sdf kernels."
            + "No issues expected."
        )
        return False
    return True


def warp_support_kernel_key(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.2.1"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.2.1, using, creating global constant to trigger kernel generation."
        )
        return False
    return True
