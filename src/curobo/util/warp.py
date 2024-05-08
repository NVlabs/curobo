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

# CuRobo
from curobo.types.base import TensorDeviceType


def init_warp(quiet=True, tensor_args: TensorDeviceType = TensorDeviceType()):
    wp.config.quiet = quiet
    # wp.config.print_launches = True
    # wp.config.verbose = True
    # wp.config.mode = "debug"
    # wp.config.enable_backward = True
    wp.init()

    # wp.force_load(wp.device_from_torch(tensor_args.device))
    return True
