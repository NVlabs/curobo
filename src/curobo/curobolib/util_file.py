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

# Standard Library
import os


def get_cpp_path():
    path = os.path.dirname(__file__)
    return os.path.join(path, "cpp")


def join_path(path1, path2):
    if isinstance(path2, str):
        return os.path.join(path1, path2)
    else:
        return path2


def add_cpp_path(sources):
    cpp_path = get_cpp_path()
    new_list = []
    for s in sources:
        s = join_path(cpp_path, s)
        new_list.append(s)
    return new_list
