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
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import CSpaceConfig, JointState


def test_cspace_config():
    cspace_config = CSpaceConfig(
        ["j0", "j1", "j2", "j3"],
        retract_config=[i for i in range(4)],
        cspace_distance_weight=[i for i in range(4)],
        null_space_weight=[i for i in range(4)],
    )

    new_order = ["j3", "j1"]
    cspace_config.inplace_reindex(new_order)
    assert cspace_config.retract_config[0] == 3 and cspace_config.retract_config[1] == 1
    assert cspace_config.null_space_weight[0] == 3 and cspace_config.null_space_weight[1] == 1
    assert (
        cspace_config.cspace_distance_weight[0] == 3
        and cspace_config.cspace_distance_weight[1] == 1
    )


def test_joint_state():
    tensor_args = TensorDeviceType()
    j_names = ["j0", "j1", "j2", "j3"]
    loc_j = ["j4", "jb"]
    final_j = ["jb", "j0", "j1", "j2", "j3", "j4"]

    position = tensor_args.to_device([i for i in range(len(j_names))])

    loc_position = tensor_args.to_device([i + len(j_names) for i in range(len(loc_j))])

    js_1 = JointState.from_position(position, joint_names=j_names)
    js_lock = JointState.from_position(loc_position, loc_j)

    final_js = js_1.get_augmented_joint_state(final_j, js_lock)
    assert final_js.joint_names == final_j
    assert (
        torch.linalg.norm(final_js.position - tensor_args.to_device([5, 0, 1, 2, 3, 4])).item()
        < 1e-8
    )


def test_batch_joint_state():
    tensor_args = TensorDeviceType()
    j_names = ["j0", "j1", "j2", "j3"]
    loc_j = ["j4", "jb"]
    final_j = ["jb", "j0", "j1", "j2", "j3", "j4"]

    # $position = tensor_args.to_device([i for i in range(len(j_names))])
    position = torch.zeros((10, len(j_names)), device=tensor_args.device, dtype=tensor_args.dtype)
    for i in range(len(j_names)):
        position[:, i] = i

    loc_position = tensor_args.to_device([i + len(j_names) for i in range(len(loc_j))])

    js_1 = JointState.from_position(position, joint_names=j_names)
    js_lock = JointState.from_position(loc_position, loc_j)

    final_js = js_1.get_augmented_joint_state(final_j, js_lock)
    assert final_js.joint_names == final_j
    assert (
        torch.linalg.norm(
            final_js.position - tensor_args.to_device([5, 0, 1, 2, 3, 4]).unsqueeze(0)
        ).item()
        < 1e-8
    )
