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
from curobo.curobolib.geom import get_pose_distance
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState


def test_repeat_seeds():
    tensor_args = TensorDeviceType()
    b = 10
    dof = 7
    position = torch.randn((b, 3), device=tensor_args.device, dtype=tensor_args.dtype)

    quaternion = torch.zeros((b, 4), device=tensor_args.device, dtype=tensor_args.dtype)

    quaternion[:, 0] = 1.0
    goal_pose = Pose(position, quaternion)

    current_state = JointState.from_position(
        torch.randn((b, dof), device=tensor_args.device, dtype=tensor_args.dtype)
    )
    batch_pose_idx = torch.arange(0, b, 1, device=tensor_args.device, dtype=torch.int32).unsqueeze(
        -1
    )
    goal = Goal(goal_pose=goal_pose, batch_pose_idx=batch_pose_idx, current_state=current_state)
    g = goal.repeat_seeds(4)

    start_pose = goal_pose.repeat_seeds(4)
    b = start_pose.position.shape[0]
    out_d = torch.zeros((b, 1), device=tensor_args.device, dtype=tensor_args.dtype)
    out_p_v = torch.zeros((b, 3), device=tensor_args.device, dtype=tensor_args.dtype)
    out_r_v = torch.zeros((b, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    out_idx = torch.zeros((b, 1), device=tensor_args.device, dtype=torch.int32)
    vec_weight = torch.ones((6), device=tensor_args.device, dtype=tensor_args.dtype)
    offset_waypoint = torch.zeros((6), device=tensor_args.device, dtype=tensor_args.dtype)
    offset_tstep_fraction = torch.ones((1), device=tensor_args.device, dtype=tensor_args.dtype)

    weight = tensor_args.to_device([1, 1, 1, 1])
    vec_convergence = tensor_args.to_device([0, 0])
    run_weight = tensor_args.to_device([1])
    project_distance = torch.tensor([True], device=tensor_args.device, dtype=torch.uint8)
    r = get_pose_distance(
        out_d,
        out_d.clone(),
        out_d.clone(),
        out_p_v,
        out_r_v,
        out_idx,
        start_pose.position,
        g.goal_pose.position,
        start_pose.quaternion,
        g.goal_pose.quaternion,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        vec_weight.clone() * 0.0,
        offset_waypoint,
        offset_tstep_fraction,
        g.batch_pose_idx,
        project_distance,
        start_pose.position.shape[0],
        1,
        1,
        1,
        False,
        False,
        True,
    )

    assert torch.sum(r[0]).item() <= 1e-5


def test_horizon_repeat_seeds():
    tensor_args = TensorDeviceType()
    b = 2
    dof = 7
    h = 30
    position = torch.randn((b, h, 3), device=tensor_args.device, dtype=tensor_args.dtype)
    # position[:,:,1] = 1.0
    quaternion = torch.zeros((b, h, 4), device=tensor_args.device, dtype=tensor_args.dtype)

    quaternion[:, 0] = 1.0
    quaternion[1, 1] = 1.0
    quaternion[1, 0] = 0.0
    goal_pose = Pose(position[:, 0, :], quaternion[:, 0, :]).clone()

    current_state = JointState.from_position(
        torch.randn((b, dof), device=tensor_args.device, dtype=tensor_args.dtype)
    )
    batch_pose_idx = torch.arange(0, b, 1, device=tensor_args.device, dtype=torch.int32).unsqueeze(
        -1
    )
    project_distance = torch.tensor([True], device=tensor_args.device, dtype=torch.uint8)

    goal = Goal(goal_pose=goal_pose, batch_pose_idx=batch_pose_idx, current_state=current_state)
    g = goal  # .repeat_seeds(4)

    start_pose = Pose(
        goal_pose.position.view(-1, 1, 3).repeat(1, h, 1),
        quaternion=goal_pose.quaternion.view(-1, 1, 4).repeat(1, h, 1),
    )
    b = start_pose.position.shape[0]
    out_d = torch.zeros((b, h, 1), device=tensor_args.device, dtype=tensor_args.dtype)
    out_p_v = torch.zeros((b, h, 3), device=tensor_args.device, dtype=tensor_args.dtype)
    out_r_v = torch.zeros((b, h, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    out_idx = torch.zeros((b, h, 1), device=tensor_args.device, dtype=torch.int32)
    vec_weight = torch.ones((6), device=tensor_args.device, dtype=tensor_args.dtype)
    offset_waypoint = torch.zeros((6), device=tensor_args.device, dtype=tensor_args.dtype)
    offset_tstep_fraction = torch.ones((1), device=tensor_args.device, dtype=tensor_args.dtype)

    weight = tensor_args.to_device([1, 1, 1, 1])
    vec_convergence = tensor_args.to_device([0, 0])
    run_weight = torch.zeros((1, h), device=tensor_args.device)
    run_weight[-1] = 1
    r = get_pose_distance(
        out_d,
        out_d.clone(),
        out_d.clone(),
        out_p_v,
        out_r_v,
        out_idx,
        start_pose.position,
        g.goal_pose.position,
        start_pose.quaternion,
        g.goal_pose.quaternion,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        vec_weight.clone() * 0.0,
        offset_waypoint,
        offset_tstep_fraction,
        g.batch_pose_idx,
        project_distance,
        start_pose.position.shape[0],
        h,
        1,
        1,
        True,
        False,
        False,
    )
    assert torch.sum(r[0]).item() < 1e-5
