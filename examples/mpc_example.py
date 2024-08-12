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
import time

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig


def plot_traj(trajectory, dof):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(3, 1)
    q = trajectory[:, :dof]
    qd = trajectory[:, dof : dof * 2]
    qdd = trajectory[:, dof * 2 : dof * 3]

    for i in range(q.shape[-1]):
        axs[0].plot(q[:, i], label=str(i))
        axs[1].plot(qd[:, i], label=str(i))
        axs[2].plot(qdd[:, i], label=str(i))
    plt.legend()
    plt.savefig("test.png")
    # plt.show()


def demo_full_config_mpc():
    PLOT = True
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_file,
        store_rollouts=True,
        step_dt=0.03,
    )
    mpc = MpcSolver(mpc_config)

    # retract_cfg = robot_cfg.cspace.retract_config.view(1, -1)
    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0)
    joint_names = mpc.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg + 0.5, joint_names=joint_names)
    )
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    start_state = JointState.from_position(retract_cfg, joint_names=joint_names)

    goal = Goal(
        current_state=start_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )
    goal_buffer = mpc.setup_solve_single(goal, 1)

    # test_q = tensor_args.to_device( [2.7735, -1.6737,  0.4998, -2.9865,  0.3386,  0.8413,  0.4371])
    # start_state.position[:] = test_q
    converged = False
    tstep = 0
    traj_list = []
    mpc_time = []
    mpc.update_goal(goal_buffer)
    current_state = start_state  # .clone()
    while not converged:
        st_time = time.time()
        # current_state.position += 0.1
        # print(current_state.position)
        result = mpc.step(current_state, 1)

        # print(mpc.get_visual_rollouts().shape)
        # exit()
        torch.cuda.synchronize()
        if tstep > 5:
            mpc_time.append(time.time() - st_time)
        # goal_buffer.current_state.position[:] = result.action.position
        # result.action.position += 0.1
        current_state.copy_(result.action)
        # goal_buffer.current_state.velocity[:] = result.action.vel
        traj_list.append(result.action.get_state_tensor())
        tstep += 1
        # if tstep % 10 == 0:
        #    print(result.metrics.pose_error.item(), result.solve_time, mpc_time[-1])
        if result.metrics.pose_error.item() < 0.01:
            converged = True
        if tstep > 1000:
            break
    print(
        "MPC (converged, error, steps, opt_time, mpc_time): ",
        converged,
        result.metrics.pose_error.item(),
        tstep,
        result.solve_time,
        np.mean(mpc_time),
    )
    if PLOT:
        plot_traj(torch.cat(traj_list, dim=0).cpu().numpy(), dof=retract_cfg.shape[-1])


if __name__ == "__main__":
    demo_full_config_mpc()
    # demo_full_config_mesh_mpc()
