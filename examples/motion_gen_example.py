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

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def plot_traj(trajectory, dt, file_name="test.png"):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(4, 1)
    q = trajectory.position.cpu().numpy()
    qd = trajectory.velocity.cpu().numpy()
    qdd = trajectory.acceleration.cpu().numpy()
    qddd = trajectory.jerk.cpu().numpy()
    timesteps = [i * dt for i in range(q.shape[0])]
    for i in range(q.shape[-1]):
        axs[0].plot(timesteps, q[:, i], label=str(i))
        axs[1].plot(timesteps, qd[:, i], label=str(i))
        axs[2].plot(timesteps, qdd[:, i], label=str(i))
        axs[3].plot(timesteps, qddd[:, i], label=str(i))

    plt.legend()
    plt.savefig(file_name)
    plt.close()
    # plt.show()


def plot_iters_traj(trajectory, d_id=1, dof=7, seed=0):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(len(trajectory), 1)
    if len(trajectory) == 1:
        axs = [axs]
    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):
            axs[k].plot(
                q[i][seed, :-1, d_id].cpu(),
                "r+-",
                label=str(i),
                alpha=0.1 + min(0.9, float(i) / (len(q))),
            )
    plt.legend()
    plt.show()


def plot_iters_traj_3d(trajectory, d_id=1, dof=7, seed=0):
    # Third Party
    import matplotlib.pyplot as plt

    ax = plt.axes(projection="3d")
    c = 0
    h = trajectory[0][0].shape[1] - 1
    x = [x for x in range(h)]

    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):
            # ax.plot3D(x,[c for _ in range(h)],  q[i][seed, :, d_id].cpu())#, 'r')
            ax.scatter3D(
                x, [c for _ in range(h)], q[i][seed, :h, d_id].cpu(), c=q[i][seed, :, d_id].cpu()
            )
            # @plt.show()
            c += 1
    # plt.legend()
    plt.show()


def demo_motion_gen_simple():
    world_config = {
        "mesh": {
            "base_scene": {
                "pose": [10.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
                "file_path": "scene/nvblox/srl_ur10_bins.obj",
            },
        },
        "cuboid": {
            "table": {
                "dims": [5.0, 5.0, 0.2],  # x, y, z
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "ur5e.yml",
        world_config,
        interpolation_dt=0.01,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    goal_pose = Pose.from_list([-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])  # x, y, z, qw, qx, qy, qz
    start_state = JointState.from_position(
        torch.zeros(1, 6).cuda(),
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
    )

    result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))
    traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
    print("Trajectory Generated: ", result.success)


def demo_motion_gen_mesh():
    PLOT = False
    tensor_args = TensorDeviceType()
    world_file = "collision_mesh_scene.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        # trajopt_tsteps=40,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = robot_cfg.cpsace.retract_config
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.5)
    result = motion_gen.plan(
        start_state,
        retract_pose,
        enable_graph=False,
        enable_opt=True,
        max_attempts=1,
        num_trajopt_seeds=10,
        num_graph_seeds=10,
    )
    print(result.status, result.attempts, result.trajopt_time)
    traj = result.raw_plan  # optimized plan
    print("Trajectory Generated: ", result.success)
    if PLOT:
        plot_traj(traj.cpu().numpy())


def demo_motion_gen(js=False):
    # Standard Library
    PLOT = True
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,
        # trajopt_dt=0.15,
        # velocity_scale=0.1,
        use_cuda_graph=True,
        # finetune_dt_scale=2.5,
        interpolation_steps=10000,
    )

    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    # motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js, parallel_finetune=True)
    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    # robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1))
    goal_state = start_state.clone()

    start_state.position[0, 0] += 0.25
    # goal_state.position[0,0] += 0.5
    if js:
        result = motion_gen.plan_single_js(
            start_state,
            goal_state,
            MotionGenPlanConfig(max_attempts=1, time_dilation_factor=0.5),
        )
    else:
        result = motion_gen.plan_single(
            start_state,
            retract_pose,
            MotionGenPlanConfig(
                max_attempts=1,
                timeout=5,
                time_dilation_factor=0.5,
            ),
        )
        new_result = result.clone()
        new_result.retime_trajectory(0.5, create_interpolation_buffer=True)
        print(new_result.optimized_dt, new_result.motion_time, result.motion_time)
    print(
        "Trajectory Generated: ",
        result.success,
        result.solve_time,
        result.status,
        result.optimized_dt,
    )
    if PLOT and result.success.item():
        traj = result.get_interpolated_plan()

        plot_traj(traj, result.interpolation_dt)
        plot_traj(new_result.get_interpolated_plan(), new_result.interpolation_dt, "test_slow.png")
        # plt.save("test.png")
        # plt.close()
        # traj = result.debug_info["opt_solution"].position.view(-1,7)
        # plot_traj(traj.cpu().numpy())


def demo_motion_gen_debug():
    PLOT = True
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=24,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=1,
        num_graph_seeds=1,
        store_ik_debug=True,
        store_trajopt_debug=True,
        trajopt_particle_opt=False,
        grad_trajopt_iters=100,
    )
    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = robot_cfg.cspace.retract_config
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.4)
    result = motion_gen.plan(start_state, retract_pose, enable_graph=True, enable_opt=True)
    if result.status not in [None, "Opt Fail"]:
        return
    traj = result.plan.view(-1, 7)  # optimized plan
    print("Trajectory Generated: ", result.success)
    trajectory_iter_steps = result.debug_info["trajopt_result"].debug_info["solver"]["steps"]

    if PLOT:
        plot_iters_traj_3d(trajectory_iter_steps, d_id=6)
        # plot_traj(traj.cpu().numpy())


def demo_motion_gen_batch():
    PLOT = False
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=1,
        num_ik_seeds=30,
    )
    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.6)

    retract_pose = retract_pose.repeat_seeds(2)
    retract_pose.position[0, 0] = -0.3
    result = motion_gen.plan_batch(
        start_state.repeat_seeds(2),
        retract_pose,
        MotionGenPlanConfig(
            max_attempts=5, enable_graph=False, enable_graph_attempt=1, enable_opt=True
        ),
    )
    traj = result.optimized_plan.position.view(2, -1, 7)  # optimized plan
    print("Trajectory Generated: ", result.success)
    if PLOT:
        plot_traj(traj[0, : result.path_buffer_last_tstep[0], :].cpu().numpy())
        plot_traj(traj[1, : result.path_buffer_last_tstep[1], :].cpu().numpy())


def demo_motion_gen_goalset():
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=1,
        num_ik_seeds=30,
    )
    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.6)

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(2, 1).view(1, -1, 3),
        quaternion=state.ee_quat_seq.repeat(2, 1).view(1, -1, 4),
    )
    goal_pose.position[0, 0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    m_config = MotionGenPlanConfig(False, True, num_trajopt_seeds=10)

    result = motion_gen.plan_goalset(start_state, goal_pose, m_config)


def demo_motion_gen_api():
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    interpolation_dt = 0.02
    # create motion gen with a cuboid cache to be able to load obstacles later:
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args,
        trajopt_tsteps=34,
        interpolation_steps=5000,
        num_ik_seeds=50,
        num_trajopt_seeds=6,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        grad_trajopt_iters=500,
        trajopt_dt=0.5,
        interpolation_dt=interpolation_dt,
        evaluate_interpolated_trajectory=True,
        js_trajopt_dt=0.5,
        js_trajopt_tsteps=34,
    )
    motion_gen = MotionGen(motion_gen_cfg)

    motion_gen.warmup(warmup_js_trajopt=False)

    # create world representation:

    cuboids = [Cuboid(name="obs_1", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]
    world = WorldConfig(cuboid=cuboids)

    motion_gen.update_world(world)

    q_start = JointState.from_position(
        tensor_args.to_device([[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]]),
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
    )

    goal_pose = Pose(
        position=tensor_args.to_device([[0.5, 0.0, 0.3]]),
        quaternion=tensor_args.to_device([[1, 0, 0, 0]]),
    )
    print(goal_pose)

    result = motion_gen.plan_single(q_start, goal_pose)

    # get result:

    interpolated_solution = result.get_interpolated_plan()

    print(result.get_interpolated_plan())


def demo_motion_gen_batch_env(n_envs: int = 10):
    tensor_args = TensorDeviceType()
    world_files = ["collision_table.yml" for _ in range(int(n_envs / 2))] + [
        "collision_test.yml" for _ in range(int(n_envs / 2))
    ]

    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        for world_file in world_files
    ]
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_cfg,
        tensor_args,
        trajopt_tsteps=30,
        use_cuda_graph=False,
        num_trajopt_seeds=4,
        num_ik_seeds=30,
        num_batch_ik_seeds=30,
        evaluate_interpolated_trajectory=True,
        interpolation_dt=0.05,
        interpolation_steps=500,
        grad_trajopt_iters=30,
    )
    motion_gen_batch_env = MotionGen(motion_gen_config)
    motion_gen_batch_env.reset()
    motion_gen_batch_env.warmup(
        enable_graph=False, batch=n_envs, warmup_js_trajopt=False, batch_env_mode=True
    )
    retract_cfg = motion_gen_batch_env.get_retract_config().clone()
    state = motion_gen_batch_env.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    goal_pose = Pose(
        state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze()
    ).repeat_seeds(n_envs)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3).repeat_seeds(n_envs)

    goal_pose.position[1, 0] -= 0.2

    m_config = MotionGenPlanConfig(
        False, True, max_attempts=1, enable_graph_attempt=None, enable_finetune_trajopt=False
    )
    result = motion_gen_batch_env.plan_batch_env(start_state, goal_pose, m_config)

    print(n_envs, result.total_time, result.total_time / n_envs)


if __name__ == "__main__":
    setup_curobo_logger("error")
    demo_motion_gen(js=False)
    # demo_motion_gen_simple()
    # demo_motion_gen_batch()
    # demo_motion_gen_goalset()
    # n = [2, 10]
    # for n_envs in n:
    #    demo_motion_gen_batch_env(n_envs=n_envs)
