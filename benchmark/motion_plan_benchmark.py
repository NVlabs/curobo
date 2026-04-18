# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
import argparse
import random
from copy import deepcopy
from typing import Optional, Tuple

# Third Party
import numpy as np
import pinocchio as pin
import seaborn as sns
import torch
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

# CuRobo
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.motion import MotionPlanner, MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.types.robot import RobotCfg
from curobo._src.util.benchmark_metrics import CuroboGroupMetrics, CuroboMetrics
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util.usd_writer import UsdWriter
from curobo._src.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_scene_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)

torch._dynamo.config.compiled_autograd = True
torch._dynamo.config.cache_size_limit = 20

# set seeds

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
np.random.seed(2)
random.seed(2)
torch.manual_seed(2)


def load_robot_model_for_dynamics(
    robot_name: str = "franka",
    attached_object_mass: float = 0.0,
) -> Tuple:
    """Load and configure Pinocchio robot model for dynamics computation.

    Args:
        robot_name: Name of the robot configuration
        attached_object_mass: Mass to set for the attached_object link (kg)

    Returns:
        Tuple of (model, data, torque_limits)
    """
    from curobo._src.types.robot import RobotCfg

    # Load robot configuration
    config_file = load_yaml(str(join_path(get_robot_configs_path(), robot_name + ".yml")))
    if "robot_cfg" in config_file:
        config_file = config_file["robot_cfg"]
    robot_cfg = RobotCfg.create(config_file)
    kinematics_config = robot_cfg.kinematics.kinematics_config

    # Update attached_object mass in kinematics config BEFORE exporting URDF
    if attached_object_mass >= 0.0 and "attached_object" in kinematics_config.link_name_to_idx_map:
        link_idx = kinematics_config.link_name_to_idx_map["attached_object"]
        # Update mass in link_masses_com tensor (last column is mass)
        kinematics_config.link_masses_com[link_idx, 3] = attached_object_mass

    # Export URDF to temporary file
    urdf_path = f"temp_robot_dynamics_{int(attached_object_mass * 1000)}mg.urdf"
    kinematics_config.export_to_urdf(
        robot_name="dynamics",
        output_path=urdf_path,
        include_spheres=False,
    )

    # Load with Pinocchio
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=None)
    data = model.createData()

    # Default Franka Panda torque limits (N·m) for 7 DOF
    torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

    return model, data, torque_limits


def compute_trajectory_energy(
    trajectory: JointState,
    robot_model_data: Tuple,
) -> dict:
    """Compute torques and energy for a trajectory using Pinocchio inverse dynamics.

    Args:
        trajectory: JointState with position, velocity, acceleration, and dt
        robot_model_data: Tuple of (model, data, torque_limits) from load_robot_model_for_dynamics()

    Returns:
        Dictionary with:
            - energy: float - total energy consumed (J) = sum of |power| * dt
            - max_torque: float - maximum torque magnitude (N·m)
            - torque_violation: bool - True if any joint exceeds its limit
    """
    model, data, torque_limits = robot_model_data

    # Extract trajectory data
    positions = trajectory.position.cpu().squeeze().numpy()
    velocities = trajectory.velocity.cpu().squeeze().numpy()
    accelerations = trajectory.acceleration.cpu().squeeze().numpy()
    dt = trajectory.dt.item()

    horizon = positions.shape[0]
    num_dof = model.nq

    # Compute torques for each timestep using inverse dynamics
    torques = np.zeros((horizon, num_dof))
    for t in range(horizon):
        q = positions[t, :num_dof]
        qd = velocities[t, :num_dof] if t < len(velocities) else velocities[-1, :num_dof]
        qdd = accelerations[t, :num_dof] if t < len(accelerations) else accelerations[-1, :num_dof]

        # Compute inverse dynamics: tau = M(q)qdd + C(q,qd) + g(q)
        tau = pin.rnea(model, data, q, qd, qdd)
        torques[t] = tau[:num_dof]

    # Compute power: P(t) = tau(t) * qd(t) for each joint
    power = torques[: len(velocities), :num_dof] * velocities[:, :num_dof]

    # Compute energy: E = sum over time of |P(t)| * dt
    energy = np.sum(np.abs(power)) * dt

    # Check torque violations
    max_torque = np.max(np.abs(torques), axis=0)

    torque_violation = np.any(max_torque > torque_limits)

    return {
        "energy": float(energy),
        "max_torque": float(np.max(max_torque)),
        "torque_violation": bool(torque_violation),
        "torques": torques,
        "power": power,
    }


def plot_traj(
    act_seq: JointState,
    dt=0.25,
    title="",
    style="-",
    save_path="plot.png",
    sma_filter=False,
    new_plot=True,
    save_plot=True,
    ax=None,
    tau=None,
    energy=None,
):
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    with sns.color_palette(n_colors=7):
        # Third Party
        import matplotlib.pyplot as plt

        if new_plot:
            if tau is not None:
                fig, ax = plt.subplots(6, 1, figsize=(5, 10), sharex=True)
            else:
                fig, ax = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
        else:
            plt.gca().set_prop_cycle(None)
        act_seq = act_seq.squeeze(0)
        t_steps = np.linspace(0, act_seq.position.shape[0] * dt, act_seq.position.shape[0])
        if sma_filter:
            kernel = 5
            sma = torch.nn.AvgPool1d(
                kernel_size=kernel, stride=1, padding=2, ceil_mode=False
            ).cuda()
        for i in range(len(joint_names)):
            ax[0].plot(t_steps, act_seq.position[:, i].cpu(), label=joint_names[i], linestyle=style)

            ax[1].plot(
                t_steps[: act_seq.velocity.shape[0]], act_seq.velocity[:, i].cpu(), linestyle=style
            )
            if sma_filter:
                act_seq.acceleration[:, i] = sma(
                    act_seq.acceleration[:, i].view(1, -1)
                ).squeeze()  # @[1:-2]

            ax[2].plot(
                t_steps[: act_seq.acceleration.shape[0]],
                act_seq.acceleration[:, i].cpu(),
                linestyle=style,
            )
            if sma_filter:
                act_seq.jerk[:, i] = sma(act_seq.jerk[:, i].view(1, -1)).squeeze()  # @[1:-2]\

            ax[3].plot(t_steps[: act_seq.jerk.shape[0]], act_seq.jerk[:, i].cpu(), linestyle=style)
            if tau is not None:
                ax[4].plot(t_steps[: tau.shape[0]], tau[:, i].cpu(), linestyle=style)
        if energy is not None:
            energy = torch.sum(energy, dim=-1)
        if tau is not None:
            ax[5].plot(t_steps[: energy.shape[0]], energy[:].cpu(), linestyle=style, color="black")
            ax[5].set_ylabel("Energy")
            ax[5].grid()
        if tau is not None:
            ax[4].set_ylabel("Torque")
            ax[4].grid()
        ax[0].set_title(title + " dt = " + "{:.3f}".format(dt))
        ax[3].set_xlabel("Time(s)")
        ax[3].set_ylabel("Jerk rad. s$^{-3}$")

        ax[0].set_ylabel("Position rad.")
        ax[1].set_ylabel("Velocity rad. s$^{-1}$")
        ax[2].set_ylabel("Acceleration rad. s$^{-2}$")
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()
        ax[0].legend(bbox_to_anchor=(0.5, 1.6), loc="upper center", ncol=3)
        if save_plot:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            return ax


def load_curobo(
    n_cubes: int,
    ik_seeds=None,
    trajopt_seeds: int = 4,
    mpinets: bool = False,
    collision_buffer: float = 0.0,
    args=None,
):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    if "robot_cfg" in robot_cfg:
        robot_cfg = robot_cfg["robot_cfg"]

    robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_buffer
    if "attached_object" in robot_cfg["kinematics"]["collision_link_names"]:
        robot_cfg["kinematics"]["collision_link_names"].remove("attached_object")
    robot_cfg["kinematics"]["tool_frames"] = ["panda_hand"]

    collision_activation_distance = 0.0025

    if args.graph:
        trajopt_seeds = 4
        collision_activation_distance = 0.0
        interpolation_steps = 100

    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }
    if args.use_dynamics:
        robot_cfg["load_dynamics"] = True
    else:
        robot_cfg["load_dynamics"] = False
    scene_cfg = SceneCfg.create(
        load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))
    ).get_obb_world()

    c_cache = {"obb": n_cubes}

    if args.mesh:
        c_cache = {"mesh": n_cubes}
        scene_cfg = scene_cfg.get_mesh_world()
    import copy

    robot_cfg_instance = RobotCfg.create(
        copy.deepcopy(robot_cfg), device_cfg=DeviceCfg()
    )

    K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
    K.position[0, :] -= 0.2
    K.position[1, :] += 0.2


    motion_planner_config = MotionPlannerCfg.create(
        robot=robot_cfg_instance,
        scene_model=scene_cfg,
        ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"][:],
        ik_transition_model="ik/transition_ik.yml",
        metrics_rollout="metrics_base.yml",
        trajopt_optimizer_configs=["trajopt/particle_trajopt.yml",
        "trajopt/lbfgs_bspline_trajopt.yml"][:],
        trajopt_transition_model="trajopt/transition_bspline_trajopt.yml",
        use_cuda_graph=not args.disable_cuda_graph,
        num_ik_seeds=ik_seeds,
        num_trajopt_seeds=trajopt_seeds,
        collision_cache=c_cache,
        store_debug=False,
        optimizer_collision_activation_distance=collision_activation_distance,
    )
    motion_planner = MotionPlanner(motion_planner_config)
    if args.use_dynamics:
        motion_planner.update_links_inertial(
            {
                "attached_object": {"mass": args.mass},
            }
        )

    return motion_planner, robot_cfg


def benchmark_mb(
    write_usd=False,
    write_plot=False,
    write_benchmark=False,
    override_tsteps: Optional[int] = None,
    graph_mode=False,
    args=None,
    prefix: str = "",
    mg_init=None,
):
    np.random.seed(2)
    random.seed(2)
    torch.manual_seed(2)
    # load dataset:
    force_graph = False

    file_paths = [motion_benchmaker_raw, mpinets_raw][:]
    # args.demo = True
    if args.demo:
        file_paths = [demo_raw]

    enable_profile = args.enable_profile
    all_files = []
    og_trajopt_seeds = 4
    og_ik_seeds = 32

    for file_path in file_paths:
        all_groups = []
        mpinets_data = False
        problems = file_path()
        if "dresser_task_oriented" in list(problems.keys()):
            mpinets_data = True

        for key, v in tqdm(problems.items()):
            force_graph = False
            trajopt_seeds = og_trajopt_seeds
            ik_seeds = og_ik_seeds
            scene_problems = problems[key]  # [:2]#[1:4]

            mg_init = None
            robot_model_data = None
            if mg_init is None:
                n_cubes = check_problems(scene_problems)

                mg, robot_cfg = load_curobo(
                    n_cubes,
                    ik_seeds,
                    trajopt_seeds,
                    mpinets_data,
                    collision_buffer=0.0,
                    args=args,
                )
                mg.warmup(enable_graph=True)

                # Load robot model for dynamics (energy computation)
                robot_model_data = load_robot_model_for_dynamics(
                    robot_name="franka",
                    attached_object_mass=args.mass,
                )
            else:
                print("Using cached planner")
                mg = mg_init

            m_list = []
            i = 0
            ik_fail = 0
            start_idx = 0
            for problem in tqdm(scene_problems[:], leave=False):
                i += 1
                if problem["collision_buffer_ik"] < 0.0:
                    continue

                q_start = problem["start"]
                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )
                problem_name = key + "_" + str(i)

                if args.mesh:
                    world = SceneCfg.create(deepcopy(problem["obstacles"])).get_mesh_world(
                        merge_meshes=False
                    )
                else:
                    world = SceneCfg.create(deepcopy(problem["obstacles"])).get_obb_world()
                mg.scene_collision_checker.clear_cache()
                mg.update_world(world)
                mg.reset_seed()
                # run planner
                start_state = JointState.from_position(mg.device_cfg.to_device([q_start]))
                goal_pose = Pose.from_list(pose)
                goal_tool_poses = GoalToolPose.from_poses(
                    {mg.tool_frames[0]: goal_pose},
                    ordered_tool_frames=mg.tool_frames,
                )
                debug_info = None
                if i == 1:
                    for _ in range(3):
                        mg.reset_seed()

                        result = mg.plan_pose(
                            goal_tool_poses,
                            start_state,
                            max_attempts=1,
                            enable_graph_attempt=1,
                        )

                if enable_profile:
                    emit_nvtx = args.emit_nvtx
                    if emit_nvtx:
                        with torch.autograd.profiler.emit_nvtx():
                            torch.cuda.profiler.start()
                            result = mg.plan_pose(
                                goal_tool_poses,
                                start_state,
                                max_attempts=1,
                                enable_graph_attempt=10,
                            )
                            torch.cuda.profiler.stop()
                        print("nvtx done")
                        exit()

                    mg.reset_seed()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        result = mg.plan_pose(
                            goal_tool_poses,
                            start_state,
                            max_attempts=1,
                            enable_graph_attempt=10,
                        )
                    print(
                        prof.key_averages(group_by_stack_n=10).table(
                            sort_by="self_cuda_time_total", row_limit=10
                        )
                    )
                    print(
                        prof.key_averages(group_by_input_shape=True).table(
                            sort_by="cpu_time_total", row_limit=10
                        )
                    )
                    print("Exporting the trace..")
                    prof.export_chrome_trace(
                        join_path("benchmark/log/trace", "motion_plan_no_dyn") + ".json"
                    )
                    print(result.success)
                    exit()
                mg.reset_seed()
                result = mg.plan_pose(
                    goal_tool_poses,
                    start_state,
                    max_attempts=100,
                    enable_graph_attempt=1,
                )
                debug_info = None

                q_usd_traj = None

                problem["solution"] = None

                # get costs:

                if result is not None and result.success.item():
                    # compute torque and energy:
                    q_traj = result.js_solution  # result.get_interpolated_plan()
                    q_traj_interpolated = result.get_interpolated_plan()
                    problem["solution_interpolated"] = {
                        "position": q_traj_interpolated.position.cpu().squeeze().numpy().tolist(),
                        "velocity": q_traj_interpolated.velocity.cpu().squeeze().numpy().tolist(),
                        "acceleration": q_traj_interpolated.acceleration.cpu()
                        .squeeze()
                        .numpy()
                        .tolist(),
                        "jerk": q_traj_interpolated.jerk.cpu().squeeze().numpy().tolist(),
                        "dt": q_traj_interpolated.dt.item(),
                    }
                    problem["goal_ik"] = q_traj.position.cpu().squeeze().numpy()[-1, :].tolist()
                    problem["solution"] = {
                        "position": q_traj.position.cpu().squeeze().numpy().tolist(),
                        "velocity": q_traj.velocity.cpu().squeeze().numpy().tolist(),
                        "acceleration": q_traj.acceleration.cpu().squeeze().numpy().tolist(),
                        "jerk": q_traj.jerk.cpu().squeeze().numpy().tolist(),
                        "dt": q_traj.dt.item(),
                    }
                    solve_time = result.total_time
                    # compute path length:
                    path_length = torch.sum(
                        torch.linalg.norm(
                            (torch.roll(q_traj.position, -1, dims=-2) - q_traj.position)[
                                ..., :-1, :
                            ],
                            dim=-1,
                        )
                    ).item()
                    offset = 1
                    if q_traj.control_space in ControlSpace.bspline_types():
                        offset = 2 * mg.trajopt_solver.interpolation_steps

                    motion_time = q_traj.dt.item() * (q_traj.position.shape[-2] - offset)
                    if False:
                        if q_traj.control_space in ControlSpace.bspline_types():
                            q_traj = q_traj.trim_trajectory(
                                mg.trajopt_solver.interpolation_steps,
                                -(mg.trajopt_solver.interpolation_steps),
                            )


                    # Compute energy using Pinocchio inverse dynamics
                    energy_val = 0.0
                    max_torque_val = 0.0
                    if robot_model_data is not None:
                        try:
                            dynamics_result = compute_trajectory_energy(q_traj, robot_model_data)
                            energy_val = dynamics_result["energy"]
                            max_torque_val = dynamics_result["max_torque"]
                        except Exception as e:
                            print(f"Warning: Failed to compute energy: {e}")
                    current_metrics = CuroboMetrics(
                        skip=False,
                        success=True,
                        time=result.total_time,
                        collision=False,
                        joint_limit_violation=False,
                        self_collision=False,
                        position_error=result.position_error.item() * 1000.0,
                        orientation_error=result.rotation_error.item() * 180.0 / np.pi,
                        eef_position_path_length=10,
                        eef_orientation_path_length=10,
                        attempts=1,
                        motion_time=motion_time,
                        solve_time=result.solve_time,
                        cspace_path_length=path_length,
                        jerk=torch.max(torch.abs(q_traj.jerk)).item(),
                        energy=energy_val,
                        torque=max_torque_val,
                    )

                    if write_usd:  # or motion_time > 2.5:
                        # CuRobo
                        if True:
                            world.randomize_color(r=[0.5, 0.9], g=[0.2, 0.5], b=[0.0, 0.2])
                            gripper_mesh = Mesh(
                                name="robot_target_gripper",
                                file_path=join_path(
                                    get_assets_path(),
                                    "robot/franka_description/meshes/visual/hand.dae",
                                ),
                                color=[0.0, 0.8, 0.1, 1.0],
                                pose=pose,
                            )
                            world.add_obstacle(gripper_mesh)
                        # CuRobo
                        if q_usd_traj is None:
                            q_usd_traj = q_traj.squeeze(0).squeeze(0)
                        UsdWriter.write_trajectory_animation_with_robot_usd(
                            robot_cfg,
                            world,
                            start_state,
                            q_usd_traj,
                            dt=q_usd_traj.dt.item(),
                            save_path=join_path("benchmark/log/usd/", problem_name)[1:] + ".usd",
                            interpolation_steps=1,
                            write_robot_usd_path="benchmark/log/usd/assets/",
                            base_frame="/world_" + problem_name,
                            visualize_robot_spheres=True,
                            flatten_usd=True,
                        )

                    if write_plot:
                        problem_name = problem_name + "_" + prefix
                        interpolated_traj = q_traj
                        if True:
                            interpolated_traj = q_traj.squeeze(0)
                            interpolated_traj.joint_names = mg.trajopt_solver.joint_names
                            dt = q_traj.dt.item()
                            ax = plot_traj(
                                interpolated_traj,
                                dt,
                                style="-",
                                title=problem_name,
                                save_path=join_path("benchmark/log/plot", problem_name + ".png")[
                                    :
                                ],
                                save_plot=True,
                                new_plot=True,
                            )

                    m_list.append(current_metrics)
                    all_groups.append(current_metrics)
                elif True:
                    current_metrics = CuroboMetrics()
                    debug = {
                        "used_graph": False,
                        "attempts": 1,
                        "ik_time": 0.0,
                        "graph_time": 0.0,
                        "trajopt_time": 0.0,
                        "total_time": 0.0,
                        "solve_time": 0.0,
                        "status": "Failure",
                        "valid_query": True,
                    }
                    problem["solution_debug"] = debug

                    m_list.append(current_metrics)
                    all_groups.append(current_metrics)
            mg.destroy()

            g_m = CuroboGroupMetrics.from_list(m_list)
            if not args.kpi:
                print(
                    key,
                    f"{g_m.success:2.2f}",
                    f"{g_m.time.mean:2.2f}",
                    f"{g_m.time.percent_98:2.2f}",
                    f"{g_m.position_error.percent_98:2.4f}",
                    f"{g_m.orientation_error.percent_98:2.4f}",
                    f"{g_m.cspace_path_length.percent_98:2.2f}",
                    f"{g_m.motion_time.percent_98:2.2f}",
                )
                print(g_m.attempts)
        g_m = CuroboGroupMetrics.from_list(all_groups)

        if not args.kpi:
            try:
                # Third Party
                from tabulate import tabulate

                headers = ["Metric", "Value"]

                table = [
                    ["Success %", f"{g_m.success:2.2f}"],
                    ["Plan Time (s)", g_m.time],
                    ["Solve Time (s)", g_m.solve_time],
                    ["Position Error (mm)", g_m.position_error],
                    ["Path Length (rad.)", g_m.cspace_path_length],
                    ["Motion Time(s)", g_m.motion_time],
                    ["Jerk", g_m.jerk],
                    ["Energy (J)", g_m.energy],
                    ["Torque (N·m)", g_m.torque],
                ]
                print(tabulate(table, headers, tablefmt="grid"))
            except ImportError:
                print(
                    "All: ",
                    f"{g_m.success:2.2f}",
                    g_m.motion_time.percent_98,
                    g_m.time.mean,
                    g_m.time.percent_75,
                    g_m.position_error.percent_75,
                    g_m.orientation_error.percent_75,
                )
        if write_benchmark:
            if not mpinets_data:
                write_yaml(problems, "benchmark/log/" + args.file_name + "_mb_solution.yaml")
            else:
                write_yaml(problems, "benchmark/log/" + args.file_name + "_mpinets_solution.yaml")
        all_files += all_groups
    g_m = CuroboGroupMetrics.from_list(all_files)
    mg = None

    # Clean up temporary URDF file
    import os

    urdf_path = "temp_robot_dynamics_0mg.urdf"
    if os.path.exists(urdf_path):
        os.remove(urdf_path)

    if len(file_paths) == 1:
        return g_m
    try:
        # Third Party
        from tabulate import tabulate

        headers = ["Metric", "Value"]
        table = [
            ["Success %", f"{g_m.success:2.2f}"],
            ["Plan Time (s)", g_m.time],
            ["Solve Time (s)", g_m.solve_time],
            ["Position Error (mm)", g_m.position_error],
            ["Path Length (rad.)", g_m.cspace_path_length],
            ["Motion Time(s)", g_m.motion_time],
            ["Jerk", g_m.jerk],
            ["Energy (J)", g_m.energy],
            ["Torque (N·m)", g_m.torque],
        ]

        print(tabulate(table, headers, tablefmt="grid"))
        if write_benchmark:
            data = {
                "Success": float(g_m.success),
                "Planning Time": {
                    "mean": float(g_m.time.mean),
                    "std": float(g_m.time.std),
                    "median": float(g_m.time.median),
                    "75th": float(g_m.time.percent_75),
                    "98th": float(g_m.time.percent_98),
                },
                "Position Error (mm)": {
                    "mean": float(g_m.position_error.mean),
                    "std": float(g_m.position_error.std),
                    "median": float(g_m.position_error.median),
                    "75th": float(g_m.position_error.percent_75),
                    "98th": float(g_m.position_error.percent_98),
                },
                "Orientation Error (deg)": {
                    "mean": float(g_m.orientation_error.mean),
                    "std": float(g_m.orientation_error.std),
                    "median": float(g_m.orientation_error.median),
                    "75th": float(g_m.orientation_error.percent_75),
                    "98th": float(g_m.orientation_error.percent_98),
                },
                "Path Length (rad.)": {
                    "mean": float(g_m.cspace_path_length.mean),
                    "std": float(g_m.cspace_path_length.std),
                    "median": float(g_m.cspace_path_length.median),
                    "75th": float(g_m.cspace_path_length.percent_75),
                    "98th": float(g_m.cspace_path_length.percent_98),
                },
                "Motion Time(s)": {
                    "mean": float(g_m.motion_time.mean),
                    "std": float(g_m.motion_time.std),
                    "median": float(g_m.motion_time.median),
                    "75th": float(g_m.motion_time.percent_75),
                    "98th": float(g_m.motion_time.percent_98),
                },
                "Jerk": {
                    "mean": float(g_m.jerk.mean),
                    "std": float(g_m.jerk.std),
                    "median": float(g_m.jerk.median),
                    "75th": float(g_m.jerk.percent_75),
                    "98th": float(g_m.jerk.percent_98),
                },
                "Energy (J)": {
                    "mean": float(g_m.energy.mean),
                    "std": float(g_m.energy.std),
                    "median": float(g_m.energy.median),
                    "75th": float(g_m.energy.percent_75),
                    "98th": float(g_m.energy.percent_98),
                },
                "Torque (N·m)": {
                    "mean": float(g_m.torque.mean),
                    "std": float(g_m.torque.std),
                    "median": float(g_m.torque.median),
                    "75th": float(g_m.torque.percent_75),
                    "98th": float(g_m.torque.percent_98),
                },
                "Solve Time (s)": {
                    "mean": float(g_m.solve_time.mean),
                    "std": float(g_m.solve_time.std),
                    "median": float(g_m.solve_time.median),
                    "75th": float(g_m.solve_time.percent_75),
                    "98th": float(g_m.solve_time.percent_98),
                },
            }
            out_path = join_path("benchmark/log", "table_" + args.file_name + ".yml")
            print(out_path)

            write_yaml(data, out_path)
    except ImportError:
        print("######## FULL SET ############")
        print("All: ", f"{g_m.success:2.2f}")
        print("MT: ", g_m.motion_time)
        print("path-length: ", g_m.cspace_path_length)
        print("PT:", g_m.time)
        print("ST: ", g_m.solve_time)
        print("position error (mm): ", g_m.position_error)
        print("orientation error(%): ", g_m.orientation_error)
        print("jerk: ", g_m.jerk)

    if args.kpi:
        kpi_data = {
            "Success": g_m.success,
            "Planning Time": float(g_m.time.mean),
            "Planning Time Std": float(g_m.time.std),
            "Planning Time Median": float(g_m.time.median),
            "Planning Time 75th": float(g_m.time.percent_75),
            "Planning Time 98th": float(g_m.time.percent_98),
        }
        write_yaml(kpi_data, join_path(args.save_path, args.file_name + ".yml"))
    return


def check_problems(all_problems):
    n_cube = 0
    for problem in all_problems:
        cache = (
            SceneCfg.create(deepcopy(problem["obstacles"])).get_obb_world().get_cache_dict()
        )
        n_cube = max(n_cube, cache["obb"])
    return n_cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable_profile",
        action="store_true",
        help="When True, enables profiling",
        default=False,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="curobov2_0kg",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="When True, runs only geometric planner",
        default=False,
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="When True, converts obstacles to meshes",
        default=False,
    )
    parser.add_argument(
        "--kpi",
        action="store_true",
        help="When True, saves minimal metrics",
        default=False,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="When True, runs only on small dataaset",
        default=False,
    )
    parser.add_argument(
        "--disable_cuda_graph",
        action="store_true",
        help="When True, disable cuda graph during benchmarking",
        default=False,
    )
    parser.add_argument(
        "--use-dynamics",
        action="store_true",
        help="When True, disable cuda graph during benchmarking",
        default=False,
    )
    parser.add_argument(
        "--write_benchmark",
        action="store_true",
        help="When True, writes paths to file",
        default=False,
    )
    parser.add_argument(
        "--save_usd",
        action="store_true",
        help="When True, writes paths to file",
        default=False,
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="When True, writes paths to file",
        default=False,
    )
    parser.add_argument(
        "--emit_nvtx",
        action="store_true",
        help="When True, emits nvtx markers",
        default=False,
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=3.0,
        help="Attached object mass in kg for dynamics (default: 3.0)",
    )
    args = parser.parse_args()
    mg = None
    setup_curobo_logger("error")
    if False:
        args.demo = True

        mg = benchmark_mb(
            write_usd=args.save_usd,
            write_plot=args.save_plot,
            write_benchmark=args.write_benchmark,
            graph_mode=args.graph,
            args=args,
            prefix="warmup",
            mg_init=mg,
        )
        args.demo = False
    previous_gm = None
    for i in range(1):
        print("*****RUN: " + str(i))
        result = benchmark_mb(
            write_usd=args.save_usd,
            write_plot=args.save_plot,
            write_benchmark=args.write_benchmark,
            graph_mode=args.graph,
            args=args,
            prefix=str(i),
            mg_init=mg,
        )
