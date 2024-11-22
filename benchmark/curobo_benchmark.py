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
import argparse
import random
from copy import deepcopy
from typing import Optional

# Third Party
import numpy as np
import torch
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.geom.types import Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.metrics import CuroboGroupMetrics, CuroboMetrics
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# set seeds
torch.manual_seed(2)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(2)
random.seed(2)


def plot_cost_iteration(cost: torch.Tensor, save_path="cost", title="", log_scale=False):
    # Third Party
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    cost = cost.cpu().numpy()
    # save to csv:
    np.savetxt(save_path + ".csv", cost, delimiter=",")

    # if cost.shape[0] > 1:
    colormap = plt.cm.winter
    plt.gca().set_prop_cycle(plt.cycler("color", colormap(np.linspace(0, 1, cost.shape[0]))))
    x = [i for i in range(cost.shape[-1])]
    for i in range(cost.shape[0]):
        plt.plot(x, cost[i], label="seed_" + str(i))
    plt.tight_layout()
    # plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    if log_scale:
        plt.yscale("log")
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + ".pdf")
    plt.close()


def plot_traj(act_seq: JointState, dt=0.25, title="", save_path="plot.png", sma_filter=False):
    # Third Party
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    t_steps = np.linspace(0, act_seq.position.shape[0] * dt, act_seq.position.shape[0])
    # compute acceleration from finite difference of velocity:
    # act_seq.acceleration = (torch.roll(act_seq.velocity, -1, 0) - act_seq.velocity) / dt
    # act_seq.acceleration = ( act_seq.velocity - torch.roll(act_seq.velocity, 1, 0)) / dt
    # act_seq.acceleration[0,:] = 0.0
    # act_seq.jerk = ( act_seq.acceleration - torch.roll(act_seq.acceleration, 1, 0)) / dt
    # act_seq.jerk[0,:] = 0.0
    if sma_filter:
        kernel = 5
        sma = torch.nn.AvgPool1d(kernel_size=kernel, stride=1, padding=2, ceil_mode=False).cuda()
    # act_seq.jerk = sma(act_seq.jerk)
    # act_seq.acceleration[-1,:] = 0.0
    for i in range(act_seq.position.shape[-1]):
        ax[0].plot(t_steps, act_seq.position[:, i].cpu(), "-", label=str(i))
        # act_seq.velocity[1:-1, i] = sma(act_seq.velocity[:,i].view(1,-1)).squeeze()#@[1:-2]

        ax[1].plot(t_steps[: act_seq.velocity.shape[0]], act_seq.velocity[:, i].cpu(), "-")
        if sma_filter:
            act_seq.acceleration[:, i] = sma(
                act_seq.acceleration[:, i].view(1, -1)
            ).squeeze()  # @[1:-2]

        ax[2].plot(t_steps[: act_seq.acceleration.shape[0]], act_seq.acceleration[:, i].cpu(), "-")
        if sma_filter:
            act_seq.jerk[:, i] = sma(act_seq.jerk[:, i].view(1, -1)).squeeze()  # @[1:-2]\

        ax[3].plot(t_steps[: act_seq.jerk.shape[0]], act_seq.jerk[:, i].cpu(), "-")
    ax[0].set_title(title + " dt=" + "{:.3f}".format(dt))
    ax[3].set_xlabel("Time(s)")
    ax[3].set_ylabel("Jerk rad. s$^{-3}$")
    ax[0].set_ylabel("Position rad.")
    ax[1].set_ylabel("Velocity rad. s$^{-1}$")
    ax[2].set_ylabel("Acceleration rad. s$^{-2}$")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    # ax[0].legend(loc="upper right")
    ax[0].legend(bbox_to_anchor=(0.5, 1.6), loc="upper center", ncol=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # plt.legend()


def load_curobo(
    n_cubes: int,
    enable_debug: bool = False,
    tsteps: int = 30,
    trajopt_seeds: int = 4,
    mpinets: bool = False,
    graph_mode: bool = True,
    mesh_mode: bool = False,
    cuda_graph: bool = True,
    collision_buffer: float = -0.01,
    finetune_dt_scale: float = 0.9,
    collision_activation_distance: float = 0.02,
    args=None,
    parallel_finetune=False,
    ik_seeds=None,
):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_buffer
    robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_mesh.yml"
    robot_cfg["kinematics"]["collision_link_names"].remove("attached_object")
    robot_cfg["kinematics"]["ee_link"] = "panda_hand"

    if ik_seeds is None:
        ik_seeds = 32

    if graph_mode:
        trajopt_seeds = 4
        collision_activation_distance = 0.0
    if trajopt_seeds >= 16:
        ik_seeds = 100
    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_obb_world()
    interpolation_steps = 1000
    c_checker = CollisionCheckerType.PRIMITIVE
    c_cache = {"obb": n_cubes}
    if mesh_mode:
        c_checker = CollisionCheckerType.MESH
        c_cache = {"mesh": n_cubes}
        world_cfg = world_cfg.get_mesh_world()
    if graph_mode:
        interpolation_steps = 100

    robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())

    K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
    K.position[0, :] -= 0.2
    K.position[1, :] += 0.2
    finetune_iters = 200
    grad_iters = None
    if args.report_edition:
        finetune_iters = 200
        grad_iters = 100
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_instance,
        world_cfg,
        finetune_trajopt_iters=finetune_iters,
        grad_trajopt_iters=grad_iters,
        trajopt_tsteps=tsteps,
        collision_checker_type=c_checker,
        use_cuda_graph=cuda_graph,
        collision_cache=c_cache,
        position_threshold=0.005,  # 5 mm
        rotation_threshold=0.05,
        num_ik_seeds=ik_seeds,
        num_graph_seeds=trajopt_seeds,
        num_trajopt_seeds=trajopt_seeds,
        interpolation_dt=0.025,
        store_ik_debug=enable_debug,
        store_trajopt_debug=enable_debug,
        interpolation_steps=interpolation_steps,
        collision_activation_distance=collision_activation_distance,
        trajopt_dt=0.25,
        finetune_dt_scale=finetune_dt_scale,
        high_precision=args.high_precision,
        use_cuda_graph_trajopt_metrics=cuda_graph,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=parallel_finetune)

    return mg, robot_cfg


def benchmark_mb(
    write_usd=False,
    save_log=False,
    write_plot=False,
    write_benchmark=False,
    plot_cost=False,
    override_tsteps: Optional[int] = None,
    graph_mode=False,
    args=None,
):
    # load dataset:
    force_graph = False

    file_paths = [motion_benchmaker_raw, mpinets_raw][:]
    if args.demo:
        file_paths = [demo_raw]

    enable_debug = save_log or plot_cost
    all_files = []
    og_tsteps = 32
    if override_tsteps is not None:
        og_tsteps = override_tsteps
    og_finetune_dt_scale = 0.8
    og_trajopt_seeds = 4
    og_parallel_finetune = True
    og_collision_activation_distance = 0.01
    og_ik_seeds = None
    for file_path in file_paths:
        all_groups = []
        mpinets_data = False
        problems = file_path()
        if "dresser_task_oriented" in list(problems.keys()):
            mpinets_data = True
        for key, v in tqdm(problems.items()):

            finetune_dt_scale = og_finetune_dt_scale
            force_graph = False
            tsteps = og_tsteps
            trajopt_seeds = og_trajopt_seeds
            collision_activation_distance = og_collision_activation_distance
            parallel_finetune = og_parallel_finetune
            ik_seeds = og_ik_seeds

            scene_problems = problems[key]
            n_cubes = check_problems(scene_problems)

            if "cubby_task_oriented" in key and "merged" not in key:
                trajopt_seeds = 8

            mg, robot_cfg = load_curobo(
                n_cubes,
                enable_debug,
                tsteps,
                trajopt_seeds,
                mpinets_data,
                graph_mode,
                args.mesh,
                not args.disable_cuda_graph,
                collision_buffer=args.collision_buffer,
                finetune_dt_scale=finetune_dt_scale,
                collision_activation_distance=collision_activation_distance,
                args=args,
                parallel_finetune=parallel_finetune,
                ik_seeds=ik_seeds,
            )
            m_list = []
            i = 0
            ik_fail = 0
            for problem in tqdm(scene_problems, leave=False):
                i += 1
                if problem["collision_buffer_ik"] < 0.0:
                    continue

                plan_config = MotionGenPlanConfig(
                    max_attempts=20,  # 20,
                    enable_graph_attempt=1,
                    disable_graph_attempt=10,
                    enable_finetune_trajopt=not args.no_finetune,
                    enable_graph=graph_mode or force_graph,
                    timeout=60,
                    enable_opt=not graph_mode,
                    need_graph_success=force_graph,
                    parallel_finetune=parallel_finetune,
                    finetune_dt_scale=finetune_dt_scale,
                )
                q_start = problem["start"]
                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )
                problem_name = "d_" + key + "_" + str(i)

                # reset planner
                mg.reset(reset_seed=False)
                if args.mesh:
                    world = WorldConfig.from_dict(deepcopy(problem["obstacles"])).get_mesh_world(
                        merge_meshes=False
                    )
                else:
                    world = WorldConfig.from_dict(deepcopy(problem["obstacles"])).get_obb_world()
                mg.world_coll_checker.clear_cache()
                mg.update_world(world)

                # run planner
                start_state = JointState.from_position(mg.tensor_args.to_device([q_start]))
                goal_pose = Pose.from_list(pose)
                if i == 1:
                    for _ in range(3):
                        result = mg.plan_single(
                            start_state,
                            goal_pose,
                            plan_config.clone(),
                        )
                result = mg.plan_single(
                    start_state,
                    goal_pose,
                    plan_config,
                )
                if result.status == "IK Fail":
                    ik_fail += 1
                problem["solution"] = None
                problem_name = key + "_" + str(i)

                if write_usd or save_log:
                    # CuRobo
                    from curobo.util.usd_helper import UsdHelper

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
                # get costs:
                if plot_cost and not result.success.item():
                    dt = 0.5
                    problem_name = "approx_wolfe_p" + problem_name
                    if result.optimized_dt is not None:
                        dt = result.optimized_dt.item()
                    if "trajopt_result" in result.debug_info:
                        success = result.success.item()
                        traj_cost = result.debug_info["trajopt_result"].debug_info["solver"][
                            "cost"
                        ][-1]
                        traj_cost = torch.cat(traj_cost, dim=-1)
                        plot_cost_iteration(
                            traj_cost,
                            title=problem_name + "_" + str(success) + "_" + str(dt),
                            save_path=join_path("benchmark/log/plot/", problem_name + "_cost"),
                            log_scale=False,
                        )
                        if "finetune_trajopt_result" in result.debug_info:
                            traj_cost = result.debug_info["finetune_trajopt_result"].debug_info[
                                "solver"
                            ]["cost"][0]
                            traj_cost = torch.cat(traj_cost, dim=-1)
                            plot_cost_iteration(
                                traj_cost,
                                title=problem_name + "_" + str(success) + "_" + str(dt),
                                save_path=join_path(
                                    "benchmark/log/plot/", problem_name + "_ft_cost"
                                ),
                                log_scale=False,
                            )
                if result.success.item():
                    q_traj = result.get_interpolated_plan()
                    problem["goal_ik"] = q_traj.position.cpu().squeeze().numpy()[-1, :].tolist()
                    problem["solution"] = {
                        "position": result.get_interpolated_plan()
                        .position.cpu()
                        .squeeze()
                        .numpy()
                        .tolist(),
                        "velocity": result.get_interpolated_plan()
                        .velocity.cpu()
                        .squeeze()
                        .numpy()
                        .tolist(),
                        "acceleration": result.get_interpolated_plan()
                        .acceleration.cpu()
                        .squeeze()
                        .numpy()
                        .tolist(),
                        "jerk": result.get_interpolated_plan()
                        .jerk.cpu()
                        .squeeze()
                        .numpy()
                        .tolist(),
                        "dt": result.interpolation_dt,
                    }

                    debug = {
                        "used_graph": result.used_graph,
                        "attempts": result.attempts,
                        "ik_time": result.ik_time,
                        "graph_time": result.graph_time,
                        "trajopt_time": result.trajopt_time,
                        "total_time": result.total_time,
                        "solve_time": result.solve_time,
                        "opt_traj": {
                            "position": result.optimized_plan.position.cpu()
                            .squeeze()
                            .numpy()
                            .tolist(),
                            "velocity": result.optimized_plan.velocity.cpu()
                            .squeeze()
                            .numpy()
                            .tolist(),
                            "acceleration": result.optimized_plan.acceleration.cpu()
                            .squeeze()
                            .numpy()
                            .tolist(),
                            "jerk": result.optimized_plan.jerk.cpu().squeeze().numpy().tolist(),
                            "dt": result.optimized_dt.item(),
                        },
                        "valid_query": result.valid_query,
                    }
                    problem["solution_debug"] = debug

                    reached_pose = mg.compute_kinematics(result.optimized_plan[-1]).ee_pose
                    rot_error = goal_pose.angular_distance(reached_pose) * 100.0
                    if args.graph:
                        solve_time = result.graph_time
                    else:
                        solve_time = result.solve_time
                    # compute path length:
                    path_length = torch.sum(
                        torch.linalg.norm(
                            (
                                torch.roll(result.optimized_plan.position, -1, dims=-2)
                                - result.optimized_plan.position
                            )[..., :-1, :],
                            dim=-1,
                        )
                    ).item()
                    current_metrics = CuroboMetrics(
                        skip=False,
                        success=True,
                        time=result.total_time,
                        collision=False,
                        joint_limit_violation=False,
                        self_collision=False,
                        position_error=result.position_error.item() * 1000.0,
                        orientation_error=rot_error.item(),
                        eef_position_path_length=10,
                        eef_orientation_path_length=10,
                        attempts=result.attempts,
                        motion_time=result.motion_time.item(),
                        solve_time=solve_time,
                        cspace_path_length=path_length,
                        jerk=torch.max(torch.abs(result.optimized_plan.jerk)).item(),
                    )

                    if write_usd:
                        # CuRobo

                        q_traj = result.get_interpolated_plan()
                        UsdHelper.write_trajectory_animation_with_robot_usd(
                            robot_cfg,
                            world,
                            start_state,
                            q_traj,
                            dt=result.interpolation_dt,
                            save_path=join_path("benchmark/log/usd/", problem_name)[1:] + ".usd",
                            interpolation_steps=1,
                            write_robot_usd_path="benchmark/log/usd/assets/",
                            robot_usd_local_reference="assets/",
                            base_frame="/world_" + problem_name,
                            visualize_robot_spheres=True,
                            flatten_usd=True,
                        )

                    if write_plot:  # and result.optimized_dt.item() > 0.06:
                        problem_name = problem_name
                        plot_traj(
                            result.optimized_plan,
                            result.optimized_dt.item(),
                            title=problem_name,
                            save_path=join_path("benchmark/log/plot/", problem_name + ".png")[1:],
                        )

                    m_list.append(current_metrics)
                    all_groups.append(current_metrics)
                elif result.valid_query:
                    current_metrics = CuroboMetrics()
                    debug = {
                        "used_graph": result.used_graph,
                        "attempts": result.attempts,
                        "ik_time": result.ik_time,
                        "graph_time": result.graph_time,
                        "trajopt_time": result.trajopt_time,
                        "total_time": result.total_time,
                        "solve_time": result.solve_time,
                        "status": result.status,
                        "valid_query": result.valid_query,
                    }
                    problem["solution_debug"] = debug

                    m_list.append(current_metrics)
                    all_groups.append(current_metrics)
                else:
                    debug = {
                        "used_graph": result.used_graph,
                        "attempts": result.attempts,
                        "ik_time": result.ik_time,
                        "graph_time": result.graph_time,
                        "trajopt_time": result.trajopt_time,
                        "total_time": result.total_time,
                        "solve_time": result.solve_time,
                        "status": result.status,
                        "valid_query": result.valid_query,
                    }

                    problem["solution_debug"] = debug
                if save_log and not result.success.item():
                    UsdHelper.write_motion_gen_log(
                        result,
                        robot_cfg,
                        world,
                        start_state,
                        Pose.from_list(pose),
                        join_path("benchmark/log/usd/", problem_name) + "_debug",
                        write_ik=True,
                        write_trajopt=True,
                        visualize_robot_spheres=True,
                        grid_space=2,
                        write_robot_usd_path="benchmark/log/usd/assets/",
                        flatten_usd=True,
                    )
                    print(result.status)

            g_m = CuroboGroupMetrics.from_list(m_list)
            if not args.kpi:
                print(
                    key,
                    f"{g_m.success:2.2f}",
                    g_m.time.mean,
                    g_m.time.percent_98,
                    g_m.position_error.mean,
                    g_m.orientation_error.mean,
                    g_m.cspace_path_length.percent_98,
                    g_m.motion_time.percent_98,
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
                    ["Motion Time(s)", g_m.motion_time],
                    ["Path Length (rad.)", g_m.cspace_path_length],
                    ["Jerk", g_m.jerk],
                    ["Position Error (mm)", g_m.position_error],
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
                write_yaml(problems, args.file_name + "_mb_solution.yaml")
            else:
                write_yaml(problems, args.file_name + "_mpinets_solution.yaml")
        all_files += all_groups
    g_m = CuroboGroupMetrics.from_list(all_files)

    try:
        # Third Party
        from tabulate import tabulate

        headers = ["Metric", "Value"]

        table = [
            ["Success %", f"{g_m.success:2.2f}"],
            ["Plan Time (s)", g_m.time],
            ["Motion Time(s)", g_m.motion_time],
            ["Path Length (rad.)", g_m.cspace_path_length],
            ["Jerk", g_m.jerk],
            ["Position Error (mm)", g_m.position_error],
        ]
        print(tabulate(table, headers, tablefmt="grid"))
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


def check_problems(all_problems):
    n_cube = 0
    for problem in all_problems:
        cache = (
            WorldConfig.from_dict(deepcopy(problem["obstacles"])).get_obb_world().get_cache_dict()
        )
        n_cube = max(n_cube, cache["obb"])
    return n_cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="mg_curobo_",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--collision_buffer",
        type=float,
        default=0.00,  # in meters
        help="Robot collision buffer",
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
        "--report_edition",
        action="store_true",
        help="When True, runs benchmark with parameters from technical report",
        default=False,
    )
    parser.add_argument(
        "--jetson",
        action="store_true",
        help="When True, runs benchmark with parameters for jetson",
        default=False,
    )
    parser.add_argument(
        "--no_finetune",
        action="store_true",
        help="When True, runs benchmark with parameters for jetson",
        default=False,
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="When True, runs benchmark with parameters for jetson",
        default=False,
    )

    args = parser.parse_args()

    setup_curobo_logger("error")
    for i in range(1):
        print("*****RUN: " + str(i))
        benchmark_mb(
            save_log=False,
            write_usd=args.save_usd,
            write_plot=args.save_plot,
            write_benchmark=args.write_benchmark,
            plot_cost=False,
            graph_mode=args.graph,
            args=args,
        )
