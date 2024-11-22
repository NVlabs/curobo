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
from copy import deepcopy
from typing import Optional

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.geom.types import Cuboid
from curobo.geom.types import Cuboid as curobo_Cuboid
from curobo.geom.types import Mesh, VoxelGrid
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
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
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(0)


def plot_cost_iteration(cost: torch.Tensor, save_path="cost", title="", log_scale=False):
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
    fig, ax = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    t_steps = np.linspace(0, act_seq.position.shape[0] * dt, act_seq.position.shape[0])

    if sma_filter:
        kernel = 5
        sma = torch.nn.AvgPool1d(kernel_size=kernel, stride=1, padding=2, ceil_mode=False).cuda()

    for i in range(act_seq.position.shape[-1]):
        ax[0].plot(t_steps, act_seq.position[:, i].cpu(), "-", label=str(i))

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
    ax[0].legend(bbox_to_anchor=(0.5, 1.6), loc="upper center", ncol=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_curobo(
    n_cubes: int,
    enable_debug: bool = False,
    tsteps: int = 30,
    trajopt_seeds: int = 4,
    mpinets: bool = False,
    graph_mode: bool = False,
    cuda_graph: bool = True,
    collision_activation_distance: float = 0.025,
    finetune_dt_scale: float = 1.0,
    parallel_finetune: bool = True,
    args=None,
):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = -0.00

    ik_seeds = 32
    if graph_mode:
        trajopt_seeds = 4
    if trajopt_seeds >= 14:
        ik_seeds = max(100, trajopt_seeds * 2)
    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }
    world_cfg = WorldConfig.from_dict(
        {
            "voxel": {
                "base": {
                    "dims": [2.4, 2.4, 2.4],
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "voxel_size": 0.02,
                    "feature_dtype": torch.bfloat16,
                },
            }
        }
    )
    interpolation_steps = 2000
    if graph_mode:
        interpolation_steps = 100
    robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())
    K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
    K.position[0, :] -= 0.2
    K.position[1, :] += 0.2

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_instance,
        world_cfg,
        trajopt_tsteps=tsteps,
        collision_checker_type=CollisionCheckerType.VOXEL,
        use_cuda_graph=cuda_graph,
        position_threshold=0.005,  # 0.5 cm
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
        maximum_trajectory_dt=0.15,
        finetune_trajopt_iters=200,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(enable_graph=True, warmup_js_trajopt=False)
    # create a ground truth collision checker:
    world_model = WorldConfig.from_dict(
        {
            "cuboid": {
                "table": {
                    "dims": [1, 1, 1],
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                }
            }
        }
    )
    if args.mesh:
        world_model = world_model.get_mesh_world()
    config = RobotWorldConfig.load_from_config(
        robot_cfg_instance,
        world_model,
        collision_activation_distance=0.0,
        collision_checker_type=CollisionCheckerType.MESH,
        n_cuboids=50,
        n_meshes=50,
        max_collision_distance=100.0,
    )
    robot_world = RobotWorld(config)

    return mg, robot_cfg, robot_world


def benchmark_mb(
    write_usd=False,
    save_log=False,
    write_plot=False,
    write_benchmark=False,
    plot_cost=False,
    override_tsteps: Optional[int] = None,
    args=None,
):
    # load dataset:
    graph_mode = args.graph
    interpolation_dt = 0.02
    file_paths = [demo_raw, motion_benchmaker_raw, mpinets_raw][1:]

    enable_debug = save_log or plot_cost
    all_files = []
    og_tsteps = 32
    if override_tsteps is not None:
        og_tsteps = override_tsteps

    og_trajopt_seeds = 4
    og_collision_activation_distance = 0.01
    if args.graph:
        og_trajopt_seeds = 4
    for file_path in file_paths:
        all_groups = []
        mpinets_data = False
        problems = file_path()
        if "dresser_task_oriented" in list(problems.keys()):
            mpinets_data = True
        for key, v in tqdm(problems.items()):
            scene_problems = problems[key]
            m_list = []
            i = -1
            ik_fail = 0
            trajopt_seeds = og_trajopt_seeds
            tsteps = og_tsteps
            collision_activation_distance = og_collision_activation_distance
            finetune_dt_scale = 0.9
            parallel_finetune = True
            if "cubby_task_oriented" in key and "merged" not in key:
                trajopt_seeds = 8

            mg, robot_cfg, robot_world = load_curobo(
                0,
                enable_debug,
                tsteps,
                trajopt_seeds,
                mpinets_data,
                graph_mode,
                not args.disable_cuda_graph,
                collision_activation_distance=collision_activation_distance,
                finetune_dt_scale=finetune_dt_scale,
                parallel_finetune=parallel_finetune,
                args=args,
            )
            for problem in tqdm(scene_problems, leave=False):
                i += 1
                if problem["collision_buffer_ik"] < 0.0:
                    continue
                plan_config = MotionGenPlanConfig(
                    max_attempts=20,
                    enable_graph_attempt=1,
                    disable_graph_attempt=10,
                    partial_ik_opt=False,
                    timeout=60,
                )

                q_start = problem["start"]
                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )

                problem_name = key + "_" + str(i)
                if args.mesh:
                    problem_name = "mesh_" + problem_name
                # reset planner
                mg.reset(reset_seed=False)
                world = WorldConfig.from_dict(problem["obstacles"])

                # mg.world_coll_checker.clear_cache()
                world_coll = WorldConfig.from_dict(problem["obstacles"]).get_collision_check_world()
                if args.mesh:
                    world_coll = world_coll.get_mesh_world(merge_meshes=False)
                robot_world.clear_world_cache()
                robot_world.update_world(world_coll)

                esdf = robot_world.world_model.get_esdf_in_bounding_box(
                    Cuboid(name="base", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2.4, 2.4, 2.4]),
                    voxel_size=0.02,
                    dtype=torch.float32,
                )
                # esdf.feature_tensor[esdf.feature_tensor < -1.0] = -1000.0
                world_voxel_collision = mg.world_coll_checker
                world_voxel_collision.update_voxel_data(esdf)

                torch.cuda.synchronize()

                start_state = JointState.from_position(mg.tensor_args.to_device([q_start]))
                if i == 0:
                    for _ in range(3):
                        result = mg.plan_single(
                            start_state,
                            Pose.from_list(pose),
                            plan_config,
                        )
                result = mg.plan_single(
                    start_state,
                    Pose.from_list(pose),
                    plan_config,
                )
                # if not result.success.item():
                #    world = write_yaml(problem["obstacles"], "dresser_task.yml")
                #    exit()

                # print(result.total_time, result.ik_time, result.trajopt_time, result.finetune_time)
                if result.status == "IK Fail":
                    ik_fail += 1
                problem["solution"] = None
                if save_log or write_usd:
                    world.randomize_color(r=[0.5, 0.9], g=[0.2, 0.5], b=[0.0, 0.2])

                    coll_mesh = mg.world_coll_checker.get_mesh_in_bounding_box(
                        curobo_Cuboid(
                            name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2.4, 2.4, 2.4]
                        ),
                        voxel_size=0.02,
                    )

                    coll_mesh.color = [0.0, 0.8, 0.8, 0.2]

                    coll_mesh.name = "voxel_world"

                    # world = WorldConfig(mesh=[coll_mesh])
                    world.add_obstacle(coll_mesh)
                # get costs:
                if plot_cost:
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
                            save_path=join_path("benchmark/log/plot/", problem_name + "_cost")[1:],
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
                                )[1:],
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
                        "dt": interpolation_dt,
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

                    # check if path is collision free w.r.t. ground truth mesh:
                    # robot_world.world_model.clear_cache()

                    q_int_traj = result.get_interpolated_plan().position.unsqueeze(0)
                    d_int_mask = (
                        torch.count_nonzero(~robot_world.validate_trajectory(q_int_traj)) == 0
                    ).item()

                    q_traj = result.optimized_plan.position.unsqueeze(0)
                    d_mask = (
                        torch.count_nonzero(~robot_world.validate_trajectory(q_traj)) == 0
                    ).item()
                    # d_world, _ = robot_world.get_world_self_collision_distance_from_joint_trajectory(
                    # q_traj)
                    # thres_dist = robot_world.contact_distance
                    # in_collision = d_world.squeeze(0) > thres_dist
                    # d_mask = not torch.any(in_collision, dim=-1).item()
                    # if not d_mask:
                    #    write_usd = True
                    #    #print(torch.max(d_world).item(), problem_name)
                    current_metrics = CuroboMetrics(
                        skip=False,
                        success=True,
                        perception_success=d_mask,
                        perception_interpolated_success=d_int_mask,
                        time=result.total_time,
                        collision=False,
                        joint_limit_violation=False,
                        self_collision=False,
                        position_error=result.position_error.item() * 100.0,
                        orientation_error=result.rotation_error.item() * 100.0,
                        eef_position_path_length=10,
                        eef_orientation_path_length=10,
                        attempts=result.attempts,
                        motion_time=result.motion_time.item(),
                        solve_time=result.solve_time,
                        jerk=torch.max(torch.abs(result.optimized_plan.jerk)).item(),
                    )

                    # run planner
                    if write_usd:  # and not d_int_mask:
                        # CuRobo
                        from curobo.util.usd_helper import UsdHelper

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
                            # flatten_usd=True,
                        )
                        # write_usd = False
                        # exit()
                    if write_plot:
                        problem_name = problem_name
                        plot_traj(
                            result.optimized_plan,
                            result.optimized_dt.item(),
                            # result.get_interpolated_plan(),
                            # result.interpolation_dt,
                            title=problem_name,
                            save_path=join_path("benchmark/log/plot/", problem_name + ".pdf")[1:],
                        )
                        plot_traj(
                            # result.optimized_plan,
                            # result.optimized_dt.item(),
                            result.get_interpolated_plan(),
                            result.interpolation_dt,
                            title=problem_name,
                            save_path=join_path("benchmark/log/plot/", problem_name + "_int.pdf")[
                                1:
                            ],
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
                    # print("invalid: " + problem_name)
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
                if save_log:  # and not result.success.item():
                    # CuRobo
                    from curobo.util.usd_helper import UsdHelper

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
                        # flatten_usd=True,
                    )
                    exit()
            g_m = CuroboGroupMetrics.from_list(m_list)
            print(
                key,
                f"{g_m.success:2.2f}",
                g_m.time.mean,
                # g_m.time.percent_75,
                g_m.time.percent_98,
                g_m.position_error.percent_98,
                # g_m.position_error.median,
                g_m.orientation_error.percent_98,
                g_m.cspace_path_length.percent_98,
                g_m.motion_time.percent_98,
                g_m.perception_interpolated_success,
                # g_m.orientation_error.median,
            )
            print(g_m.attempts)
        g_m = CuroboGroupMetrics.from_list(all_groups)
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
                write_yaml(problems, "mb_curobo_solution_voxel.yaml")
            else:
                write_yaml(problems, "mpinets_curobo_solution_voxel.yaml")
        all_files += all_groups
    g_m = CuroboGroupMetrics.from_list(all_files)
    print("######## FULL SET ############")
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

        print("All: ", f"{g_m.success:2.2f}")
        print(
            "Perception Success (coarse, interpolated):",
            g_m.perception_success,
            g_m.perception_interpolated_success,
        )
        print("MT: ", g_m.motion_time)
        print("PT:", g_m.time)
        print("ST: ", g_m.solve_time)
        print("accuracy: ", g_m.position_error, g_m.orientation_error)
        print("Jerk: ", g_m.jerk)


if __name__ == "__main__":
    setup_curobo_logger("error")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesh",
        action="store_true",
        help="When True, runs only geometric planner",
        default=False,
    )

    parser.add_argument(
        "--graph",
        action="store_true",
        help="When True, runs only geometric planner",
        default=False,
    )

    parser.add_argument(
        "--disable_cuda_graph",
        action="store_true",
        help="When True, disable cuda graph during benchmarking",
        default=False,
    )
    args = parser.parse_args()
    benchmark_mb(
        save_log=False,
        write_usd=False,
        write_plot=False,
        write_benchmark=False,
        plot_cost=False,
        args=args,
    )
