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
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import torch
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from metrics import CuroboGroupMetrics, CuroboMetrics
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from robometrics.evaluator import Evaluator
from robometrics.robot import CollisionSphereConfig, Robot
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(0)


def load_curobo(
    n_cubes: int,
    enable_debug: bool = False,
    tsteps: int = 30,
    trajopt_seeds: int = 4,
    mpinets: bool = False,
    graph_mode: bool = True,
):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
    robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_mesh.yml"
    robot_cfg["kinematics"]["collision_link_names"].remove("attached_object")

    ik_seeds = 30  # 500
    if graph_mode:
        trajopt_seeds = 4
    if trajopt_seeds >= 14:
        ik_seeds = max(100, trajopt_seeds * 4)
    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": -0.025,
        }
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_obb_world()
    interpolation_steps = 2000
    if graph_mode:
        interpolation_steps = 100
    robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())

    K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
    K.position[0, :] -= 0.1
    K.position[1, :] += 0.1
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_instance,
        world_cfg,
        trajopt_tsteps=tsteps,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        collision_cache={"obb": n_cubes},
        position_threshold=0.005,  # 0.5 cm
        rotation_threshold=0.05,
        num_ik_seeds=ik_seeds,
        num_graph_seeds=trajopt_seeds,
        num_trajopt_seeds=trajopt_seeds,
        interpolation_dt=0.025,
        interpolation_steps=interpolation_steps,
        collision_activation_distance=0.03,
        state_finite_difference_mode="CENTRAL",
        trajopt_dt=0.25,
        minimize_jerk=True,
        finetune_dt_scale=1.05,  # 1.05,
        maximum_trajectory_dt=0.1,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(enable_graph=True, warmup_js_trajopt=False)
    robot_cfg["kinematics"]["ee_link"] = "panda_hand"
    return mg, robot_cfg


def benchmark_mb(
    write_usd=False,
    save_log=False,
    write_plot=False,
    write_benchmark=False,
    plot_cost=False,
    override_tsteps: Optional[int] = None,
    save_kpi=False,
    graph_mode=False,
    args=None,
):
    interpolation_dt = 0.02

    enable_debug = False
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    urdf = join_path(get_assets_path(), robot_cfg["kinematics"]["urdf_path"])
    spheres = robot_cfg["kinematics"]["collision_spheres"]

    if isinstance(spheres, str):
        spheres = load_yaml(join_path(get_robot_configs_path(), spheres))["collision_spheres"]
    for s in spheres:
        for k in spheres[s]:
            k["radius"] = max(0.001, k["radius"] - 0.015)
    data = {
        "collision_spheres": spheres,
        "self_collision_ignore": robot_cfg["kinematics"]["self_collision_ignore"],
        "self_collision_buffer": robot_cfg["kinematics"]["self_collision_buffer"],
    }
    config = CollisionSphereConfig.load_from_dictionary(data)
    metrics_robot = Robot(urdf, config)
    evaluator = Evaluator(metrics_robot)
    all_files = []

    # modify robot joint limits as some start states in the dataset are at the joint limits:
    if True:
        for k in evaluator.robot.actuated_joints:
            k.limit.lower -= 0.1
            k.limit.upper += 0.1

    plan_config = MotionGenPlanConfig(
        max_attempts=60,
        enable_graph_attempt=3,
        enable_finetune_trajopt=False,
        partial_ik_opt=True,
    )
    file_paths = [motion_benchmaker_raw, mpinets_raw]
    if args.demo:
        file_paths = [demo_raw]
    # load dataset:
    og_tsteps = 32
    if override_tsteps is not None:
        og_tsteps = override_tsteps

    og_trajopt_seeds = 12
    for file_path in file_paths:
        all_groups = []
        mpinets_data = False
        problems = file_path()
        if "dresser_task_oriented" in list(problems.keys()):
            mpinets_data = True
        for key, v in tqdm(problems.items()):
            tsteps = og_tsteps
            trajopt_seeds = og_trajopt_seeds
            if "cage_panda" in key:
                trajopt_seeds = 16
            if "table_under_pick_panda" in key:
                tsteps = 44
                trajopt_seeds = 28

            if "cubby_task_oriented" in key and "merged" not in key:
                trajopt_seeds = 16

            if "dresser_task_oriented" in key:
                trajopt_seeds = 16

            scene_problems = problems[key]  # [:4]  # [:1]  # [:20]  # [0:10]
            n_cubes = check_problems(scene_problems)
            mg, robot_cfg = load_curobo(
                n_cubes, enable_debug, tsteps, trajopt_seeds, mpinets_data, graph_mode
            )
            m_list = []
            i = 0
            ik_fail = 0
            for problem in tqdm(scene_problems, leave=False):
                i += 1
                if problem["collision_buffer_ik"] < 0.0:
                    continue

                plan_config = MotionGenPlanConfig(
                    max_attempts=100,  # 00,  # 00,  # 100,  # 00,  # 000,#,00,#00,  # 5000,
                    enable_graph_attempt=3,
                    enable_finetune_trajopt=True,
                    partial_ik_opt=False,
                    enable_graph=graph_mode,
                    timeout=60,
                    enable_opt=not graph_mode,
                )
                q_start = problem["start"]
                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )

                problem_name = "d_" + key + "_" + str(i)

                # reset planner
                mg.reset(reset_seed=False)
                world = WorldConfig.from_dict(deepcopy(problem["obstacles"])).get_obb_world()
                # world.save_world_as_mesh(problem_name + ".stl")
                mg.world_coll_checker.clear_cache()
                mg.update_world(world)
                # continue
                # load obstacles

                # run planner
                start_state = JointState.from_position(mg.tensor_args.to_device([q_start]))
                result = mg.plan_single(
                    start_state,
                    Pose.from_list(pose),
                    plan_config,
                )
                if result.status == "IK Fail":
                    ik_fail += 1
                if result.success.item():
                    eval_obs = get_metrics_obstacles(problem["obstacles"])

                    q_traj = result.get_interpolated_plan()
                    ee_trajectory = mg.compute_kinematics(q_traj)

                    q_metrics = q_traj.position.cpu().numpy()
                    ee_m = ee_trajectory.ee_pose
                    ee_pos = ee_m.position.cpu().numpy()
                    ee_q = ee_m.quaternion.cpu().numpy()
                    se3_list = [
                        SE3(np.ravel(ee_pos[p]), np.ravel(ee_q[p]))
                        for p in range(ee_m.position.shape[0])
                    ]
                    # add gripper position:
                    q_met = np.zeros((q_metrics.shape[0], 8))
                    q_met[:, :7] = q_metrics
                    q_met[:, 7] = 0.04
                    if mpinets_data:
                        q_met[:, 7] = 0.025
                    st_time = time.time()

                    current_metrics = evaluator.evaluate_trajectory(
                        q_met,
                        se3_list,
                        SE3(np.ravel(pose[:3]), np.ravel(pose[3:])),
                        eval_obs,
                        result.total_time,
                    )
                    # if not current_metrics.success:
                    #    print(current_metrics)
                    #    write_usd = True
                    # else:
                    #    write_usd = False
                # rint(plan_config.enable_graph, plan_config.enable_graph_attempt)
                problem["solution"] = None
                if plan_config.enable_finetune_trajopt:
                    problem_name = key + "_" + str(i)
                else:
                    problem_name = "noft_" + key + "_" + str(i)
                problem_name = "nosw_" + problem_name
                if write_usd or save_log:
                    # CuRobo
                    from curobo.util.usd_helper import UsdHelper

                    world.randomize_color(r=[0.5, 0.9], g=[0.2, 0.5], b=[0.0, 0.2])

                    gripper_mesh = Mesh(
                        name="target_gripper",
                        file_path=join_path(
                            get_assets_path(),
                            "robot/franka_description/meshes/visual/hand_ee_link.dae",
                        ),
                        color=[0.0, 0.8, 0.1, 1.0],
                        pose=pose,
                    )
                    world.add_obstacle(gripper_mesh)
                # get costs:
                if plot_cost:
                    dt = 0.5
                    problem_name = "approx_wolfe_p" + problem_name
                    if result.optimized_dt is not None:
                        dt = result.optimized_dt.item()
                    if "trajopt_result" in result.debug_info:
                        success = result.success.item()
                        traj_cost = (
                            # result.debug_info["trajopt_result"].debug_info["solver"]["cost"][0]
                            result.debug_info["trajopt_result"].debug_info["solver"]["cost"][-1]
                        )
                        # print(traj_cost[0])
                        traj_cost = torch.cat(traj_cost, dim=-1)
                        plot_cost_iteration(
                            traj_cost,
                            title=problem_name + "_" + str(success) + "_" + str(dt),
                            save_path=join_path("log/plot/", problem_name + "_cost"),
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
                                save_path=join_path("log/plot/", problem_name + "_ft_cost"),
                                log_scale=False,
                            )
                if result.success.item() and current_metrics.success:
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
                    # print(problem["solution"]["position"])
                    # exit()
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
                    # print(
                    #    "T: ",
                    #    result.motion_time.item(),
                    #    result.optimized_dt.item(),
                    #    (len(problem["solution"]["position"]) - 1 ) * result.interpolation_dt,
                    #    result.interpolation_dt,
                    #    )
                    # exit()
                    current_metrics = CuroboMetrics(
                        skip=False,
                        success=True,
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
                            save_path=join_path("log/usd/", problem_name) + ".usd",
                            interpolation_steps=1,
                            write_robot_usd_path="log/usd/assets/",
                            robot_usd_local_reference="assets/",
                            base_frame="/world_" + problem_name,
                            visualize_robot_spheres=True,
                        )

                    if write_plot:
                        problem_name = problem_name
                        plot_traj(
                            result.optimized_plan,
                            result.optimized_dt.item(),
                            # result.get_interpolated_plan(),
                            # result.interpolation_dt,
                            title=problem_name,
                            save_path=join_path("log/plot/", problem_name + ".pdf"),
                        )
                        plot_traj(
                            # result.optimized_plan,
                            # result.optimized_dt.item(),
                            result.get_interpolated_plan(),
                            result.interpolation_dt,
                            title=problem_name,
                            save_path=join_path("log/plot/", problem_name + "_int.pdf"),
                        )
                        # exit()

                    m_list.append(current_metrics)
                    all_groups.append(current_metrics)
                elif result.valid_query:
                    # print("fail")
                    # print(result.status)
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
                    # print("save log")
                    UsdHelper.write_motion_gen_log(
                        result,
                        robot_cfg,
                        world,
                        start_state,
                        Pose.from_list(pose),
                        join_path("log/usd/", problem_name) + "_debug",
                        write_ik=False,
                        write_trajopt=True,
                        visualize_robot_spheres=False,
                        grid_space=2,
                    )
                # exit()

            g_m = CuroboGroupMetrics.from_list(m_list)
            print(
                key,
                f"{g_m.success:2.2f}",
                # g_m.motion_time,
                g_m.time.mean,
                # g_m.time.percent_75,
                g_m.time.percent_98,
                g_m.position_error.percent_98,
                # g_m.position_error.median,
                g_m.orientation_error.percent_98,
                # g_m.orientation_error.median,
            )  # , g_m.attempts)
            print(g_m.attempts)
            # print("MT: ", g_m.motion_time)
            # $print(ik_fail)
            # exit()
            # print(g_m.position_error, g_m.orientation_error)

        g_m = CuroboGroupMetrics.from_list(all_groups)
        print(
            "All: ",
            f"{g_m.success:2.2f}",
            g_m.motion_time.percent_98,
            g_m.time.mean,
            g_m.time.percent_75,
            g_m.position_error.percent_75,
            g_m.orientation_error.percent_75,
        )  # g_m.time, g_m.attempts)
        # print("MT: ", g_m.motion_time)

        # print(g_m.position_error, g_m.orientation_error)

        # exit()
        if write_benchmark:
            if not mpinets_data:
                write_yaml(problems, "robometrics_mb_curobo_solution.yaml")
            else:
                write_yaml(problems, "robometrics_mpinets_curobo_solution.yaml")
        all_files += all_groups
    g_m = CuroboGroupMetrics.from_list(all_files)
    # print(g_m.success, g_m.time, g_m.attempts, g_m.position_error, g_m.orientation_error)
    print("######## FULL SET ############")
    print("All: ", f"{g_m.success:2.2f}")
    print("MT: ", g_m.motion_time)
    print("PT:", g_m.time)
    print("ST: ", g_m.solve_time)
    print("position accuracy: ", g_m.position_error)
    print("orientation accuracy: ", g_m.orientation_error)

    if args.kpi:
        kpi_data = {"Success": g_m.success, "Planning Time": float(g_m.time.mean)}
        write_yaml(kpi_data, join_path(args.save_path, args.file_name + ".yml"))


def get_metrics_obstacles(obs: Dict[str, List[Any]]):
    obs_list = []
    if "cylinder" in obs and len(obs["cylinder"].items()) > 0:
        for _, vi in enumerate(obs["cylinder"].values()):
            obs_list.append(
                Cylinder(
                    np.ravel(vi["pose"][:3]), vi["radius"], vi["height"], np.ravel(vi["pose"][3:])
                )
            )
    if "cuboid" in obs and len(obs["cuboid"].items()) > 0:
        for _, vi in enumerate(obs["cuboid"].values()):
            obs_list.append(
                Cuboid(np.ravel(vi["pose"][:3]), np.ravel(vi["dims"]), np.ravel(vi["pose"][3:]))
            )
    return obs_list


def check_problems(all_problems):
    n_cube = 0
    for problem in all_problems:
        cache = WorldConfig.from_dict(problem["obstacles"]).get_obb_world().get_cache_dict()
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
        default="mg",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="When True, runs only geometric planner",
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

    args = parser.parse_args()
    setup_curobo_logger("error")
    benchmark_mb(args=args, save_kpi=args.kpi)
