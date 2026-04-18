# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark forward kinematics, pose cost, self/scene collision gradient time.

Measures per-call time and memory for:
1. Forward kinematics only.
2. Forward kinematics + tool pose cost + backward pass (gradient descent steps).
3. Above + self collision cost.
4. Above + scene (world) collision cost.
"""

import numpy as np
import pandas as pd
import tabulate
import torch

import curobo.runtime as runtime

# Enable CUDA event timing for accurate benchmark measurements.
runtime.cuda_event_timers = True

from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.geom.collision import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.config_io import join_path, load_yaml, write_yaml
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import create_graph_executor
from curobo.content import get_robot_configs_path


DEFAULT_ROBOTS = ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"]


def _load_kinematics(robot_file_name: str, keep_collision: bool) -> Kinematics:
    """Load a Kinematics instance from a robot YAML, optionally stripping collision."""
    robot_file = load_yaml(join_path(get_robot_configs_path(), robot_file_name))
    if "robot_cfg" in robot_file:
        robot_file = robot_file["robot_cfg"]
    if not keep_collision:
        robot_file["kinematics"]["collision_link_names"] = None
    robot_file["kinematics"]["lock_joints"] = {}
    return Kinematics(KinematicsCfg.from_data_dict(robot_file))


def _forward_kinematics_fn(kin: Kinematics):
    """Return a callable that takes a [batch, dof] tensor and runs FK."""

    def _run(q_sample: torch.Tensor):
        state = kin.compute_kinematics(JointState.from_position(q_sample))
        return state.tool_poses.position, state.tool_poses.quaternion

    return _run


def _make_q_test(batch_size: int, dof: int, device: torch.device) -> torch.Tensor:
    """Create 5 random joint configurations of shape (5, batch_size, dof)."""
    return (
        torch.rand((batch_size * 5, dof), device=device)
        .view(5, batch_size, dof)
        .contiguous()
    )


def run_forward_kinematics_benchmark(
    b_list: list[int], use_cuda_graph: bool = False, prefix: str = ""
):
    """Benchmark forward kinematics only."""
    print(f"run_forward_kinematics_benchmark with use_cuda_graph: {use_cuda_graph}")
    robot_list = DEFAULT_ROBOTS

    results = {
        "robot": [],
        "b_size": [],
        "time_ms": [],
        "time_per_sample_ms": [],
        "memory_mb": [],
        "first_call_time_ms": [],
    }
    device = torch.device("cuda:0")

    for robot in robot_list:
        kin = _load_kinematics(robot, keep_collision=False)
        fk_fn = _forward_kinematics_fn(kin)

        for b_size in b_list:
            torch.cuda.reset_peak_memory_stats(device=device)
            q_test = _make_q_test(b_size, kin.get_dof(), device)

            first_call_time = 0.0
            for i in range(5):
                timer = CudaEventTimer().start()
                fk_fn(q_test[i])
                if i == 0:
                    first_call_time = 1000.0 * timer.stop()

            if use_cuda_graph:
                graph_timer = CudaEventTimer().start()
                executor = create_graph_executor(
                    capture_fn=fk_fn,
                    device=device,
                    use_cuda_graph=True,
                    clone_outputs=False,
                )
                executor.warmup(q_test[0])
                first_call_time += 1000.0 * graph_timer.stop()
            else:
                executor = None

            results["first_call_time_ms"].append(first_call_time)

            dt_list = []
            for i in range(5):
                timer = CudaEventTimer().start()
                if executor is not None:
                    executor(q_test[i])
                else:
                    fk_fn(q_test[i])
                dt_list.append(timer.stop())

            mean_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2
            mean_time = 1000.0 * float(np.mean(dt_list))
            results["robot"].append(robot)
            results["b_size"].append(b_size)
            results["time_ms"].append(mean_time)
            results["time_per_sample_ms"].append(mean_time / b_size)
            results["memory_mb"].append(mean_memory)

    df = pd.DataFrame(results)
    print(tabulate.tabulate(df, headers="keys", tablefmt="grid"))
    suffix = "cuda_graph" if use_cuda_graph else "no_cuda_graph"
    write_yaml(
        results,
        join_path(
            "benchmark/log", prefix + "forward_kinematics_benchmark_curobo" + suffix + ".yml"
        ),
    )


def _make_goal_poses(kin: Kinematics, device: torch.device):
    """Create a single-goal GoalToolPose from a random joint configuration."""
    q_goal = torch.rand((1, kin.get_dof()), device=device)
    state = kin.compute_kinematics(JointState.from_position(q_goal))
    return state.tool_poses.as_goal()


def _build_pose_cost(kin: Kinematics, device_cfg: DeviceCfg) -> ToolPoseCost:
    """Build a ToolPoseCost configured for this robot's tool frames."""
    cfg = ToolPoseCostCfg(
        weight=torch.tensor([1.0, 1.0], device=device_cfg.device),
        tool_frames=kin.tool_frames,
        use_lie_group=False,
        device_cfg=device_cfg,
    )
    return ToolPoseCost(cfg)


def run_kinematics_pose_gradient_benchmark(
    b_list: list[int], use_cuda_graph: bool = False, prefix: str = ""
):
    """Benchmark FK + tool pose cost + backward pass."""
    print(f"run_kinematics_pose_gradient_benchmark with use_cuda_graph: {use_cuda_graph}")
    gd_iters = 10
    robot_list = DEFAULT_ROBOTS

    results = {
        "robot": [],
        "b_size": [],
        "time_ms": [],
        "time_per_sample_ms": [],
        "memory_mb": [],
        "first_call_time_ms": [],
    }
    device = torch.device("cuda:0")
    device_cfg = DeviceCfg(device=device)

    for robot in robot_list:
        kin = _load_kinematics(robot, keep_collision=False)
        pose_cost = _build_pose_cost(kin, device_cfg)
        goal_poses = _make_goal_poses(kin, device)

        for b_size in b_list:
            torch.cuda.reset_peak_memory_stats(device=device)
            pose_cost.setup_batch_tensors(b_size, 1)
            idxs_goal = torch.zeros((b_size, 1), device=device, dtype=torch.int32)
            q_test = _make_q_test(b_size, kin.get_dof(), device)

            def run_gd_steps(q_sample: torch.Tensor):
                q_sample = q_sample.detach()
                for _ in range(gd_iters):
                    q_sample.requires_grad_(True)
                    state = kin.compute_kinematics(JointState.from_position(q_sample))
                    cost, _, _, _ = pose_cost.forward(
                        state.tool_poses, goal_poses, idxs_goal
                    )
                    loss = torch.sum(cost)
                    loss.backward()
                    q_sample = q_sample.detach() - 0.01 * q_sample.grad
                return q_sample.detach(), loss.detach()

            first_call_time = 0.0
            for i in range(5):
                timer = CudaEventTimer().start()
                run_gd_steps(q_test[i].clone())
                if i == 0:
                    first_call_time = 1000.0 * timer.stop()

            if use_cuda_graph:
                graph_timer = CudaEventTimer().start()
                executor = create_graph_executor(
                    capture_fn=run_gd_steps,
                    device=device,
                    use_cuda_graph=True,
                    clone_outputs=False,
                )
                executor.warmup(q_test[0].clone().detach())
                first_call_time += 1000.0 * graph_timer.stop()
            else:
                executor = None

            results["first_call_time_ms"].append(first_call_time)

            dt_list = []
            for i in range(5):
                timer = CudaEventTimer().start()
                if executor is not None:
                    executor(q_test[i].clone().detach())
                else:
                    run_gd_steps(q_test[i].clone())
                dt_list.append(timer.stop())

            mean_time = 1000.0 * float(np.mean(dt_list)) / gd_iters
            mean_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2
            results["robot"].append(robot)
            results["b_size"].append(b_size)
            results["time_ms"].append(mean_time)
            results["time_per_sample_ms"].append(mean_time / b_size)
            results["memory_mb"].append(mean_memory)

    df = pd.DataFrame(results)
    print(tabulate.tabulate(df, headers="keys", tablefmt="grid"))
    suffix = "cuda_graph" if use_cuda_graph else "no_cuda_graph"
    write_yaml(
        results,
        join_path(
            "benchmark/log", prefix + "kinematics_pose_gradient_benchmark_curobo" + suffix + ".yml"
        ),
    )


def run_kinematics_pose_gradient_self_collision_benchmark(
    b_list: list[int], use_cuda_graph: bool = False, prefix: str = ""
):
    """Benchmark FK + tool pose cost + self collision cost + backward pass."""
    print(f"run_kinematics_pose_gradient_self_collision_benchmark with use_cuda_graph: {use_cuda_graph}")
    gd_iters = 10
    robot_list = DEFAULT_ROBOTS

    results = {
        "robot": [],
        "b_size": [],
        "time_ms": [],
        "time_per_sample_ms": [],
        "memory_mb": [],
        "first_call_time_ms": [],
    }
    device = torch.device("cuda:0")
    device_cfg = DeviceCfg(device=device)

    for robot in robot_list:
        kin = _load_kinematics(robot, keep_collision=True)
        pose_cost = _build_pose_cost(kin, device_cfg)

        self_collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=torch.tensor([1.0], device=device),
                self_collision_kin_config=kin.get_self_collision_config(),
                device_cfg=device_cfg,
            )
        )

        goal_poses = _make_goal_poses(kin, device)

        for b_size in b_list:
            torch.cuda.reset_peak_memory_stats(device=device)
            pose_cost.setup_batch_tensors(b_size, 1)
            self_collision_cost.setup_batch_tensors(b_size, 1)
            idxs_goal = torch.zeros((b_size, 1), device=device, dtype=torch.int32)
            q_test = _make_q_test(b_size, kin.get_dof(), device)

            def run_gd_steps(q_sample: torch.Tensor):
                q_sample = q_sample.detach()
                for _ in range(gd_iters):
                    q_sample.requires_grad_(True)
                    state = kin.compute_kinematics(JointState.from_position(q_sample))
                    pose_cost_value, _, _, _ = pose_cost.forward(
                        state.tool_poses, goal_poses, idxs_goal
                    )
                    self_collision_value = self_collision_cost.forward(state.robot_spheres)
                    cost = torch.cat([pose_cost_value, self_collision_value], dim=-1)
                    loss = torch.sum(cost)
                    loss.backward()
                    q_sample = q_sample.detach() - 0.01 * q_sample.grad
                return q_sample.detach()

            first_call_time = 0.0
            for i in range(5):
                timer = CudaEventTimer().start()
                run_gd_steps(q_test[i].clone())
                if i == 0:
                    first_call_time = 1000.0 * timer.stop()

            if use_cuda_graph:
                graph_timer = CudaEventTimer().start()
                executor = create_graph_executor(
                    capture_fn=run_gd_steps,
                    device=device,
                    use_cuda_graph=True,
                    clone_outputs=False,
                )
                executor.warmup(q_test[0].clone().detach())
                first_call_time += 1000.0 * graph_timer.stop()
            else:
                executor = None

            results["first_call_time_ms"].append(first_call_time)

            dt_list = []
            for i in range(5):
                timer = CudaEventTimer().start()
                if executor is not None:
                    executor(q_test[i].clone().detach())
                else:
                    run_gd_steps(q_test[i].clone())
                dt_list.append(timer.stop())

            mean_time = 1000.0 * float(np.mean(dt_list)) / gd_iters
            mean_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2
            results["robot"].append(robot)
            results["b_size"].append(b_size)
            results["time_ms"].append(mean_time)
            results["time_per_sample_ms"].append(mean_time / b_size)
            results["memory_mb"].append(mean_memory)

    df = pd.DataFrame(results)
    print(tabulate.tabulate(df, headers="keys", tablefmt="grid"))
    suffix = "cuda_graph" if use_cuda_graph else "no_cuda_graph"
    write_yaml(
        results,
        join_path(
            "benchmark/log",
            prefix + "kinematics_pose_gradient_self_collision_benchmark_curobo" + suffix + ".yml",
        ),
    )


def run_kinematics_pose_gradient_self_collision_world_collision_benchmark(
    b_list: list[int], use_cuda_graph: bool = False, prefix: str = ""
):
    """Benchmark FK + tool pose cost + self collision + scene collision + backward pass."""
    print(f"run_kinematics_pose_gradient_self_collision_world_collision_benchmark with use_cuda_graph: {use_cuda_graph}")
    gd_iters = 10
    robot_list = DEFAULT_ROBOTS

    results = {
        "robot": [],
        "b_size": [],
        "time_ms": [],
        "time_per_sample_ms": [],
        "memory_mb": [],
        "first_call_time_ms": [],
    }
    device = torch.device("cuda:0")
    device_cfg = DeviceCfg(device=device)

    scene_model_dict = {
        "cuboid": {
            "table": {
                "dims": [2.2, 2.2, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            },
            "cube6": {
                "dims": [0.1, 0.1, 1.5],
                "pose": [0.45, 0.0, 0.3, 1, 0, 0, 0.0],
            },
        },
    }

    for robot in robot_list:
        kin = _load_kinematics(robot, keep_collision=True)
        pose_cost = _build_pose_cost(kin, device_cfg)

        self_collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=torch.tensor([1.0], device=device),
                self_collision_kin_config=kin.get_self_collision_config(),
                device_cfg=device_cfg,
            )
        )

        scene_collision_checker = SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=SceneCfg.create(scene_model_dict),
                cache={"cuboid": 10},
            )
        )

        scene_collision_cost_cfg = SceneCollisionCostCfg(
            weight=torch.tensor([1.0], device=device),
            device_cfg=device_cfg,
        )
        scene_collision_cost_cfg.scene_collision_checker = scene_collision_checker
        scene_collision_cost_cfg.update_num_spheres(kin.total_spheres)
        scene_collision_cost = SceneCollisionCost(scene_collision_cost_cfg)

        goal_poses = _make_goal_poses(kin, device)

        for b_size in b_list:
            torch.cuda.reset_peak_memory_stats(device=device)
            pose_cost.setup_batch_tensors(b_size, 1)
            self_collision_cost.setup_batch_tensors(b_size, 1)
            scene_collision_cost.setup_batch_tensors(b_size, 1)
            idxs_goal = torch.zeros((b_size, 1), device=device, dtype=torch.int32)
            idxs_env = torch.zeros((b_size,), device=device, dtype=torch.int32)
            q_test = _make_q_test(b_size, kin.get_dof(), device)

            def run_gd_steps(q_sample: torch.Tensor):
                q_sample = q_sample.detach()
                for _ in range(gd_iters):
                    q_sample.requires_grad_(True)
                    state = kin.compute_kinematics(JointState.from_position(q_sample))
                    pose_cost_value, _, _, _ = pose_cost.forward(
                        state.tool_poses, goal_poses, idxs_goal
                    )
                    self_collision_value = self_collision_cost.forward(state.robot_spheres)
                    scene_collision_value = scene_collision_cost.forward(state, idxs_env)
                    cost = torch.cat(
                        [pose_cost_value, self_collision_value, scene_collision_value], dim=-1
                    )
                    loss = torch.sum(cost)
                    loss.backward()
                    q_sample = q_sample.detach() - 0.01 * q_sample.grad
                return q_sample.detach()

            first_call_time = 0.0
            for i in range(5):
                timer = CudaEventTimer().start()
                run_gd_steps(q_test[i].clone())
                if i == 0:
                    first_call_time = 1000.0 * timer.stop()

            if use_cuda_graph:
                graph_timer = CudaEventTimer().start()
                executor = create_graph_executor(
                    capture_fn=run_gd_steps,
                    device=device,
                    use_cuda_graph=True,
                    clone_outputs=False,
                )
                executor.warmup(q_test[0].clone().detach())
                first_call_time += 1000.0 * graph_timer.stop()
            else:
                executor = None

            results["first_call_time_ms"].append(first_call_time)

            dt_list = []
            for i in range(5):
                timer = CudaEventTimer().start()
                if executor is not None:
                    executor(q_test[i].clone().detach())
                else:
                    run_gd_steps(q_test[i].clone())
                dt_list.append(timer.stop())

            mean_time = 1000.0 * float(np.mean(dt_list)) / gd_iters
            mean_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2
            results["robot"].append(robot)
            results["b_size"].append(b_size)
            results["time_ms"].append(mean_time)
            results["time_per_sample_ms"].append(mean_time / b_size)
            results["memory_mb"].append(mean_memory)

    df = pd.DataFrame(results)
    print(tabulate.tabulate(df, headers="keys", tablefmt="grid"))
    suffix = "cuda_graph" if use_cuda_graph else "no_cuda_graph"
    write_yaml(
        results,
        join_path(
            "benchmark/log",
            prefix
            + "kinematics_pose_gradient_self_collision_world_collision_benchmark_curobo"
            + suffix
            + ".yml",
        ),
    )


def main(
    b_list: list[int] | None = None, use_cuda_graph: bool = True, prefix: str = "curobov2"
):
    if b_list is None:
        b_list = [1000]

    _ = torch.zeros((10, 10), device="cuda")
    run_forward_kinematics_benchmark(
        b_list=b_list, use_cuda_graph=use_cuda_graph, prefix=prefix
    )
    run_kinematics_pose_gradient_benchmark(
        b_list=b_list, use_cuda_graph=use_cuda_graph, prefix=prefix
    )
    run_kinematics_pose_gradient_self_collision_benchmark(
        b_list=b_list, use_cuda_graph=use_cuda_graph, prefix=prefix
    )
    run_kinematics_pose_gradient_self_collision_world_collision_benchmark(
        b_list=b_list, use_cuda_graph=use_cuda_graph, prefix=prefix
    )


if __name__ == "__main__":
    main()
