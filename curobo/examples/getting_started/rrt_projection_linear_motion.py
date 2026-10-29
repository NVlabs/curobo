# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constrain projected RRT to a straight Cartesian end-effector path.

This example gives projected RRT a 1-D task manifold running from the current
end-effector position to a draggable Cartesian target. It calls the raw graph
planner after IK, deliberately bypassing TrajOpt and the B-spline seed fitter,
so the reported Cartesian error belongs to projected RRT itself. Obstacles
come from ``collision_test.yml`` and are also draggable.

Run the interactive example with:

.. code-block:: bash

   python -m curobo.examples.getting_started.rrt_projection_linear_motion

Open ``http://localhost:8080``, drag the Cartesian target, and click
``Plan projected RRT``. Run the one-shot headless check with:

.. code-block:: bash

   python -m curobo.examples.getting_started.rrt_projection_linear_motion --headless

Press Ctrl+C to stop the interactive example.
"""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

import curobo.runtime as runtime
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame
from curobo._src.graph_planner.rrt import BiRRTConfig
from curobo._src.graph_planner.rrt.rrt_projection_sample import RRTSampleProjectionConfig
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose, ToolPoseCriteria

_ROBOT_FILE = "franka.yml"
_SCENE_FILE = "collision_test.yml"
_INTERPOLATION_STEPS = 200
_LINE_TOLERANCE_M = 0.002


def _create_planner() -> MotionPlanner:
    """Create the one-seed BiRRT sample-projection experiment."""
    projection = RRTSampleProjectionConfig(
        iterations=25,
        projection_slots=24,
        history=7,
        line_search_scales=(0.0, 0.1, 0.5, 1.0),
        step_scale=1.0,
        initial_step_scale=0.001,
        use_cuda_graph=True,
    )
    rrt = BiRRTConfig(
        samples_per_iteration=24,
        max_iterations=60,
        failure_extension_iterations=60,
        max_nodes_per_tree=4096,
        cspace_resolution=0.025,
        fixed_steer_steps=64,
        shortcut_path=True,
        sample_projection=projection,
    )
    config = MotionPlannerCfg.create(
        robot=_ROBOT_FILE,
        scene_model=_SCENE_FILE,
        graph_planner_algorithm="rrt",
        rrt_planner_config=rrt,
        num_ik_seeds=1,
        num_trajopt_seeds=1,
        use_cuda_graph=True,
        enable_trajopt=False,
        fit_graph_seed_to_bspline=False,
    )
    planner = MotionPlanner(config)
    planner.ik_solver.update_tool_pose_criteria(
        {
            frame: ToolPoseCriteria.track_position(device_cfg=planner.device_cfg)
            for frame in planner.tool_frames
        }
    )
    return planner


def _initial_state(planner: MotionPlanner) -> JointState:
    """Return the nontrivial starting configuration used by the example."""
    start = planner.default_joint_state.clone()
    start.position += planner.device_cfg.to_device([0.4, -0.2, 0.3, 0.0, -0.2, 0.1, 0.0])
    return start.unsqueeze(0)


def _line_frame_pose(start_position: torch.Tensor, target_position: torch.Tensor) -> Pose:
    """Return a pose whose local Z axis follows the requested Cartesian line."""
    direction = target_position - start_position
    distance = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool(torch.any(distance < 1.0e-4).item()):
        raise ValueError("Target must be at least 0.1 mm from the current tool position")
    z_axis = direction / distance
    reference = torch.zeros_like(z_axis)
    reference[..., 2] = 1.0
    parallel = torch.abs(torch.sum(reference * z_axis, dim=-1, keepdim=True)) > 0.9
    alternate = torch.zeros_like(z_axis)
    alternate[..., 1] = 1.0
    reference = torch.where(parallel, alternate, reference)
    x_axis = torch.linalg.cross(reference, z_axis, dim=-1)
    x_axis /= torch.linalg.vector_norm(x_axis, dim=-1, keepdim=True)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return Pose(position=target_position.clone(), rotation=rotation)


def _make_goals(
    planner: MotionPlanner,
    start: JointState,
    target_pose: Pose,
) -> Tuple[GoalToolPose, GoalToolPose]:
    """Create separate terminal-IK and goal-aligned line-manifold goals."""
    start_poses = planner.compute_kinematics(start).tool_poses.to_dict()
    target_link = planner.tool_frames[0]
    ik_poses = {name: pose.clone() for name, pose in start_poses.items()}
    ik_poses[target_link] = target_pose
    ik_goal = GoalToolPose.from_poses(
        ik_poses,
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )
    constraint_poses = {name: pose.clone() for name, pose in start_poses.items()}
    constraint_poses[target_link] = _line_frame_pose(
        start_poses[target_link].position,
        target_pose.position,
    )
    constraint_goal = GoalToolPose.from_poses(
        constraint_poses,
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )
    return ik_goal, constraint_goal


def _make_problem(
    planner: MotionPlanner,
    line_distance: float,
) -> Tuple[JointState, GoalToolPose, GoalToolPose]:
    """Create the headless check with an initial vertical target offset."""
    start = _initial_state(planner)
    target_link = planner.tool_frames[0]
    target_pose = planner.compute_kinematics(start).tool_poses.to_dict()[target_link]
    target_pose.position[:, 2] += line_distance
    ik_goal, constraint_goal = _make_goals(planner, start, target_pose)
    return start, ik_goal, constraint_goal


def _solve(
    planner: MotionPlanner,
    start: JointState,
    ik_goal: GoalToolPose,
    constraint_goal: GoalToolPose,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], float]:
    """Solve IK, then call only the raw projected graph planner."""
    ik_result = planner.ik_solver.solve_pose(
        goal_tool_poses=ik_goal,
        current_state=start,
        return_seeds=1,
    )
    if not bool(ik_result.success.reshape(-1)[0].item()):
        raise RuntimeError("IK failed for the line endpoint")

    goal_configuration = ik_result.solution.reshape(-1, planner.action_dim)[0].unsqueeze(0)
    path_criteria = {
        frame: ToolPoseCriteria(
            terminal_position_projector=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            terminal_rotation_projector=[[0.0] * 3 for _ in range(3)],
            path_position_projector=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            path_rotation_projector=[[0.0] * 3 for _ in range(3)],
            terminal_tolerance=[0.5 * _LINE_TOLERANCE_M, 0.0],
            path_tolerance=[0.5 * _LINE_TOLERANCE_M, 0.0],
            error_frame=PoseErrorFrame.GOAL,
            device_cfg=planner.device_cfg,
        )
        for frame in planner.tool_frames
    }
    planner.graph_planner.update_tool_pose_constraint(constraint_goal, path_criteria)
    graph_result = planner.graph_planner.find_path(
        start.position,
        goal_configuration,
        interpolate_waypoints=True,
        interpolation_steps=_INTERPOLATION_STEPS,
        interpolation_type=TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory=False,
    )
    if not bool(graph_result.success[0].item()):
        debug = graph_result.debug_info
        projection = debug.get("projection", {})
        raise RuntimeError(
            "no constrained path found after "
            f"{debug.get('growth_steps', '?')}/{debug.get('step_budget', '?')} waves; "
            "the straight line may be blocked or kinematically infeasible "
            f"({projection.get('repaired', 0)} projected candidates were repaired)"
        )

    path = graph_result.interpolated_waypoints[0]
    path_state = JointState.from_position(path, joint_names=planner.joint_names)
    tool_positions = planner.compute_kinematics(path_state).tool_poses.position[:, 0, 0, :]
    projection_stats = graph_result.debug_info["projection"]
    return path, tool_positions, projection_stats, graph_result.solve_time


def _line_errors(
    tool_positions: torch.Tensor,
    line_start: torch.Tensor,
    line_target: torch.Tensor,
) -> torch.Tensor:
    """Return perpendicular distance to the requested infinite line."""
    unit_direction = line_target - line_start
    unit_direction /= torch.linalg.vector_norm(unit_direction)
    offset = tool_positions - line_start
    along_line = torch.sum(offset * unit_direction, dim=-1, keepdim=True)
    perpendicular = offset - along_line * unit_direction
    return torch.linalg.vector_norm(perpendicular, dim=-1)


def _save_plot(
    tool_positions: torch.Tensor,
    line_errors: torch.Tensor,
    line_start: torch.Tensor,
    line_target: torch.Tensor,
    output: Path,
) -> None:
    """Save a Cartesian path and line-error diagnostic."""
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    points = tool_positions.detach().cpu().numpy()
    error_mm = 1000.0 * line_errors.detach().cpu().numpy()
    exact_line = torch.stack((line_start, line_target)).detach().cpu().numpy()

    figure = plt.figure(figsize=(11, 4.5))
    path_axis = figure.add_subplot(1, 2, 1, projection="3d")
    path_axis.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="tab:blue",
        linewidth=2.5,
        label="projected RRT",
    )
    path_axis.plot(
        exact_line[:, 0],
        exact_line[:, 1],
        exact_line[:, 2],
        "--",
        color="black",
        linewidth=1.5,
        label="exact line",
    )
    path_axis.scatter(*points[0], color="tab:green", s=40, label="start")
    path_axis.scatter(*points[-1], color="tab:red", s=40, label="goal")
    path_axis.set_xlabel("X [m]")
    path_axis.set_ylabel("Y [m]")
    path_axis.set_zlabel("Z [m]")
    path_axis.set_title("End-effector path")
    all_points = torch.cat((tool_positions, line_start[None], line_target[None]), dim=0)
    data_min = all_points.min(dim=0).values.detach().cpu().numpy()
    data_max = all_points.max(dim=0).values.detach().cpu().numpy()
    spatial_span = max(float((data_max - data_min).max()), 1.0e-3)
    center = 0.5 * (data_min + data_max)
    half_span = 0.55 * spatial_span
    path_axis.set_xlim(center[0] - half_span, center[0] + half_span)
    path_axis.set_ylim(center[1] - half_span, center[1] + half_span)
    path_axis.set_zlim(center[2] - half_span, center[2] + half_span)
    path_axis.set_box_aspect((1.0, 1.0, 1.0))
    path_axis.legend()

    error_axis = figure.add_subplot(1, 2, 2)
    error_axis.plot(error_mm, color="tab:orange", linewidth=2.0)
    error_axis.axhline(
        1000.0 * _LINE_TOLERANCE_M,
        color="tab:red",
        linestyle="--",
        label="example acceptance",
    )
    error_axis.set_xlabel("Interpolated waypoint")
    error_axis.set_ylabel("Distance to line [mm]")
    error_axis.set_title("Cartesian line error")
    error_axis.grid(alpha=0.3)
    error_axis.legend()

    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _add_path_visuals(
    server,
    tool_positions: torch.Tensor,
    line_start: torch.Tensor,
    line_target: torch.Tensor,
) -> None:
    """Update the planned trace and exact line overlays in Viser."""
    import numpy as np

    points = tool_positions.detach().cpu().numpy().astype(np.float32)
    planned_segments = np.stack((points[:-1], points[1:]), axis=1)
    exact_line = torch.stack((line_start, line_target)).detach().cpu().numpy().astype(np.float32)
    server.scene.add_line_segments(
        "/rrt_projection/tool_path",
        points=planned_segments,
        colors=np.array([30, 144, 255], dtype=np.uint8),
        line_width=4.0,
    )
    server.scene.add_line_segments(
        "/rrt_projection/exact_line",
        points=exact_line[None, ...],
        colors=np.array([50, 205, 50], dtype=np.uint8),
        line_width=2.0,
    )


def _add_exact_line(server, start_position, target_position) -> None:
    """Update the green ideal-line overlay before a plan is available."""
    import numpy as np

    line = np.stack((start_position, target_position), axis=0).astype(np.float32)
    server.scene.add_line_segments(
        "/rrt_projection/exact_line",
        points=line[None, ...],
        colors=np.array([50, 205, 50], dtype=np.uint8),
        line_width=2.0,
    )


def _interactive(line_distance: float, output_dir: Path, port: int) -> None:
    """Run the draggable Cartesian target and replanning UI."""
    import numpy as np

    from curobo.viewer import ViserVisualizer

    planner = _create_planner()
    visualizer = ViserVisualizer(
        ContentPath(robot_config_file=_ROBOT_FILE),
        add_robot_to_scene=True,
        add_control_frames=False,
        connect_port=port,
    )
    server = visualizer._server
    scene_cfg = planner.config.scene_collision_cfg.scene_model
    obstacle_frames = visualizer.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        name: Pose.from_numpy(frame.position, frame.wxyz)
        for name, frame in obstacle_frames.items()
    }
    initial_state = _initial_state(planner)
    current_state = initial_state.clone()
    visualizer.set_joint_state(current_state.squeeze(0))

    initial_pose = planner.compute_kinematics(current_state).tool_poses.to_dict()[
        planner.tool_frames[0]
    ]
    target_position = initial_pose.position[0].detach().cpu().numpy().copy()
    target_position[2] += line_distance
    target = server.scene.add_transform_controls(
        "/rrt_projection/target",
        position=target_position,
        wxyz=initial_pose.quaternion[0].detach().cpu().numpy(),
        scale=0.12,
        disable_rotations=True,
    )
    _add_exact_line(
        server,
        initial_pose.position[0].detach().cpu().numpy(),
        target_position,
    )
    initial_tool_position = initial_pose.position[0].detach().cpu().numpy().copy()
    current_tool_position = initial_tool_position.copy()

    with server.gui.add_folder("Projected RRT"):
        plan_button = server.gui.add_button("Plan projected RRT", color="green")
        reset_button = server.gui.add_button("Reset robot")
        status = server.gui.add_text(
            "Status",
            initial_value="Drag the target, then click Plan",
            disabled=True,
        )
        max_error_text = server.gui.add_text(
            "Maximum line error",
            initial_value="--",
            disabled=True,
        )
        projection_text = server.gui.add_text(
            "Projection work",
            initial_value="--",
            disabled=True,
        )

    busy = False
    has_path = False
    command_queue: queue.Queue = queue.Queue()

    def update_obstacles() -> None:
        """Copy any dragged Viser obstacle poses into the collision checker."""
        for name, frame in obstacle_frames.items():
            new_pose = Pose.from_numpy(frame.position, frame.wxyz)
            if new_pose != old_obstacle_poses[name]:
                planner.scene_collision_checker.update_obstacle_pose(name, new_pose)
                old_obstacle_poses[name] = new_pose.clone()

    def sync_target_line() -> None:
        target_position_now = np.asarray(target.position).copy()
        _add_exact_line(server, current_tool_position.copy(), target_position_now)

    @target.on_update
    def _target_updated(_):
        nonlocal has_path
        if not busy:
            sync_target_line()
            if has_path:
                server.scene.remove_by_name("/rrt_projection/tool_path")
                has_path = False
            status.value = "Target changed; click Plan"
            max_error_text.value = "--"
            projection_text.value = "--"

    @plan_button.on_click
    def _plan(_):
        nonlocal busy
        if busy:
            return
        busy = True
        plan_button.disabled = True
        reset_button.disabled = True
        status.value = "Planning..."
        command_queue.put(
            (
                "plan",
                np.asarray(target.position).copy(),
                np.asarray(target.wxyz).copy(),
            )
        )

    @reset_button.on_click
    def _reset(_):
        nonlocal busy
        if busy:
            return
        busy = True
        plan_button.disabled = True
        reset_button.disabled = True
        command_queue.put(("reset",))

    def plan_and_execute(requested_position, requested_quaternion) -> None:
        """Run all CUDA planner work on the main thread."""
        nonlocal busy, current_state, current_tool_position, has_path
        try:
            update_obstacles()
            target_pose = Pose.from_numpy(
                requested_position,
                requested_quaternion,
                device_cfg=planner.device_cfg,
            )
            ik_goal, constraint_goal = _make_goals(
                planner,
                current_state,
                target_pose,
            )
            line_start = (
                planner.compute_kinematics(current_state).tool_poses.position[0, 0, 0].clone()
            )
            line_target = constraint_goal.position[0, 0, 0, 0].clone()
            path, tool_positions, projection_stats, solve_time = _solve(
                planner,
                current_state,
                ik_goal,
                constraint_goal,
            )
            line_errors = _line_errors(tool_positions, line_start, line_target)
            max_error = float(line_errors.max().item())
            repaired = int(projection_stats.get("repaired", 0))
            calls = int(projection_stats.get("calls", 0))
            _add_path_visuals(
                server,
                tool_positions,
                line_start,
                line_target,
            )
            has_path = True
            _save_plot(
                tool_positions,
                line_errors,
                line_start,
                line_target,
                output_dir / "rrt_projection_linear_motion.pdf",
            )
            max_error_text.value = f"{1000.0 * max_error:.3f} mm"
            projection_text.value = f"{repaired} repairs in {calls} calls"
            status.value = f"Plan succeeded in {1000.0 * solve_time:.1f} ms"

            for position in path:
                visualizer.set_joint_state(
                    JointState.from_position(position, joint_names=planner.joint_names)
                )
                time.sleep(0.02)
            current_state = JointState.from_position(
                path[-1].unsqueeze(0),
                joint_names=planner.joint_names,
            )
            current_tool_position = tool_positions[-1].detach().cpu().numpy().copy()
        except Exception as error:
            status.value = f"Planning failed: {error}"
            print(status.value)
        finally:
            busy = False
            plan_button.disabled = False
            reset_button.disabled = False

    def reset_robot() -> None:
        """Apply reset state on the CUDA-owning main thread."""
        nonlocal busy, current_state, current_tool_position, has_path
        try:
            current_state = initial_state.clone()
            current_tool_position = initial_tool_position.copy()
            visualizer.set_joint_state(current_state.squeeze(0))
            reset_target = initial_tool_position.copy()
            reset_target[2] += line_distance
            target.position = reset_target
            target.wxyz = initial_pose.quaternion[0].detach().cpu().numpy()
            _add_exact_line(server, initial_tool_position, reset_target)
            if has_path:
                server.scene.remove_by_name("/rrt_projection/tool_path")
                has_path = False
            status.value = "Robot reset; drag the Cartesian target"
            max_error_text.value = "--"
            projection_text.value = "--"
        finally:
            busy = False
            plan_button.disabled = False
            reset_button.disabled = False

    print(f"\nInteractive projected RRT running at http://localhost:{port}")
    print("  - Drag the Cartesian position target gizmo")
    print("  - Drag collision_test.yml obstacles to reposition them")
    print("  - Click 'Plan projected RRT' to plan and animate")
    print("  - Click 'Reset robot' to return to the initial state")
    print("Press Ctrl+C to exit.\n")
    try:
        while True:
            try:
                command = command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command[0] == "plan":
                plan_and_execute(command[1], command[2])
            elif command[0] == "reset":
                reset_robot()
    finally:
        planner.destroy()


def run_example(
    line_distance: float = 0.2,
    output_dir: Path | None = None,
) -> bool:
    """Run projected RRT and return whether its path passes all checks."""
    output_dir = output_dir or (
        Path(runtime.cache_dir) / "examples" / "rrt_projection_linear_motion"
    )
    planner = _create_planner()
    try:
        start, ik_goal, constraint_goal = _make_problem(planner, line_distance)
        line_start = planner.compute_kinematics(start).tool_poses.position[0, 0, 0].clone()
        line_target = constraint_goal.position[0, 0, 0, 0].clone()
        path, tool_positions, projection_stats, solve_time = _solve(
            planner,
            start,
            ik_goal,
            constraint_goal,
        )
        line_errors = _line_errors(tool_positions, line_start, line_target)
        max_error = float(line_errors.max().item())
        rms_error = float(torch.sqrt(torch.mean(line_errors.square())).item())
        plot_path = output_dir / "rrt_projection_linear_motion.pdf"
        _save_plot(tool_positions, line_errors, line_start, line_target, plot_path)

        repaired = int(projection_stats.get("repaired", 0))
        projection_calls = int(projection_stats.get("calls", 0))
        success = max_error <= _LINE_TOLERANCE_M and projection_calls > 0 and repaired > 0
        print("\n=== Projected RRT Cartesian-line experiment ===")
        print(f"Graph solve time:       {1000.0 * solve_time:.2f} ms")
        print(f"Interpolated waypoints: {path.shape[0]}")
        print(f"Maximum line error:     {1000.0 * max_error:.3f} mm")
        print(f"RMS line error:         {1000.0 * rms_error:.3f} mm")
        print(f"Projection calls:       {projection_calls}")
        print(f"Candidates repaired:    {repaired}")
        print(f"Diagnostic plot:        {plot_path}")
        print(f"Result:                 {'PASS' if success else 'FAIL'}")

        return success
    finally:
        planner.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively verify projected RRT on a Cartesian line constraint"
    )
    parser.add_argument(
        "--line-distance",
        type=float,
        default=0.2,
        help="Initial target's world-Z offset in meters (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the diagnostic PDF",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        action="store_true",
        help="Run one plan, save the diagnostic PDF, and exit",
    )
    mode.add_argument(
        "--visualize",
        action="store_true",
        help="Run the interactive Viser UI (the default)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    return parser.parse_args()


def main() -> None:
    """Run the command-line example."""
    args = _parse_args()
    try:
        output_dir = args.output_dir or (
            Path(runtime.cache_dir) / "examples" / "rrt_projection_linear_motion"
        )
        if not args.headless:
            _interactive(args.line_distance, output_dir, args.port)
            return
        success = run_example(line_distance=args.line_distance, output_dir=output_dir)
    except KeyboardInterrupt:
        print("\nStopped projected-RRT visualization.")
        return
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
