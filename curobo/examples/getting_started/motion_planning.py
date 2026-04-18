# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan collision-free trajectories and grasp motions with GPU-accelerated optimization.

.. raw:: html

    <video autoplay="True" loop="True" muted="True" preload="auto" width="100%"><source src="../videos/get_started_motion_plan.webm" type="video/webm"></video>

Motion planning finds a smooth, collision-free path from a start joint
configuration to a goal end-effector pose. cuRobo formulates this as a
trajectory optimization problem: it parameterizes the path as a sequence of
knot points and jointly optimizes them for smoothness (velocity, acceleration,
jerk costs) and feasibility (collision avoidance, joint limits). Multiple
trajectory seeds are optimized in parallel on the GPU, and the best
collision-free result is returned.

cuRobo also supports **grasp planning**, which chains three trajectory segments
-- approach, grasp, and lift -- into a single call. The planner first solves
for a collision-free path to a pre-grasp pose (offset along the approach axis),
then plans the final approach to the grasp pose with finger-link collisions
disabled, and finally plans a lift motion away from the surface.

By the end of this tutorial you will have:

- Initialized a GPU-accelerated motion planner for the Franka Panda robot
- Defined a box obstacle in the planning scene
- Planned a collision-free trajectory from a start configuration to a goal pose
- Planned a three-phase grasp motion (approach, grasp, lift)
- Saved trajectory plots as PDF files

Step 1: Run the tutorial
--------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.motion_planning

This runs both the pose-to-pose planner and the grasp planner in sequence.
To run only one mode:

.. code-block:: bash

   python -m curobo.examples.getting_started.motion_planning --mode pose
   python -m curobo.examples.getting_started.motion_planning --mode grasp

Step 2: Check the output
--------------------------

When the tutorial finishes successfully you will see::

    === Pose-to-Pose Motion Planning ===
    ✓ Planning succeeded!
    Trajectory has 250 waypoints
    Duration: 5.00s
    Trajectory plot saved to: ~/.cache/curobo/examples/motion_planning/motion_plan.pdf

    === Grasp Planning ===
    ✓ Grasp planning succeeded!
      Approach: 120 waypoints
      Grasp:    45 waypoints
      Lift:     60 waypoints
    Trajectory plot saved to: ~/.cache/curobo/examples/motion_planning/grasp_plan.pdf

Output files are written to ``~/.cache/curobo/`` by default. Override with
``--output-dir`` or by setting ``curobo._src.runtime.cache_dir`` in Python.

Step 3: Understand the pipeline
---------------------------------

**Pose-to-pose planning** walks through five stages:

1. **Initialize MotionPlanner**: Load the robot config (``franka.yml``) and a
   scene (``collision_test.yml``). The ``warmup()`` call pre-compiles CUDA
   graphs for faster subsequent planning calls.

2. **Set start and goal**: The start is a ``JointState`` (joint angles in
   radians), and the goal is a ``Pose`` (position + quaternion for the
   end-effector). Obstacle poses use the format
   ``[x, y, z, qw, qx, qy, qz]`` (meters, wxyz quaternion).

3. **Plan**: ``plan_pose`` runs IK to find a goal configuration, then
   optimizes a trajectory across multiple seeds. It returns a result with
   ``success``, the optimized trajectory, and an interpolated version.

4. **Save**: The interpolated joint trajectory is plotted with
   ``matplotlib`` and saved as a PDF.

**Grasp planning** extends this with ``plan_grasp``, which chains three
trajectory segments:

1. **Approach**: Plan from the current configuration to a pre-grasp pose,
   offset along ``grasp_approach_axis`` by ``grasp_approach_offset`` (default
   -15 cm along the tool Z axis). Finger-link collisions are disabled so the
   gripper can reach into tight spaces.

2. **Grasp**: Plan the final approach from the pre-grasp pose to the grasp
   pose itself.

3. **Lift**: Plan a retraction from the grasp pose, offset along
   ``grasp_lift_axis`` by ``grasp_lift_offset``.

The planner accepts a *goal set* of candidate grasp poses via ``ToolPose``,
selects the most reachable one, and returns all three trajectory segments in
a ``GraspPlanResult``.

Step 4: Interactive motion planning with Viser
------------------------------------------------

For an interactive version with a web-based 3D viewer, run:

.. code-block:: bash

   python -m curobo.examples.getting_started.motion_planning --visualize

Open http://localhost:8080 in your browser. Drag the target frame to set
the goal pose, move obstacles, and click "Move" to plan and execute.
Click "Grasp" to plan and execute a three-phase grasp motion.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

import curobo.runtime as runtime
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer
import time

def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "motion_planning"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: float,
    save_path: str,
    title: str = "Joint Trajectory",
    phase_boundaries: Optional[List[int]] = None,
    phase_labels: Optional[List[str]] = None,
):
    """Plot joint positions over time and save to *save_path*.

    Args:
        positions: Joint positions tensor of shape ``(timesteps, n_joints)``.
        joint_names: Label for each joint.
        dt: Time step between waypoints (seconds).
        save_path: Output file path (e.g. ``"trajectory.pdf"``).
        title: Plot title.
        phase_boundaries: Timestep indices where a new phase starts.
        phase_labels: Label for each phase (length must match *phase_boundaries*).
    """
    import matplotlib.pyplot as plt

    pos_np = positions.cpu().numpy()
    n_steps = pos_np.shape[0]
    t = [i * dt for i in range(n_steps)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)

    if phase_boundaries and phase_labels:
        for idx, label in zip(phase_boundaries, phase_labels):
            ax.axvline(x=idx * dt, color="grey", linestyle="--", linewidth=0.8)
            ax.text(
                idx * dt, ax.get_ylim()[1], f" {label}", fontsize=8, va="top",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def pose_planning_example(output_dir: Optional[Path] = None):
    """Plan a collision-free trajectory to a goal pose.

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/motion_planning/``.

    Returns:
        True if planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )

    result = planner.plan_pose(goal_pose, q_start)

    interp_dt = planner.trajopt_solver.config.interpolation_dt
    if result is not None and result.success.any():
        print("✓ Planning succeeded!")
        interpolated = result.get_interpolated_plan()
        n_waypoints = interpolated.position.shape[-2]
        print(f"Trajectory has {n_waypoints} waypoints")
        print(f"Duration: {n_waypoints * interp_dt:.2f}s")

        _plot_trajectory(
            interpolated.position.squeeze(0),
            planner.joint_names,
            dt=interp_dt,
            save_path=str(output_dir / "motion_plan.pdf"),
            title="Pose-to-Pose Trajectory",
        )
        return True
    else:
        print("✗ Planning failed - try adjusting the goal or obstacles")
        return False


def grasp_planning_example(output_dir: Optional[Path] = None):
    """Plan a three-phase grasp motion (approach, grasp, lift).

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/motion_planning/``.

    Returns:
        True if grasp planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        max_goalset=10,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    n_grasps = 3
    positions = torch.zeros(1, 1, 1, n_grasps, 3, device="cuda", dtype=torch.float32)
    positions[..., 0] = 0.5
    positions[0, 0, 0, :, 1] = torch.linspace(-0.15, 0.15, n_grasps)
    positions[..., 2] = 0.3

    quaternions = torch.zeros(1, 1, 1, n_grasps, 4, device="cuda", dtype=torch.float32)
    quaternions[..., 0] = 1.0

    grasp_poses = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=positions,
        quaternion=quaternions,
    )

    result = planner.plan_grasp(
        current_state=q_start,
        grasp_poses=grasp_poses,
        grasp_approach_offset=0.1,
        grasp_lift_offset=0.1,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
        grasp_lift_in_tool_frame=True,
    )

    if result.success is not None and result.success.any():
        print("✓ Grasp planning succeeded!")

        approach = result.approach_interpolated_trajectory
        grasp = result.grasp_interpolated_trajectory
        lift = result.lift_interpolated_trajectory

        if approach is not None:
            print(f"  Approach: {approach.position.shape[-2]} waypoints")
        if grasp is not None:
            print(f"  Grasp:    {grasp.position.shape[-2]} waypoints")
        if lift is not None:
            print(f"  Lift:     {lift.position.shape[-2]} waypoints")

        all_positions = []
        phase_boundaries = []
        phase_labels = []
        offset = 0
        n_joints = approach.position.shape[-1] if approach is not None else None
        for traj, label in [
            (approach, "Approach"),
            (grasp, "Grasp"),
            (lift, "Lift"),
        ]:
            if traj is not None:
                phase_boundaries.append(offset)
                phase_labels.append(label)
                pos = traj.position.reshape(-1, traj.position.shape[-1])
                all_positions.append(pos)
                offset += pos.shape[0]

        if all_positions:
            combined = torch.cat(all_positions, dim=0)
            _plot_trajectory(
                combined,
                planner.joint_names,
                dt=0.02,
                save_path=str(output_dir / "grasp_plan.pdf"),
                title="Grasp Trajectory (Approach → Grasp → Lift)",
                phase_boundaries=phase_boundaries,
                phase_labels=phase_labels,
            )
        return True
    else:
        status = getattr(result, "status", "unknown")
        print(f"✗ Grasp planning failed: {status}")
        return False


def interactive_motion_planning(robot_file="franka.yml", scene_file="collision_test.yml", port=8080):
    """Launch an interactive Viser viewer for motion planning.

    Provides a web-based 3D viewer where you can:
    - Drag the target frame to set the goal pose
    - Drag obstacles to reposition them
    - Click "Move" to plan and execute a collision-free trajectory
    - Click "Grasp" to plan a three-phase grasp motion (approach, grasp, lift)
    """
    import threading

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    config = MotionPlannerCfg.create(robot=robot_file, scene_model=scene_file)
    planner = MotionPlanner(config)

    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    current_state = planner.default_joint_state.clone().unsqueeze(0)

    print("Warming up motion planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    is_moving = False

    def _create_trajectory_image(trajectory, joint_names, title=""):
        """Render a joint trajectory as a PNG image array for the Viser GUI."""
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        traj = trajectory.squeeze(0)
        pos = np.atleast_2d(traj.position[0].cpu().numpy())  # (horizon, dof)
        dt_val = traj.dt.item() if traj.dt is not None else 0.02
        t = np.arange(pos.shape[0]) * dt_val

        vel = np.atleast_2d(traj.velocity[0].cpu().numpy()) if traj.velocity is not None else None
        acc = np.atleast_2d(traj.acceleration[0].cpu().numpy()) if traj.acceleration is not None else None
        jrk = np.atleast_2d(traj.jerk[0].cpu().numpy()) if traj.jerk is not None else None

        n_plots = 1 + (vel is not None) + (acc is not None) + (jrk is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=(5, 2 * n_plots), dpi=100, sharex=True)
        if n_plots == 1:
            axes = [axes]

        plot_data = [(pos, "Position (rad)")]
        if vel is not None:
            plot_data.append((vel, "Velocity (rad/s)"))
        if acc is not None:
            plot_data.append((acc, "Accel (rad/s²)"))
        if jrk is not None:
            plot_data.append((jrk, "Jerk (rad/s³)"))

        for ax, (data, ylabel) in zip(axes, plot_data):
            for j in range(data.shape[1]):
                label = joint_names[j] if j < len(joint_names) else f"J{j}"
                if len(label) > 8:
                    label = label[:6] + ".."
                ax.plot(t, data[:, j], linewidth=1.0, label=label)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        axes[0].legend(loc="upper right", fontsize=7, ncol=2)
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        from PIL import Image
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        buf.close()
        return img_array

    server = viser_viz._server
    traj_plot = server.gui.add_image(
        _create_trajectory_image(
            JointState.from_position(
                planner.default_joint_state.position.unsqueeze(0).unsqueeze(0),
                joint_names=planner.joint_names,
            ),
            planner.joint_names,
            title="No trajectory yet",
        ),
        label="Joint Trajectory",
        format="png",
    )

    def update_obstacles():
        for k in obstacle_frames.keys():
            new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if new_pose != old_obstacle_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                old_obstacle_poses[k] = new_pose.clone()

    def execute_trajectory(trajectory):
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    def on_move(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            if result is not None and result.success.any():
                interp = result.get_interpolated_plan()
                traj_plot.image = _create_trajectory_image(
                    interp, planner.joint_names,
                    title=f"Pose Plan  |  {result.total_time:.3f}s",
                )
                execute_trajectory(interp)
            else:
                print("Motion planning failed")
            is_moving = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    def on_grasp(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_grasp_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            grasp_poses = GoalToolPose.from_poses(target_poses, num_goalset=1)
            result = planner.plan_grasp(
                grasp_poses,
                active_js,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
                grasp_lift_in_tool_frame=True,
            )
            if result is not None and result.success is not None and result.success.any():
                traj_plot.image = _create_trajectory_image(
                    result.approach_interpolated_trajectory, planner.joint_names,
                    title="Approach",
                )
                execute_trajectory(result.approach_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.grasp_interpolated_trajectory, planner.joint_names,
                    title="Grasp",
                )
                execute_trajectory(result.grasp_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.lift_interpolated_trajectory, planner.joint_names,
                    title="Lift",
                )
                execute_trajectory(result.lift_interpolated_trajectory)
            else:
                print("Grasp planning failed")
            is_moving = False

        threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)

    print(f"\nInteractive Motion Planner running at http://localhost:{port}")
    print("  - Drag the target frame to set goal pose")
    print("  - Drag obstacles to reposition them")
    print("  - Click 'Move' for pose-to-pose planning")
    print("  - Click 'Grasp' for approach-grasp-lift planning")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def test():
    """Run motion planning examples as a self-test."""
    out = _get_output_dir()
    assert pose_planning_example(), "Pose planning failed"
    assert (out / "motion_plan.pdf").exists(), "motion_plan.pdf not created"
    assert grasp_planning_example(), "Grasp planning failed"
    assert (out / "grasp_plan.pdf").exists(), "grasp_plan.pdf not created"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Motion Planning with cuRobo",
    )
    parser.add_argument(
        "--mode",
        choices=["pose", "grasp", "all"],
        default="all",
        help="Which example to run (default: all)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run as self-test with assertions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch interactive Viser viewer with Move and Grasp buttons",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka.yml",
        help="Robot config file (default: franka.yml)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="collision_test.yml",
        help="Scene config file (default: collision_test.yml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: ~/.cache/curobo/examples/motion_planning/)",
    )
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.visualize:
        interactive_motion_planning(
            robot_file=args.robot, scene_file=args.scene, port=args.port,
        )
        return

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("pose", "all"):
        print("=== Pose-to-Pose Motion Planning ===")
        pose_planning_example(output_dir=output_dir)
        print()

    if args.mode in ("grasp", "all"):
        print("=== Grasp Planning ===")
        grasp_planning_example(output_dir=output_dir)


if __name__ == "__main__":
    main()
