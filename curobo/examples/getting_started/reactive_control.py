# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Track a moving goal pose in real time with model predictive control.

.. raw:: html

    <div style="display:flex;gap:8px;">
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:50%"><source src="../videos/get_started_mpc.webm" type="video/webm"></video>
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:50%"><source src="../videos/get_started_humanoid_mpc.webm" type="video/webm"></video>
    </div>

Model Predictive Control (MPC) re-optimizes a short trajectory at every
control step, producing smooth actions that drive the robot toward the goal
while respecting joint limits and avoiding collisions. By warm-starting each
solve from the previous solution, cuRobo's MPC achieves real-time rates
suitable for closed-loop control and teleoperation.

Unlike single-shot motion planning (which computes one trajectory up front),
MPC continuously adapts to changing goals and dynamic obstacles. This makes
it the right choice when the target is not known in advance or when the
environment changes during execution.

By the end of this tutorial you will have:

- Configured and initialized cuRobo's MPC solver
- Set up a tracking problem from the robot's retract configuration
- Offset the end-effector goal by 20 cm and converged on it
- Saved a trajectory plot as a PDF

Step 1: Run the tutorial
--------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.reactive_control

To try MPC with a humanoid robot:

.. code-block:: bash

   python -m curobo.examples.getting_started.reactive_control --visualize --robot unitree_g1.yml

Step 2: Check the output
--------------------------

When the tutorial finishes successfully you will see::

    Target link: panda_hand
    Running MPC for 100 steps...
      Step 25/100  position error: 0.XXXX
      Step 50/100  position error: 0.XXXX
      Step 75/100  position error: 0.XXXX
      Step 100/100 position error: 0.XXXX
    Trajectory plot saved to: ~/.cache/curobo/examples/reactive_control/reactive_control.pdf

Step 3: Understand the pipeline
---------------------------------

The example runs five stages:

1. **Configure MPC**: Create an ``MPCSolverCfg`` with the robot config, scene
   model, ``optimization_dt`` (time resolution of the planned trajectory), and
   ``interpolation_steps`` (intermediate steps between optimization knots).

2. **Setup the problem**: Initialize the MPC solver with the robot's retract
   configuration and zero velocity/acceleration via ``setup``.

3. **Set the goal**: Compute the current end-effector pose with FK, offset it
   by 20 cm in Y, and pass it to ``update_goal_tool_poses``.

4. **Control loop**: At each step call ``optimize_action_sequence``, extract the
   last action in the optimized horizon, and feed it back as the current state
   for the next iteration. The solver warm-starts from the previous solution.

5. **Save**: Stack the collected joint positions into a trajectory and plot
   the joint positions over time with ``matplotlib``.

Step 4: Interactive MPC with Viser
------------------------------------

For an interactive version with a web-based 3D viewer, run:

.. code-block:: bash

   python -m curobo.examples.getting_started.reactive_control --visualize

Open http://localhost:8080 in your browser. Drag the control frames to update
the target pose and watch the robot track the goal in real time. Obstacles
in the scene are also interactive. Use ``--robot`` to change the robot and
``--port`` to change the server port.
"""

# Standard Library
import argparse
import sys
import time
from pathlib import Path

# Third Party
import torch

# CuRobo
from curobo import runtime
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer


def reactive_control(num_steps: int = 100):
    """Run a headless MPC control loop that tracks a target pose.

    This function sets up an MPC solver for the Franka robot, offsets the
    end-effector goal by 20 cm in the Y direction, and runs a fixed number
    of MPC iterations to converge on the target. The resulting trajectory
    is saved as a USD animation.

    Args:
        num_steps: Number of MPC iterations to run.

    Returns:
        bool: True if the trajectory was saved successfully.
    """
    # 1. Configure MPC
    config = ModelPredictiveControlCfg.create(
        robot="franka.yml",
        scene_model="collision_table.yml",
        use_cuda_graph=True,
        optimization_dt=0.025,
        interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(config)

    # 2. Setup from retract configuration
    current_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0),
        joint_names=mpc.joint_names,
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)

    mpc.setup(current_state)

    # 3. Set goal: offset current end-effector pose by 20 cm in Y
    kin_result = mpc.compute_kinematics(current_state)
    goal_poses = kin_result.tool_poses.to_dict()
    target_link = mpc.tool_frames[0]
    goal_poses[target_link].position[..., 1] += 0.2

    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(goal_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
        run_ik=True,
    )

    print(f"Target link: {target_link}")
    print(f"Running MPC for {num_steps} steps...")

    # 4. Control loop
    trajectory_positions = [current_state.position.squeeze(0).clone()]

    for step in range(num_steps):
        result = mpc.optimize_action_sequence(current_state)

        if result.action_sequence is not None and result.action_sequence.position.shape[1] > 0:
            next_position = result.action_sequence.position[:, -1, :]
            current_state = JointState.from_position(
                next_position.clone(),
                joint_names=mpc.joint_names,
            )
            current_state.velocity = result.action_sequence.velocity[:, -1, :]
            current_state.acceleration = result.action_sequence.acceleration[:, -1, :]

            trajectory_positions.append(next_position.squeeze(0).clone())

        if (step + 1) % 25 == 0:
            pos_err = result.position_error
            err_str = f"{pos_err.item():.4f}" if pos_err is not None else "N/A"
            print(f"  Step {step + 1}/{num_steps}  position error: {err_str}")

    # 5. Save trajectory plot
    import matplotlib.pyplot as plt

    out = Path(runtime.cache_dir) / "examples" / "reactive_control"
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "reactive_control.pdf"

    positions = torch.stack(trajectory_positions, dim=0).cpu().numpy()
    t = [i * config.optimization_dt for i in range(positions.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(mpc.joint_names):
        ax.plot(t, positions[:, j], label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title("Reactive Control Trajectory")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)

    print(f"Trajectory plot saved to: {save_path}")
    return True


def interactive_mpc_example(robot_file="franka.yml", port=8080):
    """Launch an interactive Viser viewer for real-time MPC tracking.

    Sets up a web-based 3D viewer where you can drag control frames to update
    the target pose and watch the robot track the goal with MPC in real time.
    Obstacles in the scene are also interactive.

    Args:
        robot_file: Robot config file name.
        port: Viser server port.
    """
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = ModelPredictiveControlCfg.create(
        robot=robot_file,
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        optimization_dt=0.03,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.03,
    )

    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    mpc = ModelPredictiveControl(config)

    current_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0),
        joint_names=mpc.joint_names,
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)

    mpc.setup(current_state)

    kin_result = mpc.compute_kinematics(current_state)
    target_link_poses = kin_result.tool_poses.to_dict()
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(
            target_link_poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1,
        ),
        run_ik=False,
    )

    print(f"\nInteractive MPC running at http://localhost:{port}")
    print(f"Target links: {mpc.tool_frames}")
    print("Drag the end-effector gizmo to update the goal pose.")
    print("Press Ctrl+C to exit.\n")

    previous_target_poses = None
    pose_changed = False

    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }

        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                mpc.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                pose_changed = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        target_poses = viser_viz.get_control_frame_pose()

        if previous_target_poses is None:
            previous_target_poses = target_poses
        else:
            for frame_name in target_poses.keys():
                if target_poses[frame_name] != previous_target_poses[frame_name]:
                    previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                    pose_changed = True
                    break

        if pose_changed:
            target_link_poses = {
                k.replace("target_", ""): v for k, v in target_poses.items()
            }
            mpc.update_goal_tool_poses(
                GoalToolPose.from_poses(
                    target_link_poses,
                    ordered_tool_frames=mpc.tool_frames,
                    num_goalset=1,
                ),
                run_ik=False,
            )
            pose_changed = False

        mpc_result = mpc.optimize_action_sequence(current_state)

        if mpc_result.action_sequence is not None and mpc_result.action_sequence.position.shape[1] > 0:
            next_position = mpc_result.action_sequence.position[:, -1, :]
            current_state = JointState.from_position(
                next_position.clone(),
                joint_names=mpc.joint_names,
            )
            current_state.velocity = mpc_result.action_sequence.velocity[:, -1, :]
            current_state.acceleration = mpc_result.action_sequence.acceleration[:, -1, :]

            viser_viz.set_joint_state(current_state.squeeze(0))

        time.sleep(0.001)


def test():
    """Run reactive control example as a self-test."""
    assert reactive_control(), "Reactive control failed"
    out = Path(runtime.cache_dir) / "examples" / "reactive_control"
    assert (out / "reactive_control.pdf").exists(), "reactive_control.pdf not created"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reactive Control with cuRobo")
    parser.add_argument("--test", action="store_true", help="Run as self-test with assertions")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of MPC steps (default: 100)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch interactive Viser viewer for real-time MPC",
    )
    parser.add_argument(
        "--robot", type=str, default="franka.yml", help="Robot config file (default: franka.yml)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Viser server port (default: 8080)",
    )
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.visualize:
        interactive_mpc_example(robot_file=args.robot, port=args.port)
        return

    reactive_control(num_steps=args.num_steps)


if __name__ == "__main__":
    main()
