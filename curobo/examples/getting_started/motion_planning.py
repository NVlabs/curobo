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

The pose example uses RRT graph planning. Compare its fitted graph output
against graph-seeded TrajOpt with:

.. code-block:: bash

   # RRT path fitted, retimed, and validated; TrajOpt is never invoked.
   python -m curobo.examples.getting_started.motion_planning --mode pose --no-enable-trajopt

   # RRT path followed by TrajOpt refinement (the default).
   python -m curobo.examples.getting_started.motion_planning --mode pose

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
   -15 cm along the tool Z axis). Contact-link collisions remain enabled during
   this phase.

2. **Grasp**: Plan the final approach from the pre-grasp pose to the grasp
   pose itself. Contact-link collisions are disabled and projected RRT is used
   by default, with constrained TrajOpt as a failure fallback.

3. **Lift**: Plan a retraction from the grasp pose, offset along
   ``grasp_lift_axis`` by ``grasp_lift_offset``. This constrained phase also
   uses projected RRT by default.

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
The Planner panel has pose-planning RRT/TrajOpt controls and separate checkboxes
for opting grasp selection and the constrained grasp/lift phases into TrajOpt.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

import curobo.runtime as runtime
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

_DEFAULT_PLAYBACK_DT = 0.02


def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "motion_planning"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prepare_interpolated_trajectory(
    trajectory: JointState,
    last_tstep: Optional[torch.Tensor] = None,
) -> JointState:
    """Return one valid trajectory with shape ``[1, horizon, dof]``."""
    if trajectory.position.ndim == 4:
        trajectory = trajectory[:, 0]
    elif trajectory.position.ndim == 2:
        trajectory = trajectory.unsqueeze(0)
    elif trajectory.position.ndim != 3:
        raise ValueError(
            "interpolated trajectory must have shape [batch, seed, horizon, dof] "
            "or [batch, horizon, dof]"
        )

    valid_horizon = trajectory.position.shape[-2]
    if last_tstep is not None:
        valid_horizon = int(last_tstep.reshape(-1)[0].item())
        if valid_horizon <= 0:
            valid_horizon = trajectory.position.shape[-2]
    if valid_horizon > trajectory.position.shape[-2]:
        raise ValueError("interpolated trajectory valid horizon exceeds its buffer")
    return trim_joint_state_trajectory(trajectory, 0, valid_horizon)


def _trajectory_dt(trajectory: JointState) -> float:
    """Return the constant timestep stored on an interpolated trajectory."""
    if trajectory.dt is None:
        return _DEFAULT_PLAYBACK_DT
    dt = float(trajectory.dt.reshape(-1)[0].item())
    if dt <= 0.0:
        raise ValueError("interpolated trajectory dt must be positive")
    return dt


def _combine_grasp_phases(
    phases: List[tuple[str, JointState]],
) -> tuple[torch.Tensor, torch.Tensor, List[int], List[str]]:
    """Concatenate phases using phase-local timesteps and no duplicate endpoints."""
    positions = []
    timestamps = []
    phase_boundaries = []
    phase_labels = []
    offset = 0
    elapsed = 0.0

    for phase_index, (label, trajectory) in enumerate(phases):
        phase_position = trajectory.position[0]
        first_index = 0 if phase_index == 0 else 1
        if phase_position.shape[0] <= first_index:
            continue
        phase_position = phase_position[first_index:]
        dt = _trajectory_dt(trajectory)
        phase_time = (
            elapsed
            + torch.arange(
                first_index,
                first_index + phase_position.shape[0],
                device=phase_position.device,
                dtype=phase_position.dtype,
            )
            * dt
        )

        phase_boundaries.append(offset)
        phase_labels.append(label)
        positions.append(phase_position)
        timestamps.append(phase_time)
        offset += phase_position.shape[0]
        elapsed = float(phase_time[-1].item())

    if not positions:
        raise ValueError("at least one non-empty grasp phase is required")
    return (
        torch.cat(positions, dim=0),
        torch.cat(timestamps, dim=0),
        phase_boundaries,
        phase_labels,
    )


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: Optional[float],
    save_path: str,
    title: str = "Joint Trajectory",
    phase_boundaries: Optional[List[int]] = None,
    phase_labels: Optional[List[str]] = None,
    timestamps: Optional[torch.Tensor] = None,
) -> None:
    """Plot joint positions over time and save to *save_path*.

    Args:
        positions: Joint positions tensor of shape ``(timesteps, n_joints)``.
        joint_names: Label for each joint.
        dt: Constant time between waypoints. Required when ``timestamps`` is None.
        save_path: Output file path (e.g. ``"trajectory.pdf"``).
        title: Plot title.
        phase_boundaries: Timestep indices where a new phase starts.
        phase_labels: Label for each phase (length must match *phase_boundaries*).
        timestamps: Optional timestamp for every waypoint. Use this when phases
            have different constant timesteps.
    """
    import matplotlib.pyplot as plt

    if positions.ndim != 2:
        raise ValueError("positions must have shape [horizon, dof]")
    pos_np = positions.cpu().numpy()
    n_steps = pos_np.shape[0]
    if timestamps is not None:
        if timestamps.numel() != n_steps:
            raise ValueError("timestamps must contain one value per waypoint")
        t = timestamps.reshape(-1).cpu().numpy()
    else:
        if dt is None:
            raise ValueError("dt is required when timestamps are not provided")
        t = torch.arange(n_steps, dtype=torch.float32).numpy() * dt

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)

    if phase_boundaries and phase_labels:
        for idx, label in zip(phase_boundaries, phase_labels):
            boundary_time = t[idx]
            ax.axvline(x=boundary_time, color="grey", linestyle="--", linewidth=0.8)
            ax.text(
                boundary_time,
                ax.get_ylim()[1],
                f" {label}",
                fontsize=8,
                va="top",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def pose_planning_example(
    output_dir: Optional[Path] = None,
    robot_file: str = "franka.yml",
    scene_file: str = "collision_test.yml",
    enable_trajopt: bool = True,
    always_run_trajopt: bool = True,
    run_trajopt_on_graph_seed: bool = True,
):
    """Plan a collision-free trajectory to a goal pose.

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/motion_planning/``.
        robot_file: Robot configuration file.
        scene_file: Collision-scene configuration file.
        enable_trajopt: Disable to return the fitted RRT seed without TrajOpt.
        always_run_trajopt: Always refine eligible fitted seeds with TrajOpt.
        run_trajopt_on_graph_seed: Refine RRT seeds even when
            ``always_run_trajopt`` is false.

    Returns:
        True if planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()
    config = MotionPlannerCfg.create(
        robot=robot_file,
        scene_model=scene_file,
        graph_planner_algorithm="rrt",
        enable_trajopt=enable_trajopt,
        always_run_trajopt=always_run_trajopt,
        run_trajopt_on_graph_seed=run_trajopt_on_graph_seed,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    q_goal = q_start.clone()
    q_goal.position[:, 0] += 0.2
    goal_pose = planner.compute_kinematics(q_goal).tool_poses.as_goal()

    result = planner.plan_pose(goal_pose, q_start, max_attempts=1, enable_graph_attempt=0)

    if result is not None:
        success = bool(result.success.any())
        status = "✓ Planning succeeded!" if success else "✗ Dense validation failed"
        print(status)
        print(f"Trajectory source: {result.trajectory_source}")
        print(f"RRT solve time: {planner.last_graph_time * 1000.0:.3f} ms")
        print(f"TrajOpt invoked: {result.trajectory_source == 'trajopt'}")
        print(f"Pipeline solver time: {result.solve_time * 1000.0:.3f} ms")
        interpolated = _prepare_interpolated_trajectory(result.get_interpolated_plan())
        interp_dt = _trajectory_dt(interpolated)
        n_waypoints = interpolated.position.shape[-2]
        path_length = torch.linalg.vector_norm(
            torch.diff(interpolated.position, dim=-2),
            dim=-1,
        ).sum()
        print(f"Trajectory has {n_waypoints} waypoints")
        print(f"Duration: {(n_waypoints - 1) * interp_dt:.2f}s")
        print(f"Joint-space path length: {path_length.item():.3f} rad")

        _plot_trajectory(
            interpolated.position[0],
            planner.joint_names,
            dt=interp_dt,
            save_path=str(output_dir / "motion_plan.pdf"),
            title="Pose-to-Pose Trajectory",
        )
        return success
    else:
        print("✗ Planning failed - try adjusting the goal or obstacles")
        return False


def grasp_planning_example(
    output_dir: Optional[Path] = None,
    robot_file: str = "franka.yml",
    scene_file: str = "collision_test.yml",
    enable_trajopt: bool = True,
    always_run_trajopt: bool = True,
    run_trajopt_on_graph_seed: bool = True,
    use_rrt_for_constrained_phases: bool = False,
    use_double_pass_trajopt_for_constrained_phases: bool = False,
) -> bool:
    """Plan a three-phase grasp motion (approach, grasp, lift).

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/motion_planning/``.
        robot_file: Robot configuration file.
        scene_file: Collision-scene configuration file.
        enable_trajopt: Disable to return fitted graph seeds without TrajOpt.
        always_run_trajopt: Always refine eligible graph seeds with TrajOpt.
        run_trajopt_on_graph_seed: Refine graph seeds when the general fast path
            is enabled.
        use_rrt_for_constrained_phases: Use projected RRT for grasp and lift.
        use_double_pass_trajopt_for_constrained_phases: Add a second TrajOpt pass
            to grasp and lift. Cannot be combined with the RRT option.

    Returns:
        True if grasp planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()
    config = MotionPlannerCfg.create(
        robot=robot_file,
        scene_model=scene_file,
        max_goalset=10,
        graph_planner_algorithm="rrt",
        enable_trajopt=enable_trajopt,
        always_run_trajopt=always_run_trajopt,
        run_trajopt_on_graph_seed=run_trajopt_on_graph_seed,
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
        use_rrt_for_constrained_phases=use_rrt_for_constrained_phases,
        use_double_pass_trajopt_for_constrained_phases=(
            use_double_pass_trajopt_for_constrained_phases
        ),
    )

    if result.success is not None and result.success.any():
        print("✓ Grasp planning succeeded!")

        prepared_phases: List[tuple[str, JointState]] = []
        for label, trajectory, last_tstep in (
            (
                "Approach",
                result.approach_interpolated_trajectory,
                result.approach_interpolated_last_tstep,
            ),
            (
                "Grasp",
                result.grasp_interpolated_trajectory,
                result.grasp_interpolated_last_tstep,
            ),
            (
                "Lift",
                result.lift_interpolated_trajectory,
                result.lift_interpolated_last_tstep,
            ),
        ):
            if trajectory is not None:
                valid_trajectory = _prepare_interpolated_trajectory(
                    trajectory,
                    last_tstep,
                )
                prepared_phases.append((label, valid_trajectory))
                print(
                    f"  {label + ':':<9} {valid_trajectory.position.shape[-2]} waypoints "
                    f"at {_trajectory_dt(valid_trajectory):.4f}s"
                )

        if prepared_phases:
            combined, timestamps, phase_boundaries, phase_labels = _combine_grasp_phases(
                prepared_phases
            )
            _plot_trajectory(
                combined,
                planner.joint_names,
                dt=None,
                save_path=str(output_dir / "grasp_plan.pdf"),
                title="Grasp Trajectory (Approach → Grasp → Lift)",
                phase_boundaries=phase_boundaries,
                phase_labels=phase_labels,
                timestamps=timestamps,
            )
        return True
    else:
        status = getattr(result, "status", "unknown")
        print(f"✗ Grasp planning failed: {status}")
        return False


def interactive_motion_planning(
    robot_file: str = "franka.yml",
    scene_file: str = "collision_test.yml",
    port: int = 8080,
    enable_trajopt: bool = True,
    always_run_trajopt: bool = True,
    run_trajopt_on_graph_seed: bool = True,
    use_rrt_for_constrained_phases: bool = False,
    use_double_pass_trajopt_for_constrained_phases: bool = False,
) -> None:
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

    config = MotionPlannerCfg.create(
        robot=robot_file,
        scene_model=scene_file,
        graph_planner_algorithm="rrt",
        enable_trajopt=enable_trajopt,
        always_run_trajopt=always_run_trajopt,
        run_trajopt_on_graph_seed=run_trajopt_on_graph_seed,
    )
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

        traj = _prepare_interpolated_trajectory(trajectory)
        pos = np.atleast_2d(traj.position[0].cpu().numpy())
        dt_val = _trajectory_dt(traj)
        t = np.arange(pos.shape[0]) * dt_val

        vel = np.atleast_2d(traj.velocity[0].cpu().numpy()) if traj.velocity is not None else None
        acc = (
            np.atleast_2d(traj.acceleration[0].cpu().numpy())
            if traj.acceleration is not None
            else None
        )
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
    with server.gui.add_folder("Planner"):
        rrt_only_btn = server.gui.add_button("Pose: RRT only")
        rrt_trajopt_btn = server.gui.add_button("Pose: RRT + TrajOpt")
        enable_trajopt_cb = server.gui.add_checkbox(
            "Enable TrajOpt",
            initial_value=enable_trajopt,
        )
        always_run_trajopt_cb = server.gui.add_checkbox(
            "Always run TrajOpt",
            initial_value=always_run_trajopt,
        )
        run_trajopt_on_graph_seed_cb = server.gui.add_checkbox(
            "TrajOpt on RRT seed",
            initial_value=run_trajopt_on_graph_seed,
        )
        trajopt_grasp_selection_cb = server.gui.add_checkbox(
            "TrajOpt grasp selection",
            initial_value=False,
        )
        rrt_constrained_phases_cb = server.gui.add_checkbox(
            "RRT grasp/lift phases",
            initial_value=use_rrt_for_constrained_phases,
        )
        double_pass_constrained_phases_cb = server.gui.add_checkbox(
            "Double-pass TrajOpt grasp/lift",
            initial_value=use_double_pass_trajopt_for_constrained_phases,
        )
        planner_mode_text = server.gui.add_text(
            "Active mode",
            initial_value="",
            disabled=True,
        )

    def _sync_planner_policy():
        planner.config.enable_trajopt = bool(enable_trajopt_cb.value)
        planner.config.always_run_trajopt = bool(always_run_trajopt_cb.value)
        planner.config.run_trajopt_on_graph_seed = bool(run_trajopt_on_graph_seed_cb.value)
        if not planner.config.enable_trajopt:
            mode = "RRT only"
        elif planner.config.always_run_trajopt:
            mode = "RRT + TrajOpt"
        elif planner.config.run_trajopt_on_graph_seed:
            mode = "RRT + TrajOpt; linear fast path"
        else:
            mode = "Validated seed; TrajOpt fallback"
        grasp_selection = (
            "TrajOpt selection" if trajopt_grasp_selection_cb.value else "IK selection"
        )
        if rrt_constrained_phases_cb.value:
            constrained_phases = "projected RRT phases"
        elif double_pass_constrained_phases_cb.value:
            constrained_phases = "double-pass TrajOpt phases"
        else:
            constrained_phases = "single-pass TrajOpt phases"
        planner_mode_text.value = f"Pose: {mode} | Grasp: {grasp_selection} + {constrained_phases}"

    @rrt_only_btn.on_click
    def _use_rrt_only(_):
        enable_trajopt_cb.value = False
        always_run_trajopt_cb.value = False
        run_trajopt_on_graph_seed_cb.value = False
        _sync_planner_policy()

    @rrt_trajopt_btn.on_click
    def _use_rrt_with_trajopt(_):
        enable_trajopt_cb.value = True
        always_run_trajopt_cb.value = True
        run_trajopt_on_graph_seed_cb.value = True
        _sync_planner_policy()

    @enable_trajopt_cb.on_update
    def _update_enable_trajopt(_):
        _sync_planner_policy()

    @always_run_trajopt_cb.on_update
    def _update_always_run_trajopt(_):
        _sync_planner_policy()

    @run_trajopt_on_graph_seed_cb.on_update
    def _update_run_trajopt_on_graph_seed(_):
        _sync_planner_policy()

    @trajopt_grasp_selection_cb.on_update
    def _update_trajopt_grasp_selection(_):
        _sync_planner_policy()

    @rrt_constrained_phases_cb.on_update
    def _update_rrt_constrained_phases(_):
        if rrt_constrained_phases_cb.value:
            double_pass_constrained_phases_cb.value = False
        _sync_planner_policy()

    @double_pass_constrained_phases_cb.on_update
    def _update_double_pass_constrained_phases(_):
        if double_pass_constrained_phases_cb.value:
            rrt_constrained_phases_cb.value = False
        _sync_planner_policy()

    _sync_planner_policy()

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
        traj = _prepare_interpolated_trajectory(trajectory)
        playback_dt = _trajectory_dt(traj)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(playback_dt)
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
                enable_graph_attempt=0,
            )
            if result is not None and result.success.any():
                interp = result.get_interpolated_plan()
                traj_plot.image = _create_trajectory_image(
                    interp,
                    planner.joint_names,
                    title=(f"Pose Plan ({result.trajectory_source})  |  {result.total_time:.3f}s"),
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
                use_trajopt_for_goalset_selection=bool(trajopt_grasp_selection_cb.value),
                use_rrt_for_constrained_phases=bool(rrt_constrained_phases_cb.value),
                use_double_pass_trajopt_for_constrained_phases=bool(
                    double_pass_constrained_phases_cb.value
                ),
            )
            if result is not None and result.success is not None and result.success.any():
                for label, trajectory, last_tstep in (
                    (
                        "Approach",
                        result.approach_interpolated_trajectory,
                        result.approach_interpolated_last_tstep,
                    ),
                    (
                        "Grasp",
                        result.grasp_interpolated_trajectory,
                        result.grasp_interpolated_last_tstep,
                    ),
                    (
                        "Lift",
                        result.lift_interpolated_trajectory,
                        result.lift_interpolated_last_tstep,
                    ),
                ):
                    if trajectory is None:
                        continue
                    valid_trajectory = _prepare_interpolated_trajectory(
                        trajectory,
                        last_tstep,
                    )
                    traj_plot.image = _create_trajectory_image(
                        valid_trajectory,
                        planner.joint_names,
                        title=label,
                    )
                    execute_trajectory(valid_trajectory)
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
    parser.add_argument(
        "--enable-trajopt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable TrajOpt (default). Use --no-enable-trajopt to return the fitted, "
            "retimed RRT path without optimization or fallback."
        ),
    )
    parser.add_argument(
        "--always-run-trajopt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always refine eligible seeds with TrajOpt (default: enabled).",
    )
    parser.add_argument(
        "--run-trajopt-on-graph-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Refine RRT seeds when --no-always-run-trajopt is used (default: enabled)."),
    )
    grasp_phase_group = parser.add_mutually_exclusive_group()
    grasp_phase_group.add_argument(
        "--grasp-rrt-constrained-phases",
        action="store_true",
        help="Use projected RRT for grasp and lift instead of single-pass TrajOpt.",
    )
    grasp_phase_group.add_argument(
        "--grasp-double-pass-trajopt",
        action="store_true",
        help="Add one TrajOpt refinement pass to grasp and lift.",
    )
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.visualize:
        interactive_motion_planning(
            robot_file=args.robot,
            scene_file=args.scene,
            port=args.port,
            enable_trajopt=args.enable_trajopt,
            always_run_trajopt=args.always_run_trajopt,
            run_trajopt_on_graph_seed=args.run_trajopt_on_graph_seed,
            use_rrt_for_constrained_phases=args.grasp_rrt_constrained_phases,
            use_double_pass_trajopt_for_constrained_phases=(args.grasp_double_pass_trajopt),
        )
        return

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("pose", "all"):
        print("=== Pose-to-Pose Motion Planning ===")
        pose_planning_example(
            output_dir=output_dir,
            robot_file=args.robot,
            scene_file=args.scene,
            enable_trajopt=args.enable_trajopt,
            always_run_trajopt=args.always_run_trajopt,
            run_trajopt_on_graph_seed=args.run_trajopt_on_graph_seed,
        )
        print()

    if args.mode in ("grasp", "all"):
        print("=== Grasp Planning ===")
        grasp_planning_example(
            output_dir=output_dir,
            robot_file=args.robot,
            scene_file=args.scene,
            enable_trajopt=args.enable_trajopt,
            always_run_trajopt=args.always_run_trajopt,
            run_trajopt_on_graph_seed=args.run_trajopt_on_graph_seed,
            use_rrt_for_constrained_phases=args.grasp_rrt_constrained_phases,
            use_double_pass_trajopt_for_constrained_phases=(args.grasp_double_pass_trajopt),
        )


if __name__ == "__main__":
    main()
