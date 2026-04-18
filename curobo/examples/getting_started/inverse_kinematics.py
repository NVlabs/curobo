# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Solve inverse kinematics on the GPU with multi-seed optimization.

.. raw:: html

    <div style="display:flex;gap:8px;">
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:50%"><source src="../videos/get_started_ik.webm" type="video/webm"></video>
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:50%"><source src="../videos/get_started_differential_ik.webm" type="video/webm"></video>
    </div>

Inverse kinematics (IK) finds the joint angles that place a robot's
end-effector at a desired 6-DOF pose. cuRobo formulates IK as a nonlinear
optimization problem and solves many random seeds in parallel on the GPU,
achieving high success rates and sub-millimeter accuracy in a single call.
The same solver supports batched targets, scene-aware collision avoidance,
self-collision checking, and runtime world updates.

By the end of this tutorial you will have:

- Solved IK for a single target pose and verified the result with FK
- Solved IK for 100 target poses in a single batched call
- Added obstacles and self-collision checking for collision-free IK
- Updated the obstacle scene at runtime without recreating the solver
- Launched an interactive Viser viewer for real-time drag-and-solve IK
- Visualized a 3D reachability map as a colored point cloud

Step 1: Run the tutorial
--------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.inverse_kinematics

This runs single, batched, and collision-free IK in sequence. Three interactive
visualization modes are also available:

.. code-block:: bash

   python -m curobo.examples.getting_started.inverse_kinematics --visualize

Full IK (LM + LBFGS): drag the end-effector gizmo to solve IK in real time;
drag obstacles to see collision avoidance adapt.

.. code-block:: bash

   python -m curobo.examples.getting_started.inverse_kinematics --differential

Differential IK: smooth reactive control that tracks the gizmo continuously,
suitable for teleoperation-style interaction.

.. code-block:: bash

   python -m curobo.examples.getting_started.inverse_kinematics --reachability

To try reachability with a dual-arm robot:

.. code-block:: bash

   python -m curobo.examples.getting_started.inverse_kinematics --reachability --robot dual_ur10e.yml

Reachability map: a 100x100 grid of IK queries is solved on a 2D slice plane
and displayed as an image (green = reachable, red = unreachable). Drag the
gizmo to move or rotate the plane; use the slider to resize it.

All three modes open at http://localhost:8080 in your browser.

Step 2: Check the output
--------------------------

When the tutorial finishes successfully you will see::

    === Single IK ===
    IK solved!
      Joint angles: tensor(...)
      Position error: 0.XXX mm

    === Batched IK (100 poses) ===
    Solved 95/100 poses (95% success)
    Mean position error: 0.XXX mm
    Max position error:  0.XXX mm

    === Collision-Free IK ===
    Single collision-free IK solved!
      Position error: 0.XXX mm

    After adding obstacle -- still solved!
      Position error: 0.XXX mm

    Batched collision-free IK: 48/50 solved

Step 3: Understand the pipeline
---------------------------------

The tutorial demonstrates four levels of IK solving:

1. **Single IK**: Create an ``InverseKinematics`` with ``num_seeds=32``, define a target
   ``Pose`` (position + quaternion in wxyz order), and call ``solve_pose``.
   The solver runs 32 parallel optimization seeds and returns the best
   solution. Verify by running FK on the solution and checking the position
   error.

2. **Batched IK**: Pass a ``Pose`` with shape ``(B, 3)`` positions and
   ``(B, 4)`` quaternions to ``solve_pose``. All B targets are solved in
   parallel on the GPU, useful for reachability analysis or grasp pose
   evaluation.

3. **Collision-free IK**: Pass ``scene_model`` and ``self_collision_check=True``
   when creating the solver config. The optimizer adds collision cost terms
   that keep every robot link sphere clear of obstacles and other links.

4. **Runtime world updates**: Call ``update_world`` with a new ``Scene``
   to add, remove, or move obstacles without recreating the solver.

Step 4: Interactive IK with Viser (advanced)
----------------------------------------------

All three interactive modes launch a `Viser <https://viser.studio>`_ web viewer
that renders the robot, its collision spheres, and scene obstacles.

- **Full IK** (``--visualize``): Uses the complete LM + LBFGS optimization
  pipeline. A 6-DOF gizmo lets you drag the end-effector target; the solver
  re-runs on every gizmo update. Obstacles are interactive -- drag them to new
  positions and the solver automatically avoids them.

- **Differential IK** (``--differential``): Smooth, continuous tracking where
  each solve warm-starts from the previous solution, producing small
  joint-space steps ideal for reactive control and teleoperation.

- **Reachability map** (``--reachability``): Solves a dense grid of IK queries
  on a 2D slice plane and visualizes success as a colored point cloud
  (green = reachable, red = unreachable). Drag the gizmo to reposition or
  rotate the plane; use the slider to resize it.

Step 5: Reachability analysis with batched IK
-----------------------------------------------

.. raw:: html

    <video autoplay="True" loop="True" muted="True" preload="auto" width="100%"><source src="../videos/get_started_reachability.webm" type="video/webm"></video>

A common use of batched IK is **reachability analysis**: given a robot
configuration and scene, which end-effector poses can the robot actually reach?
cuRobo's ``solve_pose`` solves thousands of IK queries in a single GPU call,
making it practical to sweep a dense grid of candidate poses and classify each
as reachable or unreachable.

The ``--reachability`` mode demonstrates this by sampling a 100x100 grid of
positions on a 2D slice plane, solving IK for every grid point, and coloring
the result (green = solved, red = failed). Because the solver is
collision-aware, the reachability map automatically reflects obstacles and
self-collision constraints -- adding an obstacle carves out an unreachable
region in real time.

This workflow is useful for:

- **Workcell design**: verify that a robot can reach all required task poses
  before committing to a layout.
- **Grasp filtering**: pre-screen candidate grasp poses and discard
  unreachable ones before running the full motion planner.
- **Multi-arm coverage**: with robots like ``dual_ur10e.yml``, visualize the
  overlapping workspace of two arms to plan handoff zones.
"""

import argparse
import sys
import time
import copy

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Cuboid, Scene
from curobo.types import ContentPath, GoalToolPose, Pose
from curobo.viewer import ViserVisualizer


def single_ik_example():
    """Solve IK for a single target pose.

    Returns:
        True if IK succeeded.
    """
    config = InverseKinematicsCfg.create(
        robot="franka.yml",
        num_seeds=32,
    )
    ik = InverseKinematics(config)
    target_link = ik.tool_frames[0]

    goal_pose = Pose(
        position=torch.tensor([[0.4, 0.0, 0.4]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
    )

    result = ik.solve_pose(GoalToolPose.from_poses({target_link: goal_pose}, num_goalset=1))

    if result.success.item():
        print("IK solved!")
        print(f"  Joint angles: {result.js_solution.position}")
        print(f"  Position error: {result.position_error.item() * 1000:.3f} mm")
        return True
    else:
        print("IK failed -- target may be unreachable")
        return False


def batched_ik_example():
    """Solve IK for a batch of target poses.

    Returns:
        True if at least one pose was solved.
    """
    n_poses = 100
    config = InverseKinematicsCfg.create(
        robot="franka.yml",
        num_seeds=32,
        max_batch_size=n_poses,
    )
    ik = InverseKinematics(config)
    target_link = ik.tool_frames[0]
    positions = torch.zeros(n_poses, 3, device="cuda", dtype=torch.float32)
    positions[:, 0] = torch.linspace(0.2, 0.8, n_poses)
    positions[:, 1] = 0.0
    positions[:, 2] = 0.4

    quaternions = torch.zeros(n_poses, 4, device="cuda", dtype=torch.float32)
    quaternions[:, 0] = 1.0

    goal_poses = Pose(position=positions, quaternion=quaternions)

    result = ik.solve_pose(GoalToolPose.from_poses({target_link: goal_poses}, num_goalset=1))

    n_success = result.success.sum().item()
    print(f"Solved {n_success}/{n_poses} poses ({100 * n_success / n_poses:.0f}% success)")

    successful = result.success.squeeze()
    if n_success > 0:
        pos_errors = result.position_error[successful]
        print(f"Mean position error: {pos_errors.mean().item() * 1000:.3f} mm")
        print(f"Max position error:  {pos_errors.max().item() * 1000:.3f} mm")
    return n_success > 0


def collision_free_ik_example():
    """Solve collision-free IK with obstacles.

    Returns:
        True if single collision-free IK succeeded.
    """
    goal_pose = Pose(
        position=torch.tensor([[0.4, 0.0, 0.4]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
    )

    # Solve single collision-free IK with table scene
    config = InverseKinematicsCfg.create(
        robot="franka.yml",
        scene_model="collision_table.yml",
        num_seeds=32,
        self_collision_check=True,
    )
    ik = InverseKinematics(config)
    target_link = ik.tool_frames[0]

    result = ik.solve_pose(GoalToolPose.from_poses({target_link: goal_pose}, num_goalset=1))

    if result.success.item():
        print("Single collision-free IK solved!")
        print(f"  Position error: {result.position_error.item() * 1000:.3f} mm")
    else:
        print("Single collision-free IK failed")

    # Solve with an additional obstacle (new solver instance with collision cache)
    new_obstacle = Cuboid(
        name="box_1",
        pose=[0.5, 0.0, 0.3, 1, 0, 0, 0],
        dims=[0.1, 0.3, 0.2],
    )
    config_with_cache = InverseKinematicsCfg.create(
        robot="franka.yml",
        scene_model="collision_table.yml",
        num_seeds=32,
        self_collision_check=True,
        collision_cache={"cuboid": 10},
    )
    ik2 = InverseKinematics(config_with_cache)
    ik2.update_world(Scene(cuboid=[new_obstacle]))

    result = ik2.solve_pose(GoalToolPose.from_poses({target_link: goal_pose}, num_goalset=1))
    if result.success.item():
        print("\nAfter adding obstacle -- still solved!")
        print(f"  Position error: {result.position_error.item() * 1000:.3f} mm")

    # Batched collision-free IK (fresh solver instance)
    n_poses = 50
    positions = torch.zeros(n_poses, 3, device="cuda", dtype=torch.float32)
    positions[:, 0] = torch.linspace(0.3, 0.7, n_poses)
    positions[:, 1] = 0.0
    positions[:, 2] = 0.4

    quaternions = torch.zeros(n_poses, 4, device="cuda", dtype=torch.float32)
    quaternions[:, 0] = 1.0

    goal_poses = Pose(position=positions, quaternion=quaternions)

    config_batch = InverseKinematicsCfg.create(
        robot="franka.yml",
        scene_model="collision_table.yml",
        num_seeds=32,
        self_collision_check=True,
        max_batch_size=n_poses,
    )
    ik3 = InverseKinematics(config_batch)
    result = ik3.solve_pose(GoalToolPose.from_poses({target_link: goal_poses}, num_goalset=1))

    n_success = result.success.sum().item()
    print(f"\nBatched collision-free IK: {n_success}/{n_poses} solved")
    return True


def interactive_ik_example(robot_file="franka.yml", port=8080):
    """Launch an interactive Viser viewer for real-time IK solving."""

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
    )
    config.scene_collision_cfg.use_warp_collision = True
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    ik_solver = InverseKinematics(config)
    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False
    #ik_solver.config.use_lm_seed = False

    goal_state = ik_solver.default_joint_state.clone()
    kin_state = ik_solver.compute_kinematics(goal_state).clone()
    goal_tool_poses = kin_state.tool_poses.to_dict()

    current_state = ik_solver.get_active_js(ik_solver.default_joint_state.clone())
    current_state = current_state.unsqueeze(0)

    ik_solver.solve_pose(
        goal_tool_poses=GoalToolPose.from_poses(
            goal_tool_poses,
            ordered_tool_frames=ik_solver.tool_frames,
            num_goalset=1,
        ),
        current_state=current_state.clone(),
        return_seeds=1,
    )

    print(f"\nInteractive IK running at http://localhost:{port}")
    print("Drag the end-effector gizmo to solve IK in real time.")
    print("Press Ctrl+C to exit.\n")

    previous_target_poses = None
    pose_changed = False
    current_state = current_state.clone()

    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }

        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                ik_solver.scene_collision_checker.update_obstacle_pose(
                    k, obstacle_poses[k]
                )
                pose_changed = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = copy.deepcopy(target_poses)
        else:
            for frame_name in target_poses.keys():
                if target_poses[frame_name] != previous_target_poses[frame_name]:
                    previous_target_poses = {
                        k: v.clone() for k, v in target_poses.items()
                    }
                    pose_changed = True
                    break

        if pose_changed:
            active_js = ik_solver.get_active_js(current_state)
            target_link_poses = {
                k.replace("target_", ""): v for k, v in target_poses.items()
            }
            result = ik_solver.solve_pose(
                goal_tool_poses=GoalToolPose.from_poses(
                    target_link_poses,
                    ordered_tool_frames=ik_solver.tool_frames,
                    num_goalset=1,
                ),
                current_state=active_js.squeeze(1).clone(),
                return_seeds=1,
                run_optimizer=True,
            )
            if result.success:
                pose_changed = False

                current_state = result.js_solution.clone()
                viser_viz.set_joint_state(
                    result.js_solution.squeeze(0).squeeze(0)
                )
        time.sleep(0.001)


def differential_ik_example(robot_file="franka.yml", port=8080):
    """Launch an interactive Viser viewer using differential (LM-based) IK.

    Unlike the full IK pipeline that uses LBFGS optimization, differential IK
    uses only the Levenberg-Marquardt seed solver with velocity and acceleration
    regularization. This produces smooth, reactive motions that stay in the
    current homotopy class, which is particularly important for 6-DOF robots where
    the LBFGS can jump between isolated IK solution branches.

    The solver minimizes a weighted combination of:
    - Pose error (position + orientation tracking)
    - Velocity regularization (damping, prevents overshoot)
    - Acceleration regularization (smoothing, prevents jerky transitions)
    """

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
        acceleration_regularization_weight=100.0,
        velocity_regularization_weight=1.0,
        seed_velocity_weight=1.0,
        seed_acceleration_weight=1.0,
        optimization_dt=0.1,
        success_requires_convergence=False,
        seed_position_weight=1.0,
        seed_orientation_weight=0.1,

    )
    config.exit_early = False
    config.scene_collision_cfg.use_warp_collision = True
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    ik_solver = InverseKinematics(config)

    goal_state = ik_solver.default_joint_state.clone()
    kin_state = ik_solver.compute_kinematics(goal_state).clone()
    goal_tool_poses = kin_state.tool_poses.to_dict()

    current_state = ik_solver.default_joint_state.clone()
    current_state = current_state.unsqueeze(0)

    ik_solver.solve_pose(
        goal_tool_poses=GoalToolPose.from_poses(
            goal_tool_poses,
            ordered_tool_frames=ik_solver.tool_frames,
            num_goalset=1,
        ),
        current_state=current_state,
        return_seeds=1,
        run_optimizer=False,
    )

    print(f"\nDifferential IK running at http://localhost:{port}")
    print("Drag the end-effector gizmo to solve IK in real time.")
    print("Uses LM solver with velocity/acceleration regularization.")
    print("Press Ctrl+C to exit.\n")

    previous_target_poses = None
    current_state = current_state.clone()

    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }

        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                ik_solver.scene_collision_checker.update_obstacle_pose(
                    k, obstacle_poses[k]
                )
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = target_poses
        else:
            for frame_name in target_poses.keys():
                if target_poses[frame_name] != previous_target_poses[frame_name]:
                    previous_target_poses = {
                        k: v.clone() for k, v in target_poses.items()
                    }
                    pose_changed = True
                    break

        active_js = ik_solver.get_active_js(current_state)
        target_link_poses = {
            k.replace("target_", ""): v for k, v in target_poses.items()
        }
        result = ik_solver.solve_pose(
            goal_tool_poses=GoalToolPose.from_poses(
                target_link_poses,
                ordered_tool_frames=ik_solver.tool_frames,
                num_goalset=1,
            ),
            current_state=active_js.squeeze(1),
            return_seeds=1,
            run_optimizer=False,
        )
        if result.success.any():
            current_state = result.js_solution.clone()
            viser_viz.set_joint_state(
                result.js_solution.squeeze(0).squeeze(0)
            )
        time.sleep(0.001)


def reachability_example(robot_file="franka.yml", port=8080):
    """Launch an interactive reachability slice viewer in Viser.

    Displays a 2D reachability heatmap on a draggable slice plane, similar to
    the ESDF slice in the volumetric-mapping tutorial.  A transform-controls
    gizmo defines the plane; the solver evaluates ~10 000 IK queries on the
    plane's local XY grid (all at the default tool orientation) and renders the
    result as an image: green = reachable, red = unreachable.  A yellow
    wireframe square outlines the query region.  Drag the gizmo or adjust the
    extent slider to explore different workspace cross-sections.
    """
    import viser.transforms as vtf

    BATCH_TARGET = 500
    n_per_axis = int(BATCH_TARGET ** 0.5)
    actual_batch = n_per_axis * n_per_axis

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )
    server = viser_viz._server

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        #num_seeds=12,
        self_collision_check=True,
        scene_model="collision_test.yml",
        #seed_solver_num_seeds=12,
    )
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)

    ik = InverseKinematics(config)
    ik.exit_early = False
    all_target_links = ik.tool_frames
    primary_link = all_target_links[0]

    kin_state = ik.compute_kinematics(ik.default_joint_state)
    tool_pose = kin_state.tool_poses[primary_link]
    center = tool_pose.position.squeeze().cpu().numpy()
    orientation = tool_pose.quaternion.squeeze()

    slice_gizmo = server.scene.add_transform_controls(
        "/reachability_gizmo",
        scale=0.15,
        position=tuple(center.tolist()),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    tool_frame_gizmos = {}
    for link_name in all_target_links:
        link_pose = kin_state.tool_poses[link_name]
        link_pos = link_pose.position.squeeze().cpu().numpy()
        link_quat = link_pose.quaternion.squeeze().cpu().numpy()
        tool_frame_gizmos[link_name] = server.scene.add_transform_controls(
            f"/tool_frame_{link_name}",
            scale=0.10,
            position=tuple(link_pos.tolist()),
            wxyz=tuple(link_quat.tolist()),
        )

    with server.gui.add_folder("Reachability"):
        grid_extent_slider = server.gui.add_slider(
            "Grid Extent (m)",
            min=0.1,
            max=2.0,
            step=0.05,
            initial_value=1.0,
        )

    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }
    prev_gizmo_pos = np.array(slice_gizmo.position, dtype=np.float32)
    prev_gizmo_wxyz = np.array(slice_gizmo.wxyz, dtype=np.float32)
    prev_tool_poses = {
        name: (
            np.array(g.position, dtype=np.float32),
            np.array(g.wxyz, dtype=np.float32),
        )
        for name, g in tool_frame_gizmos.items()
    }
    prev_extent = grid_extent_slider.value

    print(f"\nReachability viewer running at http://localhost:{port}")
    print(f"Slice: {n_per_axis}x{n_per_axis} = {actual_batch} IK queries per update")
    print("Drag the gizmo to move/rotate the slice plane.")
    print("Adjust the 'Grid Extent' slider to resize it.")
    print("Press Ctrl+C to exit.\n")

    needs_update = True
    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }
        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                ik.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                needs_update = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        cur_pos = np.array(slice_gizmo.position, dtype=np.float32)
        cur_wxyz = np.array(slice_gizmo.wxyz, dtype=np.float32)
        cur_tool_poses = {
            name: (
                np.array(g.position, dtype=np.float32),
                np.array(g.wxyz, dtype=np.float32),
            )
            for name, g in tool_frame_gizmos.items()
        }
        cur_extent = grid_extent_slider.value
        if (
            not np.allclose(cur_pos, prev_gizmo_pos)
            or not np.allclose(cur_wxyz, prev_gizmo_wxyz)
            or cur_extent != prev_extent
            or any(
                not np.allclose(cur_tool_poses[n][0], prev_tool_poses[n][0])
                or not np.allclose(cur_tool_poses[n][1], prev_tool_poses[n][1])
                for n in all_target_links
            )
        ):
            needs_update = True
            prev_gizmo_pos = cur_pos
            prev_gizmo_wxyz = cur_wxyz
            prev_tool_poses = cur_tool_poses
            prev_extent = cur_extent

        if not needs_update:
            time.sleep(0.02)
            continue
        needs_update = False

        extent = cur_extent
        half = extent / 2.0
        gizmo_pos = cur_pos
        rot = vtf.SO3(cur_wxyz).as_matrix().astype(np.float32)
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = rot
        pose_matrix[:3, 3] = gizmo_pos
        pose_t = torch.tensor(pose_matrix, device="cuda", dtype=torch.float32)

        lin = torch.linspace(
            -half, half, n_per_axis, device="cuda", dtype=torch.float32
        )
        uu, vv = torch.meshgrid(lin, lin, indexing="xy")
        local_pts = torch.stack(
            [
                uu.reshape(-1),
                vv.reshape(-1),
                torch.zeros(actual_batch, device="cuda", dtype=torch.float32),
                torch.ones(actual_batch, device="cuda", dtype=torch.float32),
            ],
            dim=-1,
        )
        grid_world_pts = (pose_t @ local_pts.T).T[:, :3]

        total_batch = actual_batch + 1

        primary_pos_np, primary_quat_np = cur_tool_poses[primary_link]
        primary_quat = torch.tensor(
            primary_quat_np, device="cuda", dtype=torch.float32
        )
        primary_pos = torch.tensor(
            primary_pos_np, device="cuda", dtype=torch.float32
        ).unsqueeze(0)

        all_positions = torch.cat([grid_world_pts, primary_pos], dim=0)
        all_quaternions = primary_quat.unsqueeze(0).expand(
            total_batch, -1
        ).contiguous()

        goal_dict = {
            primary_link: Pose(
                position=all_positions, quaternion=all_quaternions
            ).clone(),
        }
        for link_name in all_target_links[1:]:
            lp, lq = cur_tool_poses[link_name]
            link_pos = torch.tensor(
                lp, device="cuda", dtype=torch.float32
            ).unsqueeze(0).expand(total_batch, -1).contiguous()
            link_quat = torch.tensor(
                lq, device="cuda", dtype=torch.float32
            ).unsqueeze(0).expand(total_batch, -1).contiguous()
            goal_dict[link_name] = Pose(
                position=link_pos, quaternion=link_quat,
            )

        result = ik.solve_pose(
            GoalToolPose.from_poses(
                goal_dict,
                ordered_tool_frames=all_target_links,
                num_goalset=1,
            ),
        )

        all_success = result.success.squeeze().cpu().numpy().astype(bool)
        grid_success = all_success[:actual_batch].reshape(n_per_axis, n_per_axis)
        gizmo_success = all_success[actual_batch]

        img = np.zeros((n_per_axis, n_per_axis, 3), dtype=np.uint8)
        img[grid_success] = [0, 200, 0]
        img[~grid_success] = [200, 0, 0]

        if gizmo_success:
            gizmo_js = result.js_solution[actual_batch]
            viser_viz.set_joint_state(gizmo_js.squeeze(0))

        server.scene.add_image(
            name="/reachability_gizmo/slice_image",
            image=img,
            render_width=extent,
            render_height=extent,
        )

        corners_local = np.array(
            [[-half, -half, 0], [half, -half, 0],
             [half, half, 0], [-half, half, 0]],
            dtype=np.float32,
        )
        corners_world = (rot @ corners_local.T).T + gizmo_pos
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lines = np.array(
            [[corners_world[i], corners_world[j]] for i, j in edges],
            dtype=np.float32,
        )
        yellow = np.array([255, 255, 0], dtype=np.uint8)
        server.scene.add_line_segments(
            "/reachability_bounds",
            points=lines,
            colors=yellow,
            line_width=3.0,
        )

        n_success = int(grid_success.sum())
        print(
            f"Reachability: {n_success}/{actual_batch} "
            f"({100 * n_success / actual_batch:.0f}%) | "
            f"Grid: {n_per_axis}x{n_per_axis} = {actual_batch} | "
            f"Extent: {extent:.2f} m"
        )


def test():
    """Run IK examples as a self-test."""
    assert single_ik_example(), "Single IK failed"
    assert batched_ik_example(), "Batched IK failed"
    assert collision_free_ik_example(), "Collision-free IK failed"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inverse Kinematics with cuRobo",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "collision_free", "all"],
        default="all",
        help="Which IK example to run (default: all)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run as self-test with assertions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch interactive Viser viewer with full IK (LM + LBFGS)",
    )
    parser.add_argument(
        "--differential",
        action="store_true",
        help="Launch interactive Viser viewer with differential IK (LM only, smooth reactive control)",
    )
    parser.add_argument(
        "--reachability",
        action="store_true",
        help="Launch interactive reachability map (3D grid colored by IK success)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka.yml",
        help="Robot config file (default: franka.yml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.reachability:
        reachability_example(robot_file=args.robot, port=args.port)
        return

    if args.differential:
        differential_ik_example(robot_file=args.robot, port=args.port)
        return

    if args.visualize:
        interactive_ik_example(robot_file=args.robot, port=args.port)
        return

    if args.mode in ("single", "all"):
        print("=== Single IK ===")
        single_ik_example()
        print()

    if args.mode in ("batch", "all"):
        print(f"=== Batched IK (100 poses) ===")
        batched_ik_example()
        print()

    if args.mode in ("collision_free", "all"):
        print("=== Collision-Free IK ===")
        collision_free_ik_example()


if __name__ == "__main__":
    main()
