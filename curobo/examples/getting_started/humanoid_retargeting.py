# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA adapter for retargeting BVH motion capture to Unitree G1 using cuRobo IK and MPC.

.. raw:: html

    <div style="display:flex;gap:8px;">
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:33%"><source src="../videos/retargeting_ik.webm" type="video/webm"></video>
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:33%"><source src="../videos/retargeting_cfree_ik.webm" type="video/webm"></video>
    <video autoplay="True" loop="True" muted="True" preload="auto" style="width:33%"><source src="../videos/retargeting_mpc.webm" type="video/webm"></video>
    </div>

This guide walks through setting up cuRobo for retargeting human motion capture data
to a humanoid robot. We use the Unitree G1 29-DOF robot as an example, but the same
approach applies to any humanoid with a URDF.

This script is a SOMA adapter that bridges ``soma_retargeter`` I/O with the
cuRobo :class:`~curobo.motion_retargeter.MotionRetargeter` API. All solver
logic (IK, MPC, warm-starting) lives in ``MotionRetargeter``; this file
handles only BVH loading, effector conversion, CSV output, and visualization.

What is Motion Retargeting?
---------------------------

Motion retargeting transfers a motion from one character (the *source*) to another
(the *target*) that may have a different body shape, limb proportions, or degrees of
freedom. In humanoid robotics, the source is typically a human performer captured via
optical motion capture, and the target is a robot. Copying joint angles directly does
not work as the source and target have different kinematic structures, limb lengths,
joint limits, and degrees of freedom.

Retargeting breaks down into three sub-problems:

1. **Link mapping**: which human joint corresponds to which robot link?
2. **Proportion rescaling**: rescale and offset the human mocap poses to match the
   robot's limb lengths, body proportions, and joint frame conventions.
3. **Solving for joint angles**: find robot joint angles at every frame that best
   reproduce the rescaled target poses while respecting kinematic constraints.

`SOMA Retargeter <https://github.com/NVIDIA/soma-retargeter>`_ handles all three:
it loads a BVH file, applies a per-robot link mapping and per-joint scale
factors/pose offsets, and includes its own IK solver for step 3. This guide shows
how to replace the IK solver with cuRobo, leveraging GPU-accelerated collision-free
trajectory optimization.

How cuRobo Retargets Motion
---------------------------

cuRobo sets up an optimization problem to find robot joint angles that best reproduce
the target poses across all *N* tracked links, subject to joint limits and (optionally)
self-collision avoidance. Each tracked robot link is a ``tool_frame`` that receives an
independent pose target. Links can be weighted differently: feet and pelvis get high
position weight to maintain balance, while shoulder rotations get low weight since the
elbow and wrist targets already constrain the arm.

**Two-phase solving.** Retargeting a motion clip proceeds frame by frame using two
phases:

1. **Global IK (frame 0):** The first frame has no prior solution, so the solver
   runs a broad search using many random seeds (default 64) with no velocity limit.
   This explores the full configuration space to find a good initial pose.

2. **Local solver (frames 1 … N):** Each subsequent frame uses the previous frame's
   solution as a warm start. This can be either:

   - **Warm-started IK** (``use_mpc=False``): a single-seed, velocity-limited IK
     solve that tracks the motion frame by frame.
   - **MPC** (``use_mpc=True``): model predictive control that optimizes a
     trajectory over a planning horizon, producing smoother results with
     acceleration and jerk costs.

.. graphviz::

   digraph {
      rankdir=LR;
      edge [color="#2B4162"; fontsize=10];
      node [shape="box", style="rounded, filled", fontsize=12, color="#cccccc"];

      target [label="Per-Frame\\nTool Pose Targets", color="#708090", fontcolor="white"];
      global [label="Global IK\\n64 seeds, no v-limit", color="#76b900", fontcolor="white"];
      local  [label="Local IK  or  MPC\\n1 seed, v-limited", color="#558c8c", fontcolor="white"];
      output [label="Joint Trajectory\\n(num_envs, num_frames, num_dof)", color="#708090", fontcolor="white"];

      target -> global [label="frame 0"];
      global -> local  [label="warm start"];
      local  -> local  [label="warm start\\nframes 1…N"];
      local  -> output;
   }

Warm-starting is what connects isolated per-frame IK solutions into a temporally
coherent motion: the solver starts from a configuration that is already close to the
answer, and the velocity limit prevents it from jumping to a distant solution even if
one has lower cost.

The :class:`~curobo.motion_retargeter.MotionRetargeter` API encapsulates this
two-phase loop. You provide target poses; it manages solver construction,
warm-starting, and mode switching internally.

Step 1: Floating Base via ``extra_links``
------------------------------------------

Humanoid retargeting requires a **floating base**: the robot's pelvis moves freely in
space. cuRobo supports this via ``extra_links`` with ``child_link_name`` in the YAML
config -- no URDF modification needed. Six virtual joints (3 prismatic + 3 revolute)
are inserted between ``base_link`` and ``pelvis``:

.. code-block:: yaml

    extra_links:
      base_link_x:
        parent_link_name: base_link
        child_link_name: null       # chains to next virtual link
        joint_name: base_j_x
        joint_type: X_PRISM
        ...
      base_link_ztheta:
        parent_link_name: base_link_ytheta
        child_link_name: pelvis     # re-parents pelvis under the virtual chain
        joint_name: base_link_ztheta
        joint_type: Z_ROT
        ...

The URDF itself (``g1_29dof_rev_1_0.urdf``) has a fixed joint from ``base_link`` to
``pelvis``. The virtual chain is injected at load time by the ``extra_links`` mechanism.

Mesh paths in the URDF can use either relative paths (e.g.,
``meshes/pelvis.STL``) or ``package://`` URIs -- cuRobo strips the
``package://`` prefix automatically and resolves the remainder against the
``--asset-path`` directory.


Step 2: Define Tool Frames
--------------------------

Each human skeleton joint maps to a robot link, becoming a **tool frame** in cuRobo's
IK solver. Each tool frame carries an independent position and rotation weight: feet
and hips get high weight to maintain balance, while mid-chain links like shoulders get
low weight since the elbow and wrist targets already constrain the arm.

This mapping is read automatically from SOMA's retargeter config via
``pipeline_utils.get_retargeter_config(source_type, target_type)``. To customize the
mapping or weights for a different robot, edit the corresponding SOMA retargeter config
(``ik_map`` section).


Step 3: Build the Robot Configuration
--------------------------------------

Use the ``build_robot_model`` script to build the cuRobo config with
collision spheres and the self-collision matrix:

.. code-block:: bash

   python -m curobo.examples.getting_started.build_robot_model \\
     --urdf curobo/content/assets/robot/g1/g1_29dof_rev_1_0.urdf \\
     --asset-path curobo/content/assets/robot/g1 \\
     --tool-frames \\
       pelvis torso_link \\
       left_shoulder_roll_link left_elbow_link left_wrist_yaw_link \\
       right_shoulder_roll_link right_elbow_link right_wrist_yaw_link \\
       left_hip_roll_link left_knee_link left_ankle_roll_link \\
       right_hip_roll_link right_knee_link right_ankle_roll_link \\
     --output curobo/content/configs/robot/unitree_g1_29dof_retarget.yml \\
     --visualize

The ``--visualize`` flag opens a Viser web viewer (default ``http://localhost:8080``)
where you can inspect the fitted collision spheres.

A pre-built config is included at
``curobo/content/configs/robot/unitree_g1_29dof_retarget.yml``.


Step 4: Three Levels of Retargeting
-------------------------------------

:class:`~curobo.motion_retargeter.MotionRetargeter` is the high-level API that
implements the two-phase loop described above. You configure it once, then call
``solve_sequence`` (offline) or ``solve_frame`` (streaming); it manages solver
construction, warm-starting, and mode switching internally.

Both IK solvers use ``ik/lbfgs_retarget_ik.yml``, a specialized LBFGS optimizer config
tuned for retargeting. The MPC solver uses ``mpc/lbfgs_retarget_mpc.yml``.

**Per-link weighting**: each tool frame receives independent position and rotation
weights via :class:`~curobo._src.cost.tool_pose_criteria.ToolPoseCriteria`, passed
as a dict in ``MotionRetargeterCfg.tool_pose_criteria``.

The same two-phase solver runs at three fidelity levels. Start with Level 1 to get
results quickly, then upgrade to the higher levels as needed.

.. list-table::
   :header-rows: 1
   :widths: 10 25 25 40

   * - Level
     - Mode
     - Flag
     - Trade-off
   * - 1
     - IK, no self-collision
     - ``self_collision_check=False``
     - Fastest. Robot links may interpenetrate on constrained poses.
   * - 2
     - IK, with self-collision
     - ``self_collision_check=True``
     - ~10–20 % slower; prevents robot links interpenetration.
   * - 3
     - MPC
     - ``use_mpc=True``
     - 2–4× slower; adds trajectory smoothness (acceleration/jerk costs).


Level 1: IK without Self-Collision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/retargeting_ik.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">
       IK without Self-Collision
     </figcaption>
   </figure>

The fastest option. Collision spheres are not loaded, so initialization and
per-frame solve time are both minimized. Use this level when speed matters more than
physical plausibility, or as a first pass to verify the link mapping.

.. code-block:: python

    from curobo.motion_retargeter import (
        MotionRetargeter,
        MotionRetargeterCfg,
        SequenceGoalToolPose,
        ToolPoseCriteria,
    )

    cfg = MotionRetargeterCfg.create(
        robot="unitree_g1_29dof_retarget.yml",
        tool_pose_criteria={
            "pelvis": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.067, 0.067, 0.067],
            ),
            "left_ankle_roll_link": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.067, 0.067, 0.067],
            ),
            # ... one entry per tool frame
        },
        num_envs=1,
        self_collision_check=False,   # no self-collision
    )
    retargeter = MotionRetargeter(cfg)

    # positions: (num_frames, num_envs, num_links, num_goalset, 3) torch tensor
    # quaternions: (num_frames, num_envs, num_links, num_goalset, 4) torch tensor, wxyz convention
    seq = SequenceGoalToolPose(
        tool_frames=cfg.tool_frames,
        position=positions,
        quaternion=quaternions,
    )

    result = retargeter.solve_sequence(seq)
    # result.joint_state.position: (num_envs, num_frames, num_dof)

CLI equivalent:

.. code-block:: bash

    python -m curobo.examples.getting_started.humanoid_retargeting \\
        --input motion.bvh --output motion.csv \\
        --no-self-collision


Level 2: IK with Self-Collision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/retargeting_cfree_ik.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">
       IK with Self-Collision Avoidance
     </figcaption>
   </figure>

Enables cuRobo's self-collision avoidance. The solver penalizes configurations where
collision spheres overlap, preventing interpenetrating limbs. This is the recommended
default for generating training data or deployment motions.

.. code-block:: python

    cfg = MotionRetargeterCfg.create(
        robot="unitree_g1_29dof_retarget.yml",
        tool_pose_criteria={...},
        num_envs=1,
        self_collision_check=True,    # self-collision avoidance enabled
    )
    retargeter = MotionRetargeter(cfg)
    result = retargeter.solve_sequence(seq)

CLI equivalent:

.. code-block:: bash

    python -m curobo.examples.getting_started.humanoid_retargeting \\
        --input motion.bvh --output motion.csv


Level 3: MPC (Trajectory-Based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/retargeting_mpc.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">
       MPC with Self-Collision Avoidance
     </figcaption>
   </figure>

Switches the local solver from frame-by-frame IK to Model Predictive Control. Instead
of solving IK independently per frame, MPC optimizes a trajectory over a planning
horizon, producing smoother results with acceleration and jerk costs.

.. code-block:: python

    cfg = MotionRetargeterCfg.create(
        robot="unitree_g1_29dof_retarget.yml",
        tool_pose_criteria={...},
        num_envs=1,
        self_collision_check=True,
        use_mpc=True,                 # MPC local solver
        steps_per_target=4,
    )
    retargeter = MotionRetargeter(cfg)
    result = retargeter.solve_sequence(seq)

    # result.joint_state: one frame per input target (num_envs, num_frames, num_dof)
    # result.trajectory: smooth intermediate frames (MPC only)

CLI equivalent:

.. code-block:: bash

    python -m curobo.examples.getting_started.humanoid_retargeting \\
        --input motion.bvh --output motion.csv \\
        --mpc --steps-per-target 4

**Key points (all levels):**

- **Quaternion convention**: cuRobo uses ``(w, x, y, z)`` order. Warp and many mocap
  formats use ``(x, y, z, w)``; convert before constructing ``SequenceGoalToolPose``.
- **Output DOFs**: The solution contains 35 DOFs: 6 virtual base joints
  ``[x, y, z, roll, pitch, yaw]`` followed by 29 body joints.
- **Streaming mode**: call ``retargeter.solve_frame(tool_pose)`` one frame at a time
  for online/teleoperation use cases.


For a full description of all configuration parameters and their tuning guidance, see
:class:`~curobo._src.motion.motion_retargeter_cfg.MotionRetargeterCfg`.


Step 5: Running the SOMA Example
----------------------------------

Requires ``soma-retargeter`` to be installed. See the
`README <https://github.com/NVIDIA/soma-retargeter#installation>`_.

**Single file**:

.. code-block:: bash

    python -m curobo.examples.getting_started.humanoid_retargeting \\
        --input /path/to/motion.bvh \\
        --output /path/to/output.csv

**Batch (folder of BVH files)**:

.. code-block:: bash

    python -m curobo.examples.getting_started.humanoid_retargeting \\
        --input /path/to/bvh_folder/ \\
        --output /path/to/csv_folder/

**With visualization** (opens a web viewer at ``http://localhost:8080``):

.. code-block:: bash

    python -m curobo.examples.getting_started.humanoid_retargeting \\
        --input motion.bvh --output motion.csv --visualize

The viewer provides play/pause, loop, speed, frame slider, target sphere visibility,
and a motion dropdown (when multiple clips are loaded).

**Additional flags**:

- ``--robot-config``: cuRobo robot YAML config (default: ``unitree_g1_29dof_retarget.yml``).
- ``--no-self-collision``: disable self-collision checking (Level 1).
- ``--mpc``: enable MPC mode (Level 3).
- ``--steps-per-target N``: MPC steps per input frame (default: 4).
- ``--max-frames N``: limit retargeting to the first N frames.
- ``--max-batch N``: max clips to retarget in parallel (default: 100).


Additional Details
-------------------

- The virtual base joints (``base_link_x/y/z``, ``base_link_xtheta/ytheta/ztheta``) give
  the pelvis 6-DOF freedom. Their limits should be wide enough for the expected motion
  range.
- The SOMA example outputs joint data in ``CSVAnimationBuffer`` format. The first 7
  values per frame are the root transform ``[x, y, z, qx, qy, qz, qw]`` (converted
  from the 6 virtual base DOFs via ``Pose.from_euler_xyz_intrinsic``), followed by the
  29 body joint angles reordered to SOMA CSV convention.
"""

import argparse
import pathlib
import time
from typing import Dict, List

import numpy as np
import torch
import warp as wp

from curobo.motion_retargeter import (
    MotionRetargeter,
    MotionRetargeterCfg,
    RetargetResult,
    SequenceGoalToolPose,
    ToolPoseCriteria,
)
from curobo.scene import Sphere
from curobo.types import DeviceCfg, JointState, Pose


def _build_tool_pose_criteria(ik_map: dict) -> Dict[str, ToolPoseCriteria]:
    """Parse a SOMA ``ik_map`` config into :class:`ToolPoseCriteria`.

    The ik_map has entries like ``{"joint_name": {"t_body": "link", "t_weight": 1.0, "r_weight": 0.5}}``.
    Weights are normalized by the global maximum so that the strongest
    tracked link has weight 1.0.
    """
    max_weight = max(
        max(ik_map[j]["t_weight"], ik_map[j]["r_weight"]) for j in ik_map
    )
    criteria = {}
    for j in ik_map:
        link = ik_map[j]["t_body"]
        pw = 100.0 * ik_map[j]["t_weight"] / max_weight
        rw = 10.0 * ik_map[j]["r_weight"] / max_weight
        criteria[link] = ToolPoseCriteria.track_position_and_orientation(
            xyz=[pw, pw, pw], rpy=[rw, rw, rw], non_terminal_scale=0.5,
        )
    return criteria


def _effectors_to_sequence_goal_tool_pose(
    effector_arrays: List[np.ndarray],
    tool_frame_names: List[str],
    device_cfg: DeviceCfg = DeviceCfg(),
) -> SequenceGoalToolPose:
    """Convert SOMA effector arrays to a :class:`SequenceGoalToolPose`.

    Args:
        effector_arrays: List of arrays, one per environment/clip.
            Each has shape ``(num_frames, num_links, 7)`` with
            ``[px, py, pz, qx, qy, qz, qw]`` (SOMA xyzw convention).
        tool_frame_names: Link names matching dim 1.
        device_cfg: Device and dtype for output tensors.

    Returns:
        SequenceGoalToolPose with shape ``(num_frames, num_envs, num_links, num_goalset, 3/4)``
        (``num_goalset=1``) and wxyz quaternion convention.
    """
    max_frames = max(arr.shape[0] for arr in effector_arrays)
    num_envs = len(effector_arrays)
    num_links = len(tool_frame_names)

    positions = np.zeros((max_frames, num_envs, num_links, 1, 3), dtype=np.float32)
    quaternions = np.zeros((max_frames, num_envs, num_links, 1, 4), dtype=np.float32)
    # Default quaternion to identity (wxyz)
    quaternions[..., 0] = 1.0

    for env_idx, arr in enumerate(effector_arrays):
        num_frames = arr.shape[0]
        positions[:num_frames, env_idx, :, 0, :] = arr[:, :, :3]
        # xyzw -> wxyz
        quaternions[:num_frames, env_idx, :, 0, 0] = arr[:, :, 6]  # w
        quaternions[:num_frames, env_idx, :, 0, 1] = arr[:, :, 3]  # x
        quaternions[:num_frames, env_idx, :, 0, 2] = arr[:, :, 4]  # y
        quaternions[:num_frames, env_idx, :, 0, 3] = arr[:, :, 5]  # z
        if num_frames < max_frames:
            positions[num_frames:, env_idx, :, 0, :] = positions[
                num_frames - 1, env_idx, :, 0, :
            ]
            quaternions[num_frames:, env_idx, :, 0, :] = quaternions[
                num_frames - 1, env_idx, :, 0, :
            ]

    pos_tensor = torch.as_tensor(
        positions, device=device_cfg.device, dtype=device_cfg.dtype
    )
    quat_tensor = torch.as_tensor(
        quaternions, device=device_cfg.device, dtype=device_cfg.dtype
    )

    return SequenceGoalToolPose(
        tool_frames=tool_frame_names,
        position=pos_tensor,
        quaternion=quat_tensor,
    )


def _compute_body_reorder(joint_names: List[str]) -> List[int]:
    """Compute index mapping from cuRobo body joint order to SOMA CSV order.

    cuRobo orders joints by URDF tree traversal (waist -> arms -> legs),
    while SOMA CSV expects (legs -> waist -> arms).
    """
    from soma_retargeter.assets.csv import UnitreeG129DOF_CSVConfig

    csv_config = UnitreeG129DOF_CSVConfig()
    soma_body_names = [h.replace("_dof", "") for h in csv_config.csv_header[7:]]
    curobo_body_names = list(joint_names[6:])
    return [curobo_body_names.index(name) for name in soma_body_names]


def _convert_results_to_csv(
    result: RetargetResult,
    joint_names: List[str],
    sample_rate: float,
):
    """Convert :class:`RetargetResult` to SOMA CSV animation buffers.

    Handles the cuRobo virtual base format ``[x, y, z, roll, pitch, yaw]``
    to SOMA root format ``[px, py, pz, qx, qy, qz, qw]`` conversion using
    :meth:`Pose.from_euler_xyz_intrinsic`, and reorders body joints from
    cuRobo's URDF order to SOMA's CSV order.

    Args:
        result: RetargetResult from ``MotionRetargeter.solve_sequence()``.
            ``joint_state.position`` shape: ``(num_envs, num_frames, num_dof)``.
        joint_names: Joint names in cuRobo DOF order.
        sample_rate: Output sample rate in FPS.

    Returns:
        list[CSVAnimationBuffer]: One per environment.
    """
    from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer

    body_reorder = _compute_body_reorder(joint_names)
    js_positions = result.joint_state.position  # (num_envs, num_frames, num_dof)
    num_envs = js_positions.shape[0]

    csv_buffers = []
    for env in range(num_envs):
        env_positions = js_positions[env].cpu().numpy()  # (num_frames, num_dof)
        csv_rows = []
        for frame_idx in range(env_positions.shape[0]):
            jp = env_positions[frame_idx]
            base_xyz = jp[0:3]
            euler_xyz = torch.as_tensor(
                jp[3:6], dtype=torch.float32
            ).unsqueeze(0)
            pose = Pose.from_euler_xyz_intrinsic(euler_xyz)
            quat_wxyz = pose.quaternion.squeeze(0).cpu().numpy()
            # wxyz -> xyzw for SOMA
            quat_xyzw = np.array([
                quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
            ])
            body_joints = jp[6:][body_reorder]
            csv_rows.append(np.concatenate([base_xyz, quat_xyzw, body_joints]))

        csv_buffers.append(
            CSVAnimationBuffer.create_from_raw_data(csv_rows, sample_rate)
        )
    return csv_buffers


def visualize_motion(
    robot_config: str,
    joint_solutions_list: List[np.ndarray],
    joint_names: List[str],
    sample_rate: float,
    port: int = 8080,
    target_positions_list: List[np.ndarray] = None,
    tool_frames: List[str] = None,
    motion_names: List[str] = None,
    default_frame_skip: int = 1,
):
    """Play back retargeted motions in a viser web viewer.

    Args:
        robot_config: cuRobo robot YAML config name or path.
        joint_solutions_list: List of arrays, each ``(num_frames, num_dof)``.
        joint_names: Joint names matching the solution DOF order.
        sample_rate: Playback FPS.
        port: Viser server port.
        target_positions_list: Optional list of ``(num_frames, num_links, 3)``
            target positions, one per motion.
        tool_frames: Optional link names for target labeling.
        motion_names: Optional display names for the dropdown. Defaults to
            ``["Motion 0", "Motion 1", ...]``.
        default_frame_skip: Initial frame skip value. For MPC trajectories,
            set to ``steps_per_target * interpolation_steps`` to play back
            at the original input frame rate.
    """
    from curobo.types import ContentPath, JointState as JS
    from curobo.viewer import ViserVisualizer

    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_config),
        add_robot_to_scene=True,
        connect_port=port,
        add_control_frames=False,
    )

    server = viz._server
    n_motions = len(joint_solutions_list)
    if motion_names is None:
        motion_names = [f"Motion {i}" for i in range(n_motions)]
    if target_positions_list is None:
        target_positions_list = [None] * n_motions

    current_motion = [0]
    joint_solutions = [joint_solutions_list[0]]
    target_positions = [target_positions_list[0]]
    num_frames = [len(joint_solutions[0])]
    frame_dt = 1.0 / (sample_rate * default_frame_skip)

    show_targets = target_positions[0] is not None and len(target_positions[0]) > 0
    n_target_links = target_positions[0].shape[1] if show_targets else 0
    target_radius = 0.025
    target_color = (0, 100, 0)

    playing = [True]
    looping = [True]
    speed = [1.0]
    current_frame = [0]
    targets_visible = [show_targets]

    with server.gui.add_folder("Playback"):
        if n_motions > 1:
            gui_motion = server.gui.add_dropdown(
                "Motion", options=motion_names, initial_value=motion_names[0]
            )
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=num_frames[0] - 1, step=1, initial_value=0
        )
        gui_play = server.gui.add_button("Pause")
        gui_loop = server.gui.add_checkbox("Loop", initial_value=True)
        gui_speed = server.gui.add_slider(
            "Speed", min=0.1, max=20.0, step=0.1, initial_value=1.0
        )
        gui_skip = server.gui.add_slider(
            "Frame Skip", min=1, max=100, step=1,
            initial_value=min(default_frame_skip, 100),
        )
        if show_targets:
            gui_targets = server.gui.add_checkbox(
                "Show Targets", initial_value=True
            )

    def _switch_motion(idx):
        current_motion[0] = idx
        joint_solutions[0] = joint_solutions_list[idx]
        target_positions[0] = target_positions_list[idx]
        num_frames[0] = len(joint_solutions[0])
        current_frame[0] = 0
        gui_frame.max = num_frames[0] - 1
        gui_frame.value = 0
        _set_frame(0)

    if n_motions > 1:
        @gui_motion.on_update
        def _on_motion(_):
            idx = motion_names.index(gui_motion.value)
            _switch_motion(idx)

    @gui_play.on_click
    def _on_play(_):
        playing[0] = not playing[0]
        gui_play.name = "Pause" if playing[0] else "Play"

    @gui_loop.on_update
    def _on_loop(_):
        looping[0] = gui_loop.value

    @gui_speed.on_update
    def _on_speed(_):
        speed[0] = gui_speed.value

    skip = [max(1, default_frame_skip)]
    gui_frame.step = skip[0]

    @gui_skip.on_update
    def _on_skip(_):
        skip[0] = max(1, int(gui_skip.value))
        gui_frame.step = skip[0]

    @gui_frame.on_update
    def _on_frame(_):
        if not playing[0]:
            current_frame[0] = int(gui_frame.value)
            _set_frame(current_frame[0])

    target_sphere_handles = []
    if show_targets:
        for i in range(n_target_links):
            label = (
                tool_frames[i]
                if tool_frames
                else f"target_{i}"
            )
            handle = server.scene.add_icosphere(
                name=f"/targets/{label}",
                position=target_positions[0][0][i],
                radius=target_radius,
                color=target_color,
            )
            target_sphere_handles.append(handle)

        @gui_targets.on_update
        def _on_targets(_):
            targets_visible[0] = gui_targets.value
            for h in target_sphere_handles:
                h.visible = targets_visible[0]

    def _update_target_spheres(idx):
        tp = target_positions[0]
        if tp is not None and idx < len(tp):
            for i, h in enumerate(target_sphere_handles):
                h.position = tp[idx][i]

    def _set_frame(idx):
        q = torch.tensor(
            joint_solutions[0][idx], dtype=torch.float32, device="cuda:0"
        )
        js = JS.from_position(q.unsqueeze(0), joint_names=joint_names)
        viz.set_joint_state(js)
        if show_targets:
            _update_target_spheres(idx)

    print(f"\n[INFO] Visualization server running at http://localhost:{port}")
    print(f"[INFO] {n_motions} motion(s) loaded")
    print(f"[INFO] Playing at {sample_rate} FPS")
    if show_targets:
        print(f"[INFO] Showing {n_target_links} target link spheres")
    print("[INFO] Press Ctrl+C to exit\n")

    _set_frame(0)

    try:
        while True:
            if playing[0]:
                _set_frame(current_frame[0])
                gui_frame.value = current_frame[0]

                current_frame[0] += skip[0]
                if current_frame[0] >= num_frames[0]:
                    if looping[0]:
                        current_frame[0] = 0
                    else:
                        current_frame[0] = num_frames[0] - 1
                        playing[0] = False
                        gui_play.name = "Play"

                time.sleep(frame_dt * skip[0] / speed[0])
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Visualization stopped.")


def export_motion_to_usd(
    robot_config: str,
    joint_solutions_list: List[np.ndarray],
    joint_names: List[str],
    traj_dt: float,
    output_paths: List[str],
    frame_skip: int = 1,
    target_positions_list: List[np.ndarray] = None,
    tool_frames: List[str] = None,
    target_radius: float = 0.025,
    flatten_usd: bool = False,
):
    """Export retargeted motions to USD animation files.

    Uses :meth:`UsdWriter.write_trajectory_animation` to produce animated USD
    files with robot link meshes and optional target spheres showing the IK
    effector goals.

    Args:
        robot_config: cuRobo robot YAML config name or path.
        joint_solutions_list: List of arrays, each ``(num_frames, num_dof)``.
        joint_names: Joint names matching the solution DOF order.
        traj_dt: Time between consecutive trajectory frames in seconds.
            For non-MPC this is ``1 / sample_rate``. For MPC this is
            ``1 / (sample_rate * steps_per_target * interpolation_steps)``.
        output_paths: List of output ``.usd`` file paths, one per motion.
        frame_skip: Write every Nth frame (default: 1 = all frames).
        target_positions_list: Optional list of ``(num_frames, num_links, 3)``
            target positions to visualize as animated spheres.
        tool_frames: Optional link names for target sphere labeling.
        target_radius: Radius for target spheres (meters).
        flatten_usd: Flatten USD stage (single file, no references).
    """
    from curobo.viewer import UsdWriter

    if target_positions_list is None:
        target_positions_list = [None] * len(joint_solutions_list)

    for idx, (js_np, tp_np, out_path) in enumerate(
        zip(joint_solutions_list, target_positions_list, output_paths)
    ):
        js_subsampled = js_np[::frame_skip]
        dt = traj_dt * frame_skip

        q_traj_tensor = torch.tensor(
            js_subsampled, dtype=torch.float32, device="cuda:0"
        )
        q_traj = JointState.from_position(q_traj_tensor, joint_names=joint_names)

        q_start = JointState.from_position(
            q_traj_tensor[0:1], joint_names=joint_names
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config,
            scene_model=None,
            q_start=q_start,
            q_traj=q_traj,
            dt=dt,
            save_path=str(out_path),
            visualize_robot_spheres=False,
            flatten_usd=False,
        )

        if tp_np is not None:
            tp_subsampled = tp_np[::frame_skip]
            num_links = tp_subsampled.shape[1]
            target_color = [0.0, 0.4, 0.0, 1.0]

            sphere_traj = []
            for t in range(len(tp_subsampled)):
                frame_spheres = []
                for li in range(num_links):
                    label = (
                        tool_frames[li]
                        if tool_frames
                        else f"target_{li}"
                    )
                    pos = tp_subsampled[t, li].tolist()
                    frame_spheres.append(
                        Sphere(
                            name=label,
                            pose=pos + [1, 0, 0, 0],
                            radius=target_radius,
                            color=target_color,
                        )
                    )
                sphere_traj.append(frame_spheres)

            writer = UsdWriter()
            writer.load_stage_from_file(str(out_path))
            writer.interpolation_steps = 1
            writer.create_obstacle_animation(
                sphere_traj,
                base_frame="/world",
                obstacles_frame="targets",
            )
            writer.write_stage_to_file(str(out_path), flatten=flatten_usd)

        n_written = len(js_subsampled)
        print(
            f"[INFO] USD saved: {out_path}"
            f" ({n_written} frames, skip={frame_skip}, dt={dt:.4f}s)"
        )


def retarget_bvh_files(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    robot_config: str = "unitree_g1_29dof_retarget.yml",
    do_visualize: bool = False,
    viz_port: int = 8080,
    use_mpc: bool = False,
    steps_per_target: int = 4,
    max_frames: int = None,
    self_collision_check: bool = True,
    max_num_envs: int = 4,
    velocity_regularization_weight: float = None,
    acceleration_regularization_weight: float = None,
    usd_output: str = None,
):
    """Load BVH files, retarget via MotionRetargeter, save as SOMA CSV.

    Args:
        input_path: BVH file or directory of BVH files.
        output_path: Output CSV file or directory.
        robot_config: cuRobo robot YAML config name.
        do_visualize: Open viser web viewer after retargeting.
        viz_port: Viser server port.
        use_mpc: Use MPC instead of frame-by-frame IK.
        steps_per_target: MPC steps per target frame (only with use_mpc).
        max_frames: Limit retargeting to the first N frames.
        self_collision_check: Enable self-collision checking.
        max_num_envs: Maximum batch size per retargeting pass. Clips are
            processed in chunks of this size to avoid GPU OOM.
        usd_output: Path for USD animation export (None to skip).
    """
    import soma_retargeter.assets.bvh as bvh_utils
    import soma_retargeter.assets.csv as csv_utils
    import soma_retargeter.pipelines.utils as pipeline_utils
    import soma_retargeter.utils.io_utils as io_utils
    from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
    from soma_retargeter.utils.space_conversion_utils import (
        SpaceConverter,
        get_facing_direction_type_from_str,
    )

    # --- Discover BVH files ---
    bvh_files = []
    if input_path.is_dir():
        bvh_files = sorted(input_path.glob("*.bvh"))
        if not bvh_files:
            print(f"[ERROR] No .bvh files found in {input_path}")
            return
        output_path.mkdir(parents=True, exist_ok=True)
    elif input_path.is_file():
        bvh_files = [input_path]
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        return

    # --- Load skeleton and animations ---
    skeleton, _ = bvh_utils.load_bvh(str(bvh_files[0]))

    converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
    root_xform = converter.transform(wp.transform_identity())

    animations = []
    for bvh_file in bvh_files:
        _, anim = bvh_utils.load_bvh(str(bvh_file), skeleton)
        animations.append(anim)
    print(f"[INFO] Loaded {len(animations)} BVH file(s)")

    # --- Load SOMA retarget config ---
    source_type = pipeline_utils.get_source_type_from_str("soma")
    target_type = pipeline_utils.get_target_type_from_str("unitree_g1")
    retargeter_config = pipeline_utils.get_retargeter_config(source_type, target_type)

    ik_map = retargeter_config["ik_map"]
    mapped_joints = list(ik_map.keys())
    tool_frame_names = [ik_map[j]["t_body"] for j in mapped_joints]

    tool_pose_criteria = _build_tool_pose_criteria(ik_map)

    # --- Convert animations to effectors ---
    human_robot_scaler = HumanToRobotScaler(
        skeleton,
        retargeter_config["model_height"],
        io_utils.get_config_file(retargeter_config["human_robot_scaler_config"]),
    )

    effector_names = human_robot_scaler.effector_names()
    target_effector_indices = [
        effector_names.index(name) for name in mapped_joints
    ]

    # --- Handle initialization/stabilization frames ---
    num_init_frames = retargeter_config.get("num_initialization_frames", 5)
    num_stab_frames = retargeter_config.get("num_stabilization_frames", 5)
    num_frames_to_remove = num_init_frames + num_stab_frames

    init_pose = None
    if retargeter_config.get("initialization_pose") and num_init_frames > 0:
        import soma_retargeter.assets.bvh as bvh_init_utils
        from soma_retargeter.animation.skeleton import SkeletonInstance

        init_skel, init_anim = bvh_init_utils.load_bvh(
            io_utils.get_config_file(retargeter_config["initialization_pose"])
        )
        init_pose = SkeletonInstance(
            init_skel, [0, 0, 0], wp.transform_identity()
        )
        init_pose.set_local_transforms(init_anim.get_local_transforms(0))

    offsets = [root_xform] * len(animations)
    all_effector_arrays = []
    for i, anim in enumerate(animations):
        buffer = anim
        if init_pose is not None:
            import soma_retargeter.utils.newton_utils as newton_utils

            buffer = newton_utils.create_buffer_with_initialization_frames(
                init_pose, anim, num_init_frames, num_stab_frames,
            )
        buffer_effectors = human_robot_scaler.compute_effectors_from_buffer(
            buffer, True, offsets[i]
        )
        all_effector_arrays.append(buffer_effectors[:, target_effector_indices, :])

    if max_frames is not None:
        all_effector_arrays = [arr[:max_frames] for arr in all_effector_arrays]

    # --- Retarget in chunks of max_num_envs ---
    sample_rate = animations[0].sample_rate
    total_envs = len(animations)
    all_js_np = []
    all_tp_np = []
    all_csv_buffers = []
    joint_names_out = None
    traj_ratio = 1

    for chunk_start in range(0, total_envs, max_num_envs):
        chunk_end = min(chunk_start + max_num_envs, total_envs)
        chunk_effectors = all_effector_arrays[chunk_start:chunk_end]
        chunk_size = len(chunk_effectors)

        print(
            f"[INFO] Retargeting chunk {chunk_start + 1}-{chunk_end} "
            f"of {total_envs}"
        )

        cfg = MotionRetargeterCfg.create(
            robot=robot_config,
            tool_pose_criteria=tool_pose_criteria,
            num_envs=chunk_size,
            use_mpc=use_mpc,
            steps_per_target=steps_per_target,
            self_collision_check=self_collision_check,
            velocity_regularization_weight=velocity_regularization_weight,
            acceleration_regularization_weight=acceleration_regularization_weight,
        )
        retargeter = MotionRetargeter(cfg)
        if joint_names_out is None:
            joint_names_out = retargeter.joint_names

        seq_tool_pose = _effectors_to_sequence_goal_tool_pose(
            chunk_effectors, tool_frame_names, cfg.device_cfg,
        )

        result = retargeter.solve_sequence(seq_tool_pose)

        # Trim initialization/stabilization frames
        if num_frames_to_remove > 0:
            trimmed_traj = None
            if result.trajectory is not None:
                traj_frames_per_input = (
                    result.trajectory.position.shape[1]
                    // result.joint_state.position.shape[1]
                )
                traj_frames_to_remove = (
                    num_frames_to_remove * traj_frames_per_input
                )
                trimmed_traj = JointState.from_position(
                    result.trajectory.position[:, traj_frames_to_remove:, :],
                    joint_names=retargeter.joint_names,
                )
            result = RetargetResult(
                joint_state=JointState.from_position(
                    result.joint_state.position[:, num_frames_to_remove:, :],
                    joint_names=retargeter.joint_names,
                ),
                trajectory=trimmed_traj,
            )

        # Convert to CSV
        chunk_csv = _convert_results_to_csv(
            result, retargeter.joint_names, sample_rate,
        )
        all_csv_buffers.extend(chunk_csv)

        # Collect visualization data; use trajectory (smoother) when
        # available, repeating each target to match frame count.
        for env_idx in range(chunk_size):
            target_pos = (
                seq_tool_pose.position[:, env_idx, :, :].cpu().numpy()
            )
            if num_frames_to_remove > 0:
                target_pos = target_pos[num_frames_to_remove:]
            if result.trajectory is not None:
                js_np = result.trajectory.position[env_idx].cpu().numpy()
                traj_ratio = js_np.shape[0] // max(target_pos.shape[0], 1)
                if traj_ratio > 1:
                    target_pos = np.repeat(target_pos, traj_ratio, axis=0)
            else:
                js_np = result.joint_state.position[env_idx].cpu().numpy()
            all_js_np.append(js_np)
            all_tp_np.append(target_pos)

    # --- Save CSV files ---
    for i, buf in enumerate(all_csv_buffers):
        if input_path.is_dir():
            out_file = output_path / bvh_files[i].with_suffix(".csv").name
        else:
            out_file = output_path
        csv_utils.save_csv(str(out_file), buf)
        print(f"[INFO] Saved: {out_file}")

    # --- Export USD ---
    if usd_output is not None and len(all_js_np) > 0:
        usd_path = pathlib.Path(usd_output)
        if total_envs == 1:
            usd_paths = [str(usd_path)]
        else:
            usd_path.mkdir(parents=True, exist_ok=True)
            usd_paths = [
                str(usd_path / (bvh_files[i].stem + ".usd"))
                for i in range(total_envs)
            ]
        export_motion_to_usd(
            robot_config=robot_config,
            joint_solutions_list=all_js_np,
            joint_names=joint_names_out,
            traj_dt=1.0 / (sample_rate * traj_ratio),
            output_paths=usd_paths,
            frame_skip=traj_ratio,
            target_positions_list=all_tp_np,
            tool_frames=tool_frame_names,
        )

    # --- Visualize ---
    if do_visualize:
        motion_names = [
            bvh_files[i].stem if input_path.is_dir() else bvh_files[i].name
            for i in range(total_envs)
        ]
        visualize_motion(
            robot_config=robot_config,
            joint_solutions_list=all_js_np,
            joint_names=joint_names_out,
            sample_rate=sample_rate * traj_ratio,
            port=viz_port,
            target_positions_list=all_tp_np,
            tool_frames=tool_frame_names,
            motion_names=motion_names,
            default_frame_skip=traj_ratio,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Retarget BVH motion to Unitree G1 using cuRobo MotionRetargeter"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input BVH file or directory of BVH files",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV file or directory for CSV files",
    )
    parser.add_argument(
        "--robot-config", type=str, default="unitree_g1_29dof_retarget.yml",
        help="cuRobo robot YAML config (default: unitree_g1_29dof_retarget.yml)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open viser web viewer to play back the retargeted motion",
    )
    parser.add_argument(
        "--viz-port", type=int, default=8080,
        help="Viser visualization server port (default: 8080)",
    )
    parser.add_argument(
        "--mpc", action="store_true",
        help="Use MPC pipeline instead of frame-by-frame IK",
    )
    parser.add_argument(
        "--steps-per-target", type=int, default=4,
        help="MPC steps per target frame (default: 2, only used with --mpc)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit retargeting to the first N frames (default: all)",
    )
    parser.add_argument(
        "--no-self-collision", action="store_true",
        help="Disable self-collision checking",
    )
    parser.add_argument(
        "--max-batch", type=int, default=100,
        help="Max clips to retarget in parallel (default: 100)",
    )
    parser.add_argument(
        "--vel-reg", type=float, default=None,
        help="Velocity regularization weight (default: use YAML value)",
    )
    parser.add_argument(
        "--acc-reg", type=float, default=None,
        help="Acceleration regularization weight (default: use YAML value)",
    )
    parser.add_argument(
        "--usd", type=str, default=None,
        help="Export USD animation to this path (file for single clip, directory for batch)",
    )
    args = parser.parse_args()

    wp.init()
    retarget_bvh_files(
        pathlib.Path(args.input),
        pathlib.Path(args.output),
        args.robot_config,
        do_visualize=args.visualize,
        viz_port=args.viz_port,
        use_mpc=args.mpc,
        steps_per_target=args.steps_per_target,
        max_frames=args.max_frames,
        self_collision_check=not args.no_self_collision,
        max_num_envs=args.max_batch,
        velocity_regularization_weight=args.vel_reg,
        acceleration_regularization_weight=args.acc_reg,
        usd_output=args.usd,
    )


if __name__ == "__main__":
    main()
