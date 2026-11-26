# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test collision-avoiding motion planning for a UR10e on a linear track.

The main goal is to check obstacle avoidance.

Viewer controls:

- Target: drag the ``target`` gizmo to set the goal tool pose. It starts at the far
  end of the track, so the arm has to go around the blocks.
- Move: plan from the current pose to the target, then play the trajectory.
- Reset robot: return the robot to the start pose.
- Reset target: return the target to the far end of the track.
- Reset blocks: remove all blocks, then restore the default blocks.
- Remove all blocks: delete every block.
- Planner: ``RRT + TrajOpt`` refines the RRT seed with trajectory optimization.
  ``RRT only`` skips that step, so it is faster.
- Gizmo size: scale the target and every block gizmo together. The same size sets
  the click area around them.
- Select: choose ``none``, ``target``, or a block from the dropdown. Clicking the
  target or a block also selects it, and clicking empty space selects ``none``.
  Only the selected block shows its gizmo and size sliders.
- Add block / Remove selected: add a block at the default pose, or delete the
  selected block. The target cannot be removed.

Run with ``python -m curobo.examples.getting_started.linear_track_planning``.
Use ``--port`` to change the viewer port. Use ``--test`` to run a headless self-test.
"""

import argparse
import copy
import itertools
import threading
import time

import numpy as np

from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

PLAYBACK_DT_S = 0.02
TRACK_JOINT = "track_joint"
TRACK_START_M = -0.65
# Wider than the physical travel (-5.65 to -0.65 m), since cuRobo clips starts near a limit.
TRACK_LIMITS_M = [-5.8, -0.5]
# Rough rail beam under the track travel. It is a planning obstacle and is drawn in the viewer.
TRACK_DIMS_M = [5.4, 0.3, 0.1]
TRACK_POSE = [-3.1, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0]
MAX_BLOCKS = 15
DEFAULT_BLOCK_POSE = [-3.0, 1.0, 1.353, 1.0, 0.0, 0.0, 0.0]
# Far end of the track. The planner may stop short of it, because the track and arm are redundant.
TARGET_TRACK_M = -5.5
# Staggered in y and z around the default tool path, so the arm has to go around them.
DEFAULT_BLOCKS = [
    ([-1.9, 0.7, 1.25], 0.25),
    ([-2.7, -0.25, 1.55], 0.25),
    ([-3.5, 0.45, 1.15], 0.25),
    ([-4.3, -0.15, 1.45], 0.25),
    ([-5.1, 0.6, 1.3], 0.25),
]


def _ray_point_distance(origin, direction, point, radius):
    """Return the ray distance to a gizmo center if the ray passes within radius."""
    to_point = point - origin
    t = float(to_point @ direction)
    if t < 0.0 or np.linalg.norm(to_point - t * direction) > radius:
        return None
    return t


def _ray_box_distance(origin, direction, center, wxyz, size):
    """Return the ray distance to the entry point of an oriented box, or None on a miss."""
    w, x, y, z = wxyz
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local_origin = rot.T @ (origin - center)  # Move the ray into the box's local frame.
    local_dir = rot.T @ direction
    half = np.asarray(size, dtype=float) / 2.0
    t_near, t_far = -np.inf, np.inf
    for axis in range(3):
        if abs(local_dir[axis]) < 1e-9:
            if abs(local_origin[axis]) > half[axis]:
                return None
            continue
        t1 = (-half[axis] - local_origin[axis]) / local_dir[axis]
        t2 = (half[axis] - local_origin[axis]) / local_dir[axis]
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
    if t_near > t_far or t_far < 0.0:
        return None
    return max(t_near, 0.0)


def track_robot_cfg():
    """Return the UR10e config with a linear track inserted via ``extra_links``."""
    data = load_yaml(join_path(get_robot_configs_path(), "ur10e.yml"))
    kin = data["robot_cfg"]["kinematics"]
    kin["extra_links"] = {
        "track_carriage": {
            "parent_link_name": "base_link",
            "link_name": "track_carriage",
            "joint_name": TRACK_JOINT,
            "joint_type": "X_PRISM",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": TRACK_LIMITS_M,
            "joint_velocity_limits": [-1.5, 1.5],
        },
        # Places the arm base 0.61 m above the track, turned 90 degrees about z.
        # base_link_inertia is re-parented under this mount, keeping its URDF offset.
        "track_mount": {
            "parent_link_name": "track_carriage",
            "child_link_name": "base_link_inertia",
            "link_name": "track_mount",
            "joint_name": "track_mount_joint",
            "joint_type": "FIXED",
            "fixed_transform": [0, 0, 0.61, 0.7071068, 0, 0, 0.7071068],
        },
    }
    cspace = kin["cspace"]
    cspace["joint_names"] = [TRACK_JOINT] + cspace["joint_names"]
    cspace["default_joint_position"] = [TRACK_START_M] + cspace["default_joint_position"]
    cspace["null_space_weight"] = [1.0] + cspace["null_space_weight"]
    cspace["cspace_distance_weight"] = [1.0] + cspace["cspace_distance_weight"]
    return data


def build_planner(robot_data):
    """Build a motion planner for the track robot and warm it up."""
    cfg = MotionPlannerCfg.create(
        robot=robot_data,
        scene_model=None,
        # One slot per block, plus the track.
        collision_cache={"cuboid": MAX_BLOCKS + 1},
        # CUDA graph capture failed on the test GPU, so it is disabled.
        use_cuda_graph=False,
        graph_planner_algorithm="rrt",
        enable_trajopt=True,
        always_run_trajopt=True,
        run_trajopt_on_graph_seed=True,
    )
    planner = MotionPlanner(cfg)
    print("Warming up motion planner...")
    # Graph warmup is skipped, so the first RRT call (the first Move) is slower.
    planner.warmup(enable_graph=False, num_warmup_iterations=5)
    return planner


def make_scene(blocks):
    """Return a scene with the track and every block as cuboids."""
    cuboids = [Cuboid(name="track", dims=TRACK_DIMS_M, pose=TRACK_POSE)]
    for block in blocks:
        pose = [float(v) for v in (*block["gizmo"].position, *block["gizmo"].wxyz)]
        cuboids.append(Cuboid(name=block["label"], dims=block["size"], pose=pose))
    return Scene(cuboid=cuboids)


def plan_to_pose(planner, current_q, position, wxyz):
    """Return the joint trajectory from current_q to a tool pose, or None if planning fails."""
    goal = GoalToolPose.from_poses(
        {planner.tool_frames[0]: Pose.from_numpy(position, wxyz)},
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )
    # 0 runs RRT on the first attempt. The default of 1 only uses RRT on retries.
    result = planner.plan_pose(
        goal,
        current_q,
        max_attempts=3,
        enable_graph_attempt=0,
        use_implicit_goal=True,
    )
    if result is None or not bool(result.success.any()):
        return None
    return result.get_interpolated_plan().position.reshape(-1, len(planner.joint_names))


def test():
    """Plan around the default blocks to the far end of the track."""
    planner = build_planner(track_robot_cfg())
    cuboids = [Cuboid(name="track", dims=TRACK_DIMS_M, pose=TRACK_POSE)]
    for index, (xyz, size) in enumerate(DEFAULT_BLOCKS):
        cuboids.append(
            Cuboid(name=f"block_{index}", dims=[size] * 3, pose=[*xyz, 1.0, 0.0, 0.0, 0.0])
        )
    planner.update_world(Scene(cuboid=cuboids))

    tool_frame = planner.tool_frames[0]
    start_q = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    end_q = start_q.clone()
    end_q.position[:, 0] = TARGET_TRACK_M
    target = planner.compute_kinematics(end_q).tool_poses[tool_frame]
    target_xyz = target.position.reshape(-1).cpu().numpy()
    target_wxyz = target.quaternion.reshape(-1).cpu().numpy()

    traj = plan_to_pose(planner, start_q, target_xyz, target_wxyz)
    assert traj is not None, "Planning around the default blocks failed"
    end = planner.compute_kinematics(
        JointState.from_position(traj[-1].unsqueeze(0), joint_names=planner.joint_names)
    ).tool_poses[tool_frame]
    end_error = (end.position.reshape(-1) - target.position.reshape(-1)).norm().item()
    assert end_error < 0.005, f"Tool ends {end_error:.4f} m from the target"
    assert traj[-1, 0].item() < -4.0, "Track did not travel past the blocks"
    print("linear_track_planning self-test passed")


def main():
    """Run the viewer, or the headless self-test with --test."""
    parser = argparse.ArgumentParser(description="UR10e on a linear track")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--test", action="store_true", help="Run as self-test with assertions")
    args = parser.parse_args()
    if args.test:
        test()
        return

    robot_data = track_robot_cfg()
    # ViserVisualizer edits the config it receives, so the viewer gets a copy.
    viz = ViserVisualizer(
        content_path=copy.deepcopy(robot_data),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )
    # Uses the viser server behind ViserVisualizer, as motion_planning.py does.
    server = viz._server
    server.scene.add_box(
        "/track",
        color=(150, 150, 150),
        dimensions=tuple(TRACK_DIMS_M),
        position=tuple(TRACK_POSE[:3]),
    )
    planner = build_planner(robot_data)
    tool_frame = planner.tool_frames[0]
    start_q = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    viz.set_joint_state(start_q)

    end_q = start_q.clone()
    end_q.position[:, 0] = TARGET_TRACK_M
    default_target = planner.compute_kinematics(end_q).tool_poses[tool_frame]
    default_target_xyz = tuple(default_target.position.reshape(-1).tolist())
    default_target_wxyz = tuple(default_target.quaternion.reshape(-1).tolist())
    move_button = server.gui.add_button("Move")
    status = server.gui.add_text("Status", initial_value="Ready", disabled=True)
    current = {"q": start_q}
    move_lock = threading.Lock()
    blocks = []
    block_ids = itertools.count()
    selected = {"label": "none"}
    syncing = {"on": False}

    size_sliders = [
        server.gui.add_slider(axis, min=0.05, max=1.0, step=0.01, initial_value=0.2, disabled=True)
        for axis in ("x", "y", "z")
    ]
    select_dropdown = server.gui.add_dropdown("Select", options=["none"], initial_value="none")
    gizmo_size = server.gui.add_slider(
        "Gizmo size", min=0.05, max=0.5, step=0.01, initial_value=0.2
    )

    def make_target():
        gizmo = server.scene.add_transform_controls(
            "/target",
            scale=gizmo_size.value,
            position=default_target_xyz,
            wxyz=default_target_wxyz,
            depth_test=False,
        )
        text = server.scene.add_label(
            "/target_text", "target", anchor="bottom-center", position=default_target_xyz
        )
        target = {"label": "target", "gizmo": gizmo, "text": text}
        follow_target(target)
        gizmo.on_update(lambda _, t=target: follow_target(t))
        return target

    def follow_target(target):
        # The label sits just above its gizmo.
        center = np.asarray(target["gizmo"].position, dtype=float)
        target["text"].position = tuple(center + np.array([0.0, 0.0, 0.15]))

    def find_block(label):
        return next((b for b in blocks if b["label"] == label), None)

    def refresh_box(block):
        if block["box"] is not None:
            block["box"].remove()
        # The selected block is drawn as a wireframe, so the gizmo inside it stays visible.
        block["box"] = server.scene.add_box(
            f"/blocks/{block['label']}",
            color=(200, 120, 60),
            dimensions=tuple(block["size"]),
            position=tuple(block["gizmo"].position),
            wxyz=tuple(block["gizmo"].wxyz),
            wireframe=block["label"] == selected["label"],
            opacity=None if block["label"] == selected["label"] else 0.4,
        )

    def follow_gizmo(block):
        block["box"].position = block["gizmo"].position
        block["box"].wxyz = block["gizmo"].wxyz

    def make_block(index, pose, size):
        block = {"label": f"block_{index}", "size": list(size), "box": None}
        block["gizmo"] = viz.add_control_frame(
            f"/blocks/{block['label']}_gizmo", Pose.from_list(pose), scale=gizmo_size.value
        )
        refresh_box(block)
        block["gizmo"].on_update(lambda _, b=block: follow_gizmo(b))
        return block

    def select(label):
        if selected["label"] == label:
            return
        selected["label"] = label
        for block in blocks:
            block["gizmo"].visible = block["label"] == label
            refresh_box(block)
        block = find_block(label)
        # Setting slider values fires on_update, so this flag makes on_size ignore those calls.
        syncing["on"] = True
        for axis, slider in enumerate(size_sliders):
            slider.disabled = block is None
            if block is not None:
                slider.value = block["size"][axis]
        syncing["on"] = False
        remove_button.disabled = block is None
        select_dropdown.value = label

    def on_size():
        if syncing["on"]:
            return
        block = find_block(selected["label"])
        if block is None:
            return
        block["size"] = [float(s.value) for s in size_sliders]
        refresh_box(block)

    # Viser does not report what was clicked, so hits are found with rays. Clicks on a gizmo
    # keep the current selection, and clicks on empty space deselect.
    def on_scene_click(event):
        origin = np.asarray(event.ray_origin, dtype=float)
        direction = np.asarray(event.ray_direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        hits = []
        target_center = np.asarray(target["gizmo"].position, dtype=float)
        t = _ray_point_distance(origin, direction, target_center, gizmo_size.value)
        if t is not None:
            hits.append((t, "target"))
        for block in blocks:
            center = np.asarray(block["gizmo"].position, dtype=float)
            if block["label"] == selected["label"]:
                t = _ray_point_distance(origin, direction, center, gizmo_size.value)
                if t is not None:
                    hits.append((t, "gizmo"))
            t = _ray_box_distance(origin, direction, center, block["gizmo"].wxyz, block["size"])
            if t is not None:
                hits.append((t, block["label"]))
        if not hits:
            select("none")
            return
        _, label = min(hits)
        if label != "gizmo" and label != selected["label"]:
            select(label)

    server.scene.on_click()(on_scene_click)

    for slider in size_sliders:
        slider.on_update(lambda _: on_size())
    select_dropdown.on_update(lambda _: select(select_dropdown.value))

    def refresh_options():
        select_dropdown.options = ["none", "target"] + [b["label"] for b in blocks]

    def add_block():
        if len(blocks) >= MAX_BLOCKS:
            status.value = f"Up to {MAX_BLOCKS} blocks"
            return
        block = make_block(next(block_ids), DEFAULT_BLOCK_POSE, [0.2, 0.2, 0.2])
        blocks.append(block)
        refresh_options()
        select(block["label"])

    def remove_selected():
        block = find_block(selected["label"])
        if block is None:
            return
        blocks.remove(block)
        block["box"].remove()
        block["gizmo"].remove()
        select("none")
        refresh_options()

    def create_default_blocks():
        for xyz, size in DEFAULT_BLOCKS:
            blocks.append(make_block(next(block_ids), [*xyz, 1.0, 0.0, 0.0, 0.0], [size] * 3))
        refresh_options()

    target = make_target()
    create_default_blocks()

    def apply_gizmo_size():
        target["gizmo"].scale = gizmo_size.value
        for block in blocks:
            block["gizmo"].scale = gizmo_size.value

    gizmo_size.on_update(lambda _: apply_gizmo_size())

    server.gui.add_button("Add block").on_click(lambda _: add_block())
    remove_button = server.gui.add_button("Remove selected", disabled=True)
    remove_button.on_click(lambda _: remove_selected())

    def set_planner_mode(mode):
        use_trajopt = mode == "RRT + TrajOpt"
        planner.config.enable_trajopt = use_trajopt
        planner.config.always_run_trajopt = use_trajopt
        planner.config.run_trajopt_on_graph_seed = use_trajopt

    planner_dropdown = server.gui.add_dropdown(
        "Planner", options=["RRT + TrajOpt", "RRT only"], initial_value="RRT + TrajOpt"
    )
    planner_dropdown.on_update(lambda _: set_planner_mode(planner_dropdown.value))

    def reset_robot():
        # Playback would overwrite the robot pose at the end, so reset is refused while busy.
        if not move_lock.acquire(blocking=False):
            status.value = "Busy, try again after playback"
            return
        try:
            current["q"] = start_q
            viz.set_joint_state(start_q)
            status.value = "Robot reset"
        finally:
            move_lock.release()

    def reset_target():
        target["gizmo"].position = default_target_xyz
        target["gizmo"].wxyz = default_target_wxyz
        follow_target(target)
        status.value = "Target reset"

    def remove_all_blocks():
        for block in blocks:
            block["box"].remove()
            block["gizmo"].remove()
        blocks.clear()
        select("none")
        refresh_options()
        status.value = "All blocks removed"

    def reset_blocks():
        remove_all_blocks()
        create_default_blocks()
        status.value = "Blocks reset"

    server.gui.add_button("Reset robot").on_click(lambda _: reset_robot())
    server.gui.add_button("Reset target").on_click(lambda _: reset_target())
    server.gui.add_button("Reset blocks").on_click(lambda _: reset_blocks())
    server.gui.add_button("Remove all blocks").on_click(lambda _: remove_all_blocks())

    def on_move(_):
        if not move_lock.acquire(blocking=False):
            return
        try:
            planner.update_world(make_scene(blocks))
            status.value = "Planning..."
            traj = plan_to_pose(
                planner, current["q"], target["gizmo"].position, target["gizmo"].wxyz
            )
            if traj is None:
                status.value = "Planning failed"
                return
            status.value = "Playing"
            for q in traj:
                viz.set_joint_state(
                    JointState.from_position(q.unsqueeze(0), joint_names=planner.joint_names)
                )
                time.sleep(PLAYBACK_DT_S)
            current["q"] = JointState.from_position(
                traj[-1].unsqueeze(0), joint_names=planner.joint_names
            )
            status.value = "Done"
        finally:
            move_lock.release()

    move_button.on_click(on_move)
    print(f"Open http://localhost:{args.port} in your browser. Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
