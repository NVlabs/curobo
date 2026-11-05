# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edit Cartesian waypoints and preview a collision-checked robot trajectory in Viser.

.. raw:: html

    <video autoplay loop muted playsinline controls preload="metadata" width="100%">
      <source src="../videos/get_started_cartesian_waypoints.webm" type="video/webm">
    </video>

The video uses the defaults: 32 waypoints, 32 spline knots, and **Path scale** 1.25.
Path tolerances are 5 cm and 15 degrees; goal tolerances are 0.5 cm and 3 degrees.

By the end of this tutorial you will have:

- Loaded the Franka Panda and a table obstacle into a browser-based viewer
- Planned motion through an ordered set of Cartesian targets
- Edited a target and replayed the validated joint trajectory

Step 1: Start the viewer
------------------------

Run this command in the environment from :ref:`tutorial_installation`:

.. code-block:: bash

   python -m curobo.examples.getting_started.cartesian_waypoints

Open http://localhost:8080. Wait until the status reads **Ready** and the **Plan**
button is enabled. Startup warms both tracking modes with CUDA graphs enabled.
The viewer starts with 32 waypoints, 32 spline knots, and **Path scale** 1.25.
The path starts 10 cm forward (world +X) and 8 cm below the home tool pose.

Step 2: Plan your first path
----------------------------

1. In **Path presets**, click **Arc**.
2. Keep **Path scale** at **1.25** for the default path.
3. Leave **Waypoint tracking** set to **Position + orientation**.
4. Click **Plan**.

The orange line connects the requested targets. When planning succeeds, a green
line shows the validated tool trajectory and the robot plays the motion.
**Motion time** reports the trajectory duration; **Planning time** reports the
time to solve start IK and plan the path. **Native max error** reports the
largest tracking error checked by the planner, including the start pose.

The first numbered waypoint defines the start, and the last defines the goal.
The planner solves IK at the first waypoint before planning through the rest.
The home configuration is an IK seed; playback begins at the solved start pose.

Step 3: Edit and replan
-----------------------

Choose **Edit waypoint**, or click an orange marker, to select a target. Drag its
translation arrows to move it and its rotation rings to change its orientation.
Click **Plan** again to compute a trajectory through the edited targets.

Edits clear the previous preview. Playback is enabled only after planning and
validation succeed. In **Playback**, pause **Play** and move **Progress** to
inspect the motion, then resume **Play**. Use **Speed** to change playback speed.

Step 4: Try the other presets
-----------------------------

Click **Straight line**, **Circle**, **Crane pick/place**, **Front to back**, or
**Zigzag**, then click **Plan**. Presets increase the waypoint count when needed
to retain their defining corners. **Crane pick/place** describes a tool path;
it does not attach or transport an object.

At **Path scale** 1, the largest preset bounding-box dimension is 0.25 m.
The default scale of 1.25 makes that dimension 31.25 cm.
Scaling resizes the targets about the first waypoint, preserving individual
edits and group placement. Enable **Move whole path** to translate or rotate
all targets together. Disable it to return to individual target editing.
**Reset waypoints** restores the selected preset and the group's initial pose.

Step 5: Change tracking and inspect the result
----------------------------------------------

Set **Waypoint tracking** to **Position only**, then click **Plan**. This mode
ignores rotations at every target, including the goal. **Position + orientation**
tracks both translation and rotation. The requested reference uses linear
position interpolation and quaternion SLERP between waypoint rotations.

Expand **Tolerances** to inspect the path and goal tolerances. Positions are in
centimeters and orientations are in degrees. The defaults allow 5 cm and
15 degrees along the path, with 0.5 cm and 3 degrees at the goal. The green curve
may depart from the orange polyline within those tolerances. A failed plan
reports whether start IK, path tracking, collisions, or joint limits and dynamics
prevented a valid result. Shorten or reposition the path before replanning.

Use **Spline knots** to change the number of spline controls. Changing this
rebuilds and warms the planner while preserving waypoint edits. More knots can
reduce tracking error, but may increase planning time and motion duration.
The dropdown offers powers of two from 16 through 1024. A command-line count
from 12 through 1024 is also added to the selector. To start with 64 knots, run:

.. code-block:: bash

   python -m curobo.examples.getting_started.cartesian_waypoints --num-knots 64

The waypoint-count slider supports up to 256 poses when the spline has enough
reference intervals. Press Ctrl+C in the terminal to stop the viewer.

For a one-shot server and planner check without a browser, run:

.. code-block:: bash

   python -m curobo.examples.getting_started.cartesian_waypoints --test
"""

from __future__ import annotations

import argparse
import math
import queue
import time
from dataclasses import replace
from threading import Event
from typing import TYPE_CHECKING

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, ToolPoseCriteria
from curobo.viewer import ViserVisualizer

if TYPE_CHECKING:
    import viser


_PATTERN_MIN_COUNTS = {
    "Straight line": 2,
    "Arc": 3,
    "Circle": 9,
    "Crane pick/place": 4,
    "Front to back": 5,
    "Zigzag": 5,
}
_PRESET_BBOX_SIZE = 0.25
_DEFAULT_PATH_SCALE = 1.25
_DEFAULT_PATH_OFFSET = (0.10, 0.0, -0.08)
_WARMUP_PATH_SCALE = 0.3


def _pattern_offsets(pattern: str, count: int) -> np.ndarray:
    """Return target offsets ``[count,3]`` in a 0.25 m box, preserving each preset's shape."""
    if count == 1:
        return np.zeros((1, 3), dtype=np.float32)
    count -= 1  # The first target is the explicit start of every preset.
    fraction = np.arange(1, count + 1, dtype=np.float32) / count
    offsets = np.zeros((count, 3), dtype=np.float32)
    extent = np.float32(0.08)
    if pattern in ("Straight line", "Arc"):
        offsets[:, 1] = 0.08 * fraction
        if pattern == "Arc":
            offsets[:, 2] = 0.03 * np.sin(np.pi * fraction)
            offsets[-1, 2] = 0
    elif pattern == "Circle":
        angle = np.float32(2 * np.pi) * fraction
        offsets[:, 1] = 0.04 * np.sin(angle)
        offsets[:, 2] = 0.04 * (1 - np.cos(angle))
        offsets[-1] = 0
    else:
        anchors = np.asarray(
            {
                "Crane pick/place": [
                    [0, 0, 0],
                    [0, 0, 0.06],
                    [0, 0.10, 0.06],
                    [0, 0.10, 0],
                ],
                "Front to back": [
                    [0, 0, 0],
                    [0.08, 0, 0],
                    [0.08, 0, 0.06],
                    [-0.08, 0, 0.06],
                    [-0.08, 0, 0],
                ],
                "Zigzag": [
                    [0, 0, 0],
                    [0, 0.025, 0.025],
                    [0, 0.05, -0.025],
                    [0, 0.075, 0.025],
                    [0, 0.10, 0],
                ],
            }[pattern],
            dtype=np.float32,
        )
        segments = len(anchors) - 1
        extent = np.ptp(anchors, axis=0).max()
        knots = np.linspace(0, count, len(anchors), dtype=np.float32)
        if count >= segments:
            knots = np.rint(knots)
        query = np.arange(1, count + 1, dtype=np.float32)
        segment = np.searchsorted(knots, query).clip(1, segments) - 1
        blend = ((query - knots[segment]) / (knots[segment + 1] - knots[segment]))[:, None]
        offsets = anchors[segment] + blend * (anchors[segment + 1] - anchors[segment])
    return np.concatenate(
        (
            np.zeros((1, 3), dtype=np.float32),
            offsets * np.float32(_PRESET_BBOX_SIZE / extent),
        )
    )


def _transform_waypoint_poses(
    position: np.ndarray,
    quaternion: np.ndarray,
    group_position: np.ndarray,
    group_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the displayed group transform with local waypoint poses ``[K,L,3/4]``."""
    from viser.transforms import SO3

    rotation = SO3(np.asarray(group_wxyz, dtype=np.float32)).normalize()
    world_position = rotation.apply(position) + np.asarray(group_position, dtype=np.float32)
    world_quaternion = rotation.multiply(SO3(quaternion)).wxyz
    return world_position.astype(np.float32), world_quaternion.astype(np.float32)


class WaypointDemo:
    """Own the viewer and run planning/playback on the main CUDA thread."""

    def __init__(
        self,
        *,
        robot: str = "franka.yml",
        scene: str = "collision_table.yml",
        port: int = 8080,
        count: int = 32,
        num_knots: int = 32,
    ) -> None:
        """Create a graph-enabled planner and numbered pose handles for each tool."""
        self._robot = robot
        self._scene = scene
        self.planner = self._create_planner(num_knots)
        max_count = min(256, self.planner.config.spline_layout.reference_intervals + 1)
        if not 1 <= count <= max_count:
            self.planner.destroy()
            raise ValueError(f"This spline supports 1 to {max_count} displayed poses; got {count}")
        self.visualizer = ViserVisualizer(
            ContentPath(robot_config_file=robot),
            add_robot_to_scene=True,
            add_control_frames=False,
            connect_port=port,
        )
        self.server = self.visualizer._server
        self.visualizer.add_scene(self.planner.config.scene_collision_cfg.scene_model)
        self.start = JointState.from_position(
            self.planner.default_joint_state.position.view(1, -1).clone(),
            joint_names=self.planner.joint_names,
        )
        start_pose = self.planner.compute_kinematics(self.start).tool_poses
        self.start_position = start_pose.position[0, 0].cpu().numpy().copy()
        self.start_quaternion = start_pose.quaternion[0, 0].cpu().numpy().copy()
        self._initial_group_position = self.start_position.mean(axis=0) + np.asarray(
            _DEFAULT_PATH_OFFSET, dtype=np.float32
        )
        self.path_start: JointState | None = None
        self._start_position_error = 0.0
        self._start_rotation_error = 0.0
        self._ik_pose_criteria = {
            frame: ToolPoseCriteria.track_pose(
                non_terminal_scale=0.0,
                device_cfg=self.planner.device_cfg,
            )
            for frame in self.planner.tool_frames
        }
        self._ik_position_criteria = {
            frame: ToolPoseCriteria.track_position(device_cfg=self.planner.device_cfg)
            for frame in self.planner.tool_frames
        }
        self.server.initial_camera.position = (1.05, 0.9, 0.9)
        self.server.initial_camera.look_at = self._initial_group_position
        self.visualizer.set_joint_state(self.start.squeeze(0))
        self.handles: list[list[viser.TransformControlsHandle]] = []
        self.labels: list[viser.LabelHandle] = []
        self.markers: list[viser.MeshHandle] = []
        self.reference_lines: list[viser.LineSegmentsHandle] = []
        self.planned_lines: list[viser.LineSegmentsHandle] = []
        self.trajectory: JointState | None = None
        self.duration = 0.0
        self._commands: queue.Queue[str] = queue.Queue(maxsize=1)
        self._edited = Event()
        self._warming_up = False
        self._pattern = "Arc"
        self._applied_scale = _DEFAULT_PATH_SCALE
        self._last_tick = time.monotonic()
        self._last_preview = 0.0
        self._waypoint_frame = self.server.scene.add_frame(
            "/waypoint_group",
            position=self._initial_group_position.copy(),
            show_axes=False,
        )
        self.waypoint_group = self.server.scene.add_transform_controls(
            "/waypoint_group_control",
            position=self._initial_group_position.copy(),
            scale=0.14,
            visible=False,
            depth_test=False,
        )
        self.waypoint_group.on_update(self._mark_edited)
        with self.server.gui.add_folder("Waypoints"):
            self.server.gui.add_markdown(
                "Choose **Edit waypoint** or click a marker, drag its handle, "
                "then click **Plan**. The last waypoint is the goal.\n\n"
                "**Position + orientation:** edit and reach each pose. "
                "**Position only:** rotations are ignored, including at the goal.\n\n"
                "**Orange:** requested polyline. **Green:** planned path. "
                "The first waypoint is the start: Plan solves its IK before planning the path."
            )
            self.knots = self.server.gui.add_dropdown(
                "Spline knots",
                options=tuple(
                    str(value) for value in sorted({16, 32, 64, 128, 256, 512, 1024, num_knots})
                ),
                initial_value=str(num_knots),
                hint="Changing this rebuilds and warms up the planner; it may take a few seconds.",
            )
            self.count = self.server.gui.add_slider(
                "Waypoint count",
                min=1,
                max=max_count,
                step=1,
                initial_value=count,
                hint="Includes the start pose. Changing count regenerates the selected preset.",
            )
            self.tracking_mode = self.server.gui.add_dropdown(
                "Waypoint tracking",
                options=("Position + orientation", "Position only"),
                initial_value="Position + orientation",
            )
            self.selected = self.server.gui.add_dropdown(
                "Edit waypoint",
                options=tuple(str(i + 1) for i in range(count)),
                initial_value="1",
            )
            self.move_group = self.server.gui.add_checkbox(
                "Move whole path",
                initial_value=False,
                hint="Translate or rotate every target together, including the path start.",
            )
            self.plan_button = self.server.gui.add_button("Plan", color="green", disabled=True)
            self.reset_button = self.server.gui.add_button("Reset waypoints")
            self.status = self.server.gui.add_markdown("Drag waypoints, then click **Plan**.")
            self.motion_time = self.server.gui.add_text(
                "Motion time",
                initial_value="--",
                disabled=True,
            )
            self.planning_time = self.server.gui.add_text(
                "Planning time",
                initial_value="--",
                disabled=True,
            )
            self.errors = self.server.gui.add_text(
                "Native max error",
                initial_value="--",
                disabled=True,
            )
        with self.server.gui.add_folder("Path presets"):
            self.server.gui.add_markdown(
                "Load a preset, then click **Plan**. Small counts increase to retain the pattern. "
                "At scale 1, the largest dimension is **0.25 m**. "
                "Front/back follow world X; use the group handle to change direction."
            )
            self.path_scale = self.server.gui.add_slider(
                "Path scale",
                min=0.25,
                max=2.0,
                step=0.05,
                initial_value=_DEFAULT_PATH_SCALE,
                hint="Resize about the first waypoint, preserving edits and group placement.",
            )
            self.pattern_buttons = []
            for pattern in _PATTERN_MIN_COUNTS:
                button = self.server.gui.add_button(pattern)
                button.on_click(lambda _, name=pattern: self._request("pattern:" + name))
                self.pattern_buttons.append(button)
        with self.server.gui.add_folder("Tolerances", expand_by_default=False):
            self.path_position = self.server.gui.add_number(
                "Path position (cm)",
                initial_value=5.0,
                min=0.01,
                max=100.0,
                step=0.1,
            )
            self.path_rotation = self.server.gui.add_number(
                "Path orientation (deg)",
                initial_value=15.0,
                min=0.1,
                max=180.0,
                step=0.5,
            )
            self.final_position = self.server.gui.add_number(
                "Goal position (cm)",
                initial_value=0.5,
                min=0.01,
                max=100.0,
                step=0.1,
            )
            self.final_rotation = self.server.gui.add_number(
                "Goal orientation (deg)",
                initial_value=3.0,
                min=0.1,
                max=180.0,
                step=0.5,
            )
        with self.server.gui.add_folder("Playback"):
            self.play = self.server.gui.add_checkbox("Play", initial_value=False, disabled=True)
            self.progress = self.server.gui.add_slider(
                "Progress",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.0,
                disabled=True,
            )
            self.speed = self.server.gui.add_slider(
                "Speed",
                min=0.25,
                max=2.0,
                step=0.25,
                initial_value=1.0,
            )
        self.plan_button.on_click(lambda _: self._request("plan"))
        self.reset_button.on_click(lambda _: self._request("reset"))
        self.play.on_update(self._play_changed)
        self._planning_controls = (
            self.plan_button,
            self.reset_button,
            self.knots,
            self.count,
            self.tracking_mode,
            self.selected,
            self.move_group,
            *self.pattern_buttons,
            self.path_scale,
            self.path_position,
            self.path_rotation,
            self.final_position,
            self.final_rotation,
        )
        for control in (
            self.path_position,
            self.path_rotation,
            self.final_position,
            self.final_rotation,
            self.tracking_mode,
        ):
            control.on_update(self._mark_edited)
        self._reset_waypoints()
        self.warmup()

    def _create_planner(self, num_knots: int) -> MotionPlanner:
        """Create a planner with fixed controls for the current robot and scene."""
        return MotionPlanner(
            MotionPlannerCfg.create(
                robot=self._robot,
                num_knots=num_knots,
                scene_model=self._scene,
                use_cuda_graph=True,
            )
        )

    def _rebuild_planner(self, num_knots: int) -> None:
        """Warm a replacement on the main thread, retaining the current planner on failure."""
        previous = self.planner
        layout = replace(previous.config.spline_layout, num_knots=num_knots)
        max_count = min(256, layout.reference_intervals + 1)
        if len(self.handles) > max_count:
            self.knots.value = str(previous.config.spline_layout.num_knots)
            self.status.content = (
                f"{num_knots} knots support at most {max_count} displayed poses, including start. "
                "Reduce the waypoint count first."
            )
            return
        replacement = None
        self._set_busy(True)
        self._invalidate()
        self.status.content = f"Applying {num_knots} spline knots… Warming up…"
        try:
            replacement = self._create_planner(num_knots)
            self.planner = replacement
            if not self.warmup():
                raise RuntimeError("Warmup did not succeed in both tracking modes")
            self.count.max = max_count
            previous.destroy()
        except Exception as error:
            self.planner = previous
            if replacement is not None:
                replacement.destroy()
            self.knots.value = str(previous.config.spline_layout.num_knots)
            self.status.content = f"Could not change spline knots: {error}"
        finally:
            self._set_busy(False)

    @property
    def track_orientation(self) -> bool:
        """Return whether planning should constrain each waypoint's rotation."""
        return self.tracking_mode.value == "Position + orientation"

    def _set_busy(self, busy: bool) -> None:
        """Freeze edits during planning and keep orientation controls off in position-only mode."""
        for control in self._planning_controls:
            control.disabled = busy
        self.path_rotation.disabled = self.final_rotation.disabled = (
            busy or not self.track_orientation
        )
        for row in self.handles:
            for handle in row:
                handle.disable_axes = handle.disable_sliders = busy
                handle.disable_rotations = busy or not self.track_orientation
        self.waypoint_group.disable_axes = self.waypoint_group.disable_sliders = busy
        self.waypoint_group.disable_rotations = busy

    def warmup(self) -> bool:
        """Warm both modes on a reachable path, preserving all displayed pose edits."""
        initial_mode = self.tracking_mode.value
        initial_scale = self.path_scale.value
        initial_selection = self.selected.value
        initial_position = np.array(
            [[h.position for h in row] for row in self.handles], dtype=np.float32
        )
        initial_quaternion = np.array(
            [[h.wxyz for h in row] for row in self.handles], dtype=np.float32
        )
        group_position = np.asarray(self.waypoint_group.position).copy()
        group_quaternion = np.asarray(self.waypoint_group.wxyz).copy()
        initial_exit_early = self.planner.ik_solver.config.exit_early
        self._warming_up = True
        failures = []
        try:
            # Capture IK optimization even when the seed solver already succeeds.
            self.planner.ik_solver.config.exit_early = False
            # Capture on a short reachable path independently of the displayed preset size.
            self.waypoint_group.position = self.start_position.mean(axis=0)
            self.waypoint_group.wxyz = (1.0, 0.0, 0.0, 0.0)
            self.path_scale.value = min(initial_scale, _WARMUP_PATH_SCALE)
            self._reset_waypoints()
            for mode in self.tracking_mode.options:
                self.tracking_mode.value = mode
                if not self.plan(warming_up=True):
                    failures.append(mode)
            torch.cuda.synchronize(self.planner.device_cfg.device)
            self.planner.reset_seed()
        finally:
            self.planner.ik_solver.config.exit_early = initial_exit_early
            self.tracking_mode.value = initial_mode
            self.path_scale.value = initial_scale
            self.waypoint_group.position = group_position
            self.waypoint_group.wxyz = group_quaternion
            self._reset_waypoints(positions=initial_position, quaternions=initial_quaternion)
            self.selected.value = initial_selection
            self._invalidate()
            self._edited.clear()
            self._set_busy(False)
            self._warming_up = False
        self.status.content = (
            f"Warmup incomplete for {', '.join(failures)}. Adjust waypoints and click **Plan**."
            if failures
            else "Ready. Drag waypoints, then click **Plan**."
        )
        return not failures

    def _request(self, command: str) -> None:
        """Queue a GUI action without using CUDA from a Viser callback thread."""
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            pass

    def _mark_edited(self, _: object) -> None:
        """Ask the main thread to discard the previous result and redraw the reference."""
        if not self._warming_up:
            self._edited.set()

    def _play_changed(self, _: object) -> None:
        """Restart playback when Play is pressed at the end of the timeline."""
        if self.play.value and self.progress.value >= 1.0:
            self.progress.value = 0.0

    def _reset_waypoints(
        self,
        *,
        positions: np.ndarray | None = None,
        quaternions: np.ndarray | None = None,
    ) -> None:
        """Recreate local handles from the preset or saved poses ``[count,links,3/4]``."""
        for label in self.labels:
            label.remove()
        for marker in self.markers:
            marker.remove()
        for row in self.handles:
            for handle in row:
                handle.remove()
        self.handles.clear()
        self.labels.clear()
        self.markers.clear()
        count = int(self.count.value)
        self.selected.options = tuple(str(i + 1) for i in range(count))
        self.selected.value = "1"
        origin = self.start_position.mean(axis=0)
        offsets = _pattern_offsets(self._pattern, count) * np.float32(self.path_scale.value)
        self._applied_scale = float(self.path_scale.value)
        for index, offset in enumerate(offsets):
            row = []
            for link, frame in enumerate(self.planner.tool_frames):
                name = f"/waypoint_group/waypoints/{index}/{link}"
                local_position = (
                    self.start_position[link] - origin + offset
                    if positions is None
                    else positions[index, link]
                )
                local_quaternion = (
                    self.start_quaternion[link]
                    if quaternions is None
                    else quaternions[index, link]
                )
                handle = self.server.scene.add_transform_controls(
                    name,
                    position=local_position,
                    wxyz=local_quaternion,
                    scale=0.07,
                    disable_rotations=not self.track_orientation,
                    visible=index == 0 and not self.move_group.value,
                    depth_test=False,
                )
                handle.on_update(self._mark_edited)
                row.append(handle)
                suffix = (
                    " (start/goal)"
                    if count == 1
                    else (" (start)" if index == 0 else " (goal)" if index == count - 1 else "")
                )
                text = f"{index + 1}" + suffix
                if len(self.planner.tool_frames) > 1:
                    text += f" · {frame}"
                self.labels.append(
                    self.server.scene.add_label(
                        f"/waypoint_group/labels/{index}/{link}",
                        text,
                        position=local_position + (0.0, 0.0, 0.025),
                    )
                )
                marker = self.server.scene.add_icosphere(
                    f"/waypoint_group/markers/{index}/{link}",
                    radius=0.004,
                    position=handle.position,
                    color=(255, 170, 50),
                )
                marker.on_click(lambda _, i=index: setattr(self.selected, "value", str(i + 1)))
                self.markers.append(marker)
            self.handles.append(row)
        self._edited.set()

    def _load_pattern(self, pattern: str) -> None:
        """Replace targets while preserving tracking mode and the group's placement."""
        self._pattern = pattern
        self.count.value = max(int(self.count.value), _PATTERN_MIN_COUNTS[pattern])
        self._reset_waypoints()
        self._edited.clear()
        self._invalidate()
        self.status.content = f"Loaded **{pattern}**. Adjust the path, then click **Plan**."

    def _waypoints(self) -> GoalToolPose:
        """Copy displayed poses into float32 planner targets ``[1,K,L,1,3/4]``."""
        position = np.array([[h.position for h in row] for row in self.handles], dtype=np.float32)
        quaternion = np.array([[h.wxyz for h in row] for row in self.handles], dtype=np.float32)
        position, quaternion = _transform_waypoint_poses(
            position,
            quaternion,
            self.waypoint_group.position,
            self.waypoint_group.wxyz,
        )
        return GoalToolPose(
            self.planner.tool_frames,
            self.planner.device_cfg.to_device(position)[None, :, :, None].contiguous(),
            self.planner.device_cfg.to_device(quaternion)[None, :, :, None].contiguous(),
        )

    def _draw_paths(self, positions: np.ndarray, *, planned: bool) -> None:
        """Display polylines ``[samples,links,3]`` for the reference or accepted output."""
        handles = self.planned_lines if planned else self.reference_lines
        for handle in handles:
            handle.remove()
        handles.clear()
        group = "planned" if planned else "reference"
        color = (65, 210, 110) if planned else (255, 170, 50)
        if positions.shape[0] < 2:
            return
        for link in range(positions.shape[1]):
            points = np.stack((positions[:-1, link], positions[1:, link]), axis=1)
            handles.append(
                self.server.scene.add_line_segments(
                    f"/{group}/{link}",
                    points=points,
                    colors=color,
                    thickness=3.0,
                    thickness_units="screen",
                )
            )

    def _invalidate(self) -> None:
        """Remove stale playback and redraw the requested curve after an edit."""
        self.trajectory = None
        self.path_start = None
        self.play.value = False
        self.play.disabled = self.progress.disabled = True
        self.progress.value = 0.0
        for handle in self.planned_lines:
            handle.remove()
        self.planned_lines.clear()
        self.visualizer.set_joint_state(self.start.squeeze(0))
        waypoints = self._waypoints()
        self._draw_paths(waypoints.position[0, :, :, 0].cpu().numpy(), planned=False)
        self.status.content = "Waypoints or tolerances changed; click **Plan**."
        self.motion_time.value = self.planning_time.value = "--"
        self.errors.value = "--"

    def _synchronize(self) -> None:
        """Apply current GUI edits on the main thread, coalescing continuous drag updates."""
        count_changed = int(self.count.value) != len(self.handles)
        if count_changed:
            self._reset_waypoints()
        scale = float(self.path_scale.value)
        if scale != self._applied_scale:
            factor = np.float32(scale / self._applied_scale)
            anchor = np.array([h.position for h in self.handles[0]], dtype=np.float32)
            for row in self.handles:
                for link, handle in enumerate(row):
                    local_position = np.asarray(handle.position, dtype=np.float32)
                    handle.position = anchor[link] + factor * (local_position - anchor[link])
            self._applied_scale = scale
            self._edited.set()
            self._last_preview = 0.0
        locked = not self.track_orientation
        self._waypoint_frame.position = self.waypoint_group.position
        self._waypoint_frame.wxyz = self.waypoint_group.wxyz
        self.waypoint_group.visible = bool(self.move_group.value)
        self.path_rotation.disabled = self.final_rotation.disabled = locked
        for index, row in enumerate(self.handles):
            for link, handle in enumerate(row):
                handle.disable_rotations = locked
                handle.visible = not self.move_group.value and index + 1 == int(
                    self.selected.value
                )
                flat_index = index * len(self.planner.tool_frames) + link
                self.markers[flat_index].position = handle.position
                self.labels[flat_index].position = np.asarray(handle.position) + (0.0, 0.0, 0.025)
                self.labels[flat_index].visible = (
                    index + 1 == int(self.selected.value) or index == len(self.handles) - 1
                )
        now = time.monotonic()
        if self._edited.is_set() and (count_changed or now - self._last_preview > 0.05):
            self._edited.clear()
            self._invalidate()
            self._last_preview = now
        requested_knots = int(self.knots.value)
        if requested_knots != self.planner.config.spline_layout.num_knots:
            self._rebuild_planner(requested_knots)

    def _solve_start(self, waypoints: GoalToolPose) -> JointState | None:
        """Solve and validate first-target IK, with the same seed layout as terminal IK."""
        goal = GoalToolPose(
            waypoints.tool_frames,
            waypoints.position[:, :1].contiguous(),
            waypoints.quaternion[:, :1].contiguous(),
        )
        criteria = self._ik_pose_criteria if self.track_orientation else self._ik_position_criteria
        self.planner.ik_solver.update_tool_pose_criteria(criteria)
        try:
            ik = self.planner.ik_solver.solve_pose(
                goal,
                current_state=self.start.clone(),
                return_seeds=self.planner.trajopt_solver.config.num_seeds,
            )
        finally:
            self.planner.ik_solver.update_tool_pose_criteria(self._ik_pose_criteria)
        solutions = ik.solution[0]
        poses = self.planner.compute_kinematics(
            JointState.from_position(
                solutions,
                joint_names=self.planner.joint_names,
            )
        ).tool_poses
        position_error = (poses.position[:, 0] - goal.position[0, 0, :, 0]).norm(dim=-1)
        rotation_error = torch.zeros_like(position_error)
        if self.track_orientation:
            current = poses.quaternion[:, 0]
            target = goal.quaternion[0, 0, :, 0].unsqueeze(0)
            current = current / current.norm(dim=-1, keepdim=True)
            target = target / target.norm(dim=-1, keepdim=True)
            vector = (
                target[..., :1] * current[..., 1:]
                - current[..., :1] * target[..., 1:]
                - torch.linalg.cross(current[..., 1:], target[..., 1:].expand_as(current[..., 1:]))
            )
            rotation_error = 2 * torch.atan2(
                vector.norm(dim=-1),
                (current * target).sum(dim=-1).abs(),
            )
        position_bound = self.path_position.value / 100
        rotation_bound = math.radians(self.path_rotation.value)
        if waypoints.horizon == 1:
            position_bound = min(position_bound, self.final_position.value / 100)
            rotation_bound = min(rotation_bound, math.radians(self.final_rotation.value))
        valid = ik.success[0] & (position_error <= position_bound).all(dim=-1)
        valid = valid & (rotation_error <= rotation_bound).all(dim=-1)
        if not bool(valid.any()):
            return None
        index = int(valid.to(torch.int32).argmax().item())
        self._start_position_error = float(position_error[index].max().item())
        self._start_rotation_error = float(rotation_error[index].max().item())
        return JointState.from_position(
            solutions[index : index + 1].clone(),
            joint_names=self.planner.joint_names,
        )

    def plan(self, *, warming_up: bool = False) -> bool:
        """Plan from a stationary start and expose playback only for a validated result."""
        self._edited.clear()
        self._invalidate()
        self._set_busy(True)
        self.status.content = "Warming up…" if warming_up else "Planning…"
        try:
            torch.cuda.synchronize(self.planner.device_cfg.device)
            planning_started = time.perf_counter()
            try:
                waypoints = self._waypoints()
                self.path_start = self._solve_start(waypoints)
                if self.path_start is None:
                    self.status.content = "No valid start IK; move or rotate waypoint 1."
                    return False
                # One target is a stationary request at its separately solved start pose.
                first = 1 if waypoints.horizon > 1 else 0
                remaining = GoalToolPose(
                    waypoints.tool_frames,
                    waypoints.position[:, first:].contiguous(),
                    waypoints.quaternion[:, first:].contiguous(),
                )
                result = self.planner.plan_pose_waypoints(
                    remaining,
                    self.path_start,
                    path_tolerance=(
                        self.path_position.value / 100,
                        math.radians(self.path_rotation.value),
                    ),
                    final_tolerance=(
                        self.final_position.value / 100,
                        math.radians(self.final_rotation.value),
                    ),
                    max_attempts=3,
                    outer_iterations=4,
                    track_orientation=self.track_orientation,
                )
            finally:
                torch.cuda.synchronize(self.planner.device_cfg.device)
                planning_ms = 1000 * (time.perf_counter() - planning_started)
                self.planning_time.value = f"{planning_ms:.1f} ms"
            self.visualizer.set_joint_state(self.path_start.squeeze(0))
            if result is None:
                self.status.content = "No goal IK solution; move the goal or change its rotation."
                return False
            max_position_error = max(
                self._start_position_error,
                result.path_position_error.max().item(),
            )
            self.errors.value = f"{100 * max_position_error:.2f} cm"
            if self.track_orientation:
                max_rotation_error = max(
                    self._start_rotation_error,
                    result.path_rotation_error.max().item(),
                )
                self.errors.value += f" / {math.degrees(max_rotation_error):.2f} deg"
            if not bool(result.success.all()):
                descriptions = {
                    "path_valid": "path tracking",
                    "collision_valid": "collision",
                    "limits_and_dynamics_valid": "joint limits or dynamics",
                }
                failed = [
                    descriptions[name]
                    for name, valid in result.debug_info["validation"].items()
                    if not bool(valid.all())
                ]
                detail = f": {', '.join(failed)}" if failed else ""
                self.status.content = "Planning failed" + detail
                return False
            count = int(result.interpolated_last_tstep[0, 0].item())
            self.trajectory = JointState.from_position(
                result.interpolated_trajectory.position[0, 0, :count].clone(),
                joint_names=result.interpolated_trajectory.joint_names,
            )
            self.duration = float(result.motion_time()[0, 0].item())
            self.motion_time.value = f"{self.duration:.2f} s"
            active = self.planner.kinematics.get_active_js(self.trajectory)
            tool_positions = self.planner.compute_kinematics(active).tool_poses.position
            self._draw_paths(tool_positions[:, 0].cpu().numpy(), planned=True)
            self.progress.disabled = self.play.disabled = False
            self.progress.value = 0.0
            self.play.value = True
            self._last_tick = time.monotonic()
            self.status.content = "**Success**"
            return True
        except Exception as error:
            self.trajectory = None
            self.status.content = f"Planning failed: {error}"
            return False
        finally:
            self._set_busy(warming_up)

    def step(self) -> None:
        """Process one queued action, apply edits, and update preview playback."""
        self._synchronize()
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            command = None
        if command == "plan":
            self.plan()
        elif command == "reset":
            self.waypoint_group.position = self._initial_group_position.copy()
            self.waypoint_group.wxyz = (1.0, 0.0, 0.0, 0.0)
            self._reset_waypoints()
            self._edited.clear()
            self._invalidate()
        elif command is not None and command.startswith("pattern:"):
            self._load_pattern(command.removeprefix("pattern:"))
        now = time.monotonic()
        elapsed, self._last_tick = now - self._last_tick, now
        if self.trajectory is None:
            return
        if self.play.value:
            self.progress.value = min(
                1.0,
                self.progress.value + elapsed * self.speed.value / max(self.duration, 1e-6),
            )
            if self.progress.value >= 1.0:
                self.play.value = False
        index = round(self.progress.value * (self.trajectory.position.shape[0] - 1))
        self.visualizer.set_joint_state(self.trajectory[index])

    def close(self) -> None:
        """Release CUDA graphs and stop the viewer server."""
        try:
            self.planner.destroy()
        finally:
            self.server.stop()


def main() -> None:
    """Launch the interactive viewer or check the example without a browser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--scene", default="collision_table.yml")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--num-knots",
        type=int,
        default=32,
        choices=range(12, 1025),
        metavar="N",
        help="Initial spline controls (12..1024; default 32)",
    )
    parser.add_argument("--waypoints", type=int, default=32, choices=range(1, 257))
    parser.add_argument("--test", action="store_true", help="Run a one-shot server/planner check")
    args = parser.parse_args()
    demo = WaypointDemo(
        robot=args.robot,
        scene=args.scene,
        port=0 if args.test else args.port,
        count=args.waypoints,
        num_knots=args.num_knots,
    )
    try:
        if args.test:
            demo._synchronize()
            assert not demo.plan_button.disabled and demo.status.content.startswith("Ready")
            assert demo.trajectory is None and not demo.planned_lines
            assert demo.path_scale.value == _DEFAULT_PATH_SCALE
            demo.path_scale.value = _WARMUP_PATH_SCALE
            demo._synchronize()
            assert all(not h.disable_rotations for row in demo.handles for h in row)
            handle = demo.handles[-1][0]
            original = np.asarray(handle.wxyz).copy()
            # Position-only mode ignores an otherwise unreachable requested rotation.
            handle.wxyz = np.array([math.cos(0.02), 0.0, 0.0, math.sin(0.02)])
            edited = np.asarray(handle.wxyz).copy()
            demo.tracking_mode.value = "Position only"
            demo._synchronize()
            assert handle.disable_rotations
            np.testing.assert_array_equal(handle.wxyz, edited)
            handle.position = np.asarray(handle.position) + np.array([0.0, 0.005, 0.0])
            goal = demo._waypoints()
            np.testing.assert_allclose(
                goal.position[0, -1, 0, 0].cpu(),
                np.asarray(handle.position) + demo.waypoint_group.position,
            )
            assert demo.plan(), demo.status.content
            assert demo.trajectory is not None
            demo.step()
            demo.tracking_mode.value = "Position + orientation"
            handle.wxyz = original
            assert demo.plan(), demo.status.content
            # New counts invalidate playback; selecting a target reveals only its handle.
            changed_counts = (32, 1) if int(demo.count.value) == 1 else (1, 32)
            for count in changed_counts:
                demo.count.value = count
                demo._synchronize()
                assert len(demo.handles) == count
                assert demo.trajectory is None and demo.play.disabled
                demo.selected.value = str(count)
                demo._synchronize()
                assert sum(h.visible for row in demo.handles for h in row) == len(
                    demo.planner.tool_frames
                )
            demo._request("reset")
            demo.step()
            assert demo.trajectory is None and not demo.planned_lines
            print("Cartesian waypoint example passed")
            return
        print(f"Open http://localhost:{demo.server.get_port()} · Ctrl+C to stop")
        while True:
            demo.step()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()


if __name__ == "__main__":
    main()
