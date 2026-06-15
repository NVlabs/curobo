# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fuse live RGB-D frames into a cuRobo volumetric map and MPC scene.

This reference example uses an Intel RealSense backend for live RGB-D capture:
it aligns depth to the RGB stream, converts each frame into
:class:`~curobo.types.CameraObservation`, and integrates the observations into
:class:`~curobo.perception.Mapper`. The script is intended as a live mapping
and control demo: it updates the TSDF continuously, opens a Viser preview of
the reconstruction, recomputes an ESDF slice at the viewer refresh cadence,
and runs a Franka MPC loop with a draggable target against the live ESDF
collision grid. It does not write rendered PNGs or meshes.

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline controls style="width:100%;border-radius:6px;">
       <source src="../videos/mpc_live_target.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">Live volumetric mapping MPC target control</figcaption>
   </figure>

The dynamic-obstacle clip shows the robot reacting to obstacles introduced into
the live depth scene.

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline controls style="width:100%;border-radius:6px;">
       <source src="../videos/mpc_live_dynamic.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">Live MPC reacting to dynamic obstacles</figcaption>
   </figure>

Install the RealSense Python bindings before running the example:

.. code-block:: bash

   uv pip install pyrealsense2

Run live fixed-camera mapping until ``Ctrl-C``. The example opens an
interactive Viser point-cloud preview while integrating:

.. code-block:: bash

   python -m curobo.examples.reference.live_volumetric_mapping_mpc

The current RealSense backend provides RGB-D frames and intrinsics, but it does
not provide a camera-to-world pose. This example applies fixed -90 degree
X-axis and +90 degree Y-axis camera-frame rotations, then uses the first valid
depth frame to translate the camera so that the initial point cloud mean lands
in front of the robot base. Use ``--camera-pose x y z qw qx qy qz`` to set a
base transform before this fixed rotation and first-frame offset.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.profiling import CudaEventTimer
from curobo.scene import Scene, VoxelGrid
from curobo.types import (
    CameraObservation,
    ContentPath,
    DeviceCfg,
    GoalToolPose,
    JointState,
    Pose,
)
from curobo.viewer import ViserVisualizer


MPC_ROBOT_FILE = "franka.yml"
MPC_TARGET_OFFSET_M = 0.20
FIRST_FRAME_POINTCLOUD_CENTER_M = (0.6, 0.0, 0.2)
DEPTH_MAX_M = 1.0


def _import_realsense():
    """Import pyrealsense2 with an actionable error for optional dependency users."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is required for this example. Install it with:\n"
            "  uv pip install pyrealsense2"
        ) from exc
    return rs


def _pose_from_cli(values: Sequence[float], device: str) -> Pose:
    """Create a cuRobo Pose from ``x y z qw qx qy qz`` CLI values."""
    if len(values) != 7:
        raise ValueError("--camera-pose expects 7 values: x y z qw qx qy qz")
    return Pose.from_list(
        list(values),
        device_cfg=DeviceCfg(device=torch.device(device), dtype=torch.float32),
    )


def _multiply_quaternions(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _apply_camera_rotation_correction(base_pose: Pose, device: str) -> Pose:
    """Apply fixed -90 degree X and +90 degree Y camera-frame corrections."""
    correction_x = torch.tensor(
        [[0.70710678, -0.70710678, 0.0, 0.0]],
        dtype=torch.float32,
        device=torch.device(device),
    )
    correction_y = torch.tensor(
        [[0.70710678, 0.0, 0.70710678, 0.0]],
        dtype=torch.float32,
        device=torch.device(device),
    )
    quaternion = _multiply_quaternions(base_pose.quaternion, correction_x)
    quaternion = _multiply_quaternions(quaternion, correction_y)
    quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    return Pose(position=base_pose.position.clone(), quaternion=quaternion)


def _compute_valid_pointcloud_mean(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_min: float,
    depth_max: float,
) -> torch.Tensor:
    """Compute the camera-frame mean of valid depth points."""
    valid = (
        torch.isfinite(depth)
        & (depth >= depth_min)
        & (depth <= depth_max)
    )
    if not torch.any(valid):
        raise ValueError("Cannot mean-center: first frame has no valid depth pixels.")

    height, width = depth.shape
    u = torch.arange(width, dtype=torch.float32, device=depth.device)
    v = torch.arange(height, dtype=torch.float32, device=depth.device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    z = depth[valid]
    x = (uu[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (vv[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return torch.stack((x, y, z), dim=-1).mean(dim=0)


def _center_pose_from_first_frame(
    pose: Pose,
    observation: CameraObservation,
    depth_min: float,
    depth_max: float,
) -> tuple[Pose, torch.Tensor, torch.Tensor]:
    """Translate pose so the first frame's rotated point cloud mean is offset."""
    mean_camera = _compute_valid_pointcloud_mean(
        observation.depth_image,
        observation.intrinsics,
        depth_min,
        depth_max,
    )
    rotation = pose.get_rotation_matrix()[0]
    mean_world_offset = rotation @ mean_camera
    target_center = torch.tensor(
        FIRST_FRAME_POINTCLOUD_CENTER_M,
        dtype=pose.position.dtype,
        device=pose.position.device,
    )

    centered_pose = pose.clone()
    centered_pose.position = target_center.view(1, 3) - mean_world_offset.view(1, 3)
    return centered_pose, mean_camera, target_center


def _intrinsics_to_tensor(intrinsics, device: torch.device) -> torch.Tensor:
    """Convert RealSense intrinsics to cuRobo's 3x3 camera matrix."""
    matrix = torch.eye(3, dtype=torch.float32, device=device)
    matrix[0, 0] = float(intrinsics.fx)
    matrix[1, 1] = float(intrinsics.fy)
    matrix[0, 2] = float(intrinsics.ppx)
    matrix[1, 2] = float(intrinsics.ppy)
    return matrix


class RealSenseRGBDStream:
    """RealSense RGB-D stream aligned to the color camera."""

    def __init__(self, width: int, height: int, fps: int, device: str, rs_module=None):
        self.rs = rs_module if rs_module is not None else _import_realsense()
        self.device = torch.device(device)

        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()
        self.config.enable_stream(self.rs.stream.depth, width, height, self.rs.format.z16, fps)
        self.config.enable_stream(self.rs.stream.color, width, height, self.rs.format.rgb8, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = self.rs.align(self.rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_profile = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsics()
        self.intrinsics = _intrinsics_to_tensor(color_intrinsics, self.device)
        self.image_shape = (int(color_intrinsics.height), int(color_intrinsics.width))

    def __enter__(self) -> "RealSenseRGBDStream":
        return self

    def __exit__(self, *_):
        self.stop()

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        self.pipeline.stop()

    def warmup(self, num_frames: int) -> None:
        """Wait for initial frames so RealSense auto-exposure can settle."""
        for _ in range(num_frames):
            self.pipeline.wait_for_frames()

    def wait_for_observation(
        self,
        pose: Pose,
        timeout_ms: int = 5000,
    ) -> Optional[CameraObservation]:
        """Read one aligned RGB-D frame and convert it to CameraObservation."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        depth_np = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        rgb_np = np.asanyarray(color_frame.get_data())

        depth = torch.from_numpy(np.ascontiguousarray(depth_np)).to(
            device=self.device,
            dtype=torch.float32,
        )
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(
            device=self.device,
            dtype=torch.uint8,
        )

        return CameraObservation(
            name="realsense",
            rgb_image=rgb,
            depth_image=depth,
            pose=pose,
            intrinsics=self.intrinsics,
            timestamp=torch.tensor([frames.get_timestamp() * 0.001], device=self.device),
        )


def _make_mapper(args: argparse.Namespace) -> Mapper:
    """Create a Mapper sized for the RealSense capture workspace."""
    voxel_size = float(args.voxel_size)
    config = MapperCfg(
        voxel_size=voxel_size,
        esdf_voxel_size=args.esdf_voxel_size,
        extent_meters_xyz=tuple(args.extent),
        extent_esdf_meters_xyz=tuple(args.extent),
        grid_center=torch.tensor(args.grid_center, dtype=torch.float32),
        truncation_distance=voxel_size * 4.0,
        depth_minimum_distance=args.depth_min,
        depth_maximum_distance=args.depth_max,
        minimum_tsdf_weight=args.minimum_tsdf_weight,
        decay_factor=0.3,
        roughness=args.roughness,
        num_cameras=1,
        image_height=args.height,
        image_width=args.width,
        device=args.device,
        block_size=2,
    )
    return Mapper(config)


def _batch_observation(observation: CameraObservation) -> CameraObservation:
    """Add the leading camera dimension required by Mapper.integrate."""
    return CameraObservation(
        name=observation.name,
        depth_image=observation.depth_image.unsqueeze(0),
        rgb_image=observation.rgb_image.unsqueeze(0),
        pose=observation.pose,
        intrinsics=observation.intrinsics.unsqueeze(0),
        timestamp=observation.timestamp,
    )


def _filter_depth(observation: CameraObservation, depth_filter: FilterDepth) -> CameraObservation:
    """Clean invalid depth values before TSDF integration."""
    observation.depth_image = torch.nan_to_num(
        observation.depth_image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    filtered, _ = depth_filter(observation.depth_image.unsqueeze(0))
    observation.depth_image = filtered[0]
    return observation


def _extract_esdf_slice(
    esdf_grid: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    slice_pose: np.ndarray,
    slice_width: float,
    slice_height: float,
    slice_resolution: tuple[int, int],
) -> np.ndarray:
    """Sample an ESDF image on a rectangular world-space slice plane."""
    width_px, height_px = slice_resolution
    device = esdf_grid.device
    nx, ny, nz = esdf_grid.shape

    u = torch.linspace(-slice_width / 2.0, slice_width / 2.0, width_px, device=device)
    v = torch.linspace(-slice_height / 2.0, slice_height / 2.0, height_px, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    local_points = torch.stack(
        [
            uu.flatten(),
            vv.flatten(),
            torch.zeros(width_px * height_px, device=device),
            torch.ones(width_px * height_px, device=device),
        ],
        dim=1,
    )

    pose_tensor = torch.tensor(slice_pose, dtype=torch.float32, device=device)
    world_points = (pose_tensor @ local_points.T).T[:, :3]
    local_pts = world_points - origin.to(device)
    half_extent = torch.tensor(
        [
            (nx - 1) * voxel_size / 2.0,
            (ny - 1) * voxel_size / 2.0,
            (nz - 1) * voxel_size / 2.0,
        ],
        device=device,
    )
    normalized = local_pts / half_extent

    coords = normalized[:, [2, 1, 0]]
    coords = coords.view(1, 1, height_px, width_px, 3).float()

    esdf_5d = esdf_grid.float().unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        esdf_5d,
        coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    values = sampled.squeeze().view(height_px, width_px).cpu().numpy()

    max_dist = max(np.max(values), 0.1)
    max_negative_dist = max(np.abs(np.min(values)), 0.05)
    normalized_positive = np.clip(values / max_dist, -1, 1)
    normalized_negative = np.clip(values / max_negative_dist, -1, 1)

    colors = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    neg_mask = normalized_negative < 0
    colors[neg_mask, 0] = ((1 + normalized_negative[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 1] = ((1 + normalized_negative[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 2] = 255

    pos_mask = normalized_positive >= 0
    colors[pos_mask, 0] = 255
    colors[pos_mask, 1] = ((1 - normalized_positive[pos_mask]) * 255).astype(np.uint8)
    colors[pos_mask, 2] = ((1 - normalized_positive[pos_mask]) * 255).astype(np.uint8)
    colors[np.abs(values) < voxel_size * 0.5] = [0, 255, 0]
    return colors


def _add_static_esdf_slice(
    visualizer: ViserVisualizer,
    voxel_grid,
    config: MapperCfg,
) -> None:
    """Add one full-workspace ESDF XY slice through the grid center."""
    if voxel_grid.feature_tensor is None:
        return

    device = voxel_grid.feature_tensor.device
    origin = torch.tensor(
        voxel_grid.pose[:3],
        dtype=torch.float32,
        device=device,
    )
    extent_x, extent_y, _ = config.extent_meters_xyz
    long_side = max(extent_x, extent_y)
    width_px = max(64, round(512 * extent_x / long_side))
    height_px = max(64, round(512 * extent_y / long_side))

    slice_pose = np.eye(4, dtype=np.float32)
    slice_pose[:3, 3] = origin.cpu().numpy()
    colors = _extract_esdf_slice(
        esdf_grid=voxel_grid.feature_tensor,
        origin=origin,
        voxel_size=voxel_grid.voxel_size,
        slice_pose=slice_pose,
        slice_width=extent_x,
        slice_height=extent_y,
        slice_resolution=(width_px, height_px),
    )

    pose = Pose(
        position=origin.view(1, 3),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    )
    visualizer.add_image(
        image=colors,
        render_width=extent_x,
        render_height=extent_y,
        pose=pose,
        name="/esdf/static_xy_slice",
    )


def _update_live_visuals(
    visualizer: ViserVisualizer,
    mapper: Mapper,
    observation: CameraObservation,
    rgb_image_handle,
    point_size: float,
    max_voxels: int,
) -> tuple[object, float]:
    """Refresh TSDF voxels and recompute the static full-workspace ESDF slice."""
    _update_rgb_panel(rgb_image_handle, observation)
    _update_visualizer(
        visualizer,
        mapper,
        observation,
        point_size=point_size,
        max_voxels=max_voxels,
    )
    t0 = time.perf_counter()
    voxel_grid = mapper.compute_esdf()
    esdf_time = time.perf_counter() - t0
    _add_static_esdf_slice(visualizer, voxel_grid, mapper.config)
    return voxel_grid, esdf_time


def _add_rgb_panel(visualizer: ViserVisualizer, height: int, width: int):
    """Create a GUI image panel for the live RGB stream."""
    with visualizer._server.gui.add_folder("RealSense"):
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return visualizer._server.gui.add_image(
            blank,
            label="RGB",
            format="jpeg",
            jpeg_quality=80,
        )


def _update_rgb_panel(rgb_image_handle, observation: CameraObservation) -> None:
    """Refresh the GUI RGB image."""
    if rgb_image_handle is not None:
        rgb_image_handle.image = observation.rgb_image.cpu().numpy()


class LiveEsdfMpc:
    """MPC controller with a draggable target and live ESDF collision."""

    def __init__(self, voxel_grid: VoxelGrid, device: str, visualizer: ViserVisualizer):
        self.visualizer = visualizer
        self.device_cfg = DeviceCfg(device=torch.device(device), dtype=torch.float32)
        if voxel_grid.feature_tensor is None:
            raise ValueError("Cannot initialize MPC without an ESDF feature tensor.")
        config = ModelPredictiveControlCfg.create(
            robot=MPC_ROBOT_FILE,
            scene_model=Scene(voxel=[voxel_grid]),
            device_cfg=self.device_cfg,
            use_cuda_graph=True,
            optimization_dt=0.03,
            interpolation_steps=4,
            optimizer_collision_activation_distance=0.01,
            #warm_start_optimization_num_iters=80,
            #cold_start_optimization_num_iters=120,
        )
        self.mpc = ModelPredictiveControl(config)

        self.current_state = JointState.from_position(
            self.mpc.default_joint_position.clone().unsqueeze(0),
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = torch.zeros_like(self.current_state.position)
        self.current_state.acceleration = torch.zeros_like(self.current_state.position)

        self.mpc.setup(self.current_state)
        self.target_link = self.mpc.tool_frames[0]
        self.target_control = None
        self._last_target_pose = None
        self._set_interactive_goal()
        self.visualizer.set_joint_state(self.current_state.squeeze(0))

    def _set_interactive_goal(self) -> None:
        """Create a draggable Cartesian target from the robot's initial tool pose."""
        kin_result = self.mpc.compute_kinematics(self.current_state)
        goal_poses = kin_result.tool_poses.to_dict()
        target_pose = goal_poses[self.target_link]
        target_pose.position[..., 1] += MPC_TARGET_OFFSET_M
        self.target_control = self.visualizer.add_control_frame(
            f"/mpc/target_{self.target_link}",
            target_pose,
            scale=0.12,
        )
        self._update_goal_from_target(target_pose, force=True)

    def _read_target_pose(self) -> Pose:
        """Read the target transform-control pose from Viser."""
        return Pose.from_numpy(
            np.asarray(self.target_control.position),
            np.asarray(self.target_control.wxyz),
            device_cfg=self.device_cfg,
        )

    def _update_goal_from_target(self, target_pose: Pose, force: bool = False) -> bool:
        """Push the Viser target pose into the MPC goal if it changed."""
        if (
            not force
            and self._last_target_pose is not None
            and target_pose == self._last_target_pose
        ):
            return False

        kin_result = self.mpc.compute_kinematics(self.current_state)
        goal_poses = kin_result.tool_poses.to_dict()
        goal_poses[self.target_link] = target_pose
        goal_tool_poses = GoalToolPose.from_poses(
            goal_poses,
            ordered_tool_frames=self.mpc.tool_frames,
            num_goalset=1,
        )
        updated = self.mpc.update_goal_tool_poses(goal_tool_poses, run_ik=False)
        if updated:
            self._last_target_pose = target_pose.clone()
        return updated

    def step(self):
        """Advance MPC by one action using the aliased live ESDF tensor."""
        self._update_goal_from_target(self._read_target_pose())
        result = self.mpc.optimize_action_sequence(self.current_state)
        if result.action_sequence is None or result.action_sequence.position.shape[1] == 0:
            return result

        next_position = result.action_sequence.position[:, -1, :]
        self.current_state = JointState.from_position(
            next_position.clone(),
            joint_names=self.mpc.joint_names,
        )
        self.current_state.velocity = result.action_sequence.velocity[:, -1, :]
        self.current_state.acceleration = result.action_sequence.acceleration[:, -1, :]
        self.visualizer.set_joint_state(self.current_state.squeeze(0))
        return result


def _add_extent_box(visualizer: ViserVisualizer, config: MapperCfg) -> None:
    """Draw the mapper's configured voxel bounds in Viser."""
    min_corner, max_corner = config.get_grid_bounds()
    min_c = np.array(min_corner, dtype=np.float32)
    max_c = np.array(max_corner, dtype=np.float32)

    corners = np.array(
        [
            [min_c[0], min_c[1], min_c[2]],
            [max_c[0], min_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]],
            [min_c[0], max_c[1], min_c[2]],
            [min_c[0], min_c[1], max_c[2]],
            [max_c[0], min_c[1], max_c[2]],
            [max_c[0], max_c[1], max_c[2]],
            [min_c[0], max_c[1], max_c[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    lines = np.array([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)
    visualizer._server.scene.add_line_segments(
        "/extent_box",
        points=lines,
        colors=np.array([255, 255, 0], dtype=np.uint8),
        line_width=3.0,
    )


def _update_visualizer(
    visualizer: ViserVisualizer,
    mapper: Mapper,
    observation: CameraObservation,
    point_size: float,
    max_voxels: int,
) -> None:
    """Refresh the Viser reconstruction point cloud."""
    voxels = mapper.extract_occupied_voxels(surface_only=False)
    if len(voxels) > 0:
        centers = voxels.centers
        colors = voxels.colors_uint8()
        if len(centers) > max_voxels:
            stride = max(1, int(len(centers) / max_voxels))
            centers = centers[::stride]
            colors = colors[::stride]

        visualizer.add_point_cloud(
            pointcloud=centers.cpu().numpy(),
            colors=colors.cpu().numpy(),
            point_size=point_size,
            name="/reconstruction",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live volumetric mapping with MPC")
    parser.add_argument("--width", type=int, default=640, help="RGB-D stream width")
    parser.add_argument("--height", type=int, default=480, help="RGB-D stream height")
    parser.add_argument("--fps", type=int, default=30, help="RGB-D stream frame rate")
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Number of frames to integrate. Use 0 to run until Ctrl-C.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard while auto-exposure settles",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch/CUDA device")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="TSDF voxel size in meters",
    )
    parser.add_argument("--esdf-voxel-size", type=float, default=0.03, help="ESDF voxel size")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 2.0),
        metavar=("X", "Y", "Z"),
        help="Mapper extent in meters",
    )
    parser.add_argument(
        "--grid-center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Mapper grid center in meters",
    )
    parser.add_argument(
        "--camera-pose",
        type=float,
        nargs=7,
        default=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help=(
            "Base camera-to-world pose as x y z qw qx qy qz before the fixed "
            "-90 degree X / +90 degree Y rotations and first-frame offset"
        ),
    )
    parser.add_argument("--depth-min", type=float, default=0.15, help="Minimum valid depth")
    parser.add_argument("--depth-max", type=float, default=DEPTH_MAX_M, help="Maximum valid depth")
    parser.add_argument(
        "--minimum-tsdf-weight",
        type=float,
        default=1.0,
        help="Minimum TSDF weight for occupied voxels and surfaces",
    )
    parser.add_argument(
        "--roughness",
        type=float,
        default=3.0,
        help="Block allocation multiplier for rough or cluttered scenes",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=1,
        help="Refresh Viser every N integrated frames",
    )
    parser.add_argument(
        "--max-visualized-voxels",
        type=int,
        default=100_000,
        help="Downsample visualized voxels to this count",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_count < 0:
        raise ValueError("--frame-count must be non-negative")
    if args.visualize_every <= 0:
        raise ValueError("--visualize-every must be positive")

    realsense = _import_realsense()
    base_camera_pose = _pose_from_cli(args.camera_pose, args.device)
    uncentered_camera_pose = _apply_camera_rotation_correction(base_camera_pose, args.device)
    fixed_camera_pose = None
    mapper = _make_mapper(args)
    depth_filter = FilterDepth(
        image_shape=(args.height, args.width),
        depth_minimum_distance=mapper.config.depth_minimum_distance,
        depth_maximum_distance=mapper.config.depth_maximum_distance,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=3,
    )

    print("Live volumetric mapping with MPC")
    print("  RGB-D backend: RealSense")
    print(f"  Mapper memory: {mapper.memory_usage_mb():.1f} MB")
    print(f"  Workspace extent: {tuple(args.extent)} m")
    print(f"  Grid center: {tuple(args.grid_center)} m")

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=MPC_ROBOT_FILE),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )
    rgb_image_handle = _add_rgb_panel(visualizer, args.height, args.width)
    _add_extent_box(visualizer, mapper.config)
    print(f"  Visualization: http://localhost:{args.port}")
    print(f"  MPC robot: {MPC_ROBOT_FILE}")

    last_observation = None
    integrated_frames = 0
    frame_iter = range(args.frame_count) if args.frame_count > 0 else itertools.count()
    total = args.frame_count if args.frame_count > 0 else None
    integrate_times = []
    esdf_times = []
    mpc_times = []
    last_voxel_grid = None
    live_mpc = None
    last_mpc_result = None

    with RealSenseRGBDStream(
        args.width,
        args.height,
        args.fps,
        args.device,
        rs_module=realsense,
    ) as stream:
        print(f"  Depth scale: {stream.depth_scale:.8f} m/unit")
        if stream.image_shape != (args.height, args.width):
            raise ValueError(
                f"RealSense stream resolved to {stream.image_shape}, but the mapper was "
                f"configured for {(args.height, args.width)}. Re-run with matching "
                "--height/--width values."
            )
        if args.warmup_frames > 0:
            print(f"  Warming up for {args.warmup_frames} frames...")
            stream.warmup(args.warmup_frames)

        print("\nIntegrating frames...")
        pbar = tqdm(frame_iter, total=total, desc="integrating")
        try:
            for _ in pbar:
                camera_pose = (
                    fixed_camera_pose
                    if fixed_camera_pose is not None
                    else uncentered_camera_pose
                )
                observation = stream.wait_for_observation(camera_pose)
                if observation is None:
                    continue

                observation = _filter_depth(observation, depth_filter)
                if fixed_camera_pose is None:
                    fixed_camera_pose, first_mean, target_center = _center_pose_from_first_frame(
                        uncentered_camera_pose,
                        observation,
                        args.depth_min,
                        args.depth_max,
                    )
                    observation.pose = fixed_camera_pose
                    mean_values = first_mean.detach().cpu().tolist()
                    target_values = target_center.detach().cpu().tolist()
                    pose_values = fixed_camera_pose.position[0].detach().cpu().tolist()
                    print(
                        "\nCentered first frame: "
                        f"camera mean={mean_values}, target center={target_values}, "
                        f"camera position={pose_values}"
                    )
                batched = _batch_observation(observation)

                timer = CudaEventTimer().start()
                mapper.integrate(batched)
                dt_s = timer.stop()

                integrated_frames += 1
                last_observation = observation
                integrate_times.append(dt_s)
                if len(integrate_times) > 20:
                    integrate_times = integrate_times[-20:]
                if (
                    integrated_frames > 0
                    and integrated_frames % args.visualize_every == 0
                ):
                    last_voxel_grid, esdf_dt_s = _update_live_visuals(
                        visualizer,
                        mapper,
                        observation,
                        rgb_image_handle,
                        point_size=args.voxel_size,
                        max_voxels=args.max_visualized_voxels,
                    )
                    esdf_times.append(esdf_dt_s)
                    if len(esdf_times) > 20:
                        esdf_times = esdf_times[-20:]
                    if live_mpc is None:
                        live_mpc = LiveEsdfMpc(last_voxel_grid, args.device, visualizer)
                        print(
                            f"\nMPC target: drag /mpc/target_{live_mpc.target_link} "
                            "in the Viser scene"
                        )
                    mpc_t0 = time.perf_counter()
                    last_mpc_result = live_mpc.step()
                    mpc_times.append(time.perf_counter() - mpc_t0)
                    if len(mpc_times) > 20:
                        mpc_times = mpc_times[-20:]

                postfix = {
                    "integrated": integrated_frames,
                    "integrate_ms": f"{np.mean(integrate_times) * 1000:.1f}",
                }
                if esdf_times:
                    postfix["esdf_ms"] = f"{np.mean(esdf_times) * 1000:.1f}"
                if mpc_times:
                    postfix["mpc_ms"] = f"{np.mean(mpc_times) * 1000:.1f}"
                if last_mpc_result is not None and last_mpc_result.position_error is not None:
                    postfix["mpc_err"] = (
                        f"{last_mpc_result.position_error.detach().mean().item():.3f}"
                    )
                pbar.set_postfix(postfix)
        except KeyboardInterrupt:
            print("\nCapture interrupted; stopping integration.")

    if integrated_frames == 0 or last_observation is None:
        print("No frames were integrated.")
        return

    print(f"\nIntegrated {integrated_frames} frames")
    if last_voxel_grid is None:
        print("Computing ESDF...")
        last_voxel_grid = mapper.compute_esdf()
        _add_static_esdf_slice(visualizer, last_voxel_grid, mapper.config)
    if last_voxel_grid.feature_tensor is not None:
        print(
            f"  ESDF shape: {tuple(last_voxel_grid.feature_tensor.shape)}, "
            f"voxel_size: {last_voxel_grid.voxel_size:.4f} m"
        )
    if live_mpc is None and last_voxel_grid.feature_tensor is not None:
        live_mpc = LiveEsdfMpc(last_voxel_grid, args.device, visualizer)
        live_mpc.step()
        print(
            f"  MPC target: drag /mpc/target_{live_mpc.target_link} "
            "in the Viser scene"
        )

    _update_visualizer(
        visualizer,
        mapper,
        last_observation,
        point_size=args.voxel_size,
        max_voxels=args.max_visualized_voxels,
    )
    _update_rgb_panel(rgb_image_handle, last_observation)
    _add_static_esdf_slice(visualizer, last_voxel_grid, mapper.config)
    print("\nVisualization running. Press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
