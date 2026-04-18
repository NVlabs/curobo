# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Robot segmentation from depth images.

This module provides functionality to segment a robot from depth images
by computing distances between depth points and robot collision spheres.
"""

# Standard Library
from typing import Dict, Optional, Tuple, Union

# Third Party
import torch
from torch.profiler import record_function

from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float16_tensors,
    check_float32_tensors,
)
from curobo._src.geom.cv import get_projection_rays, project_depth_using_rays

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator, is_torch_compile_available
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


class RobotSegmenter:
    """Segment robot from depth images using collision spheres.

    This class computes masks for robot pixels in depth images by checking
    distances between depth points and robot collision spheres.

    Example:
        ```python
        from curobo._src.perception.robot_segmenter import RobotSegmenter

        segmenter = RobotSegmenter.from_robot_file("franka.yml")
        mask, filtered_image = segmenter.get_robot_mask(camera_obs, joint_state)
        ```
    """

    def __init__(
        self,
        kinematics: Kinematics,
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = True,
        ops_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the robot segmenter.

        Args:
            kinematics: Robot kinematics model for forward kinematics.
            distance_threshold: Distance threshold for segmentation.
            use_cuda_graph: Whether to use CUDA graphs for acceleration.
            ops_dtype: Data type for operations.
        """
        self._kinematics = kinematics
        self._projection_rays = None
        self.ready = False
        self._out_points_buffer = None
        self._out_gp = None
        self._out_gq = None
        self._out_gpt = None
        self.device_cfg = kinematics.device_cfg
        self.distance_threshold = distance_threshold
        self._ops_dtype = ops_dtype
        self._graph_executor = GraphExecutor(
            self._mask_op,
            device=self.device_cfg.device,
            use_cuda_graph=use_cuda_graph,
            clone_outputs=True,
        )

    @staticmethod
    def from_robot_file(
        robot_file: Union[str, Dict],
        collision_sphere_buffer: Optional[float] = None,
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "RobotSegmenter":
        """Create a RobotSegmenter from a robot configuration file.

        Args:
            robot_file: Path to robot configuration file or dict.
            collision_sphere_buffer: Optional buffer to add to collision spheres.
            distance_threshold: Distance threshold for segmentation.
            use_cuda_graph: Whether to use CUDA graphs.
            device_cfg: Tensor device configuration.

        Returns:
            Configured RobotSegmenter instance.
        """
        if isinstance(robot_file, str):
            robot_file = load_yaml(join_path(get_robot_configs_path(), robot_file))

        if isinstance(robot_file, dict):
            if collision_sphere_buffer is not None:
                robot_file["kinematics"]["collision_sphere_buffer"] = collision_sphere_buffer
            robot_cfg = RobotCfg.create(robot_file, device_cfg=device_cfg)
        else:
            log_and_raise("robot_file must be a string path or dict")

        kinematics = Kinematics(robot_cfg.kinematics)

        return RobotSegmenter(
            kinematics,
            distance_threshold=distance_threshold,
            use_cuda_graph=use_cuda_graph,
        )

    def get_pointcloud_from_depth(
        self, camera_obs: CameraObservation
    ) -> torch.Tensor:
        """Convert depth image to point cloud.

        Args:
            camera_obs: Camera observation with depth image.

        Returns:
            Point cloud tensor.
        """
        if self._projection_rays is None:
            self.update_camera_projection(camera_obs)
        depth_image = camera_obs.depth_image.to(dtype=self._ops_dtype)
        if len(depth_image.shape) == 2:
            depth_image = depth_image.unsqueeze(0)
        points = project_depth_using_rays(depth_image, self._projection_rays)

        return points

    def update_camera_projection(self, camera_obs: CameraObservation) -> None:
        """Update camera projection rays from observation.

        Args:
            camera_obs: Camera observation with intrinsics.
        """
        intrinsics = camera_obs.intrinsics
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        project_rays = get_projection_rays(
            camera_obs.depth_image.shape[-2],
            camera_obs.depth_image.shape[-1],
            intrinsics,
            camera_obs.depth_to_meter,
        ).to(dtype=self._ops_dtype)

        if self._projection_rays is None:
            self._projection_rays = project_rays

        self._projection_rays.copy_(project_rays)
        self.ready = True

    @record_function("robot_segmenter/get_robot_mask")
    def get_robot_mask(
        self,
        camera_obs: CameraObservation,
        joint_state: JointState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get robot mask from camera observation and joint state.

        Assumes 1 robot and batch of depth images, batch of poses.

        Args:
            camera_obs: Camera observation with depth image.
            joint_state: Current joint state of the robot.

        Returns:
            Tuple of (mask, filtered_image).

        Raises:
            ValueError: If depth image is not 3D (batch, height, width).
        """
        if len(camera_obs.depth_image.shape) != 3:
            log_and_raise("Send depth image as (batch, height, width)")

        active_js = self._kinematics.get_active_js(joint_state)

        mask, filtered_image = self.get_robot_mask_from_active_js(camera_obs, active_js)

        return mask, filtered_image

    def get_robot_mask_from_active_js(
        self, camera_obs: CameraObservation, active_joint_state: JointState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get robot mask using active joint state.

        Args:
            camera_obs: Camera observation with depth image.
            active_joint_state: Joint state with only active joints.

        Returns:
            Tuple of (mask, filtered_image).
        """
        q = active_joint_state.position
        mask, filtered_image = self._call_op(camera_obs, q)

        return mask, filtered_image

    def _call_op(
        self, cam_obs: CameraObservation, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call the mask operation with optional CUDA graph."""
        #return self._mask_op(cam_obs, q)
        return self._graph_executor(cam_obs, q)

    @record_function("robot_segmenter/_mask_op")
    def _mask_op(
        self, camera_obs: CameraObservation, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the masking operation."""
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        robot_spheres = self._kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=self._kinematics.joint_names)
        ).robot_spheres.squeeze(1)

        points = self.get_pointcloud_from_depth(camera_obs)
        T_camera_in_robot_frame = camera_obs.pose
        points = points.to(dtype=torch.float32)

        if self._out_points_buffer is None:
            self._out_points_buffer = points.clone()
        if self._out_gpt is None:
            self._out_gpt = torch.zeros(
                (points.shape[0], points.shape[1], 3), device=points.device
            )
        if self._out_gp is None:
            self._out_gp = torch.zeros(
                (T_camera_in_robot_frame.position.shape[0], 3), device=points.device
            )
        if self._out_gq is None:
            self._out_gq = torch.zeros(
                (T_camera_in_robot_frame.quaternion.shape[0], 4), device=points.device
            )

        points_in_robot_frame = T_camera_in_robot_frame.batch_transform_points(
            points,
            out_buffer=self._out_points_buffer,
            gp_out=self._out_gp,
            gq_out=self._out_gq,
            gpt_out=self._out_gpt,
        )

        robot_spheres = robot_spheres.to(dtype=self._ops_dtype)
        robot_spheres = robot_spheres.view(robot_spheres.shape[0], -1, 4)
        if robot_spheres.shape[0] != 1:
            if robot_spheres.shape[0] != points_in_robot_frame.shape[0]:
                log_and_raise(
                    "robot_spheres batch must be 1 or match points batch: "
                    f"got {robot_spheres.shape[0]} vs {points_in_robot_frame.shape[0]}"
                )
        if robot_spheres.dtype == torch.float16:
            check_float16_tensors(robot_spheres.device, robot_spheres=robot_spheres)
        else:
            check_float32_tensors(robot_spheres.device, robot_spheres=robot_spheres)
        if points_in_robot_frame.dim() == 2:
            points_in_robot_frame = points_in_robot_frame.unsqueeze(0)

        if is_torch_compile_available():
            mask, filtered_image = _mask_spheres_image(
                camera_obs.depth_image, robot_spheres, points_in_robot_frame,
                self.distance_threshold,
            )
        else:
            mask, filtered_image = _mask_spheres_image_cdist(
                camera_obs.depth_image, robot_spheres, points_in_robot_frame,
                self.distance_threshold,
            )

        return mask, filtered_image

    @property
    def kinematics(self) -> Kinematics:
        """Get the robot kinematics model."""
        return self._kinematics

    @property
    def base_link(self) -> str:
        """Get the robot base link name."""
        return self._kinematics.base_link


@get_torch_jit_decorator()
def _mask_image(
    image: torch.Tensor, distance: torch.Tensor, distance_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask image based on distance threshold."""
    distance = distance.view(
        image.shape[0],
        image.shape[1],
        image.shape[2],
    )
    mask = torch.logical_and((image > 0.0), (distance > -distance_threshold))
    filtered_image = torch.where(mask, 0, image)
    return mask, filtered_image


@get_torch_jit_decorator()
def _mask_spheres_image(
    image: torch.Tensor,
    robot_spheres: torch.Tensor,
    points: torch.Tensor,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask image based on distance to robot spheres.

    Caller must pass pre-validated tensors with normalized shapes:
    ``robot_spheres`` as ``[batch_robot, n_robot_spheres, 4]`` with
    ``batch_robot`` equal to 1 or ``points``'s batch, and ``points`` as
    ``[batch_points, n_points, 3]``.
    """
    robot_spheres = robot_spheres.unsqueeze(-3)
    robot_radius = robot_spheres[..., 3]
    points = points.unsqueeze(-2)
    sph_distance = -1 * (
        torch.linalg.norm(points - robot_spheres[..., :3], dim=-1) - robot_radius
    )
    distance = torch.max(sph_distance, dim=-1)[0]

    distance = distance.view(image.shape[0], image.shape[1], image.shape[2])
    mask = torch.logical_and((image > 0.0), (distance > -distance_threshold))
    filtered_image = torch.where(mask, 0, image)
    return mask, filtered_image


@get_torch_jit_decorator()
def _mask_spheres_image_cdist(
    image: torch.Tensor,
    robot_spheres: torch.Tensor,
    world_points: torch.Tensor,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask image based on distance to robot spheres using torch.cdist.

    Fallback for environments where torch.compile is not available.
    Uses cdist instead of broadcasting + linalg.norm, which avoids
    materializing a large intermediate tensor.

    Caller must pass pre-validated tensors with normalized shapes:
    ``robot_spheres`` as ``[batch_robot, n_robot_spheres, 4]`` with
    ``batch_robot`` equal to 1 or ``world_points``'s batch, and
    ``world_points`` as ``[batch_points, n_points, 3]``.
    """
    robot_spheres = robot_spheres.expand(world_points.shape[0], -1, -1)
    robot_points = robot_spheres[..., :3]
    robot_radius = robot_spheres[..., 3].unsqueeze(1)

    sph_distance = torch.cdist(world_points, robot_points, p=2.0)
    sph_distance = -1.0 * (sph_distance - robot_radius)
    distance = torch.max(sph_distance, dim=-1)[0]

    distance = distance.view(image.shape[0], image.shape[1], image.shape[2])
    distance_mask = distance > -distance_threshold
    mask = torch.logical_and((image > 0.0), distance_mask)
    filtered_image = torch.where(mask, 0, image)
    return mask, filtered_image

