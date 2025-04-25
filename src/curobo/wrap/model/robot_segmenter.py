#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from typing import Dict, Optional, Tuple, Union

# Third Party
import torch
from torch.profiler import record_function

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.cv import get_projection_rays, project_depth_using_rays
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error
from curobo.util.torch_utils import (
    get_torch_jit_decorator,
    is_torch_compile_available,
)
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


class RobotSegmenter:
    def __init__(
        self,
        robot_world: RobotWorld,
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = True,
        ops_dtype: torch.dtype = torch.float32,
        depth_to_meter: float = 0.001,
    ):
        self._robot_world = robot_world
        self._projection_rays = None
        self.ready = False
        self._out_points_buffer = None
        self._out_gp = None
        self._out_gq = None
        self._out_gpt = None
        self._cu_graph = None
        self._use_cuda_graph = use_cuda_graph
        self.tensor_args = robot_world.tensor_args
        self.distance_threshold = distance_threshold
        self._ops_dtype = ops_dtype
        self._depth_to_meter = depth_to_meter

    @staticmethod
    def from_robot_file(
        robot_file: Union[str, Dict],
        collision_sphere_buffer: Optional[float] = None,
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = True,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        ops_dtype: torch.dtype = torch.float32,
        depth_to_meter: float = 0.001,
    ):
        robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        if collision_sphere_buffer is not None:
            robot_dict["kinematics"]["collision_sphere_buffer"] = collision_sphere_buffer

        robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args=tensor_args)

        config = RobotWorldConfig.load_from_config(
            robot_cfg,
            None,
            collision_activation_distance=0.0,
            tensor_args=tensor_args,
        )
        robot_world = RobotWorld(config)

        return RobotSegmenter(
            robot_world,
            distance_threshold=distance_threshold,
            use_cuda_graph=use_cuda_graph,
            ops_dtype=ops_dtype,
            depth_to_meter=depth_to_meter,
        )

    def get_pointcloud_from_depth(self, camera_obs: CameraObservation):
        if self._projection_rays is None:
            self.update_camera_projection(camera_obs)
        depth_image = camera_obs.depth_image.to(dtype=self._ops_dtype)
        if len(depth_image.shape) == 2:
            depth_image = depth_image.unsqueeze(0)
        points = project_depth_using_rays(depth_image, self._projection_rays)

        return points

    def update_camera_projection(self, camera_obs: CameraObservation):
        intrinsics = camera_obs.intrinsics
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        project_rays = get_projection_rays(
            camera_obs.depth_image.shape[-2],
            camera_obs.depth_image.shape[-1],
            intrinsics,
            self._depth_to_meter,
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
    ):
        """
        Assumes 1 robot and batch of depth images, batch of poses
        """
        if len(camera_obs.depth_image.shape) != 3:
            log_error("Send depth image as (batch, height, width)")

        active_js = self._robot_world.get_active_js(joint_state)

        mask, filtered_image = self.get_robot_mask_from_active_js(camera_obs, active_js)

        return mask, filtered_image

    def get_robot_mask_from_active_js(
        self, camera_obs: CameraObservation, active_joint_state: JointState
    ):
        q = active_joint_state.position
        mask, filtered_image = self._call_op(camera_obs, q)

        return mask, filtered_image

    def _create_cg_graph(self, cam_obs, q):
        self._cu_cam_obs = cam_obs.clone()
        self._cu_q = q.clone()
        s = torch.cuda.Stream(device=self.tensor_args.device)
        s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))

        with torch.cuda.stream(s):
            for _ in range(3):
                self._cu_out, self._cu_filtered_out = self._mask_op(self._cu_cam_obs, self._cu_q)
        torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)

        self._cu_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cu_graph, stream=s):
            self._cu_out, self._cu_filtered_out = self._mask_op(
                self._cu_cam_obs,
                self._cu_q,
            )

    def _call_op(self, cam_obs, q):
        if self._use_cuda_graph:
            if self._cu_graph is None:
                self._create_cg_graph(cam_obs, q)
            self._cu_cam_obs.copy_(cam_obs)
            self._cu_q.copy_(q)
            self._cu_graph.replay()
            return self._cu_out.clone(), self._cu_filtered_out.clone()
        return self._mask_op(cam_obs, q)

    @record_function("robot_segmenter/_mask_op")
    def _mask_op(self, camera_obs, q):
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        robot_spheres = self._robot_world.get_kinematics(q).link_spheres_tensor

        points = self.get_pointcloud_from_depth(camera_obs)
        camera_to_robot = camera_obs.pose.to(TensorDeviceType(dtype=self._ops_dtype))

        if self._out_points_buffer is None:
            self._out_points_buffer = points.clone()
        if self._out_gpt is None:
            self._out_gpt = torch.zeros(
                (points.shape[0], points.shape[1], 3), device=points.device, dtype=self._ops_dtype
            )
        if self._out_gp is None:
            self._out_gp = torch.zeros(
                (camera_to_robot.position.shape[0], 3), device=points.device, dtype=self._ops_dtype
            )
        if self._out_gq is None:
            self._out_gq = torch.zeros(
                (camera_to_robot.quaternion.shape[0], 4),
                device=points.device,
                dtype=self._ops_dtype,
            )

        points_in_robot_frame = camera_to_robot.batch_transform_points(
            points,
            out_buffer=self._out_points_buffer,
            gp_out=self._out_gp,
            gq_out=self._out_gq,
            gpt_out=self._out_gpt,
        )

        robot_spheres = robot_spheres.to(dtype=self._ops_dtype)
        if is_torch_compile_available():
            mask, filtered_image = mask_spheres_image(
                camera_obs.depth_image,
                robot_spheres,
                points_in_robot_frame,
                self.distance_threshold,
            )
        else:
            mask, filtered_image = mask_spheres_image_cdist(
                camera_obs.depth_image,
                robot_spheres,
                points_in_robot_frame,
                self.distance_threshold,
            )

        return mask, filtered_image

    @property
    def kinematics(self) -> CudaRobotModel:
        return self._robot_world.kinematics

    @property
    def robot_world(self) -> RobotWorld:
        return self._robot_world

    @property
    def base_link(self) -> str:
        return self.kinematics.base_link


@get_torch_jit_decorator()
def mask_image(
    image: torch.Tensor, distance: torch.Tensor, distance_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    distance = distance.view(
        image.shape[0],
        image.shape[1],
        image.shape[2],
    )
    mask = torch.logical_and((image > 0.0), (distance > -distance_threshold))
    filtered_image = torch.where(mask, 0, image)
    return mask, filtered_image


@get_torch_jit_decorator()
def mask_spheres_image(
    image: torch.Tensor,
    link_spheres_tensor: torch.Tensor,
    points: torch.Tensor,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if link_spheres_tensor.shape[0] != 1:
        assert link_spheres_tensor.shape[0] == points.shape[0]
    if len(points.shape) == 2:
        points = points.unsqueeze(0)

    robot_spheres = link_spheres_tensor.view(link_spheres_tensor.shape[0], -1, 4).contiguous()
    robot_spheres = robot_spheres.unsqueeze(-3)

    robot_radius = robot_spheres[..., 3]
    points = points.unsqueeze(-2)

    robot_points = robot_spheres[..., :3]

    sph_distance = -1 * (
        torch.linalg.norm(points - robot_points, dim=-1) - robot_radius
    )  # b, n_spheres
    distance = torch.max(sph_distance, dim=-1)[0]

    distance = distance.view(
        image.shape[0],
        image.shape[1],
        image.shape[2],
    )
    mask = torch.logical_and((image > 0.0), (distance > -distance_threshold))
    filtered_image = torch.where(mask, 0, image)
    return mask, filtered_image


@get_torch_jit_decorator()
def mask_spheres_image_cdist(
    image: torch.Tensor,
    link_spheres_tensor: torch.Tensor,
    world_points: torch.Tensor,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if link_spheres_tensor.shape[0] != 1:
        assert link_spheres_tensor.shape[0] == world_points.shape[0]
    if len(world_points.shape) == 2:
        world_points = world_points.unsqueeze(0)

    robot_spheres = link_spheres_tensor.view(link_spheres_tensor.shape[0], -1, 4).contiguous()

    robot_spheres = robot_spheres.expand(world_points.shape[0], -1, -1)
    robot_points = robot_spheres[..., :3]

    robot_radius = robot_spheres[..., 3].unsqueeze(1)

    sph_distance = torch.cdist(world_points, robot_points, p=2.0)
    sph_distance = -1.0 * (sph_distance - robot_radius)

    distance = torch.max(sph_distance, dim=-1)[0]

    distance = distance.view(
        image.shape[0],
        image.shape[1],
        image.shape[2],
    )
    distance_mask = distance > -distance_threshold
    mask = torch.logical_and((image > 0.0), distance_mask)
    filtered_image = torch.where(mask, 0, image)
    return mask, filtered_image
