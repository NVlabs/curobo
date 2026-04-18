# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
import tempfile
from typing import Dict, List, Optional, Sequence

# Third Party
import numpy as np
import torch
import trimesh
import yourdfpy

from curobo._src.geom.types import SceneCfg, Sphere

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.state.state_joint import JointState
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util_file import join_path

try:
    import viser
    from viser.extras import ViserUrdf
except ImportError:
    raise ImportError("Viser not installed. Install with: pip install viser")


class ViserVisualizer:
    def __init__(
        self,
        content_path: Optional[ContentPath] = None,
        device_cfg: DeviceCfg = DeviceCfg(
            device=torch.device("cuda"), dtype=torch.float32
        ),
        add_robot_to_scene: bool = False,
        connect_ip: str = "0.0.0.0",
        connect_port: int = 8080,
        initialize_viser: bool = True,
        add_control_frames: bool = True,
        visualize_robot_spheres: bool = False,
        visualize_collision_meshes: bool = False,
    ):
        self._visualize_robot_spheres = visualize_robot_spheres

        self._server = viser.ViserServer(host=connect_ip, port=connect_port)

        self._server.scene.add_grid("/ground_plane", width=20, height=20)
        robot_data = None
        visualize_frames = []
        self._robot_model = None
        self._mesh_root = None
        # load robot into viser and set it at default joint configuration.
        if add_robot_to_scene and content_path is None:
            log_and_raise("Content path is required to add robot to scene")
        if content_path is not None:
            if not isinstance(content_path, Dict):
                robot_data = load_robot_yaml(content_path)
            else:
                robot_data = content_path
            if "robot_cfg" not in robot_data:
                robot_data = {"robot_cfg": robot_data}
            robot_data["robot_cfg"]["kinematics"]["load_tool_frames_with_mesh"] = True
            visualize_frames = robot_data["robot_cfg"]["kinematics"]["tool_frames"].copy()
            self._robot_model = KinematicsCfg.from_data_dict(
                robot_data["robot_cfg"]["kinematics"],
                device_cfg=device_cfg,
            )

            self._kinematics = Kinematics(self._robot_model)

            self._mesh_root = self._robot_model.generator_config.asset_root_path
            extra_links = self._robot_model.generator_config.extra_links
            if extra_links:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".urdf", delete=False, mode="w", prefix="curobo_viz_"
                )
                kin_config = self._kinematics.config.kinematics_config
                kin_config.export_to_urdf(
                    output_path=tmp.name,
                    kinematics_parser=self._robot_model.kinematics_parser,
                )
                urdf_path = tmp.name
            else:
                urdf_path = self._robot_model.generator_config.urdf_path
            self._urdf = yourdfpy.URDF.load(
                urdf_path,
                load_meshes=True,
                build_scene_graph=True,
                filename_handler=self._file_name_handler,
                build_collision_scene_graph=visualize_collision_meshes,
                load_collision_meshes=visualize_collision_meshes,
            )
            self._viser_urdf = ViserUrdf(
                self._server,
                self._urdf,
                root_node_name="/" + self._robot_model.generator_config.base_link,
                load_collision_meshes=visualize_collision_meshes,
            )
            self._viser_joint_names = list(self._viser_urdf.get_actuated_joint_names())
            self._vis_frames = visualize_frames.copy()

            self.reset_robot()
            self._robot_spheres = []
            # add robot spheres:
            self._control_frames = {}
            if add_control_frames:
                # get pose of frames at current joint state:
                joint_state = self._kinematics.default_joint_state
                kin_state = self._kinematics.compute_kinematics(joint_state)
                if kin_state.tool_poses is None:
                    log_and_raise("KinematicsState.tool_poses is None; check tool_frames / kinematics config.")
                for frame_name in self._vis_frames:
                    frame_pose = kin_state.tool_poses[frame_name]
                    self._control_frames[frame_name] = self.add_control_frame(
                        "/target_" + frame_name, frame_pose, scale=0.1
                    )

    def _file_name_handler(self, fname: str) -> str:
        # This will also remove package:// from the file name:
        fname = fname.replace("package://", "")
        return join_path(self._mesh_root, fname)

    @property
    def joint_names(self) -> List[str]:
        return self._viser_joint_names

    def update_robot_spheres(self, joint_state: JointState):
        self._robot_spheres = []
        # reindex:
        joint_state = joint_state.clone()
        joint_state = self._kinematics.get_active_js(joint_state)
        spheres = self._kinematics.get_robot_as_spheres(
            joint_state.position.contiguous().view(1, -1), filter_valid=True
        )
        spheres = spheres[0]
        self.add_batched_spheres(spheres)

        # for sphere in spheres:
        #    handle = self.add_sphere(sphere)
        #    self._robot_spheres.append(handle)

    def add_frame(self, frame_name: str, frame_pose: Pose, scale: float = 0.2) -> viser.FrameHandle:
        frame_position = frame_pose.position.cpu().squeeze().numpy()
        frame_quat = frame_pose.quaternion.cpu().squeeze().numpy()
        frame = self._server.scene.add_frame(
            frame_name,
            position=frame_position,
            wxyz=frame_quat,
        )
        return frame

    def add_batched_frames(self, frame_name: str, frame_poses: Pose):
        frame_positions = frame_poses.position.cpu().squeeze().numpy()
        frame_quats = frame_poses.quaternion.cpu().squeeze().numpy()
        self._batched_frames_handle = self._server.scene.add_batched_axes(
            name=frame_name,
            batched_positions=frame_positions,
            batched_wxyzs=frame_quats,
            axes_length=0.02,
            axes_radius=0.005,
        )

    def add_control_frame(
        self, frame_name: str, frame_pose: Pose, scale: float = 0.2
    ) -> viser.TransformControlsHandle:
        frame_position = frame_pose.position.cpu().squeeze().numpy()
        frame_quat = frame_pose.quaternion.cpu().squeeze().numpy()
        transform_control = self._server.scene.add_transform_controls(
            frame_name, scale=scale, position=frame_position, wxyz=frame_quat
        )
        return transform_control

    def reset_robot(self):
        joint_state = self._kinematics.get_full_js(self._kinematics.default_joint_state)
        self.set_joint_state(joint_state)

    def set_joint_positions(
        self, joint_positions: Sequence[float], joint_names: List[str], **kwargs
    ) -> None:
        joint_state = JointState.from_position(joint_positions, joint_names=joint_names)
        self.set_joint_state(joint_state)

    def get_control_frame_pose(self) -> Dict[str, Pose]:
        """Get pose of interactive frames.

        Note that these interactive frames are named with a "target_" prefix in the visualizer
        but this function returns the actual frame names (e.g., "target_ee_link" -> "ee_link").

        Returns:
            Dictionary of frame names and their poses.
        """
        poses = {}
        for frame_name in self._vis_frames:
            pose = Pose.from_numpy(
                self._control_frames[frame_name].position, self._control_frames[frame_name].wxyz
            )
            poses[frame_name] = pose
        return poses

    def set_joint_state(self, joint_state: JointState) -> None:
        joint_state = joint_state.clone()
        viser_set = set(self._viser_joint_names)
        js_set = set(joint_state.joint_names)
        if viser_set != js_set:
            missing_from_js = viser_set - js_set
            if missing_from_js:
                joint_state = self._kinematics.get_full_js(joint_state)

        if self._visualize_robot_spheres:
            self.update_robot_spheres(joint_state)

        joint_state.reindex(self._viser_joint_names)
        self._viser_urdf.update_cfg(joint_state.position.cpu().squeeze().numpy())

    def add_batched_spheres(self, spheres: List[Sphere]):
        # create a trimesh for the first sphere:
        sphere_mesh = spheres[0].get_trimesh_mesh()
        scale = []
        position = []
        quaternion = []
        mesh_radius = spheres[0].radius
        for i in range(0, len(spheres)):
            scale.append(spheres[i].radius / mesh_radius)
            position.append(spheres[i].pose[0:3])
            quaternion.append(spheres[i].pose[3:7])

        self._batched_spheres_handle = self._server.scene.add_batched_meshes_simple(
            name="curobo_spheres",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_scales=scale,
            batched_positions=position,
            batched_wxyzs=quaternion,
        )

    def add_batched_spheres_from_position(
        self, position: np.ndarray, radius: np.ndarray, color=None, name: str = "curobo_spheres"
    ):
        # create a trimesh for the first sphere:
        radius[radius < 0.001] = 0.001
        sphere_mesh = Sphere(
            name="curobo_spheres", pose=[0, 0, 0, 1, 0, 0, 0], radius=radius[0]
        ).get_trimesh_mesh()
        quaternion = np.zeros((len(position), 4))
        quaternion[:, 0] = 1.0
        scale = radius / radius[0]
        colors = np.zeros((len(position), 3))
        if color is not None:
            colors[:, :] = color

        self._batched_spheres_handle = self._server.scene.add_batched_meshes_simple(
            name=name,
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_scales=scale,
            batched_positions=position,
            batched_wxyzs=quaternion,
            batched_colors=colors,
        )

    def add_sphere(self, sphere: Sphere):
        if not isinstance(sphere, Sphere):
            log_and_raise("Sphere is not a valid Sphere object")
        sphere_handle = self._server.scene.add_icosphere(
            name=sphere.name,
            position=np.ravel(sphere.position),
            radius=sphere.radius,
            color=sphere.color if sphere.color is not None else (0, 200, 0),
        )
        return sphere_handle

    def add_line_segments(self, line_segments: np.ndarray, color: np.ndarray):
        points = line_segments
        colors = color

        self._server.scene.add_line_segments(
            "/line_segments",
            points=points,
            colors=colors,
            line_width=3.0,
        )

    def add_mesh(self, mesh_trimesh: trimesh.Trimesh, name: str = "mesh"):
        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            log_and_raise("Mesh is not a valid Mesh object")
        mesh_handle = self._server.scene.add_mesh_trimesh(
            name=name,
            mesh=mesh_trimesh,
        )
        return mesh_handle

    def add_scene(self, scene_cfg: SceneCfg, add_control_frames: bool = False):
        obstacle_frames = {}
        mesh_scene = SceneCfg.create_mesh_scene(scene_cfg)
        for mesh in mesh_scene.mesh:
            self.add_mesh(
                mesh.get_trimesh_mesh(transform_with_pose=not add_control_frames),
                name="/obstacles/" + mesh.name + "/mesh",
            )
            if add_control_frames:
                obstacle_frames[mesh.name] = self.add_control_frame(
                    "/obstacles/" + mesh.name, Pose.from_list(mesh.pose), scale=0.2
                )
        return obstacle_frames

    def add_point_cloud(
        self,
        pointcloud: np.ndarray,
        colors=[200, 200, 200],
        point_size: float = 0.005,
        name: str = "pointcloud",
    ):
        handle = self._server.scene.add_point_cloud(
            name=name,
            points=pointcloud,
            colors=colors,
            point_size=point_size,
        )
        return handle

    def add_image(
        self,
        image: np.ndarray,
        render_width: float,
        render_height: float,
        pose: Pose,
        name: str = "image",
    ):
        """Add an image to the 3D scene at a specific pose.

        Args:
            image: RGB image array (H, W, 3) as uint8.
            render_width: Width of the image in world units (meters).
            render_height: Height of the image in world units (meters).
            pose: Pose of the image in world coordinates.
            name: Name for the image in the scene.

        Returns:
            Handle to the image object.
        """
        handle = self._server.scene.add_image(
            name=name,
            image=image,
            render_width=render_width,
            render_height=render_height,
            position=pose.position.cpu().squeeze().numpy(),
            wxyz=pose.quaternion.cpu().squeeze().numpy(),
        )
        return handle

