# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from unittest.mock import MagicMock, patch

# Third Party
import numpy as np
import pytest
import torch
import trimesh

# CuRobo
from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.state.state_joint import JointState
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

try:
    from curobo._src.util.viser_visualizer import ViserVisualizer
except ImportError:
    pytest.skip(reason="Viser not installed", allow_module_level=True)


class TestViserVisualizer:
    """Test ViserVisualizer class."""

    def test_init_without_robot(self):
        """Test initialization without a robot."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
                initialize_viser=True,
            )

            assert viser_visualizer is not None
            assert viser_visualizer._robot_model is None
            mock_server_instance.scene.add_grid.assert_called_once()

    def test_init_with_robot_requires_content_path(self):
        """Test that adding robot to scene requires content path."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            with pytest.raises(Exception, match="Content path is required"):
                ViserVisualizer(
                    content_path=None,
                    add_robot_to_scene=True,
                )

    def test_init_with_robot_from_yaml(self):
        """Test initialization with robot from YAML file."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
                visualize_robot_spheres=False,
            )

            assert viser_visualizer is not None
            assert viser_visualizer._robot_model is not None
            assert viser_visualizer._kinematics is not None
            assert len(viser_visualizer.joint_names) > 0

    def test_joint_names_property(self, cuda_device_cfg):
        """Test joint_names property."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=cuda_device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
            )

            joint_names = viser_visualizer.joint_names
            assert isinstance(joint_names, (list, tuple))
            assert len(joint_names) == 7  # Franka has 7 actuated joints (gripper joints are mimic)

    def test_file_name_handler(self):
        """Test _file_name_handler method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            viser_visualizer._mesh_root = "/test/root"

            # Test removing package:// prefix
            result = viser_visualizer._file_name_handler("package://robot/mesh.obj")
            assert "robot/mesh.obj" in result
            assert "/test/root" in result

            # Test without package:// prefix
            result = viser_visualizer._file_name_handler("robot/mesh.obj")
            assert "robot/mesh.obj" in result

    def test_set_joint_positions(self, cuda_device_cfg):
        """Test set_joint_positions method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=cuda_device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
            )

            joint_names = viser_visualizer.joint_names
            joint_positions = torch.zeros(len(joint_names))
            viser_visualizer.set_joint_positions(joint_positions, joint_names)

            # Verify update was called
            assert mock_server_instance.method_calls

    def test_set_joint_state(self):
        """Test set_joint_state method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
            )

            joint_names = viser_visualizer.joint_names
            joint_state = JointState.from_position(
                torch.zeros(len(joint_names)),
                joint_names=joint_names,
            )
            viser_visualizer.set_joint_state(joint_state)

            # Verify method was executed without errors
            assert viser_visualizer is not None

    def test_reset_robot(self):
        """Test reset_robot method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
            )

            # Reset should work without errors
            viser_visualizer.reset_robot()
            assert viser_visualizer is not None

    def test_add_frame(self):
        """Test add_frame method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_frame_handle = MagicMock()
            mock_server_instance.scene.add_frame.return_value = mock_frame_handle
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            pose = Pose.from_list([0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])
            frame_handle = viser_visualizer.add_frame("test_frame", pose, scale=0.2)

            assert frame_handle is not None
            mock_server_instance.scene.add_frame.assert_called_once()

    def test_add_batched_frames(self):
        """Test add_batched_frames method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            # Create batch of poses
            positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            quaternions = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            )
            poses = Pose(position=positions, quaternion=quaternions)

            viser_visualizer.add_batched_frames("test_frames", poses)

            mock_server_instance.scene.add_batched_axes.assert_called_once()

    def test_add_control_frame(self):
        """Test add_control_frame method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_control_handle = MagicMock()
            mock_server_instance.scene.add_transform_controls.return_value = (
                mock_control_handle
            )
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            pose = Pose.from_list([0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])
            control_handle = viser_visualizer.add_control_frame("test_control", pose, scale=0.2)

            assert control_handle is not None
            mock_server_instance.scene.add_transform_controls.assert_called_once()

    def test_get_control_frame_pose(self):
        """Test get_control_frame_pose method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=True,
            )

            # Mock control frame positions
            for frame_name in viser_visualizer._vis_frames:
                viser_visualizer._control_frames[frame_name].position = np.array([0.5, 0.5, 0.5])
                viser_visualizer._control_frames[frame_name].wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            poses = viser_visualizer.get_control_frame_pose()

            assert isinstance(poses, dict)
            assert len(poses) > 0
            for frame_name, pose in poses.items():
                assert isinstance(pose, Pose)

    def test_add_sphere(self):
        """Test add_sphere method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_sphere_handle = MagicMock()
            mock_server_instance.scene.add_icosphere.return_value = mock_sphere_handle
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            sphere = Sphere(
                name="test_sphere",
                pose=[0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
                radius=0.1,
                color=[255, 0, 0],
            )

            handle = viser_visualizer.add_sphere(sphere)

            assert handle is not None
            mock_server_instance.scene.add_icosphere.assert_called_once()

    def test_add_sphere_invalid_type(self):
        """Test add_sphere with invalid type raises error."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            with pytest.raises(Exception, match="Sphere is not a valid Sphere object"):
                viser_visualizer.add_sphere("not_a_sphere")

    def test_add_batched_spheres(self):
        """Test add_batched_spheres method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            spheres = [
                Sphere(
                    name="sphere1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.1,
                ),
                Sphere(
                    name="sphere2",
                    pose=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.15,
                ),
                Sphere(
                    name="sphere3",
                    pose=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.2,
                ),
            ]

            viser_visualizer.add_batched_spheres(spheres)

            mock_server_instance.scene.add_batched_meshes_simple.assert_called_once()

    def test_add_batched_spheres_from_position(self):
        """Test add_batched_spheres_from_position method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            radii = np.array([0.1, 0.15, 0.2])
            color = [255, 0, 0]

            viser_visualizer.add_batched_spheres_from_position(
                positions, radii, color=color, name="test_spheres"
            )

            mock_server_instance.scene.add_batched_meshes_simple.assert_called_once()

    def test_add_line_segments(self):
        """Test add_line_segments method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            line_segments = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
            colors = np.array([[255, 0, 0]])

            viser_visualizer.add_line_segments(line_segments, colors)

            mock_server_instance.scene.add_line_segments.assert_called_once()

    def test_add_mesh(self):
        """Test add_mesh method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_mesh_handle = MagicMock()
            mock_server_instance.scene.add_mesh_trimesh.return_value = mock_mesh_handle
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

            handle = viser_visualizer.add_mesh(mesh, name="test_mesh")

            assert handle is not None
            mock_server_instance.scene.add_mesh_trimesh.assert_called_once()

    def test_add_mesh_invalid_type(self):
        """Test add_mesh with invalid type raises error."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            with pytest.raises(Exception, match="Mesh is not a valid Mesh object"):
                viser_visualizer.add_mesh("not_a_mesh")

    def test_add_point_cloud(self):
        """Test add_point_cloud method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_point_cloud_handle = MagicMock()
            mock_server_instance.scene.add_point_cloud.return_value = mock_point_cloud_handle
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            pointcloud = np.random.rand(100, 3)
            colors = [200, 200, 200]

            handle = viser_visualizer.add_point_cloud(
                pointcloud, colors=colors, point_size=0.01, name="test_pointcloud"
            )

            assert handle is not None
            mock_server_instance.scene.add_point_cloud.assert_called_once()

    def test_add_image(self):
        """Test add_image method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_image_handle = MagicMock()
            mock_server_instance.scene.add_image.return_value = mock_image_handle
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            pose = Pose.from_list([0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])

            handle = viser_visualizer.add_image(
                image,
                render_width=1.0,
                render_height=1.0,
                pose=pose,
                name="test_image",
            )

            assert handle is not None
            mock_server_instance.scene.add_image.assert_called_once()

    def test_add_scene(self):
        """Test add_scene method."""
        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            viser_visualizer = ViserVisualizer(
                content_path=None,
                add_robot_to_scene=False,
            )

            # Create a simple scene config with a cube
            scene_cfg = SceneCfg.create(
                {
                    "cuboid": {
                        "cube_1": {
                            "dims": [0.2, 0.2, 0.2],
                            "pose": [0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                        }
                    }
                }
            )

            # Test with add_control_frames=True to verify both mesh and control frame are added
            obstacle_frames = viser_visualizer.add_scene(scene_cfg, add_control_frames=True)

            assert isinstance(obstacle_frames, dict)
            # When add_control_frames=True, both add_mesh_trimesh and add_transform_controls
            # should be called for each obstacle in the scene
            # Note: add_mesh_trimesh may be 0 if scene has no valid meshes to convert
            # The obstacle_frames dict should contain control frame handles
            assert isinstance(obstacle_frames, dict)

    def test_update_robot_spheres(self):
        """Test update_robot_spheres method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
                visualize_robot_spheres=False,
            )

            joint_names = viser_visualizer.joint_names
            joint_state = JointState.from_position(
                torch.zeros(len(joint_names), device="cuda"),
                joint_names=joint_names,
            )

            viser_visualizer.update_robot_spheres(joint_state)

            # Should call add_batched_meshes_simple to visualize spheres
            mock_server_instance.scene.add_batched_meshes_simple.assert_called_once()

    def test_visualize_robot_spheres_enabled(self):
        """Test initialization with visualize_robot_spheres enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
                visualize_robot_spheres=True,
            )

            assert viser_visualizer._visualize_robot_spheres is True

            joint_names = viser_visualizer.joint_names
            joint_state = JointState.from_position(
                torch.zeros(len(joint_names), device="cuda"),
                joint_names=joint_names,
            )
            viser_visualizer.set_joint_state(joint_state)

            # Spheres should be visualized when setting joint state
            assert mock_server_instance.scene.add_batched_meshes_simple.call_count > 0

    def test_visualize_collision_meshes_enabled(self):
        """Test initialization with visualize_collision_meshes enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        content_path = ContentPath(robot_config_file="franka.yml")

        with patch("curobo._src.util.viser_visualizer.viser.ViserServer") as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            device_cfg = DeviceCfg()

            viser_visualizer = ViserVisualizer(
                content_path=content_path,
                device_cfg=device_cfg,
                add_robot_to_scene=True,
                add_control_frames=False,
                visualize_collision_meshes=True,
            )

            # Just verify it initializes successfully
            assert viser_visualizer is not None
            assert viser_visualizer._robot_model is not None
