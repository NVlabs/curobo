# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for CameraObservation."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


class TestCameraObservation:
    """Test CameraObservation class."""

    def test_initialization_default(self):
        """Test default CameraObservation initialization."""
        cam = CameraObservation()
        assert cam.name == "camera_image"
        assert cam.rgb_image is None
        assert cam.depth_image is None
        assert cam.pose is None

    def test_initialization_with_parameters(self, cuda_device_cfg):
        """Test CameraObservation with parameters."""
        rgb = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())

        cam = CameraObservation(
            name="test_camera", rgb_image=rgb, depth_image=depth, resolution=[480, 640]
        )

        assert cam.name == "test_camera"
        assert cam.rgb_image is not None
        assert cam.depth_image is not None
        assert cam.resolution == [480, 640]

    def test_filter_depth(self, cuda_device_cfg):
        """Test filtering depth values below threshold."""
        depth = torch.tensor([[0.005, 0.02, 0.001], [0.03, 0.0, 0.015]], **cuda_device_cfg.as_torch_dict())
        cam = CameraObservation(depth_image=depth)

        cam.filter_depth(distance=0.01)

        # Values below 0.01 should be set to 0
        assert cam.depth_image[0, 0] == 0  # 0.005 -> 0
        assert cam.depth_image[0, 1] == 0.02  # 0.02 stays
        assert cam.depth_image[0, 2] == 0  # 0.001 -> 0
        assert cam.depth_image[1, 1] == 0  # 0.0 stays 0

    def test_shape_property(self, cuda_device_cfg):
        """Test shape property."""
        rgb = torch.rand(2, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam = CameraObservation(rgb_image=rgb)

        assert cam.shape == (2, 480, 640, 3)

    def test_copy_(self, cuda_device_cfg):
        """Test copying from another CameraObservation."""
        rgb1 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth1 = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        pose1 = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        cam1 = CameraObservation(rgb_image=rgb1, depth_image=depth1, pose=pose1, resolution=[480, 640])

        rgb2 = torch.ones(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth2 = torch.ones(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        pose2 = Pose.from_list([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        cam2 = CameraObservation(rgb_image=rgb2, depth_image=depth2, pose=pose2, resolution=[480, 640])

        cam1.copy_(cam2)

        assert torch.allclose(cam1.rgb_image, rgb2)
        assert torch.allclose(cam1.depth_image, depth2)
        assert cam1.resolution == [480, 640]

    def test_copy_with_segmentation(self, cuda_device_cfg):
        """Test copying with segmentation data."""
        seg1 = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(image_segmentation=seg1)

        seg2 = torch.ones(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(image_segmentation=seg2)

        cam1.copy_(cam2)

        assert torch.allclose(cam1.image_segmentation, seg2)

    def test_copy_with_projection_matrix(self, cuda_device_cfg):
        """Test copying with projection matrix."""
        proj1 = torch.rand(1, 4, 4, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(projection_matrix=proj1)

        proj2 = torch.ones(1, 4, 4, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(projection_matrix=proj2)

        cam1.copy_(cam2)

        assert torch.allclose(cam1.projection_matrix, proj2)

    def test_copy_with_projection_rays(self, cuda_device_cfg):
        """Test copying with projection rays."""
        rays1 = torch.rand(480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(projection_rays=rays1)

        rays2 = torch.ones(480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(projection_rays=rays2)

        cam1.copy_(cam2)

        assert torch.allclose(cam1.projection_rays, rays2)

    def test_copy_with_timestamp(self, cuda_device_cfg):
        """Test copying with timestamp."""
        ts1 = torch.tensor([1.0], **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(timestamp=ts1)

        ts2 = torch.tensor([2.0], **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(timestamp=ts2)

        cam1.copy_(cam2)

        assert torch.allclose(cam1.timestamp, ts2)

    def test_clone(self, cuda_device_cfg):
        """Test cloning CameraObservation."""
        rgb = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        cam = CameraObservation(rgb_image=rgb, depth_image=depth, resolution=[480, 640])

        cam_clone = cam.clone()

        assert torch.allclose(cam_clone.rgb_image, rgb)
        assert torch.allclose(cam_clone.depth_image, depth)
        assert cam_clone.resolution == [480, 640]

        # Verify it's a deep copy
        cam_clone.rgb_image[0, 0, 0, 0] = 999.0
        assert cam.rgb_image[0, 0, 0, 0] != 999.0

    def test_clone_with_all_fields(self, cuda_device_cfg):
        """Test cloning with all fields populated."""
        rgb = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        seg = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        proj_mat = torch.rand(1, 4, 4, **cuda_device_cfg.as_torch_dict())
        proj_rays = torch.rand(480, 640, 3, **cuda_device_cfg.as_torch_dict())
        intrinsics = torch.rand(1, 3, 3, **cuda_device_cfg.as_torch_dict())
        pose = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        ts = torch.tensor([1.0], **cuda_device_cfg.as_torch_dict())

        cam = CameraObservation(
            name="test",
            rgb_image=rgb,
            depth_image=depth,
            image_segmentation=seg,
            projection_matrix=proj_mat,
            projection_rays=proj_rays,
            intrinsics=intrinsics,
            pose=pose,
            timestamp=ts,
            resolution=[480, 640],
        )

        cam_clone = cam.clone()

        assert cam_clone.name == "test"
        assert torch.allclose(cam_clone.depth_image, depth)
        assert torch.allclose(cam_clone.image_segmentation, seg)
        assert torch.allclose(cam_clone.intrinsics, intrinsics)
        assert torch.allclose(cam_clone.timestamp, ts)

    def test_to_device(self, cuda_device_cfg, cpu_device_cfg):
        """Test moving camera observation to device."""
        # Start on CPU
        rgb = torch.rand(1, 480, 640, 3, **cpu_device_cfg.as_torch_dict())
        depth = torch.rand(1, 480, 640, **cpu_device_cfg.as_torch_dict())
        cam = CameraObservation(rgb_image=rgb, depth_image=depth)

        cam_moved = cam.to(cuda_device_cfg.device)

        assert cam_moved.rgb_image.device == cuda_device_cfg.device
        assert cam_moved.depth_image.device == cuda_device_cfg.device

    def test_to_device_with_all_fields(self, cuda_device_cfg, cpu_device_cfg):
        """Test moving all fields to device."""
        seg = torch.rand(1, 480, 640, **cpu_device_cfg.as_torch_dict())
        proj_mat = torch.rand(1, 4, 4, **cpu_device_cfg.as_torch_dict())
        proj_rays = torch.rand(480, 640, 3, **cpu_device_cfg.as_torch_dict())
        intrinsics = torch.rand(1, 3, 3, **cpu_device_cfg.as_torch_dict())
        pose = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cpu_device_cfg)
        ts = torch.tensor([1.0], **cpu_device_cfg.as_torch_dict())

        cam = CameraObservation(
            image_segmentation=seg,
            projection_matrix=proj_mat,
            projection_rays=proj_rays,
            intrinsics=intrinsics,
            pose=pose,
            timestamp=ts,
        )

        cam_moved = cam.to(cuda_device_cfg.device)

        assert cam_moved.image_segmentation.device == cuda_device_cfg.device
        assert cam_moved.projection_matrix.device == cuda_device_cfg.device
        assert cam_moved.projection_rays.device == cuda_device_cfg.device
        assert cam_moved.intrinsics.device == cuda_device_cfg.device
        assert cam_moved.timestamp.device == cuda_device_cfg.device
        assert cam_moved.pose.device == cuda_device_cfg.device

    def test_update_projection_rays(self, cuda_device_cfg):
        """Test updating projection rays from intrinsics."""
        depth = torch.rand(480, 640, **cuda_device_cfg.as_torch_dict())
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        intrinsics[0, 0, 0] = 500.0  # fx
        intrinsics[0, 1, 1] = 500.0  # fy
        intrinsics[0, 0, 2] = 320.0  # cx
        intrinsics[0, 1, 2] = 240.0  # cy

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)

        cam.update_projection_rays()

        assert cam.projection_rays is not None
        # Shape is (batch=1, height*width, 3)
        assert cam.projection_rays.shape == (1, 480 * 640, 3)

    def test_update_projection_rays_batch(self, cuda_device_cfg):
        """Test updating projection rays with batched intrinsics."""
        depth = torch.rand(2, 480, 640, **cuda_device_cfg.as_torch_dict())
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0).repeat(2, 1, 1)
        intrinsics[:, 0, 0] = 500.0
        intrinsics[:, 1, 1] = 500.0

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)

        cam.update_projection_rays()

        assert cam.projection_rays is not None

    def test_update_projection_rays_2d_intrinsics(self, cuda_device_cfg):
        """Test updating projection rays with 2D intrinsics (without batch dimension)."""
        depth = torch.rand(480, 640, **cuda_device_cfg.as_torch_dict())
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict())
        intrinsics[0, 0] = 500.0
        intrinsics[1, 1] = 500.0
        intrinsics[0, 2] = 320.0
        intrinsics[1, 2] = 240.0

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)

        cam.update_projection_rays()

        assert cam.projection_rays is not None
        # Should handle 2D intrinsics and unsqueeze them
        assert cam.projection_rays.shape == (1, 480 * 640, 3)

    def test_get_pointcloud(self, cuda_device_cfg):
        """Test getting pointcloud from depth."""
        depth = torch.ones(480, 640, **cuda_device_cfg.as_torch_dict()) * 0.5
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        intrinsics[0, 0, 0] = 500.0
        intrinsics[0, 1, 1] = 500.0
        intrinsics[0, 0, 2] = 320.0
        intrinsics[0, 1, 2] = 240.0

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)

        pointcloud = cam.get_pointcloud()

        assert pointcloud.shape[-1] == 3  # x, y, z coordinates

    def test_get_pointcloud_with_pose_transform(self, cuda_device_cfg):
        """Test getting pointcloud with pose transformation."""
        depth = torch.ones(480, 640, **cuda_device_cfg.as_torch_dict()) * 0.5
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        intrinsics[0, 0, 0] = 500.0
        intrinsics[0, 1, 1] = 500.0
        intrinsics[0, 0, 2] = 320.0
        intrinsics[0, 1, 2] = 240.0
        pose = Pose.from_list([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics, pose=pose)

        pointcloud = cam.get_pointcloud(project_to_pose=True)

        assert pointcloud.shape[-1] == 3

    def test_get_pointcloud_batched(self, cuda_device_cfg):
        """Test getting pointcloud from batched depth."""
        depth = torch.ones(2, 480, 640, **cuda_device_cfg.as_torch_dict()) * 0.5
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        intrinsics[0, 0, 0] = 500.0
        intrinsics[0, 1, 1] = 500.0

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)

        pointcloud = cam.get_pointcloud()

        assert pointcloud.shape[0] == 2

    def test_stack(self, cuda_device_cfg):
        """Test stacking two camera observations."""
        rgb1 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth1 = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(rgb_image=rgb1, depth_image=depth1, resolution=[480, 640])

        rgb2 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        depth2 = torch.rand(1, 480, 640, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(rgb_image=rgb2, depth_image=depth2, resolution=[480, 640])

        cam_stacked = cam1.stack(cam2)

        assert cam_stacked.rgb_image.shape[0] == 2
        assert cam_stacked.depth_image.shape[0] == 2

    def test_stack_with_dim(self, cuda_device_cfg):
        """Test stacking with custom dimension."""
        rgb1 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(rgb_image=rgb1)

        rgb2 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(rgb_image=rgb2)

        cam_stacked = cam1.stack(cam2, dim=0)

        assert cam_stacked.rgb_image.shape[0] == 2

    def test_stack_with_all_fields(self, cuda_device_cfg):
        """Test stacking with all fields."""
        pose1 = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        ts1 = torch.tensor([1.0], **cuda_device_cfg.as_torch_dict())
        intr1 = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        cam1 = CameraObservation(pose=pose1, timestamp=ts1, intrinsics=intr1)

        pose2 = Pose.from_list([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        ts2 = torch.tensor([2.0], **cuda_device_cfg.as_torch_dict())
        intr2 = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        cam2 = CameraObservation(pose=pose2, timestamp=ts2, intrinsics=intr2)

        cam_stacked = cam1.stack(cam2)

        assert cam_stacked.timestamp.shape[0] == 2
        assert cam_stacked.intrinsics.shape[0] == 2
        assert cam_stacked.pose.batch_size == 2

    def test_extract_depth_from_structured_pointcloud(self, cuda_device_cfg):
        """Test extracting depth from structured pointcloud."""
        # Create pointcloud with shape (h, w, 3) - will be batched internally
        pointcloud = torch.ones(10, 10, 3, **cuda_device_cfg.as_torch_dict())
        pointcloud[:, :, 2] = 0.3  # Set z-coordinate to 0.3

        cam = CameraObservation()
        depth_reconstructed = cam.extract_depth_from_structured_pointcloud(pointcloud)

        # Output should be batched (1, h, w)
        assert depth_reconstructed.shape == (1, 10, 10)
        assert depth_reconstructed.device == cuda_device_cfg.device
        # Check that z-coordinates were extracted
        assert torch.allclose(
            depth_reconstructed, torch.ones(1, 10, 10, **cuda_device_cfg.as_torch_dict()) * 0.3
        )

    def test_extract_depth_with_output_image(self, cuda_device_cfg):
        """Test extracting depth with pre-allocated output image."""
        pointcloud = torch.ones(10, 10, 3, **cuda_device_cfg.as_torch_dict())
        pointcloud[:, :, 2] = 0.7
        output_image = torch.zeros((10, 10), **cuda_device_cfg.as_torch_dict())

        cam = CameraObservation()
        depth_reconstructed = cam.extract_depth_from_structured_pointcloud(
            pointcloud, output_image
        )

        # Output should be batched (1, h, w)
        assert depth_reconstructed.shape == (1, 10, 10)
        assert torch.is_tensor(depth_reconstructed)
        assert torch.allclose(
            depth_reconstructed, torch.ones(1, 10, 10, **cuda_device_cfg.as_torch_dict()) * 0.7
        )

    def test_extract_depth_batched(self, cuda_device_cfg):
        """Test extracting depth from batched structured pointcloud."""
        # Create batched pointcloud (b, h, w, 3)
        pointcloud = torch.ones(2, 10, 10, 3, **cuda_device_cfg.as_torch_dict())
        pointcloud[0, :, :, 2] = 0.3
        pointcloud[1, :, :, 2] = 0.6

        cam = CameraObservation()
        depth_reconstructed = cam.extract_depth_from_structured_pointcloud(pointcloud)

        assert depth_reconstructed.shape == (2, 10, 10)
        assert torch.allclose(
            depth_reconstructed[0], torch.ones(10, 10, **cuda_device_cfg.as_torch_dict()) * 0.3
        )
        assert torch.allclose(
            depth_reconstructed[1], torch.ones(10, 10, **cuda_device_cfg.as_torch_dict()) * 0.6
        )

    def test_extract_depth_from_structured_pointcloud_works_without_depth_image(self, cuda_device_cfg):
        """Test that extract_depth works without pre-existing depth_image."""
        cam = CameraObservation()
        pointcloud = torch.rand(10, 10, 3, **cuda_device_cfg.as_torch_dict())
        pointcloud[:, :, 2] = 0.5  # Set depth values

        # Should work - extracts shape from pointcloud
        depth = cam.extract_depth_from_structured_pointcloud(pointcloud)
        assert depth.shape == (1, 10, 10)
        assert torch.allclose(depth, torch.ones(1, 10, 10, **cuda_device_cfg.as_torch_dict()) * 0.5)

    def test_filter_depth_none(self):
        """Test filter_depth when depth_image is None - should expose bug."""
        cam = CameraObservation()

        # This should fail because depth_image is None
        with pytest.raises(ValueError, match="depth_image is None"):
            cam.filter_depth(distance=0.01)

    def test_shape_none(self):
        """Test shape property when rgb_image is None - should expose bug."""
        cam = CameraObservation()

        # This should fail because rgb_image is None
        with pytest.raises(ValueError, match="rgb_image is None"):
            _ = cam.shape

    def test_get_pointcloud_no_depth_image(self, cuda_device_cfg):
        """Test get_pointcloud when depth_image is None - should expose bug."""
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        cam = CameraObservation(intrinsics=intrinsics)

        # This should fail because depth_image is None
        with pytest.raises(ValueError, match="depth_image is None"):
            cam.get_pointcloud()

    def test_update_projection_rays_no_depth_image(self, cuda_device_cfg):
        """Test update_projection_rays when depth_image is None - should expose bug."""
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        cam = CameraObservation(intrinsics=intrinsics)

        # This should fail because depth_image is None
        with pytest.raises(ValueError, match="depth_image is None"):
            cam.update_projection_rays()

    def test_update_projection_rays_no_intrinsics(self, cuda_device_cfg):
        """Test update_projection_rays when intrinsics is None."""
        depth = torch.ones(480, 640, **cuda_device_cfg.as_torch_dict())
        cam = CameraObservation(depth_image=depth)

        # This should fail because intrinsics is None
        with pytest.raises(ValueError, match="intrinsics is None"):
            cam.update_projection_rays()

    def test_copy_with_none_fields(self, cuda_device_cfg):
        """Test copy_ when source has fields but destination doesn't."""
        cam1 = CameraObservation()
        rgb2 = torch.ones(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(rgb_image=rgb2)

        # copy_ only copies fields that exist in cam1 (destination)
        cam1.copy_(cam2)

        # cam1.rgb_image should still be None because it wasn't initialized
        assert cam1.rgb_image is None

    def test_clone_preserves_name(self, cuda_device_cfg):
        """Test that clone preserves the name field."""
        rgb = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam = CameraObservation(name="custom_camera", rgb_image=rgb)

        cam_clone = cam.clone()

        assert cam_clone.name == "custom_camera"

    def test_stack_preserves_name(self, cuda_device_cfg):
        """Test that stack preserves the name field."""
        rgb1 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(name="camera1", rgb_image=rgb1)

        rgb2 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(name="camera2", rgb_image=rgb2)

        cam_stacked = cam1.stack(cam2)

        assert cam_stacked.name == "camera1"

    def test_filter_depth_preserves_dtype(self, cuda_device_cfg):
        """Test that filter_depth preserves the dtype."""
        depth = torch.tensor(
            [[0.005, 0.02], [0.03, 0.015]], dtype=torch.float64, device=cuda_device_cfg.device
        )
        cam = CameraObservation(depth_image=depth)

        cam.filter_depth(distance=0.01)

        assert cam.depth_image.dtype == torch.float64

    def test_to_returns_self(self, cuda_device_cfg):
        """Test that to() method returns self for method chaining."""
        rgb = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam = CameraObservation(rgb_image=rgb)

        result = cam.to(cuda_device_cfg.device)

        assert result is cam

    def test_stack_with_none_poses(self, cuda_device_cfg):
        """Test stacking when poses are None."""
        rgb1 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam1 = CameraObservation(rgb_image=rgb1, pose=None)

        rgb2 = torch.rand(1, 480, 640, 3, **cuda_device_cfg.as_torch_dict())
        cam2 = CameraObservation(rgb_image=rgb2, pose=None)

        cam_stacked = cam1.stack(cam2)

        assert cam_stacked.pose is None

    def test_get_pointcloud_updates_rays_if_missing(self, cuda_device_cfg):
        """Test that get_pointcloud creates projection_rays if they don't exist."""
        depth = torch.ones(480, 640, **cuda_device_cfg.as_torch_dict()) * 0.5
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        intrinsics[0, 0, 0] = 500.0
        intrinsics[0, 1, 1] = 500.0

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)
        assert cam.projection_rays is None

        pointcloud = cam.get_pointcloud()

        assert cam.projection_rays is not None
        assert pointcloud is not None

    def test_update_projection_rays_updates_existing(self, cuda_device_cfg):
        """Test that update_projection_rays updates existing rays."""
        depth = torch.ones(480, 640, **cuda_device_cfg.as_torch_dict())
        intrinsics = torch.eye(3, **cuda_device_cfg.as_torch_dict()).unsqueeze(0)
        intrinsics[0, 0, 0] = 500.0
        intrinsics[0, 1, 1] = 500.0

        cam = CameraObservation(depth_image=depth, intrinsics=intrinsics)
        cam.update_projection_rays()
        old_rays = cam.projection_rays.clone()

        # Update intrinsics
        intrinsics[0, 0, 0] = 600.0
        cam.update_projection_rays()

        # Rays should be different after update
        assert not torch.allclose(cam.projection_rays, old_rays)

