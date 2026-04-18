# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    """Create tensor configuration for both CPU and GPU."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))

class TestPose:
    """Test Pose class."""

    def test_pose_creation_with_position_and_quaternion(self, device_cfg):
        """Test creating pose with position and quaternion."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        assert pose.batch_size == 1
        assert pose.position.shape == (1, 3)
        assert pose.quaternion.shape == (1, 4)

    def test_pose_creation_with_rotation_matrix(self, device_cfg):
        """Test creating pose with rotation matrix."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        rotation = torch.eye(3, **device_cfg.as_torch_dict()).unsqueeze(0)
        pose = Pose(position=position, rotation=rotation)
        assert pose.quaternion is not None
        assert pose.quaternion.shape == (1, 4)

    def test_pose_creation_position_only(self, device_cfg):
        """Test creating pose with position only (quaternion auto-initialized)."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position)
        assert pose.quaternion is not None
        assert pose.quaternion[0, 0] == 1.0  # Identity quaternion

    def test_pose_normalize_rotation(self, device_cfg):
        """Test pose with normalize_rotation flag."""
        position = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[2.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        assert torch.allclose(
            torch.linalg.norm(pose.quaternion, dim=-1), torch.ones(1, **device_cfg.as_torch_dict())
        )

    def test_pose_batch_size(self, device_cfg):
        """Test pose with batch size."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        assert pose.batch_size == 5
        assert len(pose) == 5

    def test_pose_equality(self, device_cfg):
        """Test pose equality comparison."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position.clone(), quaternion=quaternion.clone())
        pose2 = Pose(position=position.clone(), quaternion=quaternion.clone())
        assert pose1 == pose2

    def test_pose_inequality(self, device_cfg):
        """Test pose inequality."""
        position1 = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[1.0, 2.0, 3.1]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion.clone())
        pose2 = Pose(position=position2, quaternion=quaternion.clone())
        assert not (pose1 == pose2)

    def test_pose_detach(self, device_cfg):
        """Test pose detach."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position.requires_grad_(True)
        quaternion.requires_grad_(True)
        pose = Pose(position=position, quaternion=quaternion)
        pose_detached = pose.detach()
        assert not pose_detached.position.requires_grad
        assert not pose_detached.quaternion.requires_grad

    def test_pose_requires_grad(self, device_cfg):
        """Test setting requires_grad on pose."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose.requires_grad_(True)
        assert pose.position.requires_grad
        assert pose.quaternion.requires_grad

    def test_pose_from_matrix(self, device_cfg):
        """Test creating pose from transformation matrix."""
        matrix = torch.eye(4, **device_cfg.as_torch_dict())
        matrix[:3, 3] = torch.tensor([1.0, 2.0, 3.0], **device_cfg.as_torch_dict())
        pose = Pose.from_matrix(matrix)
        assert torch.allclose(pose.position, torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict()))

    def test_pose_from_matrix_numpy(self, device_cfg):
        """Test creating pose from numpy matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = np.array([1.0, 2.0, 3.0])
        pose = Pose.from_matrix(matrix)
        assert pose.position is not None

    def test_pose_get_rotation_matrix(self, device_cfg):
        """Test getting rotation matrix from pose."""
        position = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        rotation = pose.get_rotation_matrix()
        assert rotation.shape == (1, 3, 3)
        assert torch.allclose(rotation, torch.eye(3, **device_cfg.as_torch_dict()).unsqueeze(0))

    def test_pose_stack(self, device_cfg):
        """Test stacking two poses."""
        position1 = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[0.0, 1.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        stacked = pose1.stack(pose2)
        assert stacked.position.shape == (2, 3)

    def test_pose_repeat(self, device_cfg):
        """Test repeating pose."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        repeated = pose.repeat(3)
        assert repeated.batch_size == 3
        assert repeated.position.shape == (3, 3)

    def test_pose_unsqueeze(self, device_cfg):
        """Test unsqueezing pose dimensions."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_unsqueezed = pose.unsqueeze(1)
        assert pose_unsqueezed.position.shape == (1, 1, 3)

    def test_pose_squeeze(self, device_cfg):
        """Test squeezing pose dimensions."""
        position = torch.tensor([[[1.0, 2.0, 3.0]]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_squeezed = pose.squeeze(1)
        assert pose_squeezed.position.shape == (1, 3)

    def test_pose_repeat_seeds(self, device_cfg):
        """Test repeating seeds."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        repeated = pose.repeat_seeds(4)
        assert repeated.batch_size == 4

    def test_pose_getitem(self, device_cfg):
        """Test indexing pose."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        pose_indexed = pose[2]
        assert pose_indexed.position.shape == (1, 3)

    def test_pose_getitem_slice(self, device_cfg):
        """Test slicing pose."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        pose_sliced = pose[1:3]
        assert pose_sliced.position.shape == (2, 3)

    def test_pose_setitem(self, device_cfg):
        """Test setting pose by index."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        new_position = torch.tensor([[9.0, 9.0, 9.0]], **device_cfg.as_torch_dict())
        new_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        new_pose = Pose(position=new_position, quaternion=new_quaternion)
        pose[2] = new_pose
        assert torch.allclose(pose.position[2], new_position.squeeze())

    def test_pose_get_index(self, device_cfg):
        """Test getting pose at specific index."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        pose_at_2 = pose.get_index(2)
        assert pose_at_2.position.shape == (1, 3)

    def test_pose_apply_kernel(self, device_cfg):
        """Test applying kernel matrix to pose."""
        position = torch.rand(4, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(4, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        kernel = torch.eye(4, **device_cfg.as_torch_dict())
        pose_kernel = pose.apply_kernel(kernel)
        assert torch.allclose(pose_kernel.position, pose.position)

    def test_pose_from_numpy(self, device_cfg):
        """Test creating pose from numpy arrays."""
        position = np.array([[1.0, 2.0, 3.0]])
        quaternion = np.array([[1.0, 0.0, 0.0, 0.0]])
        pose = Pose.from_numpy(position, quaternion, device_cfg)
        assert pose.position.shape == (1, 3)
        assert pose.quaternion.shape == (1, 4)

    def test_pose_from_list(self, device_cfg):
        """Test creating pose from list."""
        pose_list = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        pose = Pose.from_list(pose_list, device_cfg)
        assert pose.position.shape == (1, 3)
        assert pose.quaternion.shape == (1, 4)

    def test_pose_from_list_xyzw(self, device_cfg):
        """Test creating pose from list with xyzw quaternion format."""
        pose_list = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        pose = Pose.from_list(pose_list, device_cfg, q_xyzw=True)
        assert pose.quaternion[0, 0] == 1.0  # w component

    def test_pose_from_batch_list(self, device_cfg):
        """Test creating pose from batch list."""
        pose_list = [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        pose = Pose.from_batch_list(pose_list, device_cfg)
        assert pose.batch_size == 2

    def test_pose_from_batch_list_xyzw(self, device_cfg):
        """Test creating pose from batch list with xyzw format."""
        pose_list = [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        pose = Pose.from_batch_list(pose_list, device_cfg, q_xyzw=True)
        assert pose.quaternion[0, 0] == 1.0

    def test_pose_tolist(self, device_cfg):
        """Test converting pose to list."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_list = pose.tolist()
        assert len(pose_list) == 7
        assert pose_list[0] == 1.0

    def test_pose_tolist_xyzw(self, device_cfg):
        """Test converting pose to list with xyzw format."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_list = pose.tolist(q_xyzw=True)
        assert pose_list[6] == 1.0  # w at end

    def test_pose_clone(self, device_cfg):
        """Test cloning pose."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_cloned = pose.clone()
        assert torch.allclose(pose_cloned.position, pose.position)
        # Verify it's a copy, not a reference
        pose_cloned.position[0, 0] = 999.0
        assert pose.position[0, 0] == 1.0

    def test_pose_to_device(self, device_cfg):
        """Test moving pose to device."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_moved = pose.to(device_cfg=device_cfg)
        assert pose_moved.position.device == device_cfg.device

    def test_pose_get_matrix(self, device_cfg):
        """Test getting transformation matrix."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        matrix = pose.get_matrix()
        assert matrix.shape == (1, 4, 4)
        assert torch.allclose(matrix[0, :3, 3], position.squeeze())

    def test_pose_get_affine_matrix(self, device_cfg):
        """Test getting affine transformation matrix."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        matrix = pose.get_affine_matrix()
        assert matrix.shape == (1, 3, 4)
        assert torch.allclose(matrix[0, :, 3], position.squeeze())

    def test_pose_inverse(self, device_cfg):
        """Test pose inverse."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_inv = pose.inverse()
        # Verify that pose * pose_inv gives identity
        identity = pose.multiply(pose_inv)
        assert torch.allclose(
            identity.position, torch.zeros(1, 3, **device_cfg.as_torch_dict()), atol=1e-6
        )

    def test_pose_get_pose_vector(self, device_cfg):
        """Test getting concatenated pose vector."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        pose_vec = pose.get_pose_vector()
        assert pose_vec.shape == (1, 7)

    def test_pose_copy(self, device_cfg):
        """Test copying pose data."""
        position1 = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[4.0, 5.0, 6.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[0.707, 0.707, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        pose1.copy_(pose2)
        assert torch.allclose(pose1.position, position2)

    def test_pose_cat(self, device_cfg):
        """Test concatenating list of poses."""
        pose_list = []
        for i in range(3):
            position = torch.rand(2, 3, **device_cfg.as_torch_dict())
            quaternion = torch.rand(2, 4, **device_cfg.as_torch_dict())
            pose_list.append(Pose(position=position, quaternion=quaternion, normalize_rotation=True))
        pose_cat = Pose.cat(pose_list)
        assert pose_cat.batch_size == 6

    def test_pose_distance(self, device_cfg):
        """Test computing distance between poses."""
        position1 = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        p_dist, q_dist = pose1.distance(pose2)
        assert torch.allclose(p_dist, torch.tensor([1.0], **device_cfg.as_torch_dict()))

    def test_pose_angular_distance(self, device_cfg):
        """Test computing angular distance."""
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[0.707, 0.707, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position.clone(), quaternion=quaternion1)
        pose2 = Pose(position=position.clone(), quaternion=quaternion2)
        ang_dist = pose1.angular_distance(pose2)
        assert ang_dist.shape == (1, 1)

    def test_pose_angular_distance_phi3(self, device_cfg):
        """Test computing angular distance with phi3 metric."""
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[0.707, 0.707, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position.clone(), quaternion=quaternion1)
        pose2 = Pose(position=position.clone(), quaternion=quaternion2)
        ang_dist = pose1.angular_distance(pose2, use_phi3=True)
        assert ang_dist.shape == (1,)

    def test_pose_linear_distance(self, device_cfg):
        """Test computing linear distance."""
        position1 = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[3.0, 4.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion.clone())
        pose2 = Pose(position=position2, quaternion=quaternion.clone())
        lin_dist = pose1.linear_distance(pose2)
        assert torch.allclose(lin_dist, torch.tensor([5.0], **device_cfg.as_torch_dict()))

    def test_pose_multiply(self, device_cfg):
        """Test pose multiplication."""
        position1 = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[0.0, 1.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        result = pose1.multiply(pose2)
        assert result.position.shape == (1, 3)

    def test_pose_multiply_broadcast(self, device_cfg):
        """Test pose multiplication with broadcasting."""
        position1 = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion2 = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2, normalize_rotation=True)
        result = pose1.multiply(pose2)
        assert result.batch_size == 5

    def test_pose_transform_points(self, device_cfg):
        """Test transforming points with pose."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        points = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        transformed = pose.transform_points(points)
        assert torch.allclose(transformed, position)

    def test_pose_batch_transform_points(self, device_cfg):
        """Test batch transforming points."""
        position = torch.rand(3, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(3, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, normalize_rotation=True)
        points = torch.rand(3, 10, 3, **device_cfg.as_torch_dict())
        transformed = pose.batch_transform_points(points)
        assert transformed.shape == (3, 10, 3)

    def test_pose_compute_offset_pose(self, device_cfg):
        """Test computing offset pose."""
        position = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        offset_pos = torch.tensor([[0.0, 1.0, 0.0]], **device_cfg.as_torch_dict())
        offset_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        offset = Pose(position=offset_pos, quaternion=offset_quat)
        result = pose.compute_offset_pose(offset)
        assert result.position.shape == (1, 3)

    def test_pose_compute_local_pose(self, device_cfg):
        """Test computing local pose."""
        position1 = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[2.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        local_pose = pose1.compute_local_pose(pose2)
        assert local_pose.position.shape == (1, 3)

    def test_pose_contiguous(self, device_cfg):
        """Test making pose contiguous."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict()).transpose(0, 1)
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position.transpose(0, 1), quaternion=quaternion)
        pose_contig = pose.contiguous()
        assert pose_contig.position.is_contiguous()

    def test_pose_device_property(self, device_cfg):
        """Test pose device property."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        assert pose.device == device_cfg.device

    def test_pose_shape_property(self, device_cfg):
        """Test pose shape property."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        assert pose.shape == (5, 3)

    def test_pose_ndim_property(self, device_cfg):
        """Test pose ndim property."""
        position = torch.rand(5, 3, **device_cfg.as_torch_dict())
        quaternion = torch.rand(5, 4, **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        assert pose.ndim == 2

    def test_pose_get_numpy_matrix(self, device_cfg):
        """Test getting numpy matrix."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        matrix = pose.get_numpy_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (1, 4, 4)

    def test_pose_get_numpy_affine_matrix(self, device_cfg):
        """Test getting numpy affine matrix."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        matrix = pose.get_numpy_affine_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (1, 3, 4)

    def test_pose_equality_shape_mismatch(self, device_cfg):
        """Test pose equality with shape mismatch (coverage for print statement)."""
        position1 = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        assert not (pose1 == pose2)

    def test_pose_equality_quaternion_mismatch(self, device_cfg):
        """Test pose equality with quaternion distance > 1e-6."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[0.999, 0.01, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = quaternion2 / torch.linalg.norm(quaternion2)
        pose1 = Pose(position=position.clone(), quaternion=quaternion1)
        pose2 = Pose(position=position.clone(), quaternion=quaternion2)
        assert not (pose1 == pose2)

    def test_pose_get_rotation_with_rotation_field(self, device_cfg):
        """Test get_rotation when rotation field is set."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        rotation = torch.eye(3, **device_cfg.as_torch_dict()).unsqueeze(0)
        pose = Pose(position=position, rotation=rotation)
        result = pose.get_rotation()
        assert result is not None
        assert torch.allclose(result, rotation)

    def test_pose_get_rotation_none(self, device_cfg):
        """Test get_rotation when both quaternion and rotation are None."""
        pose = Pose()
        result = pose.get_rotation()
        assert result is None

    def test_pose_repeat_single(self, device_cfg):
        """Test repeat with n=1 (should return self)."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        repeated = pose.repeat(1)
        assert repeated is pose

    def test_pose_unsqueeze_with_rotation(self, device_cfg):
        """Test unsqueeze preserves rotation field."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        rotation = torch.eye(3, **device_cfg.as_torch_dict()).unsqueeze(0)
        pose = Pose(position=position, rotation=rotation)
        pose_unsqueezed = pose.unsqueeze(0)
        assert pose_unsqueezed.rotation is not None
        assert pose_unsqueezed.rotation.shape == (1, 1, 3, 3)

    def test_pose_squeeze_with_rotation(self, device_cfg):
        """Test squeeze preserves rotation field."""
        position = torch.tensor([[[1.0, 2.0, 3.0]]], **device_cfg.as_torch_dict())
        rotation = torch.eye(3, **device_cfg.as_torch_dict()).unsqueeze(0).unsqueeze(0)
        quaternion = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion, rotation=rotation)
        pose_squeezed = pose.squeeze(0)
        assert pose_squeezed.rotation is not None

    def test_pose_repeat_seeds_none_position(self, device_cfg):
        """Test repeat_seeds with None position returns new Pose."""
        pose = Pose(position=None, quaternion=None)
        result = pose.repeat_seeds(5)
        assert result.position is None
        assert result.quaternion is None

    def test_pose_repeat_seeds_single(self, device_cfg):
        """Test repeat_seeds with num_seeds=1."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        result = pose.repeat_seeds(1)
        assert torch.allclose(result.position, position)

    def test_pose_apply_kernel_none_position(self, device_cfg):
        """Test apply_kernel with None position returns self."""
        pose = Pose(position=None, quaternion=None)
        kernel = torch.eye(2, **device_cfg.as_torch_dict())
        result = pose.apply_kernel(kernel)
        assert result is pose

    def test_pose_to_list_alias(self, device_cfg):
        """Test to_list method (alias for tolist)."""
        position = torch.tensor([1.0, 2.0, 3.0], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        result = pose.to_list()
        assert len(result) == 7
        assert result[:3] == [1.0, 2.0, 3.0]

    def test_pose_to_none_raises(self, device_cfg):
        """Test Pose.to() with neither device_cfg nor device raises error."""
        position = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        with pytest.raises(Exception):
            pose.to()

    def test_pose_copy_none_raises(self, device_cfg):
        """Test copy_() with None position/quaternion raises error."""
        position1 = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=None, quaternion=None)
        with pytest.raises(Exception):
            pose1.copy_(pose2)

    def test_pose_copy_shape_mismatch_raises(self, device_cfg):
        """Test copy_() with shape mismatch raises error."""
        position1 = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        position2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], **device_cfg.as_torch_dict())
        quaternion1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose1 = Pose(position=position1, quaternion=quaternion1)
        pose2 = Pose(position=position2, quaternion=quaternion2)
        with pytest.raises(Exception):
            pose1.copy_(pose2)

    def test_pose_transform_points_3d(self, device_cfg):
        """Test transform_points with 3D points input."""
        position = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        points = torch.tensor([[[1.0, 0.0, 0.0]]], **device_cfg.as_torch_dict())
        result = pose.transform_points(points)
        assert result.shape == (1, 3)

    def test_pose_batch_transform_points_2d_raises(self, device_cfg):
        """Test batch_transform_points with 2D points raises error."""
        position = torch.tensor([[0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose(position=position, quaternion=quaternion)
        points = torch.tensor([[1.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        with pytest.raises(Exception):
            pose.batch_transform_points(points)


class TestPoseFromEulerXyzIntrinsic:
    """Tests for Pose.from_euler_xyz_intrinsic and _euler_xyz_intrinsic_to_quaternion."""

    def test_identity_rotation(self, device_cfg):
        """Zero Euler angles should produce identity quaternion."""
        euler = torch.zeros(1, 3, **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        expected_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], **device_cfg.as_torch_dict())
        assert torch.allclose(pose.quaternion, expected_quat, atol=1e-6)

    def test_pure_yaw_90(self, device_cfg):
        """90-degree yaw (rz=pi/2) should rotate X axis to Y axis."""
        euler = torch.tensor([[0.0, 0.0, torch.pi / 2]], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        rot = pose.get_rotation_matrix().squeeze(0)
        x_axis_rotated = rot @ torch.tensor([1.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        assert torch.allclose(
            x_axis_rotated,
            torch.tensor([0.0, 1.0, 0.0], **device_cfg.as_torch_dict()),
            atol=1e-5,
        )

    def test_pure_pitch_90(self, device_cfg):
        """90-degree pitch (ry=pi/2) should rotate X axis to -Z axis."""
        euler = torch.tensor([[0.0, torch.pi / 2, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        rot = pose.get_rotation_matrix().squeeze(0)
        x_axis_rotated = rot @ torch.tensor([1.0, 0.0, 0.0], **device_cfg.as_torch_dict())
        assert torch.allclose(
            x_axis_rotated,
            torch.tensor([0.0, 0.0, -1.0], **device_cfg.as_torch_dict()),
            atol=1e-5,
        )

    def test_pure_roll_90(self, device_cfg):
        """90-degree roll (rx=pi/2) should rotate Y axis to Z axis."""
        euler = torch.tensor([[torch.pi / 2, 0.0, 0.0]], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        rot = pose.get_rotation_matrix().squeeze(0)
        y_axis_rotated = rot @ torch.tensor([0.0, 1.0, 0.0], **device_cfg.as_torch_dict())
        assert torch.allclose(
            y_axis_rotated,
            torch.tensor([0.0, 0.0, 1.0], **device_cfg.as_torch_dict()),
            atol=1e-5,
        )

    def test_unit_quaternion(self, device_cfg):
        """Result should always be a unit quaternion."""
        euler = torch.tensor([[0.3, -0.7, 1.2]], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        norm = torch.linalg.norm(pose.quaternion, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

    def test_with_position(self, device_cfg):
        """Position should be passed through unchanged."""
        euler = torch.tensor([[0.1, 0.2, 0.3]], **device_cfg.as_torch_dict())
        pos = torch.tensor([[1.0, 2.0, 3.0]], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler, position=pos)
        assert torch.allclose(pose.position, pos)

    def test_no_position_gives_zeros(self, device_cfg):
        """Omitting position should default to zeros."""
        euler = torch.tensor([[0.1, 0.2, 0.3]], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        assert torch.allclose(
            pose.position, torch.zeros(1, 3, **device_cfg.as_torch_dict())
        )

    def test_batch_input(self, device_cfg):
        """Should handle batched Euler angles."""
        euler = torch.rand(5, 3, **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        assert pose.quaternion.shape == (5, 4)
        assert pose.position.shape == (5, 3)

    def test_1d_input_unsqueezed(self, device_cfg):
        """A single 1D [3] tensor should be unsqueezed to [1, 3]."""
        euler = torch.tensor([0.1, 0.2, 0.3], **device_cfg.as_torch_dict())
        pose = Pose.from_euler_xyz_intrinsic(euler)
        assert pose.quaternion.shape == (1, 4)

    def test_differs_from_extrinsic(self, device_cfg):
        """Intrinsic and extrinsic should differ for non-trivial angles."""
        euler = torch.tensor([[0.5, 0.3, 0.7]], **device_cfg.as_torch_dict())
        pose_intrinsic = Pose.from_euler_xyz_intrinsic(euler)
        pose_extrinsic = Pose.from_euler_xyz(euler)
        assert not torch.allclose(
            pose_intrinsic.quaternion, pose_extrinsic.quaternion, atol=1e-4
        )

    def test_single_axis_matches_extrinsic(self, device_cfg):
        """For single-axis rotations, intrinsic and extrinsic should agree."""
        for axis in range(3):
            euler = torch.zeros(1, 3, **device_cfg.as_torch_dict())
            euler[0, axis] = 0.8
            pose_intrinsic = Pose.from_euler_xyz_intrinsic(euler)
            pose_extrinsic = Pose.from_euler_xyz(euler)
            assert torch.allclose(
                pose_intrinsic.quaternion, pose_extrinsic.quaternion, atol=1e-6
            )
