# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for JointState."""

# Standard Library
from typing import List

# Third Party
import numpy as np
import pytest
import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg


class TestJointState:
    """Test JointState class."""

    @pytest.fixture
    def joint_names(self) -> List[str]:
        """Get sample joint names."""
        return ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    @pytest.fixture
    def device_cfg(self) -> DeviceCfg:
        """Get tensor device configuration."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def sample_joint_state(self, joint_names, device_cfg) -> JointState:
        """Create sample joint state."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        return JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names,
            device_cfg=device_cfg,
        )

    def test_initialization_basic(self, device_cfg):
        """Test basic JointState initialization."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position)
        assert joint_state.position is not None
        assert torch.allclose(joint_state.position, position)

    def test_initialization_with_all_fields(self, joint_names, device_cfg):
        """Test JointState initialization with all fields."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.tensor([0.1], device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names,
            device_cfg=device_cfg,
            dt=dt,
        )

        assert torch.allclose(joint_state.position, position)
        assert torch.allclose(joint_state.velocity, velocity)
        assert torch.allclose(joint_state.acceleration, acceleration)
        assert torch.allclose(joint_state.jerk, jerk)
        assert joint_state.joint_names == joint_names
        assert torch.allclose(joint_state.dt, dt)

    def test_from_numpy(self, joint_names, device_cfg):
        """Test creating JointState from numpy arrays."""
        position = np.random.randn(7).astype(np.float32)
        velocity = np.random.randn(7).astype(np.float32)
        acceleration = np.random.randn(7).astype(np.float32)
        jerk = np.random.randn(7).astype(np.float32)

        joint_state = JointState.from_numpy(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            device_cfg=device_cfg,
        )

        assert isinstance(joint_state.position, torch.Tensor)
        assert isinstance(joint_state.velocity, torch.Tensor)
        assert isinstance(joint_state.acceleration, torch.Tensor)
        assert isinstance(joint_state.jerk, torch.Tensor)
        assert joint_state.position.device == device_cfg.device
        assert joint_state.joint_names == joint_names

    def test_from_numpy_without_velocity(self, joint_names, device_cfg):
        """Test creating JointState from numpy with only position."""
        position = np.random.randn(7).astype(np.float32)

        joint_state = JointState.from_numpy(
            joint_names=joint_names, position=position, device_cfg=device_cfg
        )

        assert isinstance(joint_state.position, torch.Tensor)
        assert isinstance(joint_state.velocity, torch.Tensor)
        assert torch.allclose(joint_state.velocity, torch.zeros_like(joint_state.position))

    def test_from_position(self, joint_names, device_cfg):
        """Test creating JointState from position only."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position=position, joint_names=joint_names)

        assert torch.allclose(joint_state.position, position)
        assert torch.allclose(joint_state.velocity, torch.zeros_like(position))
        assert torch.allclose(joint_state.acceleration, torch.zeros_like(position))
        assert torch.allclose(joint_state.jerk, torch.zeros_like(position))

    def test_clone(self, sample_joint_state):
        """Test cloning JointState."""
        cloned = sample_joint_state.clone()

        assert torch.allclose(cloned.position, sample_joint_state.position)
        assert torch.allclose(cloned.velocity, sample_joint_state.velocity)
        assert torch.allclose(cloned.acceleration, sample_joint_state.acceleration)
        assert torch.allclose(cloned.jerk, sample_joint_state.jerk)
        assert cloned.joint_names == sample_joint_state.joint_names

        # Verify they are different objects
        assert cloned.position.data_ptr() != sample_joint_state.position.data_ptr()

    def test_to_device(self, sample_joint_state, device_cfg):
        """Test moving JointState to device."""
        joint_state = sample_joint_state.to(device_cfg)
        assert joint_state.position.device == device_cfg.device
        assert joint_state.position.dtype == device_cfg.dtype

    def test_detach(self, sample_joint_state):
        """Test detaching JointState from computation graph."""
        detached = sample_joint_state.detach()
        assert not detached.position.requires_grad
        assert not detached.velocity.requires_grad

    def test_getitem_int(self, device_cfg, joint_names):
        """Test indexing JointState with integer."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names,
        )

        js_indexed = joint_state[0]
        assert js_indexed.position.shape == (7,)
        assert torch.allclose(js_indexed.position, position[0])

    def test_getitem_tensor(self, device_cfg, joint_names):
        """Test indexing JointState with tensor."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names,
        )

        idx = torch.tensor([0, 2, 4], device=device_cfg.device, dtype=torch.long)
        js_indexed = joint_state[idx]
        assert js_indexed.position.shape == (3, 7)
        assert torch.allclose(js_indexed.position, position[idx])

    def test_getitem_slice(self, device_cfg, joint_names):
        """Test slicing JointState."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names,
        )

        js_sliced = joint_state[2:5]
        assert js_sliced.position.shape == (3, 7)
        assert torch.allclose(js_sliced.position, position[2:5])

    def test_getitem_out_of_range(self, device_cfg, joint_names):
        """Test indexing JointState out of range."""
        position = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)

        with pytest.raises(ValueError, match="index out of range"):
            _ = joint_state[10]

    def test_setitem(self, device_cfg, joint_names):
        """Test setting item in JointState."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.ones(10, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            joint_names=joint_names,
        )

        new_position = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
        new_velocity = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
        new_acceleration = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
        new_jerk = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
        new_dt = torch.tensor([0.2], device=device_cfg.device, dtype=device_cfg.dtype)

        new_js = JointState(
            position=new_position,
            velocity=new_velocity,
            acceleration=new_acceleration,
            jerk=new_jerk,
            dt=new_dt,
        )

        joint_state[0] = new_js
        assert torch.allclose(joint_state.position[0], new_position)
        assert torch.allclose(joint_state.velocity[0], new_velocity)

    def test_len(self, device_cfg, joint_names):
        """Test length of JointState."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)
        assert len(joint_state) == 10

    def test_shape(self, device_cfg, joint_names):
        """Test shape property of JointState."""
        position = torch.randn(10, 5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)
        assert joint_state.shape == torch.Size([10, 5, 7])

    def test_ndim(self, device_cfg, joint_names):
        """Test ndim property of JointState."""
        position = torch.randn(10, 5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)
        assert joint_state.ndim == 3

    def test_device_property(self, sample_joint_state, device_cfg):
        """Test device property of JointState."""
        assert sample_joint_state.device == device_cfg.device

    def test_dtype_property(self, sample_joint_state, device_cfg):
        """Test dtype property of JointState."""
        assert sample_joint_state.dtype == device_cfg.dtype

    def test_data_ptr(self, sample_joint_state):
        """Test data_ptr method."""
        ptr = sample_joint_state.data_ptr()
        assert ptr == sample_joint_state.position.data_ptr()

    def test_get_state_tensor(self, sample_joint_state):
        """Test get_state_tensor method."""
        state_tensor = sample_joint_state.get_state_tensor()
        dof = sample_joint_state.position.shape[-1]
        expected_size = 4 * dof  # position, velocity, acceleration, jerk
        assert state_tensor.shape[-1] == expected_size

    def test_from_state_tensor(self, device_cfg, joint_names):
        """Test from_state_tensor method."""
        dof = 7
        state_tensor = torch.randn(4 * dof, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_state_tensor(state_tensor, joint_names=joint_names, dof=dof)

        assert joint_state.position.shape == (dof,)
        assert joint_state.velocity.shape == (dof,)
        assert joint_state.acceleration.shape == (dof,)
        assert joint_state.jerk.shape == (dof,)

    def test_stack(self, device_cfg, joint_names):
        """Test stacking JointStates."""
        position1 = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        stacked = js1.stack(js2)
        assert stacked.position.shape[0] == 10

    def test_blend(self, device_cfg, joint_names):
        """Test blending JointStates."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5, jerk=0.5)
        js1.blend(coeff, js2)

        # blend formula: coeff * new_state + (1 - coeff) * self
        # 0.5 * 0 + 0.5 * 1 = 0.5
        expected = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype) * 0.5
        assert torch.allclose(js1.position, expected)

    def test_unsqueeze(self, sample_joint_state):
        """Test unsqueezing JointState."""
        unsqueezed = sample_joint_state.unsqueeze(0)
        assert unsqueezed.position.shape[0] == 1

    def test_squeeze(self, device_cfg, joint_names):
        """Test squeezing JointState."""
        position = torch.randn(1, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)
        squeezed = joint_state.squeeze(0)
        assert squeezed.position.shape == (7,)

    def test_repeat(self, device_cfg, joint_names):
        """Test repeating JointState."""
        position = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)
        repeated = joint_state.repeat([3, 1])
        assert repeated.position.shape[0] == 15  # 5 * 3

    def test_repeat_seeds(self, device_cfg, joint_names):
        """Test repeat_seeds method."""
        position = torch.randn(2, 1, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(2, 1, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(2, 1, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(2, 1, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            joint_names=joint_names
        )
        repeated = joint_state.repeat_seeds(num_seeds=3)
        # repeat_seeds flattens (batch=2, seeds=1) into (batch*seeds*num_seeds,) = 6
        assert repeated.position.shape[0] == 6

    def test_zeros(self, device_cfg, joint_names):
        """Test creating zero JointState."""
        size = (10, 7)
        joint_state = JointState.zeros(size, device_cfg, joint_names)

        assert joint_state.position.shape == size
        assert torch.allclose(joint_state.position, torch.zeros(size))
        assert torch.allclose(joint_state.velocity, torch.zeros(size))
        assert joint_state.joint_names == joint_names

    def test_copy_(self, device_cfg, joint_names):
        """Test copy_ method."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        js1.copy_(js2)
        assert torch.allclose(js1.position, position2)

    def test_copy_at_index(self, device_cfg, joint_names):
        """Test copy_at_index method."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        js1.copy_at_index(js2, 0)
        assert torch.allclose(js1.position[0], position2)

    def test_copy_only_index(self, device_cfg, joint_names):
        """Test copy_only_index method."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        js1.copy_only_index(js2, 0)
        assert torch.allclose(js1.position[0], position2[0])

    def test_copy_at_batch_seed_indices(self, device_cfg, joint_names):
        """Test copy_at_batch_seed_indices method."""
        position1 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        js1.copy_at_batch_seed_indices(js2, batch_idx, seed_idx)
        assert torch.allclose(js1.position[0, 0], position2[0, 0])

    def test_inplace_reindex(self, device_cfg):
        """Test inplace_reindex method."""
        joint_names = ["j1", "j2", "j3", "j4"]
        position = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)

        new_order = ["j4", "j2", "j1", "j3"]
        joint_state.reindex(new_order)

        expected = torch.tensor([4.0, 2.0, 1.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        assert torch.allclose(joint_state.position, expected)
        assert joint_state.joint_names == new_order

    def test_get_ordered_joint_state(self, device_cfg):
        """Test get_ordered_joint_state method."""
        joint_names = ["j1", "j2", "j3", "j4"]
        position = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)

        new_order = ["j4", "j2", "j1", "j3"]
        new_js = joint_state.reorder(new_order)

        expected = torch.tensor([4.0, 2.0, 1.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        assert torch.allclose(new_js.position, expected)
        # Original should remain unchanged
        assert torch.allclose(joint_state.position, position)

    def test_append_joints_2d(self, device_cfg):
        """Test append_joints method with 2D tensors."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.tensor([[1.0, 2.0, 3.0]], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names1)
        js2 = JointState(position=position2, joint_names=joint_names2)

        new_js = js1.append_joints(js2)
        assert len(new_js.joint_names) == 5
        assert new_js.joint_names == joint_names1 + joint_names2
        assert new_js.position.shape[-1] == 5

    def test_append_joints_1d(self, device_cfg):
        """Test append_joints with 1D tensors (one_dim branch)."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=joint_names1)
        js2 = JointState(position=position2, joint_names=joint_names2)

        new_js = js1.append_joints(js2)
        assert len(new_js.joint_names) == 5
        assert new_js.position.shape == (5,)  # Should be squeezed back to 1D

    def test_append_joints_3d(self, device_cfg):
        """Test append_joints with 3D tensors."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names1)
        js2 = JointState(position=position2, joint_names=joint_names2)

        new_js = js1.append_joints(js2)
        assert len(new_js.joint_names) == 5
        assert new_js.position.shape == (2, 3, 5)

    def test_append_joints_2d_with_2d_input(self, device_cfg):
        """Test append_joints with both 2D tensors with repeat."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.randn(5, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names1)
        js2 = JointState(position=position2, joint_names=joint_names2)

        new_js = js1.append_joints(js2)
        assert len(new_js.joint_names) == 5
        assert new_js.position.shape == (5, 5)
        # Verify that js2 was repeated for each batch
        assert torch.allclose(new_js.position[0, 3:], position2)
        assert torch.allclose(new_js.position[1, 3:], position2)

    def test_append_joints_no_joint_names_error(self, device_cfg):
        """Test append_joints raises error when joint_names is None."""
        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=["j1", "j2", "j3"])
        js2 = JointState(position=position2, joint_names=None)

        with pytest.raises(Exception, match="joint_names are required"):
            js1.append_joints(js2)

    def test_append_joints_empty_joint_names_error(self, device_cfg):
        """Test append_joints raises error when joint_names is empty."""
        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=["j1", "j2", "j3"])
        js2 = JointState(position=position2, joint_names=[])

        with pytest.raises(Exception, match="joint_names are required"):
            js1.append_joints(js2)

    def test_append_joints_shape_mismatch_error(self, device_cfg):
        """Test append_joints raises error on incompatible shapes."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.randn(5, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.randn(3, 2, device=device_cfg.device, dtype=device_cfg.dtype)  # Incompatible shape

        js1 = JointState.from_position(position1, joint_names1)
        js2 = JointState(position=position2, joint_names=joint_names2)

        with pytest.raises(Exception, match="appending joints requires"):
            js1.append_joints(js2)

    def test_append_joints_with_dt(self, device_cfg):
        """Test append_joints preserves dt."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)
        dt1 = torch.tensor([0.1], device=device_cfg.device, dtype=device_cfg.dtype)
        dt2 = torch.tensor([0.2], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=joint_names1, dt=dt1)
        js2 = JointState(position=position2, joint_names=joint_names2, dt=dt2)

        new_js = js1.append_joints(js2)
        # js2.dt should override js1.dt
        assert torch.allclose(new_js.dt, dt2)

    def test_append_joints_3d_with_velocity(self, device_cfg):
        """Test append_joints with 3D tensors and velocity."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, velocity=velocity1, joint_names=joint_names1)
        js2 = JointState(position=position2, joint_names=joint_names2)

        new_js = js1.append_joints(js2)
        assert new_js.velocity is not None
        assert new_js.velocity.shape == (2, 3, 5)
        # Original velocity should be preserved
        assert torch.allclose(new_js.velocity[..., :3], velocity1)
        # New joints should have zero velocity
        assert torch.allclose(new_js.velocity[..., 3:], torch.zeros(2, 3, 2, device=device_cfg.device))

    def test_cat(self, device_cfg, joint_names):
        """Test cat method."""
        position1 = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        concatenated = js1.cat(js2, dim=0)
        assert concatenated.position.shape[0] == 10

    def test_view(self, device_cfg, joint_names):
        """Test view method."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        viewed = joint_state.view(2, 5, 7)
        assert viewed.position.shape == (2, 5, 7)

    def test_scale(self, sample_joint_state):
        """Test scale method."""
        dt = 2.0
        scaled = sample_joint_state.scale(dt)

        assert torch.allclose(scaled.velocity, sample_joint_state.velocity * dt)
        assert torch.allclose(scaled.acceleration, sample_joint_state.acceleration * dt**2)
        assert torch.allclose(scaled.jerk, sample_joint_state.jerk * dt**3)

    def test_scale_by_dt(self, device_cfg, joint_names):
        """Test scale_by_dt method."""
        position = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        new_dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            joint_names=joint_names,
        )

        scaled = joint_state.scale_by_dt(dt, new_dt)
        assert torch.allclose(scaled.dt, new_dt)

    def test_scale_time(self, device_cfg, joint_names):
        """Test scale_time method."""
        position = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        new_dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2

        joint_state = JointState(position=position, velocity=velocity, dt=dt, joint_names=joint_names)

        scaled = joint_state.scale_time(new_dt)
        assert torch.allclose(scaled.dt, new_dt)

    def test_trim_trajectory(self, device_cfg, joint_names):
        """Test trim_trajectory method."""
        position = torch.randn(10, 20, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        trimmed = joint_state.trim_trajectory(start_idx=5, end_idx=15)
        assert trimmed.position.shape[1] == 10

    def test_get_trajectory_at_horizon_index(self, device_cfg, joint_names):
        """Test get_trajectory_at_horizon_index method."""
        position = torch.randn(10, 20, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        horizon_js = joint_state.get_trajectory_at_horizon_index(5)
        assert horizon_js.position.shape == (10, 7)

    def test_calculate_fd_from_position(self, device_cfg, joint_names):
        """Test calculate_fd_from_position method."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.tensor([0.1], device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, dt=dt, joint_names=joint_names)

        joint_state.calculate_fd_from_position()
        assert joint_state.velocity is not None
        assert joint_state.acceleration is not None
        assert joint_state.jerk is not None

    def test_gather_by_seed_index(self, device_cfg, joint_names):
        """Test gather_by_seed_index method."""
        batch_size = 5
        num_seeds = 10
        horizon = 20
        dof = 7

        position = torch.randn(
            batch_size, num_seeds, horizon, dof, device=device_cfg.device, dtype=device_cfg.dtype
        )
        velocity = torch.randn(
            batch_size, num_seeds, horizon, dof, device=device_cfg.device, dtype=device_cfg.dtype
        )
        acceleration = torch.randn(
            batch_size, num_seeds, horizon, dof, device=device_cfg.device, dtype=device_cfg.dtype
        )
        jerk = torch.randn(
            batch_size, num_seeds, horizon, dof, device=device_cfg.device, dtype=device_cfg.dtype
        )
        dt = torch.ones(batch_size, num_seeds, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        joint_state = JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            joint_names=joint_names
        )

        topk = 3
        idx = torch.randint(0, num_seeds, (batch_size, topk), device=device_cfg.device)

        gathered = joint_state.gather_by_seed_index(idx)
        assert gathered.position.shape == (batch_size, topk, horizon, dof)

    def test_apply_kernel(self, device_cfg, joint_names):
        """Test apply_kernel method."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        kernel_mat = torch.eye(10, device=device_cfg.device, dtype=device_cfg.dtype)
        result = joint_state.apply_kernel(kernel_mat)

        assert torch.allclose(result.position, joint_state.position)

    def test_copy_reference(self, device_cfg, joint_names):
        """Test copy_reference method."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        js1.copy_reference(js2)
        # Should be same reference
        assert js1.position.data_ptr() == js2.position.data_ptr()

    def test_get_state_tensor_with_none_fields(self, device_cfg, joint_names):
        """Test get_state_tensor when velocity/acceleration/jerk are None."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, velocity=None, acceleration=None, jerk=None)

        state_tensor = joint_state.get_state_tensor()
        assert state_tensor.shape[-1] == 28  # 4 * 7 dof

    def test_getitem_list_index(self, device_cfg, joint_names):
        """Test indexing JointState with list."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState(position=position, velocity=velocity, joint_names=joint_names)
        idx = [0, 2, 4]
        js_indexed = joint_state[idx]

        assert js_indexed.position.shape == (3, 7)

    def test_getitem_with_knot(self, device_cfg, joint_names):
        """Test indexing JointState with knot data."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        knot = torch.randn(10, 5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        knot_dt = torch.ones(10, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        joint_state = JointState(position=position, joint_names=joint_names, knot=knot, knot_dt=knot_dt)
        js_indexed = joint_state[0]

        assert js_indexed.knot is not None
        assert js_indexed.knot.shape == (5, 7)

    def test_getitem_with_knot_scalar_dt(self, device_cfg, joint_names):
        """Test indexing JointState with scalar knot_dt."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        knot = torch.randn(10, 5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        knot_dt = torch.tensor(0.1, device=device_cfg.device, dtype=device_cfg.dtype)  # scalar

        joint_state = JointState(position=position, joint_names=joint_names, knot=knot, knot_dt=knot_dt)
        js_indexed = joint_state[0]

        assert js_indexed.knot_dt is not None

    def test_setitem_tensor_with_dt(self, device_cfg, joint_names):
        """Test setitem with tensor index when both have dt."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.ones(10, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        joint_state = JointState.from_position(position, joint_names)
        joint_state.dt = dt

        new_position = torch.zeros(2, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        new_dt = torch.ones(2, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2
        new_js = JointState.from_position(new_position, joint_names)
        new_js.dt = new_dt

        idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        joint_state[idx] = new_js

        assert torch.allclose(joint_state.position[0], new_position[0])
        assert torch.allclose(joint_state.dt[idx], new_dt)

    def test_setitem_tensor_without_dt_bugfix(self, device_cfg, joint_names):
        """Test setitem with tensor index when dt is None - tests bug fix."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)
        joint_state.dt = None  # Explicitly set to None

        new_position = torch.zeros(2, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        new_js = JointState.from_position(new_position, joint_names)
        new_js.dt = None

        idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        # This should not raise an error (bug fix)
        joint_state[idx] = new_js

        assert torch.allclose(joint_state.position[0], new_position[0])

    def test_gather_by_seed_index_invalid_idx_ndim(self, device_cfg, joint_names):
        """Test gather_by_seed_index with invalid idx dimensions."""
        position = torch.randn(5, 10, 20, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        # idx with wrong dimensions (1D instead of 2D)
        idx = torch.tensor([0, 1, 2], device=device_cfg.device, dtype=torch.long)

        with pytest.raises(Exception):
            joint_state.gather_by_seed_index(idx)

    def test_gather_by_seed_index_batch_mismatch(self, device_cfg, joint_names):
        """Test gather_by_seed_index with mismatched batch size."""
        position = torch.randn(5, 10, 20, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        # idx with wrong batch size
        idx = torch.randint(0, 10, (3, 2), device=device_cfg.device)

        with pytest.raises(Exception):
            joint_state.gather_by_seed_index(idx)

    def test_gather_by_seed_index_invalid_position_ndim(self, device_cfg, joint_names):
        """Test gather_by_seed_index with invalid position dimensions."""
        position = torch.randn(5, 10, 7, device=device_cfg.device, dtype=device_cfg.dtype)  # 3D instead of 4D
        joint_state = JointState.from_position(position, joint_names)

        idx = torch.randint(0, 10, (5, 2), device=device_cfg.device)

        with pytest.raises(Exception):
            joint_state.gather_by_seed_index(idx)

    def test_gather_by_seed_index_with_knot(self, device_cfg, joint_names):
        """Test gather_by_seed_index with knot data."""
        batch_size = 5
        num_seeds = 10
        horizon = 20
        dof = 7
        knots = 8

        position = torch.randn(
            batch_size, num_seeds, horizon, dof, device=device_cfg.device, dtype=device_cfg.dtype
        )
        knot = torch.randn(
            batch_size, num_seeds, knots, dof, device=device_cfg.device, dtype=device_cfg.dtype
        )
        knot_dt = torch.ones(batch_size, num_seeds, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        joint_state = JointState.from_position(position, joint_names)
        joint_state.knot = knot
        joint_state.knot_dt = knot_dt
        joint_state.dt = torch.ones(batch_size, num_seeds, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        topk = 3
        idx = torch.randint(0, num_seeds, (batch_size, topk), device=device_cfg.device)

        gathered = joint_state.gather_by_seed_index(idx)
        assert gathered.knot is not None
        assert gathered.knot.shape == (batch_size, topk, knots, dof)
        assert gathered.knot_dt is not None

    def test_from_list(self, device_cfg):
        """Test from_list static method."""
        position = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        velocity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        acceleration = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

        joint_state = JointState.from_list(position, velocity, acceleration, device_cfg)
        assert isinstance(joint_state.position, torch.Tensor)
        assert joint_state.position.device == device_cfg.device

    def test_copy_at_index_out_of_range(self, device_cfg, joint_names):
        """Test copy_at_index with out of range index."""
        position1 = torch.ones(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        with pytest.raises(ValueError, match="index out of range"):
            js1.copy_at_index(js2, 10)

    def test_copy_at_index_with_list(self, device_cfg, joint_names):
        """Test copy_at_index with list index."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(2, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        js1.copy_at_index(js2, [0, 1])
        assert torch.allclose(js1.position[0], position2[0])

    def test_copy_at_index_with_dt_none(self, device_cfg, joint_names):
        """Test copy_at_index when dt is None."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)
        js2.dt = None

        js1.copy_at_index(js2, 0)
        assert torch.allclose(js1.position[0], position2)

    def test_copy_at_batch_seed_indices_without_dt(self, device_cfg, joint_names):
        """Test copy_at_batch_seed_indices when dt is None."""
        position1 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)
        js1.dt = None
        js2.dt = None

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        js1.copy_at_batch_seed_indices(js2, batch_idx, seed_idx)
        assert torch.allclose(js1.position[0, 0], position2[0, 0])

    def test_copy__with_different_shapes_no_clone(self, device_cfg, joint_names):
        """Test copy_ with different shapes and allow_clone=False."""
        position1 = torch.ones(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        with pytest.raises(Exception):
            js1.copy_(js2, allow_clone=False)

    def test_inplace_reindex_no_joint_names(self, device_cfg):
        """Test inplace_reindex when joint_names is None."""
        position = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=None)

        with pytest.raises(Exception):
            joint_state.inplace_reindex(["j1", "j2"])

    def test_get_augmented_joint_state_with_lock_joints(self, device_cfg):
        """Test get_augmented_joint_state with lock_joints."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]
        target_names = ["j1", "j2", "j3", "j4", "j5"]

        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=joint_names1)
        lock_js = JointState(position=position2, joint_names=joint_names2)

        result = js1.get_augmented_joint_state(target_names, lock_js)
        assert len(result.joint_names) == 5

    def test_get_augmented_joint_state_none_joint_names(self, device_cfg):
        """Test get_augmented_joint_state with None joint_names."""
        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=None)
        lock_js = JointState(position=position2, joint_names=["j4", "j5"])

        with pytest.raises(ValueError, match="joint_names can't be None"):
            js1.get_augmented_joint_state(None, lock_js)

    def test_get_augmented_joint_state_overlapping_names(self, device_cfg):
        """Test get_augmented_joint_state with overlapping joint names."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j2", "j4"]  # j2 overlaps

        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=joint_names1)
        lock_js = JointState(position=position2, joint_names=joint_names2)

        with pytest.raises(ValueError, match="lock_joints is also listed"):
            js1.get_augmented_joint_state(["j1", "j2", "j3", "j4"], lock_js)

    def test_append_joints_with_knot_error(self, device_cfg):
        """Test append_joints with knot raises not implemented error."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)
        knot1 = torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        knot2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=joint_names1, knot=knot1)
        js2 = JointState(position=position2, joint_names=joint_names2, knot=knot2)

        with pytest.raises(Exception, match="knot append needs to be implemented"):
            js1.append_joints(js2)

    def test_trim_trajectory_no_horizon(self, device_cfg, joint_names):
        """Test trim_trajectory raises error when no horizon."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        with pytest.raises(ValueError, match="does not have horizon"):
            joint_state.trim_trajectory(0, 5)

    def test_trim_trajectory_with_none_end_idx(self, device_cfg, joint_names):
        """Test trim_trajectory with None end_idx."""
        position = torch.randn(10, 20, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        trimmed = joint_state.trim_trajectory(start_idx=5, end_idx=None)
        assert trimmed.position.shape[1] == 15  # 20 - 5

    def test_get_trajectory_at_horizon_index_no_horizon(self, device_cfg, joint_names):
        """Test get_trajectory_at_horizon_index raises error when no horizon."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState.from_position(position, joint_names)

        with pytest.raises(ValueError, match="does not have horizon"):
            joint_state.get_trajectory_at_horizon_index(0)

    def test_calculate_fd_from_position_no_dt(self, device_cfg, joint_names):
        """Test calculate_fd_from_position raises error when dt is None."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names, dt=None)

        with pytest.raises(Exception, match="dt is required"):
            joint_state.calculate_fd_from_position()

    def test_calculate_fd_from_position_with_explicit_dt(self, device_cfg, joint_names):
        """Test calculate_fd_from_position with explicit dt parameter."""
        position = torch.randn(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names, dt=None)

        dt = torch.tensor([0.1], device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state.calculate_fd_from_position(dt=dt)

        assert joint_state.velocity is not None
        assert joint_state.acceleration is not None

    def test_scale_with_knot_dt_error(self, device_cfg, joint_names):
        """Test scale raises error when knot_dt is not None."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        knot_dt = torch.tensor(0.1, device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState.from_position(position, joint_names)
        joint_state.knot_dt = knot_dt

        with pytest.raises(Exception, match="knot dt needs to be scaled"):
            joint_state.scale(2.0)

    def test_scale_by_dt_with_knot_dt(self, device_cfg, joint_names):
        """Test scale_by_dt with knot_dt."""
        position = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        new_dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2
        knot_dt = torch.ones(5, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        joint_state = JointState(position=position, velocity=velocity, dt=dt, joint_names=joint_names)
        joint_state.knot_dt = knot_dt

        scaled = joint_state.scale_by_dt(dt, new_dt)
        assert scaled.knot_dt is not None
        assert torch.allclose(scaled.knot_dt, knot_dt * (new_dt / dt))

    def test_index_dof(self, device_cfg):
        """Test index_dof method."""
        joint_names = ["j1", "j2", "j3", "j4"]
        position = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState.from_position(position, joint_names)

        idx = torch.tensor([0, 2], device=device_cfg.device, dtype=torch.long)
        indexed = joint_state.index_dof(idx)

        assert indexed.position.shape[-1] == 2
        assert torch.allclose(indexed.position, torch.tensor([1.0, 3.0], device=device_cfg.device))
        assert indexed.joint_names == ["j1", "j3"]

    def test_index_dof_with_knot(self, device_cfg):
        """Test index_dof with knot."""
        joint_names = ["j1", "j2", "j3", "j4"]
        position = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)
        knot = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState(position=position, joint_names=joint_names, knot=knot)

        idx = torch.tensor([0, 2], device=device_cfg.device, dtype=torch.long)
        indexed = joint_state.index_dof(idx)

        assert indexed.knot is not None
        assert indexed.knot.shape[-1] == 2

    def test_copy_only_index_with_dt(self, device_cfg, joint_names):
        """Test copy_only_index with dt field."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0
        dt1 = torch.ones(10, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        dt2 = torch.ones(10, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)
        js1.dt = dt1
        js2.dt = dt2

        js1.copy_only_index(js2, 0)
        assert torch.allclose(js1.dt[0], dt2[0])

    def test_view_with_dt_3d(self, device_cfg, joint_names):
        """Test view method with dt and 3D shape."""
        position = torch.randn(6, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt = torch.ones(6, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1

        joint_state = JointState.from_position(position, joint_names)
        joint_state.dt = dt

        viewed = joint_state.view(2, 3, 7)
        assert viewed.position.shape == (2, 3, 7)
        assert viewed.dt.shape == (2, 3)

    def test_copy_at_index_with_tensor_idx(self, device_cfg, joint_names):
        """Test copy_at_index with tensor index (covers line 396-397)."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(2, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        js1.copy_at_index(js2, idx)
        assert torch.allclose(js1.position[0], position2[0])

    def test_copy_at_index_with_both_dt(self, device_cfg, joint_names):
        """Test copy_at_index when both have dt (covers line 413)."""
        position1 = torch.ones(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)
        dt1 = torch.ones(10, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        dt2 = torch.ones(1, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)
        js1.dt = dt1
        js2.dt = dt2

        js1.copy_at_index(js2, 0)
        assert torch.allclose(js1.dt[0], dt2[0])

    def test_copy_at_batch_seed_indices_with_dt(self, device_cfg, joint_names):
        """Test copy_at_batch_seed_indices with dt (covers line 437)."""
        position1 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.ones(5, 3, 7, device=device_cfg.device, dtype=device_cfg.dtype) * 2.0
        dt1 = torch.ones(5, 3, device=device_cfg.device, dtype=device_cfg.dtype) * 0.1
        dt2 = torch.ones(5, 3, device=device_cfg.device, dtype=device_cfg.dtype) * 0.2

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)
        js1.dt = dt1
        js2.dt = dt2

        batch_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)
        seed_idx = torch.tensor([0, 1], device=device_cfg.device, dtype=torch.long)

        js1.copy_at_batch_seed_indices(js2, batch_idx, seed_idx)
        assert torch.allclose(js1.dt[0, 0], dt2[0, 0])

    def test_copy_data_deprecated(self, device_cfg, joint_names):
        """Test deprecated copy_data method (covers lines 447-458)."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        # Should work but log deprecation warning (using log_warn, not Python warnings)
        js1.copy_data(js2)

        assert torch.allclose(js1.position, position2)

    def test_copy_data_with_clone(self, device_cfg, joint_names):
        """Test copy_data when tensors don't match (covers clone path)."""
        position1 = torch.ones(7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(10, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        # Should clone when shapes don't match
        js1.copy_data(js2)

        assert js1.position.shape == (10,)

    def test_copy__with_allow_clone_true(self, device_cfg, joint_names):
        """Test copy_ with allow_clone=True and shape mismatch (covers lines 521-525)."""
        position1 = torch.ones(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.zeros(10, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState.from_position(position1, joint_names)
        js2 = JointState.from_position(position2, joint_names)

        # Should clone and log info
        js1.copy_(js2, allow_clone=True)
        assert js1.position.shape == (10, 7)
        assert torch.allclose(js1.position, position2)

    def test_unsqueeze_with_knot(self, device_cfg, joint_names):
        """Test unsqueeze with knot (covers line 538)."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype)
        knot = torch.randn(5, 7, device=device_cfg.device, dtype=device_cfg.dtype)

        joint_state = JointState.from_position(position, joint_names)
        joint_state.knot = knot

        unsqueezed = joint_state.unsqueeze(0)
        assert unsqueezed.knot is not None
        assert unsqueezed.knot.shape == (1, 5, 7)

    def test_detach_with_dt(self, device_cfg, joint_names):
        """Test detach with dt field (covers line 602)."""
        position = torch.randn(7, device=device_cfg.device, dtype=device_cfg.dtype, requires_grad=True)
        dt = torch.ones(1, device=device_cfg.device, dtype=device_cfg.dtype, requires_grad=True)

        joint_state = JointState.from_position(position, joint_names)
        joint_state.dt = dt

        detached = joint_state.detach()
        assert not detached.dt.requires_grad

    def test_get_augmented_joint_state_without_lock(self, device_cfg):
        """Test get_augmented_joint_state without lock_joints (covers line 643)."""
        joint_names = ["j1", "j2", "j3", "j4"]
        position = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_cfg.device, dtype=device_cfg.dtype)
        joint_state = JointState(position=position, joint_names=joint_names)

        new_order = ["j4", "j2", "j1", "j3"]
        # When lock_joints is None, should just reorder
        result = joint_state.get_augmented_joint_state(new_order, lock_joints=None)

        expected = torch.tensor([4.0, 2.0, 1.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype)
        assert torch.allclose(result.position, expected)

    def test_append_joints_knot_matching_shape(self, device_cfg):
        """Test append_joints with matching knot shapes (covers line 667)."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.tensor([[1.0, 2.0, 3.0]], device=device_cfg.device, dtype=device_cfg.dtype)
        position2 = torch.tensor([[4.0, 5.0]], device=device_cfg.device, dtype=device_cfg.dtype)
        knot1 = torch.tensor([[1.0, 2.0, 3.0]], device=device_cfg.device, dtype=device_cfg.dtype)
        knot2 = torch.tensor([[4.0, 5.0]], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(position=position1, joint_names=joint_names1, knot=knot1)
        js2 = JointState(position=position2, joint_names=joint_names2, knot=knot2)

        # When knot shapes match exactly
        with pytest.raises(Exception, match="knot append needs to be implemented"):
            js1.append_joints(js2)

    def test_append_joints_dead_code_analysis(self, device_cfg):
        """Verify that lines 682-688 appear to be dead code due to logical impossibility.

        The condition `current_shape[:-1] == joint_state.position.shape and len(current_shape) == len(joint_state.position.shape)`
        can never be True because:
        - current_shape[:-1] removes last dimension, making it length N-1
        - But the len check requires both to have same length N
        - A tuple of length N-1 cannot equal a tuple of length N

        This is likely a bug in the original code.
        """
        # Try various configurations that might hit this path
        configs = [
            # (js1_shape, js2_shape, js1_has_velocity)
            ((3,), (3,), True),  # Both 1D, same length
            ((1, 3), (1,), True),  # 2D and 1D
            ((2, 3), (2,), True),  # 2D and 1D with batch
            ((1, 3), (1, 2), True),  # Both 2D
        ]

        for js1_shape, js2_shape, has_vel in configs:
            joint_names1 = [f"j{i}" for i in range(js1_shape[-1])]
            joint_names2 = [f"jx{i}" for i in range(js2_shape[-1])]

            pos1 = torch.randn(js1_shape, device=device_cfg.device, dtype=device_cfg.dtype)
            pos2 = torch.randn(js2_shape, device=device_cfg.device, dtype=device_cfg.dtype)

            js1 = JointState(position=pos1, joint_names=joint_names1)
            if has_vel:
                js1.velocity = torch.randn_like(pos1)
            js2 = JointState(position=pos2, joint_names=joint_names2)

            # None of these should raise "Not implemented" because the condition is unreachable
            try:
                result = js1.append_joints(js2)
                # If we get here, it worked via a different code path
                assert result is not None
            except Exception as e:
                # Should not be "Not implemented" from line 688
                assert "Not implemented" not in str(e), \
                    f"Hit dead code with shapes {js1_shape}, {js2_shape}"

    def test_append_joints_3d_with_all_derivatives(self, device_cfg):
        """Test append_joints 3D with velocity, acceleration, jerk (covers lines 730-734)."""
        joint_names1 = ["j1", "j2", "j3"]
        joint_names2 = ["j4", "j5"]

        position1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        velocity1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        jerk1 = torch.randn(2, 3, 3, device=device_cfg.device, dtype=device_cfg.dtype)

        position2 = torch.tensor([4.0, 5.0], device=device_cfg.device, dtype=device_cfg.dtype)
        velocity2 = torch.tensor([0.4, 0.5], device=device_cfg.device, dtype=device_cfg.dtype)
        acceleration2 = torch.tensor([0.04, 0.05], device=device_cfg.device, dtype=device_cfg.dtype)
        jerk2 = torch.tensor([0.004, 0.005], device=device_cfg.device, dtype=device_cfg.dtype)

        js1 = JointState(
            position=position1,
            velocity=velocity1,
            acceleration=acceleration1,
            jerk=jerk1,
            joint_names=joint_names1
        )
        js2 = JointState(
            position=position2,
            velocity=velocity2,
            acceleration=acceleration2,
            jerk=jerk2,
            joint_names=joint_names2
        )

        new_js = js1.append_joints(js2)
        assert new_js.velocity is not None
        assert new_js.acceleration is not None
        assert new_js.jerk is not None
        assert new_js.position.shape == (2, 3, 5)
        # Check that js2's velocity/acceleration/jerk were added
        assert torch.allclose(new_js.velocity[0, 0, 3:], velocity2)
        assert torch.allclose(new_js.acceleration[0, 0, 3:], acceleration2)
        assert torch.allclose(new_js.jerk[0, 0, 3:], jerk2)

