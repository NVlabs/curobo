# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState

# CuRobo
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.state_filter import FilterCfg, JointStateFilter


class TestFilterCfg:
    def test_filter_cfg_init(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff,
            dt=0.01,
            control_space=ControlSpace.ACCELERATION,
            enable=True,
        )
        assert cfg.dt == 0.01
        assert cfg.enable is True

    def test_filter_cfg_create(self):
        coeff_dict = {"position": 0.5, "velocity": 0.5, "acceleration": 0.5}
        cfg = FilterCfg.create(
            coeff_dict, enable=True, dt=0.01, control_space=ControlSpace.ACCELERATION
        )
        assert cfg.dt == 0.01
        assert cfg.enable is True

    def test_filter_cfg_create_with_teleport(self):
        coeff_dict = {"position": 0.5, "velocity": 0.5, "acceleration": 0.5}
        cfg = FilterCfg.create(
            coeff_dict,
            enable=True,
            dt=0.01,
            control_space=ControlSpace.POSITION,
            teleport_mode=True,
        )
        assert cfg.teleport_mode is True


class TestJointStateFilter:
    def test_init_acceleration_control(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION
        )
        filter = JointStateFilter(cfg)
        assert filter.integrate_action == filter.integrate_acc

    def test_init_velocity_control(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(filter_coeff=coeff, dt=0.01, control_space=ControlSpace.VELOCITY)
        filter = JointStateFilter(cfg)
        assert filter.integrate_action == filter.integrate_vel

    def test_init_position_control(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(filter_coeff=coeff, dt=0.01, control_space=ControlSpace.POSITION)
        filter = JointStateFilter(cfg)
        assert filter.integrate_action == filter.integrate_pos

    def test_filter_joint_state_disabled(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=False
        )
        filter = JointStateFilter(cfg)
        state = JointState.zeros((3,), DeviceCfg())
        result = filter.filter_joint_state(state)
        assert result is state

    def test_filter_joint_state_first_call(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        state = JointState.zeros((3,), DeviceCfg())
        result = filter.filter_joint_state(state)
        assert filter.cmd_joint_state is not None

    def test_filter_joint_state_blend(self):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        state1 = JointState.zeros((3,), DeviceCfg())
        filter.filter_joint_state(state1)
        state2 = JointState.zeros((3,), DeviceCfg())
        state2.position = torch.ones(3, device=state2.position.device)
        result = filter.filter_joint_state(state2)
        assert result is not None

    def test_integrate_acc(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        qdd_des = torch.ones(3, device=cuda_device_cfg.device)
        result = filter.integrate_acc(qdd_des, cmd_state)
        assert result is not None
        assert torch.allclose(result.acceleration, qdd_des)

    def test_integrate_acc_with_existing_state(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        filter.integrate_acc(torch.ones(3, device=cuda_device_cfg.device), cmd_state)
        result = filter.integrate_acc(torch.ones(3, device=cuda_device_cfg.device) * 2)
        assert result is not None

    def test_integrate_vel(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.VELOCITY, enable=True
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        qd_des = torch.ones(3, device=cuda_device_cfg.device)
        result = filter.integrate_vel(qd_des, cmd_state)
        assert result is not None
        assert torch.allclose(result.velocity, qd_des)

    def test_integrate_pos(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.POSITION, enable=True
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        q_des = torch.ones(3, device=cuda_device_cfg.device)
        result = filter.integrate_pos(q_des, cmd_state)
        assert result is not None
        assert torch.allclose(result.position, q_des)

    def test_integrate_pos_teleport_mode(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff,
            dt=0.01,
            control_space=ControlSpace.POSITION,
            enable=True,
            teleport_mode=True,
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        q_des = torch.ones(3, device=cuda_device_cfg.device)
        result = filter.integrate_pos(q_des, cmd_state)
        assert result is not None
        assert torch.allclose(result.position, q_des)

    def test_integrate_jerk(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        qddd_des = torch.ones(3, device=cuda_device_cfg.device)
        result = filter.integrate_jerk(qddd_des, cmd_state)
        assert result is not None

    def test_integrate_jerk_with_existing_state(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        cmd_state = JointState.zeros((3,), cuda_device_cfg)
        filter.integrate_jerk(torch.ones(3, device=cuda_device_cfg.device), cmd_state)
        result = filter.integrate_jerk(torch.ones(3, device=cuda_device_cfg.device) * 2)
        assert result is not None

    def test_reset(self, cuda_device_cfg):
        coeff = FilterCoeff(position=0.5, velocity=0.5, acceleration=0.5)
        cfg = FilterCfg(
            filter_coeff=coeff, dt=0.01, control_space=ControlSpace.ACCELERATION, enable=True
        )
        filter = JointStateFilter(cfg)
        state = JointState.zeros((3,), cuda_device_cfg)
        filter.filter_joint_state(state)
        assert filter.cmd_joint_state is not None
        filter.reset()
        assert filter.cmd_joint_state is None

