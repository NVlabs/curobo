# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CUDA composite rigid body algorithm."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _load_robot(robot_file_name: str) -> tuple[KinematicsParams, DeviceCfg]:
    """Load a robot's dynamics tensors on the default CUDA device."""
    device_cfg = DeviceCfg()
    config = KinematicsCfg.from_robot_yaml_file(robot_file_name, device_cfg=device_cfg)
    return config.kinematics_config, device_cfg


def _load_unlocked_mimic_robot() -> tuple[KinematicsParams, DeviceCfg]:
    """Load a robot whose active links include a shared generalized coordinate."""
    device_cfg = DeviceCfg()
    config = KinematicsCfg.from_robot_yaml_file(
        "simple_mimic_robot.yml",
        device_cfg=device_cfg,
        lock_joints=None,
    )
    kinematics = config.kinematics_config
    active_joint_map = kinematics.joint_map[kinematics.joint_map >= 0].cpu().numpy()
    _, mapped_link_counts = np.unique(active_joint_map, return_counts=True)
    assert np.any(mapped_link_counts > 1)
    return kinematics, device_cfg


def _make_dynamics(kinematics: KinematicsParams, device_cfg: DeviceCfg) -> Dynamics:
    """Construct a dynamics model for a kinematics configuration."""
    return Dynamics(
        DynamicsCfg(
            kinematics_config=kinematics,
            device_cfg=device_cfg,
        )
    )


def _mass_matrix_from_rnea(
    dynamics: Dynamics,
    joint_position: torch.Tensor,
) -> torch.Tensor:
    """Construct ``M(q)`` from batched zero-velocity RNEA columns."""
    batch_size, num_dof = joint_position.shape
    num_columns = num_dof + 1
    dynamics.setup_batch_size(batch_size=batch_size * num_columns)

    position = (
        joint_position[:, None, :]
        .expand(batch_size, num_columns, num_dof)
        .reshape(batch_size * num_columns, num_dof)
        .contiguous()
    )
    velocity = torch.zeros_like(position)
    acceleration = torch.zeros(
        (batch_size, num_columns, num_dof),
        device=joint_position.device,
        dtype=joint_position.dtype,
    )
    acceleration[:, 1:, :] = torch.eye(
        num_dof,
        device=joint_position.device,
        dtype=joint_position.dtype,
    )
    acceleration = acceleration.reshape(batch_size * num_columns, num_dof)

    torque = dynamics.compute_inverse_dynamics(
        JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )
    ).reshape(batch_size, num_columns, num_dof)
    columns = torque[:, 1:, :] - torque[:, :1, :]
    return columns.transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "robot_file_name",
    ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"],
)
def test_crba_matches_rnea_columns(robot_file_name: str) -> None:
    """CRBA should match RNEA-derived mass matrices across tree topologies."""
    kinematics, device_cfg = _load_robot(robot_file_name)
    batch_size = 3
    torch.manual_seed(12)
    joint_position = (
        0.8
        * (
            torch.rand(
                (batch_size, kinematics.num_dof),
                device=device_cfg.device,
                dtype=torch.float32,
            )
            - 0.5
        )
    ).contiguous()

    crba_dynamics = _make_dynamics(kinematics, device_cfg)
    crba_dynamics.setup_batch_size(batch_size=batch_size)
    mass_matrix = crba_dynamics.compute_mass_matrix(joint_position)

    reference = _mass_matrix_from_rnea(
        _make_dynamics(kinematics, device_cfg),
        joint_position,
    )
    torch.testing.assert_close(mass_matrix, reference, atol=5.0e-4, rtol=5.0e-4)
    torch.testing.assert_close(
        mass_matrix,
        mass_matrix.transpose(-1, -2),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    _, factor_info = torch.linalg.cholesky_ex(mass_matrix)
    assert torch.all(factor_info == 0)


def test_crba_preserves_horizon_shape() -> None:
    """The mass matrix should preserve all leading position dimensions."""
    kinematics, device_cfg = _load_robot("franka.yml")
    batch_size = 2
    horizon = 4
    dynamics = _make_dynamics(kinematics, device_cfg)
    dynamics.setup_batch_size(batch_size=batch_size, horizon=horizon)
    joint_position = torch.zeros(
        (batch_size, horizon, kinematics.num_dof),
        device=device_cfg.device,
        dtype=torch.float32,
    )

    mass_matrix = dynamics.compute_mass_matrix(joint_position)

    assert mass_matrix.shape == (
        batch_size,
        horizon,
        kinematics.num_dof,
        kinematics.num_dof,
    )


def test_crba_matches_rnea_for_unlocked_mimic_joint() -> None:
    """Mimic multipliers should produce the reduced generalized mass matrix."""
    kinematics, device_cfg = _load_unlocked_mimic_robot()
    joint_position = torch.linspace(
        -0.3,
        0.2,
        kinematics.num_dof,
        device=device_cfg.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    dynamics = _make_dynamics(kinematics, device_cfg)
    dynamics.setup_batch_size(batch_size=1)

    mass_matrix = dynamics.compute_mass_matrix(joint_position)
    reference = _mass_matrix_from_rnea(
        _make_dynamics(kinematics, device_cfg),
        joint_position,
    )

    torch.testing.assert_close(mass_matrix, reference, atol=3.0e-4, rtol=3.0e-4)


def test_crba_backward_matches_rnea_for_unlocked_mimic_joint() -> None:
    """The CRBA VJP should include all links sharing a generalized coordinate."""
    kinematics, device_cfg = _load_unlocked_mimic_robot()
    torch.manual_seed(24)
    joint_position = torch.randn(
        (1, kinematics.num_dof),
        device=device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    upstream = torch.randn(
        (1, kinematics.num_dof, kinematics.num_dof),
        device=device_cfg.device,
        dtype=torch.float32,
    )
    dynamics = _make_dynamics(kinematics, device_cfg)
    dynamics.setup_batch_size(batch_size=1)

    mass_matrix = dynamics.compute_mass_matrix(joint_position)
    torch.autograd.backward(mass_matrix, grad_tensors=upstream)
    crba_gradient = joint_position.grad.detach().clone()

    reference_position = joint_position.detach().clone().requires_grad_(True)
    reference = _mass_matrix_from_rnea(
        _make_dynamics(kinematics, device_cfg),
        reference_position,
    )
    torch.autograd.backward(reference, grad_tensors=upstream)

    torch.testing.assert_close(
        crba_gradient,
        reference_position.grad,
        atol=7.0e-4,
        rtol=7.0e-4,
    )


@pytest.mark.parametrize(
    "robot_file_name",
    ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"],
)
def test_crba_backward_matches_rnea_vjp(robot_file_name: str) -> None:
    """The CRBA position VJP should match the equivalent RNEA construction."""
    kinematics, device_cfg = _load_robot(robot_file_name)
    batch_size = 2
    torch.manual_seed(31)
    joint_position = (
        0.5
        * torch.randn(
            (batch_size, kinematics.num_dof),
            device=device_cfg.device,
            dtype=torch.float32,
        )
    ).requires_grad_(True)
    upstream = torch.randn(
        (batch_size, kinematics.num_dof, kinematics.num_dof),
        device=device_cfg.device,
        dtype=torch.float32,
    )

    crba_dynamics = _make_dynamics(kinematics, device_cfg)
    crba_dynamics.setup_batch_size(batch_size=batch_size)
    mass_matrix = crba_dynamics.compute_mass_matrix(joint_position)
    torch.autograd.backward(mass_matrix, grad_tensors=upstream)
    crba_gradient = joint_position.grad.detach().clone()

    reference_position = joint_position.detach().clone().requires_grad_(True)
    reference = _mass_matrix_from_rnea(
        _make_dynamics(kinematics, device_cfg),
        reference_position,
    )
    torch.autograd.backward(reference, grad_tensors=upstream)

    torch.testing.assert_close(
        crba_gradient,
        reference_position.grad,
        atol=7.0e-4,
        rtol=7.0e-4,
    )


def test_crba_kinetic_energy_gradient_matches_finite_difference() -> None:
    """A kinetic-energy loss should have a correct position gradient."""
    kinematics, device_cfg = _load_robot("franka.yml")
    dynamics = _make_dynamics(kinematics, device_cfg)
    dynamics.setup_batch_size(batch_size=1)
    torch.manual_seed(47)
    joint_position = (
        0.3
        * torch.randn(
            (1, kinematics.num_dof),
            device=device_cfg.device,
            dtype=torch.float32,
        )
    ).requires_grad_(True)
    joint_velocity = torch.randn_like(joint_position)

    mass_matrix = dynamics.compute_mass_matrix(joint_position)
    energy = (
        0.5
        * torch.einsum(
            "bi,bij,bj->b",
            joint_velocity,
            mass_matrix,
            joint_velocity,
        ).sum()
    )
    energy.backward()
    analytical_gradient = joint_position.grad.detach().clone()

    difference_step = 2.0e-3
    finite_difference = torch.zeros_like(joint_position)
    with torch.no_grad():
        for joint_index in range(kinematics.num_dof):
            positive = joint_position.detach().clone()
            negative = joint_position.detach().clone()
            positive[:, joint_index] += difference_step
            negative[:, joint_index] -= difference_step
            positive_matrix = dynamics.compute_mass_matrix(positive).clone()
            negative_matrix = dynamics.compute_mass_matrix(negative)
            positive_energy = (
                0.5
                * torch.einsum(
                    "bi,bij,bj->b",
                    joint_velocity,
                    positive_matrix,
                    joint_velocity,
                ).sum()
            )
            negative_energy = (
                0.5
                * torch.einsum(
                    "bi,bij,bj->b",
                    joint_velocity,
                    negative_matrix,
                    joint_velocity,
                ).sum()
            )
            finite_difference[:, joint_index] = (positive_energy - negative_energy) / (
                2.0 * difference_step
            )

    torch.testing.assert_close(
        analytical_gradient,
        finite_difference,
        atol=2.0e-2,
        rtol=2.0e-2,
    )
