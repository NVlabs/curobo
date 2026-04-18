# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for kinematics jacobian accuracy and jacobian backward (gradient) correctness.

Tests verify:
1. Forward jacobian values match finite difference approximation on FK.
2. Backward pass through jacobian (dJ/dq) matches finite differences on a scalar loss.
3. Element-wise gradient of individual jacobian entries matches finite differences.
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

# Robot configs for jacobian testing (single-chain robots with reasonable DOF).
ROBOT_CONFIGS = [
    #"franka.yml",
    #"ur10e.yml",
    "unitree_g1.yml",
]


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in (w, x, y, z) format.

    Args:
        q1: First quaternion [batch, 4] in (w, x, y, z) format.
        q2: Second quaternion [batch, 4] in (w, x, y, z) format.

    Returns:
        Product quaternion [batch, 4].
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate (inverse for unit quaternions) in (w, x, y, z) format.

    Args:
        q: Quaternion [batch, 4] in (w, x, y, z) format.

    Returns:
        Conjugate quaternion [batch, 4].
    """
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)


def _align_quaternion_sign(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """Align quaternion sign to reference to handle double-cover ambiguity.

    Even with w >= 0 enforced by the kernel, quaternions can still flip sign at the
    w = 0 boundary (rotation angle = pi). This aligns q to q_ref via positive dot product.

    Args:
        q: Quaternion to align [batch, 4].
        q_ref: Reference quaternion [batch, 4].

    Returns:
        Aligned quaternion [batch, 4].
    """
    dot = (q * q_ref).sum(dim=-1, keepdim=True)
    return torch.where(dot < 0, -q, q)



def _create_robot_model(
    robot_file: str, device_cfg: DeviceCfg, compute_jacobian: bool = True
) -> Kinematics:
    """Create a robot kinematics model from a config file.

    Args:
        robot_file: Robot YAML config filename (e.g. "franka.yml").
        device_cfg: Device configuration for tensors.
        compute_jacobian: Whether to compute jacobian in forward pass.

    Returns:
        Kinematics model instance.
    """
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, device_cfg=device_cfg)
    return Kinematics(cfg, compute_jacobian=compute_jacobian)


def _sample_joint_positions(robot_model: Kinematics, n_random: int = 3):
    """Sample test joint configurations: zero, default, and random near default.

    Random configs are sampled within ±1 radian (or meter) of the default configuration,
    clamped to joint limits. This avoids extreme configurations (e.g. virtual base joints
    with ±100m limits) where finite difference approximations become unreliable.

    Args:
        robot_model: Kinematics model.
        n_random: Number of random configurations to sample.

    Returns:
        List of (name, q_tensor) tuples.
    """
    dof = robot_model.get_dof()
    device = robot_model.device_cfg.device
    dtype = robot_model.device_cfg.dtype

    configs = []

    # Zero configuration
    configs.append(("zero", torch.zeros(1, dof, device=device, dtype=dtype)))

    # Default configuration from robot YAML
    default_q = robot_model.default_joint_position.unsqueeze(0).to(dtype=dtype)
    configs.append(("default", default_q))

    # Random configurations near default, clamped to joint limits
    joint_limits = robot_model.get_joint_limits()
    if joint_limits is not None and joint_limits.position is not None:
        q_min = joint_limits.position[0]
        q_max = joint_limits.position[1]
        perturbation_range = 1.0  # ±1 radian (or meter for prismatic joints)
        sample_min = torch.max(q_min, default_q[0] - perturbation_range)
        sample_max = torch.min(q_max, default_q[0] + perturbation_range)
        for i in range(n_random):
            alpha = torch.rand(1, dof, device=device, dtype=dtype)
            q_rand = sample_min + (sample_max - sample_min) * alpha
            configs.append((f"random_{i}", q_rand))

    return configs


@pytest.mark.parametrize("robot_file", ROBOT_CONFIGS)
def test_jacobian_accuracy(robot_file: str, cuda_device_cfg: DeviceCfg):
    """Test jacobian values match finite difference approximation on FK.

    Verifies the forward jacobian computation by comparing each column of the
    analytical jacobian against central finite differences on the FK output
    (position for linear part, quaternion for angular part).
    """
    robot_model = _create_robot_model(robot_file, cuda_device_cfg)
    num_dof = robot_model.get_dof()
    tool_frames = robot_model.tool_frames
    test_configs = _sample_joint_positions(robot_model, n_random=3)

    eps = 1e-4
    abs_tol = 5e-3
    rel_tol = 0.02
    failures = []

    for config_name, q in test_configs:
        batch_size = q.shape[0]

        # Get jacobian at this configuration; clone to avoid buffer aliasing
        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names)
        )
        computed_jacobian = state.tool_jacobians.clone()
        ref_quat = state.tool_poses.quaternion.clone()

        if computed_jacobian is None:
            continue

        num_links = computed_jacobian.shape[2]

        for link_idx in range(num_links):
            fd_jacobian = torch.zeros(batch_size, 6, num_dof, device=q.device, dtype=q.dtype)
            quat_ref = ref_quat[:, 0, link_idx, :]

            for j in range(num_dof):
                q_plus = q.clone()
                q_plus[:, j] += eps
                state_plus = robot_model.compute_kinematics(
                    JointState.from_position(q_plus, joint_names=robot_model.joint_names)
                )
                pos_plus = state_plus.tool_poses.position[:, 0, link_idx, :].clone()
                quat_plus = state_plus.tool_poses.quaternion[:, 0, link_idx, :].clone()

                q_minus = q.clone()
                q_minus[:, j] -= eps
                state_minus = robot_model.compute_kinematics(
                    JointState.from_position(q_minus, joint_names=robot_model.joint_names)
                )
                pos_minus = state_minus.tool_poses.position[:, 0, link_idx, :].clone()
                quat_minus = state_minus.tool_poses.quaternion[:, 0, link_idx, :].clone()

                quat_plus = _align_quaternion_sign(quat_plus, quat_ref)
                quat_minus = _align_quaternion_sign(quat_minus, quat_ref)

                fd_jacobian[:, 0:3, j] = (pos_plus - pos_minus) / (2 * eps)

                dq = _quaternion_multiply(quat_plus, _quaternion_conjugate(quat_minus))
                fd_jacobian[:, 3:6, j] = 2.0 * dq[:, 1:4] / (2 * eps)

            computed_jac_link = computed_jacobian[:, 0, link_idx, :, :]
            abs_diff = torch.abs(computed_jac_link - fd_jacobian)
            tol = abs_tol + rel_tol * torch.abs(fd_jacobian)
            passed = bool(torch.all(abs_diff <= tol).item())

            link_name = (
                tool_frames[link_idx] if link_idx < len(tool_frames) else f"link_{link_idx}"
            )
            max_abs = abs_diff.max().item()

            if not passed:
                failures.append(
                    f"[{config_name}] Link '{link_name}': max_abs={max_abs:.2e}"
                )

    assert not failures, (
        f"Jacobian accuracy failures for {robot_file}:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("robot_file", ROBOT_CONFIGS)
def test_jacobian_gradcheck_manual(robot_file: str, cuda_device_cfg: DeviceCfg):
    """Test jacobian backward with manual finite difference checks.

    Verifies that d/dq [ sum(J^2) ] computed via backprop matches finite differences.
    This tests the jacobian backward kernel (dJ/dq).
    """
    robot_model = _create_robot_model(robot_file, cuda_device_cfg)
    # Use fewer configs for backward tests (slower per config)
    test_configs = _sample_joint_positions(robot_model, n_random=1)

    def jacobian_loss(joint_angles):
        kin_state = robot_model.compute_kinematics(
            JointState.from_position(joint_angles, joint_names=robot_model.joint_names)
        )
        return torch.sum(kin_state.tool_jacobians**2)

    abs_tol = 5e-2
    failures = []

    for config_name, q_base in test_configs:
        q = q_base.clone().requires_grad_(True)

        # Compute analytical gradient via backprop
        if q.grad is not None:
            q.grad.zero_()

        loss = jacobian_loss(q)
        loss.backward()

        analytical_grad = q.grad.clone()

        # Find best finite difference gradient across multiple epsilon values
        epsilons = [1e-3, 5e-4, 1e-4]
        best_error = float("inf")
        best_eps = None
        best_fd_grad = None

        for fd_eps in epsilons:
            fd_grad = torch.zeros_like(q)

            for i in range(q.shape[0]):
                for j in range(q.shape[1]):
                    q_pos = q.clone().detach()
                    q_pos[i, j] += fd_eps
                    loss_pos = jacobian_loss(q_pos)

                    q_neg = q.clone().detach()
                    q_neg[i, j] -= fd_eps
                    loss_neg = jacobian_loss(q_neg)

                    fd_grad[i, j] = (loss_pos - loss_neg) / (2 * fd_eps)

            max_error = torch.abs(analytical_grad - fd_grad).max().item()
            if max_error < best_error:
                best_error = max_error
                best_eps = fd_eps
                best_fd_grad = fd_grad

        if best_error > abs_tol:
            failures.append(
                f"[{config_name}]: eps={best_eps}, max_abs={best_error:.2e} > {abs_tol}"
            )

    assert not failures, (
        f"Jacobian gradcheck failures for {robot_file}:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("robot_file", ROBOT_CONFIGS)
def test_jacobian_gradient_accuracy(robot_file: str, cuda_device_cfg: DeviceCfg):
    """Test gradient of jacobian (dJ/dq) matches finite difference approximation.

    Verifies that the backward pass through the jacobian computation correctly
    computes dJ/dq by backpropagating through individual jacobian elements and
    comparing against numerical differentiation.
    """
    robot_model = _create_robot_model(robot_file, cuda_device_cfg)
    num_dof = robot_model.get_dof()
    eps = 1e-4

    # Use fewer configs for element-wise backward tests
    test_configs = _sample_joint_positions(robot_model, n_random=1)

    # Build test elements based on this robot's DOF: sample across rows and columns
    state = robot_model.compute_kinematics(
        JointState.from_position(test_configs[0][1], joint_names=robot_model.joint_names)
    )
    if state.tool_jacobians is None:
        pytest.skip("Jacobian computation not available")

    num_links = state.tool_jacobians.shape[2]

    # Test representative jacobian elements: [link_idx, row_idx, col_idx]
    # Rows 0-2 = linear (x,y,z), rows 3-5 = angular (x,y,z)
    test_elements = []
    for row in range(6):
        col = min(row, num_dof - 1)
        test_elements.append((0, row, col))
    # Also test last joint column
    if num_dof > 1:
        test_elements.append((0, 0, num_dof - 1))

    abs_tol = 5e-3
    rel_tol = 0.05
    failures = []

    for config_name, q_base in test_configs:
        for link_idx, row_idx, col_idx in test_elements:
            if link_idx >= num_links or col_idx >= num_dof:
                continue

            # Analytical gradient via backprop
            q = q_base.clone().requires_grad_(True)

            try:
                state = robot_model.compute_kinematics(
                    JointState.from_position(q, joint_names=robot_model.joint_names)
                )
                target = state.tool_jacobians[0, 0, link_idx, row_idx, col_idx]
                target.backward()
            except RuntimeError as e:
                if "not supported yet" in str(e):
                    pytest.skip("Jacobian backward pass not yet implemented")
                raise

            analytical_grad = q.grad.clone()

            # Finite difference gradient; clone to avoid buffer aliasing
            fd_grad = torch.zeros_like(q)
            for j in range(num_dof):
                q_plus = q_base.clone()
                q_plus[:, j] += eps
                jac_plus = robot_model.compute_kinematics(
                    JointState.from_position(q_plus, joint_names=robot_model.joint_names)
                ).tool_jacobians[0, 0, link_idx, row_idx, col_idx].clone()

                q_minus = q_base.clone()
                q_minus[:, j] -= eps
                jac_minus = robot_model.compute_kinematics(
                    JointState.from_position(q_minus, joint_names=robot_model.joint_names)
                ).tool_jacobians[0, 0, link_idx, row_idx, col_idx].clone()

                fd_grad[0, j] = (jac_plus - jac_minus) / (2 * eps)

            # Use allclose-style tolerance: |diff| <= atol + rtol * |ref|
            abs_diff = torch.abs(analytical_grad - fd_grad)
            tolerance = abs_tol + rel_tol * torch.abs(fd_grad)
            element_passed = bool(torch.all(abs_diff <= tolerance).item())

            if not element_passed:
                max_abs = abs_diff.max().item()
                failures.append(
                    f"[{config_name}] J[{link_idx},{row_idx},{col_idx}]: "
                    f"max_abs={max_abs:.2e}"
                )

    assert not failures, (
        f"Jacobian gradient accuracy failures for {robot_file}:\n" + "\n".join(failures)
    )
