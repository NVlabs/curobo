# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar and gradient tests for the end-effector velocity-limit cost."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_velocity_limit import ToolVelocityLimitCost
from curobo._src.cost.cost_tool_velocity_limit_cfg import ToolVelocityLimitCostCfg
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module", autouse=True)
def setup_warp() -> None:
    """Initialize Warp before compiling the CPU and CUDA kernels."""
    init_warp(quiet=True)


def _make_cost(
    device_cfg: DeviceCfg,
    *,
    criteria: ToolVelocityLimitCriteria,
    weight=(3.0, 5.0),
    use_grad_input: bool = True,
) -> ToolVelocityLimitCost:
    return ToolVelocityLimitCost(
        ToolVelocityLimitCostCfg(
            weight=list(weight),
            tool_frames=["tool"],
            tool_velocity_limit_criteria={"tool": criteria},
            device_cfg=device_cfg,
            use_grad_input=use_grad_input,
        )
    )


def test_scalar_values_project_scale_then_apply_squared_hinge(cpu_device_cfg: DeviceCfg) -> None:
    criteria = ToolVelocityLimitCriteria.limit(
        max_linear_speed=1.0,
        max_angular_speed=0.5,
        linear_projector=[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],
        angular_projector=torch.eye(3),
        linear_axis_scale=[2.0, 1.0, 3.0],
        angular_axis_scale=[1.0, 2.0, 1.0],
        device_cfg=cpu_device_cfg,
    )
    cost_fn = _make_cost(cpu_device_cfg, criteria=criteria)
    joint_velocity = cpu_device_cfg.to_device([[[1.0, 2.0]]])
    jacobian = torch.zeros((1, 1, 1, 6, 2), **cpu_device_cfg.as_torch_dict())
    jacobian[0, 0, 0, 0, 0] = 1.0
    jacobian[0, 0, 0, 1, 1] = 1.0
    jacobian[0, 0, 0, 3, 0] = 0.5
    jacobian[0, 0, 0, 4, 1] = 0.25

    cost, linear_violation, angular_violation = cost_fn.forward(joint_velocity, jacobian)

    linear_speed = torch.linalg.vector_norm(torch.tensor([3.0, 1.5, 0.0]))
    angular_speed = torch.linalg.vector_norm(torch.tensor([0.5, 1.0, 0.0]))
    expected_linear_violation = linear_speed - 1.0
    expected_angular_violation = angular_speed - 0.5
    torch.testing.assert_close(linear_violation[0, 0, 0], expected_linear_violation)
    torch.testing.assert_close(angular_violation[0, 0, 0], expected_angular_violation)
    torch.testing.assert_close(cost[0, 0, 0], 0.5 * 3.0 * expected_linear_violation.square())
    torch.testing.assert_close(cost[0, 0, 1], 0.5 * 5.0 * expected_angular_violation.square())


def test_below_at_and_disabled_limits_are_exactly_zero(cpu_device_cfg: DeviceCfg) -> None:
    criteria = ToolVelocityLimitCriteria.limit(
        max_linear_speed=1.0,
        max_angular_speed=0.0,
        angular_projector=torch.zeros(3, 3),
        device_cfg=cpu_device_cfg,
    )
    cost_fn = _make_cost(cpu_device_cfg, criteria=criteria)
    jacobian = torch.zeros((3, 1, 1, 6, 1), **cpu_device_cfg.as_torch_dict())
    jacobian[:, 0, 0, 0, 0] = 1.0
    joint_velocity = cpu_device_cfg.to_device([[[0.0]], [[0.5]], [[1.0]]]).requires_grad_(True)

    cost, linear_violation, angular_violation = cost_fn.forward(joint_velocity, jacobian)
    assert torch.count_nonzero(cost) == 0
    assert torch.count_nonzero(linear_violation) == 0
    assert torch.count_nonzero(angular_violation) == 0
    cost.sum().backward()
    assert torch.count_nonzero(joint_velocity.grad) == 0


def _evaluate_scalar(
    cost_fn: ToolVelocityLimitCost,
    joint_velocity: torch.Tensor,
    jacobian: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    return (cost_fn.forward(joint_velocity, jacobian)[0] * multiplier).sum()


def test_joint_velocity_and_jacobian_gradients_match_finite_difference(
    cpu_device_cfg: DeviceCfg,
) -> None:
    criteria = ToolVelocityLimitCriteria.limit(
        max_linear_speed=0.2,
        max_angular_speed=0.1,
        linear_projector=torch.eye(3),
        angular_projector=torch.diag(torch.tensor([1.0, 1.0, 0.0])),
        linear_axis_scale=[1.0, 2.0, 0.5],
        angular_axis_scale=[1.5, 0.75, 2.0],
        device_cfg=cpu_device_cfg,
    )
    cost_fn = _make_cost(cpu_device_cfg, criteria=criteria, use_grad_input=True)
    qdot_base = cpu_device_cfg.to_device([[[0.7, -0.4]]])
    jacobian_base = cpu_device_cfg.to_device(
        [[[[[0.8, -0.2], [0.1, 0.5], [-0.4, 0.3], [0.2, 0.7], [-0.6, 0.1], [0.5, -0.3]]]]]
    )
    multiplier = cpu_device_cfg.to_device([[[1.7, 0.6]]])

    qdot = qdot_base.clone().requires_grad_(True)
    jacobian = jacobian_base.clone().requires_grad_(True)
    _evaluate_scalar(cost_fn, qdot, jacobian, multiplier).backward()
    analytical_qdot = qdot.grad.clone()
    analytical_jacobian = jacobian.grad.clone()

    epsilon = 1.0e-3
    finite_qdot = torch.zeros_like(qdot_base)
    for index in range(qdot_base.numel()):
        plus = qdot_base.clone().view(-1)
        minus = qdot_base.clone().view(-1)
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_qdot.view(-1)[index] = (
            _evaluate_scalar(cost_fn, plus.view_as(qdot_base), jacobian_base, multiplier)
            - _evaluate_scalar(cost_fn, minus.view_as(qdot_base), jacobian_base, multiplier)
        ) / (2.0 * epsilon)

    finite_jacobian = torch.zeros_like(jacobian_base)
    for index in range(jacobian_base.numel()):
        plus = jacobian_base.clone().view(-1)
        minus = jacobian_base.clone().view(-1)
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_jacobian.view(-1)[index] = (
            _evaluate_scalar(cost_fn, qdot_base, plus.view_as(jacobian_base), multiplier)
            - _evaluate_scalar(cost_fn, qdot_base, minus.view_as(jacobian_base), multiplier)
        ) / (2.0 * epsilon)

    torch.testing.assert_close(analytical_qdot, finite_qdot, atol=2.0e-3, rtol=2.0e-3)
    torch.testing.assert_close(analytical_jacobian, finite_jacobian, atol=2.0e-3, rtol=2.0e-3)


def test_joint_velocity_gradient_sums_all_tool_links(cpu_device_cfg: DeviceCfg) -> None:
    criteria = {
        name: ToolVelocityLimitCriteria.limit(
            max_linear_speed=0.0,
            max_angular_speed=0.0,
            angular_projector=torch.zeros(3, 3),
            device_cfg=cpu_device_cfg,
        )
        for name in ("first", "second")
    }
    cost_fn = ToolVelocityLimitCost(
        ToolVelocityLimitCostCfg(
            weight=[1.0, 0.0],
            tool_frames=["first", "second"],
            tool_velocity_limit_criteria=criteria,
            device_cfg=cpu_device_cfg,
            use_grad_input=True,
        )
    )
    joint_velocity = cpu_device_cfg.to_device([[[2.0]]]).requires_grad_(True)
    jacobian = torch.zeros((1, 1, 2, 6, 1), **cpu_device_cfg.as_torch_dict())
    jacobian[0, 0, 0, 0, 0] = 1.0
    jacobian[0, 0, 1, 1, 0] = 3.0

    cost_fn.forward(joint_velocity, jacobian)[0].sum().backward()

    # The two link contributions to d(cost)/d(qdot) are 2 and 18.
    torch.testing.assert_close(joint_velocity.grad, torch.tensor([[[20.0]]]))


def test_joint_position_gradient_includes_jacobian_derivative(
    cuda_device_cfg: DeviceCfg,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused kinematics Jacobian backward")
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_model = Kinematics(
        KinematicsCfg.from_robot_yaml_file(
            robot_data,
            tool_frames=["panda_hand"],
            device_cfg=cuda_device_cfg,
        ),
        compute_jacobian=True,
        compute_spheres=False,
    )
    criteria = ToolVelocityLimitCriteria.limit(
        max_linear_speed=0.02,
        max_angular_speed=0.05,
        device_cfg=cuda_device_cfg,
    )
    cost_fn = _make_cost(cuda_device_cfg, criteria=criteria)
    q_base = cuda_device_cfg.to_device([[[0.31, -0.82, 0.27, -1.91, 0.44, 1.16, -0.38]]])
    qdot = cuda_device_cfg.to_device([[[0.4, -0.3, 0.2, 0.1, -0.2, 0.25, -0.15]]])

    def velocity_cost(q: torch.Tensor) -> torch.Tensor:
        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names)
        )
        return cost_fn.forward(qdot, state.tool_jacobians)[0].sum()

    q = q_base.clone().requires_grad_(True)
    velocity_cost(q).backward()
    analytical = q.grad.clone()

    epsilon = 5.0e-4
    finite = torch.zeros_like(q_base)
    for joint_index in range(q_base.shape[-1]):
        plus = q_base.clone()
        minus = q_base.clone()
        plus[0, 0, joint_index] += epsilon
        minus[0, 0, joint_index] -= epsilon
        finite[0, 0, joint_index] = (velocity_cost(plus) - velocity_cost(minus)) / (2.0 * epsilon)

    torch.testing.assert_close(analytical, finite, atol=2.0e-2, rtol=5.0e-2)


def test_forward_rejects_missing_or_misaligned_runtime_data(cpu_device_cfg: DeviceCfg) -> None:
    criteria = ToolVelocityLimitCriteria.limit(
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        device_cfg=cpu_device_cfg,
    )
    cost_fn = _make_cost(cpu_device_cfg, criteria=criteria)
    qdot = torch.zeros((1, 1, 2), **cpu_device_cfg.as_torch_dict())
    jacobian = torch.zeros((1, 1, 1, 6, 2), **cpu_device_cfg.as_torch_dict())

    with pytest.raises(ValueError, match="must be provided"):
        cost_fn.forward(None, jacobian)
    with pytest.raises(ValueError, match="same batch, horizon, and dof"):
        cost_fn.forward(qdot, jacobian[..., :1])
    with pytest.raises(ValueError, match="configured tool frames"):
        cost_fn.forward(qdot, jacobian.expand(1, 1, 2, 6, 2))
