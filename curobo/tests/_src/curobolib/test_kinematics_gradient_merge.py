# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare fused sphere-gradient inputs through the standard kinematics VJP."""

from __future__ import annotations

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend import kinematics as backend
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState


@pytest.mark.parametrize("first_scale", [0.25, 2.0])
def test_fused_sphere_gradient_inputs(monkeypatch: pytest.MonkeyPatch, first_scale: float) -> None:
    """Adding two sphere cotangents inside FK backward preserves joint gradients."""
    model = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml", ["panda_hand"]))
    positions = torch.tensor(
        [[0.0, -1.0, 0.0, -2.0, 0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32
    )

    def gradient() -> torch.Tensor:
        q = positions.clone().requires_grad_(True)
        state = model.compute_kinematics(JointState.from_position(q, joint_names=model.joint_names))
        xyz = state.robot_spheres[..., :3]
        weights = torch.arange(xyz.numel(), device="cuda", dtype=torch.float32).reshape(xyz.shape)
        loss = (xyz * weights / xyz.numel()).sum() + state.tool_poses.position.square().sum()
        return torch.autograd.grad(loss, q)[0].clone()

    expected = gradient()
    original = backend.launch_kinematics_link_wrench_backward

    def split_gradient(*args: object, **kwargs: object) -> None:
        arguments = list(args)
        spheres = arguments[3]
        arguments[3] = spheres * first_scale
        kwargs["grad_spheres_addend"] = spheres * (1.0 - first_scale)
        original(*arguments, **kwargs)

    monkeypatch.setattr(backend, "launch_kinematics_link_wrench_backward", split_gradient)
    torch.testing.assert_close(gradient(), expected, atol=2.0e-5, rtol=2.0e-5)
