# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library

# Third Party
import torch

# CuRobo
from curobo._src.util.error_metrics import rotation_error_matrix, rotation_error_quaternion


class TestRotationErrorQuaternion:
    def test_identical_quaternions(self):
        q_des = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        err = rotation_error_quaternion(q_des, q)
        assert err < 0.01

    def test_opposite_quaternions(self):
        q_des = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        err = rotation_error_quaternion(q_des, q)
        assert err < 0.01

    def test_different_quaternions(self):
        q_des = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.7071, 0.7071, 0.0, 0.0])
        err = rotation_error_quaternion(q_des, q)
        assert err > 0

    def test_quaternion_cuda(self):
        if torch.cuda.is_available():
            q_des = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
            err = rotation_error_quaternion(q_des, q)
            assert err < 0.01


class TestRotationErrorMatrix:
    def test_identical_matrices(self):
        r_des = torch.eye(3)
        r = torch.eye(3)
        cost = rotation_error_matrix(r_des, r)
        assert torch.allclose(cost, torch.tensor(0.0), atol=1e-6)

    def test_different_matrices(self):
        r_des = torch.eye(3)
        r = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        cost = rotation_error_matrix(r_des, r)
        assert cost > 0

    def test_matrix_batch(self):
        r_des = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        r = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        cost = rotation_error_matrix(r_des, r)
        assert cost.shape == (5,)
        assert torch.allclose(cost, torch.zeros(5), atol=1e-6)

    def test_matrix_cuda(self):
        if torch.cuda.is_available():
            r_des = torch.eye(3, device="cuda")
            r = torch.eye(3, device="cuda")
            cost = rotation_error_matrix(r_des, r)
            assert torch.allclose(cost, torch.tensor(0.0, device="cuda"), atol=1e-6)

