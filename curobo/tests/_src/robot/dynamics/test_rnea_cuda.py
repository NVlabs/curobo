# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for CUDA RNEA forward and backward kernels.

Validates the CUDA kernel output against the NumPy reference and Pinocchio.
"""

import numpy as np
import pytest
import torch

from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
    rnea_from_kinematics_params,
)


def _load_robot(robot_yml):
    """Load curobo robot and return KinematicsParams."""
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.device_cfg import DeviceCfg

    device_cfg = DeviceCfg()
    cfg = KinematicsCfg.from_robot_yaml_file(robot_yml, device_cfg=device_cfg)
    return cfg.kinematics_config, device_cfg


def _run_cuda_rnea(kp, device_cfg, q_np, qd_np, qdd_np, gravity=-9.81):
    """Run RNEA via the Dynamics class."""
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.state.state_joint import JointState

    cfg = DynamicsCfg(
        kinematics_config=kp,
        device_cfg=device_cfg,
        gravity=[0.0, 0.0, gravity],
    )
    dynamics = Dynamics(cfg)

    batch_size = q_np.shape[0] if q_np.ndim == 2 else 1
    dynamics.setup_batch_size(batch_size=batch_size)

    q_t = torch.tensor(q_np, dtype=torch.float32, device=device_cfg.device)
    qd_t = torch.tensor(qd_np, dtype=torch.float32, device=device_cfg.device)
    qdd_t = torch.tensor(qdd_np, dtype=torch.float32, device=device_cfg.device)

    if q_t.ndim == 1:
        q_t = q_t.unsqueeze(0)
        qd_t = qd_t.unsqueeze(0)
        qdd_t = qdd_t.unsqueeze(0)

    joint_state = JointState(position=q_t, velocity=qd_t, acceleration=qdd_t)
    tau = dynamics.compute_inverse_dynamics(joint_state)
    return tau.detach().cpu().numpy()


def _run_numpy_rnea(kp, q_np, qd_np, qdd_np, gravity=-9.81):
    """Run RNEA via NumPy reference."""
    if q_np.ndim == 1:
        tau, _, _, _ = rnea_from_kinematics_params(q_np, qd_np, qdd_np, kp, gravity)
        return tau
    else:
        taus = []
        for b in range(q_np.shape[0]):
            tau, _, _, _ = rnea_from_kinematics_params(
                q_np[b], qd_np[b], qdd_np[b], kp, gravity
            )
            taus.append(tau)
        return np.stack(taus)


# ---------------------------------------------------------------------------
# CUDA forward vs NumPy reference
# ---------------------------------------------------------------------------


class TestCUDAvsNumPy:
    """Validate CUDA RNEA forward kernel against NumPy reference."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
    ]

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_gravity_compensation(self, robot_yml):
        """Gravity-only torques (qd=0, qdd=0) should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        q = np.zeros(num_dof)
        qd = np.zeros(num_dof)
        qdd = np.zeros(num_dof)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda.flatten(), tau_numpy, atol=1e-4, rtol=1e-4,
            err_msg=f"{robot_yml}: gravity compensation mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_random_single(self, robot_yml):
        """Random single configuration should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(42)
        q = rng.randn(num_dof).astype(np.float32)
        qd = rng.randn(num_dof).astype(np.float32)
        qdd = rng.randn(num_dof).astype(np.float32)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda.flatten(), tau_numpy, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: random single config mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_batch(self, robot_yml):
        """Batch of 16 random configs should match per-element NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(123)
        batch_size = 16
        q = rng.randn(batch_size, num_dof).astype(np.float32)
        qd = rng.randn(batch_size, num_dof).astype(np.float32)
        qdd = rng.randn(batch_size, num_dof).astype(np.float32)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda, tau_numpy, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: batch mismatch",
        )

    def test_large_batch_franka(self):
        """Large batch (1000) stress test."""
        kp, device_cfg = _load_robot("franka.yml")
        num_dof = kp.num_dof

        rng = np.random.RandomState(999)
        batch_size = 1000
        q = rng.randn(batch_size, num_dof).astype(np.float32)
        qd = rng.randn(batch_size, num_dof).astype(np.float32)
        qdd = rng.randn(batch_size, num_dof).astype(np.float32)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda, tau_numpy, atol=1e-3, rtol=1e-3,
            err_msg="franka large batch mismatch",
        )


# ---------------------------------------------------------------------------
# CUDA forward vs Pinocchio (gold standard)
# ---------------------------------------------------------------------------

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except Exception:
    PINOCCHIO_AVAILABLE = False


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
class TestCUDAvsPinocchio:
    """Validate CUDA RNEA forward against Pinocchio."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
        "dual_ur10e.yml",
        "unitree_g1.yml",
    ]

    @staticmethod
    def _load_pin(robot_yml):
        """Load curobo robot and matching Pinocchio model, return joint permutation."""
        import os

        kp, device_cfg = _load_robot(robot_yml)

        urdf_path = f"/tmp/test_rnea_cuda_{robot_yml.replace('.yml', '')}.urdf"
        kp.export_to_urdf(
            robot_name="dynamics", output_path=urdf_path, include_spheres=False
        )
        pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path)
        pin_data = pin_model.createData()
        if os.path.exists(urdf_path):
            os.remove(urdf_path)

        # Build joint order permutation: perm[curobo_idx] = pin_idx
        curobo_names = list(kp.joint_names)
        pin_names = [pin_model.names[i] for i in range(1, pin_model.njoints)]
        num_dof = len(curobo_names)
        perm = np.zeros(num_dof, dtype=int)
        for ci, cn in enumerate(curobo_names):
            perm[ci] = pin_names.index(cn)

        return kp, device_cfg, pin_model, pin_data, perm

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_gravity(self, robot_yml):
        """Gravity torques match Pinocchio."""
        kp, device_cfg, pin_model, pin_data, perm = self._load_pin(robot_yml)
        num_dof = kp.num_dof
        inv_perm = np.argsort(perm)

        q = np.zeros(num_dof, dtype=np.float64)
        qd = np.zeros(num_dof, dtype=np.float64)
        qdd = np.zeros(num_dof, dtype=np.float64)

        # Pinocchio: send inputs in Pinocchio joint order (inv_perm gathers
        # curobo-ordered vector into Pinocchio order), then map output back
        # to curobo order with perm.
        tau_pin_ordered = pin.rnea(
            pin_model, pin_data, q[inv_perm], qd[inv_perm], qdd[inv_perm]
        )
        tau_pin = np.array(tau_pin_ordered)[perm]

        # CUDA
        tau_cuda = _run_cuda_rnea(
            kp, device_cfg, q.astype(np.float32), qd.astype(np.float32),
            qdd.astype(np.float32)
        )

        np.testing.assert_allclose(
            tau_cuda.flatten(), tau_pin, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: CUDA vs Pinocchio gravity",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_random(self, robot_yml):
        """Random configurations match Pinocchio."""
        kp, device_cfg, pin_model, pin_data, perm = self._load_pin(robot_yml)
        num_dof = kp.num_dof
        inv_perm = np.argsort(perm)

        rng = np.random.RandomState(42)
        for _ in range(5):
            q = rng.randn(num_dof).astype(np.float64)
            qd = rng.randn(num_dof).astype(np.float64)
            qdd = rng.randn(num_dof).astype(np.float64)

            # Pinocchio: curobo→pin reorder with inv_perm, pin→curobo with perm
            tau_pin_ordered = pin.rnea(
                pin_model, pin_data, q[inv_perm], qd[inv_perm], qdd[inv_perm]
            )
            tau_pin = np.array(tau_pin_ordered)[perm]

            # CUDA
            tau_cuda = _run_cuda_rnea(
                kp, device_cfg, q.astype(np.float32), qd.astype(np.float32),
                qdd.astype(np.float32)
            )

            np.testing.assert_allclose(
                tau_cuda.flatten(), tau_pin, atol=1e-3, rtol=1e-3,
                err_msg=f"{robot_yml}: CUDA vs Pinocchio random",
            )


# ---------------------------------------------------------------------------
# Dynamics class interface tests
# ---------------------------------------------------------------------------


class TestDynamicsInterface:
    """Test the Dynamics high-level interface."""

    def test_setup_and_compute(self):
        """Basic setup_batch_size + compute_inverse_dynamics flow."""
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.state.state_joint import JointState

        kp, device_cfg = _load_robot("franka.yml")
        cfg = DynamicsCfg(
            kinematics_config=kp,
            device_cfg=device_cfg,
        )
        dynamics = Dynamics(cfg)
        dynamics.setup_batch_size(batch_size=10)

        num_dof = kp.num_dof
        q = torch.randn(10, num_dof, device=device_cfg.device, dtype=torch.float32)
        qd = torch.randn(10, num_dof, device=device_cfg.device, dtype=torch.float32)
        qdd = torch.randn(10, num_dof, device=device_cfg.device, dtype=torch.float32)

        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau = dynamics.compute_inverse_dynamics(joint_state)

        assert tau.shape == (10, num_dof)
        assert tau.device == q.device
        assert torch.all(torch.isfinite(tau))

    def test_rebatch(self):
        """Can call setup_batch_size multiple times."""
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.state.state_joint import JointState

        kp, device_cfg = _load_robot("franka.yml")
        cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
        dynamics = Dynamics(cfg)

        # First setup
        dynamics.setup_batch_size(batch_size=5)
        num_dof = kp.num_dof
        q = torch.zeros(5, num_dof, device=device_cfg.device, dtype=torch.float32)
        qd = torch.zeros(5, num_dof, device=device_cfg.device, dtype=torch.float32)
        qdd = torch.zeros(5, num_dof, device=device_cfg.device, dtype=torch.float32)
        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau1 = dynamics.compute_inverse_dynamics(joint_state)
        assert tau1.shape == (5, num_dof)

        # Second setup with different batch size
        dynamics.setup_batch_size(batch_size=20)
        q2 = torch.zeros(20, num_dof, device=device_cfg.device, dtype=torch.float32)
        qd2 = torch.zeros(20, num_dof, device=device_cfg.device, dtype=torch.float32)
        qdd2 = torch.zeros(20, num_dof, device=device_cfg.device, dtype=torch.float32)
        js2 = JointState(position=q2, velocity=qd2, acceleration=qdd2)
        tau2 = dynamics.compute_inverse_dynamics(js2)
        assert tau2.shape == (20, num_dof)


# ---------------------------------------------------------------------------
# CUDA backward (VJP) vs NumPy backward
# ---------------------------------------------------------------------------



def _run_cuda_backward(kp, device_cfg, q_np, qd_np, qdd_np, tau_bar_np, gravity=-9.81):
    """Run RNEA backward via autograd by calling forward then backward."""
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.state.state_joint import JointState

    cfg = DynamicsCfg(
        kinematics_config=kp,
        device_cfg=device_cfg,
        gravity=[0.0, 0.0, gravity],
    )
    dynamics = Dynamics(cfg)

    # Ensure 2D from the start so tensors are leaf nodes with requires_grad
    if q_np.ndim == 1:
        q_2d = q_np[np.newaxis, :]
        qd_2d = qd_np[np.newaxis, :]
        qdd_2d = qdd_np[np.newaxis, :]
        tb_2d = tau_bar_np[np.newaxis, :]
    else:
        q_2d = q_np
        qd_2d = qd_np
        qdd_2d = qdd_np
        tb_2d = tau_bar_np

    batch_size = q_2d.shape[0]
    dynamics.setup_batch_size(batch_size=batch_size)

    q_t = torch.tensor(q_2d, dtype=torch.float32, device=device_cfg.device,
                        requires_grad=True)
    qd_t = torch.tensor(qd_2d, dtype=torch.float32, device=device_cfg.device,
                         requires_grad=True)
    qdd_t = torch.tensor(qdd_2d, dtype=torch.float32, device=device_cfg.device,
                          requires_grad=True)
    tau_bar_t = torch.tensor(tb_2d, dtype=torch.float32, device=device_cfg.device)

    joint_state = JointState(position=q_t, velocity=qd_t, acceleration=qdd_t)
    tau = dynamics.compute_inverse_dynamics(joint_state)
    tau.backward(tau_bar_t)

    return (
        q_t.grad.detach().cpu().numpy(),
        qd_t.grad.detach().cpu().numpy(),
        qdd_t.grad.detach().cpu().numpy(),
    )


def _run_numpy_backward(kp, q_np, qd_np, qdd_np, tau_bar_np, gravity=-9.81):
    """Run RNEA backward via NumPy reference."""
    from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
        rnea_backward_from_kinematics_params,
        rnea_from_kinematics_params,
    )

    if q_np.ndim == 1:
        _, v, a, f = rnea_from_kinematics_params(
            q_np, qd_np, qdd_np, kp, gravity
        )
        gq, gqd, gqdd = rnea_backward_from_kinematics_params(
            tau_bar_np, q_np, qd_np, qdd_np, v, a, f, kp, gravity
        )
        return gq, gqd, gqdd
    else:
        gqs, gqds, gqdds = [], [], []
        for b in range(q_np.shape[0]):
            _, v, a, f = rnea_from_kinematics_params(
                q_np[b], qd_np[b], qdd_np[b], kp, gravity
            )
            gq, gqd, gqdd = rnea_backward_from_kinematics_params(
                tau_bar_np[b], q_np[b], qd_np[b], qdd_np[b],
                v, a, f, kp, gravity
            )
            gqs.append(gq)
            gqds.append(gqd)
            gqdds.append(gqdd)
        return np.stack(gqs), np.stack(gqds), np.stack(gqdds)


class TestCUDABackwardVsNumPy:
    """Validate CUDA RNEA backward (VJP) kernel against NumPy reference."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
        "dual_ur10e.yml",
        "unitree_g1.yml",
    ]

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_grad_q_identity_tau_bar(self, robot_yml):
        """Gradient with tau_bar = ones should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(42)
        q = rng.randn(num_dof).astype(np.float32)
        qd = rng.randn(num_dof).astype(np.float32)
        qdd = rng.randn(num_dof).astype(np.float32)
        tau_bar = np.ones(num_dof, dtype=np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda.flatten(), gq_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_q mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda.flatten(), gqd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qd mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda.flatten(), gqdd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qdd mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_grad_random_tau_bar(self, robot_yml):
        """Gradient with random tau_bar should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(123)
        q = rng.randn(num_dof).astype(np.float32)
        qd = rng.randn(num_dof).astype(np.float32)
        qdd = rng.randn(num_dof).astype(np.float32)
        tau_bar = rng.randn(num_dof).astype(np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda.flatten(), gq_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_q random tau_bar mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda.flatten(), gqd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qd random tau_bar mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda.flatten(), gqdd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qdd random tau_bar mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ["franka.yml", "ur10e.yml", "unitree_g1.yml"])
    def test_grad_batch(self, robot_yml):
        """Batched backward should match per-element NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof
        batch_size = 8

        rng = np.random.RandomState(99)
        q = rng.randn(batch_size, num_dof).astype(np.float32)
        qd = rng.randn(batch_size, num_dof).astype(np.float32)
        qdd = rng.randn(batch_size, num_dof).astype(np.float32)
        tau_bar = rng.randn(batch_size, num_dof).astype(np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda, gq_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_q batch mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda, gqd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qd batch mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda, gqdd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qdd batch mismatch",
        )

    def test_grad_zero_input(self):
        """Gradient at q=0, qd=0, qdd=0 should match NumPy (gravity only)."""
        kp, device_cfg = _load_robot("franka.yml")
        num_dof = kp.num_dof

        q = np.zeros(num_dof, dtype=np.float32)
        qd = np.zeros(num_dof, dtype=np.float32)
        qdd = np.zeros(num_dof, dtype=np.float32)
        tau_bar = np.ones(num_dof, dtype=np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda.flatten(), gq_np, atol=1e-4, rtol=1e-4,
            err_msg="franka: grad_q at zero mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda.flatten(), gqd_np, atol=1e-4, rtol=1e-4,
            err_msg="franka: grad_qd at zero mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda.flatten(), gqdd_np, atol=1e-4, rtol=1e-4,
            err_msg="franka: grad_qdd at zero mismatch",
        )


# ---------------------------------------------------------------------------
# Optimization convergence: fwd → ||τ||² → bwd → update qdd  (no CUDA graph)
# ---------------------------------------------------------------------------


def _create_dynamics(robot_yml, batch_size):
    """Create a Dynamics model for the given robot."""
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

    kp, device_cfg = _load_robot(robot_yml)

    cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
    dynamics = Dynamics(cfg)
    dynamics.setup_batch_size(batch_size)
    return dynamics, kp, device_cfg


def _run_optimization(dynamics, num_dof, device, batch_size=10, n_steps=1000, lr=0.001):
    """Run gradient descent on qdd to minimize ||τ||².

    The unique minimum is qdd* = -M⁻¹(C·qd + g), i.e. the free-fall acceleration.
    Since M is PD, the problem is convex in qdd and must converge.

    Returns:
        losses: list of loss values per step.
        tau_final: [batch, num_dof] final torques.
        qdd_final: [batch, num_dof] optimized accelerations.
    """
    from curobo._src.state.state_joint import JointState

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_dof, device=device)
    qd = torch.randn(batch_size, num_dof, device=device)
    qdd = torch.zeros(batch_size, num_dof, device=device, requires_grad=True)

    losses = []
    for _step in range(n_steps):
        if qdd.grad is not None:
            qdd.grad.zero_()
        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau = dynamics.compute_inverse_dynamics(joint_state)
        loss = (tau ** 2).sum()
        loss.backward()
        with torch.no_grad():
            grad = qdd.grad.clamp(-1e3, 1e3)
            qdd -= lr * grad
        losses.append(loss.item())

    # Final evaluation (no grad)
    with torch.no_grad():
        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau_final = dynamics.compute_inverse_dynamics(joint_state)

    return losses, tau_final.detach(), qdd.detach()


class TestOptimizationConvergence:
    """Verify fwd+bwd drives tau→0 by optimizing qdd (gradient descent)."""

    ROBOT_CONFIGS = ["franka.yml", "ur10e.yml"]

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_curobo_converges(self, robot_yml):
        """CuRobo RNEA: loss should drop significantly, proving gradients work."""
        dynamics, kp, device_cfg = _create_dynamics(robot_yml, 10)
        losses, tau_final, _ = _run_optimization(
            dynamics, kp.num_dof, device_cfg.device
        )
        # Just verify gradients reduce the loss significantly
        assert losses[-1] < losses[0] * 0.1, (
            f"{robot_yml}: cuRobo loss didn't reduce "
            f"({losses[-1]:.2f} vs {losses[0]:.2f})"
        )
        assert tau_final.abs().max() < 10.0, (
            f"{robot_yml}: tau too large, max={tau_final.abs().max():.2f}"
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
