# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the NumPy reference RNEA implementation.

Validates against:
1. Known analytical results (zero-velocity gravity compensation)
2. Numerical properties (symmetry, energy conservation)
3. Pinocchio cross-validation (forward and backward)
"""

import numpy as np
import pytest

from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
    FIXED,
    Y_ROT,
    build_spatial_inertia,
    cross3,
    homogeneous_to_spatial_transform,
    motion_cross_product,
    rnea,
    skew,
    spatial_force_cross,
    spatial_inertia_multiply,
    spatial_transform_multiply,
    spatial_transform_transpose_multiply,
)

# ---------------------------------------------------------------------------
# Spatial algebra unit tests
# ---------------------------------------------------------------------------


class TestSpatialAlgebra:
    """Test spatial algebra helpers."""

    def test_skew_antisymmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        np.testing.assert_allclose(S + S.T, 0, atol=1e-15)

    def test_skew_cross_product(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(skew(a) @ b, cross3(a, b), atol=1e-15)

    def test_cross3_anticommutative(self):
        a = np.random.randn(3)
        b = np.random.randn(3)
        np.testing.assert_allclose(cross3(a, b), -cross3(b, a), atol=1e-14)

    def test_spatial_transform_identity(self):
        """Identity transform (R=I, p=0) should leave vectors unchanged."""
        R = np.eye(3)
        p = np.zeros(3)
        v = np.random.randn(6)
        result = spatial_transform_multiply(R, p, v)
        np.testing.assert_allclose(result, v, atol=1e-14)

    def test_spatial_transform_vs_matrix(self):
        """Structured multiply should match full 6x6 matrix multiply."""
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # 90° about Y
        p = np.array([0.1, 0.2, 0.3])
        v = np.random.randn(6)

        X = homogeneous_to_spatial_transform(R, p)
        result_matrix = X @ v
        result_structured = spatial_transform_multiply(R, p, v)
        np.testing.assert_allclose(result_structured, result_matrix, atol=1e-14)

    def test_spatial_transform_transpose_vs_matrix(self):
        """Structured X^T multiply should match full matrix."""
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        p = np.array([0.1, 0.2, 0.3])
        f = np.random.randn(6)

        X = homogeneous_to_spatial_transform(R, p)
        result_matrix = X.T @ f
        result_structured = spatial_transform_transpose_multiply(R, p, f)
        np.testing.assert_allclose(result_structured, result_matrix, atol=1e-14)

    def test_spatial_inertia_symmetric(self):
        """Spatial inertia matrix should be symmetric positive semi-definite."""
        mass = 2.5
        com = np.array([0.1, -0.05, 0.03])
        inertia = np.array([0.01, 0.02, 0.015, 0.001, 0.002, -0.001])
        I = build_spatial_inertia(mass, com, inertia)
        np.testing.assert_allclose(I, I.T, atol=1e-15)
        # Check positive semi-definite
        eigvals = np.linalg.eigvalsh(I)
        assert np.all(eigvals >= -1e-10)

    def test_spatial_inertia_multiply_vs_matrix(self):
        """On-the-fly I*v should match full 6x6 matrix multiply."""
        mass = 2.5
        com = np.array([0.1, -0.05, 0.03])
        inertia = np.array([0.01, 0.02, 0.015, 0.001, 0.002, -0.001])
        v = np.random.randn(6)

        I = build_spatial_inertia(mass, com, inertia)
        result_matrix = I @ v
        result_onthefly = spatial_inertia_multiply(mass, com, inertia, v)
        np.testing.assert_allclose(result_onthefly, result_matrix, atol=1e-13)

    def test_motion_cross_lie_bracket_antisymmetric(self):
        """Lie bracket antisymmetry: [v1, v2] = -[v2, v1]."""
        v1 = np.random.randn(6)
        v2 = np.random.randn(6)
        lhs = motion_cross_product(v1, v2)
        rhs = -motion_cross_product(v2, v1)
        np.testing.assert_allclose(lhs, rhs, atol=1e-13)

    def test_force_cross_dual(self):
        """crf(v)*f = -crm(v)^T * f."""
        v = np.random.randn(6)
        f = np.random.randn(6)
        # Build crm(v) as 6x6 matrix
        omega = v[:3]
        v_lin = v[3:]
        crm = np.zeros((6, 6))
        crm[:3, :3] = skew(omega)
        crm[3:, :3] = skew(v_lin)
        crm[3:, 3:] = skew(omega)
        result_matrix = -crm.T @ f
        result_func = spatial_force_cross(v, f)
        np.testing.assert_allclose(result_func, result_matrix, atol=1e-14)


# ---------------------------------------------------------------------------
# Simple pendulum test (1-DOF, analytical solution known)
# ---------------------------------------------------------------------------


class TestSinglePendulum:
    """Test RNEA on a single revolute joint (pendulum)."""

    def _make_pendulum(self, length=1.0, mass=1.0):
        """Create data for a single-link pendulum rotating about Y axis.

        Link extends along X, CoM at (length/2, 0, 0). Gravity along -Z.
        Y-axis rotation creates a pendulum in the XZ plane.
        """
        num_links = 1
        num_dof = 1

        # Fixed transform: identity rotation, child origin at (0,0,0) relative to parent
        fixed_transforms = np.zeros((num_links, 3, 4))
        fixed_transforms[0, :3, :3] = np.eye(3)
        fixed_transforms[0, :3, 3] = [0, 0, 0]

        link_map = np.array([-1])  # parent is world
        joint_map = np.array([0])  # joint 0
        joint_map_type = np.array([Y_ROT])  # revolute about Y
        joint_offset_map = np.array([[1.0, 0.0]])  # [multiplier, offset], no mimic

        # CoM at (length/2, 0, 0), mass
        link_masses_com = np.array([[length / 2, 0, 0, mass]])

        # Thin rod about CoM: Iyy = m*L^2/12 (rotation about Y)
        iyy = mass * length**2 / 12.0
        link_inertias = np.array([[1e-6, iyy, 1e-6, 0, 0, 0]])

        return dict(
            fixed_transforms=fixed_transforms,
            link_map=link_map,
            joint_map=joint_map,
            joint_map_type=joint_map_type,
            joint_offset_map=joint_offset_map,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
        )

    def test_gravity_compensation_at_horizontal(self):
        """Pendulum horizontal (q=0): tau = -m*g*L/2 (motor opposes gravity)."""
        L, m = 1.0, 2.0
        data = self._make_pendulum(length=L, mass=m)

        q = np.array([0.0])  # horizontal
        qd = np.array([0.0])
        qdd = np.array([0.0])

        tau, v, a, f = rnea(q, qd, qdd, gravity=-9.81, **data)

        # At q=0 (horizontal), motor torque to hold against gravity = -m*g*(L/2)
        expected_tau = -m * 9.81 * (L / 2)
        np.testing.assert_allclose(tau[0], expected_tau, rtol=1e-10)

    def test_gravity_compensation_at_vertical(self):
        """Pendulum vertical (q=-pi/2 about Y): tau ≈ 0 (link along -Z, gravity along link)."""
        L, m = 1.0, 2.0
        data = self._make_pendulum(length=L, mass=m)

        # At q=-pi/2 about Y, link that was along X now points along -Z
        # CoM in world frame: R_y(-pi/2)*[L/2,0,0] = [0, 0, -L/2]
        # Gravity force: [0, 0, -m*g]
        # Torque about Y: (r x F).y = ([0,0,-L/2] x [0,0,-m*g]).y = 0
        q = np.array([-np.pi / 2])
        qd = np.array([0.0])
        qdd = np.array([0.0])

        tau, v, a, f = rnea(q, qd, qdd, gravity=-9.81, **data)
        np.testing.assert_allclose(tau[0], 0.0, atol=1e-10)

    def test_zero_velocity_zero_accel(self):
        """With qd=0, qdd=0, tau should be pure gravity compensation."""
        data = self._make_pendulum()
        q = np.array([0.5])
        qd = np.array([0.0])

        tau_with_qdd, _, _, _ = rnea(q, qd, np.zeros(1), gravity=-9.81, **data)
        tau_without_qdd, _, _, _ = rnea(q, qd, None, gravity=-9.81, **data)
        np.testing.assert_allclose(tau_with_qdd, tau_without_qdd, atol=1e-14)

    def test_pure_inertia_no_gravity(self):
        """With gravity=0, qd=0: tau = I_eff * qdd."""
        L, m = 1.0, 2.0
        data = self._make_pendulum(length=L, mass=m)

        q = np.array([0.0])
        qd = np.array([0.0])
        qdd = np.array([1.0])

        tau, _, _, _ = rnea(q, qd, qdd, gravity=0.0, **data)

        # tau = M*qdd: effective inertia about Y = m*L^2/3
        I_eff = m * L**2 / 3.0
        np.testing.assert_allclose(tau[0], I_eff * qdd[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# Two-link serial chain test
# ---------------------------------------------------------------------------


class TestTwoLinkChain:
    """Test RNEA on a 2-link serial chain (double pendulum)."""

    def _make_double_pendulum(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
        """Create a 2-link planar pendulum (both revolute Y, swinging in XZ plane)."""
        num_links = 2

        fixed_transforms = np.zeros((num_links, 3, 4))

        # Link 0: attached to world at origin, extends along X
        fixed_transforms[0, :3, :3] = np.eye(3)
        fixed_transforms[0, :3, 3] = [0, 0, 0]

        # Link 1: attached to link 0 at (L1, 0, 0)
        fixed_transforms[1, :3, :3] = np.eye(3)
        fixed_transforms[1, :3, 3] = [L1, 0, 0]

        link_map = np.array([-1, 0])  # link 0's parent is world, link 1's parent is link 0
        joint_map = np.array([0, 1])  # joint indices
        joint_map_type = np.array([Y_ROT, Y_ROT])
        joint_offset_map = np.array([[1.0, 0.0], [1.0, 0.0]])

        # CoM at middle of each link
        link_masses_com = np.array(
            [
                [L1 / 2, 0, 0, m1],
                [L2 / 2, 0, 0, m2],
            ]
        )

        # Thin rods (inertia about Y for each)
        link_inertias = np.array(
            [
                [1e-6, m1 * L1**2 / 12, 1e-6, 0, 0, 0],
                [1e-6, m2 * L2**2 / 12, 1e-6, 0, 0, 0],
            ]
        )

        return dict(
            fixed_transforms=fixed_transforms,
            link_map=link_map,
            joint_map=joint_map,
            joint_map_type=joint_map_type,
            joint_offset_map=joint_offset_map,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
        )

    def test_gravity_at_zero_config(self):
        """At q=[0,0], both links horizontal. Known gravity torques."""
        L1, L2 = 1.0, 0.8
        m1, m2 = 2.0, 1.5
        data = self._make_double_pendulum(L1, L2, m1, m2)
        g = 9.81

        q = np.array([0.0, 0.0])
        qd = np.array([0.0, 0.0])
        qdd = np.array([0.0, 0.0])

        tau, _, _, _ = rnea(q, qd, qdd, gravity=-g, **data)

        # tau[1] = -m2*g*L2/2 (motor opposes link 2's gravity about joint 2)
        expected_tau1 = -m2 * g * L2 / 2
        np.testing.assert_allclose(tau[1], expected_tau1, rtol=1e-10)

        # tau[0] = -(m1*g*L1/2 + m2*g*(L1 + L2/2)) (motor opposes both links' gravity)
        expected_tau0 = -(m1 * g * L1 / 2 + m2 * g * (L1 + L2 / 2))
        np.testing.assert_allclose(tau[0], expected_tau0, rtol=1e-10)

    def test_numerical_gradient_check(self):
        """Finite-difference check: d(tau)/d(q) via perturbation."""
        data = self._make_double_pendulum()

        q = np.array([0.3, -0.5])
        qd = np.array([0.1, 0.2])
        qdd = np.array([0.0, 0.0])

        tau0, _, _, _ = rnea(q, qd, qdd, gravity=-9.81, **data)

        eps = 1e-7
        dtau_dq = np.zeros((2, 2))
        for j in range(2):
            q_plus = q.copy()
            q_plus[j] += eps
            tau_plus, _, _, _ = rnea(q_plus, qd, qdd, gravity=-9.81, **data)
            dtau_dq[:, j] = (tau_plus - tau0) / eps

        # Just check that the gradient is non-trivial and finite
        assert np.all(np.isfinite(dtau_dq))
        assert np.linalg.norm(dtau_dq) > 0.1  # should be non-trivial


# ---------------------------------------------------------------------------
# Test with curobo's KinematicsParams (Franka robot)
# ---------------------------------------------------------------------------


class TestWithCuroboRobot:
    """Test RNEA using a real robot loaded through curobo."""

    @pytest.fixture
    def franka_params(self):
        """Load Franka kinematics params."""
        try:
            from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

            cfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
            return cfg.kinematics_config
        except Exception as e:
            pytest.skip(f"Could not load Franka robot: {e}")

    def test_gravity_compensation_franka(self, franka_params):
        """Gravity compensation torques should be finite and non-trivial."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_from_kinematics_params,
        )

        kp = franka_params
        num_dof = kp.num_dof
        q = np.zeros(num_dof)
        qd = np.zeros(num_dof)

        tau, v, a, f = rnea_from_kinematics_params(q, qd, None, kp, gravity=-9.81)

        assert tau.shape == (num_dof,)
        assert np.all(np.isfinite(tau))
        # Gravity compensation should be non-zero for a non-trivial robot
        assert np.linalg.norm(tau) > 0.1


# ---------------------------------------------------------------------------
# Test fixed joints
# ---------------------------------------------------------------------------


class TestFixedJoints:
    """Test that fixed joints are handled correctly."""

    def test_fixed_joint_passthrough(self):
        """A fixed joint should just propagate v, a through the transform."""
        num_links = 2

        fixed_transforms = np.zeros((num_links, 3, 4))
        fixed_transforms[0, :3, :3] = np.eye(3)
        fixed_transforms[1, :3, :3] = np.eye(3)
        fixed_transforms[1, :3, 3] = [1.0, 0, 0]  # offset by 1m along X

        link_map = np.array([-1, 0])
        joint_map = np.array([0, -1])  # link 1 has a fixed joint
        joint_map_type = np.array([Y_ROT, FIXED])
        joint_offset_map = np.array([[1.0, 0.0], [1.0, 0.0]])

        link_masses_com = np.array(
            [
                [0.5, 0, 0, 1.0],  # link 0
                [0.5, 0, 0, 1.0],  # link 1 (fixed to link 0)
            ]
        )
        link_inertias = np.array(
            [
                [1e-4, 1e-4, 1e-4, 0, 0, 0],
                [1e-4, 1e-4, 1e-4, 0, 0, 0],
            ]
        )

        q = np.array([0.0])  # only 1 DOF
        qd = np.array([0.0])
        qdd = np.array([0.0])

        tau, v, a, f = rnea(
            q, qd, qdd,
            fixed_transforms=fixed_transforms,
            link_map=link_map,
            joint_map=joint_map,
            joint_map_type=joint_map_type,
            joint_offset_map=joint_offset_map,
            link_masses_com=link_masses_com,
            link_inertias=link_inertias,
            gravity=-9.81,
        )

        assert tau.shape == (1,)
        assert np.all(np.isfinite(tau))
        # Torque should account for gravity on BOTH links
        # Total gravity torque about joint 0 (at origin, Z axis):
        # link 0 CoM at (0.5, 0, 0): torque = 1.0 * 9.81 * 0.5
        # link 1 CoM at (1.0 + 0.5, 0, 0) = (1.5, 0, 0): torque = 1.0 * 9.81 * 1.5
        expected = -(1.0 * 9.81 * 0.5 + 1.0 * 9.81 * 1.5)
        np.testing.assert_allclose(tau[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Cross-validation: RNEA NumPy vs Pinocchio
# ---------------------------------------------------------------------------

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except Exception:
    PINOCCHIO_AVAILABLE = False


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
class TestRNEAvsPinocchio:
    """Cross-validate NumPy RNEA against Pinocchio inverse dynamics (gold standard)."""

    @pytest.fixture(scope="class")
    def franka_setup(self):
        """Load Franka via curobo and build a matching Pinocchio model."""
        import os

        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.types.device_cfg import DeviceCfg

        device_cfg = DeviceCfg()
        cfg = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
        kp = cfg.kinematics_config

        urdf_path = "/tmp/test_rnea_franka.urdf"
        kp.export_to_urdf(
            robot_name="dynamics",
            output_path=urdf_path,
            include_spheres=False,
        )

        pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path)
        pin_data = pin_model.createData()

        if os.path.exists(urdf_path):
            os.remove(urdf_path)

        return kp, pin_model, pin_data

    def _compare(self, franka_setup, q_np, qd_np, qdd_np, atol=1e-4):
        """Run both RNEA and Pinocchio, compare torques."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_from_kinematics_params,
        )

        kp, pin_model, pin_data = franka_setup
        num_dof = pin_model.nq

        tau_rnea, _, _, _ = rnea_from_kinematics_params(
            q_np, qd_np, qdd_np, kp, gravity=-9.81,
        )

        tau_pin = pin.rnea(pin_model, pin_data, q_np[:num_dof], qd_np[:num_dof], qdd_np[:num_dof])
        tau_pin = np.array(tau_pin).flatten()[:num_dof]

        np.testing.assert_allclose(tau_rnea, tau_pin, atol=atol)

    def test_gravity_compensation_zero_config(self, franka_setup):
        """Gravity compensation at zero configuration."""
        num_dof = franka_setup[0].num_dof
        q = np.zeros(num_dof)
        qd = np.zeros(num_dof)
        qdd = np.zeros(num_dof)
        self._compare(franka_setup, q, qd, qdd)

    def test_gravity_compensation_nonzero_config(self, franka_setup):
        """Gravity compensation at a non-trivial configuration."""
        q = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
        qd = np.zeros(7)
        qdd = np.zeros(7)
        self._compare(franka_setup, q, qd, qdd)

    def test_with_velocity(self, franka_setup):
        """Coriolis + gravity at nonzero velocity."""
        q = np.array([0.1, -0.3, 0.2, -1.0, 0.05, 0.8, -0.1])
        qd = np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.3, -0.2])
        qdd = np.zeros(7)
        self._compare(franka_setup, q, qd, qdd)

    def test_with_acceleration(self, franka_setup):
        """Full ID: gravity + Coriolis + inertia."""
        q = np.array([0.1, -0.3, 0.2, -1.0, 0.05, 0.8, -0.1])
        qd = np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.3, -0.2])
        qdd = np.array([1.0, -0.5, 0.3, 0.2, -0.8, 0.6, -0.4])
        self._compare(franka_setup, q, qd, qdd)

    def test_random_configurations(self, franka_setup):
        """Test several random configurations."""
        np.random.seed(42)
        for _ in range(5):
            q = np.random.uniform(-2.0, 2.0, 7)
            qd = np.random.uniform(-1.0, 1.0, 7)
            qdd = np.random.uniform(-1.0, 1.0, 7)
            self._compare(franka_setup, q, qd, qdd)


# ---------------------------------------------------------------------------
# RNEA Backward (VJP) Tests
# ---------------------------------------------------------------------------


class TestRNEABackwardFiniteDiff:
    """Validate RNEA backward against finite differences."""

    def _make_pendulum(self, length=1.0, mass=2.0):
        """Single Y_ROT pendulum."""
        return dict(
            fixed_transforms=np.array([np.eye(3, 4)]),
            link_map=np.array([-1]),
            joint_map=np.array([0]),
            joint_map_type=np.array([Y_ROT]),
            joint_offset_map=np.array([[1.0, 0.0]]),
            link_masses_com=np.array([[length / 2, 0, 0, mass]]),
            link_inertias=np.array(
                [[1e-6, mass * length**2 / 12.0, 1e-6, 0, 0, 0]]
            ),
        )

    def _finite_diff_check(self, data, q, qd, qdd, tau_bar, eps=1e-6):
        """Check grad_q, grad_qd, grad_qdd via finite differences."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import rnea, rnea_backward

        # Analytical gradients
        tau, v, a, f = rnea(q, qd, qdd, gravity=-9.81, **data)
        gq, gqd, gqdd = rnea_backward(
            tau_bar, q, qd, qdd, v, a, f, gravity=-9.81, **data
        )

        # Finite-diff grad_q
        fd_gq = np.zeros_like(q)
        for i in range(len(q)):
            q_p, q_m = q.copy(), q.copy()
            q_p[i] += eps
            q_m[i] -= eps
            tau_p, _, _, _ = rnea(q_p, qd, qdd, gravity=-9.81, **data)
            tau_m, _, _, _ = rnea(q_m, qd, qdd, gravity=-9.81, **data)
            fd_gq[i] = tau_bar @ (tau_p - tau_m) / (2 * eps)

        # Finite-diff grad_qd
        fd_gqd = np.zeros_like(qd)
        for i in range(len(qd)):
            qd_p, qd_m = qd.copy(), qd.copy()
            qd_p[i] += eps
            qd_m[i] -= eps
            tau_p, _, _, _ = rnea(q, qd_p, qdd, gravity=-9.81, **data)
            tau_m, _, _, _ = rnea(q, qd_m, qdd, gravity=-9.81, **data)
            fd_gqd[i] = tau_bar @ (tau_p - tau_m) / (2 * eps)

        # Finite-diff grad_qdd
        fd_gqdd = np.zeros_like(qdd)
        for i in range(len(qdd)):
            qdd_p, qdd_m = qdd.copy(), qdd.copy()
            qdd_p[i] += eps
            qdd_m[i] -= eps
            tau_p, _, _, _ = rnea(q, qd, qdd_p, gravity=-9.81, **data)
            tau_m, _, _, _ = rnea(q, qd, qdd_m, gravity=-9.81, **data)
            fd_gqdd[i] = tau_bar @ (tau_p - tau_m) / (2 * eps)

        np.testing.assert_allclose(gq, fd_gq, atol=1e-5, err_msg="grad_q mismatch")
        np.testing.assert_allclose(gqd, fd_gqd, atol=1e-5, err_msg="grad_qd mismatch")
        np.testing.assert_allclose(
            gqdd, fd_gqdd, atol=1e-5, err_msg="grad_qdd mismatch"
        )

    def test_pendulum_gravity(self):
        """Pendulum at various angles, tau_bar=1."""
        data = self._make_pendulum()
        for angle in [0.0, 0.3, -0.5, np.pi / 4]:
            q = np.array([angle])
            qd = np.zeros(1)
            qdd = np.zeros(1)
            tau_bar = np.array([1.0])
            self._finite_diff_check(data, q, qd, qdd, tau_bar)

    def test_pendulum_full_dynamics(self):
        """Pendulum with velocity and acceleration."""
        data = self._make_pendulum()
        q = np.array([0.5])
        qd = np.array([1.2])
        qdd = np.array([-0.8])
        tau_bar = np.array([2.5])
        self._finite_diff_check(data, q, qd, qdd, tau_bar)

    def test_two_link_chain(self):
        """Two-link chain at nonzero config."""
        T1 = np.eye(3, 4)
        T2 = np.eye(3, 4)
        T2[0, 3] = 1.0  # link2 starts at x=1 from link1

        data = dict(
            fixed_transforms=np.array([T1, T2]),
            link_map=np.array([-1, 0]),
            joint_map=np.array([0, 1]),
            joint_map_type=np.array([Y_ROT, Y_ROT]),
            joint_offset_map=np.array([[1.0, 0.0], [1.0, 0.0]]),
            link_masses_com=np.array([[0.5, 0, 0, 2.0], [0.4, 0, 0, 1.5]]),
            link_inertias=np.array(
                [
                    [1e-6, 2.0 / 12.0, 1e-6, 0, 0, 0],
                    [1e-6, 1.5 * 0.64 / 12.0, 1e-6, 0, 0, 0],
                ]
            ),
        )

        q = np.array([0.3, -0.5])
        qd = np.array([0.5, -0.3])
        qdd = np.array([1.0, -0.5])
        tau_bar = np.array([1.0, -0.7])
        self._finite_diff_check(data, q, qd, qdd, tau_bar)

    def test_franka_finite_diff(self):
        """Franka robot finite-diff check."""
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_backward_from_kinematics_params,
            rnea_from_kinematics_params,
        )

        cfg = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=DeviceCfg())
        kp = cfg.kinematics_config

        np.random.seed(123)
        q = np.random.uniform(-1.5, 1.5, 7)
        qd = np.random.uniform(-1.0, 1.0, 7)
        qdd = np.random.uniform(-1.0, 1.0, 7)
        tau_bar = np.random.randn(7)

        tau, v, a, f = rnea_from_kinematics_params(q, qd, qdd, kp, gravity=-9.81)
        gq, gqd, gqdd = rnea_backward_from_kinematics_params(
            tau_bar, q, qd, qdd, v, a, f, kp, gravity=-9.81
        )

        eps = 1e-6
        # Finite-diff grad_q
        fd_gq = np.zeros(7)
        for i in range(7):
            q_p, q_m = q.copy(), q.copy()
            q_p[i] += eps
            q_m[i] -= eps
            tp, _, _, _ = rnea_from_kinematics_params(q_p, qd, qdd, kp, gravity=-9.81)
            tm, _, _, _ = rnea_from_kinematics_params(q_m, qd, qdd, kp, gravity=-9.81)
            fd_gq[i] = tau_bar @ (tp - tm) / (2 * eps)

        fd_gqd = np.zeros(7)
        for i in range(7):
            qd_p, qd_m = qd.copy(), qd.copy()
            qd_p[i] += eps
            qd_m[i] -= eps
            tp, _, _, _ = rnea_from_kinematics_params(q, qd_p, qdd, kp, gravity=-9.81)
            tm, _, _, _ = rnea_from_kinematics_params(q, qd_m, qdd, kp, gravity=-9.81)
            fd_gqd[i] = tau_bar @ (tp - tm) / (2 * eps)

        fd_gqdd = np.zeros(7)
        for i in range(7):
            qdd_p, qdd_m = qdd.copy(), qdd.copy()
            qdd_p[i] += eps
            qdd_m[i] -= eps
            tp, _, _, _ = rnea_from_kinematics_params(q, qd, qdd_p, kp, gravity=-9.81)
            tm, _, _, _ = rnea_from_kinematics_params(q, qd, qdd_m, kp, gravity=-9.81)
            fd_gqdd[i] = tau_bar @ (tp - tm) / (2 * eps)

        np.testing.assert_allclose(gq, fd_gq, atol=1e-4, err_msg="Franka grad_q")
        np.testing.assert_allclose(gqd, fd_gqd, atol=1e-4, err_msg="Franka grad_qd")
        np.testing.assert_allclose(
            gqdd, fd_gqdd, atol=1e-4, err_msg="Franka grad_qdd"
        )


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
class TestRNEABackwardVsPinocchio:
    """Compare RNEA backward against Pinocchio's analytical Jacobians."""

    @pytest.fixture(scope="class")
    def franka_setup(self):
        """Load Franka and build Pinocchio model."""
        import os

        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.types.device_cfg import DeviceCfg

        device_cfg = DeviceCfg()
        cfg = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
        kp = cfg.kinematics_config

        urdf_path = "/tmp/test_rnea_backward_franka.urdf"
        kp.export_to_urdf(
            robot_name="dynamics", output_path=urdf_path, include_spheres=False
        )
        pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path)
        pin_data = pin_model.createData()
        if os.path.exists(urdf_path):
            os.remove(urdf_path)

        return kp, pin_model, pin_data

    def _compare(self, franka_setup, q_np, qd_np, qdd_np, atol=1e-4):
        """Compare VJP against Pinocchio Jacobian * tau_bar."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_backward_from_kinematics_params,
            rnea_from_kinematics_params,
        )

        kp, pin_model, pin_data = franka_setup
        num_dof = pin_model.nq

        np.random.seed(99)
        tau_bar = np.random.randn(num_dof)

        # Our VJP
        tau, v, a, f = rnea_from_kinematics_params(q_np, qd_np, qdd_np, kp)
        gq, gqd, gqdd = rnea_backward_from_kinematics_params(
            tau_bar, q_np, qd_np, qdd_np, v, a, f, kp
        )

        # Pinocchio Jacobians
        pin.computeRNEADerivatives(
            pin_model, pin_data,
            q_np[:num_dof], qd_np[:num_dof], qdd_np[:num_dof],
        )
        dtau_dq = np.array(pin_data.dtau_dq)
        dtau_dv = np.array(pin_data.dtau_dv)
        # dtau/dqdd = M (mass matrix)
        M = np.array(pin_data.M)

        # VJP = tau_bar^T * Jacobian
        pin_gq = tau_bar @ dtau_dq
        pin_gqd = tau_bar @ dtau_dv
        pin_gqdd = tau_bar @ M

        np.testing.assert_allclose(gq, pin_gq, atol=atol, err_msg="grad_q vs Pin")
        np.testing.assert_allclose(gqd, pin_gqd, atol=atol, err_msg="grad_qd vs Pin")
        np.testing.assert_allclose(
            gqdd, pin_gqdd, atol=atol, err_msg="grad_qdd vs Pin"
        )

    def test_gravity_only(self, franka_setup):
        """Gradient at zero velocity/acceleration."""
        q = np.array([0.1, -0.3, 0.2, -1.0, 0.05, 0.8, -0.1])
        self._compare(franka_setup, q, np.zeros(7), np.zeros(7))

    def test_full_dynamics(self, franka_setup):
        """Full dynamics gradient."""
        q = np.array([0.1, -0.3, 0.2, -1.0, 0.05, 0.8, -0.1])
        qd = np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.3, -0.2])
        qdd = np.array([1.0, -0.5, 0.3, 0.2, -0.8, 0.6, -0.4])
        self._compare(franka_setup, q, qd, qdd)

    def test_random_configurations(self, franka_setup):
        """Several random configs."""
        np.random.seed(42)
        for _ in range(5):
            q = np.random.uniform(-2.0, 2.0, 7)
            qd = np.random.uniform(-1.0, 1.0, 7)
            qdd = np.random.uniform(-1.0, 1.0, 7)
            self._compare(franka_setup, q, qd, qdd)


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
class TestMultiRobotVsPinocchio:
    """Forward and backward RNEA vs Pinocchio across multiple robot configs."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
        "dual_ur10e.yml",
        "unitree_g1.yml",
    ]

    @staticmethod
    def _load_robot(robot_yml):
        """Load curobo robot and build Pinocchio model.

        Returns a permutation map to handle differing joint orders between
        curobo and Pinocchio (happens for branching trees like unitree).
        """
        import os

        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.types.device_cfg import DeviceCfg

        device_cfg = DeviceCfg()
        cfg = KinematicsCfg.from_robot_yaml_file(robot_yml, device_cfg=device_cfg)
        kp = cfg.kinematics_config

        urdf_path = f"/tmp/test_rnea_{robot_yml.replace('.yml','')}.urdf"
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

        return kp, pin_model, pin_data, perm

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_forward_gravity(self, robot_yml):
        """Gravity torques match Pinocchio for each robot."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_from_kinematics_params,
        )

        kp, pin_model, pin_data, perm = self._load_robot(robot_yml)
        num_dof = len(perm)

        np.random.seed(7)
        for _ in range(3):
            q_curobo = np.random.uniform(-1.5, 1.5, num_dof)
            q_pin = q_curobo[np.argsort(perm)]  # reorder for Pinocchio
            qd = np.zeros(num_dof)

            tau_rnea, _, _, _ = rnea_from_kinematics_params(
                q_curobo, qd, qd, kp,
            )
            tau_pin_raw = np.array(
                pin.rnea(pin_model, pin_data, q_pin, qd, qd)
            ).flatten()[:num_dof]
            # Reorder Pinocchio result to curobo order
            tau_pin = tau_pin_raw[perm]

            np.testing.assert_allclose(
                tau_rnea, tau_pin, atol=1e-4,
                err_msg=f"{robot_yml} forward gravity mismatch",
            )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_forward_full_dynamics(self, robot_yml):
        """Full ID matches Pinocchio for each robot."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_from_kinematics_params,
        )

        kp, pin_model, pin_data, perm = self._load_robot(robot_yml)
        num_dof = len(perm)
        inv_perm = np.argsort(perm)  # curobo->pin reorder

        np.random.seed(11)
        for _ in range(3):
            q_c = np.random.uniform(-1.5, 1.5, num_dof)
            qd_c = np.random.uniform(-1.0, 1.0, num_dof)
            qdd_c = np.random.uniform(-1.0, 1.0, num_dof)

            tau_rnea, _, _, _ = rnea_from_kinematics_params(q_c, qd_c, qdd_c, kp)
            tau_pin_raw = np.array(
                pin.rnea(
                    pin_model, pin_data,
                    q_c[inv_perm], qd_c[inv_perm], qdd_c[inv_perm],
                )
            ).flatten()[:num_dof]
            tau_pin = tau_pin_raw[perm]

            np.testing.assert_allclose(
                tau_rnea, tau_pin, atol=1e-4,
                err_msg=f"{robot_yml} forward full dynamics mismatch",
            )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_backward_vs_pinocchio(self, robot_yml):
        """VJP matches Pinocchio Jacobians for each robot."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
            rnea_backward_from_kinematics_params,
            rnea_from_kinematics_params,
        )

        kp, pin_model, pin_data, perm = self._load_robot(robot_yml)
        num_dof = len(perm)
        inv_perm = np.argsort(perm)

        np.random.seed(13)
        for _ in range(3):
            q_c = np.random.uniform(-1.5, 1.5, num_dof)
            qd_c = np.random.uniform(-1.0, 1.0, num_dof)
            qdd_c = np.random.uniform(-1.0, 1.0, num_dof)
            tau_bar_c = np.random.randn(num_dof)

            tau, v, a, f = rnea_from_kinematics_params(q_c, qd_c, qdd_c, kp)
            gq, gqd, gqdd = rnea_backward_from_kinematics_params(
                tau_bar_c, q_c, qd_c, qdd_c, v, a, f, kp,
            )

            # Pinocchio in its own joint order
            q_p = q_c[inv_perm]
            qd_p = qd_c[inv_perm]
            qdd_p = qdd_c[inv_perm]
            tau_bar_p = tau_bar_c[inv_perm]

            pin.computeRNEADerivatives(pin_model, pin_data, q_p, qd_p, qdd_p)
            pin_gq_p = tau_bar_p @ np.array(pin_data.dtau_dq)
            pin_gqd_p = tau_bar_p @ np.array(pin_data.dtau_dv)
            pin_gqdd_p = tau_bar_p @ np.array(pin_data.M)

            # Reorder Pinocchio gradients back to curobo order
            pin_gq = pin_gq_p[perm]
            pin_gqd = pin_gqd_p[perm]
            pin_gqdd = pin_gqdd_p[perm]

            np.testing.assert_allclose(
                gq, pin_gq, atol=1e-4,
                err_msg=f"{robot_yml} grad_q mismatch",
            )
            np.testing.assert_allclose(
                gqd, pin_gqd, atol=1e-4,
                err_msg=f"{robot_yml} grad_qd mismatch",
            )
            np.testing.assert_allclose(
                gqdd, pin_gqdd, atol=1e-4,
                err_msg=f"{robot_yml} grad_qdd mismatch",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
