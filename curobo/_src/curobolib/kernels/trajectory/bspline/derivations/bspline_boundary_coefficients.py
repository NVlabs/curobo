#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive correct fixed knot coefficients for B-spline boundary constraints.

This script computes the coefficient matrices needed to enforce boundary conditions
(position, velocity, acceleration, jerk) at the start of a B-spline trajectory.
"""

from typing import Dict, Tuple

import numpy as np


def compute_cubic_bspline_basis(t: float) -> np.ndarray:
    """Compute cubic B-spline basis functions at parameter t.

    Args:
        t: Parameter value in [0, 1]

    Returns:
        Array of 4 basis function values
    """
    # Coefficient matrix for cubic B-spline (from the CUDA code)
    coeffs = np.array(
        [
            [-1 / 6, 3 / 6, -3 / 6, 1 / 6],
            [3 / 6, -6 / 6, 0, 4 / 6],
            [-3 / 6, 3 / 6, 3 / 6, 1 / 6],
            [1 / 6, 0, 0, 0],
        ]
    )

    # Powers of t: [t³, t², t, 1]
    t_powers = np.array([t**3, t**2, t, 1.0])

    # Compute basis functions
    basis = coeffs @ t_powers
    return basis


def compute_cubic_bspline_derivatives(t: float, dt: float) -> Dict[str, np.ndarray]:
    """Compute cubic B-spline basis functions and their derivatives at parameter t.

    Args:
        t: Parameter value in [0, 1]
        dt: Time step (knot spacing)

    Returns:
        Dictionary with 'position', 'velocity', 'acceleration', 'jerk' basis values
    """
    # Coefficient matrix for cubic B-spline
    coeffs = np.array(
        [
            [-1 / 6, 3 / 6, -3 / 6, 1 / 6],
            [3 / 6, -6 / 6, 0, 4 / 6],
            [-3 / 6, 3 / 6, 3 / 6, 1 / 6],
            [1 / 6, 0, 0, 0],
        ]
    )

    # Position basis: coeffs @ [t³, t², t, 1]
    t_powers_pos = np.array([t**3, t**2, t, 1.0])
    basis_pos = coeffs @ t_powers_pos

    # Velocity basis: d/dt of position basis = coeffs @ [3t², 2t, 1, 0] / dt
    t_powers_vel = np.array([3 * t**2, 2 * t, 1.0, 0.0])
    basis_vel = (coeffs @ t_powers_vel) / dt

    # Acceleration basis: d²/dt² of position basis = coeffs @ [6t, 2, 0, 0] / dt²
    t_powers_acc = np.array([6 * t, 2.0, 0.0, 0.0])
    basis_acc = (coeffs @ t_powers_acc) / (dt**2)

    # Jerk basis: d³/dt³ of position basis = coeffs @ [6, 0, 0, 0] / dt³
    t_powers_jerk = np.array([6.0, 0.0, 0.0, 0.0])
    basis_jerk = (coeffs @ t_powers_jerk) / (dt**3)

    return {
        "position": basis_pos,
        "velocity": basis_vel,
        "acceleration": basis_acc,
        "jerk": basis_jerk,
    }


def derive_fixed_knot_coefficients_degree3() -> Tuple[np.ndarray, np.ndarray]:
    """Derive the coefficient matrix for computing fixed knots from boundary conditions
    for degree 3 (cubic) B-splines.

    Returns:
        coeff_matrix: 3x4 matrix where columns are [pos, vel, acc, jerk] coefficients
        verification: The original basis matrix for verification
    """
    # We need to solve: M * knots = boundary_conditions
    # Where M is the matrix of basis functions and derivatives at t=0
    # and boundary_conditions = [pos, vel*dt, acc*dt², jerk*dt³]

    # Compute basis functions and derivatives at t=0
    # Note: We use dt=1 here and will scale in the actual computation
    t = 1.0
    dt = t
    # Coefficient matrix for cubic B-spline
    coeffs = np.array(
        [
            [-1 / 6, 3 / 6, -3 / 6, 1 / 6],
            [3 / 6, -6 / 6, 0, 4 / 6],
            [-3 / 6, 3 / 6, 3 / 6, 1 / 6],
            [1 / 6, 0, 0, 0],
        ]
    )

    # Position basis: coeffs @ [t³, t², t, 1]
    t_powers_pos = np.array([t**3, t**2, t, 1.0])
    basis_pos = coeffs @ t_powers_pos

    # Velocity basis: d/dt of position basis = coeffs @ [3t², 2t, 1, 0] / dt
    t_powers_vel = np.array([3 * t**2, 2 * t, 1.0, 0.0])
    basis_vel = (coeffs @ t_powers_vel) / dt

    # Acceleration basis: d²/dt² of position basis = coeffs @ [6t, 2, 0, 0] / dt²
    t_powers_acc = np.array([6 * t, 2.0, 0.0, 0.0])
    basis_acc = (coeffs @ t_powers_acc) / (dt**2)

    # Jerk basis: d³/dt³ of position basis = coeffs @ [6, 0, 0, 0] / dt³
    t_powers_jerk = np.array([6.0, 0.0, 0.0, 0.0])
    basis_jerk = (coeffs @ t_powers_jerk) / (dt**3)

    # For cubic B-spline at t=0, B₃(0) = 0, so we only use first 3 basis functions
    # We need 3 knots to satisfy 4 boundary conditions (overdetermined)
    # Actually, for cubic splines, we can only satisfy pos, vel, acc exactly
    # Jerk will be approximated

    # Build the system matrix using first 3 basis functions
    # We'll use position, velocity, and acceleration constraints
    M = np.zeros((4, 4))
    M[0, :] = basis_pos  # Position constraint
    M[1, :] = basis_vel  # Velocity constraint (already divided by dt)
    M[2, :] = basis_acc  # Acceleration constraint (already divided by dt²)
    M[3, :] = basis_jerk  # Jerk constraint (already divided by dt³)

    print("=== Degree 3 (Cubic) B-spline Boundary Coefficients ===\n")
    print("Basis function values at t=0:")
    print(f"  Position basis:     {basis_pos}")
    print(f"  Velocity basis:     {basis_vel}")
    print(f"  Acceleration basis: {basis_acc}")
    print(f"  Jerk basis:         {basis_jerk}")

    print("\nSystem matrix M (4x4):")
    print(M)

    # Compute the inverse to get coefficient matrix
    M_inv = np.linalg.inv(M)

    # The coefficient matrix tells us how to compute fixed knots from boundary conditions
    # fixed_knots = M_inv @ [pos, vel*dt, acc*dt²]

    # Extract coefficients for each boundary condition
    pos_coeffs = M_inv[:, 0]
    vel_coeffs = M_inv[:, 1]
    acc_coeffs = M_inv[:, 2]
    jerk_coeffs = M_inv[:, 3]

    # For jerk, we need to find the best approximation
    # We can use the pseudoinverse approach or set a specific pattern
    # For now, let's compute what jerk we get with zero jerk input
    # jerk_at_zero = derivatives['jerk'][:] @ np.zeros(4)

    # Alternative: minimize the jerk difference
    # We want: jerk_basis @ knots ≈ jerk_desired
    # Using least squares: knots = (jerk_basis)⁺ * jerk_desired
    # jerk_coeffs = np.zeros(4)  # For cubic, we can't control jerk independently

    print("\n=== Derived Fixed Knot Coefficients ===")
    print(f"Position coefficients:     {pos_coeffs}")
    print(f"Velocity coefficients:     {vel_coeffs}")
    print(f"Acceleration coefficients: {acc_coeffs}")
    print(f"Jerk coefficients:         {jerk_coeffs} (cannot be controlled independently)")

    # Verification: Check if the coefficients work
    print("\n=== Verification ===")
    test_pos, test_vel, test_acc, test_jerk = 1.0, 0.5, 0.2, 0.1
    fixed_knots = (
        pos_coeffs * test_pos
        + vel_coeffs * test_vel * dt
        + acc_coeffs * test_acc * dt**2
        + jerk_coeffs * test_jerk * dt**3
    )

    # Compute resulting position, velocity, acceleration
    result_pos = basis_pos @ fixed_knots
    result_vel = basis_vel @ fixed_knots * dt
    result_acc = basis_acc @ fixed_knots * dt**2
    result_jerk = basis_jerk @ fixed_knots * dt**3

    print(
        f"Test boundary conditions: pos={test_pos}, vel={test_vel}, acc={test_acc}, jerk={test_jerk}"
    )
    print(f"Fixed knots: {fixed_knots}")
    print(f"Resulting position:     {result_pos:.6f} (expected: {test_pos})")
    print(f"Resulting velocity:     {result_vel:.6f} (expected: {test_vel})")
    print(f"Resulting acceleration: {result_acc:.6f} (expected: {test_acc})")
    print(f"Resulting jerk:         {result_jerk:.6f} (expected: {test_jerk})")
    return np.column_stack([pos_coeffs, vel_coeffs, acc_coeffs, jerk_coeffs]), M


def derive_fixed_knot_coefficients_degree4():
    """Derive the coefficient matrix for degree 4 (quartic) B-splines."""
    print("\n\n" + "=" * 60)
    print("=== Degree 4 (Quartic) B-spline Boundary Coefficients ===\n")

    # Coefficient matrix for quartic B-spline
    coeffs = np.array(
        [
            [1 / 24, -4 / 24, 6 / 24, -4 / 24, 1 / 24],
            [-4 / 24, 12 / 24, -6 / 24, -12 / 24, 11 / 24],
            [6 / 24, -12 / 24, -6 / 24, 12 / 24, 11 / 24],
            [-4 / 24, 4 / 24, 6 / 24, 4 / 24, 1 / 24],
            [1 / 24, 0, 0, 0, 0],
        ]
    )

    dt = 1.0
    t = 0.0

    # Compute basis functions and derivatives at t=0
    t_powers_pos = np.array([t**4, t**3, t**2, t, 1.0])
    basis_pos = coeffs @ t_powers_pos

    t_powers_vel = np.array([4 * t**3, 3 * t**2, 2 * t, 1.0, 0.0])
    basis_vel = (coeffs @ t_powers_vel) / dt

    t_powers_acc = np.array([12 * t**2, 6 * t, 2.0, 0.0, 0.0])
    basis_acc = (coeffs @ t_powers_acc) / (dt**2)

    t_powers_jerk = np.array([24 * t, 6.0, 0.0, 0.0, 0.0])
    basis_jerk = (coeffs @ t_powers_jerk) / (dt**3)

    t_powers_snap = np.array([24.0, 0.0, 0.0, 0.0, 0.0])
    basis_snap = (coeffs @ t_powers_snap) / (dt**4)

    print("Basis function values at t=0:")
    print(f"  Position basis:     {basis_pos}")
    print(f"  Velocity basis:     {basis_vel}")
    print(f"  Acceleration basis: {basis_acc}")
    print(f"  Jerk basis:         {basis_jerk}")

    # Build 4x4 system matrix
    M = np.zeros((5, 5))
    M[0, :] = basis_pos
    M[1, :] = basis_vel
    M[2, :] = basis_acc
    M[3, :] = basis_jerk
    M[4, :] = basis_snap

    print("\nSystem matrix M (4x4):")
    print(M)

    # Compute the inverse
    M_inv = np.linalg.inv(M)

    print("\nInverse matrix M⁻¹:")
    print(M_inv)

    # Extract coefficients
    pos_coeffs = M_inv[:, 0]
    vel_coeffs = M_inv[:, 1]
    acc_coeffs = M_inv[:, 2]
    jerk_coeffs = M_inv[:, 3]

    print("\n=== Derived Fixed Knot Coefficients ===")
    print(f"Position coefficients:     {pos_coeffs}")
    print(f"Velocity coefficients:     {vel_coeffs}")
    print(f"Acceleration coefficients: {acc_coeffs}")
    print(f"Jerk coefficients:         {jerk_coeffs}")


def derive_fixed_knot_coefficients_degree5():
    """Derive the coefficient matrix for degree 5 (quintic) B-splines.
    For quintic, we can control up to the 4th derivative (snap), but we only use up to jerk.
    """
    print("\n\n" + "=" * 60)
    print("=== Degree 5 (Quintic) B-spline Boundary Coefficients ===\n")

    # Coefficient matrix for quintic B-spline
    coeffs = np.array(
        [
            [-1 / 120, 5 / 120, -10 / 120, 10 / 120, -5 / 120, 1 / 120],
            [5 / 120, -20 / 120, 20 / 120, 20 / 120, -50 / 120, 26 / 120],
            [-10 / 120, 30 / 120, 0 / 120, -60 / 120, 0 / 120, 66 / 120],
            [10 / 120, -20 / 120, -20 / 120, 20 / 120, 50 / 120, 26 / 120],
            [-5 / 120, 5 / 120, 10 / 120, 10 / 120, 5 / 120, 1 / 120],
            [1 / 120, 0, 0, 0, 0, 0],
        ]
    )

    dt = 1.0
    t = 0.0

    # Compute basis functions and derivatives at t=0
    t_powers_pos = np.array([t**5, t**4, t**3, t**2, t, 1.0])
    basis_pos = coeffs @ t_powers_pos

    t_powers_vel = np.array([5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1.0, 0.0])
    basis_vel = (coeffs @ t_powers_vel) / dt

    t_powers_acc = np.array([20 * t**3, 12 * t**2, 6 * t, 2.0, 0.0, 0.0])
    basis_acc = (coeffs @ t_powers_acc) / (dt**2)

    t_powers_jerk = np.array([60 * t**2, 24 * t, 6.0, 0.0, 0.0, 0.0])
    basis_jerk = (coeffs @ t_powers_jerk) / (dt**3)

    t_powers_snap = np.array([120 * t, 24.0, 0.0, 0.0, 0.0, 0.0])
    basis_snap = (coeffs @ t_powers_snap) / (dt**4)

    t_powers_crackle = np.array([120.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    basis_crackle = (coeffs @ t_powers_crackle) / (dt**5)

    print("Basis function values at t=0:")
    print(f"  Position basis:     {basis_pos}")
    print(f"  Velocity basis:     {basis_vel}")
    print(f"  Acceleration basis: {basis_acc}")
    print(f"  Jerk basis:         {basis_jerk}")
    print(f"  Snap basis:         {basis_snap}")
    # For quintic, we have 5 knots but only 4 constraints
    # We'll use the first 4 knots and use least squares
    # Or we can add a snap (4th derivative) constraint

    # Approach 1: Use only first 4 knots (underdetermined - use pseudoinverse)
    M = np.zeros((6, 6))
    M[0, :] = basis_pos
    M[1, :] = basis_vel
    M[2, :] = basis_acc
    M[3, :] = basis_jerk
    M[4, :] = basis_snap
    M[5, :] = basis_crackle

    print("\nSystem matrix M (6x6):")
    print(M)

    # Use pseudoinverse for minimum norm solution
    M_pinv = np.linalg.inv(M)

    # Extract coefficients
    pos_coeffs = M_pinv[:, 0]
    vel_coeffs = M_pinv[:, 1]
    acc_coeffs = M_pinv[:, 2]
    jerk_coeffs = M_pinv[:, 3]
    snap_coeffs = M_pinv[:, 4]

    print("\n=== Derived Fixed Knot Coefficients (5 knots) ===")
    print(f"Position coefficients:     {pos_coeffs}")
    print(f"Velocity coefficients:     {vel_coeffs}")
    print(f"Acceleration coefficients: {acc_coeffs}")
    print(f"Jerk coefficients:         {jerk_coeffs}")


if __name__ == "__main__":
    # Derive coefficients for degree 3
    coeff_matrix, basis_matrix = derive_fixed_knot_coefficients_degree3()

    # Show alternative approach discussion
    # derive_fixed_knot_coefficients_degree3_with_jerk()

    # Derive coefficients for degree 4 and 5
    derive_fixed_knot_coefficients_degree4()
    derive_fixed_knot_coefficients_degree5()

    print("\n" + "=" * 60)
