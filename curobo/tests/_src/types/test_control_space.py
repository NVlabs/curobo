# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ControlSpace enum."""

# Third Party

# CuRobo
from curobo._src.types.control_space import ControlSpace


class TestControlSpace:
    """Test ControlSpace enum."""

    def test_enum_values(self):
        """Test that enum values are correctly defined."""
        assert ControlSpace.POSITION.value == 0
        assert ControlSpace.VELOCITY.value == 1
        assert ControlSpace.ACCELERATION.value == 2
        assert ControlSpace.BSPLINE_3.value == 3
        assert ControlSpace.BSPLINE_4.value == 4
        assert ControlSpace.BSPLINE_5.value == 5

    def test_bspline_types(self):
        """Test bspline_types static method."""
        bspline_types = ControlSpace.bspline_types()
        assert len(bspline_types) == 3
        assert ControlSpace.BSPLINE_3 in bspline_types
        assert ControlSpace.BSPLINE_4 in bspline_types
        assert ControlSpace.BSPLINE_5 in bspline_types
        assert ControlSpace.POSITION not in bspline_types
        assert ControlSpace.VELOCITY not in bspline_types

    def test_position_types(self):
        """Test position_types static method."""
        position_types = ControlSpace.position_types()
        assert ControlSpace.POSITION in position_types
        assert ControlSpace.BSPLINE_3 in position_types
        assert ControlSpace.BSPLINE_4 in position_types
        assert ControlSpace.BSPLINE_5 in position_types
        assert ControlSpace.VELOCITY not in position_types
        assert ControlSpace.ACCELERATION not in position_types

    def test_spline_degree_bspline_3(self):
        """Test spline_degree for BSPLINE_3."""
        degree = ControlSpace.spline_degree(ControlSpace.BSPLINE_3)
        assert degree == 3

    def test_spline_degree_bspline_4(self):
        """Test spline_degree for BSPLINE_4."""
        degree = ControlSpace.spline_degree(ControlSpace.BSPLINE_4)
        assert degree == 4

    def test_spline_degree_bspline_5(self):
        """Test spline_degree for BSPLINE_5."""
        degree = ControlSpace.spline_degree(ControlSpace.BSPLINE_5)
        assert degree == 5

    def test_spline_degree_non_bspline(self):
        """Test spline_degree for non-bspline control spaces."""
        assert ControlSpace.spline_degree(ControlSpace.POSITION) == 0
        assert ControlSpace.spline_degree(ControlSpace.VELOCITY) == 0
        assert ControlSpace.spline_degree(ControlSpace.ACCELERATION) == 0

    def test_spline_total_knots_position(self):
        """Test spline_total_knots for POSITION control space."""
        action_knots = 10
        total_knots = ControlSpace.spline_total_knots(ControlSpace.POSITION, action_knots)
        assert total_knots == action_knots

    def test_spline_total_knots_velocity(self):
        """Test spline_total_knots for VELOCITY control space."""
        action_knots = 10
        total_knots = ControlSpace.spline_total_knots(ControlSpace.VELOCITY, action_knots)
        assert total_knots == action_knots

    def test_spline_total_knots_bspline_3(self):
        """Test spline_total_knots for BSPLINE_3."""
        action_knots = 10
        total_knots = ControlSpace.spline_total_knots(ControlSpace.BSPLINE_3, action_knots)
        # For BSPLINE_3: action_knots + degree + 1 = 10 + 3 + 1 = 14
        assert total_knots == 14

    def test_spline_total_knots_bspline_4(self):
        """Test spline_total_knots for BSPLINE_4."""
        action_knots = 10
        total_knots = ControlSpace.spline_total_knots(ControlSpace.BSPLINE_4, action_knots)
        # For BSPLINE_4: action_knots + degree + 1 = 10 + 4 + 1 = 15
        assert total_knots == 15

    def test_spline_total_knots_bspline_5(self):
        """Test spline_total_knots for BSPLINE_5."""
        action_knots = 10
        total_knots = ControlSpace.spline_total_knots(ControlSpace.BSPLINE_5, action_knots)
        # For BSPLINE_5: action_knots + degree + 1 = 10 + 5 + 1 = 16
        assert total_knots == 16

    def test_spline_total_interpolation_steps_position(self):
        """Test spline_total_interpolation_steps for POSITION."""
        action_knots = 10
        interpolation_steps = 5
        total_steps = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.POSITION, action_knots, interpolation_steps
        )
        # For POSITION: total_knots * interpolation_steps + 1 = 10 * 5 + 1 = 51
        assert total_steps == 51

    def test_spline_total_interpolation_steps_bspline_3(self):
        """Test spline_total_interpolation_steps for BSPLINE_3."""
        action_knots = 10
        interpolation_steps = 5
        total_steps = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.BSPLINE_3, action_knots, interpolation_steps
        )
        # For BSPLINE_3: total_knots * interpolation_steps + 1
        # total_knots = 14, so 14 * 5 + 1 = 71
        assert total_steps == 71

    def test_spline_total_interpolation_steps_bspline_4(self):
        """Test spline_total_interpolation_steps for BSPLINE_4."""
        action_knots = 10
        interpolation_steps = 5
        total_steps = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.BSPLINE_4, action_knots, interpolation_steps
        )
        # For BSPLINE_4: total_knots * interpolation_steps + 1
        # total_knots = 15, so 15 * 5 + 1 = 76
        assert total_steps == 76

    def test_spline_total_interpolation_steps_bspline_5(self):
        """Test spline_total_interpolation_steps for BSPLINE_5."""
        action_knots = 10
        interpolation_steps = 5
        total_steps = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.BSPLINE_5, action_knots, interpolation_steps
        )
        # For BSPLINE_5: total_knots * interpolation_steps + 1
        # total_knots = 16, so 16 * 5 + 1 = 81
        assert total_steps == 81

    def test_spline_total_interpolation_steps_velocity(self):
        """Test spline_total_interpolation_steps for VELOCITY."""
        action_knots = 8
        interpolation_steps = 3
        total_steps = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.VELOCITY, action_knots, interpolation_steps
        )
        # For VELOCITY: total_knots * interpolation_steps + 1 = 8 * 3 + 1 = 25
        assert total_steps == 25

    def test_spline_total_knots_with_zero_action_knots(self):
        """Test spline_total_knots with zero action knots."""
        total_knots = ControlSpace.spline_total_knots(ControlSpace.BSPLINE_3, 0)
        assert total_knots == 4  # 0 + 3 + 1

    def test_spline_total_interpolation_steps_with_one_knot(self):
        """Test spline_total_interpolation_steps with single knot."""
        total_steps = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.POSITION, 1, 10
        )
        assert total_steps == 11  # 1 * 10 + 1

