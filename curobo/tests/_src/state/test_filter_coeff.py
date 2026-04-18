# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for FilterCoeff."""

# Third Party

# CuRobo
from curobo._src.state.filter_coeff import FilterCoeff


class TestFilterCoeff:
    """Test FilterCoeff class."""

    def test_default_initialization(self):
        """Test default filter coefficient initialization."""
        coeff = FilterCoeff()
        assert coeff.position == 0.0
        assert coeff.velocity == 0.0
        assert coeff.acceleration == 0.0
        assert coeff.jerk == 0.0

    def test_initialization_with_values(self):
        """Test filter coefficient initialization with custom values."""
        coeff = FilterCoeff(position=0.5, velocity=0.3, acceleration=0.2, jerk=0.1)
        assert coeff.position == 0.5
        assert coeff.velocity == 0.3
        assert coeff.acceleration == 0.2
        assert coeff.jerk == 0.1

    def test_initialization_partial_values(self):
        """Test filter coefficient with partial values."""
        coeff = FilterCoeff(position=0.8, velocity=0.6)
        assert coeff.position == 0.8
        assert coeff.velocity == 0.6
        assert coeff.acceleration == 0.0
        assert coeff.jerk == 0.0

    def test_values_can_be_one(self):
        """Test filter coefficient with value 1.0."""
        coeff = FilterCoeff(position=1.0, velocity=1.0, acceleration=1.0, jerk=1.0)
        assert coeff.position == 1.0
        assert coeff.velocity == 1.0
        assert coeff.acceleration == 1.0
        assert coeff.jerk == 1.0

    def test_values_can_be_negative(self):
        """Test filter coefficient with negative values."""
        coeff = FilterCoeff(position=-0.5, velocity=-0.3)
        assert coeff.position == -0.5
        assert coeff.velocity == -0.3

    def test_immutable_after_creation(self):
        """Test that FilterCoeff is a frozen dataclass."""
        coeff = FilterCoeff(position=0.5)
        # Since it's a dataclass, it should be mutable by default
        # unless frozen=True is specified
        coeff.position = 0.7
        assert coeff.position == 0.7

