# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for cost type enums."""

# Third Party
import pytest

# CuRobo
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.cost_pose_type import PoseErrorType


class TestCSpaceCostType:
    """Test CSpaceCostType enum."""

    def test_position_value(self):
        """Test POSITION enum value."""
        assert CSpaceCostType.POSITION.value == 0

    def test_state_value(self):
        """Test STATE enum value."""
        assert CSpaceCostType.STATE.value == 1

    def test_enum_members(self):
        """Test that all enum members exist."""
        members = list(CSpaceCostType)
        assert len(members) == 2
        assert CSpaceCostType.POSITION in members
        assert CSpaceCostType.STATE in members

    def test_enum_name_access(self):
        """Test accessing enum by name."""
        assert CSpaceCostType["POSITION"] == CSpaceCostType.POSITION
        assert CSpaceCostType["STATE"] == CSpaceCostType.STATE

    def test_enum_value_to_member(self):
        """Test getting enum member from value."""
        assert CSpaceCostType(0) == CSpaceCostType.POSITION
        assert CSpaceCostType(1) == CSpaceCostType.STATE

    def test_invalid_value_raises_error(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            CSpaceCostType(99)

    def test_invalid_name_raises_error(self):
        """Test that invalid name raises KeyError."""
        with pytest.raises(KeyError):
            CSpaceCostType["INVALID"]

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert CSpaceCostType.POSITION != CSpaceCostType.STATE
        assert CSpaceCostType.POSITION == CSpaceCostType.POSITION

    def test_enum_identity(self):
        """Test enum identity."""
        a = CSpaceCostType.POSITION
        b = CSpaceCostType.POSITION
        assert a is b


class TestPoseErrorType:
    """Test PoseErrorType enum."""

    def test_single_goal_value(self):
        """Test SINGLE_GOAL enum value."""
        assert PoseErrorType.SINGLE_GOAL.value == 0

    def test_batch_goal_value(self):
        """Test BATCH_GOAL enum value."""
        assert PoseErrorType.BATCH_GOAL.value == 1

    def test_goalset_value(self):
        """Test GOALSET enum value."""
        assert PoseErrorType.GOALSET.value == 2

    def test_batch_goalset_value(self):
        """Test BATCH_GOALSET enum value."""
        assert PoseErrorType.BATCH_GOALSET.value == 3

    def test_enum_members(self):
        """Test that all enum members exist."""
        members = list(PoseErrorType)
        assert len(members) == 4
        assert PoseErrorType.SINGLE_GOAL in members
        assert PoseErrorType.BATCH_GOAL in members
        assert PoseErrorType.GOALSET in members
        assert PoseErrorType.BATCH_GOALSET in members

    def test_enum_name_access(self):
        """Test accessing enum by name."""
        assert PoseErrorType["SINGLE_GOAL"] == PoseErrorType.SINGLE_GOAL
        assert PoseErrorType["BATCH_GOAL"] == PoseErrorType.BATCH_GOAL
        assert PoseErrorType["GOALSET"] == PoseErrorType.GOALSET
        assert PoseErrorType["BATCH_GOALSET"] == PoseErrorType.BATCH_GOALSET

    def test_enum_value_to_member(self):
        """Test getting enum member from value."""
        assert PoseErrorType(0) == PoseErrorType.SINGLE_GOAL
        assert PoseErrorType(1) == PoseErrorType.BATCH_GOAL
        assert PoseErrorType(2) == PoseErrorType.GOALSET
        assert PoseErrorType(3) == PoseErrorType.BATCH_GOALSET

    def test_invalid_value_raises_error(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            PoseErrorType(99)

    def test_invalid_name_raises_error(self):
        """Test that invalid name raises KeyError."""
        with pytest.raises(KeyError):
            PoseErrorType["INVALID"]

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert PoseErrorType.SINGLE_GOAL != PoseErrorType.BATCH_GOAL
        assert PoseErrorType.GOALSET != PoseErrorType.BATCH_GOALSET
        assert PoseErrorType.SINGLE_GOAL == PoseErrorType.SINGLE_GOAL

    def test_enum_identity(self):
        """Test enum identity."""
        a = PoseErrorType.GOALSET
        b = PoseErrorType.GOALSET
        assert a is b

    def test_enum_iteration_order(self):
        """Test that enum iteration maintains order."""
        members = list(PoseErrorType)
        assert members[0] == PoseErrorType.SINGLE_GOAL
        assert members[1] == PoseErrorType.BATCH_GOAL
        assert members[2] == PoseErrorType.GOALSET
        assert members[3] == PoseErrorType.BATCH_GOALSET
