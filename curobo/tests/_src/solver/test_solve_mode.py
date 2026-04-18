# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SolveMode enum and related functions."""

# Third Party
import pytest

# CuRobo
from curobo._src.solver.solve_mode import SolveMode, parse_solve_mode


class TestSolveModeEnum:
    """Test SolveMode enum values and properties."""

    def test_single_mode_value(self):
        """Test SINGLE mode has correct value."""
        assert SolveMode.SINGLE.value == "single"

    def test_batch_mode_value(self):
        """Test BATCH mode has correct value."""
        assert SolveMode.BATCH.value == "batch"

    def test_multi_env_mode_value(self):
        """Test MULTI_ENV mode has correct value."""
        assert SolveMode.MULTI_ENV.value == "multi_env"

    def test_enum_members_count(self):
        """Test that SolveMode has exactly 3 members."""
        assert len(SolveMode) == 3

    def test_enum_members_list(self):
        """Test all enum members are present."""
        members = [m for m in SolveMode]
        assert SolveMode.SINGLE in members
        assert SolveMode.BATCH in members
        assert SolveMode.MULTI_ENV in members

    def test_enum_by_value(self):
        """Test creating enum from value."""
        assert SolveMode("single") == SolveMode.SINGLE
        assert SolveMode("batch") == SolveMode.BATCH
        assert SolveMode("multi_env") == SolveMode.MULTI_ENV

    def test_enum_invalid_value(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            SolveMode("invalid")


class TestParseSolveMode:
    """Test parse_solve_mode function."""

    def test_parse_enum_single(self):
        """Test parsing SolveMode.SINGLE enum."""
        result = parse_solve_mode(SolveMode.SINGLE)
        assert result == SolveMode.SINGLE

    def test_parse_enum_batch(self):
        """Test parsing SolveMode.BATCH enum."""
        result = parse_solve_mode(SolveMode.BATCH)
        assert result == SolveMode.BATCH

    def test_parse_enum_multi_env(self):
        """Test parsing SolveMode.MULTI_ENV enum."""
        result = parse_solve_mode(SolveMode.MULTI_ENV)
        assert result == SolveMode.MULTI_ENV

    def test_parse_string_single(self):
        """Test parsing 'single' string."""
        result = parse_solve_mode("single")
        assert result == SolveMode.SINGLE

    def test_parse_string_batch(self):
        """Test parsing 'batch' string."""
        result = parse_solve_mode("batch")
        assert result == SolveMode.BATCH

    def test_parse_string_multi_env(self):
        """Test parsing 'multi_env' string."""
        result = parse_solve_mode("multi_env")
        assert result == SolveMode.MULTI_ENV

    def test_parse_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            parse_solve_mode("invalid_mode")


class TestSolveModeComparison:
    """Test SolveMode comparison operations."""

    def test_equality_same_mode(self):
        """Test that same modes are equal."""
        assert SolveMode.SINGLE == SolveMode.SINGLE
        assert SolveMode.BATCH == SolveMode.BATCH
        assert SolveMode.MULTI_ENV == SolveMode.MULTI_ENV

    def test_inequality_different_modes(self):
        """Test that different modes are not equal."""
        assert SolveMode.SINGLE != SolveMode.BATCH
        assert SolveMode.BATCH != SolveMode.MULTI_ENV
        assert SolveMode.SINGLE != SolveMode.MULTI_ENV

    def test_identity(self):
        """Test enum identity."""
        mode1 = SolveMode.SINGLE
        mode2 = SolveMode.SINGLE
        assert mode1 is mode2


class TestSolveModeStringRepresentation:
    """Test SolveMode string representations."""

    def test_str_representation(self):
        """Test __str__ representation."""
        assert str(SolveMode.SINGLE) == "SolveMode.SINGLE"
        assert str(SolveMode.BATCH) == "SolveMode.BATCH"
        assert str(SolveMode.MULTI_ENV) == "SolveMode.MULTI_ENV"

    def test_name_property(self):
        """Test name property."""
        assert SolveMode.SINGLE.name == "SINGLE"
        assert SolveMode.BATCH.name == "BATCH"
        assert SolveMode.MULTI_ENV.name == "MULTI_ENV"

