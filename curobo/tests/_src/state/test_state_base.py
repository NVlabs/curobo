# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for State base class."""

# Third Party

# CuRobo
from curobo._src.state.state_base import State
from curobo._src.state.state_joint import JointState


class TestState:
    """Test State base class."""

    def test_state_is_dataclass(self):
        """Test that State is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(State)

    def test_state_is_abstract_sequence(self):
        """Test that State inherits from Sequence."""
        from collections.abc import Sequence as ABCSequence

        # State inherits from Sequence but can't be instantiated directly
        # We test via a concrete implementation (JointState)
        import torch
        position = torch.randn(7)
        joint_state = JointState(position=position)
        assert isinstance(joint_state, ABCSequence)
        assert isinstance(joint_state, State)

