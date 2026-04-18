# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Rollout classes that compute costs and constraints from a given action sequence.

Rollouts are defined structurally via the
:class:`curobo._src.rollout.rollout_protocol.Rollout` :class:`~typing.Protocol` -- there is no
base class to inherit from. :class:`curobo._src.rollout.rollout_robot.RobotRollout` is the
concrete implementation used for manipulators across cuRobo; it composes a transition model,
a :class:`curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager`, and a scene
collision checker.

A rollout is consumed by any class that implements the
:class:`curobo._src.optim.optimizer_protocol.Optimizer` Protocol to minimise costs while
satisfying constraints.

See :ref:`rollout_class_note` for a walkthrough of the Rollout Protocol and how to implement a
custom rollout.
"""

