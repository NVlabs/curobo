# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan a joint-space trajectory with nonzero start and goal velocities.

The default B-spline trajectory optimizer treats the derivatives supplied in
``current_state`` and ``goal_state`` as trajectory boundary conditions. Run
this example with:

.. code-block:: bash

   python -m curobo.examples.getting_started.nonzero_boundary_velocity
"""

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


def main():
    """Plan and validate a trajectory with nonzero velocity boundaries."""
    planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml"))
    tensor_args = planner.device_cfg.as_torch_dict()

    q0 = torch.tensor(
        [[0.0, -0.4, 0.2, -1.7, 0.0, 1.4, 0.7]],
        **tensor_args,
    )
    dq0 = torch.tensor(
        [[0.10, 0.00, -0.05, 0.00, 0.00, 0.00, 0.00]],
        **tensor_args,
    )
    q1 = torch.tensor(
        [[0.4, -0.2, 0.4, -1.4, 0.1, 1.6, 0.5]],
        **tensor_args,
    )
    dq1 = torch.tensor(
        [[0.00, 0.12, 0.00, -0.08, 0.00, 0.00, 0.00]],
        **tensor_args,
    )

    start_state = JointState(
        position=q0,
        velocity=dq0,
        acceleration=torch.zeros_like(q0),
        jerk=torch.zeros_like(q0),
        joint_names=planner.joint_names,
    )
    goal_state = JointState(
        position=q1,
        velocity=dq1,
        acceleration=torch.zeros_like(q1),
        jerk=torch.zeros_like(q1),
        joint_names=planner.joint_names,
    )

    result = planner.plan_cspace(
        current_state=start_state,
        goal_state=goal_state,
        max_attempts=5,
    )
    if result is None or not bool(result.success.any()):
        raise RuntimeError("No dynamically feasible collision-free trajectory was found")

    trajectory = result.get_interpolated_plan()
    dof = q0.shape[-1]
    positions = trajectory.position.reshape(-1, dof)
    velocities = trajectory.velocity.reshape(-1, dof)

    torch.testing.assert_close(positions[0], q0[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(velocities[0], dq0[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(positions[-1], q1[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(velocities[-1], dq1[0], atol=1e-4, rtol=1e-4)

    print("Planning succeeded")
    print(f"Trajectory waypoints: {positions.shape[0]}")
    print(f"Start velocity: {velocities[0].tolist()}")
    print(f"Goal velocity:  {velocities[-1].tolist()}")


if __name__ == "__main__":
    main()
