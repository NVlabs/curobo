# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Compute forward kinematics on the GPU with automatic differentiation.

Forward kinematics (FK) maps a vector of joint angles to the 6-DOF pose of
every link in the kinematic chain. cuRobo evaluates FK entirely on the GPU
using a parallel product-of-exponentials formulation, and every operation is
differentiable through PyTorch autograd. This means FK can serve as a
building block inside larger optimization loops (IK, motion planning, MPC)
with gradients flowing back to the joint angles at zero extra cost.

By the end of this tutorial you will have:

- Loaded a robot model from a cuRobo YAML configuration
- Computed the end-effector pose for a single joint configuration
- Evaluated FK for 1 000 configurations in a single batched call
- Back-propagated a position loss through FK to obtain joint-angle gradients

Step 1: Run the tutorial
--------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.forward_kinematics

Step 2: Check the output
--------------------------

When the tutorial finishes successfully you will see::

    Robot has 7 degrees of freedom
    Tool frames: ['panda_hand']

    Single FK:
      EE position: tensor(...)
      EE quaternion (wxyz): tensor(...)

    Batched FK (1000 configs): 0.XX ms
      EE positions shape: torch.Size([1000, 1, 3])

    Differentiable FK:
      Gradient w.r.t. joints: tensor(...)

Step 3: Understand the pipeline
---------------------------------

The example demonstrates three capabilities:

1. **Single FK**: Pass one joint configuration and receive the end-effector
   position and orientation (quaternion in wxyz order). The result also
   contains poses for every tool frame defined in the robot config.

2. **Batched FK**: Pass a ``(B, num_dof)`` tensor of joint configurations and
   receive ``(B, num_frames, 3)`` positions and ``(B, num_frames, 4)``
   quaternions. All configurations are evaluated in parallel on the GPU,
   making large-scale reachability analysis fast.

3. **Differentiable FK**: Because FK is implemented as standard PyTorch
   operations, calling ``.backward()`` on any scalar derived from the output
   produces gradients with respect to the input joint angles. This is the
   foundation of cuRobo's optimization-based IK and motion planning.
"""

# Standard Library
import argparse
import sys

# Third Party
import torch

# CuRobo
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


def forward_kinematics_example():
    """Demonstrate forward kinematics with cuRobo.

    Returns:
        True if all checks passed.
    """
    config = KinematicsCfg.from_robot_yaml_file("franka.yml")
    robot = Kinematics(config)

    print(f"Robot has {robot.get_dof()} degrees of freedom")
    print(f"Tool frames: {robot.tool_frames}")

    # Single FK
    q = torch.zeros(1, robot.get_dof(), device="cuda", dtype=torch.float32)
    state = robot.compute_kinematics(
        JointState.from_position(q, joint_names=robot.joint_names)
    )
    ee_pose = state.tool_poses.get_link_pose(robot.tool_frames[0])
    print("\nSingle FK:")
    print(f"  EE position: {ee_pose.position}")
    print(f"  EE quaternion (wxyz): {ee_pose.quaternion}")

    # Batched FK
    batch_size = 1000
    q_batch = torch.rand(batch_size, robot.get_dof(), device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    state_batch = robot.compute_kinematics(
        JointState.from_position(q_batch, joint_names=robot.joint_names)
    )
    end.record()
    torch.cuda.synchronize()
    print(f"\nBatched FK ({batch_size} configs): {start.elapsed_time(end):.2f} ms")
    print(f"  EE positions shape: {state_batch.tool_poses.position.shape}")

    # Differentiable FK
    q = torch.zeros(1, robot.get_dof(), device="cuda", dtype=torch.float32, requires_grad=True)
    state = robot.compute_kinematics(
        JointState.from_position(q, joint_names=robot.joint_names)
    )
    ee_pos = state.tool_poses.get_link_pose(robot.tool_frames[0]).position
    target_pos = torch.tensor([[0.5, 0.0, 0.5]], device="cuda", dtype=torch.float32)
    loss = torch.sum((ee_pos - target_pos) ** 2)
    loss.backward()
    print("\nDifferentiable FK:")
    print(f"  Gradient w.r.t. joints: {q.grad}")
    return True


def test():
    """Run forward kinematics example as a self-test."""
    assert forward_kinematics_example(), "Forward kinematics example failed"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Forward Kinematics with cuRobo")
    parser.add_argument("--test", action="store_true", help="Run as self-test with assertions")
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    forward_kinematics_example()


if __name__ == "__main__":
    main()
