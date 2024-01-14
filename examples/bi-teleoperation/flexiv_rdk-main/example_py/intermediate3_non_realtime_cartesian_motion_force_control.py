#!/usr/bin/env python

"""intermediate3_non_realtime_cartesian_motion_force_control.py

This tutorial runs non-real-time Cartesian-space unified motion-force control to apply force along
Z axis of the chosen reference frame, or to execute a simple polish action along XY plane of the
chosen reference frame.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import math
import argparse
import numpy as np

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk
# fmt: on

# Global constants
# ==================================================================================================
# TCP sine-sweep amplitude [m]
SWING_AMP = 0.1

# TCP sine-sweep frequency [Hz]
SWING_FREQ = 0.3

# Pressing force to apply during the unified motion-force control [N]
PRESSING_FORCE = 5.0


def print_description():
    """
    Print tutorial description.

    """
    print("This tutorial runs non-real-time Cartesian-space unified motion-force control to "
          "apply force along Z axis of the chosen reference frame, or to execute a simple "
          "polish action along XY plane of the chosen reference frame.")
    print()


def main():
    # Program Setup
    # ==============================================================================================
    # Parse arguments
    argparser = argparse.ArgumentParser()
    # Required arguments
    argparser.add_argument("robot_ip", help="IP address of the robot server")
    argparser.add_argument("local_ip", help="IP address of this PC")
    argparser.add_argument(
        "frequency", help="command frequency, 20 to 200 [Hz]", type=int)
    # Optional arguments
    argparser.add_argument(
        "--TCP", action="store_true",
        help="use TCP frame as reference frame, otherwise use base frame")
    argparser.add_argument(
        "--polish", action="store_true",
        help="execute a simple polish action along XY plane, otherwise apply a constant force along Z axis")
    args = argparser.parse_args()

    # Check if arguments are valid
    frequency = args.frequency
    assert (frequency >= 20 and frequency <= 200), "Invalid <frequency> input"

    # Define alias
    robot_states = flexivrdk.RobotStates()
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # Print description
    log.info("Tutorial description:")
    print_description()

    # The reference frame to use, see Robot::sendCartesianMotionForce() for more details
    frame_str = "BASE"
    if args.TCP:
        log.info("Reference frame used for motion force control: robot TCP frame")
        frame_str = "TCP"
    else:
        log.info("Reference frame used for motion force control: robot base frame")

    # Whether to enable polish action
    if args.polish:
        log.info("Robot will execute a polish action along XY plane")
    else:
        log.info("Robot will apply a constant force along Z axis")

    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot(args.robot_ip, args.local_ip)

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Wait for the robot to become operational
        seconds_waited = 0
        while not robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                log.warn(
                    "Still waiting for robot to become operational, please check that the robot 1) "
                    "has no fault, 2) is in [Auto (remote)] mode")

        log.info("Robot is now operational")

        # Move robot to home pose
        log.info("Moving to home pose")
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.executePrimitive("Home()")

        # Wait for the primitive to finish
        while (robot.isBusy()):
            time.sleep(1)

        # Non-real-time Cartesian Motion-force Control
        # ==========================================================================================
        # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        robot.executePrimitive("ZeroFTSensor()")

        # WARNING: during the process, the robot must not contact anything, otherwise the result
        # will be inaccurate and affect following operations
        log.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot")

        # Wait for primitive completion
        while robot.isBusy():
            time.sleep(1)
        log.info("Sensor zeroing complete")

        # Get latest robot states
        robot.getRobotStates(robot_states)

        # Set control mode and initial pose based on reference frame used
        init_pose = []
        if frame_str == "BASE":
            robot.setMode(mode.NRT_CARTESIAN_MOTION_FORCE_BASE)
            # If using base frame, directly read from robot states
            init_pose = robot_states.tcpPose.copy()
        elif frame_str == "TCP":
            robot.setMode(mode.NRT_CARTESIAN_MOTION_FORCE_TCP)
            # If using TCP frame, current TCP is at the reference frame's origin
            init_pose = [0, 0, 0, 1, 0, 0, 0]
        else:
            raise Exception("Invalid reference frame choice")

        print(
            "Initial TCP pose set to [position 3x1, rotation (quaternion) 4x1]: ",
            init_pose)

        # Set loop period
        period = 1.0/frequency
        print("Sending command to robot at", frequency,
              "Hz, or", period, "seconds interval")

        # Periodic loop counter
        loop_counter = 0

        # Flag indicating the initial contact is made
        is_contacted = False

        # Send command periodically at user-specified frequency
        while True:
            # Use sleep to control loop period
            time.sleep(period)

            # Monitor fault on robot server
            if robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")

            # Read robot states
            robot.getRobotStates(robot_states)

            # Compute norm of sensed external force
            ext_force = np.array([robot_states.extWrenchInBase[0],
                                  robot_states.extWrenchInBase[1], robot_states.extWrenchInBase[2]])
            ext_force_norm = np.linalg.norm(ext_force)

            # Set sign of Fz according to reference frame to achieve a "pressing down" behavior
            Fz = 0.0
            if frame_str == "BASE":
                Fz = -PRESSING_FORCE
            elif frame_str == "TCP":
                Fz = PRESSING_FORCE

            # Initialize target vectors
            target_pose = init_pose.copy()
            target_wrench = [0.0, 0.0, Fz, 0.0, 0.0, 0.0]

            # Search for contact
            if not is_contacted:
                # Send both initial pose and wrench commands, the result is
                # force control along Z axis, and motion hold along other axes
                robot.sendCartesianMotionForce(init_pose, target_wrench)

                # Contact is made
                if ext_force_norm > PRESSING_FORCE:
                    is_contacted = True
                    log.warn("Contact detected at robot TCP")

                # Skip the rest actions until contact is made
                continue

            # Repeat the following actions in a 20-second cycle: first 15 seconds
            # do unified motion-force control, the rest 5 seconds trigger smooth
            # transition to pure motion control
            time_elapsed = loop_counter * period
            if loop_counter % (20 * frequency) == 0:
                # Print info at the beginning of action cycle
                if args.polish:
                    log.info("Executing polish with pressing force [N] = "
                             + str(Fz))
                else:
                    log.info("Applying constant force [N] = " + str(Fz))

            elif loop_counter % (20 * frequency) == (15 * frequency):
                # Print info when disabling force control
                log.info(
                    "Disabling force control and transiting smoothly to pure "
                    "motion control")

            elif loop_counter % (20 * frequency) == (20 * frequency - 1):
                # Reset contact flag at the end of action cycle
                is_contacted = False

            elif time_elapsed % 20.0 < 15.0:
                # Simple polish action along XY plane of chosen reference frame
                if args.polish:
                    # Create motion command to sine-sweep along Y direction
                    target_pose[1] = init_pose[1] + SWING_AMP * \
                        math.sin(2 * math.pi * SWING_FREQ *
                                 loop_counter * period)

                    # Send both target pose and wrench commands, the result is
                    # force control along Z axis, and motion control along other
                    # axes
                    robot.sendCartesianMotionForce(target_pose, target_wrench)

                # Apply constant force along Z axis of chosen reference frame
                else:
                    # Send both initial pose and wrench commands, the result is
                    # force control along Z axis, and motion hold along other axes
                    robot.sendCartesianMotionForce(init_pose, target_wrench)

            else:
                # By not passing in targetWrench parameter, the force control will
                # be cancelled and transit smoothly back to pure motion control.
                # The previously force-controlled axis will be gently pulled toward
                # the motion target currently set for that axis. Here we use
                # initPose for example.
                robot.sendCartesianMotionForce(init_pose)

            # Increment loop counter
            loop_counter += 1

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
