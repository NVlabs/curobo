#!/usr/bin/env python

"""intermediate4_teach_by_demonstration.py

This tutorial shows a demo implementation for teach by demonstration: free-drive the robot and
record a series of Cartesian poses, which are then reproduced by the robot.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import argparse

# Utility methods
from utility import quat2eulerZYX
from utility import list2str
from utility import parse_pt_states

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk
# fmt: on

# Maximum contact wrench [fx, fy, fz, mx, my, mz] [N][Nm]
MAX_CONTACT_WRENCH = [50.0, 50.0, 50.0, 15.0, 15.0, 15.0]


def print_description():
    """
    Print tutorial description.

    """
    print("This tutorial shows a demo implementation for teach by demonstration: free-drive the "
          "robot and record a series of Cartesian poses, which are then reproduced by the robot.")
    print()


def main():
    # Program Setup
    # ==============================================================================================
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('robot_ip', help='IP address of the robot server')
    argparser.add_argument('local_ip', help='IP address of this PC')
    args = argparser.parse_args()

    # Define alias
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    # Print description
    log.info("Tutorial description:")
    print_description()

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

        # Teach By Demonstration
        # ==========================================================================================
        # Recorded robot poses
        saved_poses = []

        # Robot states data
        robot_states = flexivrdk.RobotStates()

        # Acceptable user inputs
        log.info("Accepted key inputs:")
        print("[n] - start new teaching process")
        print("[r] - record current robot pose")
        print("[e] - finish recording and start execution")

        # User input polling
        input_buffer = ""
        while True:
            input_buffer = str(input())
            # Start new teaching process
            if input_buffer == "n":
                # Clear storage
                saved_poses.clear()

                # Put robot to plan execution mode
                robot.setMode(mode.NRT_PLAN_EXECUTION)

                # Robot run free drive
                robot.executePlan("PLAN-FreeDriveAuto")

                log.info("New teaching process started")
                log.warn(
                    "Hold down the enabling button on the motion bar to activate free drive")
            # Save current robot pose
            elif input_buffer == "r":
                if not robot.isBusy():
                    log.warn("Please start a new teaching process first")
                    continue

                robot.getRobotStates(robot_states)
                saved_poses.append(robot_states.tcpPose)
                log.info("New pose saved: " + str(robot_states.tcpPose))
                log.info("Number of saved poses: " + str(len(saved_poses)))
            # Reproduce recorded poses
            elif input_buffer == "e":
                if len(saved_poses) == 0:
                    log.warn("No pose is saved yet")
                    continue

                # Put robot to primitive execution mode
                robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)

                for i in range(len(saved_poses)):
                    log.info("Executing pose " + str(i + 1) + "/"
                             + str(len(saved_poses)))

                    target_pos = [
                        saved_poses[i][0], saved_poses[i][1], saved_poses[i][2]]
                    # Convert quaternion to Euler ZYX required by MoveCompliance primitive
                    target_quat = [saved_poses[i][3],
                                   saved_poses[i][4], saved_poses[i][5], saved_poses[i][6]]

                    target_euler_deg = quat2eulerZYX(target_quat, degree=True)
                    robot.executePrimitive(
                        "MoveCompliance(target="
                        + list2str(target_pos)
                        + list2str(target_euler_deg)
                        + "WORLD WORLD_ORIGIN, maxVel=0.3, enableMaxContactWrench=1, maxContactWrench="
                        + list2str(MAX_CONTACT_WRENCH) + ")")

                    # Wait for robot to reach target location by checking for "reachedTarget = 1"
                    # in the list of current primitive states
                    while (parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1"):
                        time.sleep(1)

                log.info(
                    "All saved poses are executed, enter 'n' to start a new "
                    "teaching process, 'r' to record more poses, 'e' to repeat "
                    "execution")

                # Put robot back to free drive
                robot.setMode(mode.NRT_PLAN_EXECUTION)
                robot.executePlan("PLAN-FreeDriveAuto")
            else:
                log.warn("Invalid input")

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
