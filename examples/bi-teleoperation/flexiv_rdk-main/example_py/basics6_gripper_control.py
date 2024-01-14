#!/usr/bin/env python

"""basics6_gripper_control.py

This tutorial does position and force control (if available) for grippers supported by Flexiv.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import argparse
import threading

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk
# fmt: on

# Global flag: whether the gripper control tasks are finished
g_is_done = False


def print_description():
    """
    Print tutorial description.

    """
    print("This tutorial does position and force control (if available) for grippers "
          "supported by Flexiv.")
    print()


def print_gripper_states(gripper, log):
    """
    Print gripper states data @ 1Hz.

    """
    # Data struct storing gripper states
    gripper_states = flexivrdk.GripperStates()

    while (not g_is_done):
        # Get the latest gripper states
        gripper.getGripperStates(gripper_states)

        # Print all gripper states, round all float values to 2 decimals
        log.info("Current gripper states:")
        print("width: ", round(gripper_states.width, 2))
        print("force: ", round(gripper_states.force, 2))
        print("max_width: ", round(gripper_states.maxWidth, 2))
        print("is_moving: ", gripper_states.isMoving)
        time.sleep(1)


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

        # Gripper Control
        # ==========================================================================================
        # Gripper control is not available if the robot is in IDLE mode, so switch to some mode
        # other than IDLE
        robot.setMode(mode.NRT_PLAN_EXECUTION)
        robot.executePlan("PLAN-Home")
        time.sleep(1)

        # Instantiate gripper control interface
        gripper = flexivrdk.Gripper(robot)

        # Thread for printing gripper states
        print_thread = threading.Thread(
            target=print_gripper_states, args=[gripper, log])
        print_thread.start()

        # Position control
        log.info("Closing gripper")
        gripper.move(0.01, 0.1, 20)
        time.sleep(2)
        log.info("Opening gripper")
        gripper.move(0.09, 0.1, 20)
        time.sleep(2)

        # Stop
        log.info("Closing gripper")
        gripper.move(0.01, 0.1, 20)
        time.sleep(0.5)
        log.info("Stopping gripper")
        gripper.stop()
        time.sleep(2)
        log.info("Closing gripper")
        gripper.move(0.01, 0.1, 20)
        time.sleep(2)
        log.info("Opening gripper")
        gripper.move(0.09, 0.1, 20)
        time.sleep(0.5)
        log.info("Stopping gripper")
        gripper.stop()
        time.sleep(2)

        # Force control, if available (sensed force is not zero)
        gripper_states = flexivrdk.GripperStates()
        gripper.getGripperStates(gripper_states)
        if abs(gripper_states.force) > sys.float_info.epsilon:
            log.info("Gripper running zero force control")
            gripper.grasp(0)
            # Exit after 10 seconds
            time.sleep(10)

        # Finished, exit all threads
        gripper.stop()
        global g_is_done
        g_is_done = True
        log.info("Program finished")
        print_thread.join()

    except Exception as e:
        log.error(str(e))


if __name__ == "__main__":
    main()
