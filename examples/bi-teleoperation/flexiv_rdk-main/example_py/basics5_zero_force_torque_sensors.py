#!/usr/bin/env python

"""basics5_zero_force_torque_sensors.py

This tutorial zeros the robot's force and torque sensors, which is a recommended (but not
mandatory) step before any operations that require accurate force/torque measurement.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import argparse
from utility import list2str

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk
# fmt: on


def print_description():
    """
    Print tutorial description.

    """
    print("This tutorial zeros the robot's force and torque sensors, which is a recommended "
          "(but not mandatory) step before any operations that require accurate "
          "force/torque measurement.")
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

        # Zero Sensors
        # ==========================================================================================
        # Get and print the current TCP force/moment readings
        robot_states = flexivrdk.RobotStates()
        robot.getRobotStates(robot_states)
        log.info(
            "TCP force and moment reading in base frame BEFORE sensor zeroing: " +
            list2str(robot_states.extWrenchInBase) + "[N][Nm]")

        # Run the "ZeroFTSensor" primitive to automatically zero force and torque sensors
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.executePrimitive("ZeroFTSensor()")

        # WARNING: during the process, the robot must not contact anything, otherwise the result
        # will be inaccurate and affect following operations
        log.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot")

        # Wait for the primitive completion
        while (robot.isBusy()):
            time.sleep(1)
        log.info("Sensor zeroing complete")

        # Get and print the current TCP force/moment readings
        robot.getRobotStates(robot_states)
        log.info(
            "TCP force and moment reading in base frame AFTER sensor zeroing: " +
            list2str(robot_states.extWrenchInBase) + "[N][Nm]")

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
