#!/usr/bin/env python

"""basics7_auto_recovery.py

This tutorial runs an automatic recovery process if the robot's safety system is in recovery
state. See flexiv::Robot::isRecoveryState() and RDK manual for more details.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import argparse

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
    print("This tutorial runs an automatic recovery process if the robot's safety system is in "
          "recovery state. See flexiv::Robot::isRecoveryState() and RDK manual for more details.")
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

    # Print description
    log.info("Tutorial description:")
    print_description()

    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot(args.robot_ip, args.local_ip)

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Run Auto-recovery
        # ==========================================================================================
        # If the system is in recovery state, we can't use isOperational to tell if the enabling
        # process is done, so just wait long enough for the process to finish
        time.sleep(8)

        # Start auto recovery if the system is in recovery state, the involved joints will start to
        # move back into allowed position range
        if robot.isRecoveryState():
            robot.startAutoRecovery()
            # Block forever, must reboot the robot and restart user program after recovery is done
            while True:
                time.sleep(1)

        # Otherwise the system is normal, do nothing
        else:
            log.info(
                "Robot system is not in recovery state, nothing to be done, exiting ...")

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
