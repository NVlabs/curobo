#!/usr/bin/env python

"""basics3_primitive_execution.py

This tutorial executes several basic robot primitives (unit skills). For detailed documentation
on all available primitives, please see [Flexiv Primitives](https://www.flexiv.com/primitives/).
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import argparse

# Utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
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
    print("This tutorial executes several basic robot primitives (unit skills). For "
          "detailed documentation on all available primitives, please see [Flexiv "
          "Primitives](https://www.flexiv.com/primitives/).")
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

        # Execute Primitives
        # ==========================================================================================
        # Switch to primitive execution mode
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)

        # (1) Go to home pose
        # ------------------------------------------------------------------------------------------
        # All parameters of the "Home" primitive are optional, thus we can skip the parameters and
        # the default values will be used
        log.info("Executing primitive: Home")

        # Send command to robot
        robot.executePrimitive("Home()")

        # Wait for the primitive to finish
        while (robot.isBusy()):
            time.sleep(1)

        # (2) Move robot joints to target positions
        # ------------------------------------------------------------------------------------------
        # The required parameter <target> takes in 7 target joint positions. Unit: degrees
        log.info("Executing primitive: MoveJ")

        # Send command to robot
        robot.executePrimitive("MoveJ(target=30 -45 0 90 0 40 30)")

        # Wait for reached target
        while (parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1"):
            time.sleep(1)

        # (3) Move robot TCP to a target position in world (base) frame
        # ------------------------------------------------------------------------------------------
        # Required parameter:
        #   target: final target position
        #       [pos_x pos_y pos_z rot_x rot_y rot_z ref_frame ref_point]
        #       Unit: m, deg
        # Optional parameter:
        #   waypoints: waypoints to pass before reaching final target
        #       (same format as above, but can repeat for number of waypoints)
        #   maxVel: maximum TCP linear velocity
        #       Unit: m/s
        # NOTE: The rotations use Euler ZYX convention, rot_x means Euler ZYX angle around X axis
        log.info("Executing primitive: MoveL")

        # Send command to robot
        robot.executePrimitive(
            "MoveL(target=0.65 -0.3 0.2 180 0 180 WORLD WORLD_ORIGIN,waypoints=0.45 0.1 0.2 180 0 "
            "180 WORLD WORLD_ORIGIN 0.45 -0.3 0.2 180 0 180 WORLD WORLD_ORIGIN, maxVel=0.2)")

        # The [Move] series primitive won't terminate itself, so we determine if the robot has
        # reached target location by checking the primitive state "reachedTarget = 1" in the list
        # of current primitive states, and terminate the current primitive manually by sending a
        # new primitive command.
        while (parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1"):
            time.sleep(1)

        # (4) Another MoveL that uses TCP frame
        # ------------------------------------------------------------------------------------------
        # In this example the reference frame is changed from WORLD::WORLD_ORIGIN to TRAJ::START,
        # which represents the current TCP frame
        log.info("Executing primitive: MoveL")

        # Example to convert target quaternion [w,x,y,z] to Euler ZYX using scipy package's 'xyz'
        # extrinsic rotation
        # NOTE: scipy uses [x,y,z,w] order to represent quaternion
        target_quat = [0.9185587, 0.1767767, 0.3061862, 0.1767767]
        # ZYX = [30, 30, 30] degrees
        eulerZYX_deg = quat2eulerZYX(target_quat, degree=True)

        # Send command to robot. This motion will hold current TCP position and
        # only do TCP rotation
        robot.executePrimitive("MoveL(target=0.0 0.0 0.0 "
                               + list2str(eulerZYX_deg)
                               + "TRAJ START)")

        # Wait for reached target
        while (parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1"):
            time.sleep(1)

        # All done, stop robot and put into IDLE mode
        robot.stop()

    except Exception as e:
        # Print exception error message
        log.error(str(e))


if __name__ == "__main__":
    main()
