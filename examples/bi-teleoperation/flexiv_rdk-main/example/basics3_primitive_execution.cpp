/**
 * @example basics3_primitive_execution.cpp
 * This tutorial executes several basic robot primitives (unit skills). For detailed documentation
 * on all available primitives, please see [Flexiv Primitives](https://www.flexiv.com/primitives/).
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <thread>

/** @brief Print tutorial description */
void printDescription()
{
    std::cout << "This tutorial executes several basic robot primitives (unit skills). For "
                 "detailed documentation on all available primitives, please see [Flexiv "
                 "Primitives](https://www.flexiv.com/primitives/)."
              << std::endl
              << std::endl;
}

/** @brief Print program usage help */
void printHelp()
{
    // clang-format off
    std::cout << "Required arguments: [robot IP] [local IP]" << std::endl;
    std::cout << "    robot IP: address of the robot server" << std::endl;
    std::cout << "    local IP: address of this PC" << std::endl;
    std::cout << "Optional arguments: None" << std::endl;
    std::cout << std::endl;
    // clang-format on
}

int main(int argc, char* argv[])
{
    // Program Setup
    // =============================================================================================
    // Logger for printing message with timestamp and coloring
    flexiv::Log log;

    // Parse parameters
    if (argc < 3 || flexiv::utility::programArgsExistAny(argc, argv, {"-h", "--help"})) {
        printHelp();
        return 1;
    }
    // IP of the robot server
    std::string robotIP = argv[1];
    // IP of the workstation PC running this program
    std::string localIP = argv[2];

    // Print description
    log.info("Tutorial description:");
    printDescription();

    try {
        // RDK Initialization
        // =========================================================================================
        // Instantiate robot interface
        flexiv::Robot robot(robotIP, localIP);

        // Clear fault on robot server if any
        if (robot.isFault()) {
            log.warn("Fault occurred on robot server, trying to clear ...");
            // Try to clear the fault
            robot.clearFault();
            std::this_thread::sleep_for(std::chrono::seconds(2));
            // Check again
            if (robot.isFault()) {
                log.error("Fault cannot be cleared, exiting ...");
                return 1;
            }
            log.info("Fault on robot server is cleared");
        }

        // Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...");
        robot.enable();

        // Wait for the robot to become operational
        int secondsWaited = 0;
        while (!robot.isOperational()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (++secondsWaited == 10) {
                log.warn(
                    "Still waiting for robot to become operational, please check that the robot 1) "
                    "has no fault, 2) is in [Auto (remote)] mode");
            }
        }
        log.info("Robot is now operational");

        // Execute Primitives
        // =========================================================================================
        // Switch to primitive execution mode
        robot.setMode(flexiv::Mode::NRT_PRIMITIVE_EXECUTION);

        // (1) Go to home pose
        // -----------------------------------------------------------------------------------------
        // All parameters of the "Home" primitive are optional, thus we can skip the parameters and
        // the default values will be used
        log.info("Executing primitive: Home");

        // Send command to robot
        robot.executePrimitive("Home()");

        // Wait for the primitive to finish
        while (robot.isBusy()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // (2) Move robot joints to target positions
        // -----------------------------------------------------------------------------------------
        // The required parameter <target> takes in 7 target joint positions. Unit: degrees
        log.info("Executing primitive: MoveJ");

        // Send command to robot
        robot.executePrimitive("MoveJ(target=30 -45 0 90 0 40 30)");

        // Wait for reached target
        while (flexiv::utility::parsePtStates(robot.getPrimitiveStates(), "reachedTarget") != "1") {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // (3) Move robot TCP to a target position in world (base) frame
        // -----------------------------------------------------------------------------------------
        // Required parameter:
        //   target: final target position
        //       [pos_x pos_y pos_z rot_x rot_y rot_z ref_frame ref_point]
        //       Unit: m, deg
        // Optional parameter:
        //   waypoints: waypoints to pass before reaching final target
        //       (same format as above, but can repeat for number of waypoints)
        //   maxVel: maximum TCP linear velocity
        //       Unit: m/s
        // NOTE: The rotations use Euler ZYX convention, rot_x means Euler ZYX angle around X axis
        log.info("Executing primitive: MoveL");

        // Send command to robot
        robot.executePrimitive(
            "MoveL(target=0.65 -0.3 0.2 180 0 180 WORLD WORLD_ORIGIN,waypoints=0.45 0.1 0.2 180 0 "
            "180 WORLD WORLD_ORIGIN 0.45 -0.3 0.2 180 0 180 WORLD WORLD_ORIGIN, maxVel=0.2)");

        // The [Move] series primitive won't terminate itself, so we determine if the robot has
        // reached target location by checking the primitive state "reachedTarget = 1" in the list
        // of current primitive states, and terminate the current primitive manually by sending a
        // new primitive command.
        while (flexiv::utility::parsePtStates(robot.getPrimitiveStates(), "reachedTarget") != "1") {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // (4) Another MoveL that uses TCP frame
        // -----------------------------------------------------------------------------------------
        // In this example the reference frame is changed from WORLD::WORLD_ORIGIN to TRAJ::START,
        // which represents the current TCP frame
        log.info("Executing primitive: MoveL");

        // Example to convert target quaternion [w,x,y,z] to Euler ZYX using utility functions
        std::vector<double> targetQuat = {0.9185587, 0.1767767, 0.3061862, 0.1767767};
        // ZYX = [30, 30, 30] degrees
        auto targetEulerDeg = flexiv::utility::rad2Deg(flexiv::utility::quat2EulerZYX(targetQuat));

        // Send command to robot. This motion will hold current TCP position and only do TCP
        // rotation
        robot.executePrimitive(
            "MoveL(target=0.0 0.0 0.0 " + flexiv::utility::vec2Str(targetEulerDeg) + "TRAJ START)");

        // Wait for reached target
        while (flexiv::utility::parsePtStates(robot.getPrimitiveStates(), "reachedTarget") != "1") {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // All done, stop robot and put into IDLE mode
        robot.stop();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
