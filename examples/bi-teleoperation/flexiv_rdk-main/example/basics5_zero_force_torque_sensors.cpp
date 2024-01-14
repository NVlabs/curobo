/**
 * @example basics5_zero_force_torque_sensors.cpp
 * This tutorial zeros the robot's force and torque sensors, which is a recommended (but not
 * mandatory) step before any operations that require accurate force/torque measurement.
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
    std::cout << "This tutorial zeros the robot's force and torque sensors, which is a recommended "
                 "(but not mandatory) step before any operations that require accurate "
                 "force/torque measurement."
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

        // Zero Sensors
        // =========================================================================================
        // Get and print the current TCP force/moment readings
        flexiv::RobotStates robotStates;
        robot.getRobotStates(robotStates);
        log.info("TCP force and moment reading in base frame BEFORE sensor zeroing: "
                 + flexiv::utility::vec2Str(robotStates.extWrenchInBase) + "[N][Nm]");

        // Run the "ZeroFTSensor" primitive to automatically zero force and torque sensors
        robot.setMode(flexiv::Mode::NRT_PRIMITIVE_EXECUTION);
        robot.executePrimitive("ZeroFTSensor()");

        // WARNING: during the process, the robot must not contact anything, otherwise the result
        // will be inaccurate and affect following operations
        log.warn("Zeroing force/torque sensors, make sure nothing is in contact with the robot");

        // Wait for primitive completion
        while (robot.isBusy()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        log.info("Sensor zeroing complete");

        // Get and print the current TCP force/moment readings
        robot.getRobotStates(robotStates);
        log.info("TCP force and moment reading in base frame AFTER sensor zeroing: "
                 + flexiv::utility::vec2Str(robotStates.extWrenchInBase) + "[N][Nm]");

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
