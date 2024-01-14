/**
 * @example basics7_auto_recovery.cpp
 * This tutorial runs an automatic recovery process if the robot's safety system is in recovery
 * state. See flexiv::Robot::isRecoveryState() and RDK manual for more details.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <string>
#include <thread>

/** @brief Print tutorial description */
void printDescription()
{
    std::cout
        << "This tutorial runs an automatic recovery process if the robot's safety system is in "
           "recovery state. See flexiv::Robot::isRecoveryState() and RDK manual for more details."
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

        // Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...");
        robot.enable();

        // Run Auto-recovery
        // =========================================================================================
        // If the system is in recovery state, we can't use isOperational to tell if the enabling
        // process is done, so just wait long enough for the process to finish
        std::this_thread::sleep_for(std::chrono::seconds(8));

        // Start auto recovery if the system is in recovery state, the involved joints will start to
        // move back into allowed position range
        if (robot.isRecoveryState()) {
            robot.startAutoRecovery();
            // Block forever, must reboot the robot and restart user program after recovery is done
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        // Otherwise the system is normal, do nothing
        else {
            log.info("Robot system is not in recovery state, nothing to be done, exiting ...");
        }
    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
