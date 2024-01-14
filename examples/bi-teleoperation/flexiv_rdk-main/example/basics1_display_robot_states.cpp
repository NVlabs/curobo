/**
 * @example basics1_display_robot_states.cpp
 * This tutorial does the very first thing: check connection with the robot server and print
 * received robot states.
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
    std::cout << "This tutorial does the very first thing: check connection with the robot server "
                 "and print received robot states."
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

/** @brief Print robot states data @ 1Hz */
void printRobotStates(flexiv::Robot& robot, flexiv::Log& log)
{
    // Data struct storing robot states
    flexiv::RobotStates robotStates;

    while (true) {
        // Get the latest robot states
        robot.getRobotStates(robotStates);

        // Print all robot states in JSON format using the built-in ostream
        // operator overloading
        log.info("Current robot states:");
        std::cout << robotStates << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
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

        // Print States
        // =========================================================================================
        // Use std::thread to do scheduling so that this example can run on all OS, since not all OS
        // support flexiv::Scheduler
        std::thread lowPriorityThread(std::bind(printRobotStates, std::ref(robot), std::ref(log)));

        // Properly exit thread
        lowPriorityThread.join();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
