/**
 * @example intermediate6_robot_dynamics.cpp
 * This tutorial runs the integrated dynamics engine to obtain robot Jacobian, mass matrix, and
 * gravity force.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Model.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <mutex>

/** @brief Print tutorial description */
void printDescription()
{
    std::cout << "This tutorial runs the integrated dynamics engine to obtain robot Jacobian, mass "
                 "matrix, and gravity force."
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

/** @brief Periodic task running at 100 Hz */
int periodicTask(flexiv::Robot& robot, flexiv::Model& model)
{
    // Logger for printing message with timestamp and coloring
    flexiv::Log log;

    // Data struct for storing robot states
    flexiv::RobotStates robotStates;

    try {
        int loopCounter = 0;
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            loopCounter++;

            // Monitor fault on robot server
            if (robot.isFault()) {
                throw flexiv::ServerException(
                    "periodicTask: Fault occurred on robot server, exiting "
                    "...");
            }

            // Mark timer start point
            auto tic = std::chrono::high_resolution_clock::now();

            // Read robot states
            robot.getRobotStates(robotStates);

            // Update robot model in dynamics engine
            model.updateModel(robotStates.q, robotStates.dtheta);

            // Compute gravity vector
            auto g = model.getGravityForce();

            // Compute mass matrix
            auto M = model.getMassMatrix();

            // Compute Jacobian
            auto J = model.getJacobian("flange");

            // Mark timer end point and get loop time
            auto toc = std::chrono::high_resolution_clock::now();
            auto computeTime
                = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count();

            // Print at 1Hz
            if (loopCounter % 100 == 0) {
                // Print time used to compute g, M, J
                log.info("Computation time = " + std::to_string(computeTime) + " us");
                std::cout << std::endl;
                // Print gravity
                std::cout << std::fixed << std::setprecision(5) << "g = " << g.transpose() << "\n"
                          << std::endl;
                // Print mass matrix
                std::cout << std::fixed << std::setprecision(5) << "M = " << M << "\n" << std::endl;
                // Print Jacobian
                std::cout << std::fixed << std::setprecision(5) << "J = " << J << "\n" << std::endl;
            }
        }
    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
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

        // Move robot to home pose
        log.info("Moving to home pose");
        robot.setMode(flexiv::Mode::NRT_PRIMITIVE_EXECUTION);
        robot.executePrimitive("Home()");

        // Wait for the primitive to finish
        while (robot.isBusy()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // Robot Dynamics
        // =========================================================================================
        // Initialize dynamics engine
        flexiv::Model model(robot);

        std::thread periodicTaskThread(periodicTask, std::ref(robot), std::ref(model));
        periodicTaskThread.join();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
