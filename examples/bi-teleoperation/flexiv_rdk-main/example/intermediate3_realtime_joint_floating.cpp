/**
 * @example intermediate3_realtime_joint_floating.cpp
 * This tutorial runs real-time joint floating with gentle velocity damping, gravity compensation,
 * and soft protection against position limits. This example is ideal for verifying the system's
 * whole-loop real-timeliness, accuracy of the robot dynamics model, and joint torque control
 * performance. If everything works well, all joints should float smoothly.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Scheduler.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <string>
#include <thread>

namespace {
/** Joint velocity damping gains for floating */
const std::vector<double> k_floatingDamping = {10.0, 10.0, 5.0, 5.0, 1.0, 1.0, 1.0};
}

/** @brief Print tutorial description */
void printDescription()
{
    std::cout << "This tutorial runs real-time joint floating with gentle velocity damping, "
                 "gravity compensation, and soft protection against position limits. This example "
                 "is ideal for verifying the system's whole-loop real-timeliness, accuracy of the "
                 "robot dynamics model, and joint torque control performance. If everything works "
                 "well, all joints should float smoothly."
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

/** @brief Callback function for realtime periodic task */
void periodicTask(flexiv::Robot& robot, flexiv::Scheduler& scheduler, flexiv::Log& log,
    flexiv::RobotStates& robotStates)
{
    try {
        // Monitor fault on robot server
        if (robot.isFault()) {
            throw flexiv::ServerException(
                "periodicTask: Fault occurred on robot server, exiting ...");
        }

        // Read robot states
        robot.getRobotStates(robotStates);

        // Robot degrees of freedom
        size_t robotDOF = robotStates.tau.size();

        // Set 0 joint torques
        std::vector<double> targetTorque(robotDOF, 0.0);

        // Add some velocity damping
        for (size_t i = 0; i < robotDOF; ++i) {
            targetTorque[i] = -k_floatingDamping[i] * robotStates.dtheta[i];
        }

        // Send target joint torque to RDK server, enable gravity compensation
        // and joint limits soft protection
        robot.streamJointTorque(targetTorque, true, true);

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        scheduler.stop();
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

        // Create data struct for storing robot states
        flexiv::RobotStates robotStates;

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

        // Real-time Joint Floating
        // =========================================================================================
        // Switch to real-time joint torque control mode
        robot.setMode(flexiv::Mode::RT_JOINT_TORQUE);

        // Create real-time scheduler to run periodic tasks
        flexiv::Scheduler scheduler;
        // Add periodic task with 1ms interval and highest applicable priority
        scheduler.addTask(std::bind(periodicTask, std::ref(robot), std::ref(scheduler),
                              std::ref(log), std::ref(robotStates)),
            "HP periodic", 1, scheduler.maxPriority());
        // Start all added tasks, this is by default a blocking method
        scheduler.start();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
