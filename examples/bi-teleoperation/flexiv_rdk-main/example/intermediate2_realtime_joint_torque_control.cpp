/**
 * @example intermediate2_realtime_joint_torque_control.cpp
 * This tutorial runs real-time joint torque control to hold or sine-sweep all robot joints. An
 * outer position loop is used to generate joint torque commands. This outer position loop + inner
 * torque loop together is also known as an impedance controller.
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
#include <cmath>
#include <thread>

namespace {
/** RT loop period [sec] */
constexpr double k_loopPeriod = 0.001;

/** Outer position loop (impedance) gains, values are only for demo purpose */
const std::vector<double> k_impedanceKp = {3000.0, 3000.0, 800.0, 800.0, 200.0, 200.0, 200.0};
const std::vector<double> k_impedanceKd = {80.0, 80.0, 40.0, 40.0, 8.0, 8.0, 8.0};

/** Sine-sweep trajectory amplitude and frequency */
constexpr double k_sineAmp = 0.035;
constexpr double k_sineFreq = 0.3;
}

/** @brief Print tutorial description */
void printDescription()
{
    std::cout
        << "This tutorial runs real-time joint torque control to hold or sine-sweep all robot "
           "joints. An outer position loop is used to generate joint torque commands. This outer "
           "position loop + inner torque loop together is also known as an impedance controller."
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
    std::cout << "Optional arguments: [--hold]" << std::endl;
    std::cout << "    --hold: robot holds current joint positions, otherwise do a sine-sweep" << std::endl;
    std::cout << std::endl;
    // clang-format on
}

/** @brief Callback function for realtime periodic task */
void periodicTask(flexiv::Robot& robot, flexiv::Scheduler& scheduler, flexiv::Log& log,
    flexiv::RobotStates& robotStates, const std::string& motionType,
    const std::vector<double>& initPos)
{
    // Local periodic loop counter
    static unsigned int loopCounter = 0;

    try {
        // Monitor fault on robot server
        if (robot.isFault()) {
            throw flexiv::ServerException(
                "periodicTask: Fault occurred on robot server, exiting ...");
        }

        // Read robot states
        robot.getRobotStates(robotStates);

        // Robot degrees of freedom
        size_t robotDOF = robotStates.q.size();

        // Target joint positions
        std::vector<double> targetPos(robotDOF, 0);

        // Set target position based on motion type
        if (motionType == "hold") {
            targetPos = initPos;
        } else if (motionType == "sine-sweep") {
            for (size_t i = 0; i < robotDOF; ++i) {
                targetPos[i]
                    = initPos[i]
                      + k_sineAmp * sin(2 * M_PI * k_sineFreq * loopCounter * k_loopPeriod);
            }
        } else {
            throw flexiv::InputException(
                "periodicTask: unknown motion type. Accepted motion types: hold, sine-sweep");
        }

        // Run impedance control on all joints
        std::vector<double> targetTorque(robotDOF);
        for (size_t i = 0; i < robotDOF; ++i) {
            targetTorque[i] = k_impedanceKp[i] * (targetPos[i] - robotStates.q[i])
                              - k_impedanceKd[i] * robotStates.dtheta[i];
        }

        // Send target joint torque to RDK server
        robot.streamJointTorque(targetTorque, true);

        loopCounter++;

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

    // Type of motion specified by user
    std::string motionType = "";
    if (flexiv::utility::programArgsExist(argc, argv, "--hold")) {
        log.info("Robot holding current pose");
        motionType = "hold";
    } else {
        log.info("Robot running joint sine-sweep");
        motionType = "sine-sweep";
    }

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

        // Real-time Joint Torque Control
        // =========================================================================================
        // Switch to real-time joint torque control mode
        robot.setMode(flexiv::Mode::RT_JOINT_TORQUE);

        // Set initial joint positions
        robot.getRobotStates(robotStates);
        auto initPos = robotStates.q;
        log.info("Initial joint positions set to: " + flexiv::utility::vec2Str(initPos));

        // Create real-time scheduler to run periodic tasks
        flexiv::Scheduler scheduler;
        // Add periodic task with 1ms interval and highest applicable priority
        scheduler.addTask(
            std::bind(periodicTask, std::ref(robot), std::ref(scheduler), std::ref(log),
                std::ref(robotStates), std::ref(motionType), std::ref(initPos)),
            "HP periodic", 1, scheduler.maxPriority());
        // Start all added tasks, this is by default a blocking method
        scheduler.start();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
