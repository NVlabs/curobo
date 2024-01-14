/**
 * @example intermediate5_realtime_cartesian_motion_force_control.cpp
 * This tutorial runs real-time Cartesian-space unified motion-force control to apply force along
 * Z axis of the chosen reference frame, or to execute a simple polish action along XY plane of the
 * chosen reference frame.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Scheduler.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <cmath>
#include <thread>

namespace {
/** RT loop frequency [Hz] */
constexpr size_t k_loopFreq = 1000;

/** RT loop period [sec] */
constexpr double k_loopPeriod = 0.001;

/** TCP sine-sweep amplitude [m] */
constexpr double k_swingAmp = 0.1;

/** TCP sine-sweep frequency [Hz] */
constexpr double k_swingFreq = 0.3;

/** Pressing force to apply during the unified motion-force control [N] */
constexpr double k_pressingForce = 5.0;
}

/** @brief Print tutorial description */
void printDescription()
{
    std::cout << "This tutorial runs real-time Cartesian-space unified motion-force control to "
                 "apply force along Z axis of the chosen reference frame, or to execute a simple "
                 "polish action along XY plane of the chosen reference frame."
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
    std::cout << "Optional arguments: [--hold] [--collision]" << std::endl;
    std::cout << "    --TCP: use TCP frame as reference frame, otherwise use base frame" << std::endl;
    std::cout << "    --polish: execute a simple polish action along XY plane, otherwise apply a "
                 "constant force along Z axis"
              << std::endl
              << std::endl;
    // clang-format on
}

/** Callback function for realtime periodic task */
void periodicTask(flexiv::Robot& robot, flexiv::Scheduler& scheduler, flexiv::Log& log,
    flexiv::RobotStates& robotStates, const std::vector<double>& initPose,
    const std::string frameStr, bool enablePolish)
{
    // Local periodic loop counter
    static size_t loopCounter = 0;

    // Local flag indicating the initial contact is made
    static bool isContacted = false;

    try {
        // Monitor fault on robot server
        if (robot.isFault()) {
            throw flexiv::ServerException(
                "periodicTask: Fault occurred on robot server, exiting ...");
        }

        // Read robot states
        robot.getRobotStates(robotStates);

        // Compute norm of sensed external force
        Eigen::Vector3d extForce = {robotStates.extWrenchInBase[0], robotStates.extWrenchInBase[1],
            robotStates.extWrenchInBase[2]};
        double extForceNorm = extForce.norm();

        // Set sign of Fz according to reference frame to achieve a "pressing down" behavior
        double Fz = 0.0;
        if (frameStr == "BASE") {
            Fz = -k_pressingForce;
        } else if (frameStr == "TCP") {
            Fz = k_pressingForce;
        }

        // Initialize target vectors
        std::vector<double> targetPose = initPose;
        std::vector<double> targetWrench = {0.0, 0.0, Fz, 0.0, 0.0, 0.0};

        // Search for contact
        if (!isContacted) {
            // Send both initial pose and wrench commands, the result is force control along Z axis,
            // and motion hold along other axes
            robot.streamCartesianMotionForce(initPose, targetWrench);

            // Contact is made
            if (extForceNorm > k_pressingForce) {
                isContacted = true;
                log.warn("Contact detected at robot TCP");
            }

            // Skip the rest actions until contact is made
            return;
        }

        // Repeat the following actions in a 20-second cycle: first 15 seconds do unified
        // motion-force control, the rest 5 seconds trigger smooth transition to pure motion control
        if (loopCounter % (20 * k_loopFreq) == 0) {
            // Print info at the beginning of action cycle
            if (enablePolish) {
                log.info("Executing polish with pressing force [N] = " + std::to_string(Fz));
            } else {
                log.info("Applying constant force [N] = " + std::to_string(Fz));
            }

        } else if (loopCounter % (20 * k_loopFreq) == (15 * k_loopFreq)) {
            log.info("Disabling force control and transiting smoothly to pure motion control");

        } else if (loopCounter % (20 * k_loopFreq) == (20 * k_loopFreq - 1)) {
            // Reset contact flag at the end of action cycle
            isContacted = false;

        } else if (loopCounter % (20 * k_loopFreq) < (15 * k_loopFreq)) {
            // Simple polish action along XY plane of chosen reference frame
            if (enablePolish) {
                // Create motion command to sine-sweep along Y direction
                targetPose[1]
                    = initPose[1]
                      + k_swingAmp * sin(2 * M_PI * k_swingFreq * loopCounter * k_loopPeriod);

                // Send both target pose and wrench commands, the result is force control along Z
                // axis, and motion control along other axes
                robot.streamCartesianMotionForce(targetPose, targetWrench);
            }
            // Apply constant force along Z axis of chosen reference frame
            else {
                // Send both initial pose and wrench commands, the result is force control along Z
                // axis, and motion hold along other axes
                robot.streamCartesianMotionForce(initPose, targetWrench);
            }

        } else {
            // By not passing in targetWrench parameter, the force control will be cancelled and
            // transit smoothly back to pure motion control. The previously force-controlled axis
            // will be gently pulled toward the motion target currently set for that axis. Here we
            // use initPose for example.
            robot.streamCartesianMotionForce(initPose);
        }

        // Increment loop counter
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

    // The reference frame to use, see Robot::streamCartesianMotionForce() for more details
    std::string frameStr = "BASE";
    if (flexiv::utility::programArgsExist(argc, argv, "--TCP")) {
        log.info("Reference frame used for motion force control: robot TCP frame");
        frameStr = "TCP";
    } else {
        log.info("Reference frame used for motion force control: robot base frame");
    }

    // Whether to enable polish action
    bool enablePolish = false;
    if (flexiv::utility::programArgsExist(argc, argv, "--polish")) {
        log.info("Robot will execute a polish action along XY plane");
        enablePolish = true;
    } else {
        log.info("Robot will apply a constant force along Z axis");
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

        // Real-time Cartesian Motion-force Control
        // =========================================================================================
        // IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        robot.executePrimitive("ZeroFTSensor()");

        // WARNING: during the process, the robot must not contact anything, otherwise the result
        // will be inaccurate and affect following operations
        log.warn("Zeroing force/torque sensors, make sure nothing is in contact with the robot");

        // Wait for primitive completion
        while (robot.isBusy()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        log.info("Sensor zeroing complete");

        // Get latest robot states
        robot.getRobotStates(robotStates);

        // Set control mode and initial pose based on reference frame used
        std::vector<double> initPose;
        if (frameStr == "BASE") {
            robot.setMode(flexiv::Mode::RT_CARTESIAN_MOTION_FORCE_BASE);
            // If using base frame, directly read from robot states
            initPose = robotStates.tcpPose;
        } else if (frameStr == "TCP") {
            robot.setMode(flexiv::Mode::RT_CARTESIAN_MOTION_FORCE_TCP);
            // If using TCP frame, current TCP is at the reference frame's origin
            initPose = {0, 0, 0, 1, 0, 0, 0};
        } else {
            throw flexiv::InputException("Invalid reference frame choice");
        }

        log.info("Initial TCP pose set to [position 3x1, rotation (quaternion) 4x1]: "
                 + flexiv::utility::vec2Str(initPose));

        // Create real-time scheduler to run periodic tasks
        flexiv::Scheduler scheduler;
        // Add periodic task with 1ms interval and highest applicable priority
        scheduler.addTask(
            std::bind(periodicTask, std::ref(robot), std::ref(scheduler), std::ref(log),
                std::ref(robotStates), std::ref(initPose), std::ref(frameStr), enablePolish),
            "HP periodic", 1, scheduler.maxPriority());
        // Start all added tasks, this is by default a blocking method
        scheduler.start();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
