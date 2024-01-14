/**
 * @test test_dynamics_with_tool.cpp
 * A test to evaluate the dynamics engine with tool mounted.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Model.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Scheduler.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <mutex>

namespace {
/** M, G ground truth from MATLAB */
struct GroundTruth
{
    Eigen::Matrix<double, 7, 7> M;
    Eigen::Matrix<double, 7, 1> G;
} g_groundTruth;

/** Data shared between threads */
struct SharedData
{
    int64_t loopTime;
    Eigen::VectorXd G;
    Eigen::MatrixXd M;
} g_data;

/** Mutex on shared data */
std::mutex g_mutex;

}

/** User-defined high-priority periodic task @ 1kHz */
void highPriorityTask(flexiv::Robot& robot, flexiv::Scheduler& scheduler, flexiv::Log& log,
    flexiv::Model& model, flexiv::RobotStates& robotStates)
{
    try {
        // Monitor fault on robot server
        if (robot.isFault()) {
            throw flexiv::ServerException(
                "highPriorityTask: Fault occurred on robot server, exiting "
                "...");
        }

        // Read robot states
        robot.getRobotStates(robotStates);

        // Update robot model in dynamics engine
        model.updateModel(robotStates.q, robotStates.dtheta);

        // Mark timer start point
        auto tic = std::chrono::high_resolution_clock::now();

        // Get M and G after setTool from dynamic engine
        auto M = model.getMassMatrix();
        auto G = model.getGravityForce();

        // mark timer end point and get loop time
        auto toc = std::chrono::high_resolution_clock::now();
        auto loopTime = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count();

        // Safely write shared data
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_data.loopTime = loopTime;
            g_data.M = M;
            g_data.G = G;
        }

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        scheduler.stop();
    }
}

/** User-defined low-priority periodic task @ 1Hz */
void lowPriorityTask()
{
    // wake up every second to do something
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Safely read shared data
    int loopTime;
    Eigen::MatrixXd M;
    Eigen::VectorXd G;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        loopTime = g_data.loopTime;
        M = g_data.M;
        G = g_data.G;
    }

    // print time interval of high-priority periodic task
    std::cout << "=====================================================" << std::endl;
    std::cout << "Loop time = " << loopTime << " us" << std::endl;

    // evaluate M, G after setTool and compute their norm
    auto deltaM = M - g_groundTruth.M;
    auto deltaG = G - g_groundTruth.G;

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Difference of M between ground truth (MATLAB) and "
                 "integrated dynamics engine after setTool() = "
              << std::endl
              << deltaM << std::endl;
    std::cout << "Norm of delta M: " << deltaM.norm() << '\n' << std::endl;

    std::cout << "Difference of G between ground truth (MATLAB) and "
                 "integrated dynamics engine after setTool() = "
              << std::endl
              << deltaG.transpose() << std::endl;
    std::cout << "Norm of delta G: " << deltaG.norm() << '\n' << std::endl;
}

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
    // log object for printing message with timestamp and coloring
    flexiv::Log log;

    // Parse Parameters
    //=============================================================================
    if (argc < 3 || flexiv::utility::programArgsExistAny(argc, argv, {"-h", "--help"})) {
        printHelp();
        return 1;
    }

    // IP of the robot server
    std::string robotIP = argv[1];

    // IP of the workstation PC running this program
    std::string localIP = argv[2];

    try {
        // RDK Initialization
        //=============================================================================
        // Instantiate robot interface
        flexiv::Robot robot(robotIP, localIP);

        // create data struct for storing robot states
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

        // enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...");
        robot.enable();

        // Wait for the robot to become operational
        int secondsWaited = 0;
        while (!robot.isOperational()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (++secondsWaited == 10) {
                log.warn(
                    "Still waiting for robot to become operational, please "
                    "check that the robot 1) has no fault, 2) is booted "
                    "into Auto mode");
            }
        }
        log.info("Robot is now operational");

        // set mode after robot is operational
        robot.setMode(flexiv::Mode::NRT_PLAN_EXECUTION);

        // Bring Robot To Home
        //=============================================================================
        robot.executePlan("PLAN-Home");

        // wait for the execution to finish
        do {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } while (robot.isBusy());

        // put mode back to IDLE
        robot.setMode(flexiv::Mode::IDLE);

        // Robot Model (Dynamics Engine) Initialization
        //=============================================================================
        flexiv::Model model(robot);

        // Set Tool
        //=============================================================================
        // artificial tool parameters for verification
        double mass = 0.9;
        // com is relative to tcp frame
        Eigen::Vector3d com = {0.0, 0.0, -0.093};
        // inertia relative to com
        Eigen::Matrix3d inertia;
        inertia << 2.768e-03, 0, 0, 0, 3.149e-03, 0, 0, 0, 5.64e-04;

        log.info("Artificial tool parameters:");
        std::cout << "mass = " << mass << std::endl;
        std::cout << "CoM = " << com << std::endl;
        std::cout << "inertia = " << inertia << std::endl;

        // Hard-coded Dynamics Ground Truth from MATLAB
        //=============================================================================
        // clang-format off
        g_groundTruth.M << 
        2.916316686749461, -0.052869517013466,  1.903540434220357, -0.124348845003517, -0.041914639740668,  0.027649255000000, -0.001464000000000,
        -0.052869517013466,  3.222619358431081,  0.053667041477633, -1.414236317529289, -0.184390078851855,  0.259867572215541, -0.000000000000000,
        1.903540434220357,  0.053667041477633,  1.416023230140298, -0.002724223477633,  0.000093550780077,  0.001155982477633, -0.001121489064726,
        -0.124348845003517, -1.414236317529289, -0.002724223477633,  1.287676548627496,  0.177147398851855, -0.244344118313748,  0.000000000000000,
        -0.041914639740668, -0.184390078851855,  0.000093550780077,  0.177147398851855,  0.054219756143365, -0.038829881851855,  0.000941041060581,
        0.027649255000000,  0.259867572215541,  0.001155982477633, -0.244344118313748, -0.038829881851855,  0.082171270000000,                  0,
        -0.001464000000000, -0.000000000000000, -0.001121489064726,  0.000000000000000,  0.000941041060581,                  0,  0.001464000000000;

        // Matlab value is the joint torque to resist gravity
        g_groundTruth.G << 
        -0.000000000000001,  52.664497076609663,  0.830964961569619, -22.968509865473024, -2.721399343355234, 3.272076450000000,                 0;
        // clang-format on

        // Periodic Tasks
        //=============================================================================
        flexiv::Scheduler scheduler;
        // Add periodic task with 1ms interval and highest applicable priority
        scheduler.addTask(std::bind(highPriorityTask, std::ref(robot), std::ref(scheduler),
                              std::ref(log), std::ref(model), std::ref(robotStates)),
            "HP periodic", 1, scheduler.maxPriority());
        // Add periodic task with 1s interval and lowest applicable priority
        scheduler.addTask(lowPriorityTask, "LP periodic", 1000, 0);
        // Start all added tasks, this is by default a blocking method
        scheduler.start();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
