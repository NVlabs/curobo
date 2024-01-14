/**
 * @test test_dynamics_engine.cpp
 * A test to evaluate the dynamics engine (J, M, G)
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Model.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Scheduler.hpp>
#include <flexiv/Utility.hpp>

#include <Eigen/Eigen>

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <mutex>

namespace {
/** J, M, G ground truth from MATLAB */
struct GroundTruth
{
    Eigen::Matrix<double, 6, 7> J;
    Eigen::Matrix<double, 7, 7> M;
    Eigen::Matrix<double, 7, 1> G;
} g_groundTruth;

/** Data shared between threads */
struct SharedData
{
    int64_t loopTime;
    Eigen::MatrixXd J;
    Eigen::MatrixXd M;
    Eigen::VectorXd G;
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

        // Mark timer start point
        auto tic = std::chrono::high_resolution_clock::now();

        // Get new robot states
        robot.getRobotStates(robotStates);

        // Update robot model in dynamics engine
        model.updateModel(robotStates.q, robotStates.dtheta);

        // Get J, M, G from dynamic engine
        Eigen::MatrixXd J = model.getJacobian("flange");
        Eigen::MatrixXd M = model.getMassMatrix();
        Eigen::VectorXd G = model.getGravityForce();

        // Mark timer end point and get loop time
        auto toc = std::chrono::high_resolution_clock::now();
        auto loopTime = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count();

        // Safely write shared data
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_data.loopTime = loopTime;
            g_data.J = J;
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
    // Safely read shared data
    int loopTime;
    Eigen::MatrixXd J;
    Eigen::MatrixXd M;
    Eigen::VectorXd G;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        loopTime = g_data.loopTime;
        J = g_data.J;
        M = g_data.M;
        G = g_data.G;
    }

    // Print time interval of high-priority periodic task
    std::cout << "=====================================================" << std::endl;
    std::cout << "Loop time = " << loopTime << " us" << std::endl;

    // Evaluate J, M, G with true value
    auto deltaJ = J - g_groundTruth.J;
    auto deltaM = M - g_groundTruth.M;
    auto deltaG = G - g_groundTruth.G;

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Difference of J between ground truth (MATLAB) and "
                 "integrated dynamics engine = "
              << std::endl
              << deltaJ << std::endl;
    std::cout << "Norm of delta J: " << deltaJ.norm() << '\n' << std::endl;

    std::cout << "Difference of M between ground truth (MATLAB) and "
                 "integrated dynamics engine = "
              << std::endl
              << deltaM << std::endl;
    std::cout << "Norm of delta M: " << deltaM.norm() << '\n' << std::endl;

    std::cout << "Difference of G between ground truth (MATLAB) and "
                 "integrated dynamics engine = "
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
                    "Still waiting for robot to become operational, please "
                    "check that the robot 1) has no fault, 2) is booted "
                    "into Auto mode");
            }
        }
        log.info("Robot is now operational");

        // Set mode after robot is operational
        robot.setMode(flexiv::Mode::NRT_PLAN_EXECUTION);

        // Bring Robot To Home
        //=============================================================================
        robot.executePlan("PLAN-Home");

        // Wait for the execution to finish
        do {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } while (robot.isBusy());

        // Put mode back to IDLE
        robot.setMode(flexiv::Mode::IDLE);

        // Robot Model (Dynamics Engine) Initialization
        //=============================================================================
        flexiv::Model model(robot);

        // Hard-coded Dynamics Ground Truth from MATLAB
        //=============================================================================
        // clang-format off
        g_groundTruth.J <<
        0.110000000000000,  0.078420538028673,  0.034471999940354, -0.368152340866938, -0.064278760968654,  0.136000000000000, -0.000000000000000,
        0.687004857483100, -0.000000000000000,  0.576684003660457,  0.000000000000000,  0.033475407198662, -0.000000000000000, -0.000000000000000,
        0.000000000000000,  0.687004857483100, -0.028925442435894, -0.417782862794537, -0.076604444311898,  0.110000000000000, -0.000000000000000,
        0.000000000000000, -0.000000000000000,  0.642787609686539,  0.000000000000000,  0.766044443118978, -0.000000000000000,  0.000000000000000,
                        0, -1.000000000000000, -0.000000000000000,  1.000000000000000,  0.000000000000000, -1.000000000000000,  0.000000000000000,
        1.000000000000000, -0.000000000000000,  0.766044443118978,  0.000000000000000, -0.642787609686539, -0.000000000000000, -1.000000000000000;

        g_groundTruth.M << 
        2.480084579964625, -0.066276150278304,  1.520475428405090, -0.082258763257690, -0.082884472612488,  0.008542255000000, -0.000900000000000,
        -0.066276150278304,  2.778187401738267,  0.067350373889147, -1.100953424157635, -0.129191018084721,  0.165182543869134, -0.000000000000000,
        1.520475428405090,  0.067350373889147,  1.074177611590570, -0.000410055889147, -0.043572229972025, -0.001968185110853, -0.000689439998807,
        -0.082258763257690, -1.100953424157635, -0.000410055889147,  0.964760218577003,  0.123748338084721, -0.125985653288502,  0.000000000000000,
        -0.082884472612488, -0.129191018084721, -0.043572229972025,  0.123748338084721,  0.038006882479315, -0.020080821084721,  0.000578508848718,
        0.008542255000000,  0.165182543869134, -0.001968185110853, -0.125985653288502, -0.020080821084721,  0.034608170000000,                  0,
        -0.000900000000000, -0.000000000000000, -0.000689439998807,  0.000000000000000,  0.000578508848718,                  0,  0.000900000000000;

        // Matlab value is the joint torque to resist gravity
        g_groundTruth.G << 
        0.000000000000000, 46.598931189891374, 1.086347692836129, -19.279904969860052, -2.045058704525489,  2.300886450000000,                  0;
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
