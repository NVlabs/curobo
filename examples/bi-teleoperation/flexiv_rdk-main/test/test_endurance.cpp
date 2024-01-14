/**
 * @test test_endurance.cpp
 * Endurance test running Cartesian impedance control to slowly sine-sweep near
 * home for a duration of user-specified hours. Raw data will be logged to CSV
 * files continuously.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Robot.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Log.hpp>
#include <flexiv/Scheduler.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <thread>

namespace {

// size of Cartesian pose vector [position 3x1 + rotation (quaternion) 4x1 ]
const unsigned int k_cartPoseSize = 7;

// RT loop period [sec]
const double k_loopPeriod = 0.001;

// TCP sine-sweep amplitude [m]
const double k_swingAmp = 0.1;

// TCP sine-sweep frequency [Hz]
const double k_swingFreq = 0.025; // = 10mm/s linear velocity

// current Cartesian-space pose (position + rotation) of robot TCP
std::vector<double> g_currentTcpPose;

// high-priority task loop counter
uint64_t g_hpLoopCounter = 0;

// test duration converted from user-specified hours to loop counts
uint64_t g_testDurationLoopCounts = 0;

// time duration for each log file [loop counts]
const unsigned int k_logDurationLoopCounts = 10 * 60 * 1000; // = 10 min/file

// data to be logged in low-priority thread
struct LogData
{
    std::vector<double> tcpPose;
    std::vector<double> tcpForce;
} g_logData;

}

void highPriorityTask(flexiv::Robot& robot, flexiv::Scheduler& scheduler, flexiv::Log& log,
    flexiv::RobotStates& robotStates)
{
    // flag whether initial Cartesian position is set
    static bool isInitPoseSet = false;

    // Initial Cartesian-space pose (position + rotation) of robot TCP
    static std::vector<double> initTcpPose;

    try {
        // Monitor fault on robot server
        if (robot.isFault()) {
            throw flexiv::ServerException(
                "highPriorityTask: Fault occurred on robot server, exiting "
                "...");
        }

        // Read robot states
        robot.getRobotStates(robotStates);

        // TCP movement control
        //=====================================================================
        // set initial TCP pose
        if (!isInitPoseSet) {
            // check vector size before saving
            if (robotStates.tcpPose.size() == k_cartPoseSize) {
                initTcpPose = robotStates.tcpPose;
                g_currentTcpPose = initTcpPose;
                isInitPoseSet = true;
            }
        }
        // run control after initial pose is set
        else {
            // move along Z direction
            g_currentTcpPose[2]
                = initTcpPose[2]
                  + k_swingAmp * sin(2 * M_PI * k_swingFreq * g_hpLoopCounter * k_loopPeriod);
            robot.streamCartesianMotionForce(g_currentTcpPose);
        }

        // save data to global buffer, not using mutex to avoid interruption on
        // RT loop from potential priority inversion
        g_logData.tcpPose = robotStates.tcpPose;
        g_logData.tcpForce = robotStates.extWrenchInBase;

        // increment loop counter
        g_hpLoopCounter++;

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        scheduler.stop();
    }
}

void lowPriorityTask()
{
    // low-priority task loop counter
    uint64_t lpLoopCounter = 0;

    // data logging CSV file
    std::ofstream csvFile;

    // CSV file name
    std::string csvFileName;

    // CSV file counter (data during the test is divided to multiple files)
    unsigned int fileCounter = 0;

    // log object for printing message with timestamp and coloring
    flexiv::Log log;

    // wait for a while for the robot states data to be available
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // use while loop to prevent this thread from return
    while (true) {
        // run at 1kHz
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Data logging
        //=====================================================================
        // close existing log file and create a new one periodically
        if (lpLoopCounter % k_logDurationLoopCounts == 0) {
            // close file if exist
            if (csvFile.is_open()) {
                csvFile.close();
                log.info("Saved log file: " + csvFileName);
            }

            // increment log file counter
            fileCounter++;

            // create new file name using the updated counter as suffix
            csvFileName = "endurance_test_data_" + std::to_string(fileCounter) + ".csv";

            // open new log file
            csvFile.open(csvFileName);
            if (csvFile.is_open()) {
                log.info("Created new log file: " + csvFileName);
            } else {
                log.error("Failed to create log file: " + csvFileName);
            }
        }

        // log data to file in CSV format, avoid logging too much data otherwise
        // the RT loop will not hold
        if (csvFile.is_open()) {
            // loop counter x1, TCP pose x7, TCP external force x6
            csvFile << lpLoopCounter << ",";

            for (const auto& i : g_logData.tcpPose) {
                csvFile << i << ",";
            }

            for (const auto& i : g_logData.tcpForce) {
                csvFile << i << ",";
            }

            // end of line
            csvFile << '\n';
        }

        // check if the test duration has elapsed
        if (g_hpLoopCounter > g_testDurationLoopCounts) {
            log.info("Test duration has elapsed, saving any open log file ...");

            // close log file
            if (csvFile.is_open()) {
                csvFile.close();
                log.info("Saved log file: " + csvFileName);
            }

            // exit program
            return;
        }

        // increment loop counter
        lpLoopCounter++;
    }
}

void printHelp()
{
    // clang-format off
    std::cout << "Required arguments: [robot IP] [local IP] [test hours]" << std::endl;
    std::cout << "    robot IP: address of the robot server" << std::endl;
    std::cout << "    local IP: address of this PC" << std::endl;
    std::cout << "    test hours: duration of the test, can have decimals" << std::endl;
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
    if (argc < 4 || flexiv::utility::programArgsExistAny(argc, argv, {"-h", "--help"})) {
        printHelp();
        return 1;
    }

    // IP of the robot server
    std::string robotIP = argv[1];

    // IP of the workstation PC running this program
    std::string localIP = argv[2];

    // test duration in hours
    double testHours = std::stof(argv[3]);
    // convert duration in hours to loop counts
    g_testDurationLoopCounts = (uint64_t)(testHours * 3600.0 * 1000.0);
    log.info("Test duration: " + std::to_string(testHours)
             + " hours = " + std::to_string(g_testDurationLoopCounts) + " cycles");

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

        // Bring Robot To Home
        //=============================================================================
        // set mode after robot is operational
        robot.setMode(flexiv::Mode::NRT_PLAN_EXECUTION);

        robot.executePlan("PLAN-Home");

        // Wait fot the plan to finish
        do {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } while (robot.isBusy());

        // set mode after robot is at home
        robot.setMode(flexiv::Mode::RT_CARTESIAN_MOTION_FORCE_BASE);

        // Periodic Tasks
        //=============================================================================
        flexiv::Scheduler scheduler;
        // Add periodic task with 1ms interval and highest applicable priority
        scheduler.addTask(std::bind(highPriorityTask, std::ref(robot), std::ref(scheduler),
                              std::ref(log), std::ref(robotStates)),
            "HP periodic", 1, scheduler.maxPriority());
        // Start all added tasks, not blocking
        scheduler.start(false);

        // Use std::thread for logging task without strict chronology
        std::thread lowPriorityThread(lowPriorityTask);

        // lowPriorityThread is responsible to release blocking
        lowPriorityThread.join();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
