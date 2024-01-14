/**
 * @test test_scheduler.cpp
 * A test to evaluate RDK's internal real-time scheduler.
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Log.hpp>
#include <flexiv/Scheduler.hpp>
#include <flexiv/Exception.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>

namespace {

/** Data shared between threads */
struct SharedData
{
    int64_t measuredInterval = 0;
} g_data;

/** Mutex on the shared data */
std::mutex g_mutex;

}

/** User-defined high-priority periodic task @ 1kHz */
void highPriorityTask(flexiv::Scheduler& scheduler, flexiv::Log& log)
{
    static unsigned int loopCounter = 0;

    // Scheduler loop interval start time point
    static std::chrono::high_resolution_clock::time_point tic;

    try {
        // Mark loop interval end point
        auto toc = std::chrono::high_resolution_clock::now();

        // Calculate scheduler's interrupt interval and print
        auto measuredInterval
            = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count();

        // Safely write shared data
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_data.measuredInterval = measuredInterval;
        }

        // Stop scheduler after 5 seconds
        if (++loopCounter > 5000) {
            loopCounter = 0;
            scheduler.stop();
        }

        // Mark loop interval start point
        tic = std::chrono::high_resolution_clock::now();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        scheduler.stop();
    }
}

/** User-defined low-priority periodic task @1Hz */
void lowPriorityTask(flexiv::Log& log)
{
    static uint64_t accumulatedTime = 0;
    static uint64_t numMeasures = 0;
    static float avgInterval = 0.0;
    int measuredInterval = 0;

    // Safely read shared data
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        measuredInterval = g_data.measuredInterval;
    }

    // calculate average time interval
    accumulatedTime += measuredInterval;
    numMeasures++;
    avgInterval = (float)accumulatedTime / (float)numMeasures;

    // print time interval of high-priority periodic task
    log.info("High-priority task interval (curr | avg) = " + std::to_string(measuredInterval)
             + " | " + std::to_string(avgInterval) + " us");
}

void printHelp()
{
    // clang-format off
    std::cout << "Required arguments: None" << std::endl;
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
    if (flexiv::utility::programArgsExistAny(argc, argv, {"-h", "--help"})) {
        printHelp();
        return 1;
    }

    try {
        // Periodic Tasks
        //=============================================================================
        flexiv::Scheduler scheduler;
        // Add periodic task with 1ms interval and highest applicable priority
        scheduler.addTask(std::bind(highPriorityTask, std::ref(scheduler), std::ref(log)),
            "HP periodic", 1, scheduler.maxPriority());
        // Add periodic task with 1s interval and lowest applicable priority
        scheduler.addTask(std::bind(lowPriorityTask, std::ref(log)), "LP periodic", 1000, 0);
        // Start all added tasks, this is by default a blocking method
        scheduler.start();

        // Restart scheduler after 2 seconds
        log.warn("Scheduler will restart in 2 seconds");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        scheduler.start();

    } catch (const flexiv::Exception& e) {
        log.error(e.what());
        return 1;
    }

    return 0;
}
