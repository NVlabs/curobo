/**
 * @test test_log.cpp
 * Test log functions of the flexiv::Log class
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 * @author Flexiv
 */

#include <flexiv/Log.hpp>
#include <flexiv/Utility.hpp>

#include <iostream>
#include <thread>

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
    if (flexiv::utility::programArgsExistAny(argc, argv, {"-h", "--help"})) {
        printHelp();
        return 1;
    }

    // log object for printing message with timestamp and coloring
    flexiv::Log log;

    // print info message
    log.info("This is an INFO message with timestamp and GREEN coloring");
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // print warning message
    log.warn("This is a WARNING message with timestamp and YELLOW coloring");
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // print error message
    log.error("This is an ERROR message with timestamp and RED coloring");

    return 0;
}
