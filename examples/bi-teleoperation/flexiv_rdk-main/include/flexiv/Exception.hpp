/**
 * @file Exception.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_EXCEPTION_HPP_
#define FLEXIVRDK_EXCEPTION_HPP_

#include <stdexcept>

namespace flexiv {

/**
 * @class Exception
 * @brief Base class for all custom runtime exception classes.
 */
class Exception : public std::runtime_error
{
    // Using parent constructor to take in error message
    using std::runtime_error::runtime_error;
};

/**
 * @class InitException
 * @brief Thrown if the initialization of a functional module has failed.
 */
class InitException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class CommException
 * @brief Thrown if the communication/connection with the robot server has an
 * error.
 */
class CommException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class ServerException
 * @brief Thrown if the robot server is not operational or has fault.
 */
class ServerException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class ClientException
 * @brief Thrown if an error is triggered by the client computer.
 */
class ClientException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class InputException
 * @brief Thrown if the user input is not valid.
 */
class InputException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class ExecutionException
 * @brief Thrown if an error occurred when executing a requested operation or
 * computation.
 */
class ExecutionException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class LogicException
 * @brief Thrown if the logic of the program is incorrect.
 */
class LogicException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

/**
 * @class CompatibilityException
 * @brief Thrown if version of the RDK client is not compatible with the robot
 * server.
 */
class CompatibilityException : public Exception
{
    // Using parent constructor to take in error message
    using Exception::Exception;
};

} /* namespace flexiv */

#endif /* FLEXIVRDK_EXCEPTION_HPP_ */
