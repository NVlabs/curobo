/**
 * @file Utility.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_UTILITY_HPP_
#define FLEXIVRDK_UTILITY_HPP_

#include <flexiv/Exception.hpp>
#include <Eigen/Eigen>
#include <vector>

namespace flexiv {
namespace utility {

/**
 * @brief Convert quaternion to Euler angles with ZYX axis rotations.
 * @param[in] quat Quaternion input in [w,x,y,z] order.
 * @return Euler angles in [x,y,z] order [rad].
 * @note The return value, when converted to degrees, is the same Euler angles
 * used by Move primitives.
 * @throw InputException if input is of wrong size.
 */
inline std::vector<double> quat2EulerZYX(const std::vector<double>& quat)
{
    // Check input size
    if (quat.size() != 4) {
        throw InputException(
            "[flexiv::utility] quat2EulerZYX: input vector is not size 4");
    }

    // Form quaternion
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);

    // The returned vector is in [z,y,x] order
    auto eulerZYX = q.toRotationMatrix().eulerAngles(2, 1, 0);

    // Convert to general [x,y,z] order
    return (std::vector<double> {eulerZYX[2], eulerZYX[1], eulerZYX[0]});
}

/**
 * @brief Convert radians to degrees for a single value.
 */
inline double rad2Deg(double rad)
{
    constexpr double k_pi = 3.14159265358979323846;
    return (rad / k_pi * 180.0);
}

/**
 * @brief Convert radians to degrees for a vector.
 */
inline std::vector<double> rad2Deg(const std::vector<double>& radVec)
{
    std::vector<double> degVec = {};
    for (const auto& v : radVec) {
        degVec.push_back(rad2Deg(v));
    }
    return degVec;
}

/**
 * @brief Convert a std::vector to a string.
 * @param[in] vec std::vector of any type and size.
 * @param[in] decimal Decimal places to keep for each number in the vector.
 * @param[in] trailingSpace Whether to include a space after the last element.
 * @return A string with format "vec[0] vec[1] ... vec[n] ", i.e. each element
 * followed by a space, including the last one if trailingSpace = true.
 */
template <typename T>
inline std::string vec2Str(
    const std::vector<T>& vec, size_t decimal = 3, bool trailingSpace = true)
{
    auto padding = "";
    std::stringstream ss;
    ss.precision(decimal);
    ss << std::fixed;

    for (const auto& v : vec) {
        ss << padding << v;
        padding = " ";
    }

    if (trailingSpace) {
        ss << " ";
    }
    return ss.str();
}

/**
 * @brief Check if any provided strings exist in the program arguments.
 * @param[in] argc Argument count passed to main() of the program.
 * @param[in] argv Argument vector passed to main() of the program, where
 * argv[0] is the program name.
 * @param[in] refStrs Reference strings to check against.
 * @return True if the program arguments contain one or more reference strings.
 */
inline bool programArgsExistAny(
    int argc, char** argv, const std::vector<std::string>& refStrs)
{
    for (int i = 0; i < argc; i++) {
        for (const auto& v : refStrs) {
            if (v == std::string(argv[i])) {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Check if one specific string exists in the program arguments.
 * @param[in] argc Argument count passed to main() of the program.
 * @param[in] argv Argument vector passed to main() of the program, where
 * argv[0] is the program name.
 * @param[in] refStr Reference string to check against.
 * @return True if the program arguments contain this specific reference string.
 */
inline bool programArgsExist(int argc, char** argv, const std::string& refStr)
{
    return programArgsExistAny(argc, argv, {refStr});
}

/**
 * @brief Parse the value of a specified primitive state from the ptStates
 * string list.
 * @param[in] ptStates Primitive states string list returned from
 * Robot::getPrimitiveStates().
 * @param[in] parseTarget Name of the primitive state to parse for.
 * @return Value of the specified primitive state in string format. Empty string
 * is returned if parseTarget does not exist.
 */
inline std::string parsePtStates(
    const std::vector<std::string>& ptStates, const std::string& parseTarget)
{
    for (const auto& state : ptStates) {
        std::stringstream ss(state);
        std::string buffer;
        std::vector<std::string> parsedState;
        while (ss >> buffer) {
            parsedState.push_back(buffer);
        }
        if (parsedState.front() == parseTarget) {
            return parsedState.back();
        }
    }

    return "";
}

} /* namespace utility */
} /* namespace flexiv */

#endif /* FLEXIVRDK_UTILITY_HPP_ */
