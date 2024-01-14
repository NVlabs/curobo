/**
 * @file Gripper.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_GRIPPER_HPP_
#define FLEXIVRDK_GRIPPER_HPP_

#include "Robot.hpp"
#include <memory>

namespace flexiv {

/**
 * @class Gripper
 * @brief Interface with grippers supported by Flexiv.
 */
class Gripper
{
public:
    /**
     * @brief Create a flexiv::Gripper instance for gripper control.
     * @param[in] robot Reference to the instance of flexiv::Robot.
     * @throw InitException if the instance failed to initialize.
     */
    Gripper(const Robot& robot);
    virtual ~Gripper();

    /**
     * @brief Grasp with direct force control. Requires the mounted gripper to
     * support direct force control.
     * @param[in] force Target gripping force. Positive: closing force,
     * negative: opening force [N].
     * @note Applicable operation modes: all modes except IDLE.
     * @warning Target inputs outside the valid range (specified in gripper's
     * configuration file) will be saturated.
     * @throw ExecutionException if error occurred during execution.
     */
    void grasp(double force);

    /**
     * @brief Move the gripper fingers with position control.
     * @param[in] width Target opening width [m].
     * @param[in] velocity Closing/opening velocity, cannot be 0 [m/s].
     * @param[in] forceLimit Maximum output force during movement [N]. If not
     * specified, default force limit of the mounted gripper will be used.
     * @note Applicable operation modes: all modes except IDLE.
     * @warning Target inputs outside the valid range (specified in gripper's
     * configuration file) will be saturated.
     * @throw ExecutionException if error occurred during execution.
     */
    void move(double width, double velocity, double forceLimit = 0);

    /**
     * @brief Stop the gripper.
     * @note Applicable operation modes: all modes.
     * @throw ExecutionException if error occurred during execution.
     */
    void stop(void);

    /**
     * @brief Get current gripper states.
     * @param[out] output Reference of output data object.
     * @throw CommException if there's no response from server.
     * @throw ExecutionException if error occurred during execution.
     * @warning This method will block until the request-reply operation with
     * the server is done. The blocking time varies by communication latency.
     */
    void getGripperStates(GripperStates& output);

private:
    class Impl;
    std::unique_ptr<Impl> m_pimpl;
};

} /* namespace flexiv */

#endif /* FLEXIVRDK_GRIPPER_HPP_ */
