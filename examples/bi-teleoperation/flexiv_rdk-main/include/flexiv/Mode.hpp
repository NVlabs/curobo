/**
 * @file Mode.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_MODE_HPP_
#define FLEXIVRDK_MODE_HPP_

namespace flexiv {

/**
 * @enum Mode
 * @brief Operation modes of the robot.
 */
enum Mode
{
    /** Mode not set */
    UNKNOWN = -1,

    /**
     * No operation to execute, the robot holds position and waits for new
     * command.
     */
    IDLE,

    /**
     * Run real-time joint torque control to track continuous commands @1kHz.
     * @note Real-time (RT) mode
     * @see flexiv::Robot::streamJointTorque()
     */
    RT_JOINT_TORQUE,

    /**
     * Run real-time joint position control to track continuous commands @ 1kHz.
     * @note Real-time (RT) mode
     * @see flexiv::Robot::streamJointPosition()
     */
    RT_JOINT_POSITION,

    /**
     * Run non-real-time joint position control to track discrete commands
     * (smoothened by internal motion generator).
     * @note Non-real-time (NRT) mode
     * @see flexiv::Robot::sendJointPosition()
     */
    NRT_JOINT_POSITION,

    /**
     * Execute pre-configured robot task plans.
     * @note Non-real-time (NRT) mode
     * @see flexiv::Robot::executePlan()
     */
    NRT_PLAN_EXECUTION,

    /**
     * Execute robot primitives (unit skills).
     * @note Non-real-time (NRT) mode
     * @see flexiv::Robot::executePrimitive()
     * @see [Flexiv Primitives](https://www.flexiv.com/primitives/)
     * documentation
     */
    NRT_PRIMITIVE_EXECUTION,

    /**
     * Run real-time Cartesian motion-force control to track continuous commands
     * in base frame @ 1kHz.
     * @note Real-time (RT) mode
     * @see flexiv::Robot::streamCartesianMotionForce()
     */
    RT_CARTESIAN_MOTION_FORCE_BASE,

    /**
     * Run real-time Cartesian motion-force control to track continuous commands
     * in TCP frame @ 1kHz.
     * @note Real-time (RT) mode
     * @see flexiv::Robot::streamCartesianMotionForce()
     */
    RT_CARTESIAN_MOTION_FORCE_TCP,

    /**
     * Run non-real-time Cartesian motion-force control to track discrete
     * commands (smoothened by internal motion generator) in base frame.
     * @note Non-real-time (NRT) mode
     * @see flexiv::Robot::sendCartesianMotionForce()
     */
    NRT_CARTESIAN_MOTION_FORCE_BASE,

    /**
     * Run non-real-time Cartesian motion-force control to track discrete
     * commands (smoothened by internal motion generator) in TCP frame.
     * @note Non-real-time (NRT) mode
     * @see flexiv::Robot::sendCartesianMotionForce()
     */
    NRT_CARTESIAN_MOTION_FORCE_TCP,
};

} /* namespace flexiv */

#endif /* FLEXIVRDK_MODE_HPP_ */
