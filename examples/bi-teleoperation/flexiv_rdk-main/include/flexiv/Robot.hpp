/**
 * @file Robot.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_ROBOT_HPP_
#define FLEXIVRDK_ROBOT_HPP_

#include "Data.hpp"
#include "Mode.hpp"

#include <vector>
#include <memory>
#include <functional>

namespace flexiv {

/**
 * @class Robot
 * @brief Main interface with the robot, containing system, control, motion, and
 * IO methods. This interface is also responsible for communication.
 */
class Robot
{
public:
    /**
     * @brief Create a flexiv::Robot instance as the main robot interface. RDK
     * services will start and connection with robot server will be established.
     * @param[in] serverIP IP address of the robot server (remote).
     * @param[in] localIP IP address of the workstation PC (local).
     * @throw InitException if the instance failed to initialize.
     * @throw CompatibilityException if the RDK library version is incompatible
     * with robot server.
     * @throw CommException if the connection with robot server failed to
     * establish.
     */
    Robot(const std::string& serverIP, const std::string& localIP);
    virtual ~Robot();

    /**
     * @brief Access general information of the robot.
     * @return RobotInfo struct.
     */
    const RobotInfo info(void);

    //============================= SYSTEM CONTROL =============================
    /**
     * @brief Enable the robot, if E-stop is released and there's no fault, the
     * robot will release brakes, and becomes operational a few seconds later.
     * @throw ExecutionException if error occurred during execution.
     * @throw CommException if connection with robot server is lost.
     */
    void enable(void);

    /**
     * @brief Stop the robot and transit robot mode to Idle.
     * @throw ExecutionException if error occurred during execution.
     */
    void stop(void);

    /**
     * @brief Check if the robot is normally operational, which requires the
     * following conditions to be met: enabled, brakes fully released, in auto
     * mode, no fault, and not in reduced state.
     * @warning The robot won't execute any user command until it becomes
     * normally operational.
     * @return True: operational, false: not operational.
     */
    bool isOperational(void) const;

    /**
     * @brief Check if the robot is currently executing a task. This includes
     * any user commanded operations that requires the robot to execute. For
     * example, plans, primitives, Cartesian and joint motions, etc.
     * @warning Some exceptions exist for primitives, see executePrimitive()
     * warning for more details.
     * @return True: busy, false: idle.
     */
    bool isBusy(void) const;

    /**
     * @brief Check if the connection with the robot server is established.
     * @return True: connected, false: disconnected.
     */
    bool isConnected(void) const;

    /**
     * @brief Check if the robot is in fault state.
     * @return True: robot has fault, false: robot normal.
     */
    bool isFault(void) const;

    /**
     * @brief Check if the Emergency Stop is released.
     * @note True: E-stop released, false: E-stop pressed
     */
    bool isEstopReleased(void) const;

    /**
     * @brief Check if the robot system is in recovery state.
     * @return True: in recovery state, false: not in recovery state.
     * @note Use startAutoRecovery() to carry out recovery operation.
     * @par Recovery state
     * The robot system will enter recovery state if it needs to recover
     * from joint position limit violation (a critical system fault that
     * requires a recovery operation, during which the joints that moved outside
     * the allowed position range will need to move very slowly back into the
     * allowed range). Refer to user manual for more details about system
     * recovery state.
     */
    bool isRecoveryState(void) const;

    /**
     * @brief Try establishing connection with the robot server.
     * @throw CommException if failed to establish connection.
     */
    void connect(void);

    /**
     * @brief Disconnect with the robot server.
     * @throw ExecutionException if error occurred during execution.
     */
    void disconnect(void);

    /**
     * @brief Clear minor fault of the robot.
     * @throw ExecutionException if error occurred during execution.
     */
    void clearFault(void);

    /**
     * @brief Set a new operation mode to the robot and wait until the mode
     * transition is finished.
     * @param[in] mode flexiv::Mode enum.
     * @warning To avoid unexpected behavior, it's recommended to call stop()
     * and check if the robot has come to a complete stop using isStopped()
     * before switching mode.
     * @throw InputException if requested mode is invalid.
     * @throw LogicException if robot is in an unknown operation mode.
     * @throw ServerException if robot is not operational.
     * @throw ExecutionException if failed to transit the robot into specified
     * operation mode after several attempts.
     * @warning This method will block until the robot has successfully
     * transited into the specified operation mode.
     */
    void setMode(Mode mode);

    /**
     * @brief Get the current operation mode of the robot.
     * @return flexiv::Mode enum.
     */
    Mode getMode(void) const;

    //============================ ROBOT OPERATION =============================
    /**
     * @brief Get robot states like joint position, velocity, torque, TCP
     * pose, velocity, etc.
     * @param[out] output Reference of output data object.
     * @note Call this method periodically to get the latest states.
     */
    void getRobotStates(RobotStates& output);

    /**
     * @brief Execute a plan by specifying its index.
     * @param[in] index Index of the plan to execute, can be obtained via
     * getPlanNameList().
     * @note Applicable operation mode: NRT_PLAN_EXECUTION.
     * @throw InputException if index is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @note isBusy() can be used to check if a plan task has finished.
     * @warning This method will block for 50 ms for the non-real-time command
     * to be transmitted and fully processed by the robot server.
     */
    void executePlan(unsigned int index);

    /**
     * @brief Execute a plan by specifying its name.
     * @param[in] name Name of the plan to execute, can be obtained via
     * getPlanNameList().
     * @note Applicable operation mode: NRT_PLAN_EXECUTION.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @note isBusy() can be used to check if a plan task has finished.
     * @warning This method will block for 50 ms for the non-real-time command
     * to be transmitted and fully processed by the robot server.
     */
    void executePlan(const std::string& name);

    /**
     * @brief Pause or resume the execution of the current plan.
     * @param[in] pause True: pause plan, false: resume plan.
     * @note Applicable operation mode: NRT_PLAN_EXECUTION.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @warning This method will block for 50 ms for the non-real-time command
     * to be transmitted and fully processed by the robot server.
     */
    void pausePlan(bool pause);

    /**
     * @brief Get a list of all available plans.
     * @return Available plans in the format of a string list.
     * @throw CommException if there's no response from server.
     * @throw ExecutionException if error occurred during execution.
     * @warning This method will block until the request-reply operation with
     * the server is done. The blocking time varies by communication latency.
     */
    std::vector<std::string> getPlanNameList(void) const;

    /**
     * @brief Get detailed information about the currently running plan.
     * Contains information like plan name, primitive name, node name, node
     * path, node path time period, etc.
     * @param[out] output Reference of output data object.
     * @throw CommException if there's no response from server.
     * @throw ExecutionException if error occurred during execution.
     * @warning This method will block until the request-reply operation with
     * the server is done. The blocking time varies by communication latency.
     */
    void getPlanInfo(PlanInfo& output);

    /**
     * @brief Execute a primitive by specifying its name and parameters, which
     * can be found in the [Flexiv Primitives
     * documentation](https://www.flexiv.com/primitives/).
     * @param[in] ptCmd Primitive command with the following string format:
     * "primitiveName(inputParam1=xxx, inputParam2=xxx, ...)".
     * @note Applicable operation mode: NRT_PRIMITIVE_EXECUTION.
     * @throw InputException if size of the input string is greater than 5kb.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @note A primitive won't terminate itself upon finish, thus isBusy()
     * cannot be used to check if a primitive task is finished, use primitive
     * states like "reachedTarget" instead.
     * @warning The primitive input parameters may not use SI units, please
     * refer to the Flexiv Primitives documentation for exact unit definition.
     * @warning Some primitives may not terminate automatically and require
     * users to manually terminate them based on specific primitive states,
     * for example, most [Move] primitives. In such case, isBusy() will stay
     * true even if it seems everything is done for that primitive.
     * @warning This method will block for 50 ms for the non-real-time command
     * to be transmitted and fully processed by the robot server.
     */
    void executePrimitive(const std::string& ptCmd);

    /**
     * @brief Get feedback states of the currently executing primitive.
     * @return Primitive states in the format of a string list.
     * @throw CommException if there's no response from server.
     * @throw ExecutionException if error occurred during execution.
     * @warning This method will block until the request-reply operation with
     * the server is done. The blocking time varies by communication latency.
     */
    std::vector<std::string> getPrimitiveStates(void) const;

    /**
     * @brief Set global variables for the robot by specifying name and value.
     * @param[in] globalVars Command to set global variables using the format:
     * globalVar1=value(s), globalVar2=value(s), ...
     * @note Applicable operation mode: NRT_PLAN_EXECUTION.
     * @throw InputException if size of the input string is greater than 5kb.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @warning The specified global variable(s) must have already been created
     * in the robot server, otherwise setting a nonexistent global variable will
     * have no effect. To check if a global variable is successfully set, use
     * getGlobalVariables().
     * @warning This method will block for 50 ms for the non-real-time command
     * to be transmitted and fully processed by the robot server.
     */
    void setGlobalVariables(const std::string& globalVars);

    /**
     * @brief Get available global variables from the robot.
     * @return Global variables in the format of a string list.
     * @throw CommException if there's no response from server.
     * @throw ExecutionException if error occurred during execution.
     * @warning This method will block until the request-reply operation with
     * the server is done. The blocking time varies by communication latency.
     */
    std::vector<std::string> getGlobalVariables(void) const;

    /**
     * @brief Check if the robot has come to a complete stop.
     * @return True: stopped, false: still moving.
     */
    bool isStopped(void) const;

    /**
     * @brief If the mounted tool has more than one TCP, switch the TCP being
     * used by the robot server. Default to the 1st one (index = 0).
     * @param[in] index Index of the TCP on the mounted tool to switch to.
     * @note No need to call this method if the mounted tool on the robot has
     * only one TCP, it'll be used by default.
     * @throw ExecutionException if error occurred during execution.
     */
    void switchTcp(unsigned int index);

    /**
     * @brief Start auto recovery to bring joints that are outside the allowed
     * position range back into allowed range.
     * @note Refer to user manual for more details.
     * @see isRecoveryState()
     * @throw ExecutionException if error occurred during execution.
     */
    void startAutoRecovery(void);

    //====================== DIRECT MOTION/FORCE CONTROL =======================
    /**
     * @brief Continuously send joint torque command to robot.
     * @param[in] torques Target joint torques: \f$ {\tau_J}_d \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [Nm] \f$.
     * @param[in] enableGravityComp Enable/disable robot gravity compensation.
     * @param[in] enableSoftLimits Enable/disable soft limits to keep the
     * joints from moving outside the allowed position range, which will
     * trigger a safety fault that requires recovery operation.
     * @note Applicable operation mode: RT_JOINT_TORQUE.
     * @note Real-time (RT).
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @warning Always stream smooth and continuous commands to avoid sudden
     * movements.
     */
    void streamJointTorque(const std::vector<double>& torques,
        bool enableGravityComp = true, bool enableSoftLimits = true);

    /**
     * @brief Continuously send joint position, velocity, and acceleration
     * command.
     * @param[in] positions Target joint positions: \f$ q_d \in \mathbb{R}^{DOF
     * \times 1} \f$. Unit: \f$ [rad] \f$.
     * @param[in] velocities Target joint velocities: \f$ \dot{q}_d \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s] \f$.
     * @param[in] accelerations Target joint accelerations: \f$ \ddot{q}_d \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s^2] \f$.
     * @note Applicable operation mode: RT_JOINT_POSITION.
     * @note Real-time (RT).
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @warning Always stream smooth and continuous commands to avoid sudden
     * movements.
     */
    void streamJointPosition(const std::vector<double>& positions,
        const std::vector<double>& velocities,
        const std::vector<double>& accelerations);

    /**
     * @brief Discretely send joint position, velocity, and acceleration
     * command. The internal trajectory generator will interpolate between two
     * set points and make the motion smooth.
     * @param[in] positions Target joint positions: \f$ q_d \in \mathbb{R}^{DOF
     * \times 1} \f$. Unit: \f$ [rad] \f$.
     * @param[in] velocities Target joint velocities: \f$ \dot{q}_d \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s] \f$.
     * @param[in] accelerations Target joint accelerations: \f$ \ddot{q}_d \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s^2] \f$.
     * @param[in] maxVel Maximum joint velocities: \f$ \dot{q}_{max} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s] \f$.
     * @param[in] maxAcc Maximum joint accelerations: \f$ \ddot{q}_{max} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s^2] \f$.
     * @note Applicable operation mode: NRT_JOINT_POSITION.
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     */
    void sendJointPosition(const std::vector<double>& positions,
        const std::vector<double>& velocities,
        const std::vector<double>& accelerations,
        const std::vector<double>& maxVel, const std::vector<double>& maxAcc);

    /**
     * @brief Continuously command Cartesian motion and force command for the
     * robot to track using its unified motion-force controller.
     * @param[in] pose Target TCP pose in base or TCP frame (depends on
     * operation mode): \f$ {^{O}T_{TCP}}_{d} \f$ or \f$ {^{TCP}T_{TCP}}_{d} \in
     * \mathbb{R}^{7 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * position and \f$ \mathbb{R}^{4 \times 1} \f$ quaternion: \f$ [x, y, z,
     * q_w, q_x, q_y, q_z]^T \f$. Unit: \f$ [m]~[] \f$.
     * @param[in] wrench  Target TCP wrench in base or TCP frame (depends on
     * operation mode): \f$ ^{0}F_d \f$ or \f$ ^{TCP}F_d \in \mathbb{R}^{6
     * \times 1} \f$. If TCP frame is used, unlike motion control, the reference
     * frame for force control is always the robot's current TCP frame. When the
     * target value of a direction is set to non-zero, this direction will
     * smoothly transit from motion control to force control, and the robot will
     * track the target force/moment in this direction using an explicit force
     * controller. When the target value is reset to 0, this direction will then
     * smoothly transit from force control back to motion control, and the robot
     * will gently move to the target motion point even if the set point is
     * distant. Calling with default parameter (all zeros) will result in pure
     * motion control in all directions. Consists of \f$ \mathbb{R}^{3 \times 1}
     * \f$ force and \f$ \mathbb{R}^{3 \times 1} \f$ moment: \f$ [f_x, f_y, f_z,
     * m_x, m_y, m_z]^T \f$. Unit: \f$ [N]~[Nm] \f$.
     * @note Applicable operation modes: RT_CARTESIAN_MOTION_FORCE_BASE,
     * RT_CARTESIAN_MOTION_FORCE_TCP.
     * @note Real-time (RT).
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @warning Reference frame non-orthogonality between motion- and force-
     * controlled directions can happen when using the TCP frame mode
     * (RT_CARTESIAN_MOTION_FORCE_TCP). The reference frame for motion control
     * is defined as the robot TCP frame at the time point when the operation
     * mode is switched into RT_CARTESIAN_MOTION_FORCE_TCP and is updated only
     * upon each mode entrance, since motion control requires a fixed reference
     * frame. The reference frame for force control is defined as the current
     * (latest) robot TCP frame, since force control does not require a fixed
     * reference frame. Such difference in frame definition means that, when
     * force control is enabled for one or more directions, the force-controlled
     * directions and motion-controlled directions are not guaranteed to stay
     * orthogonal to each other. When non-orthogonality happens, the affected
     * directions will see some control performance degradation. To avoid
     * reference frame non-orthogonality and retain maximum control performance,
     * it's recommended to keep the robot's Cartesian orientation unchanged when
     * running motion-force control in TCP frame mode. Note that the base frame
     * mode (RT_CARTESIAN_MOTION_FORCE_BASE) does not have such restriction.
     * @warning Always stream smooth and continuous commands to avoid sudden
     * movements.
     */
    void streamCartesianMotionForce(const std::vector<double>& pose,
        const std::vector<double>& wrench = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    /**
     * @brief Discretely command Cartesian motion and force command for the
     * robot to track using its unified motion-force controller. An internal
     * motion generator will smooth the discrete commands.
     * @param[in] pose Target TCP pose in base or TCP frame (depends on
     * operation mode): \f$ {^{O}T_{TCP}}_{d} \f$ or \f$ {^{TCP}T_{TCP}}_{d} \in
     * \mathbb{R}^{7 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * position and \f$ \mathbb{R}^{4 \times 1} \f$ quaternion: \f$ [x, y, z,
     * q_w, q_x, q_y, q_z]^T \f$. Unit: \f$ [m]~[] \f$.
     * @param[in] wrench  Target TCP wrench in base or TCP frame (depends on
     * operation mode): \f$ ^{0}F_d \f$ or \f$ ^{TCP}F_d \in \mathbb{R}^{6
     * \times 1} \f$. If TCP frame is used, unlike motion control, the reference
     * frame for force control is always the robot's current TCP frame. When the
     * target value of a direction is set to non-zero, this direction will
     * smoothly transit from motion control to force control, and the robot will
     * track the target force/moment in this direction using an explicit force
     * controller. When the target value is reset to 0, this direction will then
     * smoothly transit from force control back to motion control, and the robot
     * will gently move to the target motion point even if the set point is
     * distant. Calling with default parameter (all zeros) will result in pure
     * motion control in all directions. Consists of \f$ \mathbb{R}^{3 \times 1}
     * \f$ force and \f$ \mathbb{R}^{3 \times 1} \f$ moment: \f$ [f_x, f_y, f_z,
     * m_x, m_y, m_z]^T \f$. Unit: \f$ [N]~[Nm] \f$.
     * @note Applicable operation modes: NRT_CARTESIAN_MOTION_FORCE_BASE,
     * NRT_CARTESIAN_MOTION_FORCE_TCP.
     * @note Real-time (RT).
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     * @warning Reference frame non-orthogonality between motion- and force-
     * controlled directions can happen when using the TCP frame mode
     * (NRT_CARTESIAN_MOTION_FORCE_TCP). The reference frame for motion control
     * is defined as the robot TCP frame at the time point when the operation
     * mode is switched into NRT_CARTESIAN_MOTION_FORCE_TCP and is updated only
     * upon each mode entrance, since motion control requires a fixed reference
     * frame. The reference frame for force control is defined as the current
     * (latest) robot TCP frame, since force control does not require a fixed
     * reference frame. Such difference in frame definition means that, when
     * force control is enabled for one or more directions, the force-controlled
     * directions and motion-controlled directions are not guaranteed to stay
     * orthogonal to each other. When non-orthogonality happens, the affected
     * directions will see some control performance degradation. To avoid
     * reference frame non-orthogonality and retain maximum control performance,
     * it's recommended to keep the robot's Cartesian orientation unchanged when
     * running motion-force control in TCP frame mode. Note that the base frame
     * mode (NRT_CARTESIAN_MOTION_FORCE_BASE) does not have such restriction.
     */
    void sendCartesianMotionForce(const std::vector<double>& pose,
        const std::vector<double>& wrench = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    /**
     * @brief Set motion stiffness for the Cartesian motion-force control modes.
     * @param[in] stiffness Desired Cartesian motion stiffness: \f$ K_d \in
     * \mathbb{R}^{6 \times 1} \f$. Calling with default parameter (all zeros)
     * will reset to the robot's nominal stiffness. Consists of \f$
     * \mathbb{R}^{3 \times 1} \f$ linear stiffness and \f$ \mathbb{R}^{3 \times
     * 1} \f$ angular stiffness: \f$ [k_x, k_y, k_z, k_{Rx}, k_{Ry},
     * k_{Rz}]^T \f$. Unit: \f$ [N/m]~[Nm/rad] \f$.
     * @note Applicable operation modes: RT/NRT_CARTESIAN_MOTION_FORCE_BASE,
     * RT/NRT_CARTESIAN_MOTION_FORCE_TCP.
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     */
    void setCartesianStiffness(
        const std::vector<double>& stiffness = {0, 0, 0, 0, 0, 0});

    /**
     * @brief Set preferred joint positions for the null-space posture control
     * module used in the Cartesian motion-force control modes.
     * @param[in] preferredPositions Preferred joint positions for the
     * null-space posture control: \f$ q_{ns} \in \mathbb{R}^{DOF \times 1} \f$.
     * Calling with default parameter (all zeros) will reset to the robot's
     * nominal preferred joint positions, which is the home posture.
     * Unit: \f$ [rad] \f$.
     * @par Null-space posture control
     * Similar to human arm, a robotic arm with redundant degree(s) of
     * freedom (DOF > 6) can change its overall posture without affecting the
     * ongoing primary task. This is achieved through a technique called
     * "null-space control". After the preferred joint positions for a desired
     * robot posture is set using this method, the robot's null-space control
     * module will try to pull the arm as close to this posture as possible
     * without affecting the primary Cartesian motion-force control task.
     * @note Applicable operation modes: RT/NRT_CARTESIAN_MOTION_FORCE_BASE,
     * RT/NRT_CARTESIAN_MOTION_FORCE_TCP.
     * @throw InputException if input is invalid.
     * @throw LogicException if robot is not in the correct operation mode.
     * @throw ExecutionException if error occurred during execution.
     */
    void setNullSpacePosture(
        const std::vector<double>& preferredPositions = {0, 0, 0, 0, 0, 0, 0});

    //=============================== IO CONTROL ===============================
    /**
     * @brief Set digital output on the control box.
     * @param[in] portNumber Port to set value to [0 ~ 15].
     * @param[in] value True: set high, false: set low.
     * @throw ExecutionException if error occurred during execution.
     * @throw InputException if input is invalid.
     */
    void writeDigitalOutput(unsigned int portNumber, bool value);

    /**
     * @brief Read digital input on the control box.
     * @param[in] portNumber Port to read value from [0 ~ 15].
     * @return True: port high, false: port low.
     * @throw CommException if there's no response from server.
     * @throw ExecutionException if error occurred during execution.
     * @throw InputException if input is invalid.
     */
    bool readDigitalInput(unsigned int portNumber);

private:
    class Impl;
    std::unique_ptr<Impl> m_pimpl;

    friend class Model;
    friend class Gripper;
};

} /* namespace flexiv */

#endif /* FLEXIVRDK_ROBOT_HPP_ */
