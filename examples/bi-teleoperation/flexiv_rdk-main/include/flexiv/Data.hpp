/**
 * @file Data.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_DATA_HPP_
#define FLEXIVRDK_DATA_HPP_

#include <vector>
#include <string>
#include <ostream>

namespace flexiv {

/**
 * @struct RobotInfo
 * @brief General information of the connected robot.
 */
struct RobotInfo
{
    /** Robot serial number */
    std::string serialNum = {};

    /** Robot degrees of freedom (DOF) */
    unsigned int DOF = {};

    /** Robot software version */
    std::string softwareVer = {};

    /**
     * Nominal motion stiffness of the Cartesian motion-force control modes: \f$
     * K_{nom} \in \mathbb{R}^{6 \times 1} \f$. Consists of \f$ \mathbb{R}^{3
     * \times 1} \f$ linear stiffness and \f$ \mathbb{R}^{3 \times 1} \f$
     * angular stiffness: \f$ [k_x, k_y, k_z, k_{Rx}, k_{Ry}, k_{Rz}]^T \f$.
     * Unit: \f$ [N/m]~[Nm/rad] \f$.
     */
    std::vector<double> nominalK = {};

    /**
     * Joint positions of the robot's home posture: \f$ q_{home} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad] \f$.
     */
    std::vector<double> qHome = {};

    /**
     * Lower limits of joint positions: \f$ q_{min} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad] \f$.
     */
    std::vector<double> qMin = {};

    /**
     * Upper limits of joint positions: \f$ q_{max} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad] \f$.
     */
    std::vector<double> qMax = {};

    /**
     * Upper limits of joint velocities: \f$ \dot{q}_{max} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [rad/s] \f$.
     */
    std::vector<double> dqMax = {};

    /**
     * Upper limits of joint torques: \f$ \tau_{max} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [Nm] \f$.
     */
    std::vector<double> tauMax = {};
};

/**
 * @struct RobotStates
 * @brief Data struct containing the joint- and Cartesian-space robot states.
 */
struct RobotStates
{
    /**
     * Measured joint positions using link-side encoder: \f$ q \in
     * \mathbb{R}^{DOF \times 1} \f$. This is the direct measurement of joint
     * positions, preferred for most cases. Unit: \f$ [rad] \f$.
     */
    std::vector<double> q = {};

    /**
     * Measured joint positions using motor-side encoder: \f$ \theta \in
     * \mathbb{R}^{DOF \times 1} \f$. This is the indirect measurement of joint
     * positions. \f$ \theta = q + \Delta \f$, where \f$ \Delta \f$ is the
     * joint's internal deflection between motor and link. Unit: \f$ [rad] \f$.
     */
    std::vector<double> theta = {};

    /**
     * Measured joint velocities using link-side encoder: \f$ \dot{q} \in
     * \mathbb{R}^{DOF \times 1} \f$. This is the direct but more noisy
     * measurement of joint velocities. Unit: \f$ [rad/s] \f$.
     */
    std::vector<double> dq = {};

    /**
     * Measured joint velocities using motor-side encoder: \f$ \dot{\theta} \in
     * \mathbb{R}^{DOF \times 1} \f$. This is the indirect but less noisy
     * measurement of joint velocities, preferred for most cases.
     * Unit: \f$ [rad/s] \f$.
     */
    std::vector<double> dtheta = {};

    /**
     * Measured joint torques: \f$ \tau \in \mathbb{R}^{DOF \times 1} \f$.
     * Unit: \f$ [Nm] \f$.
     */
    std::vector<double> tau = {};

    /**
     * Desired joint torques: \f$ \tau_{d} \in \mathbb{R}^{DOF \times 1} \f$.
     * Compensation of nonlinear dynamics (gravity, centrifugal, and Coriolis)
     * is excluded. Unit: \f$ [Nm] \f$.
     */
    std::vector<double> tauDes = {};

    /**
     * Numerical derivative of measured joint torques: \f$ \dot{\tau} \in
     * \mathbb{R}^{DOF \times 1} \f$. Unit: \f$ [Nm/s] \f$.
     */
    std::vector<double> tauDot = {};

    /**
     * Estimated external joint torques: \f$ \hat \tau_{ext} \in \mathbb{R}^{DOF
     * \times 1} \f$. Produced by any external contact (with robot body or
     * end-effector) that does not belong to the known robot model.
     * Unit: \f$ [Nm] \f$.
     */
    std::vector<double> tauExt = {};

    /**
     * Measured TCP pose expressed in base frame: \f$ ^{O}T_{TCP} \in
     * \mathbb{R}^{7 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * position and \f$ \mathbb{R}^{4 \times 1} \f$ quaternion: \f$ [x, y, z,
     * q_w, q_x, q_y, q_z]^T \f$. Unit: \f$ [m]~[] \f$.
     */
    std::vector<double> tcpPose = {};

    /**
     * Desired TCP pose expressed in base frame: \f$ {^{O}T_{TCP}}_{d} \in
     * \mathbb{R}^{7 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * position and \f$ \mathbb{R}^{4 \times 1} \f$ quaternion: \f$ [x, y, z,
     * q_w, q_x, q_y, q_z]^T \f$. Unit: \f$ [m]~[] \f$.
     */
    std::vector<double> tcpPoseDes = {};

    /**
     * Measured TCP velocity expressed in base frame: \f$ ^{O}\dot{X} \in
     * \mathbb{R}^{6 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * linear velocity and \f$ \mathbb{R}^{3 \times 1} \f$ angular velocity: \f$
     * [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^T \f$.
     * Unit: \f$ [m/s]~[rad/s] \f$.
     */
    std::vector<double> tcpVel = {};

    /**
     * Measured camera pose expressed in base frame: \f$ ^{O}T_{cam} \in
     * \mathbb{R}^{7 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * position and \f$ \mathbb{R}^{4 \times 1} \f$ quaternion: \f$ [x, y, z,
     * q_w, q_x, q_y, q_z]^T \f$. Unit: \f$ [m]~[] \f$.
     */
    std::vector<double> camPose = {};

    /**
     * Measured flange pose expressed in base frame: \f$ ^{O}T_{flange} \in
     * \mathbb{R}^{7 \times 1} \f$. Consists of \f$ \mathbb{R}^{3 \times 1} \f$
     * position and \f$ \mathbb{R}^{4 \times 1} \f$ quaternion: \f$ [x, y, z,
     * q_w, q_x, q_y, q_z]^T \f$. Unit: \f$ [m]~[] \f$.
     */
    std::vector<double> flangePose = {};

    /**
     * Force-torque (FT) sensor raw reading in flange frame: \f$
     * ^{flange}F_{raw} \in \mathbb{R}^{6 \times 1} \f$. The value is 0 if no FT
     * sensor is installed. Consists of \f$ \mathbb{R}^{3 \times 1} \f$ force
     * and \f$ \mathbb{R}^{3 \times 1} \f$ moment: \f$ [f_x, f_y, f_z, m_x, m_y,
     * m_z]^T \f$. Unit: \f$ [N]~[Nm] \f$.
     */
    std::vector<double> ftSensorRaw = {};

    /**
     * Estimated external wrench applied on TCP and expressed in TCP frame: \f$
     * ^{TCP}F_{ext} \in \mathbb{R}^{6 \times 1} \f$. Consists of \f$
     * \mathbb{R}^{3 \times 1} \f$ force and \f$ \mathbb{R}^{3 \times 1} \f$
     * moment: \f$ [f_x, f_y, f_z, m_x, m_y, m_z]^T \f$. Unit: \f$ [N]~[Nm] \f$.
     */
    std::vector<double> extWrenchInTcp = {};

    /**
     * Estimated external wrench applied on TCP and expressed in base frame: \f$
     * ^{0}F_{ext} \in \mathbb{R}^{6 \times 1} \f$. Consists of \f$
     * \mathbb{R}^{3 \times 1} \f$ force and \f$ \mathbb{R}^{3 \times 1} \f$
     * moment: \f$ [f_x, f_y, f_z, m_x, m_y, m_z]^T \f$. Unit: \f$ [N]~[Nm] \f$.
     */
    std::vector<double> extWrenchInBase = {};
};

/**
 * @struct PlanInfo
 * @brief Data struct containing information of the on-going primitive/plan.
 */
struct PlanInfo
{
    /** Current primitive name */
    std::string ptName = {};

    /** Current node name */
    std::string nodeName = {};

    /** Current node path */
    std::string nodePath = {};

    /** Current node path time period */
    std::string nodePathTimePeriod = {};

    /** Current node path number */
    std::string nodePathNumber = {};

    /** Assigned plan name */
    std::string assignedPlanName = {};

    /** Velocity scale */
    double velocityScale = {};
};

/**
 * @struct GripperStates
 * @brief Data struct containing the gripper states.
 */
struct GripperStates
{
    /** Measured finger opening width [m] */
    double width = {};

    /** Measured finger force. Positive: opening force, negative: closing force.
     * 0 if the mounted gripper has no force sensing capability [N] */
    double force = {};

    /** Maximum finger opening width of the mounted gripper [m] */
    double maxWidth = {};

    /** Whether the fingers are moving */
    bool isMoving = {};
};

/**
 * @brief Operator overloading to out stream all robot info in JSON format:
 * {"info_1": [val1,val2,val3,...], "info_2": [val1,val2,val3,...], ...}.
 * @param[in] ostream Ostream instance.
 * @param[in] robotInfo RobotInfo data struct to out stream.
 * @return Updated ostream instance.
 */
std::ostream& operator<<(
    std::ostream& ostream, const flexiv::RobotInfo& robotInfo);

/**
 * @brief Operator overloading to out stream all robot states in JSON format:
 * {"state_1": [val1,val2,val3,...], "state_2": [val1,val2,val3,...], ...}.
 * @param[in] ostream Ostream instance.
 * @param[in] robotStates RobotStates data struct to out stream.
 * @return Updated ostream instance.
 */
std::ostream& operator<<(
    std::ostream& ostream, const flexiv::RobotStates& robotStates);

/**
 * @brief Operator overloading to out stream all plan info in JSON format:
 * {"info_1": [val1,val2,val3,...], "info_2": [val1,val2,val3,...], ...}.
 * @param[in] ostream Ostream instance.
 * @param[in] planInfo PlanInfo data struct to out stream.
 * @return Updated ostream instance.
 */
std::ostream& operator<<(
    std::ostream& ostream, const flexiv::PlanInfo& planInfo);

/**
 * @brief Operator overloading to out stream all gripper states in JSON format:
 * {"state_1": [val1,val2,val3,...], "state_2": [val1,val2,val3,...], ...}.
 * @param[in] ostream Ostream instance.
 * @param[in] gripperStates GripperStates data struct to out stream.
 * @return Updated ostream instance.
 */
std::ostream& operator<<(
    std::ostream& ostream, const flexiv::GripperStates& gripperStates);

} /* namespace flexiv */

#endif /* FLEXIVRDK_DATA_HPP_ */
