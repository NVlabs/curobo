/**
 * @file Scheduler.hpp
 * @copyright Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIVRDK_SCHEDULER_HPP_
#define FLEXIVRDK_SCHEDULER_HPP_

#include <string>
#include <functional>
#include <memory>

namespace flexiv {

/**
 * @class Scheduler
 * @brief Real-time scheduler that can simultaneously run multiple periodic
 * tasks. Parameters for each task are configured independently.
 */
class Scheduler
{
public:
    /**
     * @brief Create and initialize a flexiv::Scheduler instance.
     * @throw InitException if the instance failed to initialize.
     * @throw ClientException if an error is triggered by the client computer.
     */
    Scheduler();
    virtual ~Scheduler();

    /**
     * @brief Add a new periodic task to the scheduler's task pool. Each task in
     * the pool is assigned to a dedicated thread with independent thread
     * configuration.
     * @param[in] callback Callback function of user task.
     * @param[in] taskName A unique name for this task.
     * @param[in] interval Execution interval of this periodic task [ms]. The
     * minimum available interval is 1 ms, equivalent to 1 kHz loop frequency.
     * @param[in] priority Priority for this task thread, can be set to 0 ~
     * <max_priority>, with 0 being the lowest, and <max_priority> being the
     * highest. <max_priority> can be obtained from maxPriority(). When the
     * priority is set to non-zero, this thread becomes a real-time thread and
     * can only be interrupted by threads with higher priority. When the
     * priority is set to 0, this thread becomes a non-real-time thread and can
     * be interrupted by any real-time threads. The common practice is to set
     * priority of the most critical tasks to <max_priority> or near, and set
     * priority of other non-critical tasks to 0 or near. To avoid race
     * conditions, the same priority should be assigned to only one task.
     * @param[in] cpuAffinity CPU core for this task thread to bind to, can be
     * set to 2 ~ (<num_cores> - 1). This task thread will only run on
     * the specified CPU core. If left with the default value (-1), then this
     * task thread will not bind to any CPU core, and the system will decide
     * which core to run this task thread on according to the system's own
     * strategy. The common practice is to bind the high-priority task to a
     * dedicated spare core, and bind low-priority tasks to other cores or just
     * leave them unbound (cpuAffinity = -1).
     * @throw LogicException if the scheduler is already started or is not
     * fully initialized yet.
     * @throw InputException if the specified interval/priority/affinity is
     * invalid or the specified task name is duplicate.
     * @throw ClientException if an error is triggered by the client computer.
     * @note Setting CPU affinity on macOS has no effect, as its Mach kernel
     * takes full control of thread placement so CPU binding is not supported.
     * @warning Calling this method after start() is not allowed.
     * @warning For maximum scheduling performance, setting CPU affinity to 0 or
     * 1 is not allowed: core 0 is usually the default core for system processes
     * and can be crowded; core 1 is reserved for the scheduler itself.
     */
    void addTask(std::function<void(void)>&& callback,
        const std::string& taskName, int interval, int priority,
        int cpuAffinity = -1);

    /**
     * @brief Start to execute all added tasks periodically.
     * @param[in] isBlocking Whether to block the thread from which this
     * method is called until the scheduler is stopped. A common usage is to
     * call this method from main() with this parameter set to true to keep
     * main() from returning.
     * @throw LogicException if the scheduler is not fully initialized yet.
     * @throw ClientException if an error is triggered by the client computer.
     */
    void start(bool isBlocking = true);

    /**
     * @brief Stop all added tasks. start() will stop blocking and return.
     * @throw LogicException if the scheduler is not started or is not fully
     * initialized yet.
     * @throw ClientException if an error is triggered by the client computer.
     * @note Call start() again to restart the scheduler.
     */
    void stop();

    /**
     * @brief Get maximum available priority for the user task.
     * @return The maximum priority that can be set to a user task when calling
     * addTask().
     */
    int maxPriority() const;

    /**
     * @brief Get number of tasks added to the scheduler.
     * @return Number of added tasks.
     */
    size_t numTasks() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_pimpl;
};

} /* namespace flexiv */

#endif /* FLEXIVRDK_SCHEDULER_HPP_ */
