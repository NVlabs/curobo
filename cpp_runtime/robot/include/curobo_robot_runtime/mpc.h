// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <array>
#include <memory>
#include "curobo_robot_runtime/trajectory_optimization.h"

namespace curobo::robot {
class MotionPlanner;
namespace detail {
class MpcRuntime;
}

/** Four fixed timing candidates; no IK, RRT or continuous timestep optimization. */
struct MpcConfig {
  static constexpr int TIMING_CANDIDATES = 4;
  static constexpr float MINIMUM_KNOT_DT = 0.01F;
  static constexpr float DEFAULT_MAXIMUM_KNOT_DT = 0.1F;
  static constexpr float MAXIMUM_DT_SCALE = 2.F;
  int iterations{50};
  /** Retry failed probe fans within the fixed CUDA-graph iteration budget. */
  bool line_search_backtracking{true};
  /** Initial plan's speed-scaled sample period; never changed by a failed solve. */
  float trajectory_dt{0.025F};
  float collision_activation_distance{0.005F};
  /** Position/orientation weights from lbfgs_mpc.yml. */
  std::array<float, 2> tool_pose_weight{5000.0F, 200.0F};
};

/** Cartesian path criterion, shared by the artifact's tool frames. */
struct MpcCartesianConfig {
  /** Nonzero translation axis, normalized internally in world or goal frame. */
  std::array<float, 3> direction{0.0F, 0.0F, 1.0F};
  bool direction_in_goal_frame{false};
  /** Running cost multiplier in [0,1]; terminal pose tracking stays full strength. */
  float along_direction_weight{1.0F};
  /** Check perpendicular position/orientation error, independent of running cost weights. */
  bool constrain_path{true};
};

/** Candidate buffers remain owned by MPC and are overwritten by the next step. */
struct MpcResult {
  bool feasible{false};
  float cost{0.0F};
  float collision_cost{0.0F};
  /** Selected uniform sample dt, not knot spacing. Invalid when infeasible. */
  float trajectory_dt{0.0F};
  int selected_candidate{-1};
  const float *position{nullptr};
  const float *velocity{nullptr};
  const float *acceleration{nullptr};
  const float *jerk{nullptr};
};

/**
 * Persistent c-space/Cartesian TrajOpt using shared pre-FK collision parameters.
 *
 * The planner and its scene must outlive this object. Scene/sphere updates and
 * solver calls must be serialized. All input tensors are CUDA float32. reset()
 * latches a joint goal and seed; reset_cartesian() starts a free-endpoint pose
 * phase. Inactive costs have zero GPU weights. Both modes reuse one solve graph.
 * step() shifts the accepted seed by executed whole knot intervals, broadcasts
 * it to four fixed timing candidates, clears L-BFGS, and solves in one graph.
 * accept() commits the selected knots only after external execution admission.
 * An infeasible result is NOT an executable command: the execution layer may
 * retry before its handoff deadline while the last accepted command continues,
 * otherwise decelerate/hold. Validate the actual execution and
 * braking splines before issuing them; a collision-free stop may not exist if
 * an obstacle enters the stopping distance. This class does not drive hardware.
 */
class MpcController final {
 public:
  explicit MpcController(const MotionPlanner &planner, const MpcConfig &config = {});
  ~MpcController();
  MpcController(const MpcController &) = delete;
  MpcController &operator=(const MpcController &) = delete;

  TrajectoryOptimizationInfo info() const;
  /**
   * Latch the planner's uniform sample dt (not knot spacing) at phase entry.
   * Four geometric spacings start at max(initial knot dt,10 ms) and end at
   * max(100 ms, twice that spacing). The 100 ms default is not a hard cap.
   * Candidates stay fixed until the next phase; a solve only selects an index.
   * Updates GPU timing and dt-dependent costs without recapture. A changed dt
   * requires reset/reset_cartesian with matching seed knots before step().
   * Rejects grids exceeding artifact timing or dense-validation capacity;
   * never speeds a plan up or truncates validation to fit those bounds.
   */
  void update_dt(float trajectory_dt, cudaStream_t stream = nullptr);
  /** Fixed sample dt values for the current phase, in increasing order. */
  std::array<float, MpcConfig::TIMING_CANDIDATES> candidate_dts() const;
  /** Goal [dof], initial free knots [num_knots,dof], at config.trajectory_dt. */
  void reset(const float *goal, const float *seed_knots, cudaStream_t stream = nullptr);
  /**
   * Start a Cartesian phase, clear history, and unpin the final joint state.
   * CUDA positions [num_tool_frames,3], normalized wxyz quaternions
   * [num_tool_frames,4], seed [num_knots,dof]. Full terminal pose tracking;
   * path validation permits translation along config.direction. No IK is run.
   */
  void reset_cartesian(const float *positions, const float *quaternions, const float *seed_knots,
                       const MpcCartesianConfig &config = {}, cudaStream_t stream = nullptr);
  /** Update CUDA Cartesian targets without clearing history or recapturing. */
  void update_cartesian_target(const float *positions, const float *quaternions,
                               cudaStream_t stream = nullptr);
  /**
   * Synchronous candidate solve. q/v/a/jerk are CUDA [dof] at the intended handoff.
   * Use the accepted command's right-sided jerk at the knot-aligned boundary.
   * executed_knots advances the last accepted seed by whole knot intervals.
   * The shift is consumed once even on failure: pass zero when retrying at the
   * same handoff. Rejected candidates never replace the accepted seed.
   * After reset, advance only by intervals consumed from the initial seed.
   * Fractional shifts are unsupported; the executor must schedule aligned handoffs.
   * C-space outputs end at the implicit joint goal; Cartesian outputs have a
   * free joint endpoint. Feasibility is not measured-state goal completion.
   * Cartesian acceptance checks path, collisions and joint limits, not terminal
   * pose convergence: a valid horizon may represent partial progress.
   */
  MpcResult step(const float *position, const float *velocity, const float *acceleration,
                 const float *jerk, int executed_knots, cudaStream_t stream = nullptr);
  /** Commit the latest feasible proposal after admission/ACK, before the next step. */
  void accept(cudaStream_t stream = nullptr);
  /**
   * Check execution-spline samples [trajectory_steps,dof] without optimizing.
   * Uses swept world collision, self collision and q/v/a/jerk bounds. Sampling
   * intervals moving a sphere farther than twice the collision clearance are
   * rejected: the finite-step sweep alone can miss sparsely sampled obstacles.
   * Sampling and the swept sphere approximation are not a continuous-time safety proof.
   */
  bool validate_trajectory(const float *position, const float *velocity, const float *acceleration,
                           const float *jerk, float sample_dt, cudaStream_t stream = nullptr);
  /** Number of graph captures; goal/state/scene value changes do not recapture. */
  int graph_capture_count() const;

 private:
  std::unique_ptr<detail::MpcRuntime> runtime_;
};
}  // namespace curobo::robot
