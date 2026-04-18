# Task Configuration Structure

Configuration files are stored in `curobo/content/configs/task/` for various robot control tasks, organized by task type and solver strategy.

## Naming Convention

We follow a strict **Solver-First, Task-Last** naming convention to ensure consistency and clarity across different task types.

### Standard Format
```
[Solver]_[Variant]_[Task].yml
```

- **Solver**: The optimization method used (e.g., `lbfgs`, `particle`).
- **Variant** (Optional): Specific variations or sub-components (e.g., `retarget`, `bspline`).
- **Task**: The specific task type, used as a suffix (e.g., `ik`, `trajopt`, `mpc`).

### Special Prefixes

- **`transition_`**: Used for transition model configurations.
  - Format: `transition_[Variant]_[Task].yml`
  - Example: `transition_bspline_mpc.yml`
- **`metrics_`**: Used for metric and validation configurations.
  - Format: `metrics_[Scope].yml`
  - Example: `metrics_base.yml`

## Directory Structure

### `ik/` (Inverse Kinematics)
Configurations for solving inverse kinematics problems.
- `particle_ik.yml`: Particle-based (MPPI) solver.
- `lbfgs_ik.yml`: L-BFGS solver.
- `lbfgs_retarget_ik.yml`: Retargeting-specific L-BFGS variant.
- `transition_ik.yml`: Transition model for IK.

### `trajopt/` (Trajectory Optimization)
Configurations for generating full trajectories.
- `particle_trajopt.yml`: Particle-based trajectory optimization.
- `lbfgs_bspline_trajopt.yml`: L-BFGS B-spline trajectory optimization.
- `transition_bspline_trajopt.yml`: Transition model for B-spline trajopt.

### `mpc/` (Model Predictive Control)
Configurations for real-time control loops.
- `lbfgs_mpc.yml`: L-BFGS MPC solver.
- `lbfgs_retarget_mpc.yml`: Retargeting-specific L-BFGS MPC variant.
- `transition_bspline_mpc.yml`: Transition model for MPC.

### `graph_planner/` (Global Planning)
Configurations for graph-based path planners.
- `exact_graph_planner.yml`: Graph planner settings.
- `transition_graph_planner.yml`: Transition model for graph planning.

## Shared Configurations

- **`metrics_base.yml`**: The root validation configuration defining default weights, collision parameters, and evaluation metrics used across tasks.
