# Zordi Motion Planning Examples

Copyright 2024 Zordi, Inc. All rights reserved.

This folder contains advanced motion planning examples for Zordi robotics applications using Isaac Sim and CuRobo. The examples demonstrate collision-free motion generation for XArm7 robots operating in complex plant environments with obstacle avoidance.

## Overview

The Zordi motion planning system provides multiple approaches for reactive, collision-aware motion generation:

1. **Two-Phase Motion Generation** - Reactive approach with pre-grasp and incremental approach phases
2. **Multi-Candidate Motion Planning** - Batch evaluation of multiple gripper orientations with optimal trajectory selection
3. **Expert Policy System** - Reusable motion policies for integration into larger systems

## Files Description

### Core Examples

- **`zordi_motion_gen.py`** - Main two-phase reactive motion generation example
- **`example_expert_usage.py`** - Demonstrates integration of motion expert policy

### Motion Planning Components

- **`zordi_motion_planner.py`** - Multi-candidate motion planner with batch trajectory generation and cost-based selection
- **`zordi_motion_expert.py`** - Expert policy for two-phase motion planning that can be integrated into other systems
- **`franka_motion_gen.py`** - Franka robot motion generation example (reference implementation)
- **`helper.py`** - Utility functions for robot setup, physics configuration, and visualization

## Key Features

### Two-Phase Motion Approach

- **Phase 1**: Move to pre-grasp position (15cm offset from target) - Fast planning
- **Phase 2**: Incremental approach toward target - Precise collision avoidance

### Multi-Candidate Planning

- Generates trajectories for 5 different gripper orientations simultaneously
- Evaluates trajectories using multiple cost metrics
- Selects optimal trajectory using weighted cost function

### Expert Policy System

- State machine-based motion execution
- Reusable for integration into larger robotic systems
- Configurable parameters for different use cases

## Robot Configuration

The examples use XArm7 robot with Zordi-specific configurations located in:

- `curobo/src/curobo/content/configs/robot/xarm7.yml`
- `curobo/src/curobo/content/configs/robot/spheres/xarm7_improved_gripper.yml`

### Key Configuration Features

- **URDF-based kinematics** for accurate joint modeling
- **Collision sphere approximation** for fast collision checking
- **Locked gripper joint** at open position (0.2 rad)
- **Self-collision avoidance** with appropriate buffers
- **Asset paths** pointing to Zordi simulation assets

## Prerequisites

### Software Requirements

- Isaac Sim 2023.1+ or Isaac Sim 4.5+
- CuRobo (latest version)
- PyTorch with CUDA support
- NumPy, SciPy

### Hardware Requirements

- NVIDIA GPU with CUDA support
- Minimum 8GB GPU memory recommended

### Asset Requirements

- Zordi simulation assets located at `/home/gilwoo/workspace/zordi_sim_assets/`
- Strawberry plant USD files at `/home/gilwoo/workspace/zordi_sim_assets/lightwheel/`
- XArm7 robot URDF and USD files

## Installation

1. **Install CuRobo** in your Isaac Sim environment:

   <https://curobo.org/get_started/1_install_instructions.html>

   ```bash
   # Clone CuRobo repository (if not already available)
   git clone https://github.com/NVlabs/curobo.git

   # Install CuRobo following the official installation guide
   cd curobo
   pip install -e . --no-build-isolation
   ```

Note: I had to downgrade to cuda 11.8-toolkit while keeping the driver to 550.x. Nvidia-smi still shows 12.4.
I'm also running it with isaaclab, though I think issaclab's native python should be sufficient.

## Usage

### Running the Two-Phase Motion Generation Example

```bash
# Navigate to the Zordi examples directory
cd curobo/examples/isaac_sim/zordi

# Run with GUI (default)
python zordi_motion_gen.py

# Run with specific gripper orientation
python zordi_motion_gen.py --orientation_index 2

# Run in headless mode
python zordi_motion_gen.py --headless
```

#### Command Line Arguments

- `--headless`: Run simulation without GUI
- `--orientation_index N`: Use specific gripper orientation (0-4, default: 0)

### Running the Expert Policy Example

```bash
# Navigate to the Zordi examples directory
cd curobo/examples/isaac_sim/zordi

# Run expert policy demonstration
python example_expert_usage.py

# Run in headless mode
python example_expert_usage.py --headless
```

## Motion Planning Parameters

### Gripper Orientations

The system supports 5 different gripper orientations optimized for strawberry picking:

| Index | Description | Vector Direction |
|-------|-------------|------------------|
| 0 | 90° orientation | [1.0, 0.0, 0] |
| 1 | 67.5° orientation | [0.92, 0.38, 0] |
| 2 | 45° orientation | [0.71, 0.71, 0] |
| 3 | 22.5° orientation | [0.38, 0.92, 0] |
| 4 | 0° orientation | [0.0, 1.0, 0] |

### Planning Configuration

- **Position threshold**: 1mm accuracy
- **Rotation threshold**: 15° tolerance
- **Pre-grasp offset**: 15cm from target
- **Approach step size**: 2cm increments
- **Target threshold**: 5mm final accuracy

## Target Configuration

The examples target specific strawberry spheres in the plant environment:

```python
target_prim_path = "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_20/Sphere"
associated_stem_path = "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_20/Stem_20"
```

## Architecture

### Motion Generation Flow

1. **World Setup**: Load plant environment and robot
2. **Obstacle Detection**: Extract collision geometry from USD stage
3. **Target Selection**: Identify strawberry sphere and associated stem (manually set for now)
4. **Orientation Alignment**: Align gripper with stem direction (manually set to be one of 5 orientations)
5. **Phase 1 Planning**: Plan collision-free path to pre-grasp position
6. **Phase 2 Planning**: Plan incremental approach to target
7. **Execution**: Execute trajectories with real-time monitoring

### Multi-Candidate Planning Flow

1. **Batch Generation**: Generate trajectories for all orientations
2. **Collision Evaluation**: Check each trajectory for collisions
3. **Cost Computation**: Calculate weighted cost metrics
4. **Trajectory Selection**: Choose optimal trajectory
5. **Execution**: Execute selected trajectory
