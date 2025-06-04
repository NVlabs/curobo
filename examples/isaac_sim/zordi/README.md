# Zordi CuRobo Integration Examples

This directory contains CuRobo integration examples specifically designed for Zordi plant simulation environments.

## Overview

- **`config_utils.py`** - Path management and YAML loading with environment variable resolution

### Motion Generation Examples

- **`example_expert_usage.py`** - Demonstrates expert motion policies
- **`zordi_motion_gen.py`** - Core motion generation for Zordi environments
- **`zordi_motion_expert.py`** - Expert-level motion planning policies
- **`franka_motion_gen.py`** - Franka robot motion generation (useful for debugging in case Zordi robot is not configured correctly.)

## Usage

1. **Set Environment Variable:**

   ```bash
   export ZORDI_SIM_ASSETS_PATH=/path/to/zordi_sim_assets
   ```

2. **Run Expert Usage Example:**

   ```bash
   isaaclab_py example_expert_usage.py
   ```

3. **Run Motion Generation:**

   ```bash
   isaaclab_py zordi_motion_gen.py
   ```

## Dependencies

- **CuRobo**: Motion planning and optimization
- **Isaac Sim**: Simulation environment
- **ZORDI_SIM_ASSETS**: External asset repository
