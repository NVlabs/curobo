# cuRobo C++ Runtime

Native runtime source and export tooling versioned with cuRobo:

- `robot/`: backend-neutral C++ APIs for robot computation.
- `mapper/`: C++ runtime for cuRobo's block-sparse mapper.
- [`benchmarks/`](benchmarks/README.md): reproducible native IK and motion-planning benchmarks.

Runtime implementations use CUDA and Warp kernels. Public
headers and APIs must not require callers to select or interact with a specific
backend.

Generated artifacts are intentionally kept out of source control. Use
component-local `artifacts/` directories only as ignored local output
locations, or write artifacts under `/tmp`.

## Mapper Runtime

Export mapper AOT artifacts from a cuRobo checkout:

```bash
python cpp_runtime/mapper/tools/export_mapper_aot.py \
  --config cpp_runtime/mapper/configs/small_mapper_features.json \
  --out cpp_runtime/mapper/artifacts/small_mapper_features
```

Build the C++ runtime:

```bash
WARP_NATIVE_INCLUDE_DIR=$(python - <<'PY'
import pathlib
import warp
print(pathlib.Path(warp.__file__).parent / "native")
PY
)

cmake -S cpp_runtime/mapper -B cpp_runtime/build/mapper \
  -DWARP_NATIVE_INCLUDE_DIR="${WARP_NATIVE_INCLUDE_DIR}"
cmake --build cpp_runtime/build/mapper -j
```

## Robot Runtime

The [`robot/`](robot/README.md) runtime provides fused forward kinematics,
geometric Jacobians, multi-seed goalset inverse kinematics, fixed-capacity
single-scene obstacle storage, voxel-volume sphere fitting, and grasped-object
attachment through preallocated robot-link spheres. Full cuRobo robot
configurations also export a paired-candidate joint-space TrajOpt runtime with
uniform timestep refinement and a standalone bounded bidirectional RRT planner
that searches paired IK solutions together. `TrajectorySeedFitter` exposes the
production arc-length, B-spline, and time-optimal IRLS fit. `MotionPlanner`
preserves candidate lanes through direct and RRT-seeded planning, reports the
selected seed, and exposes single-pose `plan_pose()` orchestration from
multi-solution IK through TrajOpt. It also exposes `plan_grasp()` goalset
selection, approach, Cartesian-line insertion, and lift with contact-link
collision control.
The exporter accepts a cuRobo robot configuration or a basic URDF.

## Tests

Run the public runtime tests:

```bash
pytest cpp_runtime/mapper/tests cpp_runtime/robot/tests
```

Set `CUROBO_SOURCE_DIR` only when intentionally exporting from another cuRobo checkout.
