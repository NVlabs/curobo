# cuRobo Mapper C++ Runtime

Source-only C++ runtime for cuRobo's block-sparse mapper. The runtime loads
Warp and cuda.core AOT CUBINs exported from shared cuRobo kernels and launches
them from C++.

The Python mapper bundle is the single source of truth for Warp kernels, and
the cuda.core backend owns the shared pose-refinement LM kernels. The C++
runtime does not carry private copies of either implementation.

## Layout

- `include/`: public C ABI and Warp ABI mirror types.
- `src/`: runtime implementation and CUDA utility helpers.
- `tools/`: AOT artifact export tools.
- `configs/`: example export configs.

The checked-in configs are intentionally minimal:

- `small_mapper.json`: quick smoke/export config.
- `small_mapper_features.json`: quick fused-feature and semantic-filter smoke config.
- `720p_4m_8mm_color_grid.json`: realistic 720p RGB-D mapper config with color grid enabled.

Generated artifacts, caches, benchmarks, and local smoke experiments are not
part of this source boundary.

## Export AOT Artifacts

```bash
python cpp_runtime/mapper/tools/export_mapper_aot.py \
  --config cpp_runtime/mapper/configs/small_mapper.json \
  --out cpp_runtime/mapper/artifacts/small_mapper
```

Artifacts are tied to the exported config and GPU architecture. Re-run the
exporter when image size, camera count, block size, grid shape, color-grid size,
feature-grid size, texture camera shape, support capacity, ESDF shape, or GPU
architecture changes.

## Build Runtime

```bash
WARP_NATIVE_INCLUDE_DIR=$(python - <<'PY'
import pathlib
import warp
print(pathlib.Path(warp.__file__).parent / "native")
PY
)

cmake -S cpp_runtime/mapper \
  -B cpp_runtime/build/mapper \
  -DWARP_NATIVE_INCLUDE_DIR="${WARP_NATIVE_INCLUDE_DIR}"
cmake --build cpp_runtime/build/mapper -j
```

If `nvcc` is not on `PATH`, CMake uses `/usr/local/cuda/bin/nvcc` when present.

## Runtime Scope

Supported runtime calls include:

- RGB-D TSDF integration from caller-owned CUDA buffers.
- Optional float16 camera-feature integration into the block feature grid.
- Optional uint8 instance-slot fusion into a packed per-block ownership grid.
- Occupied voxel extraction.
- Dense ESDF computation using gather seeding plus cuRobo PBA.
- Batched depth/color raycast rendering.
- Compact feature raycast references and float16/float32 feature materialization.
- Cosine and/or argmax semantic filters for occupied voxels, ESDF, raycasts,
  feature raycasts, and mesh extraction.
- Time/frustum decay and empty-block recycling.
- Live updates to the time-decay factor and minimum TSDF site weight; changing
  the site threshold invalidates and recaptures the fixed-shape ESDF graph.
- Triangle-soup mesh extraction into caller-owned CUDA buffers.
- Visibility-tested mesh UV projection with row-major RGB texture atlas output
  and voxel-color fallback patches.
- Ray/SDF camera pose refinement.

Static obstacle stamping and LiDAR integration are not part of this C++ runtime path.

## C ABI

The public C ABI lives in `include/curobo_mapper_runtime/api.h`. A runtime
instance owns one map and its scratch buffers; callers own input/output CUDA
buffers and pass a CUDA stream for ordered execution.

`curobo_mapper_extract_occupied_points` samples observed TSDF zero-crossing squares into
caller-owned point/source-index buffers with a GPU-resident accepted count. It supports instance,
semantic, and source-voxel filters, partial-corner interpolation, and CUDA Graph capture after
warm-up.

The exported artifact directory fixes compile-time mapper dimensions, including
image shape, camera count, block size, grid shape, color-grid size, feature-grid
size, instance-grid size, texture camera shape, ESDF shape, support capacity,
and GPU architecture.
Changing one of these requires re-exporting artifacts and creating a new runtime
instance.

Integration frames use caller-owned CUDA buffers for:

- `float32[num_cameras, image_height, image_width]` depth images.
- `uint8[num_cameras, image_height, image_width, 3]` RGB images.
- `float32[num_cameras, 3, 3]` intrinsics.
- `float32[num_cameras, 3]` camera positions.
- `float32[num_cameras, 4]` camera quaternions in cuRobo `wxyz` order.
- Optional `uint8[num_cameras, image_height, image_width]` compact instance
  slots. Slot zero is background and slots 1 through 255 are registry-owned.

Feature-enabled artifacts accept `CuroboMapperFeatureFrame` through
`curobo_mapper_integrate_features`. Its feature grid is a caller-owned,
channels-last
`float16[num_cameras, feature_grid_height, feature_grid_width, feature_dim]`
CUDA buffer. The feature dimensions and grouped/tiled integration policy are
fixed by the export config.

Semantic APIs accept a `CuroboMapperSemanticFilter`. Use either a caller-owned
`float32[feature_dim]` CUDA `target` (`target_class = -1`), or set a direct
non-negative `target_class` for categorical argmax filtering without a host
round trip. Set `use_min_similarity` for inclusive cosine matching,
`require_argmax` for channel-argmax matching, and `exclude` to remove matching
geometry instead of retaining it.

`curobo_mapper_get_device_view` exposes borrowed, read-only CUDA pointers for
the dense per-block RGB and feature accumulators, packed instance
owner/confidence state, and the most recently computed FP16 ESDF. These
allocations remain stable for the mapper lifetime. Retire a reused registry
slot with `curobo_mapper_clear_instance_slot` before assigning it to a new
detector track. To remove an object's dynamic TSDF, RGB, and features as well
as its ownership, call `curobo_mapper_remove_instance_geometry`. Its semantic
class bitmask can restrict removal to object-like feature argmax classes, so a
coarse instance cell does not erase nearby environment geometry. With a class
mask, matching voxels are removed from blocks touching the instance and their
one-block neighborhood so coarse-cell and block boundaries do not leave
collision geometry behind. A zero mask removes only voxels directly owned by
the slot. The operation stays asynchronous on the caller's CUDA stream after
its active-block query.

Texture frames use RGB images, visibility depths, intrinsics, and camera poses.
If visibility depth is omitted, the runtime renders TSDF depth for the supplied
texture cameras and uses that depth for the visibility test.

## Notes

The C++ runtime reads `mapper_aot_manifest.json` from the artifact directory at
runtime. The `warp-lang` version used to run the C++ executable must match the
version used to generate the artifact manifest.

The exporter uses the in-tree cuRobo package by default. Set `CUROBO_SOURCE_DIR` only when
intentionally exporting from another checkout.

Run the opt-in native feature/semantic smoke test against an exported artifact:

```bash
CUROBO_MAPPER_TEST_ARTIFACT_DIR=/path/to/artifacts \
  pytest -q cpp_runtime/mapper/tests/test_mapper_runtime.py
```
