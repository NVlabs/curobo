// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/pose_calibration.h"

#include "cuda_check.h"
#include "pose_calibration_cuda.h"
#include "warp_abi.h"
#ifdef CUROBO_ROBOT_HAS_WARP_MESH
#include <version.h>

#include "warp_mesh_backend.h"
#endif
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

namespace curobo::robot {
namespace {
using namespace detail;
using Point = std::array<float, 3>;
using Matrix = std::array<float, 16>;
constexpr Matrix IDENTITY{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

class Buffer {
 public:
  explicit Buffer(size_t bytes) { CRR_CUDA_CHECK(cudaMalloc(&pointer_, bytes)); }
  ~Buffer() {
    if (pointer_) cudaFree(pointer_);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template <class Type>
  Type* get() const {
    return static_cast<Type*>(pointer_);
  }
  void upload(const void* data, size_t bytes, cudaStream_t stream) const {
    CRR_CUDA_CHECK(cudaMemcpyAsync(pointer_, data, bytes, cudaMemcpyHostToDevice, stream));
  }

 private:
  void* pointer_{};
};

class Kernel {
 public:
  ~Kernel() {
    if (module_) cuModuleUnload(module_);
  }
  void load(const std::string& path, const std::string& symbol, int shared) {
    CRR_CU_CHECK(cuModuleLoad(&module_, path.c_str()));
    CRR_CU_CHECK(cuModuleGetFunction(&function_, module_, symbol.c_str()));
    shared_ = shared;
    if (shared_)
      CRR_CU_CHECK(
          cuFuncSetAttribute(function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_));
  }
  void launch(int count, cudaStream_t stream, std::vector<void*> args) const {
    CRR_CU_CHECK(cuLaunchKernel(function_, (count + 255) / 256, 1, 1, 256, 1, 1, shared_, stream,
                                args.data(), nullptr));
  }

 private:
  CUmodule module_{};
  CUfunction function_{};
  int shared_{};
};

template <class Action>
void graph_run(cudaStream_t stream, Action action) {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  try {
    action();
  } catch (...) {
    cudaStreamEndCapture(stream, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  try {
    CRR_CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    if (executable) cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    throw;
  }
  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
}

bool finite(const Point& p) {
  return std::all_of(p.begin(), p.end(), [](float x) { return std::isfinite(x); });
}
void validate(const Matrix& t) {
  for (float x : t)
    if (!std::isfinite(x)) throw std::invalid_argument("calibration seed must be finite");
  for (int i = 0; i < 4; ++i)
    if (std::abs(t[12 + i] - (i == 3 ? 1.F : 0.F)) > 1.e-5F)
      throw std::invalid_argument("calibration seed must be homogeneous");
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float dot = 0;
      for (int k = 0; k < 3; ++k) dot += t[k * 4 + i] * t[k * 4 + j];
      if (std::abs(dot - (i == j ? 1.F : 0.F)) > 1.e-4F)
        throw std::invalid_argument("calibration seed rotation must be orthonormal");
    }
  float determinant = t[0] * (t[5] * t[10] - t[6] * t[9]) - t[1] * (t[4] * t[10] - t[6] * t[8]) +
                      t[2] * (t[4] * t[9] - t[5] * t[8]);
  if (determinant < .999F)
    throw std::invalid_argument("calibration seed must be a proper rotation");
}

Matrix quaternion_matrix(const float* q, const float* position) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  return {1 - 2 * (y * y + z * z),
          2 * (x * y - w * z),
          2 * (x * z + w * y),
          position[0],
          2 * (x * y + w * z),
          1 - 2 * (x * x + z * z),
          2 * (y * z - w * x),
          position[1],
          2 * (x * z - w * y),
          2 * (y * z + w * x),
          1 - 2 * (x * x + y * y),
          position[2],
          0,
          0,
          0,
          1};
}
std::array<float, 4> quaternion(const Matrix& m) {
  std::array<float, 4> q{};
  float trace = m[0] + m[5] + m[10];
  if (trace > 0) {
    float s = 2 * std::sqrt(trace + 1);
    q = {.25F * s, (m[9] - m[6]) / s, (m[2] - m[8]) / s, (m[4] - m[1]) / s};
  } else {
    int i = m[5] > m[0] ? 1 : 0;
    if (m[10] > m[i * 4 + i]) i = 2;
    int j = (i + 1) % 3, k = (i + 2) % 3;
    float s = 2 * std::sqrt(1 + m[i * 4 + i] - m[j * 4 + j] - m[k * 4 + k]);
    q[0] = (m[k * 4 + j] - m[j * 4 + k]) / s;
    q[i + 1] = .25F * s;
    q[j + 1] = (m[j * 4 + i] + m[i * 4 + j]) / s;
    q[k + 1] = (m[k * 4 + i] + m[i * 4 + k]) / s;
  }
  return q;
}

std::vector<Point> resample(const std::vector<Point>& input, int count, std::mt19937& rng) {
  std::vector<int> indices(std::max(static_cast<int>(input.size()), count));
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = i % input.size();
  std::shuffle(indices.begin(), indices.end(), rng);
  std::vector<Point> result(count);
  for (int i = 0; i < count; ++i) result[i] = input[indices[i]];
  return result;
}

std::pair<std::vector<Point>, std::vector<Point>> sample_surface(const CalibrationMesh& mesh,
                                                                 int count, std::mt19937& rng) {
  if (mesh.vertices.empty() || mesh.triangles.empty() || mesh.vertices.size() > 10000000 ||
      mesh.triangles.size() > 10000000)
    throw std::invalid_argument("calibration requires a nonempty bounded triangle mesh");
  for (auto p : mesh.vertices)
    if (!finite(p)) throw std::invalid_argument("mesh has nonfinite vertices");
  std::vector<float> weights;
  std::vector<Point> normals;
  for (auto face : mesh.triangles) {
    for (int index : face)
      if (index < 0 || static_cast<size_t>(index) >= mesh.vertices.size())
        throw std::invalid_argument("invalid calibration triangle index");
    Point u{}, v{}, n{};
    for (int i = 0; i < 3; ++i) {
      u[i] = mesh.vertices[face[1]][i] - mesh.vertices[face[0]][i];
      v[i] = mesh.vertices[face[2]][i] - mesh.vertices[face[0]][i];
    }
    n = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    float area = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    weights.push_back(area);
    if (area > 0)
      for (float& value : n) value /= area;
    normals.push_back(n);
  }
  float total = 0;
  for (float w : weights) total += w;
  if (!(total > 0) || !std::isfinite(total))
    throw std::invalid_argument("calibration mesh has no finite surface area");
  std::discrete_distribution<int> choose(weights.begin(), weights.end());
  std::uniform_real_distribution<float> uniform;
  std::vector<Point> points(count), sample_normals(count);
  for (int i = 0; i < count; ++i) {
    int f = choose(rng);
    float u = uniform(rng), v = uniform(rng);
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    auto face = mesh.triangles[f];
    for (int k = 0; k < 3; ++k)
      points[i][k] = mesh.vertices[face[0]][k] +
                     u * (mesh.vertices[face[1]][k] - mesh.vertices[face[0]][k]) +
                     v * (mesh.vertices[face[2]][k] - mesh.vertices[face[0]][k]);
    sample_normals[i] = normals[f];
  }
  return {points, sample_normals};
}

void validate_config(const PoseCalibrationConfig& c) {
  for (int x : {c.coarse_mesh_points, c.fine_mesh_points, c.coarse_observed_points,
                c.fine_observed_points, c.surface_points})
    if (x < 10 || x > 65536)
      throw std::invalid_argument("calibration sample counts must be in [10,65536]");
  if (c.rotation_samples < 1 || c.rotation_samples > 256)
    throw std::invalid_argument("rotation hypotheses must be in [1,256]");
  for (int x : {c.coarse_iterations, c.fine_iterations, c.surface_iterations})
    if (x < 1 || x > 1000) throw std::invalid_argument("invalid calibration iteration count");
  for (float x : {c.coarse_distance, c.fine_distance, c.icp_huber_delta, c.surface_distance,
                  c.surface_huber_delta})
    if (!std::isfinite(x) || x <= 0)
      throw std::invalid_argument("calibration thresholds must be finite and positive");
  if (!std::isfinite(c.minimum_valid_ratio) || c.minimum_valid_ratio <= 0 ||
      c.minimum_valid_ratio > 1)
    throw std::invalid_argument("invalid calibration overlap ratio");
}
}  // namespace

class PoseCalibration::Impl {
 public:
  Impl(const std::string& directory, PoseCalibrationConfig config) : config_(config) {
    validate_config(config_);
    CRR_CUDA_CHECK(cudaGetDevice(&device_));
    CRR_CUDA_CHECK(cudaFree(nullptr));
    if (config_.refine_surface) {
#ifdef CUROBO_ROBOT_HAS_WARP_MESH
      std::ifstream metadata(directory + "/pose_calibration.meta");
      std::string format, warp_version, query_symbol, reduce_symbol;
      int version = 0, arch = 0, query_shared = 0, reduce_shared = 0;
      metadata >> format >> version >> warp_version >> arch >> query_symbol >> query_shared >>
          reduce_symbol >> reduce_shared;
      if (!metadata || format != "curobo-pose-calibration" || version != 1 || query_shared < 0 ||
          reduce_shared < 0)
        throw std::invalid_argument("invalid target-exported calibration kernel metadata");
      if (warp_version != WP_VERSION_STRING)
        throw std::invalid_argument("calibration export and native Warp versions differ");
      cudaDeviceProp properties{};
      CRR_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_));
      if (arch != properties.major * 10 + properties.minor)
        throw std::invalid_argument("calibration kernels do not match target GPU");
      auto cubin = directory + "/pose_calibration.sm_" + std::to_string(arch) + ".cubin";
      query_.load(cubin, query_symbol, query_shared);
      reduce_.load(cubin, reduce_symbol, reduce_shared);
      backend_ = std::make_unique<detail::WarpMeshBackend>();
#else
      throw std::runtime_error("surface refinement requires WARP_NATIVE_INCLUDE_DIR at build time");
#endif
    }
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  ~Impl() {
    if (stream_) cudaStreamDestroy(stream_);
  }

  PoseCalibrationResult detect(const CalibrationMesh& mesh, const std::vector<Point>& input,
                               const std::optional<Matrix>& seed) {
    int current = 0;
    CRR_CUDA_CHECK(cudaGetDevice(&current));
    if (current != device_)
      throw std::invalid_argument("calibration instance belongs to another CUDA device");
    if (seed) validate(*seed);
    std::vector<Point> observed;
    observed.reserve(input.size());
    for (auto p : input)
      if (finite(p)) observed.push_back(p);
    if (observed.size() < 10 || observed.size() > 10000000)
      throw std::invalid_argument("calibration requires 10 to 10000000 finite observed points");
    std::mt19937 rng(config_.random_seed);
    auto fine = sample_surface(mesh, config_.fine_mesh_points, rng);
    auto coarse = sample_surface(mesh, config_.coarse_mesh_points, rng);
    auto fine_obs = resample(observed, config_.fine_observed_points, rng),
         coarse_obs = resample(observed, config_.coarse_observed_points, rng);
    Buffer points(fine.first.size() * sizeof(Point)), normals(fine.second.size() * sizeof(Point)),
        obs(fine_obs.size() * sizeof(Point));
    Buffer cp(coarse.first.size() * sizeof(Point)), cn(coarse.second.size() * sizeof(Point)),
        co(coarse_obs.size() * sizeof(Point));
    points.upload(fine.first.data(), fine.first.size() * sizeof(Point), stream_);
    normals.upload(fine.second.data(), fine.second.size() * sizeof(Point), stream_);
    obs.upload(fine_obs.data(), fine_obs.size() * sizeof(Point), stream_);
    cp.upload(coarse.first.data(), coarse.first.size() * sizeof(Point), stream_);
    cn.upload(coarse.second.data(), coarse.second.size() * sizeof(Point), stream_);
    co.upload(coarse_obs.data(), coarse_obs.size() * sizeof(Point), stream_);
    int hypotheses = seed ? 1 : config_.rotation_samples;
    std::vector<CalibrationPose> initial(hypotheses);
    std::uniform_real_distribution<float> uniform;
    Point center{};
    for (auto p : coarse_obs)
      for (int i = 0; i < 3; ++i) center[i] += p[i] / coarse_obs.size();
    for (int i = 0; i < hypotheses; ++i) {
      Matrix t = seed ? *seed : IDENTITY;
      if (!seed) {
        float u = uniform(rng), v = uniform(rng), w = uniform(rng), pi = 3.14159265358979323846F;
        float q[4] = {std::sqrt(1 - u) * std::sin(2 * pi * v),
                      std::sqrt(1 - u) * std::cos(2 * pi * v), std::sqrt(u) * std::sin(2 * pi * w),
                      std::sqrt(u) * std::cos(2 * pi * w)};
        t = quaternion_matrix(q, center.data());
      }
      std::copy(t.begin(), t.end(), initial[i].matrix);
      initial[i].hypothesis = i;
    }
    Buffer poses(hypotheses * sizeof(CalibrationPose)), best(sizeof(CalibrationPose));
    poses.upload(initial.data(), initial.size() * sizeof(CalibrationPose), stream_);
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
    graph_run(stream_, [&] {
      if (!seed)
        launch_calibration_icp(cp.get<float>(), cn.get<float>(), config_.coarse_mesh_points,
                               co.get<float>(), config_.coarse_observed_points,
                               poses.get<CalibrationPose>(), hypotheses, config_.coarse_iterations,
                               config_.coarse_distance, config_.icp_huber_delta, stream_);
      launch_calibration_select(poses.get<CalibrationPose>(), hypotheses,
                                best.get<CalibrationPose>(), stream_);
      launch_calibration_icp(points.get<float>(), normals.get<float>(), config_.fine_mesh_points,
                             obs.get<float>(), config_.fine_observed_points,
                             best.get<CalibrationPose>(), 1, config_.fine_iterations,
                             config_.fine_distance, config_.icp_huber_delta, stream_);
    });
    CalibrationPose fitted{};
    CRR_CUDA_CHECK(cudaMemcpy(&fitted, best.get<void>(), sizeof(fitted), cudaMemcpyDeviceToHost));
    PoseCalibrationResult result;
    std::copy(fitted.matrix, fitted.matrix + 16, result.camera_from_base.begin());
    result.icp_mean_distance = fitted.error;
    result.fine_iterations = fitted.iterations;
    result.best_hypothesis = fitted.hypothesis;
    result.valid_ratio = static_cast<float>(fitted.valid) / config_.fine_mesh_points;
    result.surface_rms = std::numeric_limits<float>::infinity();
    if (!std::isfinite(fitted.error) || (!config_.refine_surface && fitted.valid < 10)) {
      result.error = "insufficient ICP correspondences";
      return result;
    }
    if (config_.refine_surface)
      refine(mesh, observed, result, rng);
    else
      result.success = result.valid_ratio >= config_.minimum_valid_ratio;
    if (!result.success && result.error.empty())
      result.error = "insufficient mesh-distance overlap";
    return result;
  }

 private:
  void refine(const CalibrationMesh& mesh, const std::vector<Point>& observed,
              PoseCalibrationResult& result, std::mt19937& rng) {
#ifdef CUROBO_ROBOT_HAS_WARP_MESH
    auto samples = resample(
        observed, std::min(static_cast<int>(observed.size()), config_.surface_points), rng);
    int count = static_cast<int>(samples.size());
    Buffer vertices(mesh.vertices.size() * sizeof(Point)),
        triangles(mesh.triangles.size() * sizeof(std::array<int, 3>)),
        points(samples.size() * sizeof(Point));
    vertices.upload(mesh.vertices.data(), mesh.vertices.size() * sizeof(Point), stream_);
    triangles.upload(mesh.triangles.data(), mesh.triangles.size() * sizeof(std::array<int, 3>),
                     stream_);
    points.upload(samples.data(), samples.size() * sizeof(Point), stream_);
    Buffer position(3 * sizeof(float)), rotation(4 * sizeof(float)),
        distances(count * sizeof(float)), gradients(count * sizeof(Point)),
        mask(count * sizeof(int));
    Buffer hessian(36 * sizeof(float)), gradient(6 * sizeof(float)), squared(sizeof(float)),
        valid(sizeof(int)), state(sizeof(CalibrationLmState));
    Point p{result.camera_from_base[3], result.camera_from_base[7], result.camera_from_base[11]};
    auto q = quaternion(result.camera_from_base);
    position.upload(p.data(), sizeof(p), stream_);
    rotation.upload(q.data(), sizeof(q), stream_);
    CRR_CUDA_CHECK(cudaMemsetAsync(state.get<void>(), 0, sizeof(CalibrationLmState), stream_));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUcontext context{};
    CRR_CU_CHECK(cuCtxGetCurrent(&context));
    uint64_t mesh_id =
        backend_->create_mesh(context, vertices.get<void>(), static_cast<int>(mesh.vertices.size()),
                              triangles.get<void>(), static_cast<int>(mesh.triangles.size()));
    try {
      // BVH creation uses Warp's context stream, which can differ from our
      // nonblocking stream. Complete it before capturing queries on our stream.
      backend_->synchronize(context);
      graph_run(stream_, [&] {
        for (int iteration = 0; iteration <= config_.surface_iterations; ++iteration) {
          CRR_CUDA_CHECK(cudaMemsetAsync(mask.get<void>(), 0, count * sizeof(int), stream_));
          CRR_CUDA_CHECK(cudaMemsetAsync(distances.get<void>(), 0, count * sizeof(float), stream_));
          CRR_CUDA_CHECK(cudaMemsetAsync(gradients.get<void>(), 0, count * sizeof(Point), stream_));
          CRR_CUDA_CHECK(cudaMemsetAsync(hessian.get<void>(), 0, 36 * sizeof(float), stream_));
          CRR_CUDA_CHECK(cudaMemsetAsync(gradient.get<void>(), 0, 6 * sizeof(float), stream_));
          CRR_CUDA_CHECK(cudaMemsetAsync(squared.get<void>(), 0, sizeof(float), stream_));
          CRR_CUDA_CHECK(cudaMemsetAsync(valid.get<void>(), 0, sizeof(int), stream_));
          auto bounds = warp_bounds_1d(count);
          auto pa =
              warp_array_1d(reinterpret_cast<uint64_t>(points.get<void>()), count, sizeof(Point));
          auto po =
              warp_array_1d(reinterpret_cast<uint64_t>(position.get<void>()), 3, sizeof(float));
          auto qa =
              warp_array_1d(reinterpret_cast<uint64_t>(rotation.get<void>()), 4, sizeof(float));
          auto da = warp_array_1d(reinterpret_cast<uint64_t>(distances.get<void>()), count,
                                  sizeof(float));
          auto ga = warp_array_1d(reinterpret_cast<uint64_t>(gradients.get<void>()), count,
                                  sizeof(Point));
          auto ma = warp_array_1d(reinterpret_cast<uint64_t>(mask.get<void>()), count, sizeof(int));
          float threshold = config_.surface_distance;
          query_.launch(
              count, stream_,
              {&bounds, &pa, &count, &po, &qa, &mesh_id, &threshold, &threshold, &da, &ga, &ma});
          auto ha =
              warp_array_1d(reinterpret_cast<uint64_t>(hessian.get<void>()), 36, sizeof(float));
          auto ja =
              warp_array_1d(reinterpret_cast<uint64_t>(gradient.get<void>()), 6, sizeof(float));
          auto sa =
              warp_array_1d(reinterpret_cast<uint64_t>(squared.get<void>()), 1, sizeof(float));
          auto va = warp_array_1d(reinterpret_cast<uint64_t>(valid.get<void>()), 1, sizeof(int));
          int huber = 1;
          float delta = config_.surface_huber_delta;
          reduce_.launch(count, stream_,
                         {&bounds, &pa, &da, &ga, &ma, &count, &huber, &delta, &ha, &ja, &sa, &va});
          launch_calibration_lm(hessian.get<float>(), gradient.get<float>(), squared.get<float>(),
                                valid.get<int>(), position.get<float>(), rotation.get<float>(),
                                state.get<CalibrationLmState>(), stream_);
        }
      });
      CalibrationLmState fitted{};
      CRR_CUDA_CHECK(
          cudaMemcpy(&fitted, state.get<void>(), sizeof(fitted), cudaMemcpyDeviceToHost));
      result.camera_from_base = quaternion_matrix(fitted.quaternion, fitted.position);
      result.valid_ratio = static_cast<float>(fitted.valid) / count;
      result.surface_iterations = fitted.iterations;
      result.surface_rms = std::sqrt(fitted.sum_squared / std::max(fitted.valid, 1));
      result.success = fitted.valid > 10 && result.valid_ratio >= config_.minimum_valid_ratio &&
                       std::isfinite(result.surface_rms);
      if (result.success) validate(result.camera_from_base);
    } catch (...) {
      cudaStreamSynchronize(stream_);
      backend_->destroy_mesh(mesh_id);
      throw;
    }
    backend_->destroy_mesh(mesh_id);
#endif
  }
  PoseCalibrationConfig config_;
  int device_{};
  cudaStream_t stream_{};
  Kernel query_, reduce_;
#ifdef CUROBO_ROBOT_HAS_WARP_MESH
  std::unique_ptr<detail::WarpMeshBackend> backend_;
#endif
};

PoseCalibration::PoseCalibration(const std::string& directory, PoseCalibrationConfig config)
    : impl_(std::make_unique<Impl>(directory, config)) {}
PoseCalibration::~PoseCalibration() = default;
PoseCalibrationResult PoseCalibration::detect(const CalibrationMesh& mesh,
                                              const std::vector<Point>& observed,
                                              const std::optional<Matrix>& seed) {
  return impl_->detect(mesh, observed, seed);
}
}  // namespace curobo::robot
