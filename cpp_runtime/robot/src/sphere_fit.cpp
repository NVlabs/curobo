// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/sphere_fit.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace curobo::robot {
namespace {

constexpr double GEOMETRY_EPSILON = 1.0E-12;
constexpr size_t MAX_GRID_POINTS = 50'000'000;

struct Vector3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

Vector3 operator+(const Vector3& left, const Vector3& right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vector3 operator-(const Vector3& left, const Vector3& right) {
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vector3 operator*(const Vector3& value, double scale) {
  return {value.x * scale, value.y * scale, value.z * scale};
}

double dot(const Vector3& left, const Vector3& right) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vector3 cross(const Vector3& left, const Vector3& right) {
  return {
      left.y * right.z - left.z * right.y,
      left.z * right.x - left.x * right.z,
      left.x * right.y - left.y * right.x,
  };
}

double squared_norm(const Vector3& value) { return dot(value, value); }

Vector3 vertex(const Mesh& mesh, int32_t index) {
  const auto& value = mesh.vertices.at(static_cast<size_t>(index));
  return {value.at(0), value.at(1), value.at(2)};
}

uint64_t edge_key(int32_t first, int32_t second) {
  const auto low = static_cast<uint32_t>(std::min(first, second));
  const auto high = static_cast<uint32_t>(std::max(first, second));
  return (static_cast<uint64_t>(low) << 32U) | high;
}

void validate_mesh(const Mesh& mesh) {
  if (mesh.vertices.size() < 4 || mesh.triangles.size() < 4) {
    throw std::invalid_argument(
        "sphere fitting requires a closed mesh with at least four vertices and triangles");
  }
  for (const auto& point : mesh.vertices) {
    for (const auto coordinate : point) {
      if (!std::isfinite(coordinate)) {
        throw std::invalid_argument("mesh vertices must be finite");
      }
    }
  }

  std::unordered_map<uint64_t, int> edge_counts;
  edge_counts.reserve(mesh.triangles.size() * 3);
  for (const auto& triangle : mesh.triangles) {
    for (const auto index : triangle) {
      if (index < 0 || static_cast<size_t>(index) >= mesh.vertices.size()) {
        throw std::invalid_argument("mesh triangle index is out of range");
      }
    }
    if (triangle.at(0) == triangle.at(1) ||
        triangle.at(1) == triangle.at(2) ||
        triangle.at(2) == triangle.at(0)) {
      throw std::invalid_argument("mesh contains a degenerate triangle");
    }
    const auto first = vertex(mesh, triangle.at(0));
    const auto second = vertex(mesh, triangle.at(1));
    const auto third = vertex(mesh, triangle.at(2));
    if (squared_norm(cross(second - first, third - first)) <=
        GEOMETRY_EPSILON) {
      throw std::invalid_argument("mesh contains a zero-area triangle");
    }
    ++edge_counts[edge_key(triangle.at(0), triangle.at(1))];
    ++edge_counts[edge_key(triangle.at(1), triangle.at(2))];
    ++edge_counts[edge_key(triangle.at(2), triangle.at(0))];
  }
  for (const auto& [unused_edge, count] : edge_counts) {
    static_cast<void>(unused_edge);
    if (count != 2) {
      throw std::invalid_argument("voxel sphere fitting requires a watertight mesh");
    }
  }
}

double point_triangle_squared_distance(const Vector3& point,
                                       const Vector3& first,
                                       const Vector3& second,
                                       const Vector3& third) {
  const auto first_second = second - first;
  const auto first_third = third - first;
  const auto first_point = point - first;
  const double d1 = dot(first_second, first_point);
  const double d2 = dot(first_third, first_point);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return squared_norm(first_point);
  }

  const auto second_point = point - second;
  const double d3 = dot(first_second, second_point);
  const double d4 = dot(first_third, second_point);
  if (d3 >= 0.0 && d4 <= d3) {
    return squared_norm(second_point);
  }

  const double first_edge_area = d1 * d4 - d3 * d2;
  if (first_edge_area <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double weight = d1 / (d1 - d3);
    return squared_norm(point - (first + first_second * weight));
  }

  const auto third_point = point - third;
  const double d5 = dot(first_second, third_point);
  const double d6 = dot(first_third, third_point);
  if (d6 >= 0.0 && d5 <= d6) {
    return squared_norm(third_point);
  }

  const double second_edge_area = d5 * d2 - d1 * d6;
  if (second_edge_area <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double weight = d2 / (d2 - d6);
    return squared_norm(point - (first + first_third * weight));
  }

  const double third_edge_area = d3 * d6 - d5 * d4;
  if (third_edge_area <= 0.0 && (d4 - d3) >= 0.0 &&
      (d5 - d6) >= 0.0) {
    const double weight = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return squared_norm(point - (second + (third - second) * weight));
  }

  const double denominator = 1.0 /
                             (first_edge_area + second_edge_area +
                              third_edge_area);
  const double second_weight = second_edge_area * denominator;
  const double third_weight = third_edge_area * denominator;
  const auto projection =
      first + first_second * second_weight + first_third * third_weight;
  return squared_norm(point - projection);
}

bool ray_triangle_intersection(const Vector3& origin, const Vector3& direction,
                               const Vector3& first, const Vector3& second,
                               const Vector3& third, double* distance) {
  const auto first_second = second - first;
  const auto first_third = third - first;
  const auto cross_direction = cross(direction, first_third);
  const double determinant = dot(first_second, cross_direction);
  if (std::abs(determinant) <= GEOMETRY_EPSILON) {
    return false;
  }
  const double inverse_determinant = 1.0 / determinant;
  const auto origin_offset = origin - first;
  const double first_coordinate =
      dot(origin_offset, cross_direction) * inverse_determinant;
  constexpr double BARYCENTRIC_EPSILON = 1.0E-10;
  if (first_coordinate < -BARYCENTRIC_EPSILON ||
      first_coordinate > 1.0 + BARYCENTRIC_EPSILON) {
    return false;
  }
  const auto second_cross = cross(origin_offset, first_second);
  const double second_coordinate =
      dot(direction, second_cross) * inverse_determinant;
  if (second_coordinate < -BARYCENTRIC_EPSILON ||
      first_coordinate + second_coordinate > 1.0 + BARYCENTRIC_EPSILON) {
    return false;
  }
  const double hit_distance =
      dot(first_third, second_cross) * inverse_determinant;
  if (hit_distance <= BARYCENTRIC_EPSILON) {
    return false;
  }
  *distance = hit_distance;
  return true;
}

bool is_inside(const Mesh& mesh, const Vector3& point) {
  // The non-axis-aligned ray avoids systematic coincidences with voxel grids.
  constexpr Vector3 RAY_DIRECTION{1.0, 0.3713906763541037,
                                  0.1298734321456901};
  std::vector<double> intersections;
  intersections.reserve(mesh.triangles.size() / 2);
  for (const auto& triangle : mesh.triangles) {
    double distance = 0.0;
    if (ray_triangle_intersection(
            point, RAY_DIRECTION, vertex(mesh, triangle.at(0)),
            vertex(mesh, triangle.at(1)), vertex(mesh, triangle.at(2)),
            &distance)) {
      intersections.push_back(distance);
    }
  }
  std::sort(intersections.begin(), intersections.end());
  size_t unique_hits = 0;
  double previous = -std::numeric_limits<double>::infinity();
  for (const double distance : intersections) {
    const double tolerance = 1.0E-9 * std::max(1.0, std::abs(distance));
    if (unique_hits == 0 || std::abs(distance - previous) > tolerance) {
      ++unique_hits;
      previous = distance;
    }
  }
  return unique_hits % 2 == 1;
}

double mesh_clearance(const Mesh& mesh, const Vector3& point) {
  double squared_distance = std::numeric_limits<double>::infinity();
  for (const auto& triangle : mesh.triangles) {
    squared_distance = std::min(
        squared_distance,
        point_triangle_squared_distance(
            point, vertex(mesh, triangle.at(0)),
            vertex(mesh, triangle.at(1)), vertex(mesh, triangle.at(2))));
  }
  return std::sqrt(std::max(0.0, squared_distance));
}

size_t grid_size(int x, int y, int z) {
  const auto x_size = static_cast<size_t>(x);
  const auto y_size = static_cast<size_t>(y);
  const auto z_size = static_cast<size_t>(z);
  if (y_size != 0 && x_size > MAX_GRID_POINTS / y_size) {
    throw std::invalid_argument("sphere fitting grid is too large");
  }
  const auto xy = x_size * y_size;
  if (z_size != 0 && xy > MAX_GRID_POINTS / z_size) {
    throw std::invalid_argument("sphere fitting grid is too large");
  }
  const auto total = xy * z_size;
  if (total > MAX_GRID_POINTS) {
    throw std::invalid_argument("sphere fitting grid is too large");
  }
  return total;
}

}  // namespace

SphereFitResult fit_spheres_to_mesh(const Mesh& mesh,
                                    const SphereFitConfig& config) {
  if (config.num_spheres <= 0) {
    throw std::invalid_argument("num_spheres must be positive");
  }
  if (!std::isfinite(config.minimum_radius) ||
      config.minimum_radius < 0.0F) {
    throw std::invalid_argument("minimum_radius must be finite and non-negative");
  }
  validate_mesh(mesh);

  Vector3 lower = vertex(mesh, 0);
  Vector3 upper = lower;
  for (const auto& value : mesh.vertices) {
    lower.x = std::min(lower.x, static_cast<double>(value.at(0)));
    lower.y = std::min(lower.y, static_cast<double>(value.at(1)));
    lower.z = std::min(lower.z, static_cast<double>(value.at(2)));
    upper.x = std::max(upper.x, static_cast<double>(value.at(0)));
    upper.y = std::max(upper.y, static_cast<double>(value.at(1)));
    upper.z = std::max(upper.z, static_cast<double>(value.at(2)));
  }
  const auto extents = upper - lower;
  const double volume = extents.x * extents.y * extents.z;
  if (!(volume > 0.0) || !std::isfinite(volume)) {
    throw std::invalid_argument("mesh bounding box must have positive volume");
  }
  const double pitch =
      std::cbrt(volume / static_cast<double>(config.num_spheres));
  const auto axis_count = [pitch](double extent) {
    constexpr double INTEGER_TOLERANCE = 1.0E-12;
    return std::max(
        1, static_cast<int>(std::ceil(extent / pitch - INTEGER_TOLERANCE)));
  };
  const int x_count = axis_count(extents.x);
  const int y_count = axis_count(extents.y);
  const int z_count = axis_count(extents.z);
  const auto point_count = grid_size(x_count, y_count, z_count);

  SphereFitResult result;
  result.voxel_pitch = static_cast<float>(pitch);
  result.spheres.reserve(
      std::min(point_count, static_cast<size_t>(config.num_spheres)));
  const Vector3 cell_size{
      extents.x / static_cast<double>(x_count),
      extents.y / static_cast<double>(y_count),
      extents.z / static_cast<double>(z_count),
  };
  for (int x = 0; x < x_count; ++x) {
    for (int y = 0; y < y_count; ++y) {
      for (int z = 0; z < z_count; ++z) {
        const Vector3 point{
            lower.x + (static_cast<double>(x) + 0.5) * cell_size.x,
            lower.y + (static_cast<double>(y) + 0.5) * cell_size.y,
            lower.z + (static_cast<double>(z) + 0.5) * cell_size.z,
        };
        if (!is_inside(mesh, point)) {
          continue;
        }
        const auto radius = static_cast<float>(mesh_clearance(mesh, point));
        if (radius <= config.minimum_radius) {
          continue;
        }
        result.spheres.push_back({
            {static_cast<float>(point.x), static_cast<float>(point.y),
             static_cast<float>(point.z)},
            radius,
        });
      }
    }
  }
  result.candidate_count = static_cast<int>(result.spheres.size());
  std::sort(result.spheres.begin(), result.spheres.end(),
            [](const CollisionSphere& left, const CollisionSphere& right) {
              if (left.radius != right.radius) {
                return left.radius > right.radius;
              }
              return left.center < right.center;
            });
  if (result.spheres.size() > static_cast<size_t>(config.num_spheres)) {
    result.spheres.resize(static_cast<size_t>(config.num_spheres));
  }
  if (result.spheres.empty()) {
    throw std::runtime_error(
        "voxel sphere fitting found no interior centers; use a closed volume or a finer fit");
  }
  return result;
}

}  // namespace curobo::robot
