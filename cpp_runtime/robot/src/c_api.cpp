// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "robot_runtime.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

using curobo_robot_runtime::RobotRuntimeImpl;

struct CuroboRobotRuntime {
  std::unique_ptr<RobotRuntimeImpl> impl;
};

namespace {

thread_local std::string g_last_error;

int fail(const std::exception& exception) {
  g_last_error = exception.what();
  return -1;
}

int fail_message(const char* message) {
  g_last_error = message;
  return -1;
}

template <typename Function>
int wrap(Function&& function) {
  try {
    function();
    g_last_error.clear();
    return 0;
  } catch (const std::exception& exception) {
    return fail(exception);
  } catch (...) {
    return fail_message("unknown C++ exception");
  }
}

RobotRuntimeImpl& checked(CuroboRobotRuntime* runtime) {
  if (runtime == nullptr || runtime->impl == nullptr) {
    throw std::runtime_error("CuroboRobotRuntime is null");
  }
  return *runtime->impl;
}

const char* name_or_error(CuroboRobotRuntime* runtime, int index, bool tool_frame) {
  try {
    const auto& name = tool_frame ? checked(runtime).tool_frame_name(index)
                                  : checked(runtime).joint_name(index);
    g_last_error.clear();
    return name.c_str();
  } catch (const std::exception& exception) {
    fail(exception);
    return nullptr;
  } catch (...) {
    fail_message("unknown C++ exception");
    return nullptr;
  }
}

}  // namespace

extern "C" {

const char* curobo_robot_last_error(void) { return g_last_error.c_str(); }

int curobo_robot_create(const char* artifact_dir, int device_ordinal,
                        CuroboRobotRuntime** runtime) {
  if (runtime == nullptr) {
    return fail_message("runtime output pointer is null");
  }
  *runtime = nullptr;
  return wrap([&] {
    if (artifact_dir == nullptr) {
      throw std::runtime_error("artifact_dir is null");
    }
    auto holder = std::make_unique<CuroboRobotRuntime>();
    holder->impl = std::make_unique<RobotRuntimeImpl>(artifact_dir, device_ordinal);
    *runtime = holder.release();
  });
}

void curobo_robot_destroy(CuroboRobotRuntime* runtime) { delete runtime; }

int curobo_robot_get_info(CuroboRobotRuntime* runtime, CuroboRobotInfo* info) {
  return wrap([&] {
    if (info == nullptr) {
      throw std::runtime_error("info output pointer is null");
    }
    *info = checked(runtime).info();
  });
}

const char* curobo_robot_joint_name(CuroboRobotRuntime* runtime, int joint_index) {
  return name_or_error(runtime, joint_index, false);
}

const char* curobo_robot_tool_frame_name(CuroboRobotRuntime* runtime,
                                         int tool_frame_index) {
  return name_or_error(runtime, tool_frame_index, true);
}

int curobo_robot_compute_kinematics(
    CuroboRobotRuntime* runtime, const CuroboRobotKinematicsRequest* request) {
  return wrap([&] {
    if (request == nullptr) {
      throw std::runtime_error("kinematics request is null");
    }
    checked(runtime).compute_kinematics(*request);
  });
}

}  // extern "C"
