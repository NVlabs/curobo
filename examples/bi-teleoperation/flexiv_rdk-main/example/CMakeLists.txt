
cmake_minimum_required(VERSION 3.4)
project(flexiv_rdk-examples)

set(CMAKE_CXX_STANDARD 14)

message("OS: ${CMAKE_SYSTEM_NAME}")
message("Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
endif()

# Minumum example list for all OS
set(EXAMPLE_LIST
  basics1_display_robot_states
  basics2_clear_fault
  basics3_primitive_execution
  basics4_plan_execution
  basics5_zero_force_torque_sensors
  basics6_gripper_control
  basics7_auto_recovery
  intermediate7_teach_by_demonstration
)


# Additional examples for Linux and Mac
if(CMAKE_HOST_UNIX)
  list(APPEND EXAMPLE_LIST
    intermediate1_realtime_joint_position_control
    intermediate2_realtime_joint_torque_control
    intermediate3_realtime_joint_floating
    intermediate4_realtime_cartesian_pure_motion_control
    intermediate5_realtime_cartesian_motion_force_control
    intermediate6_robot_dynamics
  )
endif()

# Find flexiv_rdk INTERFACE library
find_package(flexiv_rdk REQUIRED)

# Build all selected examples
foreach(example ${EXAMPLE_LIST})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} flexiv::flexiv_rdk)
endforeach()
