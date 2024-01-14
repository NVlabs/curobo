cmake_minimum_required(VERSION 3.4)

# ===================================================================
#      PROJECT CONFIG
# ===================================================================
project(flexiv_rdk VERSION 0.8.0)

# C++14 required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
endif()

# Set static library
message("OS: ${CMAKE_SYSTEM_NAME}")
message("Processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    set(RDK_STATIC_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/libFlexivRdk.x86_64-linux-gnu.a")
  elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(RDK_STATIC_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/libFlexivRdk.aarch64-linux-gnu.a")
  else()
    message(FATAL_ERROR "Linux with ${CMAKE_SYSTEM_PROCESSOR} processor is currently not supported.")
  endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64")
    set(RDK_STATIC_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/libFlexivRdk.arm64-darwin.a")
  else()
    message(FATAL_ERROR "Mac with ${CMAKE_SYSTEM_PROCESSOR} processor is currently not supported.")
  endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "AMD64")
    set(RDK_STATIC_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/FlexivRdk.win_amd64.lib")
  else()
    message(FATAL_ERROR "Windows with ${CMAKE_SYSTEM_PROCESSOR} processor is currently not supported.")
  endif()
endif()

# ===================================================================
#      PROJECT DEPENDENCIES
# ===================================================================
# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ===================================================================
#      PROJECT LIBRARIES
# ===================================================================
# Create an INTERFACE library with no source file to compile
add_library(${PROJECT_NAME} INTERFACE)

# Create an alias of the library using flexiv namespace, 
# to imitate the install target which uses flexiv namespace.
add_library(flexiv::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} INTERFACE
  ${RDK_STATIC_LIBRARY}
  Threads::Threads
)

# Use moderate compiler warning option 
if(MSVC)
  target_compile_options(${PROJECT_NAME} INTERFACE /W1)
else()
  target_compile_options(${PROJECT_NAME} INTERFACE -Wall -Wextra)
endif()

# Install the INTERFACE library
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlexivInstallLibrary.cmake)
FlexivInstallLibrary()

# Also install Eigen headers
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen3/Eigen
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
)
