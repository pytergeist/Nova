cmake_minimum_required(VERSION 3.22)
project(Nova LANGUAGES CXX)

# ------------------------------------------------------------
# Project config
# ------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions OPTIONAL)
include(Dependencies OPTIONAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# ASAN build
# ------------------------------------------------------------
option(NOVA_ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(NOVA_ENABLE_ASAN)
  message(STATUS "ASAN enabled")

  # REQUIRED: ASAN + LTO do not mix on macOS
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

  add_compile_options(
          -fsanitize=address
          -fno-omit-frame-pointer
          -g
  )

  add_link_options(
          -fsanitize=address
  )
endif()


# ------------------------------------------------------------
# Python / pybind11
# ------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)

if(NOT DEFINED Python_EXECUTABLE)
  if(DEFINED ENV{VIRTUAL_ENV})
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
  elseif(DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
  endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

message(STATUS "Using Python: ${Python_EXECUTABLE} (${Python_VERSION})")

# ------------------------------------------------------------
# BLAS / Accelerate
# ------------------------------------------------------------
find_package(BLAS REQUIRED)
set(ACCELERATE_FRAMEWORK "")
if(APPLE)
  set(ACCELERATE_FRAMEWORK "-framework Accelerate")
endif()

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
set(FUSION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusion/src/Fusion")
set(FUSION_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fusion/src")

# ============================================================
# Layered Architecture
# ============================================================

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------
# allocation layer
# ------------------------------------------------------------
add_library(fusion_alloc STATIC
        ${FUSION_SRC_DIR}/alloc/BFCPoolAllocator.cpp
        ${FUSION_SRC_DIR}/alloc/CPUSubAllocator.cpp
)

target_include_directories(fusion_alloc PUBLIC ${FUSION_INCLUDE_ROOT})

# ------------------------------------------------------------
# device abstraction layer
# ------------------------------------------------------------
add_library(fusion_device INTERFACE)
target_include_directories(fusion_device INTERFACE ${FUSION_INCLUDE_ROOT})

# ------------------------------------------------------------
# storage abstraction layer
# ------------------------------------------------------------
add_library(fusion_storage INTERFACE)
target_include_directories(fusion_storage INTERFACE ${FUSION_INCLUDE_ROOT})
target_link_libraries(fusion_storage INTERFACE
        fusion_alloc
        fusion_device
)
# ------------------------------------------------------------
# core layer (Tensor semantics)
# ------------------------------------------------------------
add_library(fusion_core STATIC
        ${FUSION_SRC_DIR}/core/TensorPlan.cpp
        ${FUSION_SRC_DIR}/core/ThreadPool.cpp
)

set_target_properties(fusion_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(fusion_core
        PUBLIC
        ${FUSION_INCLUDE_ROOT}
        PRIVATE
        ${FUSION_SRC_DIR}/cpu/simd
)

target_link_libraries(fusion_core PUBLIC
        fusion_storage
        fusion_device
        fusion_alloc
        ${BLAS_LIBRARIES}
        ${ACCELERATE_FRAMEWORK}
        OpenBLAS::OpenBLAS
#        Eigen3::Eigen
)

if(TARGET sleef::sleef)
  target_link_libraries(fusion_core PRIVATE sleef::sleef)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
  target_compile_definitions(fusion_core PRIVATE FUSION_ENABLE_NEON=1)
endif()


# ------------------------------------------------------------
# ops
# ------------------------------------------------------------
add_library(fusion_ops INTERFACE)

target_include_directories(fusion_ops INTERFACE ${FUSION_INCLUDE_ROOT})
target_link_libraries(fusion_ops INTERFACE fusion_core)

# ------------------------------------------------------------
# autodiff
# ------------------------------------------------------------
add_library(fusion_autodiff INTERFACE)

target_include_directories(fusion_autodiff INTERFACE ${FUSION_INCLUDE_ROOT})
target_link_libraries(fusion_autodiff INTERFACE
        fusion_ops
        fusion_core
)


# ------------------------------------------------------------
# physics
# ------------------------------------------------------------
add_library(fusion_physics INTERFACE)

target_include_directories(fusion_physics INTERFACE ${FUSION_INCLUDE_ROOT})
target_link_libraries(fusion_physics INTERFACE
        fusion_autodiff
        fusion_ops
        fusion_core
)

# ============================================================
# Python extension
# ============================================================
set(PY_EXT_INSTALL_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/nova/src/backend/core/clib
)
file(MAKE_DIRECTORY ${PY_EXT_INSTALL_DIR})

pybind11_add_module(fusion
        ${FUSION_SRC_DIR}/python/Bindings.cpp
)

target_link_libraries(fusion PRIVATE
        fusion_physics
        fusion_autodiff
        fusion_ops
        fusion_core
        fusion_storage
        fusion_device
        fusion_alloc
        ${BLAS_LIBRARIES}
        ${ACCELERATE_FRAMEWORK}
)

set_target_properties(fusion PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}
)


install(TARGETS fusion LIBRARY DESTINATION ${PY_EXT_INSTALL_DIR})

# ============================================================
# Optional CLI
# ============================================================
option(NOVA_BUILD_CLI "Build CLI FusionExecutable" OFF)

if(NOVA_BUILD_CLI)
  add_executable(FusionExecutable
          ${FUSION_SRC_DIR}/main.cpp
  )
  target_link_libraries(FusionExecutable PRIVATE fusion_core)
endif()

## ============================================================
## Benchmarks
## ============================================================
#add_subdirectory(fusion/benchmarks)

# ============================================================
# Tests
# ============================================================
#option(NOVA_BUILD_TEST "Build tests" OFF)
#
#if(NOVA_BUILD_TEST)
#  enable_testing()
#  add_executable(fusion_test
#          ${FUSION_SRC_DIR}/tests/tensor/cpu/operations.cpp
#  )
#  target_link_libraries(fusion_test PRIVATE
#          fusion_core
#          GTest::gtest
#          GTest::gtest_main
#  )
#  include(GoogleTest)
#  gtest_discover_tests(fusion_test)
#endif()
