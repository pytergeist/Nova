cmake_minimum_required(VERSION 3.22)
project(Nova LANGUAGES CXX)

# ------------------------------------------------------------
# Optional local modules (don't fail if missing)
# ------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions OPTIONAL RESULT_VARIABLE _po_included)
include(Dependencies  OPTIONAL RESULT_VARIABLE _dep_included)

# ------------------------------------------------------------
# Python (use the active venv if present)
# ------------------------------------------------------------
# Make pybind11 rely on CMake's FindPython, not its legacy finder.
set(PYBIND11_FINDPYTHON ON)

# If a venv/conda env is active, point CMake to that interpreter.
if(NOT DEFINED Python_EXECUTABLE)
  if(DEFINED ENV{VIRTUAL_ENV})
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
  elseif(DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
  endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python: ${Python_EXECUTABLE} (version ${Python_VERSION})")

find_package(pybind11 REQUIRED)

# ------------------------------------------------------------
# BLAS (Accelerate on macOS)
# ------------------------------------------------------------
find_package(BLAS REQUIRED)
message(STATUS "BLAS_LIBRARIES='${BLAS_LIBRARIES}'")

set(ACCELERATE_FRAMEWORK "")
if(APPLE)
  set(ACCELERATE_FRAMEWORK "-framework Accelerate")
endif()

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
set(FUSION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusion/src/Fusion")

# ------------------------------------------------------------
# Core C++ library (only non-kernel .cpps!)
# kernels/*.cpp are #included by headers; don't compile them twice.
# ------------------------------------------------------------
add_library(fusion_core STATIC
        ${FUSION_SRC_DIR}/core/Broadcast.cpp
)

set_target_properties(fusion_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(fusion_core
        PUBLIC
        ${FUSION_SRC_DIR}
        ${FUSION_SRC_DIR}/cpu/simd
)

# If sleef was provided by Dependencies.cmake, link it.
if(TARGET sleef::sleef)
  target_link_libraries(fusion_core PUBLIC sleef::sleef)
endif()

# Make sure <cblas.h> is visible on macOS SDKs (robust include path).
if(APPLE)
  if(NOT CMAKE_OSX_SYSROOT)
    execute_process(
            COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE _sdk
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  else()
    set(_sdk "${CMAKE_OSX_SYSROOT}")
  endif()
  if(EXISTS "${_sdk}/System/Library/Frameworks/Accelerate.framework/Headers")
    target_include_directories(fusion_core PUBLIC
            "${_sdk}/System/Library/Frameworks/Accelerate.framework/Headers"
            "${_sdk}/System/Library/Frameworks/vecLib.framework/Headers"
    )
  endif()
endif()

# Link BLAS to core (and force Accelerate if needed).
target_link_libraries(fusion_core PUBLIC ${BLAS_LIBRARIES} ${ACCELERATE_FRAMEWORK})

# Optional NEON toggles on ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
  option(FUSION_ENABLE_NEON "Enable NEON SIMD optimizations" ON)
  if(FUSION_ENABLE_NEON)
    target_compile_definitions(fusion_core PUBLIC FUSION_ENABLE_NEON=1)
  endif()
endif()

# ------------------------------------------------------------
# Python extension module
# ------------------------------------------------------------
pybind11_add_module(fusion
        ${FUSION_SRC_DIR}/python/Bindings.cpp
)

target_link_libraries(fusion PRIVATE
        fusion_core
        ${BLAS_LIBRARIES} ${ACCELERATE_FRAMEWORK}
)

# Drop the built module into nova/src/backend/core/clib/
set(PY_EXT_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nova/src/backend/core/clib)
file(MAKE_DIRECTORY ${PY_EXT_INSTALL_DIR})
set_target_properties(fusion PROPERTIES
        PREFIX ""  # produce fusion.cpython-*.so
        LIBRARY_OUTPUT_DIRECTORY  ${PY_EXT_INSTALL_DIR}
        RUNTIME_OUTPUT_DIRECTORY  ${PY_EXT_INSTALL_DIR}
        ARCHIVE_OUTPUT_DIRECTORY  ${PY_EXT_INSTALL_DIR}
)
install(TARGETS fusion LIBRARY DESTINATION ${PY_EXT_INSTALL_DIR})

# ------------------------------------------------------------
# Optional native example
# ------------------------------------------------------------
add_executable(FusionExecutable
        ${FUSION_SRC_DIR}/main.cpp
)
target_link_libraries(FusionExecutable PRIVATE
        fusion_core
        ${BLAS_LIBRARIES} ${ACCELERATE_FRAMEWORK}
)

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
add_subdirectory(fusion/benchmarks)
