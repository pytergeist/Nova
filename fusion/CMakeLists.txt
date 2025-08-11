cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

#--------------------------------------------------------------------------------
# Optional policy tweak to suppress install() normalization warnings
cmake_policy(SET CMP0177 NEW)

#--------------------------------------------------------------------------------
# ———— FetchContent support for missing dependencies ————
include(FetchContent)

# —— xsimd (header-only) —— try system first, then pull from GitHub if needed
find_path(XSIMD_INCLUDE_DIR
        NAMES xsimd/xsimd.hpp
        HINTS /usr/include /usr/local/include
)
if (NOT XSIMD_INCLUDE_DIR)
    message(STATUS "xsimd not found on system; fetching via FetchContent")
    FetchContent_Declare(
            xsimd
            GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
            GIT_TAG        13.2.0
    )
    FetchContent_MakeAvailable(xsimd)
    set(XSIMD_INCLUDE_DIR ${xsimd_SOURCE_DIR}/include)
endif()

#--------------------------------------------------------------------------------
# Platform & Compiler settings
# Target Apple Silicon on macOS
set(CMAKE_OSX_ARCHITECTURES "arm64")

# C++20, no GNU extensions
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS       OFF)

# Enable all warnings
add_compile_options(-Wall)

# Allow Homebrew dependencies (macOS); no-op on Linux
set(CMAKE_PREFIX_PATH "/opt/homebrew" CACHE STRING "Prefix path for dependencies")

#--------------------------------------------------------------------------------
# pybind11
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")
find_package(pybind11 REQUIRED)

#--------------------------------------------------------------------------------
# OpenBLAS
find_library(OPENBLAS_LIB
        NAMES openblas
        PATHS /opt/homebrew/opt/openblas/lib /usr/lib /usr/local/lib
)
if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS library not found")
endif()
set(OPENBLAS_INCLUDE_DIR "/opt/homebrew/opt/openblas/include" CACHE PATH "OpenBLAS include dir")

#--------------------------------------------------------------------------------
# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# Eigen3_INCLUDE_DIRS provided by the package

#--------------------------------------------------------------------------------
# Python extension module (_C)
pybind11_add_module(fusion
        src/Fusion/python/bindings.cpp
)

target_sources(fusion PRIVATE
        src/Fusion/core/broadcast.cpp
)

# ---- Add NEON macro definition ----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    option(FUSION_ENABLE_NEON "Enable NEON SIMD optimizations" ON)
    if(FUSION_ENABLE_NEON)
        target_compile_definitions(fusion PRIVATE FUSION_ENABLE_NEON=1)
    endif()
endif()


target_include_directories(fusion PRIVATE
        ${PROJECT_SOURCE_DIR}/src/Fusion/cpu/simd
        ${OPENBLAS_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIRS}
        ${XSIMD_INCLUDE_DIR}
)


target_link_libraries(fusion PRIVATE
        ${OPENBLAS_LIB}
)

# Install the .so into your Python package directory
set(PY_EXT_INSTALL_DIR
        ${PROJECT_SOURCE_DIR}/../nova/src/backend/core/_C
)
file(MAKE_DIRECTORY ${PY_EXT_INSTALL_DIR})

set_target_properties(fusion PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}
)

install(TARGETS fusion
        LIBRARY DESTINATION ${PY_EXT_INSTALL_DIR}
)

#--------------------------------------------------------------------------------
# Native executable
add_executable(FusionExecutable
        src/Fusion/main.cpp
        src/Fusion/core/broadcast.cpp
)

target_include_directories(FusionExecutable PRIVATE
        ${PROJECT_SOURCE_DIR}/src/Fusion/tensor
        ${PROJECT_SOURCE_DIR}/src/Fusion/cpu/simd
        ${OPENBLAS_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIRS}
        ${XSIMD_INCLUDE_DIR}
)

target_link_libraries(FusionExecutable PRIVATE
        ${OPENBLAS_LIB}
)

#--------------------------------------------------------------------------------
## Tests (GoogleTest)
#enable_testing()
#
#set(GTest_DIR "/opt/homebrew/Cellar/googletest/1.11.0/lib/cmake/GTest" CACHE PATH "GTest CMake dir")
#find_package(GTest REQUIRED)
#
#add_executable(FusionTests
#        src/Fusion/tests/tensor_cpu_tests.cpp
#)
#
#target_include_directories(FusionTests PRIVATE
#        src/Fusion/tensor
#        ${OPENBLAS_INCLUDE_DIR}
#        ${Eigen3_INCLUDE_DIRS}
#        ${XSIMD_INCLUDE_DIR}
#)
#
#target_link_libraries(FusionTests PRIVATE
#        GTest::gtest_main
#        pthread
#        ${OPENBLAS_LIB}
#)
#
#add_test(NAME FusionTests COMMAND FusionTests)

#--------------------------------------------------------------------------------
# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build Fusion benchmarks" ON)

if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
