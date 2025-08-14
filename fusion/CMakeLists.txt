cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

#--------------------------------------------------------------------------------
include(FetchContent)

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
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS       OFF)

add_compile_options(-Wall -g)

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

#--------------------------------------------------------------------------------
# SLEEF (
find_package(SLEEF CONFIG QUIET)
if(NOT SLEEF_FOUND)
    message(STATUS "SLEEF not found; fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
            sleef
            GIT_REPOSITORY https://github.com/shibatch/sleef.git
            GIT_TAG        3.9.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
    )
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_DFT OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_QUAD OFF CACHE BOOL "" FORCE)
    set(SLEEF_DISABLE_GFNI ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(sleef)

    if(NOT TARGET sleef::sleef)
        add_library(sleef::sleef ALIAS sleef)
    endif()
endif()
#--------------------------------------------------------------------------------
# Python extension module (_C)
pybind11_add_module(fusion
        src/Fusion/python/bindings.cpp
)

target_sources(fusion PRIVATE
        src/Fusion/core/Broadcast.cpp
)

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
        sleef::sleef
        ${OPENBLAS_LIB}
)

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
add_executable(FusionExecutable
        src/Fusion/main.cpp
        src/Fusion/core/Broadcast.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(FUSION_ENABLE_NEON)
        target_compile_definitions(FusionExecutable PRIVATE FUSION_ENABLE_NEON=1)
    endif()
endif()

target_include_directories(FusionExecutable PRIVATE
        ${PROJECT_SOURCE_DIR}/src/Fusion/tensor
        ${PROJECT_SOURCE_DIR}/src/Fusion/cpu/simd
        ${OPENBLAS_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIRS}
        ${XSIMD_INCLUDE_DIR}
)

target_link_libraries(FusionExecutable PRIVATE
        sleef::sleef
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
