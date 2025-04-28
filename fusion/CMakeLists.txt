cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

#------------------------- Platform & Compiler -------------------------#
# Set target architecture for macOS (Apple Silicon)
set(CMAKE_OSX_ARCHITECTURES "arm64")

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Wall warnings
add_compile_options(-Wall)

# If you install via Homebrew, make sure CMake searches there
set(CMAKE_PREFIX_PATH "/opt/homebrew" CACHE STRING "Prefix path for dependencies")

#------------------------- pybind11 Setup -------------------------#
# Force pybind11 to use the CMake FindPython module
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")
find_package(pybind11 REQUIRED)

#------------------------- OpenBLAS -------------------------#
find_library(OPENBLAS_LIB openblas
        PATHS /opt/homebrew/opt/openblas/lib
)
if (NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS library not found in /opt/homebrew/opt/openblas/lib")
endif()
set(OPENBLAS_INCLUDE_DIR /opt/homebrew/opt/openblas/include)

#------------------------- Eigen3 -------------------------#
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# fallback:
# set(EIGEN3_INCLUDE_DIR "/opt/homebrew/include/eigen3")

#------------------------- Python Extension (_C) -------------------------#
# Build fusion_python as a Python extension
pybind11_add_module(fusion
        src/Fusion/python/bindings.cpp
)

# Include paths for both headers and templates
target_include_directories(fusion PRIVATE
        src/Fusion/templates
        ${OPENBLAS_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

# Link against OpenBLAS (Eigen is header-only)
target_link_libraries(fusion PRIVATE
        ${OPENBLAS_LIB}
)

# Make sure the output directory exists in your python package
# We want the .so to land in: ../nova/src/backend/core/_C
set(PY_EXT_INSTALL_DIR
        ${PROJECT_SOURCE_DIR}/../nova/src/backend/core/_C
)
file(MAKE_DIRECTORY ${PY_EXT_INSTALL_DIR})

# Override the target’s output so that the Python extension
# is placed into the _C folder
set_target_properties(fusion PROPERTIES
        PREFIX ""                                          # no lib prefix
        LIBRARY_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}     # .so/.dylib
        RUNTIME_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}     # for multi-config
        ARCHIVE_OUTPUT_DIRECTORY ${PY_EXT_INSTALL_DIR}
)

# Optional: install rule
install(TARGETS fusion
        LIBRARY DESTINATION ${PY_EXT_INSTALL_DIR}
)

#------------------------- Native Executable -------------------------#
add_executable(FusionExecutable
        src/Fusion/main.cpp
)

target_include_directories(FusionExecutable PRIVATE
        src/Fusion/tensor
        ${OPENBLAS_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(FusionExecutable PRIVATE
        ${OPENBLAS_LIB}
)

#------------------------- Tests (GoogleTest) -------------------------#
enable_testing()

# Adjust this path if your GTest is somewhere else
set(GTest_DIR "/opt/homebrew/Cellar/googletest/1.11.0/lib/cmake/GTest")
find_package(GTest REQUIRED)

add_executable(FusionTests
        src/Fusion/tests/tensor_ops_tests.cpp
)

target_include_directories(FusionTests PRIVATE
        src/Fusion/tensor
        ${OPENBLAS_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(FusionTests PRIVATE
        GTest::gtest
        GTest::gtest_main
        pthread
        ${OPENBLAS_LIB}
)

add_test(NAME FusionTests COMMAND FusionTests)
