cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

# Set target architecture for macOS (Apple Silicon)
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Set C++ standard and compiler flags.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Set the CMAKE_PREFIX_PATH so CMake searches Homebrew directories.
set(CMAKE_PREFIX_PATH "/opt/homebrew")

# ------------------- pybind11 Setup -------------------
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")
find_package(pybind11 REQUIRED)

# ------------------- Find OpenBLAS -------------------
find_library(OPENBLAS_LIB openblas PATHS /opt/homebrew/opt/openblas/lib)
if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS library not found in /opt/homebrew/opt/openblas/lib")
endif()
set(OPENBLAS_INCLUDE_DIR /opt/homebrew/opt/openblas/include)

# ------------------- Find Eigen3 -------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# If find_package fails, you can instead manually add:
# set(EIGEN3_INCLUDE_DIR "/opt/homebrew/include/eigen3")

# ------------------- pybind11 Module -------------------
pybind11_add_module(fusion_math src/Fusion/python/bindings.cpp)
target_include_directories(fusion_math PRIVATE
        src/Fusion/templates
        ${OPENBLAS_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}  # Or directly use /opt/homebrew/include/eigen3 if not found by find_package
)
target_link_libraries(fusion_math PRIVATE ${OPENBLAS_LIB}
        # Alternatively, if using find_package: Eigen3::Eigen
)

# ------------------- Fusion Executable -------------------
add_executable(FusionExecutable src/Fusion/main.cpp)
target_include_directories(FusionExecutable PRIVATE
        src/Fusion/tensor
        ${OPENBLAS_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}  # Add Eigen include path here as well
)
target_link_libraries(FusionExecutable PRIVATE ${OPENBLAS_LIB})

# ------------------- Google Test Setup -------------------
enable_testing()

set(GTest_DIR "/opt/homebrew/Cellar/googletest/1.11.0/lib/cmake/GTest")
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(FusionTests src/Fusion/tests/tensor_ops_tests.cpp)
target_include_directories(FusionTests PRIVATE
        src/Fusion/tensor
        ${OPENBLAS_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}  # And here if tests use Eigen as well
)
target_link_libraries(FusionTests PRIVATE
        GTest::gtest
        GTest::gtest_main
        pthread
        ${OPENBLAS_LIB}
)

add_test(NAME FusionTests COMMAND FusionTests)
