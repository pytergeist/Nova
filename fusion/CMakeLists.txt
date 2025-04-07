cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_PREFIX_PATH "/opt/homebrew")

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")
find_package(pybind11 REQUIRED)

pybind11_add_module(fusion_math src/Fusion/python/bindings.cpp)
target_include_directories(fusion_math PRIVATE src/Fusion/templates)

add_executable(FusionExecutable
        src/Fusion/core/main.cpp
        src/Fusion/core/tensor_ops.tpp
)
target_include_directories(FusionExecutable PRIVATE src/Fusion/core)

# ------------------- Google Test Setup -------------------
enable_testing()

set(GTest_DIR "/opt/homebrew/Cellar/googletest/1.11.0/lib/cmake/GTest")
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(FusionTests src/Fusion/tests/tensor_ops_tests.cpp)
target_include_directories(FusionTests PRIVATE src/Fusion/core)
target_link_libraries(FusionTests PRIVATE GTest::gtest GTest::gtest_main pthread)

add_test(NAME FusionTests COMMAND FusionTests)
