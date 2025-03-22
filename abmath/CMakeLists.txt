cmake_minimum_required(VERSION 3.16)
project(AbMath LANGUAGES CXX)

# Specify the target architecture (Apple Silicon)
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")

find_package(pybind11 REQUIRED)

pybind11_add_module(abmath src/AbMath/python/bindings.cpp)
target_include_directories(abmath PRIVATE src/AbMath/templates)

add_executable(AbMathExecutable main.cpp)
target_include_directories(AbMathExecutable PRIVATE src/AbMath/templates)
add_executable(AbMathExecutable tensor.cpp)
target_include_directories(AbMathExecutable PRIVATE src/AbMath/core)
