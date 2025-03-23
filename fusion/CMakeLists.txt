cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

# Specify the target architecture (Apple Silicon)
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")

find_package(pybind11 REQUIRED)

pybind11_add_module(fusion src/Fusion/python/bindings.cpp)
target_include_directories(fusion PRIVATE src/Fusion/templates)

add_executable(FusionExecutable src/Fusion/core/main.cpp src/Fusion/core/tensor.tpp src/Fusion/core/tensor.tpp)
target_include_directories(FusionExecutable PRIVATE src/Fusion/core)
