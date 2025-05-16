cmake_minimum_required(VERSION 3.16)
project(Fusion LANGUAGES CXX)

#------------------------- Platform & Compiler -------------------------#
# Target Apple Silicon on macOS
set(CMAKE_OSX_ARCHITECTURES "arm64")

# C++20, no GNU extensions
set(CMAKE_CXX_STANDARD           20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# Enable all warnings
add_compile_options(-Wall)

# Allow Homebrew dependencies
set(CMAKE_PREFIX_PATH "/opt/homebrew" CACHE STRING "Prefix path for dependencies")

#------------------------- xsimd (header-only) -------------------------#
# Point at your Spack‐installed xsimd headers
set(XSIMD_INCLUDE_DIR
        "/opt/homebrew/Cellar/spack/0.23.1/opt/spack/darwin-sequoia-m1/apple-clang-17.0.0/xsimd-8.1.0-s3yjkaw2c7iwa6wpqkcofp5fgzk6xhsw/include"
)

#------------------------- pybind11 Setup -------------------------#
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")
find_package(pybind11 REQUIRED)

#------------------------- OpenBLAS -------------------------#
find_library(OPENBLAS_LIB
        NAMES openblas
        PATHS /opt/homebrew/opt/openblas/lib
)
if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS library not found in /opt/homebrew/opt/openblas/lib")
endif()
set(OPENBLAS_INCLUDE_DIR "/opt/homebrew/opt/openblas/include")

#------------------------- Eigen3 -------------------------#
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# Eigen3_INCLUDE_DIRS is provided by the config

#------------------------- Python Extension (_C) -------------------------#
pybind11_add_module(fusion
        src/Fusion/python/bindings.cpp
)

target_include_directories(fusion PRIVATE
        src/Fusion/templates
        ${OPENBLAS_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${XSIMD_INCLUDE_DIR}
)

target_link_libraries(fusion PRIVATE
        ${OPENBLAS_LIB}
)

# Place the .so into your python package
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

#------------------------- Native Executable -------------------------#
add_executable(FusionExecutable
        src/Fusion/main.cpp
)

target_include_directories(FusionExecutable PRIVATE
        src/Fusion/tensor
        ${OPENBLAS_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${XSIMD_INCLUDE_DIR}
)

target_link_libraries(FusionExecutable PRIVATE
        ${OPENBLAS_LIB}
)

##------------------------- Tests (GoogleTest) -------------------------#
#enable_testing()
#
#set(GTest_DIR "/opt/homebrew/Cellar/googletest/1.11.0/lib/cmake/GTest")
#find_package(GTest REQUIRED)
#
#add_executable(FusionTests
#        src/Fusion/tests/tensor_ops_tests.cpp
#)
#
#target_include_directories(FusionTests PRIVATE
#        src/Fusion/tensor
#        ${OPENBLAS_INCLUDE_DIR}
#        ${Eigen3_INCLUDE_DIR}
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
