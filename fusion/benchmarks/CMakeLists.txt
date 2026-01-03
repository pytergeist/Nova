cmake_minimum_required(VERSION 3.22)

if(NOT NOVA_USE_OPENBLAS)
    message(STATUS "Skipping benchmarks (BLAS disabled)")
    return()
endif()

# Provide nanobench if your top-level Dependencies didn’t already
include(FetchContent)
FetchContent_Declare(nanobench
        GIT_REPOSITORY https://github.com/martinus/nanobench.git
        GIT_TAG v4.3.11
)
FetchContent_MakeAvailable(nanobench)

add_subdirectory(eigen)
add_subdirectory(fusion)
