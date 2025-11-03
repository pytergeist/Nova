cmake_minimum_required(VERSION 3.22)

add_executable(FusionBenchMark
        ${CMAKE_CURRENT_SOURCE_DIR}/FusionBenchmark.cpp
)

# Link core + nanobench + BLAS explicitly (ensures Accelerate appears)
target_link_libraries(FusionBenchMark PRIVATE
        fusion_core
        nanobench
        ${BLAS_LIBRARIES}
)

set_property(TARGET FusionBenchMark PROPERTY CXX_CLANG_TIDY "")
