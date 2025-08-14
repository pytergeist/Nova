find_package(BLAS REQUIRED)

add_executable(FusionBenchMark FusionBenchmark.cpp ${PROJECT_SOURCE_DIR}/src/Fusion/core/Broadcast.cpp)

target_include_directories(FusionBenchMark PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/Fusion/cpu/simd
        ${OPENBLAS_INCLUDE_DIR}
        ${XSIMD_INCLUDE_DIR}
)


target_link_libraries(FusionBenchMark PRIVATE
        nanobench
        sleef::sleef
        ${OPENBLAS_LIB}
)
