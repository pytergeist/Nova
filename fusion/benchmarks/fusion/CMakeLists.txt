find_package(BLAS REQUIRED)

add_executable(FusionBenchMark fusion_benchmark.cpp)

target_include_directories(FusionBenchMark PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${OPENBLAS_INCLUDE_DIR}
        ${XSIMD_INCLUDE_DIR}
)


target_link_libraries(FusionBenchMark PRIVATE
        nanobench
        ${OPENBLAS_LIB}
)
