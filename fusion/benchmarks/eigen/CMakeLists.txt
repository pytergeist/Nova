find_package(BLAS REQUIRED)

add_executable(EigenBenchMark eigen_benchmark.cpp)

target_include_directories(EigenBenchMark PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${OPENBLAS_INCLUDE_DIR}
        ${XSIMD_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIRS}
)


target_link_libraries(EigenBenchMark PRIVATE
        nanobench
        Eigen3::Eigen
        ${OPENBLAS_LIB}
)
