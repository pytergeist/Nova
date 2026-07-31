add_executable(fusion_tensor_cpu_tests
        operations.cpp
)

target_include_directories(fusion_tensor_cpu_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_tensor_cpu_tests PRIVATE
        fusion_core
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(fusion_tensor_cpu_tests)