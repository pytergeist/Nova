add_executable(fusion_core_tests
        test_dtype.cpp
        test_raw_tensor.cpp
        test_plan_meta.cpp
        test_dense_iter.cpp
        test_plan_builders_elementwise.cpp
        test_plan_builders_contraction.cpp
        test_plan_builders_reduction.cpp
)

target_include_directories(fusion_core_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_core_tests PRIVATE fusion_core
        fusion_storage
        fusion_alloc
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(fusion_core_tests)