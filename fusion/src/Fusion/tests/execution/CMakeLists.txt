add_executable(fusion_execution_tests
        test_binary_elementwise.cpp
        test_contraction.cpp
        test_reduction.cpp
        test_unary_elementwise.cpp
)

target_include_directories(fusion_execution_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_execution_tests PRIVATE fusion_execution
        fusion_core
        fusion_storage
        fusion_alloc
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(fusion_execution_tests)