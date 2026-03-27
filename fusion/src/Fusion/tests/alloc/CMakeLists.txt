add_executable(fusion_alloc_tests
        test_region_manager.cpp
        test_pool_allocator_basic.cpp
        test_pool_allocator_buckets.cpp
        test_pool_allocator_split_merge.cpp
        test_pool_allocator_fuzz.cpp
)

target_include_directories(fusion_alloc_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_alloc_tests PRIVATE
        fusion_alloc
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(fusion_alloc_tests)