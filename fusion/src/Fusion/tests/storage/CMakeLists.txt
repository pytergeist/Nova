add_executable(fusion_storage_tests
        test_dense_storage.cpp
        test_tensor_buffer.cpp
        test_tensor_view.cpp
)

target_include_directories(fusion_storage_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_storage_tests PRIVATE
        fusion_core
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(fusion_storage_tests)