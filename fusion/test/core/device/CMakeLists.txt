add_executable(fusion_device_tests
        test_device.cpp
)

target_include_directories(fusion_device_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_device_tests PRIVATE
        fusion_core
        GTest::gtest_main
)

gtest_discover_tests(fusion_device_tests)