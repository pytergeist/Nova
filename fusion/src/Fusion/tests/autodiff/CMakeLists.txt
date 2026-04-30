add_executable(fusion_autodiff_tests
        test_adtensor_core.cpp
        test_adtensor_eager.cpp
        test_adtensor_trace.cpp
        test_autodiff_meta.cpp
        test_autodiff_mode.cpp
        test_dispatch.cpp
        test_engine.cpp
        test_grad_store.cpp
        test_autodiff_context.cpp
        test_grad_store.cpp
        test_graph.cpp

)

target_include_directories(fusion_autodiff_tests PRIVATE
        ${FUSION_INCLUDE_ROOT}
)

target_link_libraries(fusion_autodiff_tests PRIVATE fusion_autodiff
        fusion_core
        fusion_storage
        fusion_alloc
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(fusion_autodiff_tests)