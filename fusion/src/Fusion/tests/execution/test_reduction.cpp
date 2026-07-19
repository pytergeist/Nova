#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "Fusion/core/tensor/DenseTensor.hpp"

TEST(ExecutionCPUReductionTest,
     tag_fallback_reduction_accumulates_into_output) {
   const float a[] = {1., 2., 3., 4., 5.};
   float out[] = {1.};
   fusion::execution::cpu::detail::reduction_scalar_fallback<float, SumSIMD>(
       out, a, 0, 1, 6);

   EXPECT_FLOAT_EQ(out[0], 16.);
}

TEST(ExecutionCPUReductionTest, tag_fallback_reduction_respects_strides) {
   const float a[] = {1., 2., 3., 4., 5.};
   float out[] = {1.};
   fusion::execution::cpu::detail::reduction_scalar_fallback<float, SumSIMD>(
       out, a, 0, 2, 3);

   EXPECT_FLOAT_EQ(out[0], 10.);
}

TEST(ExecutionCPUReductionTest,
     reduction_tag_global_fastpath_sums_all_elements) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({1}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   fusion::planning::ReductionContext meta =
       fusion::planning::make_reduction_context(
           a, fusion::planning::kGlobalReduceAxis, false);
   ASSERT_TRUE(meta.fastpath);
   ASSERT_EQ(meta.fast_len, 6);

   fusion::execution::cpu::reduction<float, SumSIMD>(a, meta, out);

   EXPECT_FLOAT_EQ(out[0], 21.);
}

TEST(ExecutionCPUReductionTest,
     reduction_tag_axis_path_reduces_requested_axis) {
   DenseTensor<float> a({2, 3},
                        std::vector<float>{
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                        },
                        DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   fusion::planning::ReductionContext meta =
       fusion::planning::make_reduction_context(a, 1, false);
   ASSERT_FALSE(meta.fastpath);
   ASSERT_EQ(meta.out_shape, (std::vector<std::size_t>{2}));

   fusion::execution::cpu::reduction<float, SumSIMD>(a, meta, out);

   EXPECT_FLOAT_EQ(out[0], 6.);
   EXPECT_FLOAT_EQ(out[1], 15.);
}