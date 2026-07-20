#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "Fusion/core/tensor/DenseTensor.hpp"

TEST(ExecutionCPUUnaryEwiseTest, tag_fallback_unary_respects_strides) {
   const float a[] = {4., 99., 9., 99., 16., 99.};
   float out[] = {0., 0., 0.};

   fusion::execution::cpu::detail::unary_scalar_fallback<float, SqrtSIMD>(
       out, a,
       1, // out stride
       2, // a stride
       3  // len
   );
   EXPECT_FLOAT_EQ(out[0], 2.);
   EXPECT_FLOAT_EQ(out[1], 3.);
   EXPECT_FLOAT_EQ(out[2], 4.);
}

TEST(ExecutionCPUUnaryEwiseTest, unary_ewise_tag_fastpath_computes_square) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 4, 9, 16, 25, 36},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   fusion::planning::UnaryEwiseContext ctx =
       fusion::planning::make_unary_ewise_context(a);
   ASSERT_TRUE(ctx.fastpath);
   ASSERT_EQ(ctx.fast_len, 6);


   fusion::execution::cpu::unary_elementwise<float, SqrtTag>(out.get_ptr(), a.get_ptr(), ctx);
   EXPECT_FLOAT_EQ(out[0], 1.);
   EXPECT_FLOAT_EQ(out[1], 2.);
   EXPECT_FLOAT_EQ(out[2], 3.);
   EXPECT_FLOAT_EQ(out[3], 4.);
   EXPECT_FLOAT_EQ(out[4], 5.);
   EXPECT_FLOAT_EQ(out[5], 6.);
}