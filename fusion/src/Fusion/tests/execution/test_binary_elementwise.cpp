#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "Fusion/core/tensor/DenseTensor.hpp"

TEST(ExecutionCPUBinaryEwiseTest, tag_fallback_binary_respects_strides) {
   const float a[] = {1., 99., 2., 99., 3., 99.};
   const float b[] = {10., 88., 20., 88., 30., 88.};
   float out[] = {0., 0., 0.};

   fusion::execution::cpu::detail::binary_scalar_fallback<float, AddSIMD>(
       out, a, b,
       1, // out stride
       2, // a stride
       2, // b stride
       3  // len
   );

   EXPECT_FLOAT_EQ(out[0], 11.);
   EXPECT_FLOAT_EQ(out[1], 22.);
   EXPECT_FLOAT_EQ(out[2], 33.);
}

TEST(ExecutionCPUBinaryEwiseTest,
     binary_ewise_tag_fastpath_computes_elementwise_add) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> b({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   fusion::planning::BinaryEwiseContext ctx =
       fusion::planning::make_binary_ewise_context(a, b);
   EXPECT_EQ(ctx.exec, fusion::planning::BinaryExecKind::FlatContiguous);
   EXPECT_EQ(ctx.fast_len, 6);

   fusion::execution::cpu::binary_elementwise<float, AddTag>(out.get_ptr(), a.get_ptr(), b.get_ptr(), ctx);

   EXPECT_FLOAT_EQ(out[0], 2.);
   EXPECT_FLOAT_EQ(out[1], 4.);
   EXPECT_FLOAT_EQ(out[2], 6.);
   EXPECT_FLOAT_EQ(out[3], 8.);
   EXPECT_FLOAT_EQ(out[4], 10.);
   EXPECT_FLOAT_EQ(out[5], 12.);
}

TEST(ExecutionCPUBinaryEwiseTest,
     binary_ewise_tag_broadcast_path_computes_elementwise_add) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> b({1, 3}, std::vector<float>{1, 2, 3}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   fusion::planning::BinaryEwiseContext ctx =
       fusion::planning::make_binary_ewise_context(a, b);
   ASSERT_EQ(ctx.exec,
             fusion::planning::BinaryExecKind::FlatContiguousBroadcastRHS);
   ASSERT_EQ(ctx.out_shape, (std::vector<size_t>{2, 3}));

   fusion::execution::cpu::binary_elementwise<float, AddTag>(out.get_ptr(), a.get_ptr(), b.get_ptr(), ctx);

   EXPECT_FLOAT_EQ(out[0], 2.);
   EXPECT_FLOAT_EQ(out[1], 4.);
   EXPECT_FLOAT_EQ(out[2], 6.);
   EXPECT_FLOAT_EQ(out[3], 5.);
   EXPECT_FLOAT_EQ(out[4], 7.);
   EXPECT_FLOAT_EQ(out[5], 9.);
}