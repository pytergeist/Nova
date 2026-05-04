#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"

#include "fixtures.h"


TEST(ADTensorEagerTest, unary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   EXPECT_FALSE(AutodiffContext<float>::has());
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0F, 4.0F, 9.0F, 16.0F, 25.0F, 36.0F},
                     DType::FLOAT32,
                     Device{DeviceType::CPU, 0},
                     false);

   ADTensor<float> y = x.sqrt();

   RawTensor<float> expected({2, 3},
                             std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                             DType::FLOAT32,
                             Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}


TEST(ADTensorEagerTest, binary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   EXPECT_FALSE(AutodiffContext<float>::has());
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32,
                     Device{DeviceType::CPU, 0},
                     false);

   ADTensor<float> y({2, 3},
                  std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                  DType::FLOAT32,
                  Device{DeviceType::CPU, 0},
                  false);

   ADTensor<float> z = x + y;

   RawTensor<float> expected({2, 3},
                             std::vector<float>{2.0, 5.0, 10.0, 17.0, 26.0, 37.0},
                             DType::FLOAT32,
                             Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(z.raw(), expected);
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}


TEST(ADTensorEagerTest, reduction_op_on_non_grad_tensor_uses_eager_path_without_context) {
   EXPECT_FALSE(AutodiffContext<float>::has());
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0F, 2.0F, 3.0F,
                                        4.0F, 5.0F, 6.0F},
                     DType::FLOAT32,
                     Device{DeviceType::CPU, 0},
                     false);

   ADTensor<float> y = x.sum(1, false);

   RawTensor<float> expected({2},
                             std::vector<float>{6.0F, 15.0F},
                             DType::FLOAT32,
                             Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

