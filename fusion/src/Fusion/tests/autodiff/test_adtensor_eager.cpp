#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"

#include "fixtures.h"

// TODO: abstract tensor creation out of tests

TEST(ADTensorEagerTest,
     unary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y = x.sqrt();

   RawTensor<float> expected({2, 3},
                             std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                             DType::FLOAT32, Device{DeviceType::CPU, 0});
   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     binary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y({2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> z = x + y;

   RawTensor<float> expected(
       {2, 3}, std::vector<float>{2.0, 5.0, 10.0, 17.0, 26.0, 37.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0});
   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(z.raw(), expected);
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(ADTensorEagerTest,
     reduction_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x({2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y = x.sum(1, false);

   RawTensor<float> expected({2}, std::vector<float>{6.0, 15.0}, DType::FLOAT32,
                             Device{DeviceType::CPU, 0});
   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     rhs_scalar_binary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x({2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> z = x * 2;

   RawTensor<float> expected({2, 3},
                             std::vector<float>{2.0, 4.0, 6.0, 8.0, 10.0, 12.0},
                             DType::FLOAT32, Device{DeviceType::CPU, 0});
   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(z.raw(), expected);
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(
    ADTensorEagerTest,
    binary_op_uses_eager_path_when_both_inputs_require_grad_and_no_grad_guard_active) {
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);
   autodiff::NoGradGuard const _;
   ADTensor<float> z = x + y;

   RawTensor<float> expected(
       {2, 3}, std::vector<float>{2.0, 5.0, 10.0, 17.0, 26.0, 37.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(z.raw(), expected);
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(
    ADTensorEagerTest,
    binary_op_uses_eager_path_when_one_input_requires_grad_and_no_grad_guard_active) {
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   autodiff::NoGradGuard const _;

   ADTensor<float> z = x + y;

   RawTensor<float> expected(
       {2, 3}, std::vector<float>{2.0, 5.0, 10.0, 17.0, 26.0, 37.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(z.raw(), expected);
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(
    ADTensorEagerTest,
    unary_op_uses_eager_path_when_input_requires_grad_and_no_grad_guard_active) {
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   autodiff::NoGradGuard const _;
   ADTensor<float> y = x.sqrt();

   RawTensor<float> expected({2, 3},
                             std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                             DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     reduction_op_uses_eager_path_on_grad_tensor_and_no_grad_guard_active) {
   ADTensor<float> x({2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);
   autodiff::NoGradGuard const _;
   ADTensor<float> y = x.sum(1, false);

   RawTensor<float> expected({2}, std::vector<float>{6.0, 15.0}, DType::FLOAT32,
                             Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     grad_returns_nullopt_when_tensor_doesnt_participate_in_backward) {
   ADTensor<float> x({2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y = x.sqrt();
   EXPECT_FALSE(y.grad().has_value());
}

TEST(
    ADTensorEagerTest,
    binary_op_on_non_grad_tensors_inside_active_context_still_uses_eager_path) {
   EngineScope<float> scope;
   scope.enter();

   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y({2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> z = x + y;

   RawTensor<float> expected(
       {2, 3}, std::vector<float>{2.0, 5.0, 10.0, 17.0, 26.0, 37.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(z.raw(), expected);
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());

   scope.exit();
}

TEST(ADTensorEagerTest,
     unary_op_on_non_grad_tensor_inside_active_context_still_uses_eager_path) {
   EngineScope<float> scope;
   scope.enter();

   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y = x.sqrt();

   RawTensor<float> expected({2, 3},
                             std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                             DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());

   scope.exit();
}

TEST(
    ADTensorEagerTest,
    reduction_op_on_non_grad_tensor_inside_active_context_still_uses_eager_path) {
   EngineScope<float> scope;
   scope.enter();

   ADTensor<float> x({2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y = x.sum(1, false);

   RawTensor<float> expected({2}, std::vector<float>{6.0, 15.0}, DType::FLOAT32,
                             Device{DeviceType::CPU, 0});

   EXPECT_TENSOR_EQ(y.raw(), expected);
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());

   scope.exit();
}
