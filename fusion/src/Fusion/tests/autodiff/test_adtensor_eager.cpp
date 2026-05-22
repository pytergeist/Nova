#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"

#include "fixtures.h"
#include "test_builders.h"

TEST(ADTensorEagerTest,
     unary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   const ADTensor<float> x = test_builders::ad_square_inputs(false);

   ADTensor<float> y = x.sqrt();

   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(y.base(), test_builders::raw_sqrt_expected());
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     binary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x = test_builders::ad_square_inputs(false);
   ADTensor<float> y = test_builders::ad_ones_inputs(false);

   ADTensor<float> z = x + y;

   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(z.base(), test_builders::raw_square_plus_ones_expected());
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(ADTensorEagerTest,
     reduction_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x = test_builders::ad_linear_inputs(false);

   ADTensor<float> y = x.sum(1, false);

   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(y.base(), test_builders::raw_sum_axis1_expected());
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     rhs_scalar_binary_op_on_non_grad_tensor_uses_eager_path_without_context) {
   ADTensor<float> x = test_builders::ad_linear_inputs(false);

   ADTensor<float> z = x * 2;

   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_TENSOR_EQ(z.base(), test_builders::raw_linear_times_two_expected());
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(
    ADTensorEagerTest,
    binary_op_uses_eager_path_when_both_inputs_require_grad_and_no_grad_guard_active) {
   ADTensor<float> x = test_builders::ad_square_inputs(true);
   ADTensor<float> y = test_builders::ad_ones_inputs(true);
   autodiff::NoGradGuard const _;

   ADTensor<float> z = x + y;

   EXPECT_TENSOR_EQ(z.base(), test_builders::raw_square_plus_ones_expected());
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(
    ADTensorEagerTest,
    binary_op_uses_eager_path_when_one_input_requires_grad_and_no_grad_guard_active) {
   ADTensor<float> x = test_builders::ad_square_inputs(true);
   ADTensor<float> y = test_builders::ad_ones_inputs(false);

   autodiff::NoGradGuard const _;

   ADTensor<float> z = x + y;

   EXPECT_TENSOR_EQ(z.base(), test_builders::raw_square_plus_ones_expected());
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());
}

TEST(
    ADTensorEagerTest,
    unary_op_uses_eager_path_when_input_requires_grad_and_no_grad_guard_active) {
   ADTensor<float> x = test_builders::ad_square_inputs(true);

   autodiff::NoGradGuard const _;
   ADTensor<float> y = x.sqrt();

   EXPECT_TENSOR_EQ(y.base(), test_builders::raw_sqrt_expected());
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     reduction_op_uses_eager_path_on_grad_tensor_and_no_grad_guard_active) {
   ADTensor<float> x = test_builders::ad_linear_inputs(true);
   autodiff::NoGradGuard const _;
   ADTensor<float> y = x.sum(1, false);

   EXPECT_TENSOR_EQ(y.base(), test_builders::raw_sum_axis1_expected());
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());
}

TEST(ADTensorEagerTest,
     grad_returns_nullopt_when_tensor_doesnt_participate_in_backward) {
   ADTensor<float> x = test_builders::ad_square_inputs(false);

   ADTensor<float> y = x.sqrt();
   EXPECT_FALSE(y.grad().has_value());
}

TEST(
    ADTensorEagerTest,
    binary_op_on_non_grad_tensors_inside_active_context_still_uses_eager_path) {
   EngineScope<float> scope;
   scope.enter();

   ADTensor<float> x = test_builders::ad_square_inputs(false);
   ADTensor<float> y = test_builders::ad_ones_inputs(false);

   ADTensor<float> z = x + y;

   EXPECT_TENSOR_EQ(z.base(), test_builders::raw_square_plus_ones_expected());
   EXPECT_FALSE(z.requires_grad());
   EXPECT_FALSE(z.has_vid());

   scope.exit();
}

TEST(ADTensorEagerTest,
     unary_op_on_non_grad_tensor_inside_active_context_still_uses_eager_path) {
   EngineScope<float> scope;
   scope.enter();

   ADTensor<float> x = test_builders::ad_square_inputs(false);

   ADTensor<float> y = x.sqrt();

   EXPECT_TENSOR_EQ(y.base(), test_builders::raw_sqrt_expected());
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());

   scope.exit();
}

TEST(
    ADTensorEagerTest,
    reduction_op_on_non_grad_tensor_inside_active_context_still_uses_eager_path) {
   EngineScope<float> scope;
   scope.enter();

   ADTensor<float> x = test_builders::ad_linear_inputs(false);

   ADTensor<float> y = x.sum(1, false);

   EXPECT_TENSOR_EQ(y.base(), test_builders::raw_sum_axis1_expected());
   EXPECT_FALSE(y.requires_grad());
   EXPECT_FALSE(y.has_vid());

   scope.exit();
}
