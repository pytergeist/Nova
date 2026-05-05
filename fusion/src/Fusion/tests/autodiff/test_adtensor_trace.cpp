#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"

#include "fixtures.h"

TEST(ADTensorTraceTest, ensure_vid_without_active_context_throws) {
   AutodiffContextReset reset;
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_THROW(x.ensure_vid(), std::runtime_error);
}

TEST(ADTensorTraceTest, ensure_vid_inside_active_context_registers_value) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_EQ(x.vid(), -1);
   x.ensure_vid();
   EXPECT_TRUE(x.has_vid());
}

TEST(ADTensorTraceTest,
     ensure_vid_called_twice_in_same_context_returns_same_vid) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_EQ(x.vid(), -1);
   x.ensure_vid();
   EXPECT_EQ(x.vid(), ValueID{0});
   x.ensure_vid();
   EXPECT_EQ(x.vid(), ValueID{0});
}

TEST(ADTensorTraceTest, ensure_vid_inside_active_context_sets_has_vid_true) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_EQ(x.vid(), -1);
   x.ensure_vid();
   EXPECT_TRUE(x.has_vid());
}

TEST(ADTensorTraceTest, backward_on_leaf_tensor_exposes_seed_gradient) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_EQ(x.vid(), -1);
   x.ensure_vid();
   x.backward();
   EXPECT_TRUE(x.grad().has_value());
}

TEST(
    ADTensorTraceTest,
    unary_op_on_grad_tensor_inside_active_context_produces_grad_requiring_result) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y = x.sqrt();
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(y.requires_grad());
}

TEST(ADTensorTraceTest, traced_unary_result_can_be_backwarded_to_leaf) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y = x.sqrt();
   y.backward();
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(x.grad().has_value());
}

TEST(
    ADTensorTraceTest,
    reduction_op_on_grad_tensor_inside_active_context_produces_grad_requiring_result) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y = x.sum(1, false);
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(y.requires_grad());
}

TEST(ADTensorTraceTest, traced_reduction_result_can_be_backwarded_to_leaf) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y = x.sum(1, false);
   y.backward();
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(x.grad().has_value());
}

TEST(
    ADTensorTraceTest,
    binary_op_when_both_inputs_require_grad_inside_active_context_produces_grad_requiring_result) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(z.has_vid());
   EXPECT_TRUE(z.requires_grad());
}

TEST(
    ADTensorTraceTest,
    binary_op_when_lhs_input_requires_grad_inside_active_context_produces_grad_requiring_result) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> z = x + y;
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(z.has_vid());
   EXPECT_TRUE(z.requires_grad());
}

TEST(
    ADTensorTraceTest,
    binary_op_when_rhs_input_requires_grad_inside_active_context_produces_grad_requiring_result) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(z.has_vid());
   EXPECT_TRUE(z.requires_grad());
}

TEST(ADTensorTraceTest,
     backward_on_untracked_tensor_without_active_context_throws) {
   AutodiffContextReset reset;
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   EXPECT_THROW(z.backward(), std::runtime_error);
}

TEST(ADTensorTraceTest, backward_on_traced_unary_result_populates_leaf_grad) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y = x.sqrt();
   y.backward();
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(x.grad().has_value());
}

TEST(ADTensorTraceTest,
     backward_on_traced_binary_result_populates_all_leaf_grads) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   z.backward();
   EXPECT_TRUE(y.has_vid());
   EXPECT_TRUE(x.has_vid());
   EXPECT_TRUE(z.has_vid());
   EXPECT_TRUE(x.grad().has_value());
   EXPECT_TRUE(y.grad().has_value());
}

TEST(ADTensorTraceTest, grad_returns_nullopt_before_backward) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   EXPECT_FALSE(x.grad().has_value());
   EXPECT_FALSE(y.grad().has_value());
}

TEST(ADTensorTraceTest, returned_grad_tensor_does_not_require_grad) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   z.backward();
   ADTensor<float> x_grad = x.grad().value();
   ADTensor<float> y_grad = y.grad().value();
   EXPECT_FALSE(x_grad.requires_grad());
   EXPECT_FALSE(y_grad.requires_grad());
}

TEST(ADTensorTraceTest, grad_persists_after_engine_scope_exit) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   z.backward();
   EXPECT_TRUE(x.grad().has_value());
   EXPECT_TRUE(y.grad().has_value());
   scope.exit();
   EXPECT_TRUE(x.grad().has_value());
   EXPECT_TRUE(y.grad().has_value());
}

TEST(ADTensorTraceTest,
     backward_on_mixed_grad_binary_result_populates_only_grad_requiring_leaf) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   ADTensor<float> z = x + y;
   z.backward();
   EXPECT_TRUE(x.grad().has_value());
   EXPECT_FALSE(y.grad().has_value());
}

TEST(
    ADTensorTraceTest,
    intermediate_tensor_grad_is_not_exposed_when_only_leaf_export_is_supported) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();
   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> y({2, 3},
                     std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ADTensor<float> z = x + y;
   z.backward();
   EXPECT_TRUE(x.grad().has_value());
   EXPECT_TRUE(y.grad().has_value());
   EXPECT_FALSE(z.grad().has_value());
}

TEST(ADTensorTraceTest,
     tensor_can_be_retracked_in_new_context_after_old_context_exits) {
   AutodiffContextReset reset;

   ADTensor<float> x({2, 3},
                     std::vector<float>{1.0F, 4.0F, 9.0F, 16.0F, 25.0F, 36.0F},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, true);

   ValueID first_vid{-1};
   {
      EngineScope<float> scope{};
      scope.enter();

      first_vid = x.ensure_vid();
      EXPECT_TRUE(x.has_vid());
      EXPECT_EQ(first_vid, ValueID{0});

      scope.exit();
   }
   EXPECT_FALSE(AutodiffContext<float>::has());
   {
      EngineScope<float> scope{};
      scope.enter();

      ValueID second_vid = x.ensure_vid();
      EXPECT_TRUE(x.has_vid());
      EXPECT_EQ(second_vid, ValueID{0});
      EXPECT_NE(&scope.eng(), nullptr);

      scope.exit();
   }
}
