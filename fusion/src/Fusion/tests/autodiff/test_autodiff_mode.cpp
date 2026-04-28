#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/Engine.hpp"
#include "Fusion/autodiff/AutodiffContext.hpp"

#include "fixtures.h"

TEST(AutodiffModeTest, no_grad_guard_disables_grad) {
   EXPECT_TRUE(autodiff::grad_enabled());
   autodiff::NoGradGuard _;
   EXPECT_FALSE(autodiff::grad_enabled());
}

TEST(AutodiffModeTest, no_grad_guard_enables_grad_after_disable_out_of_scope) {
   {
      EXPECT_TRUE(autodiff::grad_enabled());
      autodiff::NoGradGuard _;
      EXPECT_FALSE(autodiff::grad_enabled());
   }
   EXPECT_TRUE(autodiff::grad_enabled());
}

TEST(AutodiffModeTest, nested_no_grad_guards_restore_previous_state) {
   EXPECT_TRUE(autodiff::grad_enabled());

   {
      autodiff::NoGradGuard g1;
      EXPECT_FALSE(autodiff::grad_enabled());

      {
         autodiff::NoGradGuard g2;
         EXPECT_FALSE(autodiff::grad_enabled());
      }

      EXPECT_FALSE(autodiff::grad_enabled());
   }

   EXPECT_TRUE(autodiff::grad_enabled());
}

TEST(AutodiffModeTest,
     should_trace_unary_returns_true_when_engine_set_and_not_guarded) {
   AutodiffContextReset reset;
   ADTensor<float> t = make_test_tensor(true);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);

   EXPECT_TRUE(autodiff::should_trace(t));
}

TEST(AutodiffModeTest, should_trace_unary_returns_false_when_engine_not_set) {
   ADTensor<float> t = make_test_tensor(true);

   Engine<float> enginet;
   EXPECT_FALSE(autodiff::should_trace(t));
}

TEST(AutodiffModeTest,
     should_trace_unary_returns_false_when_no_engine_set_in_context) {
   AutodiffContextReset reset;
   ADTensor<float> t = make_test_tensor(true);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   AutodiffContext<float>::pop();
   EXPECT_FALSE(autodiff::should_trace(t));
}

TEST(AutodiffModeTest,
     should_trace_unary_returns_false_when_requires_grad_false) {
   AutodiffContextReset reset;
   ADTensor<float> t = make_test_tensor(false);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   EXPECT_FALSE(autodiff::should_trace(t));
}

TEST(AutodiffModeTest, should_trace_unary_returns_false_when_no_grad_guard) {
   AutodiffContextReset reset;
   ADTensor<float> t = make_test_tensor(true);

   autodiff::NoGradGuard _;

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   EXPECT_FALSE(autodiff::should_trace(t));
}

TEST(
    AutodiffModeTest,
    should_trace_binary_returns_true_when_engine_set_and_not_guarded_lhs_req_grad) {
   AutodiffContextReset reset;
   ADTensor<float> t1 = make_test_tensor(false);
   ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   EXPECT_TRUE(autodiff::should_trace(t1, t2));
}

TEST(
    AutodiffModeTest,
    should_trace_binary_returns_true_when_engine_set_and_not_guarded_rhs_req_grad) {
   AutodiffContextReset reset;
   ADTensor<float> t1 = make_test_tensor(true);
   ADTensor<float> t2 = make_test_tensor(false);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   EXPECT_TRUE(autodiff::should_trace(t1, t2));
}

TEST(AutodiffModeTest, should_trace_binary_returns_false_when_engine_not_set) {
   ADTensor<float> t1 = make_test_tensor(true);
   ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> enginet;
   EXPECT_FALSE(autodiff::should_trace(t1, t2));
}

TEST(AutodiffModeTest,
     should_trace_binary_returns_false_when_both_requires_grad_false) {
   AutodiffContextReset reset;
   ADTensor<float> t1 = make_test_tensor(false);
   ADTensor<float> t2 = make_test_tensor(false);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   EXPECT_FALSE(autodiff::should_trace(t1, t2));
}

TEST(AutodiffModeTest, should_trace_binary_returns_false_when_no_grad_guard) {
   AutodiffContextReset reset;
   ADTensor<float> t1 = make_test_tensor(true);
   ADTensor<float> t2 = make_test_tensor(true);

   autodiff::NoGradGuard _;

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   EXPECT_FALSE(autodiff::should_trace(t1, t2));
}

TEST(AutodiffModeTest,
     should_trace_binary_returns_false_when_no_engine_set_in_context) {
   AutodiffContextReset reset;
   ADTensor<float> t1 = make_test_tensor(true);
   ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> enginet;
   AutodiffContext<float>::set(&enginet);
   AutodiffContext<float>::pop();
   EXPECT_FALSE(autodiff::should_trace(t1, t2));
}
