#include <gtest/gtest.h>

#include "fixtures.h"

#include "Fusion/autodiff/Dispatch.hpp"

struct TestParam {
   int p;
   bool operator==(const TestParam &) const = default;
};

TEST(AutodiffDispatchTest,
     construct_meta_binary_produces_meta_with_two_tensors) {
   ADTensor<float> t1 = make_test_tensor(true);
   ADTensor<float> t2 = make_test_tensor(true);

   AutodiffMeta<float> meta = autodiff::construct_meta<float>(t1, t2);

   EXPECT_EQ(meta.size(), 2);

   EXPECT_EQ(meta[0].shape(), t1.shape());
   EXPECT_EQ(meta[1].shape(), t2.shape());
   EXPECT_EQ(meta[0].strides(), t1.strides());
   EXPECT_EQ(meta[1].strides(), t2.strides());
   EXPECT_EQ(meta[0].dtype(), t1.dtype());
   EXPECT_EQ(meta[1].dtype(), t2.dtype());
   EXPECT_EQ(meta[0].device(), t2.device());
   EXPECT_EQ(meta[1].device(), t2.device());
}

TEST(AutodiffDispatchTest,
     construct_meta_unary_produces_meta_with_single_tensor) {
   ADTensor<float> t = make_test_tensor(true);

   AutodiffMeta<float> meta = autodiff::construct_meta<float>(t);

   EXPECT_EQ(meta.size(), 1);

   EXPECT_EQ(meta[0].shape(), t.shape());
   EXPECT_EQ(meta[0].strides(), t.strides());
   EXPECT_EQ(meta[0].dtype(), t.dtype());
   EXPECT_EQ(meta[0].device(), t.device());
}

TEST(
    AutodiffDispatchTest,
    construct_meta_unary_with_param_produces_meta_with_single_tensor_and_param) {
   ADTensor<float> t = make_test_tensor(true);

   const TestParam p{1};

   AutodiffMeta<float> meta = autodiff::construct_meta<float>(t, p);

   EXPECT_EQ(meta.size(), 1);

   EXPECT_EQ(meta[0].shape(), t.shape());
   EXPECT_EQ(meta[0].strides(), t.strides());
   EXPECT_EQ(meta[0].dtype(), t.dtype());
   EXPECT_EQ(meta[0].device(), t.device());
   EXPECT_EQ(std::any_cast<TestParam>(meta.op_param), p);
}

/// Below we are only testing eager paths. Tracing paths should live in
/// integration tests and use a real engine/context.
/// NB: The mock Ops do nothing to the underlying data, therefore we test
/// shape/strides between input[0] and the return tensor, as no transformations
/// have taken place.

TEST(AutodiffDispatchTest,
     binary_dispatch_uses_eager_when_both_not_requires_grad) {
   const ADTensor<float> t1 = make_test_tensor(false);
   const ADTensor<float> t2 = make_test_tensor(false);
   bool eager_called = false;
   auto eager = [&](const ADTensor<float> &x, const ADTensor<float> &y) {
      eager_called = true;
      return x;
   };

   ADTensor<float> out =
       autodiff::binary<float, Operation<float, TestBinaryOp<float>>>(t1, t2,
                                                                      eager);
   EXPECT_TRUE(eager_called);
   EXPECT_FALSE(out.requires_grad());
   EXPECT_TENSOR_EQ(out.base(), t1.base());
}

TEST(AutodiffDispatchTest, unary_dispatch_uses_eager_when_not_requires_grad) {
   const ADTensor<float> t = make_test_tensor(false);
   bool eager_called = false;
   auto eager = [&](const ADTensor<float> &x) {
      eager_called = true;
      return x;
   };

   ADTensor<float> out =
       autodiff::unary<float, Operation<float, TestUnaryOp<float>>>(t, eager);
   EXPECT_TRUE(eager_called);
   EXPECT_FALSE(out.requires_grad());
   EXPECT_TENSOR_EQ(out.base(), t.base());
}

TEST(AutodiffDispatchTest,
     unary_with_param_dispatch_uses_eager_when_not_requires_grad) {
   const ADTensor<float> t = make_test_tensor(false);
   const TestParam p{1};
   bool eager_called = false;
   auto eager = [&](const ADTensor<float> &x, const TestParam &p) {
      eager_called = true;
      return x;
   };

   ADTensor<float> out =
       autodiff::unary<float, Operation<float, TestUnaryOp<float>>, TestParam>(
           t, p, eager);
   EXPECT_TRUE(eager_called);
   EXPECT_FALSE(out.requires_grad());
   EXPECT_TENSOR_EQ(out.base(), t.base());
}

TEST(AutodiffDispatchTest,
     unary_dispatch_uses_eager_when_grad_is_globally_disabled) {
   const ADTensor<float> t = make_test_tensor(true);

   bool eager_called = false;
   auto eager = [&](const ADTensor<float> &x) {
      eager_called = true;
      return x;
   };

   const autodiff::NoGradGuard _;
   ADTensor<float> out =
       autodiff::unary<float, Operation<float, TestUnaryOp<float>>>(t, eager);

   EXPECT_TRUE(eager_called);
   EXPECT_TENSOR_EQ(out.base(), t.base());
}

TEST(
    AutodiffDispatchTest,
    binary_dispatch_uses_eager_when_one_tensor_requires_grad_and_grad_is_globally_disabled) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   bool eager_called = false;
   auto eager = [&](const ADTensor<float> &x, const ADTensor<float> &y) {
      eager_called = true;
      return y;
   };

   const autodiff::NoGradGuard _;
   ADTensor<float> out =
       autodiff::binary<float, Operation<float, TestBinaryOp<float>>>(t1, t2,
                                                                      eager);

   EXPECT_TRUE(eager_called);
   EXPECT_TENSOR_EQ(out.base(), t2.base());
}

TEST(
    AutodiffDispatchTest,
    binary_dispatch_uses_eager_when_both_tensors_require_grad_and_grad_is_globally_disabled) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   bool eager_called = false;
   auto eager = [&](const ADTensor<float> &x, const ADTensor<float> &y) {
      eager_called = true;
      return y;
   };

   const autodiff::NoGradGuard _;
   ADTensor<float> out =
       autodiff::binary<float, Operation<float, TestBinaryOp<float>>>(t1, t2,
                                                                      eager);

   EXPECT_TRUE(eager_called);
   EXPECT_TENSOR_EQ(out.base(), t2.base());
}