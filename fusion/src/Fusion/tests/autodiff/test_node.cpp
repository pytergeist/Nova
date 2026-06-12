#include <gtest/gtest.h>

#include "Fusion/autodiff/registry/aligned/Add.hpp"
#include "Fusion/autodiff/registry/aligned/Sqrt.hpp"
#include "Fusion/autodiff/Node.hpp"

#include "fixtures.h"

TEST(NodeTest, binary_node_run_forward_delegates_to_op_forward) {
   using Op = Add<float>;
   Node<float, Op> node{Op{}};

   AutodiffMeta<float> input;
   Tensor<float> x = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));
   Tensor<float> y = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0}));
   input.push_back(x);
   input.push_back(y);

   AutodiffMeta<float> out = node.run_forward(input);

   ASSERT_EQ(out.size(), 1);
   Tensor<float> expected = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{11.0, 22.0, 33.0, 44.0, 55.0, 66.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0}));

   EXPECT_TENSOR_EQ(out.at(0), expected);
}

TEST(NodeTest, binary_node_run_backward_delegates_to_op_backward) {
   using Op = Add<float>;
   Node<float, Op> node{Op{}};

   AutodiffMeta<float> grad_out;
   Tensor<float> x = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));
   grad_out.push_back(x);

   AutodiffMeta<float> grad_in = node.run_backward(grad_out);

   ASSERT_EQ(grad_in.size(), 2);

   Tensor<float> expected = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));

   EXPECT_TENSOR_EQ(grad_in.at(0), expected);
   EXPECT_TENSOR_EQ(grad_in.at(1), expected);
}

TEST(NodeTest, unary_node_run_forward_delegates_to_op_forward) {
   using Op = Sqrt<float>;
   Node<float, Op> node{Op{}};

   AutodiffMeta<float> input;
   Tensor<float> x = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 4.0, 9.0, 16.0, 25.0, 36.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0}));
   input.push_back(x);

   AutodiffMeta<float> out = node.run_forward(input);

   ASSERT_EQ(out.size(), 1);

   Tensor<float> expected = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));

   EXPECT_TENSOR_EQ(out.at(0), expected);
}

TEST(NodeTest, node_exposes_operation_name_via_static_constant) {
   using Op = Add<float>;
   std::string_view op_name = OpTraits<Op::tag>::name;
   const std::string_view node_op_name = Node<float, Op>::KName;
   EXPECT_EQ(op_name, node_op_name);
}