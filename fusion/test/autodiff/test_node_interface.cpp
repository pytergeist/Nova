#include <gtest/gtest.h>
#include <typeinfo>

#include "Fusion/autodiff/NodeInterface.hpp"
#include "Fusion/autodiff/registry/elementwise/Add.hpp"
#include "Fusion/autodiff/registry/elementwise/Sqrt.hpp"

#include "fixtures.h"

TEST(INodeTest, default_public_input_output_state_is_empty_for_new_node) {
   INode<float> node{Add<float>{}};

   EXPECT_FALSE(node.has_inputs());
   EXPECT_FALSE(node.has_outputs());
   EXPECT_EQ(node.num_inputs(), 0);
   EXPECT_EQ(node.num_outputs(), 0);
   EXPECT_TRUE(node.inputs().empty());
   EXPECT_TRUE(node.outputs().empty());
}

TEST(INodeTest, name_forwards_from_erased_node_model) {
   INode<float> node{Add<float>{}};
   std::string_view op_name = OpTraits<typename Add<float>::tag>::name;
   EXPECT_EQ(node.name(), op_name);
}

TEST(INodeTest, get_input_and_output_arity_forward_from_operation) {
   INode<float> node{Add<float>{}};
   std::size_t input_arity =
       OpTraits<typename Add<float>::tag>::schema.inputs.arity;
   std::size_t output_arity =
       OpTraits<typename Add<float>::tag>::schema.outputs.arity;
   EXPECT_EQ(node.get_input_arity(), input_arity);
   EXPECT_EQ(node.get_output_arity(), output_arity);
}

TEST(INodeTest, type_info_methods_match_operation_types) {
   using Op = Add<float>;
   INode<float> node{Op{}};

   EXPECT_EQ(node.in_type(), typeid(typename Op::In));
   EXPECT_EQ(node.out_type(), typeid(typename Op::Out));
   EXPECT_EQ(node.grad_in_type(), typeid(typename Op::GradIn));
   EXPECT_EQ(node.grad_out_type(), typeid(typename Op::GradOut));
}

TEST(INodeTest, apply_forward_dispatches_through_type_erasure) {
   INode<float> node{Add<float>{}};

   AutodiffMeta<float> input;
   Tensor<float> x = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));
   Tensor<float> y = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0}));
   input.push_back(x);
   input.push_back(y);

   AutodiffMeta<float> out = node.apply_forward(input);

   ASSERT_EQ(out.size(), 1);

   Tensor<float> expected = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{11.0, 22.0, 33.0, 44.0, 55.0, 66.0},
       DType::FLOAT32, Device{DeviceType::CPU, 0}));

   EXPECT_TENSOR_EQ(out.at(0), expected);
}

TEST(INodeTest, apply_backward_dispatches_through_type_erasure) {
   INode<float> node{Add<float>{}};

   AutodiffMeta<float> grad_out;
   Tensor<float> x = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));
   grad_out.push_back(x);

   AutodiffMeta<float> grad_in = node.apply_backward(grad_out);

   ASSERT_EQ(grad_in.size(), 2);

   Tensor<float> expected = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, DType::FLOAT32,
       Device{DeviceType::CPU, 0}));

   EXPECT_TENSOR_EQ(grad_in.at(0), expected);
   EXPECT_TENSOR_EQ(grad_in.at(1), expected);
}

TEST(INodeTest, moved_from_node_throws_on_apply_backward) {
   INode<float> src{Add<float>{}};
   INode<float> dst{std::move(src)};

   AutodiffMeta<float> grad_out;
   Tensor<float> x = Tensor<float>::from_dense(DenseTensor<float>(
       {2, 3}, std::vector<float>{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F},
       DType::FLOAT32, Device{DeviceType::CPU, 0}));
   grad_out.push_back(x);

   EXPECT_THROW(src.apply_backward(grad_out), std::runtime_error);
}