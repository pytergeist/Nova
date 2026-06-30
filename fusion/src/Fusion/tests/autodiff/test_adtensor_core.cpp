#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"

#include "fixtures.h"
#include "test_builders.h"

TEST(ADTensorCoreTest, default_constructed_tensor_is_uninitialised) {
   const ADTensor<float> t;
   EXPECT_FALSE(t.is_initialised());
   EXPECT_TRUE(t.empty());
   EXPECT_TRUE(t.shape().empty());
   EXPECT_TRUE(t.strides().empty());
   EXPECT_EQ(t.vid(), -1);
   EXPECT_FALSE(t.requires_grad());
}

TEST(ADTensorCoreTest, construction_from_raw_tensor_is_initialised) {
   Tensor<float> rt = make_test_tensor();
   const ADTensor<float> t(rt, true);
   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_FALSE(t.shape().empty());
   EXPECT_FALSE(t.strides().empty());
   EXPECT_TRUE(t.requires_grad());
   EXPECT_EQ(t.vid(), -1);
   EXPECT_TRUE(t.requires_grad());
   EXPECT_TENSOR_EQ(t.base(), rt);
}

TEST(ADTensorCoreTest, construct_from_shape_only_initialises_storage) {
   Device device{DeviceType::CPU, 0};
   ADTensor<float> t({2, 3}, DType::FLOAT32, device, false);

   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_EQ(t.vid(), -1);
   EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(t.dtype(), DType::FLOAT32);
   EXPECT_EQ(t.device(), device);

   EXPECT_FALSE(t.requires_grad());
}

TEST(ADTensorCoreTest, construct_from_shape_and_data_initialises_storage) {
   ADTensor<float> t(make_test_tensor(), false);

   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_EQ(t.vid(), -1);
   EXPECT_FALSE(t.requires_grad());
}

TEST(ADTensorCoreTest, construct_from_shape_and_data_forwards_metadata) {
   Device device{DeviceType::CPU, 0};
   ADTensor<float> t(make_test_tensor(), false);

   EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(t.strides(), (std::vector<std::int64_t>{3, 1}));
   EXPECT_EQ(t.rank(), 2);
   EXPECT_EQ(t.ndims(), 2);
   EXPECT_EQ(t.dtype(), DType::FLOAT32);
   EXPECT_EQ(t.device(), device);
   EXPECT_FALSE(t.requires_grad());
}

TEST(ADTensorCoreTest,
     construction_from_raw_tensor_preserves_shape_and_strides) {
   Tensor<float> rt = make_test_tensor();
   const ADTensor<float> t(rt, true);

   EXPECT_EQ(t.shape(), rt.shape());
   EXPECT_EQ(t.strides(), rt.strides());
   EXPECT_EQ(t.rank(), rt.rank());
   EXPECT_EQ(t.ndims(), rt.ndims());
   EXPECT_EQ(t.dtype(), rt.dtype());
   EXPECT_EQ(t.device(), rt.device());
   EXPECT_EQ(t.vid(), -1);
   EXPECT_TRUE(t.requires_grad());
}

TEST(ADTensorCoreTest, size_and_flat_size_forward_to_raw_tensor) {
   ADTensor<float> t = test_builders::ad_linear_inputs(false);

   EXPECT_EQ(t.size(), 6);
   EXPECT_EQ(t.flat_size(), 6);
}

TEST(ADTensorCoreTest, set_vid_sets_tensor_value_id) {
   Tensor<float> rt = make_test_tensor();
   ADTensor<float> t(rt, true);
   EXPECT_EQ(t.vid(), -1);
   const ValueID vid{2};
   t.set_vid(vid);
   EXPECT_EQ(t.vid(), vid);
}

TEST(ADTensorCoreTest, set_vid_on_tensor_with_valid_vid_throws) {
   Tensor<float> rt = make_test_tensor();
   ADTensor<float> t(rt, true);
   EXPECT_EQ(t.vid(), -1);
   const ValueID vid1{1};
   const ValueID vid2{2};
   t.set_vid(vid1);
   EXPECT_EQ(t.vid(), vid1);
   EXPECT_THROW(t.set_vid(vid2), std::runtime_error);
}

TEST(ADTensorCoreTest, has_vid_returns_false_when_vid_invalid) {
   Tensor<float> rt = make_test_tensor();
   ADTensor<float> t(rt, true);
   EXPECT_EQ(t.vid(), -1);
   EXPECT_FALSE(t.has_vid());
}

TEST(ADTensorCoreTest, has_vid_returns_true_when_vid_valid) {
   Tensor<float> rt = make_test_tensor();
   ADTensor<float> t(rt, true);
   EXPECT_EQ(t.vid(), -1);
   const ValueID vid{1};
   t.set_vid(vid);
   EXPECT_TRUE(t.has_vid());
}

TEST(ADTensorCoreTest, set_requires_grad_changes_false_to_true) {
   Tensor<float> rt = make_test_tensor();
   ADTensor<float> t(rt, false);
   EXPECT_FALSE(t.requires_grad());
   t.set_requires_grad(true);
   EXPECT_TRUE(t.requires_grad());
}

TEST(ADTensorCoreTest, set_requires_grad_changes_true_to_false) {
   Tensor<float> rt = make_test_tensor();
   ADTensor<float> t(rt, true);
   EXPECT_TRUE(t.requires_grad());
   t.set_requires_grad(false);
   EXPECT_FALSE(t.requires_grad());
}

TEST(ADTensorCoreTest, raw_nonconst_accessor_allows_mutation_via_clear) {
   ADTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                     DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   Tensor<float> &raw = t.base();
   raw.clear();

   Tensor<float> expected = Tensor<float>::from_dense(
       DenseTensor<float>({2, 3}, std::vector<float>{0, 0, 0, 0, 0, 0},
                          DType::FLOAT32, Device{DeviceType::CPU, 0}));

   EXPECT_TENSOR_EQ(t.base(), expected);
}

TEST(ADTensorCoreTest, begin_end_span_whole_tensor) {
   ADTensor<float> t({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                     Device{DeviceType::CPU, 0}, false);

   EXPECT_EQ(t.end() - t.begin(), 4);
   EXPECT_FLOAT_EQ(t.begin()[0], 1.0);
   EXPECT_FLOAT_EQ(t.begin()[3], 4.0);
}