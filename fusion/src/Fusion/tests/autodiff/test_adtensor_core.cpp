#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"

#include "fixtures.h"


TEST(ADTensorCoreTest, default_constructed_tensor_is_uninitialised) {
   const ADTensor<float> t;
   EXPECT_FALSE(t.is_initialised());
   EXPECT_TRUE(t.empty());
   EXPECT_TRUE(t.shape().empty());
   EXPECT_TRUE(t.strides().empty());
}

TEST(ADTensorCoreTest, construction_from_raw_tensor_is_initialised) {
   RawTensor<float> rt = make_test_raw_tensor();
   const ADTensor<float> t(rt, true);
   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_FALSE(t.shape().empty());
   EXPECT_FALSE(t.strides().empty());
   EXPECT_TRUE(t.requires_grad());
   EXPECT_TENSOR_EQ(t.raw(), rt);
}


TEST(ADTensorCoreTest, construct_from_shape_and_data_initialises_storage) {
   ADTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0}, false);

   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
}

TEST(ADTensorCoreTest, construct_from_shape_and_data_correctly_fowards_metadata) {
   Device device{DeviceType::CPU, 0};
   ADTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, device,  false);

   EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(t.strides(), (std::vector<std::int64_t>{3, 1}));
   EXPECT_EQ(t.rank(), 2);
   EXPECT_EQ(t.ndims(), 2);
   EXPECT_EQ(t.dtype(), DType::FLOAT32);
   EXPECT_EQ(t.device(), device);
}


TEST(ADTensorCoreTest, construction_from_raw_tensor_preserves_shape_and_strides) {
   RawTensor<float> rt = make_test_raw_tensor();
   const ADTensor<float> t(rt, true);

   EXPECT_EQ(t.shape(), rt.shape());
   EXPECT_EQ(t.strides(), rt.strides());
   EXPECT_EQ(t.rank(), rt.rank());
   EXPECT_EQ(t.ndims(), rt.ndims());
   EXPECT_EQ(t.dtype(), rt.dtype());
   EXPECT_EQ(t.device(), rt.device());
}





