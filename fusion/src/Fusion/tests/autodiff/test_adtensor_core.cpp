#include <gtest/gtest.h>

#include "Fusion/autodiff/ADTensor.hpp"

#include "Fixtures.h"


TEST(ADTensorCoreTest, default_constructed_tensor_is_uninitialised) {
   const ADTensor<float> t;
   EXPECT_FALSE(t.is_initialised());
   EXPECT_TRUE(t.empty());
   EXPECT_TRUE(t.shape().empty());
}

TEST(ADTensorCoreTest, construction_from_raw_tensor_is_initialised) {
   const ADTensor<float> t;
   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_FALSE(t.shape().empty());
}



