#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "Fusion/core/Dtype.h"

TEST(DTypeTest, Float32SizeMatchesFloat) {
   EXPECT_EQ(get_dtype_size(DType::FLOAT32), sizeof(float));
}

TEST(DTypeTest, Float64SizeMatchesDouble) {
   EXPECT_EQ(get_dtype_size(DType::FLOAT64), sizeof(double));
}

TEST(DTypeTest, Int32SizeMatchesInt32) {
   EXPECT_EQ(get_dtype_size(DType::INT32), sizeof(int32_t));
}

TEST(DTypeTest, Int64SizeMatchesInt64) {
   EXPECT_EQ(get_dtype_size(DType::INT64), sizeof(int64_t));
}

TEST(DTypeTest, BoolSizeMatchesBool) {
   EXPECT_EQ(get_dtype_size(DType::BOOL), sizeof(bool));
}

TEST(DTypeTest, AliasConstantsMatchEnumValues) {
   EXPECT_EQ(kFloat32, DType::FLOAT32);
   EXPECT_EQ(kFloat64, DType::FLOAT64);
   EXPECT_EQ(kInt32, DType::INT32);
   EXPECT_EQ(kInt64, DType::INT64);
   EXPECT_EQ(kBool, DType::BOOL);
}