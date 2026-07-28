#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "../../core/dtype/Dtype.h"

TEST(DTypeTest, float32_size_matches_float) {
   EXPECT_EQ(get_dtype_size(DType::FLOAT32), sizeof(float));
}

TEST(DTypeTest, float64_size_matches_double) {
   EXPECT_EQ(get_dtype_size(DType::FLOAT64), sizeof(double));
}

TEST(DTypeTest, int32_size_matches_int32) {
   EXPECT_EQ(get_dtype_size(DType::INT32), sizeof(int32_t));
}

TEST(DTypeTest, int64_size_matches_int64) {
   EXPECT_EQ(get_dtype_size(DType::INT64), sizeof(int64_t));
}

TEST(DTypeTest, bool_size_matches_bool) {
   EXPECT_EQ(get_dtype_size(DType::BOOL), sizeof(bool));
}

TEST(DTypeTest, alias_constants_match_enum_values) {
   EXPECT_EQ(kFloat32, DType::FLOAT32);
   EXPECT_EQ(kFloat64, DType::FLOAT64);
   EXPECT_EQ(kInt32, DType::INT32);
   EXPECT_EQ(kInt64, DType::INT64);
   EXPECT_EQ(kBool, DType::BOOL);
}