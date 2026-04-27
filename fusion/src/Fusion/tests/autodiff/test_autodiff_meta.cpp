#include <gtest/gtest.h>

#include "Fusion/autodiff/AutodiffMeta.hpp"

#include "Fusion/core/RawTensor.hpp"

#include "fixtures.h"

TEST(AutodiffMetaTest, default_constructor_produces_empty_data) {
   AutodiffMeta<float> meta{};
   EXPECT_TRUE(meta.empty());
   EXPECT_EQ(meta.size(), 0);
}

TEST(AutodiffMetaTest, reserve_constructor_reserves_correct_capacity) {
   AutodiffMeta<float> meta{10};
   EXPECT_EQ(meta.data.capacity(), 10);
}

TEST(AutodiffMetaTest, emplace_back_copies_single_tensor_to_data) {
   RawTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   AutodiffMeta<float> meta{};
   meta.emplace_back(t);
   EXPECT_EQ(meta.size(), 1);
   EXPECT_TENSOR_EQ(meta[0], t);
}

TEST(AutodiffMetaTest, push_back_copies_single_tensor_to_data) {
   RawTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   AutodiffMeta<float> meta{};
   meta.push_back(t);
   EXPECT_EQ(meta.size(), 1);
   EXPECT_TENSOR_EQ(meta[0], t);
}

TEST(AutodiffMetaTest, emplace_back_copies_multiple_tensors_to_data_in_order) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> t2({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.emplace_back(t1);
   meta.emplace_back(t2);

   EXPECT_EQ(meta.size(), 2);

   EXPECT_TENSOR_EQ(meta[0], t1);
   EXPECT_TENSOR_EQ(meta[1], t2);
}

TEST(AutodiffMetaTest, push_back_copies_multiple_tensors_to_data_in_order) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> t2({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.push_back(t1);
   meta.push_back(t2);

   EXPECT_EQ(meta.size(), 2);

   EXPECT_TENSOR_EQ(meta[0], t1);
   EXPECT_TENSOR_EQ(meta[1], t2);
}

TEST(AutodiffMetaTest, at_method_returns_correct_tensor) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> t2({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.emplace_back(t1);
   meta.emplace_back(t2);

   EXPECT_EQ(meta.size(), 2);

   EXPECT_TENSOR_EQ(meta.at(0), t1);
   EXPECT_TENSOR_EQ(meta.at(1), t2);
}

TEST(AutodiffMetaTest, index_operator_returns_correct_meta) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> t2({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.emplace_back(t1);
   meta.emplace_back(t2);

   EXPECT_EQ(meta.size(), 2);

   EXPECT_TENSOR_EQ(meta[0], t1);
   EXPECT_TENSOR_EQ(meta[1], t2);
}

TEST(RawTensorTest, begin_end_span_whole_tensor) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> t2({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.emplace_back(t1);
   meta.emplace_back(t2);

   EXPECT_EQ(meta.end() - meta.begin(), 2);

   EXPECT_TENSOR_EQ(meta.begin()[0], t1);
   EXPECT_TENSOR_EQ(meta.begin()[1], t2);
}

TEST(AutodiffMetaTest, at_method_throws_when_index_out_of_range) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.emplace_back(t1);

   EXPECT_THROW(meta.at(1), std::out_of_range);
}

TEST(AutodiffMetaTest, index_operator_throws_when_index_out_of_range) {
   RawTensor<float> t1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                       DType::FLOAT32, Device{DeviceType::CPU, 0});

   AutodiffMeta<float> meta{};
   meta.emplace_back(t1);

   EXPECT_THROW(meta[1], std::out_of_range);
}