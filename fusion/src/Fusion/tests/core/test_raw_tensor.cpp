#include <gtest/gtest.h>

#include <vector>

#include "Fusion/core/tensor/DenseTensor.hpp"

// TODO: create a make_tensor helper

TEST(RawTensorTest, default_constructed_tensor_is_uninitialised) {
   DenseTensor<float> t;

   EXPECT_FALSE(t.is_initialised());
   EXPECT_TRUE(t.empty());
   EXPECT_TRUE(t.shape().empty());
   EXPECT_TRUE(t.strides().empty());
}

TEST(RawTensorTest, construct_from_shape_and_data_initialises_storage) {
   DenseTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_NE(t.get_ptr(), nullptr);
}

TEST(RawTensorTest, construct_from_shape_and_data_preserves_shape_and_strides) {
   DenseTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(t.strides(), (std::vector<std::int64_t>{3, 1}));
   EXPECT_EQ(t.rank(), 2);
   EXPECT_EQ(t.ndims(), 2);
   EXPECT_TRUE(t.is_contiguous());
}

TEST(RawTensorTest, construct_from_shape_and_data_copies_values) {
   DenseTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   ASSERT_NE(t.get_ptr(), nullptr);
   EXPECT_FLOAT_EQ(t[0], 1.0);
   EXPECT_FLOAT_EQ(t[1], 2.0);
   EXPECT_FLOAT_EQ(t[2], 3.0);
   EXPECT_FLOAT_EQ(t[3], 4.0);
   EXPECT_FLOAT_EQ(t[4], 5.0);
   EXPECT_FLOAT_EQ(t[5], 6.0);
}

TEST(RawTensorTest, construct_from_shape_only_allocates_storage) {
   DenseTensor<float> t({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_TRUE(t.is_initialised());
   EXPECT_FALSE(t.empty());
   EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(t.strides(), (std::vector<std::int64_t>{3, 1}));
   EXPECT_EQ(t.flat_size(), 6);
   EXPECT_NE(t.get_ptr(), nullptr);
}

TEST(RawTensorTest, construct_rejects_empty_shape) {
   EXPECT_THROW((DenseTensor<float>({}, std::vector<float>{}, DType::FLOAT32,
                                    Device{DeviceType::CPU, 0})),
                std::runtime_error);
}

TEST(RawTensorTest, construct_rejects_data_size_mismatch) {
   EXPECT_THROW(
       (DenseTensor<float>({2, 3}, std::vector<float>{1, 2, 3}, DType::FLOAT32,
                           Device{DeviceType::CPU, 0})),
       std::runtime_error);
}

// TODO: Depricate this test once multiple backends added
TEST(RawTensorTest, construct_rejects_non_cpu_device) {
   EXPECT_THROW(
       (DenseTensor<float>({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                           DType::FLOAT32, Device{DeviceType::CUDA, 0})),
       std::runtime_error);
}

TEST(RawTensorTest, dtype_size_matches_dtype) {
   DenseTensor<float> t({2}, std::vector<float>{1, 2}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   EXPECT_EQ(t.dtype(), DType::FLOAT32);
   EXPECT_EQ(t.dtype_size(), sizeof(float));
}

TEST(RawTensorTest, size_and_flat_size_match_number_of_elements) {
   DenseTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   EXPECT_EQ(t.size(), 6);
   EXPECT_EQ(t.flat_size(), 6);
}

TEST(RawTensorTest, get_ptr_returns_underlying_storage_pointer) {
   DenseTensor<float> t({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   EXPECT_EQ(t.get_ptr(), t.raw_data().data<float>());
}

TEST(RawTensorTest, begin_end_span_whole_tensor) {
   DenseTensor<float> t({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   EXPECT_EQ(t.end() - t.begin(), 4);
   EXPECT_FLOAT_EQ(t.begin()[0], 1.0);
   EXPECT_FLOAT_EQ(t.begin()[3], 4.0);
}

TEST(RawTensorTest, clear_sets_all_elements_to_zero) {
   DenseTensor<float> t({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   t.clear();

   EXPECT_FLOAT_EQ(t[0], 0.0);
   EXPECT_FLOAT_EQ(t[1], 0.0);
   EXPECT_FLOAT_EQ(t[2], 0.0);
   EXPECT_FLOAT_EQ(t[3], 0.0);
}

TEST(RawTensorTest, view_returns_matching_metadata_and_pointer) {
   DenseTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   TensorView<float> v = t.view();

   EXPECT_EQ(v.shape(), t.shape());
   EXPECT_EQ(v.strides(), t.strides());
   EXPECT_EQ(v.rank(), t.rank());
   EXPECT_EQ(v.ndims(), t.ndims());
   EXPECT_EQ(v.data(), t.get_ptr());
}

TEST(RawTensorTest, copy_construction_preserves_contents) {
   DenseTensor<float> a({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   DenseTensor<float> b(a);

   EXPECT_EQ(b.shape(), a.shape());
   EXPECT_EQ(b.strides(), a.strides());
   EXPECT_FLOAT_EQ(b[0], 1.0);
   EXPECT_FLOAT_EQ(b[3], 4.0);
}

TEST(RawTensorTest, move_construction_preserves_contents) {
   DenseTensor<float> a({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   DenseTensor<float> b(std::move(a));

   EXPECT_TRUE(b.is_initialised());
   EXPECT_EQ(b.shape(), (std::vector<std::size_t>{2, 2}));
   EXPECT_EQ(b.strides(), (std::vector<std::int64_t>{2, 1}));
   EXPECT_FLOAT_EQ(b[0], 1.0);
   EXPECT_FLOAT_EQ(b[3], 4.0);
}

TEST(RawTensorTest, constructed_tensor_has_single_ownership_of_storage) {
   DenseTensor<float> a({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   EXPECT_EQ(a.storage_use_count(), 1);
}

TEST(RawTensorTest, move_construction_transfers_single_ownership_of_storage) {
   DenseTensor<float> a({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   DenseTensor<float> b(std::move(a));
   EXPECT_EQ(b.storage_use_count(), 1);
   EXPECT_EQ(a.storage_use_count(), 0);
}

TEST(RawTensorTest, move_assignment_transfers_single_ownership) {
   DenseTensor<float> a({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   DenseTensor<float> b;
   b = std::move(a);
   EXPECT_EQ(b.storage_use_count(), 1);
   EXPECT_EQ(a.storage_use_count(), 0);
}

TEST(RawTensorTest, tensor_view_does_not_increment_use_count) {
   DenseTensor<float> a({2, 2}, std::vector<float>{1, 2, 3, 4}, DType::FLOAT32,
                        Device{DeviceType::CPU, 0});

   TensorView<float> b = a.view();

   EXPECT_EQ(a.storage_use_count(), 1);
}
