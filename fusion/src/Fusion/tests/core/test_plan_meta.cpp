#include <gtest/gtest.h>

#include <vector>

#include "Fusion/core/PlanMeta.hpp"
#include "Fusion/core/RawTensor.hpp"

TEST(PlanMetaTest, contig_elem_strides_returns_row_major_strides_for_2d_shape) {
   std::vector<std::size_t> shape{2, 3};
   EXPECT_EQ(contig_elem_strides(shape), (std::vector<std::int64_t>{3, 1}));
}

TEST(PlanMetaTest, contig_elem_strides_returns_row_major_strides_for_3d_shape) {
   std::vector<std::size_t> shape{2, 3, 4};
   EXPECT_EQ(contig_elem_strides(shape), (std::vector<std::int64_t>{12, 4, 1}));
}

TEST(PlanMetaTest, contig_elem_strides_returns_row_major_strides_for_4d_shape) {
   std::vector<std::size_t> shape{2, 3, 4, 5};
   EXPECT_EQ(contig_elem_strides(shape),
             (std::vector<std::int64_t>{60, 20, 5, 1}));
}

TEST(PlanMetaTest, make_desc_from_shape_uses_provided_strides) {
   std::vector<std::size_t> shape{2, 3};
   const std::int64_t strides[] = {5, 1};

   OperandDescription desc = make_desc_from_shape<float>(shape, strides);

   EXPECT_EQ(desc.ndims(), 2);
   EXPECT_EQ(desc.shape, shape);
   EXPECT_EQ(desc.strides, (std::vector<std::int64_t>{5, 1}));
   EXPECT_EQ(desc.itemsize, sizeof(float));
}

TEST(PlanMetaTest,
     make_desc_from_shape_builds_contiguous_strides_when_strides_null) {
   std::vector<std::size_t> shape{2, 3, 4};

   OperandDescription desc = make_desc_from_shape<float>(shape, nullptr);

   EXPECT_EQ(desc.ndims(), 3);
   EXPECT_EQ(desc.shape, shape);
   EXPECT_EQ(desc.strides, (std::vector<std::int64_t>{12, 4, 1}));
   EXPECT_EQ(desc.itemsize, sizeof(float));
}

TEST(PlanMetaTest, make_desc_from_tensor_copies_shape_and_strides) {
   RawTensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});

   OperandDescription desc = make_desc_from_tensor(t);

   EXPECT_EQ(desc.ndims(), 2);
   EXPECT_EQ(desc.shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(desc.strides, (std::vector<std::int64_t>{3, 1}));
   EXPECT_EQ(desc.itemsize, sizeof(float));
}

TEST(PlanMetaTest,
     make_binary_meta_uses_fastpath_for_same_shape_contiguous_inputs) {
   RawTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> b({2, 3}, std::vector<float>{6, 5, 4, 3, 2, 1},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});

   BinaryEwiseMeta meta = make_binary_meta(a, b);

   EXPECT_EQ(meta.exec, BinaryExecKind::FlatContiguous);
   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(meta.fast_len, 6);
}

TEST(PlanMetaTest,
     make_binary_meta_builds_broadcast_output_shape_for_broadcast_case) {
   RawTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   RawTensor<float> b({1, 3}, std::vector<float>{10, 20, 30}, DType::FLOAT32,
                      Device{DeviceType::CPU, 0});

   BinaryEwiseMeta meta = make_binary_meta(a, b);

   EXPECT_NE(meta.exec, BinaryExecKind::FlatContiguous);
   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(meta.dA.shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(meta.dB.shape, (std::vector<std::size_t>{1, 3}));
   EXPECT_EQ(meta.dOut.shape, (std::vector<std::size_t>{2, 3}));
}

TEST(PlanMetaTest, make_unary_meta_uses_fastpath_for_contiguous_input) {
   RawTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   UnaryEwiseMeta meta = make_unary_meta(a);

   EXPECT_TRUE(meta.fastpath);
   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(meta.fast_len, 6);
}

TEST(PlanMetaTest,
     make_reduction_meta_global_reduce_without_keepdim_uses_fastpath) {
   RawTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});

   ReductionMeta meta = make_reduction_meta(a, kGlobalReduceAxis, false);

   EXPECT_TRUE(meta.fastpath);
   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{1}));
   EXPECT_EQ(meta.fast_len, 6);
   EXPECT_EQ(meta.reduce_len, 6);
}

TEST(
    PlanMetaTest,
    make_reduction_meta_axis_reduce_without_keepdim_sets_expected_output_shape) {
   RawTensor<float> a({2, 3, 4}, std::vector<float>(24, 1), DType::FLOAT32,
                      Device{DeviceType::CPU, 0});

   ReductionMeta meta = make_reduction_meta(a, 1, false);

   EXPECT_FALSE(meta.fastpath);
   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{2, 4}));
   EXPECT_EQ(meta.reduction_axis, 1);
   EXPECT_FALSE(meta.keepdim);
   EXPECT_EQ(meta.reduce_len, 3);
}

TEST(PlanMetaTest,
     make_reduction_meta_axis_reduce_with_keepdim_sets_expected_output_shape) {
   RawTensor<float> a({2, 3, 4}, std::vector<float>(24, 1.0), DType::FLOAT32,
                      Device{DeviceType::CPU, 0});

   ReductionMeta meta = make_reduction_meta(a, 1, true);

   EXPECT_FALSE(meta.fastpath);
   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{2, 1, 4}));
   EXPECT_EQ(meta.reduction_axis, 1);
   EXPECT_TRUE(meta.keepdim);
   EXPECT_EQ(meta.reduce_len, 3);
}

TEST(PlanMetaTest, make_contraction_meta_einsum_infers_matmul_output_shape) {
   RawTensor<float> a({2, 4}, std::vector<float>(8, 1.0), DType::FLOAT32,
                      Device{DeviceType::CPU, 0});
   RawTensor<float> b({4, 3}, std::vector<float>(12, 1.0), DType::FLOAT32,
                      Device{DeviceType::CPU, 0});

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1}, // Out: [i, j]
               {0, 2}, // A:     [i, k]
               {2, 1}, // B:     [k, j]
           },
       .out_labels = {0, 1},
   };

   ContractionMeta meta = make_contraction_meta_einsum(a, b, binding);

   EXPECT_EQ(meta.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(meta.dA.shape, (std::vector<std::size_t>{2, 4}));
   EXPECT_EQ(meta.dB.shape, (std::vector<std::size_t>{4, 3}));
   EXPECT_EQ(meta.dOut.shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_TRUE(meta.fastpath);
}

TEST(PlanMetaTest, make_contraction_meta_einsum_stores_binding) {
   RawTensor<float> a({2, 4}, std::vector<float>(8, 1.0f), DType::FLOAT32,
                      Device{DeviceType::CPU, 0});
   RawTensor<float> b({4, 3}, std::vector<float>(12, 1.0f), DType::FLOAT32,
                      Device{DeviceType::CPU, 0});

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1}, // Out: [i, j]
               {0, 2}, // A:     [i, k]
               {2, 1}, // B:     [k, j]
           },
       .out_labels = {0, 1},
   };

   ContractionMeta meta = make_contraction_meta_einsum(a, b, binding);

   EXPECT_EQ(meta.binding.out_labels, (std::vector<Label>{0, 1}));
   ASSERT_EQ(meta.binding.op_axis_labels.size(), 3);
   EXPECT_EQ(meta.binding.op_axis_labels[0], (std::vector<Label>{0, 1}));
   EXPECT_EQ(meta.binding.op_axis_labels[1], (std::vector<Label>{0, 2}));
   EXPECT_EQ(meta.binding.op_axis_labels[2], (std::vector<Label>{2, 1}));
}