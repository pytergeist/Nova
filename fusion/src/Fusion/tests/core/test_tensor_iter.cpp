#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "Fusion/core/iter/TensorIter.hpp"
#include "Fusion/core/planning/PlanMeta.hpp"
#include "Fusion/core/tensor/DenseTensor.hpp"

#include "Fusion/cpu/simd/SimdTags.hpp"

TEST(TensorIterTest, for_each_outer_then_inner_with_zero_dim_calls_inner_once) {
   BroadcastPlan plan{};
   plan.num_operands = 3;
   plan.out_ndim = 0;
   plan.itemsize = sizeof(float);

   plan.op_access.resize(3);
   plan.op_access[0].access = AccessKind::Affine;
   plan.op_access[1].access = AccessKind::Affine;
   plan.op_access[2].access = AccessKind::Affine;

   float out = 0.0;
   float a = 1.0;
   float b = 1.0;

   std::array<uint8_t *, 3> base = {reinterpret_cast<uint8_t *>(&out),
                                    reinterpret_cast<uint8_t *>(&a),
                                    reinterpret_cast<uint8_t *>(&b)};

   int calls = 0;

   fusion::iter::for_each_outer_then_inner<BroadcastPlan, 3>(
       plan, base, [&](fusion::iter::InnerSegment<3> &segment) {
          ++calls;
          EXPECT_EQ(segment.len, 1);
          EXPECT_EQ(segment.ptrs[0], reinterpret_cast<uint8_t *>(&out));
          EXPECT_EQ(segment.ptrs[1], reinterpret_cast<uint8_t *>(&a));
          EXPECT_EQ(segment.ptrs[2], reinterpret_cast<uint8_t *>(&b));
          EXPECT_EQ(segment.step[0].byte_stride, 0);
          EXPECT_EQ(segment.step[1].byte_stride, 0);
          EXPECT_EQ(segment.step[2].byte_stride, 0);
       });

   EXPECT_EQ(calls, 1);
}

TEST(TensorIterTest,
     for_each_outer_then_inner_2_dim_calls_inner_per_outer_row) {
   BroadcastPlan plan{};
   plan.num_operands = 3;
   plan.out_ndim = 2;
   plan.itemsize = sizeof(float);

   plan.loop = {
       LoopDim{.size = 2, .kind = IndexKind::Independent},
       LoopDim{.size = 3, .kind = IndexKind::Independent},
   };

   plan.op_access.resize(3);
   for (auto &access : plan.op_access) {
      access.access = AccessKind::Affine;
      access.affine.byte_stride_per_loop = {
          static_cast<std::int64_t>(3 * sizeof(float)),
          static_cast<std::int64_t>(sizeof(float)),
      };
   }

   TensorBuffer out;
   out.allocate_elements_with<float>(&default_allocator(), 6);
   TensorBuffer a;
   a.allocate_elements_with<float>(&default_allocator(), 6);
   TensorBuffer b;
   b.allocate_elements_with<float>(&default_allocator(), 6);

   float *outp = out.data<float>();
   float *ap = a.data<float>();
   float *bp = b.data<float>();

   std::array<uint8_t *, 3> base = {
       reinterpret_cast<uint8_t *>(outp),
       reinterpret_cast<uint8_t *>(ap),
       reinterpret_cast<uint8_t *>(bp),
   };

   int calls = 0;

   fusion::iter::for_each_outer_then_inner<BroadcastPlan, 3>(
       plan, base, [&](fusion::iter::InnerSegment<3> &segment) {
          ++calls;
          EXPECT_EQ(segment.len, 3);
          EXPECT_EQ(segment.step[0].byte_stride, sizeof(float));
          EXPECT_EQ(segment.step[1].byte_stride, sizeof(float));
          EXPECT_EQ(segment.step[2].byte_stride, sizeof(float));
       });

   EXPECT_EQ(calls, 2);
}

TEST(TensorIterTest, tag_fallback_binary_respects_strides) {
   const float a[] = {1., 99., 2., 99., 3., 99.};
   const float b[] = {10., 88., 20., 88., 30., 88.};
   float out[] = {0., 0., 0.};

   fusion::iter::tag_fallback_binary<float, AddSIMD>(out, a, b,
                                                     1, // out stride
                                                     2, // a stride
                                                     2, // b stride
                                                     3  // len
   );

   EXPECT_FLOAT_EQ(out[0], 11.);
   EXPECT_FLOAT_EQ(out[1], 22.);
   EXPECT_FLOAT_EQ(out[2], 33.);
}

TEST(TensorIterTest, tag_fallback_unary_respects_strides) {
   const float a[] = {4., 99., 9., 99., 16., 99.};
   float out[] = {0., 0., 0.};

   fusion::iter::tag_fallback_unary<float, SqrtSIMD>(out, a,
                                                     1, // out stride
                                                     2, // a stride
                                                     3  // len
   );
   EXPECT_FLOAT_EQ(out[0], 2.);
   EXPECT_FLOAT_EQ(out[1], 3.);
   EXPECT_FLOAT_EQ(out[2], 4.);
}

TEST(TensorIterTest, tag_fallback_reduction_accumulates_into_output) {
   const float a[] = {1., 2., 3., 4., 5.};
   float out[] = {1.};
   fusion::iter::tag_fallback_reduction<float, SumSIMD>(out, a, 0, 1, 6);

   EXPECT_FLOAT_EQ(out[0], 16.);
}

TEST(TensorIterTest, tag_fallback_reduction_respects_strides) {
   const float a[] = {1., 2., 3., 4., 5.};
   float out[] = {1.};
   fusion::iter::tag_fallback_reduction<float, SumSIMD>(out, a, 0, 2, 3);

   EXPECT_FLOAT_EQ(out[0], 10.);
}

TEST(TensorIterTest, tag_fallback_contraction_accumulates_products) {
   const float a[] = {1., 90., 2., 33., 3., 99.};
   const float b[] = {4., 67., 5., 14., 7., 88.};
   float out[] = {1.};

   fusion::iter::tag_fallback_contraction<float, MultiplySIMD>(
       out, a, b,
       0, // accumulate into single output element
       2, // a stride
       2, // b stride
       3  // len
   );
   EXPECT_EQ(out[0], 1. + 1. * 4. + 2. * 5. + 3. * 7.);
}

TEST(TensorIterTest, binary_ewise_tag_fastpath_computes_elementwise_add) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> b({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   BinaryEwiseMeta meta = make_binary_meta(a, b);
   EXPECT_EQ(meta.exec, BinaryExecKind::FlatContiguous);
   EXPECT_EQ(meta.fast_len, 6);

   fusion::iter::binary_ewise_tag<float, AddSIMD>(a, b, meta, out);

   EXPECT_FLOAT_EQ(out[0], 2.);
   EXPECT_FLOAT_EQ(out[1], 4.);
   EXPECT_FLOAT_EQ(out[2], 6.);
   EXPECT_FLOAT_EQ(out[3], 8.);
   EXPECT_FLOAT_EQ(out[4], 10.);
   EXPECT_FLOAT_EQ(out[5], 12.);
}

TEST(TensorIterTest, binary_ewise_tag_broadcast_path_computes_elementwise_add) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> b({1, 3}, std::vector<float>{1, 2, 3}, DType::FLOAT32,
                      Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   BinaryEwiseMeta meta = make_binary_meta(a, b);
   ASSERT_EQ(meta.exec, BinaryExecKind::FlatContiguousBroadcastRHS);
   ASSERT_EQ(meta.out_shape, (std::vector<size_t>{2, 3}));

   fusion::iter::binary_ewise_tag<float, AddSIMD>(a, b, meta, out);

   EXPECT_FLOAT_EQ(out[0], 2.);
   EXPECT_FLOAT_EQ(out[1], 4.);
   EXPECT_FLOAT_EQ(out[2], 6.);
   EXPECT_FLOAT_EQ(out[3], 5.);
   EXPECT_FLOAT_EQ(out[4], 7.);
   EXPECT_FLOAT_EQ(out[5], 9.);
}

TEST(TensorIterTest, unary_ewise_tag_fastpath_computes_square) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 4, 9, 16, 25, 36},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   UnaryEwiseMeta meta = make_unary_meta(a);
   ASSERT_TRUE(meta.fastpath);
   ASSERT_EQ(meta.fast_len, 6);

   fusion::iter::unary_ewise_tag<float, SqrtSIMD>(a, meta, out);

   EXPECT_FLOAT_EQ(out[0], 1.);
   EXPECT_FLOAT_EQ(out[1], 2.);
   EXPECT_FLOAT_EQ(out[2], 3.);
   EXPECT_FLOAT_EQ(out[3], 4.);
   EXPECT_FLOAT_EQ(out[4], 5.);
   EXPECT_FLOAT_EQ(out[5], 6.);
}

TEST(TensorIterTest, reduction_tag_global_fastpath_sums_all_elements) {
   DenseTensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({1}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   ReductionMeta meta = make_reduction_meta(a, kGlobalReduceAxis, false);
   ASSERT_TRUE(meta.fastpath);
   ASSERT_EQ(meta.fast_len, 6);

   fusion::iter::reduction_tag<float, SumSIMD>(a, meta, out);

   EXPECT_FLOAT_EQ(out[0], 21.);
}

TEST(TensorIterTest, reduction_tag_axis_path_reduces_requested_axis) {
   DenseTensor<float> a({2, 3},
                      std::vector<float>{
                          1,
                          2,
                          3,
                          4,
                          5,
                          6,
                      },
                      DType::FLOAT32, Device{DeviceType::CPU, 0});
   DenseTensor<float> out({2}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   ReductionMeta meta = make_reduction_meta(a, 1, false);
   ASSERT_FALSE(meta.fastpath);
   ASSERT_EQ(meta.out_shape, (std::vector<std::size_t>{2}));

   fusion::iter::reduction_tag<float, SumSIMD>(a, meta, out);

   EXPECT_FLOAT_EQ(out[0], 6.);
   EXPECT_FLOAT_EQ(out[1], 15.);
}

TEST(TensorIterTest, contraction_tag_computes_matrix_multiplication_result) {
   DenseTensor<float> a({2, 4},
                      std::vector<float>{
                          1,
                          2,
                          3,
                          4,
                          5,
                          6,
                          7,
                          8,
                      },
                      DType::FLOAT32, Device{DeviceType::CPU, 0});

   DenseTensor<float> b({4, 3},
                      std::vector<float>{
                          1,
                          2,
                          3,
                          4,
                          5,
                          6,
                          7,
                          8,
                          9,
                          10,
                          11,
                          12,
                      },
                      DType::FLOAT32, Device{DeviceType::CPU, 0});

   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

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

   fusion::iter::contraction_tag<float, MultiplySIMD, MultiplySIMD>(a, b, meta,
                                                                    out);

   EXPECT_FLOAT_EQ(out[0], 70.);
   EXPECT_FLOAT_EQ(out[1], 80.);
   EXPECT_FLOAT_EQ(out[2], 90.);
   EXPECT_FLOAT_EQ(out[3], 158.);
   EXPECT_FLOAT_EQ(out[4], 184.);
   EXPECT_FLOAT_EQ(out[5], 210.);
}