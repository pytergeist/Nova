#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "Fusion/execution/iter/DenseIter.hpp"
#include "Fusion/compiler/planning/OpContextBuilders.h"
#include "Fusion/core/tensor/DenseTensor.hpp"

TEST(DenseIterTest, for_each_outer_then_inner_with_zero_dim_calls_inner_once) {

   fusion::planning::ElementwisePlan plan{};
   plan.exec.core.num_operands = 3;
   plan.exec.core.out_ndim = 0;
   plan.exec.core.itemsize = sizeof(float);

   plan.exec.access.operands.resize(3);
   plan.exec.access.operands[0].access = fusion::fuir::AccessKind::Affine;
   plan.exec.access.operands[1].access = fusion::fuir::AccessKind::Affine;
   plan.exec.access.operands[2].access = fusion::fuir::AccessKind::Affine;

   struct DenseIterPlanView {
      std::size_t num_operands{};
      std::span<const fusion::fuir::LoopDim> loop;
      std::span<const fusion::fuir::OperandAccess> operands;
   };

   fusion::dense::iter::DenseIterPlanView view =
       fusion::dense::iter::dense_iter_view(plan);

   float out = 0.0;
   float a = 1.0;
   float b = 1.0;

   std::array<std::byte *, 1> outputs{
       reinterpret_cast<std::byte *>(&out),
   };

   std::array<const std::byte *, 2> inputs{
       reinterpret_cast<const std::byte *>(&a),
       reinterpret_cast<const std::byte *>(&b),
   };

   int calls = 0;

   fusion::dense::iter::for_each_outer_then_inner<2, 1>(
       view, outputs, inputs,
       [&](const fusion::dense::iter::DenseSegmentView<2, 1> &segment) {
          ++calls;
          EXPECT_EQ(segment.len, 1);
          EXPECT_EQ(segment.outputs[0], reinterpret_cast<std::byte *>(&out));
          EXPECT_EQ(segment.inputs[0], reinterpret_cast<std::byte *>(&a));
          EXPECT_EQ(segment.inputs[1], reinterpret_cast<std::byte *>(&b));
          EXPECT_EQ(segment.output_byte_stride[0].stride, 0);
          EXPECT_EQ(segment.output_byte_stride[0].stride, 0);
          EXPECT_EQ(segment.output_byte_stride[1].stride, 0);
       });

   EXPECT_EQ(calls, 1);
}

TEST(DenseIterTest, for_each_outer_then_inner_2_dim_calls_inner_per_outer_row) {
   fusion::planning::ElementwisePlan plan{};
   plan.exec.core.num_operands = 3;
   plan.exec.core.out_ndim = 2;
   plan.exec.core.itemsize = sizeof(float);

   fusion::planning::DenseTraversalPlan &dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   dense.loop = {
       fusion::fuir::LoopDim{.size = 2,
                             .kind = fusion::fuir::IndexKind::Independent},
       fusion::fuir::LoopDim{.size = 3,
                             .kind = fusion::fuir::IndexKind::Independent},
   };

   plan.exec.access.operands.resize(3);
   fusion::dense::iter::DenseIterPlanView view =
       fusion::dense::iter::dense_iter_view(plan);

   for (auto &access : plan.exec.access.operands) {
      access.access = fusion::fuir::AccessKind::Affine;
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

   std::array<std::byte *, 1> outputs{
       reinterpret_cast<std::byte *>(&outp),
   };

   std::array<const std::byte *, 2> inputs{
       reinterpret_cast<const std::byte *>(&ap),
       reinterpret_cast<const std::byte *>(&bp),
   };
   int calls = 0;

   fusion::dense::iter::for_each_outer_then_inner<2, 1>(
       view, outputs, inputs,
       [&](const fusion::dense::iter::DenseSegmentView<2, 1> &segment) {
          ++calls;
          EXPECT_EQ(segment.len, 3);
          EXPECT_EQ(segment.output_byte_stride[0].stride, sizeof(float));
          EXPECT_EQ(segment.input_byte_stride[0].stride, sizeof(float));
          EXPECT_EQ(segment.input_byte_stride[1].stride, sizeof(float));
       });

   EXPECT_EQ(calls, 2);
}
