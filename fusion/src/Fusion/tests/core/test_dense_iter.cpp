#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContextBuilders.h"
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

   std::array<uint8_t *, 3> base = {reinterpret_cast<uint8_t *>(&out),
                                    reinterpret_cast<uint8_t *>(&a),
                                    reinterpret_cast<uint8_t *>(&b)};

   int calls = 0;

   fusion::dense::iter::for_each_outer_then_inner<3>(
       view, base, [&](fusion::dense::iter::DenseSegment<3> &segment) {
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

   std::array<uint8_t *, 3> base = {
       reinterpret_cast<uint8_t *>(outp),
       reinterpret_cast<uint8_t *>(ap),
       reinterpret_cast<uint8_t *>(bp),
   };

   int calls = 0;

   fusion::dense::iter::for_each_outer_then_inner<3>(
       view, base, [&](fusion::dense::iter::DenseSegment<3> &segment) {
          ++calls;
          EXPECT_EQ(segment.len, 3);
          EXPECT_EQ(segment.step[0].byte_stride, sizeof(float));
          EXPECT_EQ(segment.step[1].byte_stride, sizeof(float));
          EXPECT_EQ(segment.step[2].byte_stride, sizeof(float));
       });

   EXPECT_EQ(calls, 2);
}
