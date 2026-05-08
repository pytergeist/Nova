#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/core/TensorPlan.h"

TEST(TensorPlanReductionTest,
     make_reduction_plan_keepdim_false_reduces_middle_axis) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
      .shape = {2, 3, 4},
      .strides = {12, 4, 1},
      .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 1, false);

   EXPECT_EQ(plan.num_operands, 2);
   EXPECT_EQ(plan.out_ndim, 2);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 4}));
   EXPECT_EQ(plan.reduction_axis, 1);
   EXPECT_FALSE(plan.keep_dim);
   EXPECT_EQ(plan.itemsize, sizeof(float));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_keepdim_true_reduces_middle_axis) {
   OperandDescription out{
       .shape = {2, 1, 4},
       .strides = {4, 1, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 1, true);

   EXPECT_EQ(plan.num_operands, 2);
   EXPECT_EQ(plan.out_ndim, 3);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 1, 4}));
   EXPECT_EQ(plan.reduction_axis, 1);
   EXPECT_TRUE(plan.keep_dim);
   EXPECT_EQ(plan.itemsize, sizeof(float));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_keepdim_true_rejects_bad_out_shape) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in}, 1, true), std::runtime_error);
}

TEST(TensorPlanReductionTest, make_reduction_plan_normalises_negative_axis) {
   OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan =
       make_reduction_plan({out, in}, static_cast<std::size_t>(-1), false);

   EXPECT_EQ(plan.reduction_axis, 2);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_builds_independent_loops_then_reduction_loop) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 1, false);

   ASSERT_EQ(plan.loop.size(), 3);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].kind, IndexKind::Independent);

   EXPECT_EQ(plan.loop[1].size, 4);
   EXPECT_EQ(plan.loop[1].kind, IndexKind::Independent);

   EXPECT_EQ(plan.loop[2].size, 3);
   EXPECT_EQ(plan.loop[2].kind, IndexKind::Reduction);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_sets_stride_bytes_for_keepdim_false) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 1, false);

   ASSERT_EQ(plan.loop.size(), 3);
   ASSERT_EQ(plan.op_access.size(), 3);

   const std::size_t item = static_cast<std::int64_t>(sizeof(float));

   EXPECT_EQ(plan.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{4 * item, 12 * item}));

   EXPECT_EQ(plan.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{1 * item, 1 * item}));

   EXPECT_EQ(plan.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{0, 4 * item}));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_sets_stride_bytes_for_keepdim_true) {
   OperandDescription out{
       .shape = {2, 1, 4},
       .strides = {4, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 1, true);

   ASSERT_EQ(plan.loop.size(), 3);

   const std::size_t item = static_cast<std::int64_t>(sizeof(float));

   EXPECT_EQ(plan.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{4 * item, 12 * item}));
   EXPECT_EQ(plan.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{1 * item, 1 * item}));
   EXPECT_EQ(plan.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{0 * item, 4 * item}));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_reducing_first_axis_without_keepdim_is_valid) {
   OperandDescription out{
       .shape = {3, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 0, false);

   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{3, 4}));
   EXPECT_EQ(plan.reduction_axis, 0);

   ASSERT_EQ(plan.loop.size(), 3);
   EXPECT_EQ(plan.loop[0].size, 3);
   EXPECT_EQ(plan.loop[1].size, 4);
   EXPECT_EQ(plan.loop[2].size, 2);
   EXPECT_EQ(plan.loop[2].kind, IndexKind::Reduction);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_reducing_last_axis_without_keepdim_is_valid) {
   OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   ReductionPlan plan = make_reduction_plan({out, in}, 2, false);

   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(plan.reduction_axis, 2);

   ASSERT_EQ(plan.loop.size(), 3);
   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[2].size, 4);
   EXPECT_EQ(plan.loop[2].kind, IndexKind::Reduction);
}

TEST(TensorPlanReductionTest, make_reduction_plan_rejects_empty_descs) {
   EXPECT_THROW(make_reduction_plan({}, 0, false), std::runtime_error);
}

TEST(TensorPlanReductionTest, make_reduction_plan_rejects_axis_out_of_range) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in}, 3, false), std::runtime_error);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_rejects_keepdim_false_with_wrong_output_rank) {
   OperandDescription out{
       .shape = {2, 1, 4},
       .strides = {4, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in}, 1, false), std::runtime_error);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_rejects_keepdim_true_with_wrong_output_rank) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in}, 1, true), std::runtime_error);
}

TEST(
    TensorPlanReductionTest,
    make_reduction_plan_rejects_keepdim_true_when_reduced_axis_not_one_in_output) {
   OperandDescription out{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in}, 1, true), std::runtime_error);
}

TEST(TensorPlanReductionTest, make_reduction_plan_rejects_mixed_itemsize) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(double),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in}, 1, false), std::runtime_error);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_rejects_input_rank_mismatch_across_inputs) {
   OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in0{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   OperandDescription in1{
       .shape = {3, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
      .layout = LayoutKind::Dense,
      .access = AccessKind::Affine,
      .storage = StorageKind::Owned,
      .update = UpdateKind::ReadOnly,
      .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_reduction_plan({out, in0, in1}, 1, false),
                std::runtime_error);
}