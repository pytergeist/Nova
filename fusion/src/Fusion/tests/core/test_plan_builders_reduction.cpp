#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/core/planning/PlanBuilders.h"

TEST(TensorPlanReductionTest,
     make_reduction_plan_keepdim_false_reduces_middle_axis) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 1, false);

   EXPECT_EQ(plan.exec.core.num_operands, 2);
   EXPECT_EQ(plan.exec.core.out_ndim, 2);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 4}));
   EXPECT_EQ(plan.reduction_axis, 1);
   EXPECT_FALSE(plan.keep_dim);
   EXPECT_EQ(plan.exec.core.itemsize, sizeof(float));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_keepdim_true_reduces_middle_axis) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 1, 4},
       .strides = {4, 1, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 1, true);

   EXPECT_EQ(plan.exec.core.num_operands, 2);
   EXPECT_EQ(plan.exec.core.out_ndim, 3);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 1, 4}));
   EXPECT_EQ(plan.reduction_axis, 1);
   EXPECT_TRUE(plan.keep_dim);
   EXPECT_EQ(plan.exec.core.itemsize, sizeof(float));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_keepdim_true_rejects_bad_out_shape) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_reduction_plan({out, in}, 1, true),
                std::runtime_error);
}

TEST(TensorPlanReductionTest, make_reduction_plan_normalises_negative_axis) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan = fusion::planning::make_reduction_plan(
       {out, in}, static_cast<std::size_t>(-1), false);

   EXPECT_EQ(plan.reduction_axis, 2);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_builds_independent_loops_then_reduction_loop) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 1, false);
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   ASSERT_EQ(dense.loop.size(), 3);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(dense.loop[0].kind, fusion::fuir::IndexKind::Independent);

   EXPECT_EQ(dense.loop[1].size, 4);
   EXPECT_EQ(dense.loop[1].kind, fusion::fuir::IndexKind::Independent);

   EXPECT_EQ(dense.loop[2].size, 3);
   EXPECT_EQ(dense.loop[2].kind, fusion::fuir::IndexKind::Reduction);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_sets_stride_bytes_for_keepdim_false) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 1, false);
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   ASSERT_EQ(dense.loop.size(), 3);
   ASSERT_EQ(plan.exec.access.operands.size(), 2);

   const std::size_t item = static_cast<std::int64_t>(sizeof(float));

   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{4 * item, 1 * item, 0}));

   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{12 * item, 1 * item, 4 * item}));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_sets_stride_bytes_for_keepdim_true) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 1, 4},
       .strides = {4, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 1, true);
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   ASSERT_EQ(dense.loop.size(), 3);
   ASSERT_EQ(plan.exec.access.operands.size(), 2);

   const std::size_t item = static_cast<std::int64_t>(sizeof(float));

   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{4 * item, 1 * item, 0}));
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{12 * item, 1 * item, 4 * item}));
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_reducing_first_axis_without_keepdim_is_valid) {
   fusion::fuir::OperandDescription out{
       .shape = {3, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 0, false);
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);

   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{3, 4}));
   EXPECT_EQ(plan.reduction_axis, 0);

   ASSERT_EQ(dense.loop.size(), 3);
   EXPECT_EQ(dense.loop[0].size, 3);
   EXPECT_EQ(dense.loop[1].size, 4);
   EXPECT_EQ(dense.loop[2].size, 2);
   EXPECT_EQ(dense.loop[2].kind, fusion::fuir::IndexKind::Reduction);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_reducing_last_axis_without_keepdim_is_valid) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ReductionPlan plan =
       fusion::planning::make_reduction_plan({out, in}, 2, false);
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(plan.reduction_axis, 2);

   ASSERT_EQ(dense.loop.size(), 3);
   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(dense.loop[2].size, 4);
   EXPECT_EQ(dense.loop[2].kind, fusion::fuir::IndexKind::Reduction);
}

TEST(TensorPlanReductionTest, make_reduction_plan_rejects_empty_descs) {
   EXPECT_THROW(fusion::planning::make_reduction_plan({}, 0, false),
                std::runtime_error);
}

TEST(TensorPlanReductionTest, make_reduction_plan_rejects_axis_out_of_range) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_reduction_plan({out, in}, 3, false),
                std::runtime_error);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_rejects_keepdim_false_with_wrong_output_rank) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 1, 4},
       .strides = {4, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_reduction_plan({out, in}, 1, false),
                std::runtime_error);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_rejects_keepdim_true_with_wrong_output_rank) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_reduction_plan({out, in}, 1, true),
                std::runtime_error);
}

TEST(
    TensorPlanReductionTest,
    make_reduction_plan_rejects_keepdim_true_when_reduced_axis_not_one_in_output) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_reduction_plan({out, in}, 1, true),
                std::runtime_error);
}

TEST(TensorPlanReductionTest, make_reduction_plan_rejects_mixed_itemsize) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(double),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_reduction_plan({out, in}, 1, false),
                std::runtime_error);
}

TEST(TensorPlanReductionTest,
     make_reduction_plan_rejects_input_rank_mismatch_across_inputs) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in0{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::fuir::OperandDescription in1{
       .shape = {3, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::fuir::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(
       fusion::planning::make_reduction_plan({out, in0, in1}, 1, false),
       std::runtime_error);
}