#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/compiler/planning/PlanBuilders.h"

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_single_operand_preserves_shape) {
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({a});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);

   EXPECT_EQ(plan.exec.core.num_operands, 1);
   EXPECT_EQ(plan.exec.core.out_ndim, 2);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));

   ASSERT_EQ(dense.loop.size(), 2);
   ASSERT_EQ(plan.exec.access.operands.size(), 1);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(dense.loop[0].kind, fusion::fuir::IndexKind::Independent);
   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(dense.loop[1].kind, fusion::fuir::IndexKind::Independent);
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_same_shape_preserves_output_shape) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({out, a, b});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);

   EXPECT_EQ(plan.exec.core.num_operands, 3);
   EXPECT_EQ(plan.exec.core.out_ndim, 2);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));

   ASSERT_EQ(dense.loop.size(), 2);
   ASSERT_EQ(plan.exec.access.operands.size(), 3);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.exec.access.operands[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_broadcasts_leading_dimension) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {1, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({out, a, b});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(dense.loop.size(), 2);
   ASSERT_EQ(plan.exec.core.num_operands, 3);
   ASSERT_EQ(plan.exec.access.operands.size(), 3);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.exec.access.operands[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_broadcasts_trailing_dimension) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {2, 1},
       .strides = {1, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({out, a, b});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(dense.loop.size(), 2);
   ASSERT_EQ(plan.exec.core.num_operands, 3);
   ASSERT_EQ(plan.exec.access.operands.size(), 3);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.exec.access.operands[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)), 0}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_right_aligns_lower_rank_operand) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {3},
       .strides = {1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({out, a, b});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(dense.loop.size(), 2);
   ASSERT_EQ(plan.exec.core.num_operands, 3);
   ASSERT_EQ(plan.exec.access.operands.size(), 3);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)),
             }));

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.exec.access.operands[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_supports_multiple_broadcast_operands) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription a{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {1, 3, 1},
       .strides = {3, 1, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription c{
       .shape = {4},
       .strides = {1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({out, a, b, c});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);
   EXPECT_EQ(plan.exec.core.num_operands, 4);
   EXPECT_EQ(plan.exec.core.out_ndim, 3);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3, 4}));

   ASSERT_EQ(dense.loop.size(), 3);
   ASSERT_EQ(plan.exec.access.operands.size(), 4);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 12 * static_cast<std::int64_t>(sizeof(float)),
                 4 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 12 * static_cast<std::int64_t>(sizeof(float)),
                 4 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(dense.loop[2].size, 4);
   EXPECT_EQ(plan.exec.access.operands[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 1 * static_cast<std::int64_t>(sizeof(float)), 0}));

   EXPECT_EQ(plan.exec.access.operands[3].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 0, 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_scalar_like_all_ones_shape_broadcasts) {
   fusion::fuir::OperandDescription out{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {1, 1},
       .strides = {1, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   fusion::planning::ElementwisePlan plan =
       fusion::planning::make_elementwise_plan({out, a, b});
   fusion::planning::DenseTraversalPlan dense =
       std::get<fusion::planning::DenseTraversalPlan>(plan.exec.traversal);

   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(dense.loop.size(), 2);
   ASSERT_EQ(plan.exec.access.operands.size(), 3);

   EXPECT_EQ(plan.exec.access.operands[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{12, 4}));
   EXPECT_EQ(plan.exec.access.operands[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{12, 4}));

   EXPECT_EQ(plan.exec.access.operands[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{0, 0}));
}

TEST(TensorPlanAlignedTest, make_elementwise_plan_rejects_mixed_itemsize) {
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(double),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_elementwise_plan({a, b}),
                std::runtime_error);
}

TEST(TensorPlanAlignedTest, make_elementwise_plan_rejects_shape_mismatch) {
   fusion::fuir::OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };
   fusion::fuir::OperandDescription b{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_elementwise_plan({a, b}),
                std::runtime_error);
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_rejects_bad_tensor_description_shape_rank_mismatch) {
   fusion::fuir::OperandDescription bad{
       .shape = {2, 3},
       .strides = {1},
       .itemsize = sizeof(float),
       .layout = fusion::core::LayoutKind::Dense,
       .access = fusion::fuir::AccessKind::Affine,
       .storage = fusion::fuir::StorageKind::Owned,
       .update = fusion::fuir::UpdateKind::ReadOnly,
       .type = fusion::fuir::OperandDescType::Tensor,
   };

   EXPECT_THROW(fusion::planning::make_elementwise_plan({bad}),
                std::runtime_error);
}

TEST(TensorPlanAlignedTest, make_elementwise_plan_rejects_empty_operands) {
   EXPECT_THROW(fusion::planning::make_elementwise_plan({}),
                std::runtime_error);
}