#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/core/planning/TensorPlan.h"

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_single_operand_preserves_shape) {
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({a});

   EXPECT_EQ(plan.core.num_operands, 1);
   EXPECT_EQ(plan.core.out_ndim, 2);
   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3}));

   ASSERT_EQ(plan.core.loop.size(), 2);
   ASSERT_EQ(plan.core.op_access.size(), 1);

   EXPECT_EQ(plan.core.loop[0].size, 2);
   EXPECT_EQ(plan.core.loop[0].kind, IndexKind::Independent);
   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.loop[1].size, 3);
   EXPECT_EQ(plan.core.loop[1].kind, IndexKind::Independent);
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_same_shape_preserves_output_shape) {
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
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({out, a, b});

   EXPECT_EQ(plan.core.num_operands, 3);
   EXPECT_EQ(plan.core.out_ndim, 2);
   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3}));

   ASSERT_EQ(plan.core.loop.size(), 2);
   ASSERT_EQ(plan.core.op_access.size(), 3);

   EXPECT_EQ(plan.core.loop[0].size, 2);
   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.loop[1].size, 3);
   EXPECT_EQ(plan.core.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_broadcasts_leading_dimension) {
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
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {1, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({out, a, b});

   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.core.loop.size(), 2);
   ASSERT_EQ(plan.core.num_operands, 3);
   ASSERT_EQ(plan.core.op_access.size(), 3);

   EXPECT_EQ(plan.core.loop[0].size, 2);
   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.loop[1].size, 3);
   EXPECT_EQ(plan.core.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_broadcasts_trailing_dimension) {
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
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {2, 1},
       .strides = {1, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({out, a, b});

   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.core.loop.size(), 2);
   ASSERT_EQ(plan.core.num_operands, 3);
   ASSERT_EQ(plan.core.op_access.size(), 3);

   EXPECT_EQ(plan.core.loop[0].size, 2);
   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.loop[1].size, 3);
   EXPECT_EQ(plan.core.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)), 0}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_right_aligns_lower_rank_operand) {
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
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {3},
       .strides = {1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({out, a, b});

   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.core.loop.size(), 2);
   ASSERT_EQ(plan.core.num_operands, 3);
   ASSERT_EQ(plan.core.op_access.size(), 3);

   EXPECT_EQ(plan.core.loop[0].size, 2);
   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)),
             }));

   EXPECT_EQ(plan.core.loop[1].size, 3);
   EXPECT_EQ(plan.core.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_supports_multiple_broadcast_operands) {
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
   OperandDescription a{
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {1, 3, 1},
       .strides = {3, 1, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription c{
       .shape = {4},
       .strides = {1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({out, a, b, c});

   EXPECT_EQ(plan.core.num_operands, 4);
   EXPECT_EQ(plan.core.out_ndim, 3);
   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3, 4}));

   ASSERT_EQ(plan.core.loop.size(), 3);
   ASSERT_EQ(plan.core.op_access.size(), 4);

   EXPECT_EQ(plan.core.loop[0].size, 2);
   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 12 * static_cast<std::int64_t>(sizeof(float)),
                 4 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.loop[1].size, 3);
   EXPECT_EQ(plan.core.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 12 * static_cast<std::int64_t>(sizeof(float)),
                 4 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.core.loop[2].size, 4);
   EXPECT_EQ(plan.core.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 1 * static_cast<std::int64_t>(sizeof(float)), 0}));

   EXPECT_EQ(plan.core.op_access[3].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{
                 0, 0, 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_scalar_like_all_ones_shape_broadcasts) {
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
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {1, 1},
       .strides = {1, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   ElementWisePlan plan = make_elementwise_plan({out, a, b});

   EXPECT_EQ(plan.core.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.core.loop.size(), 2);
   ASSERT_EQ(plan.core.op_access.size(), 3);

   EXPECT_EQ(plan.core.op_access[0].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{12, 4}));
   EXPECT_EQ(plan.core.op_access[1].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{12, 4}));

   EXPECT_EQ(plan.core.op_access[2].affine.byte_stride_per_loop,
             (std::vector<std::int64_t>{0, 0}));
}

TEST(TensorPlanAlignedTest, make_elementwise_plan_rejects_mixed_itemsize) {
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(double),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_elementwise_plan({a, b}), std::runtime_error);
}

TEST(TensorPlanAlignedTest, make_elementwise_plan_rejects_shape_mismatch) {
   OperandDescription a{
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };
   OperandDescription b{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_elementwise_plan({a, b}), std::runtime_error);
}

TEST(TensorPlanAlignedTest,
     make_elementwise_plan_rejects_bad_tensor_description_shape_rank_mismatch) {
   OperandDescription bad{
       .shape = {2, 3},
       .strides = {1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   EXPECT_THROW(make_elementwise_plan({bad}), std::runtime_error);
}

TEST(TensorPlanAlignedTest, make_elementwise_plan_rejects_empty_operands) {
   EXPECT_THROW(make_elementwise_plan({}), std::runtime_error);
}