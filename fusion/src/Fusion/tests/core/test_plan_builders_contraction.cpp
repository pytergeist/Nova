#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/core/planning/PlanBuilders.h"

TEST(TensorPlanContractionTest, infer_einsum_out_shape_returns_matmul_shape) {
   OperandDescription a{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   const std::vector<std::size_t> out_shape =
       infer_out_shape_from_binding({a, b}, binding);

   EXPECT_EQ(out_shape, (std::vector<std::size_t>{2, 3}));
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_sets_output_shape) {
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
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };
   ContractionPlan plan =
       make_contraction_plan_einsum_out({out, a, b}, binding);

   EXPECT_EQ(plan.exec.core.num_operands, 3);
   EXPECT_EQ(plan.exec.core.out_ndim, 2);
   EXPECT_EQ(plan.exec.core.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(plan.exec.core.itemsize, sizeof(float));
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_builds_three_loops_for_matmul) {
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
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   ContractionPlan plan =
       make_contraction_plan_einsum_out({out, a, b}, binding);

   DenseTraversalPlan dense = std::get<DenseTraversalPlan>(plan.exec.traversal);

   ASSERT_EQ(dense.loop.size(), 3);

   EXPECT_EQ(dense.loop[0].size, 2);
   EXPECT_EQ(dense.loop[0].kind, IndexKind::Independent);

   EXPECT_EQ(dense.loop[1].size, 3);
   EXPECT_EQ(dense.loop[1].kind, IndexKind::Independent);

   EXPECT_EQ(dense.loop[2].size, 4);
   EXPECT_EQ(dense.loop[2].kind, IndexKind::Reduction);
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_assigns_m_n_k_roles_for_matmul) {
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
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   ContractionPlan plan =
       make_contraction_plan_einsum_out({out, a, b}, binding);
   DenseTraversalPlan dense = std::get<DenseTraversalPlan>(plan.exec.traversal);

   ASSERT_EQ(dense.loop.size(), 3);

   EXPECT_EQ(dense.loop[0].role, IndexRole::M);
   EXPECT_EQ(dense.loop[1].role, IndexRole::N);
   EXPECT_EQ(dense.loop[2].role, IndexRole::K);
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_detects_gemm_like_matmul) {
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
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   ContractionPlan plan =
       make_contraction_plan_einsum_out({out, a, b}, binding);

   EXPECT_TRUE(plan.exec.hints.gemm_like);

   EXPECT_EQ(plan.exec.hints.gemm.batch, 1);
   EXPECT_EQ(plan.exec.hints.gemm.M, 2);
   EXPECT_EQ(plan.exec.hints.gemm.N, 3);
   EXPECT_EQ(plan.exec.hints.gemm.K, 4);

   EXPECT_EQ(plan.exec.hints.gemm.out_rs, 3);
   EXPECT_EQ(plan.exec.hints.gemm.out_cs, 1);

   EXPECT_EQ(plan.exec.hints.gemm.a_rs, 4);
   EXPECT_EQ(plan.exec.hints.gemm.a_cs, 1);

   EXPECT_EQ(plan.exec.hints.gemm.b_rs, 3);
   EXPECT_EQ(plan.exec.hints.gemm.b_cs, 1);
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_rejects_wrong_output_shape) {
   OperandDescription out{
       .shape = {2, 4}, // incorrect shape (should be {2, 3}
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription a{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   EXPECT_THROW(make_contraction_plan_einsum_out({out, a, b}, binding),
                std::runtime_error);
}

TEST(TensorPlanContractionTest,
     infer_einsum_out_shape_rejects_mismatched_contract_dimension) {
   OperandDescription a{
       .shape = {2, 5},
       .strides = {5, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };
   // cannot infer out shape from input operand shapes
   EXPECT_THROW(infer_out_shape_from_binding({a, b}, binding),
                std::runtime_error);
}

TEST(TensorPlanContractionTest, infer_einsum_out_shape_rejects_mixed_itemsize) {
   OperandDescription a{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(double),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   EXPECT_THROW(infer_out_shape_from_binding({a, b}, binding),
                std::runtime_error);
}

TEST(TensorPlanContractionTest,
     infer_einsum_out_shape_rejects_repeated_label_within_operand) {
   OperandDescription a{
       .shape = {4, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 0}, // repeated label
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   EXPECT_THROW(infer_out_shape_from_binding({a, b}, binding),
                std::runtime_error);
}

TEST(TensorPlanContractionTest,
     infer_einsum_out_shape_rejects_output_label_missing_from_operands) {
   OperandDescription a{
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandDescription b{
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
       .layout = LayoutKind::Dense,
       .access = AccessKind::Affine,
       .storage = StorageKind::Owned,
       .update = UpdateKind::ReadOnly,
       .type = OperandDescType::Tensor,
   };

   OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 0},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 9},
   };
   EXPECT_THROW(infer_out_shape_from_binding({a, b}, binding),
                std::runtime_error);
}
