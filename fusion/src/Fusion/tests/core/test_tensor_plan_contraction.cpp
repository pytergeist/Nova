#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/core/TensorPlan.h"

TEST(TensorPlanContractionTest, infer_einsum_out_shape_returns_matmul_shape) {
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   const std::vector<std::size_t> out_shape =
       infer_einsum_out_shape({a, b}, binding);

   EXPECT_EQ(out_shape, (std::vector<std::size_t>{2, 3}));
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_sets_output_shape) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
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

   EXPECT_EQ(plan.num_operands, 3);
   EXPECT_EQ(plan.out_ndim, 2);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
   EXPECT_EQ(plan.itemsize, sizeof(float));
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_builds_three_loops_for_matmul) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
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

   ASSERT_EQ(plan.loop.size(), 3);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].kind, LoopKind::Independent);

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].kind, LoopKind::Independent);

   EXPECT_EQ(plan.loop[2].size, 4);
   EXPECT_EQ(plan.loop[2].kind, LoopKind::Reduction);
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_assigns_m_n_k_roles_for_matmul) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
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

   ASSERT_EQ(plan.loop.size(), 3);

   EXPECT_EQ(plan.loop[0].role, LoopRole::M);
   EXPECT_EQ(plan.loop[1].role, LoopRole::N);
   EXPECT_EQ(plan.loop[2].role, LoopRole::K);
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_detects_gemm_like_matmul) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
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

   EXPECT_TRUE(plan.gemm_like);

   EXPECT_EQ(plan.gemm.batch, 1);
   EXPECT_EQ(plan.gemm.M, 2);
   EXPECT_EQ(plan.gemm.N, 3);
   EXPECT_EQ(plan.gemm.K, 4);

   EXPECT_EQ(plan.gemm.out_rs, 3);
   EXPECT_EQ(plan.gemm.out_cs, 1);

   EXPECT_EQ(plan.gemm.a_rs, 4);
   EXPECT_EQ(plan.gemm.a_cs, 1);

   EXPECT_EQ(plan.gemm.b_rs, 3);
   EXPECT_EQ(plan.gemm.b_cs, 1);
}

TEST(TensorPlanContractionTest,
     make_contraction_plan_einsum_out_rejects_wrong_output_shape) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 4}, // incorrect shape (should be {2, 3}
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
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
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 5},
       .strides = {5, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };
   // cannot infer out shape from input operand shapes
   EXPECT_THROW(infer_einsum_out_shape({a, b}, binding), std::runtime_error);
}

TEST(TensorPlanContractionTest, infer_einsum_out_shape_rejects_mixed_itemsize) {
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(double),
   };

   EinsumBinding binding{
       .op_axis_labels =
           {
               {0, 1},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   EXPECT_THROW(infer_einsum_out_shape({a, b}, binding), std::runtime_error);
}

TEST(TensorPlanContractionTest,
     infer_einsum_out_shape_rejects_repeated_label_within_operand) {
   TensorDescription a{
       .ndims = 2,
       .shape = {4, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
       .op_axis_labels =
           {
               {0, 0}, // repeated label
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 1},
   };

   EXPECT_THROW(infer_einsum_out_shape({a, b}, binding), std::runtime_error);
}

TEST(TensorPlanContractionTest,
     infer_einsum_out_shape_rejects_output_label_missing_from_operands) {
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   TensorDescription b{
       .ndims = 2,
       .shape = {4, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   EinsumBinding binding{
       .op_axis_labels =
           {
               {0, 0},
               {0, 2},
               {2, 1},
           },
       .out_labels = {0, 9},
   };
   EXPECT_THROW(infer_einsum_out_shape({a, b}, binding), std::runtime_error);
}
