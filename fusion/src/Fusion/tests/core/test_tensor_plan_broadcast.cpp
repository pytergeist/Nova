#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Fusion/core/TensorPlan.h"

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_single_operand_preserves_shape) {
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({a});

   EXPECT_EQ(plan.num_operands, 1);
   EXPECT_EQ(plan.out_ndim, 2);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));

   ASSERT_EQ(plan.loop.size(), 2);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].kind, LoopKind::Independent);
   EXPECT_EQ(plan.loop[0].stride_bytes,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].kind, LoopKind::Independent);
   EXPECT_EQ(plan.loop[1].stride_bytes,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_same_shape_preserves_output_shape) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({out, a, b});

   EXPECT_EQ(plan.num_operands, 3);
   EXPECT_EQ(plan.out_ndim, 2);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));

   ASSERT_EQ(plan.loop.size(), 2);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].stride_bytes,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 3 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].stride_bytes,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_broadcasts_leading_dimension) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 2,
       .shape = {1, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({out, a, b});

   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.loop.size(), 2);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].stride_bytes,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 3 * static_cast<std::int64_t>(sizeof(float)), 0}));

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].stride_bytes,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_broadcasts_trailing_dimension) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 2,
       .shape = {2, 1},
       .strides = {1, 1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({out, a, b});

   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.loop.size(), 2);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].stride_bytes,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].stride_bytes,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)), 0}));
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_right_aligns_lower_rank_operand) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 1,
       .shape = {3},
       .strides = {1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({out, a, b});

   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.loop.size(), 2);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].stride_bytes,
             (std::vector<std::int64_t>{
                 3 * static_cast<std::int64_t>(sizeof(float)),
                 3 * static_cast<std::int64_t>(sizeof(float)), 0}));

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].stride_bytes,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_supports_multiple_broadcast_operands) {
   TensorDescription out{
       .ndims = 3,
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription a{
       .ndims = 3,
       .shape = {2, 3, 4},
       .strides = {12, 4, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 3,
       .shape = {1, 3, 1},
       .strides = {3, 1, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription c{
       .ndims = 1,
       .shape = {4},
       .strides = {1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({out, a, b, c});

   EXPECT_EQ(plan.num_operands, 4);
   EXPECT_EQ(plan.out_ndim, 3);
   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3, 4}));

   ASSERT_EQ(plan.loop.size(), 3);

   EXPECT_EQ(plan.loop[0].size, 2);
   EXPECT_EQ(plan.loop[0].stride_bytes,
             (std::vector<std::int64_t>{
                 12 * static_cast<std::int64_t>(sizeof(float)),
                 12 * static_cast<std::int64_t>(sizeof(float)), 0, 0}));

   EXPECT_EQ(plan.loop[1].size, 3);
   EXPECT_EQ(plan.loop[1].stride_bytes,
             (std::vector<std::int64_t>{
                 4 * static_cast<std::int64_t>(sizeof(float)),
                 4 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)), 0}));

   EXPECT_EQ(plan.loop[2].size, 4);
   EXPECT_EQ(plan.loop[2].stride_bytes,
             (std::vector<std::int64_t>{
                 1 * static_cast<std::int64_t>(sizeof(float)),
                 1 * static_cast<std::int64_t>(sizeof(float)), 0,
                 1 * static_cast<std::int64_t>(sizeof(float))}));
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_scalar_like_all_ones_shape_broadcasts) {
   TensorDescription out{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 2,
       .shape = {1, 1},
       .strides = {1, 1},
       .itemsize = sizeof(float),
   };

   BroadcastPlan plan = make_broadcast_plan({out, a, b});

   EXPECT_EQ(plan.out_shape, (std::vector<std::size_t>{2, 3}));
   ASSERT_EQ(plan.loop.size(), 2);

   EXPECT_EQ(plan.loop[0].stride_bytes.back(), 0);
   EXPECT_EQ(plan.loop[1].stride_bytes.back(), 0);
}

TEST(TensorPlanBroadcastTest, make_broadcast_plan_rejects_mixed_itemsize) {
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(double),
   };

   EXPECT_THROW(make_broadcast_plan({a, b}), std::runtime_error);
}

TEST(TensorPlanBroadcastTest, make_broadcast_plan_rejects_shape_mismatch) {
   TensorDescription a{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {3, 1},
       .itemsize = sizeof(float),
   };
   TensorDescription b{
       .ndims = 2,
       .shape = {2, 4},
       .strides = {4, 1},
       .itemsize = sizeof(float),
   };

   EXPECT_THROW(make_broadcast_plan({a, b}), std::runtime_error);
}

TEST(TensorPlanBroadcastTest,
     make_broadcast_plan_rejects_bad_tensor_description_shape_rank_mismatch) {
   TensorDescription bad{
       .ndims = 2,
       .shape = {2, 3},
       .strides = {1},
       .itemsize = sizeof(float),
   };

   EXPECT_THROW(make_broadcast_plan({bad}), std::runtime_error);
}

TEST(TensorPlanBroadcastTest, make_broadcast_plan_rejects_empty_operands) {
   EXPECT_THROW(make_broadcast_plan({}), std::runtime_error);
}