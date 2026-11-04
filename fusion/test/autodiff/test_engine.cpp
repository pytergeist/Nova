#include <gtest/gtest.h>

#include "fixtures.h"

#include "Fusion/autodiff/Engine.hpp"

TEST(AutodiffEngineTest,
     default_constructed_engine_has_empty_grad_and_val_buffers) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   EXPECT_TRUE(engine.val_buffer_is_empty());
   EXPECT_TRUE(engine.grad_buffer_is_empty());
}

TEST(AutodiffEngineTest,
     apply_single_binary_op_returns_single_forward_result_vid) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());
   meta.push_back(t2.base());

   ValueID vid1 = engine.track_input(t1.base(), true);
   ValueID vid2 = engine.track_input(t2.base(), true);

   ASSERT_EQ(vid1, ValueID{0});
   ASSERT_EQ(vid2, ValueID{1});

   std::vector<ValueID> vids{vid1, vid2};

   using Op = Operation<float, PopTestBinaryOp<float>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_EQ(out_vid, ValueID{2});
   Tensor<float> forward_result = engine.materialise(out_vid);
   EXPECT_TENSOR_EQ(forward_result, make_test_tensor(false).base());
}

TEST(AutodiffEngineTest,
     apply_single_unary_op_returns_single_forward_result_vid) {
   const ADTensor<float> t1 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());

   ValueID vid1 = engine.track_input(t1.base(), true);

   ASSERT_EQ(vid1, ValueID{0});

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, PopTestUnaryOp<float>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_EQ(out_vid, ValueID{1});
   Tensor<float> forward_result = engine.materialise(out_vid);
   EXPECT_TENSOR_EQ(forward_result, make_test_tensor(false).base());
}

TEST(AutodiffEngineTest, apply_multi_split_op_returns_all_output_vids) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());
   meta.push_back(t2.base());

   ValueID vid1 = engine.track_input(t1.base(), true);
   ValueID vid2 = engine.track_input(t2.base(), true);

   ASSERT_EQ(vid1, ValueID{0});
   ASSERT_EQ(vid2, ValueID{1});

   std::vector<ValueID> vids{vid1, vid2};
   constexpr std::uint32_t seed = 77;
   using Op = Operation<float, PopTestSplitOp<float, seed>>;
   std::vector<ValueID> out_vids = engine.apply_multi<Op>(meta, vids);
   EXPECT_EQ(out_vids.size(), 2);
   EXPECT_EQ(out_vids[0], ValueID{2});
   EXPECT_EQ(out_vids[1], ValueID{3});
   Tensor<float> forward_result1 = engine.materialise(out_vids[0]);
   Tensor<float> forward_result2 = engine.materialise(out_vids[1]);
   EXPECT_TENSOR_EQ(forward_result1, make_test_tensor(false, seed).base());
   EXPECT_TENSOR_EQ(forward_result2, make_test_tensor(false, seed + 1).base());
}

TEST(AutodiffEngineTest, apply_single_throws_when_forward_result_empty) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());
   meta.push_back(t2.base());

   ValueID vid1 = engine.track_input(t1.base(), true);
   ValueID vid2 = engine.track_input(t2.base(), true);

   std::vector<ValueID> vids{vid1, vid2};

   using Op = Operation<float, TestBinaryOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}

TEST(AutodiffEngineTest, apply_multi_throws_when_forward_result_empty) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t.base());

   ValueID vid1 = engine.track_input(t.base(), true);

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, TestSplitOp<float>>;
   EXPECT_THROW(engine.apply_multi<Op>(meta, vids), std::runtime_error);
}

TEST(AutodiffEngineTest, apply_single_throws_when_op_returns_multiple_outputs) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t.base());

   ValueID vid1 = engine.track_input(t.base(), true);

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, TestSplitOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}

TEST(AutodiffEngineTest, track_input_with_requires_grad_true_marks_leaf) {
   AutodiffContextReset reset;
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   const ADTensor<float> t = make_test_tensor(true);
   const ValueID vid = engine.track_input(t.base(), true);

   engine.backward(vid);

   const GradSlotID slot = engine.get_grad_slot(vid);
   const GradStore<float> &store =
       AutodiffContext<float>::runtime().grad_store();

   EXPECT_TRUE(store.has(slot));
}

TEST(AutodiffEngineTest,
     track_input_with_requires_grad_false_does_not_mark_leaf) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   const ADTensor<float> t = make_test_tensor(false);
   ValueID vid = engine.track_input(t.base(), false);

   EXPECT_THROW(engine.backward(vid), std::runtime_error);
}

TEST(AutodiffEngineTest,
     maybe_mark_leaf_on_intermediate_value_does_not_mark_leaf) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   ValueID v1 = engine.track_input(t1.base(), false);
   ValueID v2 = engine.track_input(t2.base(), false);

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());
   meta.push_back(t2.base());

   std::vector<ValueID> vids{v1, v2};

   using Op = Operation<float, PopTestBinaryOp<float>>;
   ValueID out = engine.apply_single<Op>(meta, vids);

   engine.maybe_mark_leaf(out, true);

   EXPECT_THROW(engine.backward(out), std::runtime_error);
}

TEST(AutodiffEngineTest,
     maybe_mark_leaf_marks_unproduced_value_when_requires_grad_true) {
   AutodiffContextReset reset;
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   const ADTensor<float> t = make_test_tensor(false);
   ValueID vid = engine.track_input(t.base(), false);

   engine.maybe_mark_leaf(vid, true);

   engine.backward(vid);
   const GradSlotID slot = engine.get_grad_slot(vid);
   const GradStore<float> &store =
       AutodiffContext<float>::runtime().grad_store();

   EXPECT_TRUE(store.has(slot));
}

TEST(AutodiffEngineTest, track_input_create_new_vid_and_adds_tensor_to_buffer) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   const ADTensor<float> t = make_test_tensor(true);
   ValueID vid = engine.track_input(t.base(), true);
   Tensor<float> result = engine.materialise(vid);
   EXPECT_EQ(vid, ValueID{0});
   EXPECT_TENSOR_EQ(result, t.base());
}

TEST(AutodiffEngineTest, track_input_creates_and_stores_tensors_in_order) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   const ADTensor<float> t1 = make_test_tensor(true, 42);
   const ADTensor<float> t2 = make_test_tensor(true, 43);
   ValueID vid1 = engine.track_input(t1.base(), true);
   ValueID vid2 = engine.track_input(t2.base(), true);
   Tensor<float> result1 = engine.materialise(vid1);
   Tensor<float> result2 = engine.materialise(vid2);
   EXPECT_EQ(vid1, ValueID{0});
   EXPECT_EQ(vid2, ValueID{1});
   EXPECT_TENSOR_EQ(result1, t1.base());
   EXPECT_TENSOR_EQ(result2, t2.base());
}

TEST(AutodiffEngineTest, materialise_returns_tensor_by_vid) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   const ADTensor<float> t1 = make_test_tensor(true, 42);
   const ADTensor<float> t2 = make_test_tensor(true, 43);
   ValueID vid1 = engine.track_input(t1.base(), true);
   ValueID vid2 = engine.track_input(t2.base(), true);
   Tensor<float> result1 = engine.materialise(vid1);
   Tensor<float> result2 = engine.materialise(vid2);
   EXPECT_TENSOR_EQ(result1, t1.base());
   EXPECT_TENSOR_EQ(result2, t2.base());
}

TEST(AutodiffEngineTest,
     backward_single_unary_op_returns_unary_backward_result) {
   const ADTensor<float> t1 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());

   ValueID leaf_vid = engine.track_input(t1.base(), true);

   ASSERT_EQ(leaf_vid, ValueID{0});

   std::vector<ValueID> vids{leaf_vid};
   constexpr std::uint32_t seed = 77;
   using Op = Operation<float, PopTestUnaryOp<float, seed>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_TRUE(engine.has_value(out_vid));
   engine.backward(out_vid);
   const GradSlotID slot = engine.get_grad_slot(leaf_vid);
   GradStore<float> &store = AutodiffContext<float>::runtime().grad_store();

   EXPECT_TRUE(store.has(slot));
   Tensor<float> grad = store.get(slot);
   EXPECT_TENSOR_EQ(grad, make_test_tensor(false, seed).base());
}

TEST(AutodiffEngineTest,
     backward_single_binary_op_returns_binary_backward_result) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());
   meta.push_back(t2.base());

   ValueID leaf_vid1 = engine.track_input(t1.base(), true);
   ValueID leaf_vid2 = engine.track_input(t2.base(), true);

   ASSERT_EQ(leaf_vid1, ValueID{0});
   ASSERT_EQ(leaf_vid2, ValueID{1});

   std::vector<ValueID> vids{leaf_vid1, leaf_vid2};
   constexpr std::uint32_t seed = 77;
   using Op = Operation<float, PopTestBinaryOp<float, seed>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_TRUE(engine.has_value(out_vid));
   engine.backward(out_vid);
   const GradSlotID slot1 = engine.get_grad_slot(leaf_vid1);
   const GradSlotID slot2 = engine.get_grad_slot(leaf_vid2);
   GradStore<float> &store = AutodiffContext<float>::runtime().grad_store();

   EXPECT_TRUE(store.has(slot1));
   EXPECT_TRUE(store.has(slot2));

   Tensor<float> grad1 = store.get(slot1);
   Tensor<float> grad2 = store.get(slot2);

   EXPECT_TENSOR_EQ(grad1, make_test_tensor(false, seed).base());
   EXPECT_TENSOR_EQ(grad2, make_test_tensor(false, seed + 1).base());
}

TEST(AutodiffEngineTest,
     backward_multi_split_op_returns_split_backward_result) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t1.base());
   meta.push_back(t2.base());

   ValueID leaf_vid1 = engine.track_input(t1.base(), true);
   ValueID leaf_vid2 = engine.track_input(t2.base(), true);

   ASSERT_EQ(leaf_vid1, ValueID{0});
   ASSERT_EQ(leaf_vid2, ValueID{1});

   std::vector<ValueID> vids{leaf_vid1, leaf_vid2};
   constexpr std::uint32_t seed = 77;
   using Op = Operation<float, PopTestSplitOp<float, seed>>;
   std::vector<ValueID> out_vids = engine.apply_multi<Op>(meta, vids);
   ASSERT_EQ(out_vids[0], ValueID{2});
   ASSERT_EQ(out_vids[1], ValueID{3});
   engine.backward(out_vids[1]);
   const GradSlotID slot1 = engine.get_grad_slot(leaf_vid1);
   const GradSlotID slot2 = engine.get_grad_slot(leaf_vid2);
   GradStore<float> &store = AutodiffContext<float>::runtime().grad_store();

   EXPECT_TRUE(store.has(slot1));
   EXPECT_TRUE(store.has(slot2));

   Tensor<float> grad1 = store.get(slot1);
   Tensor<float> grad2 = store.get(slot2);
   EXPECT_TENSOR_EQ(grad1, make_test_tensor(false, seed).base());
   EXPECT_TENSOR_EQ(grad2, make_test_tensor(false, seed + 1).base());
}

TEST(AutodiffEngineTest, get_grad_throws_when_no_grad_exists) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t.base());

   ValueID leaf_vid = engine.track_input(t.base(), true);

   ASSERT_EQ(leaf_vid, ValueID{0});

   std::vector<ValueID> vids{leaf_vid};

   using Op = Operation<float, PopTestUnaryOp<float>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_TRUE(engine.has_value(out_vid));
   EXPECT_THROW(engine.get_grad(leaf_vid), std::out_of_range);
}

TEST(AutodiffEngineTest, has_value_returns_true_when_val_in_buffer) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   AutodiffMeta<float> meta;
   meta.push_back(t.base());

   ValueID leaf_vid = engine.track_input(t.base(), true);
   EXPECT_TRUE(engine.has_value(leaf_vid));
}

TEST(AutodiffEngineTest, has_value_returns_false_when_val_not_in_buffer) {
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   EXPECT_FALSE(engine.has_value(ValueID{0}));
}

TEST(AutodiffEngineTest,
     backward_accumulates_gradients_when_same_leaf_feeds_two_branches) {
   const ADTensor<float> x = make_test_tensor(true);
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   ValueID x_vid = engine.track_input(x.base(), true);
   ASSERT_EQ(x_vid, ValueID{0});
   constexpr std::uint32_t seed1 = 77;
   constexpr std::uint32_t seed2 = 123;
   using BranchOp = Operation<float, PopTestUnaryOp<float, seed1>>;
   using MergeOp = Operation<float, PopTestBinaryOp<float, seed2>>;

   AutodiffMeta<float> meta_u1;
   meta_u1.push_back(x.base());
   std::vector<ValueID> in1{x_vid};
   ValueID u1_vid = engine.apply_single<BranchOp>(meta_u1, in1);
   EXPECT_TRUE(engine.has_value(u1_vid));

   AutodiffMeta<float> meta_u2;
   meta_u2.push_back(x.base());
   std::vector<ValueID> in2{x_vid};
   ValueID u2_vid = engine.apply_single<BranchOp>(meta_u2, in2);
   EXPECT_TRUE(engine.has_value(u2_vid));

   AutodiffMeta<float> meta_z;
   meta_z.emplace_back(engine.materialise(u1_vid));
   meta_z.emplace_back(engine.materialise(u2_vid));
   std::vector<ValueID> merge_inputs{u1_vid, u2_vid};
   ValueID z_vid = engine.apply_single<MergeOp>(meta_z, merge_inputs);
   EXPECT_TRUE(engine.has_value(z_vid));

   engine.backward(z_vid);

   Tensor<float> grad_x = engine.get_grad(x_vid);

   Tensor<float> expected_branch = make_test_tensor(false, seed1).base();
   Tensor<float> expected = expected_branch + expected_branch;

   EXPECT_TENSOR_EQ(grad_x, expected);
}

TEST(AutodiffEngineTest, has_value_returns_false_for_negative_vid) {
   const ADTensor<float> t = make_test_tensor(true);
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   ValueID vid = engine.track_input(t.base(), true);
   EXPECT_FALSE(engine.has_value(ValueID{-1}));
}

TEST(AutodiffEngineTest, materialise_called_with_negative_vid_throws) {
   const ADTensor<float> t = make_test_tensor(true);
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   ValueID vid = engine.track_input(t.base(), true);
   EXPECT_THROW(engine.materialise(ValueID{-1}), std::out_of_range);
}

TEST(AutodiffEngineTest, materialise_called_with_out_of_range_vid_throws) {
   const ADTensor<float> t = make_test_tensor(true);
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   ValueID vid = engine.track_input(t.base(), true);
   EXPECT_THROW(engine.materialise(ValueID{5}), std::out_of_range);
}

TEST(AutodiffEngineTest,
     get_grad_on_intermediate_tensor_returns_intermediate_grad) {
   const ADTensor<float> x = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   ValueID x_vid = engine.track_input(x.base(), true);
   ASSERT_EQ(x_vid, ValueID{0});

   AutodiffMeta<float> meta_u1;
   meta_u1.push_back(x.base());
   std::vector<ValueID> x_inputs_1{x_vid};

   using UnaryOp = Operation<float, PopTestUnaryOp<float>>;
   ValueID u1_vid = engine.apply_single<UnaryOp>(meta_u1, x_inputs_1);
   EXPECT_TRUE(engine.has_value(u1_vid));

   AutodiffMeta<float> meta_u2;
   meta_u2.push_back(x.base());
   std::vector<ValueID> x_inputs_2{x_vid};

   ValueID u2_vid = engine.apply_single<UnaryOp>(meta_u2, x_inputs_2);
   EXPECT_TRUE(engine.has_value(u2_vid));

   AutodiffMeta<float> meta_z;
   meta_z.emplace_back(engine.materialise(u1_vid));
   meta_z.emplace_back(engine.materialise(u2_vid));
   std::vector<ValueID> merge_inputs{u1_vid, u2_vid};
   constexpr std::uint32_t seed = 77;
   using BinaryOp = Operation<float, PopTestBinaryOp<float, seed>>;
   ValueID z_vid = engine.apply_single<BinaryOp>(meta_z, merge_inputs);
   EXPECT_TRUE(engine.has_value(z_vid));

   engine.backward(z_vid);

   Tensor<float> grad = engine.get_grad(u1_vid);
   EXPECT_TENSOR_EQ(grad, make_test_tensor(false, seed).base());
}

TEST(AutodiffEngineTest,
     export_leaf_does_not_add_intermediate_grads_to_grad_store) {
   const ADTensor<float> x = make_test_tensor(true);

   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());

   ValueID x_vid = engine.track_input(x.base(), true);
   ASSERT_EQ(x_vid, ValueID{0});

   AutodiffMeta<float> meta_u1;
   meta_u1.push_back(x.base());
   std::vector<ValueID> x_inputs_1{x_vid};

   using UnaryOp = Operation<float, PopTestUnaryOp<float>>;
   ValueID u1_vid = engine.apply_single<UnaryOp>(meta_u1, x_inputs_1);
   EXPECT_TRUE(engine.has_value(u1_vid));

   AutodiffMeta<float> meta_u2;
   meta_u2.push_back(x.base());
   std::vector<ValueID> x_inputs_2{x_vid};

   ValueID u2_vid = engine.apply_single<UnaryOp>(meta_u2, x_inputs_2);
   EXPECT_TRUE(engine.has_value(u2_vid));

   AutodiffMeta<float> meta_z;
   meta_z.emplace_back(engine.materialise(u1_vid));
   meta_z.emplace_back(engine.materialise(u2_vid));
   std::vector<ValueID> merge_inputs{u1_vid, u2_vid};
   constexpr std::uint32_t seed = 77;
   using BinaryOp = Operation<float, PopTestBinaryOp<float, seed>>;
   ValueID z_vid = engine.apply_single<BinaryOp>(meta_z, merge_inputs);
   EXPECT_TRUE(engine.has_value(z_vid));

   engine.backward(z_vid);

   EXPECT_THROW(engine.get_grad_slot(u1_vid), std::runtime_error);
}