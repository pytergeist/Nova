#include <gtest/gtest.h>

#include "fixtures.h"

#include "Fusion/autodiff/Engine.hpp"


TEST(AutodiffEngineTest, default_constructed_engine_has_empty_grad_and_val_buffers) {
   Engine<float> engine{};
   EXPECT_TRUE(engine.val_buffer_is_empty());
   EXPECT_TRUE(engine.grad_buffer_is_empty());
}

TEST(AutodiffEngineTest, apply_single_binary_op_returns_single_forward_result_vid) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t1.raw());
   meta.push_back(t2.raw());

   ValueID vid1 = engine.track_input(t1.raw(), true);
   ValueID vid2 = engine.track_input(t2.raw(), true);

   ASSERT_EQ(vid1, ValueID{0});
   ASSERT_EQ(vid2, ValueID{1});

   std::vector<ValueID> vids{vid1, vid2};

   using Op = Operation<float, PopTestBinaryOp<float>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_EQ(out_vid, ValueID{2});
   RawTensor<float> forward_result = engine.materialise(out_vid);
   EXPECT_TENSOR_EQ(forward_result,  make_test_tensor(false).raw());
}

TEST(AutodiffEngineTest, apply_single_unary_op_returns_single_forward_result_vid) {
   const ADTensor<float> t1 = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t1.raw());

   ValueID vid1 = engine.track_input(t1.raw(), true);

   ASSERT_EQ(vid1, ValueID{0});

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, PopTestUnaryOp<float>>;
   ValueID out_vid = engine.apply_single<Op>(meta, vids);
   EXPECT_EQ(out_vid, ValueID{1});
   RawTensor<float> forward_result = engine.materialise(out_vid);
   EXPECT_TENSOR_EQ(forward_result,  make_test_tensor(false).raw());
}


TEST(AutodiffEngineTest, apply_single_split_op_returns_vector_forward_result_of_vids) {
   const ADTensor<float> t1 = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t1.raw());

   ValueID vid1 = engine.track_input(t1.raw(), true);
   ValueID vid2 = engine.track_input(t1.raw(), true);

   ASSERT_EQ(vid1, ValueID{0});
   ASSERT_EQ(vid2, ValueID{1});

   std::vector<ValueID> vids{vid1, vid2};

   using Op = Operation<float, PopTestSplitOp<float>>;
   std::vector<ValueID> out_vids = engine.apply_multi<Op>(meta, vids);
   EXPECT_EQ(out_vids.size(), 2);
   EXPECT_EQ(out_vids[0], ValueID{2});
   EXPECT_EQ(out_vids[1], ValueID{3});
   RawTensor<float> forward_result1 = engine.materialise(out_vids[0]);
   RawTensor<float> forward_result2 = engine.materialise(out_vids[1]);
   // TODO: set seed globally for fixtures, not with magic numbers
   EXPECT_TENSOR_EQ(forward_result1,  make_test_tensor(false, 42).raw());
   EXPECT_TENSOR_EQ(forward_result2,  make_test_tensor(false, 43).raw());
}


TEST(AutodiffEngineTest, apply_single_binary_op_throws_when_forward_result_empty) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t1.raw());
   meta.push_back(t2.raw());

   ValueID vid1 = engine.track_input(t1.raw(), true);
   ValueID vid2 = engine.track_input(t2.raw(), true);

   std::vector<ValueID> vids{vid1, vid2};

   using Op = Operation<float, TestBinaryOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}

TEST(AutodiffEngineTest, apply_multi_binary_op_throws_when_forward_result_empty) {
   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t1.raw());
   meta.push_back(t2.raw());

   ValueID vid1 = engine.track_input(t1.raw(), true);
   ValueID vid2 = engine.track_input(t2.raw(), true);

   std::vector<ValueID> vids{vid1, vid2};

   using Op = Operation<float, TestBinaryOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}


TEST(AutodiffEngineTest, apply_single_unary_op_throws_when_forward_result_empty) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t.raw());

   ValueID vid1 = engine.track_input(t.raw(), true);

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, TestUnaryOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}

TEST(AutodiffEngineTest, apply_multi_unary_op_throws_when_forward_result_empty) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t.raw());

   ValueID vid1 = engine.track_input(t.raw(), true);

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, TestUnaryOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}

TEST(AutodiffEngineTest, apply_single_split_op_throws) {
   const ADTensor<float> t = make_test_tensor(true);

   Engine<float> engine{};

   AutodiffMeta<float> meta;
   meta.push_back(t.raw());

   ValueID vid1 = engine.track_input(t.raw(), true);

   std::vector<ValueID> vids{vid1};

   using Op = Operation<float, TestSplitOp<float>>;
   EXPECT_THROW(engine.apply_single<Op>(meta, vids), std::runtime_error);
}



TEST(AutodiffEngineTest, track_input_with_requires_grad_true_marks_leaf) {
   Engine<float> engine{};

   const ADTensor<float> t = make_test_tensor(true);
   ValueID vid = engine.track_input(t.raw(), true);

   BackwardResult<float> result = engine.backward(vid);

   EXPECT_TRUE(result.grads.contains(vid));
}

TEST(AutodiffEngineTest, track_input_with_requires_grad_false_does_not_mark_leaf) {
   Engine<float> engine{};

   const ADTensor<float> t = make_test_tensor(false);
   ValueID vid = engine.track_input(t.raw(), false);

   EXPECT_THROW(engine.backward(vid), std::runtime_error);
}

TEST(AutodiffEngineTest, maybe_mark_leaf_on_intermediate_value_does_not_mark_leaf) {
   Engine<float> engine{};

   const ADTensor<float> t1 = make_test_tensor(true);
   const ADTensor<float> t2 = make_test_tensor(true);

   ValueID v1 = engine.track_input(t1.raw(), false);
   ValueID v2 = engine.track_input(t2.raw(), false);

   AutodiffMeta<float> meta;
   meta.push_back(t1.raw());
   meta.push_back(t2.raw());

   std::vector<ValueID> vids{v1, v2};

   using Op = Operation<float, PopTestBinaryOp<float>>;
   ValueID out = engine.apply_single<Op>(meta, vids);

   engine.maybe_mark_leaf(out, true);

   EXPECT_THROW(engine.backward(out), std::runtime_error);
}

TEST(AutodiffEngineTest, maybe_mark_leaf_marks_unproduced_value_when_requires_grad_true) {
   Engine<float> engine{};

   const ADTensor<float> t = make_test_tensor(false);
   ValueID vid = engine.track_input(t.raw(), false);

   engine.maybe_mark_leaf(vid, true);

   BackwardResult<float> result = engine.backward(vid);
   EXPECT_TRUE(result.grads.contains(vid));
}


TEST(AutodiffEngineTest, track_input_create_new_vid_and_adds_tensor_to_buffer) {
    Engine<float> engine{};
    const ADTensor<float> t = make_test_tensor(true);
    ValueID vid = engine.track_input(t.raw(), true);
    RawTensor<float> result = engine.materialise(vid);
    EXPECT_EQ(vid, ValueID{0});
    EXPECT_TENSOR_EQ(result, t.raw());
}

TEST(AutodiffEngineTest, track_input_creates_and_stores_tensors_in_order) {
    Engine<float> engine{};
    const ADTensor<float> t1 = make_test_tensor(true, 42);
    const ADTensor<float> t2 = make_test_tensor(true, 43);
    ValueID vid1 = engine.track_input(t1.raw(), true);
    ValueID vid2 = engine.track_input(t2.raw(), true);
    RawTensor<float> result1 = engine.materialise(vid1);
    RawTensor<float> result2 = engine.materialise(vid2);
    EXPECT_EQ(vid1, ValueID{0});
    EXPECT_EQ(vid2, ValueID{1});
    EXPECT_TENSOR_EQ(result1, t1.raw());
    EXPECT_TENSOR_EQ(result2, t2.raw());
}

// TO TEST:
// backward
// materialise_leaf_grads
// materialise
// get_grad
// has_value
