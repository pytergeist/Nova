#include <gtest/gtest.h>

#include "helpers.h"

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
   EXPECT_TENSOR_EQ(forward_result1,  make_test_tensor(false).raw());
   EXPECT_TENSOR_EQ(forward_result1,  make_test_tensor(false).raw());
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






// TO TEST:
// backward
// maybe_mark_leaf
// materialise_leaf_grads
// track_input
// materialise
// get_grad
// has_value
