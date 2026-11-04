#include <gtest/gtest.h>

#include "fixtures.h"

#include "Fusion/autodiff/GradStore.hpp"

TEST(TestGradStore, allocate_add_nullopt_to_slots) {
   GradStore<float> store;
   EXPECT_TRUE(store.empty());
   const GradSlotID slot = store.allocate();
   EXPECT_FALSE(store.empty());
}

TEST(TestGradStore, has_returns_false_when_slot_not_populated) {
   GradStore<float> store;
   EXPECT_TRUE(store.empty());
   const GradSlotID slot = store.allocate();
   EXPECT_FALSE(store.has(slot));
}

TEST(TestGradStore, has_returns_false_when_grad_slot_id_out_of_range) {
   GradStore<float> store;
   EXPECT_TRUE(store.empty());
   store.allocate();
   GradSlotID slot{-1};
   EXPECT_FALSE(store.has(slot));
}

TEST(TestGradStore, has_returns_true_when_grad_slot_id_out_of_range) {
   GradStore<float> store;
   EXPECT_TRUE(store.empty());
   store.allocate();
   const GradSlotID slot = store.allocate();
   const Tensor<float> grad = make_test_tensor(false).base();

   store.set(slot, grad);

   EXPECT_TRUE(store.has(slot));
}

TEST(TestGradStore, get_returns_stored_grad) {
   GradStore<float> store;

   const GradSlotID slot = store.allocate();
   const Tensor<float> grad = make_test_tensor(false).base();

   store.set(slot, grad);

   EXPECT_TRUE(store.has(slot));

   Tensor<float> result = store.get(slot);
   EXPECT_TENSOR_EQ(result, grad);
}

TEST(TestGradStore, set_stores_grad_in_correct_slot) {
   GradStore<float> store;
   store.allocate();
   const GradSlotID slot1 = store.allocate();
   const GradSlotID slot2 = store.allocate();
   const Tensor<float> grad = make_test_tensor(false).base();

   store.set(slot2, grad);
   Tensor<float> result = store.get(slot2);
   EXPECT_FALSE(store.has(slot1));
   EXPECT_TRUE(store.has(slot2));
   EXPECT_TENSOR_EQ(result, grad);
}

TEST(GradStoreTest, reset_removes_value_from_slot) {
   GradStore<float> store;
   store.allocate();
   const GradSlotID slot1 = store.allocate();
   const GradSlotID slot2 = store.allocate();
   const Tensor<float> grad = make_test_tensor(false).base();
   store.set(slot2, grad);
   EXPECT_FALSE(store.has(slot1));
   EXPECT_TRUE(store.has(slot2));
   store.reset_slot(slot2);
   EXPECT_FALSE(store.has(slot2));
}
