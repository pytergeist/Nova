#include <gtest/gtest.h>

#include "Fusion/autodiff/Engine.hpp"
#include "Fusion/autodiff/EngineContext.hpp"

#include "fixtures.h"

TEST(EngineContextTest, engine_context_set_and_get_returns_engine_instance) {
   EngineContextReset reset;
   Engine<float> engine;
   EngineContext<float>::set(&engine);
   EXPECT_EQ(&EngineContext<float>::get(), &engine);
}

TEST(EngineContextTest, engine_context_clear_results_in_empty_context) {
   EngineContextReset reset;
   Engine<float> engine1;
   Engine<float> engine2;
   EngineContext<float>::set(&engine1);
   EngineContext<float>::set(&engine2);
   EngineContext<float>::clear();
   EXPECT_FALSE(EngineContext<float>::has());
}

TEST(EngineContextTest, engine_context_sets_multiple_and_get_returns_latest_engine_instance) {
   EngineContextReset reset;
   Engine<float> engine1;
   Engine<float> engine2;
   EngineContext<float>::set(&engine1);
   EngineContext<float>::set(&engine2);
   EXPECT_EQ(&EngineContext<float>::get(), &engine2);
}

TEST(EngineContextTest, engine_context_returns_correct_engine_after_multiple_set_and_pop) {
   EngineContextReset reset;
   Engine<float> engine1;
   Engine<float> engine2;
   EngineContext<float>::set(&engine1);
   EngineContext<float>::set(&engine2);
   EngineContext<float>::pop();
   EXPECT_EQ(&EngineContext<float>::get(), &engine1);
}

TEST(EngineContextTest, empty_engine_context_throws_runtime_error) {
   EngineContextReset reset;
   Engine<float> engine;
   EXPECT_THROW(EngineContext<float>::get(), std::runtime_error);
}

TEST(EngineContextTest, engine_context_has_returns_true_when_set) {
   EngineContextReset reset;
   Engine<float> engine;
   EngineContext<float>::set(&engine);
   EXPECT_TRUE(EngineContext<float>::has());
}

TEST(EngineContextTest, engine_context_has_returns_false_when_not_set) {
   EngineContextReset reset;
   Engine<float> engine;
   EXPECT_FALSE(EngineContext<float>::has());
}

TEST(EngineScopeTest, engine_scope_enter_sets_engine) {
   EngineContextReset reset;
   EngineScope<float> scope{};
   scope.enter();

   EXPECT_TRUE(scope.active());
   EXPECT_TRUE(EngineContext<float>::has());
   EXPECT_EQ(&EngineContext<float>::get(), &scope.eng());
}

TEST(EngineScopeTest, engine_scope_exit_after_enter_clears_engine_state) {
   EngineContextReset reset;

   EngineScope<float> scope{};
   scope.enter();

   ASSERT_TRUE(scope.active());
   ASSERT_TRUE(EngineContext<float>::has());

   scope.exit();

   EXPECT_FALSE(scope.active());
   EXPECT_FALSE(EngineContext<float>::has());
   EXPECT_THROW(EngineContext<float>::get(), std::runtime_error);
}

TEST(EngineScopeTest, engine_scope_destructor_clears_engine_state_after_enter) {
   EngineContextReset reset;

   {
      EngineScope<float> scope{};
      scope.enter();

      EXPECT_TRUE(scope.active());
      EXPECT_TRUE(EngineContext<float>::has());
   }

   EXPECT_FALSE(EngineContext<float>::has());
   EXPECT_THROW(EngineContext<float>::get(), std::runtime_error);
}


TEST(EngineScopeTest, multiple_engine_scope_exit_after_enter_clears_engine_state) {
   EngineContextReset reset;

   EngineScope<float> scope1{};
   scope1.enter();
   EngineScope<float> scope2{};
   scope2.enter();

   EXPECT_TRUE(scope1.active());
   EXPECT_TRUE(EngineContext<float>::has());

   Engine<float>& engine1 = EngineContext<float>::get();

   scope1.exit();

   Engine<float>& engine2 = EngineContext<float>::get();


   EXPECT_TRUE(scope2.active());
   EXPECT_NE(&engine1, &engine2);
   scope2.exit();
   EXPECT_FALSE(EngineContext<float>::has());
   EXPECT_THROW(EngineContext<float>::get(), std::runtime_error);
}

// to test:
// test new exit?