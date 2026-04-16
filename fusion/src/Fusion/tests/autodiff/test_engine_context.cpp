#include <gtest/gtest.h>

#include "Fusion/autodiff/Engine.hpp"
#include "Fusion/autodiff/EngineContext.hpp"

#include "helpers.h"

TEST(EngineContextTest, engine_context_set_and_get_returns_engine_instance) {
   EngineContextReset reset;
   Engine<float> engine;
   EngineContext<float>::set(&engine);
   EXPECT_EQ(&EngineContext<float>::get(), &engine);
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
